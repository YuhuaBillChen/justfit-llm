//! Responses storage and streaming service.
use super::{generate, prepare_deadline, reply, safe_json, stream_ready, Sender, Service};
use crate::{bad, ApiError, Result};
use axum::{
    body::{Body, Bytes},
    response::Response,
};
use serde_json::{json, Value};
use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
};
use tokio::sync::{mpsc, OwnedSemaphorePermit};
use tokio_stream::wrappers::ReceiverStream;

async fn send_event(sender: &Sender, value: &Value) -> Result<()> {
    let bytes = if value.get("comment").is_some() {
        Bytes::from_static(b": justfit-keepalive\n\n")
    } else {
        Bytes::from(format!(
            "event: {}\ndata: {}\n\n",
            value["type"].as_str().unwrap_or("error"),
            safe_json(value)
        ))
    };
    tokio::time::timeout(std::time::Duration::from_secs(30), sender.send(Ok(bytes)))
        .await
        .map_err(|_| ApiError(408, "stream write timed out".into(), "timeout_error".into()))?
        .map_err(|_| ApiError(499, "client disconnected".into(), "cancelled".into()))
}

pub(super) fn chat_events(state: &mut StreamState, bytes: &[u8]) -> Result<Vec<Value>> {
    if bytes.starts_with(b":") {
        return Ok(state.idle());
    }
    let text = std::str::from_utf8(bytes).map_err(|_| bad("invalid internal stream"))?;
    let mut events = Vec::new();
    for line in text.lines().filter_map(|line| line.strip_prefix("data: ")) {
        if line == "[DONE]" {
            continue;
        }
        let chunk: Value =
            serde_json::from_str(line).map_err(|_| bad("invalid internal stream"))?;
        if let Some(progress) = chunk.get("prompt_progress") {
            events.extend(state.progress(progress.clone()));
        }
        let delta = &chunk["choices"][0]["delta"];
        for field in ["reasoning_content", "content"] {
            if let Some(text) = delta[field].as_str() {
                events.extend(state.delta(field, text));
            }
        }
        if let Some(deltas)=delta["tool_calls"].as_array(){for delta in deltas{events.extend(state.tool(delta)?);}}
    }
    Ok(events)
}

struct Record {
    response: Vec<u8>,
    history: Vec<u8>,
}
impl Record {
    fn size(&self) -> usize {
        self.response.len() + self.history.len()
    }
}
struct Entries {
    records: HashMap<String, Arc<Record>>,
    order: VecDeque<String>,
    bytes: usize,
    evictions: u64,
    hits: u64,
    misses: u64,
}
pub(super) struct Store {
    entries: Mutex<Entries>,
    budget: usize,
}
fn encode(value: &Value) -> Result<Vec<u8>> {
    let text = serde_json::to_string(value).map_err(|_| {
        ApiError(
            500,
            "response could not be encoded".into(),
            "internal_server_error".into(),
        )
    })?;
    Ok(text
        .replace('\u{85}', "\\u0085")
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029")
        .into_bytes())
}
impl Store {
    pub(super) fn new(budget: usize) -> Self {
        Self {
            entries: Mutex::new(Entries {
                records: HashMap::new(),
                order: VecDeque::new(),
                bytes: 0,
                evictions: 0,
                hits: 0,
                misses: 0,
            }),
            budget,
        }
    }
    fn record(&self, id: &str) -> Option<Arc<Record>> {
        let mut entries = self.entries.lock().unwrap();
        let result = entries.records.get(id).cloned();
        if result.is_some() {
            entries.hits += 1;
            entries.order.retain(|k| k != id);
            entries.order.push_back(id.into());
        } else {
            entries.misses += 1;
        }
        result
    }
    pub(super) fn get(&self, id: &str) -> Result<Value> {
        let record = self
            .record(id)
            .ok_or_else(|| ApiError(404, "response not found".into(), "not_found_error".into()))?;
        serde_json::from_slice(&record.response).map_err(|_| {
            ApiError(
                500,
                "stored response is invalid".into(),
                "internal_server_error".into(),
            )
        })
    }
    fn put(&self, response: &Value, history: &Value) -> Result<bool> {
        let record = Arc::new(Record {
            response: encode(response)?,
            history: encode(history)?,
        });
        if record.size() > self.budget {
            return Ok(false);
        }
        let id = response["id"]
            .as_str()
            .ok_or_else(|| bad("response id unavailable"))?
            .to_string();
        let mut entries = self.entries.lock().unwrap();
        if let Some(old) = entries.records.remove(&id) {
            entries.bytes -= old.size();
        }
        entries.order.retain(|key| key != &id);
        entries.order.push_back(id.clone());
        entries.bytes += record.size();
        entries.records.insert(id, record);
        while entries.bytes > self.budget {
            let id = entries.order.pop_front().unwrap();
            let old = entries.records.remove(&id).unwrap();
            entries.bytes -= old.size();
            entries.evictions += 1;
        }
        Ok(true)
    }
    pub(super) fn delete(&self, id: &str) -> Result<Value> {
        let mut entries = self.entries.lock().unwrap();
        let old = entries
            .records
            .remove(id)
            .ok_or_else(|| ApiError(404, "response not found".into(), "not_found_error".into()))?;
        entries.bytes -= old.size();
        entries.order.retain(|key| key != id);
        Ok(json!({"id":id,"object":"response","deleted":true}))
    }
    pub(super) fn stats(&self) -> Value {
        let e = self.entries.lock().unwrap();
        json!({"entries":e.records.len(),"bytes":e.bytes,"budget_bytes":self.budget,"evictions":e.evictions,"hits":e.hits,"misses":e.misses})
    }
}

fn output_item(id: &str, kind: &str, index: usize, text: &str, incomplete: bool) -> Value {
    let status = if incomplete {
        "incomplete"
    } else {
        "completed"
    };
    if kind == "reasoning" {
        json!({"id":format!("rs_{id}_{index}"),"type":"reasoning","status":status,"summary":[{"type":"summary_text","text":text}],"content":[{"type":"reasoning_text","text":text}],"encrypted_content":null})
    } else {
        json!({"id":format!("msg_{id}_{index}"),"type":"message","status":status,"role":"assistant","content":[{"type":"output_text","text":text,"annotations":[]}]})
    }
}
fn complete(
    model: &str,
    public: &str,
    chat: &Value,
    store: bool,
    previous: &Value,
    reasoning_open: bool,
) -> Value {
    let choice = &chat["choices"][0];
    let message = &choice["message"];
    let incomplete = choice["finish_reason"] == "length";
    let mut output = Vec::new();
    let reasoning = message["reasoning_content"].as_str().unwrap_or("");
    let content = message["content"].as_str().unwrap_or("");
    // A preceding reasoning block is complete once an answer starts.
    if !reasoning.is_empty() {
        output.push(output_item(
            public,
            "reasoning",
            0,
            reasoning,
            incomplete && content.is_empty() && reasoning_open,
        ));
    }
    if !content.is_empty() || output.iter().all(|v| v["type"] == "reasoning") {
        output.push(output_item(
            public,
            "text",
            output.len(),
            content,
            incomplete,
        ));
    }
    let usage = &chat["usage"];
    let input = usage["prompt_tokens"].as_u64().unwrap_or(0);
    let cached = usage["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or(0);
    json!({"id":format!("resp_{public}"),"object":"response","created_at":chat["created"],"status":if incomplete{"incomplete"}else{"completed"},
        "incomplete_details":if incomplete{json!({"reason":"max_output_tokens"})}else{Value::Null},"error":null,"model":model,"output":output,"parallel_tool_calls":true,"store":store,"previous_response_id":previous,
        "text":{"format":{"type":"text"}},"usage":{"input_tokens":input,"input_tokens_details":{"cached_tokens":cached,"cache_write_tokens":input.saturating_sub(cached)},"output_tokens":usage["completion_tokens"],"output_tokens_details":usage["completion_tokens_details"],"total_tokens":usage["total_tokens"]},"end_turn":choice["finish_reason"]=="stop"})
}
pub(super) fn fixture(body: &Value) -> Value {
    let chat = json!({"id":"chatcmpl-fixed","created":7,"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13,"prompt_tokens_details":{"cached_tokens":0},"completion_tokens_details":{"reasoning_tokens":0}},
        "choices":[{"finish_reason":body["reason"],"message":{"content":body["content"],"reasoning_content":body["reasoning"]}}]});
    complete(
        "model",
        "fixed",
        &chat,
        true,
        &Value::Null,
        body["reasoning_open"].as_bool().unwrap_or(false),
    )
}

struct OpenBlock {
    kind: &'static str,
    text: String,
    call_id: Option<String>,
    name: Option<String>,
}
fn response_text_format(chat: &Value) -> Value {
    let format=&chat["response_format"];
    if format["type"]=="json_schema" {
        let mut value=json!({"type":"json_schema"});
        if let Some(fields)=format["json_schema"].as_object() {
            value.as_object_mut().unwrap().extend(fields.clone());
        }
        value
    } else if format.is_object() {format.clone()} else {json!({"type":"text"})}
}
pub(super) struct StreamState {
    model: String,
    public: String,
    created: u64,
    store: bool,
    previous: Value,
    sequence: u64,
    begun: bool,
    open: Option<OpenBlock>,
    output: Vec<Value>,
    text_format: Value,
    parallel: bool,
    namespaces: Value,
}
impl StreamState {
    pub(super) fn new(model: String, public: String, created: u64, store: bool, previous: Value) -> Self {
        Self {
            model,
            public,
            created,
            store,
            previous,
            sequence: 0,
            begun: false,
            open: None,
            output: Vec::new(),
            text_format: json!({"type":"text"}),
            parallel: true,
            namespaces: json!({}),
        }
    }
    fn pending(&self) -> Value {
        json!({"id":format!("resp_{}",self.public),"object":"response","created_at":self.created,"status":"in_progress","incomplete_details":null,"error":null,"model":self.model,"output":[],"parallel_tool_calls":self.parallel,"store":self.store,"previous_response_id":self.previous,"text":{"format":self.text_format},"usage":null,"end_turn":false})
    }
    fn event(&mut self, name: &str, mut payload: Value) -> Value {
        payload["type"] = json!(name);
        payload["sequence_number"] = json!(self.sequence);
        self.sequence += 1;
        payload
    }
    fn begin(&mut self) -> Vec<Value> {
        if self.begun {
            return vec![];
        }
        self.begun = true;
        vec![
            self.event("response.created", json!({"response":self.pending()})),
            self.event("response.in_progress", json!({"response":self.pending()})),
        ]
    }
    fn open(&mut self, kind: &'static str) -> Vec<Value> {
        let index = self.output.len();
        let mut item = output_item(&self.public, kind, index, "", false);
        item["status"] = json!("in_progress");
        if kind == "reasoning" {
            item["summary"] = json!([]);
            item["content"] = json!([]);
        } else {
            item["content"] = json!([]);
        }
        let mut events = vec![self.event(
            "response.output_item.added",
            json!({"output_index":index,"item":item}),
        )];
        events.push(if kind=="reasoning"{self.event("response.reasoning_summary_part.added",json!({"item_id":item["id"],"output_index":index,"summary_index":0,"part":{"type":"summary_text","text":""}}))}
            else{self.event("response.content_part.added",json!({"item_id":item["id"],"output_index":index,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}))});
        self.open = Some(OpenBlock {
            kind,
            text: String::new(),
            call_id: None,
            name: None,
        });
        events
    }
    fn close(&mut self, incomplete: bool) -> Vec<Value> {
        let Some(block) = self.open.take() else {
            return vec![];
        };
        let index = self.output.len();
        let item = if block.kind=="tool" {self.tool_item(&block,index,if incomplete{"incomplete"}else{"completed"})}else{output_item(&self.public, block.kind, index, &block.text, incomplete)};
        let mut events = Vec::new();
        if block.kind=="tool" {
            events.push(self.event("response.function_call_arguments.done",json!({"item_id":item["id"],"output_index":index,"name":item["name"],"arguments":item["arguments"]})));
        }else if block.kind == "reasoning" {
            events.push(self.event("response.reasoning_summary_text.done",json!({"item_id":item["id"],"output_index":index,"summary_index":0,"text":block.text})));
            let mut payload = json!({"item_id":item["id"],"output_index":index,"summary_index":0,"part":{"type":"summary_text","text":block.text}});
            if incomplete {
                payload["status"] = json!("incomplete");
            }
            events.push(self.event("response.reasoning_summary_part.done", payload));
        } else {
            events.push(self.event("response.output_text.done",json!({"item_id":item["id"],"output_index":index,"content_index":0,"text":block.text,"logprobs":[]})));
            events.push(self.event("response.content_part.done",json!({"item_id":item["id"],"output_index":index,"content_index":0,"part":item["content"][0]})));
        }
        events.push(self.event(
            "response.output_item.done",
            json!({"output_index":index,"item":item}),
        ));
        self.output.push(item);
        events
    }
    fn delta(&mut self, field: &str, text: &str) -> Vec<Value> {
        if text.is_empty() {
            return vec![];
        }
        let kind = if field == "reasoning_content" {
            "reasoning"
        } else {
            "text"
        };
        let mut events = Vec::new();
        if self.open.as_ref().is_some_and(|b| b.kind != kind) {
            events.extend(self.close(false));
        }
        if self.open.is_none() {
            events.extend(self.open(kind));
        }
        self.open.as_mut().unwrap().text.push_str(text);
        let index = self.output.len();
        let item_id = format!(
            "{}_{}_{}",
            if kind == "reasoning" { "rs" } else { "msg" },
            self.public,
            index
        );
        let mut payload = json!({"item_id":item_id,"output_index":index,"delta":text});
        if kind == "reasoning" {
            payload["summary_index"] = json!(0);
            events.push(self.event("response.reasoning_summary_text.delta", payload));
        } else {
            payload["content_index"] = json!(0);
            payload["logprobs"] = json!([]);
            events.push(self.event("response.output_text.delta", payload));
        }
        events
    }
    fn tool_item(&self,block:&OpenBlock,index:usize,status:&str)->Value {
        let name=block.name.as_deref().unwrap_or("");let namespace=&self.namespaces[name];
        let mut item=json!({"id":format!("fc_{}_{}",self.public,index),"type":"function_call","status":status,"call_id":block.call_id,"name":namespace.get(1).and_then(Value::as_str).unwrap_or(name),"arguments":if status=="in_progress"{""}else{&block.text}});
        if let Some(namespace)=namespace.get(0){item["namespace"]=namespace.clone();}item
    }
    fn tool(&mut self,delta:&Value)->Result<Vec<Value>> {
        let function=&delta["function"];let mut events=Vec::new();
        if let Some(name)=function["name"].as_str() {
            events.extend(self.close(false));let block=OpenBlock{kind:"tool",text:String::new(),call_id:Some(delta["id"].as_str().ok_or_else(||bad("missing tool call id"))?.to_string()),name:Some(name.to_string())};
            let index=self.output.len();let item=self.tool_item(&block,index,"in_progress");events.push(self.event("response.output_item.added",json!({"output_index":index,"item":item})));self.open=Some(block);
        }
        if let Some(text)=function["arguments"].as_str().filter(|s|!s.is_empty()) {
            let block=self.open.as_mut().filter(|b|b.kind=="tool").ok_or_else(||bad("tool argument delta without header"))?;block.text.push_str(text);
            let index=self.output.len();events.push(self.event("response.function_call_arguments.delta",json!({"item_id":format!("fc_{}_{}",self.public,index),"output_index":index,"delta":text})));
        }
        Ok(events)
    }
    fn progress(&mut self, progress: Value) -> Vec<Value> {
        vec![self.event(
            "response.in_progress",
            json!({"response":self.pending(),"prompt_progress":progress}),
        )]
    }
    fn idle(&mut self) -> Vec<Value> {
        let mut events = self.begin();
        if self.open.is_none() && self.output.is_empty() {
            events.push(self.event("response.in_progress", json!({"response":self.pending()})));
        } else {
            events.push(json!({"comment":"justfit-keepalive"}));
        }
        events
    }
    pub(super) fn finish(&mut self, chat: &Value, reasoning_open: bool) -> (Vec<Value>, Value) {
        let incomplete = chat["choices"][0]["finish_reason"] == "length";
        let cut = incomplete
            && self
                .open
                .as_ref()
                .is_some_and(|b| b.kind != "reasoning" || reasoning_open);
        let mut events = self.close(cut);
        if self.output.iter().all(|v| v["type"] == "reasoning") {
            events.extend(self.open("text"));
            events.extend(self.close(incomplete));
        }
        let mut response = complete(
            &self.model,
            &self.public,
            chat,
            self.store,
            &self.previous,
            reasoning_open,
        );
        response["output"] = json!(self.output);
        response["text"]["format"] = self.text_format.clone();
        response["parallel_tool_calls"] = json!(self.parallel);
        if self.output.iter().any(|v|v["type"]=="function_call"){response["end_turn"]=json!(false);}
        (events, response)
    }
}
pub(super) fn stream_fixture(body: &Value) -> Result<Value> {
    let mut state = StreamState::new("model".into(), "fixed".into(), 7, true, Value::Null);
    state.parallel=body.get("parallel").and_then(Value::as_bool).unwrap_or(true);state.namespaces=body.get("namespaces").cloned().unwrap_or(json!({}));
    let mut events = state.begin();
    let mut content = String::new();
    let mut reasoning = String::new();
    for step in body["steps"]
        .as_array()
        .ok_or_else(|| bad("steps required"))?
    {
        if let Some(delta)=step.get("tool"){events.extend(state.tool(delta)?);continue;}
        if step["idle"] == true {
            events.extend(state.idle());
            continue;
        }
        if let Some(progress) = step.get("progress") {
            events.extend(state.progress(progress.clone()));
            continue;
        }
        let field = step["field"]
            .as_str()
            .ok_or_else(|| bad("field required"))?;
        let text = step["text"].as_str().ok_or_else(|| bad("text required"))?;
        if field == "reasoning_content" {
            reasoning.push_str(text);
        } else {
            content.push_str(text);
        }
        events.extend(state.delta(field, text));
    }
    let chat = json!({"created":7,"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13,"prompt_tokens_details":{"cached_tokens":0},"completion_tokens_details":{"reasoning_tokens":0}},"choices":[{"finish_reason":body["reason"],"message":{"reasoning_content":reasoning,"content":content}}]});
    let (tail, response) = state.finish(&chat, body["reasoning_open"].as_bool().unwrap_or(false));
    events.extend(tail);
    let event = if response["status"] == "incomplete" {
        "response.incomplete"
    } else {
        "response.completed"
    };
    events.push(state.event(event, json!({"response":response})));
    Ok(json!(events))
}

pub(super) async fn post(
    s: Service,
    mut body: Value,
    permit: Arc<OwnedSemaphorePermit>,
    mut _body_permit: super::BodyPermit,
    received: std::time::Instant,
) -> Result<Response> {
    if body
        .get("stream")
        .is_some_and(|v| !v.is_null() && !v.is_boolean())
    {
        return Err(bad("stream must be a boolean"));
    }
    let streaming = body["stream"] == true;
    if body
        .get("return_progress")
        .is_some_and(|v| !v.is_boolean() || (v == &json!(true) && !streaming))
    {
        return Err(bad(
            "return_progress requires stream: true and must be a boolean",
        ));
    }
    let store = body
        .get("store")
        .filter(|v| !v.is_null())
        .map_or(Ok(true), |v| {
            v.as_bool().ok_or_else(|| bad("store must be a boolean"))
        })?;
    let previous = body["previous_response_id"].clone();
    if !previous.is_null() && !previous.as_str().is_some_and(|s| !s.is_empty()) {
        return Err(bad("previous_response_id must be a non-empty string"));
    }
    let mut items = Vec::new();
    let mut history_permit = None;
    if let Some(id) = previous.as_str() {
        let record = s.responses.record(id).ok_or_else(|| {
            ApiError(
                404,
                "previous response not found".into(),
                "previous_response_not_found".into(),
            )
        })?;
        history_permit = Some(s.body_bytes.reserve(record.history.len()).map_err(|mut error| {
            error.1="retained input capacity is exhausted; retry shortly".into();error
        })?);
        items = serde_json::from_slice(&record.history).map_err(|_| {
            ApiError(
                500,
                "stored history invalid".into(),
                "internal_server_error".into(),
            )
        })?;
    }
    match &body["input"] {
        Value::String(text) => items.push(json!({"type":"message","role":"user","content":text})),
        Value::Array(input) => items.extend(input.iter().cloned()),
        _ => return Err(bad("input must be a non-empty string or array")),
    };
    body["input"] = json!(items);
    let mut converted = crate::responses::convert(&body)?;
    let text_format=response_text_format(&converted["chat"]);
    converted["chat"]["stream"] = json!(streaming);
    if let Some(progress) = body.get("return_progress") {
        converted["chat"]["return_progress"] = progress.clone();
    }
    let mut prepared = s.prepare(&converted["chat"], false).await?;
    let has_tools=prepared.tools.is_some();
    let parallel=prepared.tools.as_ref().is_none_or(|p|p.parallel);
    let namespaces=converted["namespaces"].clone();
    prepare_deadline(&mut prepared, received)?;
    if let Some(mut previous)=history_permit.take(){_body_permit.absorb(&mut previous);}
    _body_permit.retain(&prepared,converted["chat"].get("response_format"),if store{&items}else{&[]})?;
    if !store{items.clear();items.shrink_to_fit();}
    drop(converted);drop(body);
    let mut ticket = s.engine.submit_prepared(&prepared)?;
    prepared.frame.payload.truncate(94);
    prepared.frame.payload.shrink_to_fit();
    if streaming {
        let keepalive = stream_ready(&mut ticket, &mut prepared.timeout).await?;
        let mut state = StreamState::new(
            s.front.response_model.clone(),
            prepared.public_id.clone(),
            prepared.created_at,
            store,
            previous.clone(),
        );
        state.text_format=text_format;
        state.parallel=parallel;state.namespaces=namespaces;
        let (sender, receiver) = mpsc::channel(16);
        tokio::spawn(async move {
            let _body_permit = _body_permit;
            let run = async {
                if keepalive { send_event(&sender, &json!({"comment":"justfit-keepalive"})).await?; }
                for event in state.begin() { send_event(&sender, &event).await?; }
                let (chat_sender, mut chat_receiver) = mpsc::channel(16);
                let producer = async {
                    let result = generate(&s, prepared, ticket, Some(&chat_sender), false, permit).await;
                    drop(chat_sender);
                    result
                };
                let stream_state = &mut state;
                let output = &sender;
                let consumer = async move {
                    loop {
                        let chunk = tokio::select! {
                            chunk = chat_receiver.recv() => chunk,
                            _ = output.closed() => return Err(ApiError(499, "client disconnected".into(), "cancelled".into())),
                        };
                        let Some(Ok(bytes)) = chunk else { break; };
                        for event in chat_events(stream_state, &bytes)? { send_event(output, &event).await?; }
                    }
                    Ok::<_, ApiError>(())
                };
                // Dropping the internal receiver on disconnect cancels the
                // existing native ticket through generate's sender.closed().
                let (collected, forwarded) = tokio::join!(producer, consumer);
                forwarded?;
                let collected = collected?;
                let (events, mut response) = state.finish(&collected.response, collected.reasoning_open);
                for event in events { send_event(&sender, &event).await?; }
                if store {
                    let mut history = items;
                    history.extend(response["output"].as_array().unwrap().iter().cloned());
                    if !s.responses.put(&response, &json!(history))? { response["store"] = json!(false); }
                }
                let name = if response["status"] == "incomplete" { "response.incomplete" } else { "response.completed" };
                send_event(&sender, &state.event(name, json!({"response":response}))).await
            }.await;
            if let Err(error) = run {
                let mut response = state.pending();
                response["status"] = json!("failed");
                response["output"] = json!(state.output);
                response["error"] = error.body()["error"].clone();
                let _ = send_event(
                    &sender,
                    &state.event("response.failed", json!({"response":response})),
                )
                .await;
            }
        });
        let mut response = Response::new(Body::from_stream(ReceiverStream::new(receiver)));
        response
            .headers_mut()
            .insert("content-type", "text/event-stream".parse().unwrap());
        response
            .headers_mut()
            .insert("cache-control", "no-cache".parse().unwrap());
        return Ok(response);
    }
    if has_tools {
        let mut state=StreamState::new(s.front.response_model.clone(),prepared.public_id.clone(),prepared.created_at,store,previous.clone());state.text_format=text_format;state.parallel=parallel;state.namespaces=namespaces;
        let (chat_sender,mut chat_receiver)=mpsc::channel(16);
        let producer=async{let result=generate(&s,prepared,ticket,Some(&chat_sender),false,permit.clone()).await;drop(chat_sender);result};
        let consumer=async{while let Some(chunk)=chat_receiver.recv().await{let bytes=chunk.map_err(|_|bad("internal response stream failed"))?;chat_events(&mut state,&bytes)?;}Ok::<_,crate::ApiError>(())};
        let (collected,drained)=tokio::join!(producer,consumer);drained?;let collected=collected?;let (_,mut response)=state.finish(&collected.response,collected.reasoning_open);
        if store{let mut history=items;history.extend(response["output"].as_array().unwrap().iter().cloned());if !s.responses.put(&response,&json!(history))?{response["store"]=json!(false);}}
        return Ok(reply(response));
    }
    let collected = generate(&s, prepared, ticket, None, false, permit.clone()).await?;
    let chat = collected.response;
    let public = chat["id"]
        .as_str()
        .and_then(|s| s.strip_prefix("chatcmpl-"))
        .ok_or_else(|| bad("response id unavailable"))?;
    let mut response = complete(
        &s.front.response_model,
        public,
        &chat,
        store,
        &previous,
        collected.reasoning_open,
    );
    response["text"]["format"]=text_format;
    if store {
        let mut history = items;
        history.extend(response["output"].as_array().unwrap().iter().cloned());
        if !s.responses.put(&response, &json!(history))? {
            response["store"] = json!(false);
        }
    }
    drop(history_permit);
    Ok(reply(response))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn structured_format_is_preserved_in_created_completed_and_stored_response() {
        let expected=json!({"type":"json_schema","name":"math","strict":true,"schema":{"type":"object"}});
        let format=response_text_format(&json!({"response_format":{"type":"json_schema","json_schema":{"name":"math","strict":true,"schema":{"type":"object"}}}}));
        assert_eq!(format,expected);
        let mut state=StreamState::new("model".into(),"fixed".into(),7,true,Value::Null);
        state.text_format=format;
        assert_eq!(state.begin()[0]["response"]["text"]["format"],expected);
        let chat=json!({"created":7,"choices":[{"finish_reason":"stop","message":{"content":"{}"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2,"completion_tokens_details":{"reasoning_tokens":0}}});
        let (_,response)=state.finish(&chat,false);
        assert_eq!(response["text"]["format"],expected);
        let store=Store::new(4096);assert!(store.put(&response,&json!([])).unwrap());
        assert_eq!(store.get("resp_fixed").unwrap()["text"]["format"],expected);
    }
    #[test]
    fn chat_transport_preserves_response_reasoning_text_and_progress() {
        let mut state = StreamState::new("model".into(), "fixed".into(), 7, true, Value::Null);
        let mut events = state.begin();
        for chunk in [
            json!({"choices":[{"delta":{"reasoning_content":"check"}}]}),
            json!({"choices":[{"delta":{"content":"323"}}]}),
            json!({"prompt_progress":{"processed":12},"choices":[{"delta":{}}]}),
        ] {
            events.extend(
                chat_events(&mut state, format!("data: {}\n\n", chunk).as_bytes()).unwrap(),
            );
        }
        assert!(chat_events(&mut state, b"data: [DONE]\n\n")
            .unwrap()
            .is_empty());
        let chat = json!({"created":7,"choices":[{"finish_reason":"stop","message":{"reasoning_content":"check","content":"323"}}],"usage":{"prompt_tokens":12,"completion_tokens":3,"total_tokens":15,"completion_tokens_details":{"reasoning_tokens":1}}});
        let (tail, response) = state.finish(&chat, false);
        events.extend(tail);
        assert_eq!(response["output"][0]["summary"][0]["text"], "check");
        assert_eq!(response["output"][1]["content"][0]["text"], "323");
        assert!(events
            .iter()
            .any(|v| v["prompt_progress"]["processed"] == 12));
        for (index, event) in events.iter().enumerate() {
            assert_eq!(event["sequence_number"], index);
        }
        assert!(chat_events(&mut state, b"data: invalid\n\n").is_err());
    }
    #[test]
    fn strict_byte_lru_retains_immutable_records_across_eviction() {
        let one = json!({"id":"resp_one","output":[]});
        let two = json!({"id":"resp_two","output":[]});
        let history = json!([{"role":"user","content":"one"}]);
        let size = encode(&one).unwrap().len() + encode(&history).unwrap().len();
        let store = Store::new(size);
        assert!(store.put(&one, &history).unwrap());
        let held = store.record("resp_one").unwrap();
        assert!(store.put(&two, &history).unwrap());
        assert!(store.get("resp_one").is_err());
        assert_eq!(
            serde_json::from_slice::<Value>(&held.response).unwrap(),
            one
        );
        assert_eq!(store.stats()["evictions"], 1);
        assert_eq!(store.delete("resp_two").unwrap()["deleted"], true);
        assert_eq!(store.stats()["bytes"], 0);
    }
    #[test]
    fn oversize_record_never_evicts_previous_response() {
        let store = Store::new(120);
        let small = json!({"id":"resp_one"});
        assert!(store.put(&small, &json!([])).unwrap());
        assert!(!store
            .put(&json!({"id":"resp_big","text":"a".repeat(200)}), &json!([]))
            .unwrap());
        assert_eq!(store.get("resp_one").unwrap(), small);
    }
}
