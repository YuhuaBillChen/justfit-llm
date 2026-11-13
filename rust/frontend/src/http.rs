//! Development HTTP text service. Remaining APIs explicitly refuse as parity gaps.
mod responses;
mod anthropic;
mod scoring;
mod body_budget;
mod shutdown;
pub mod connections;
use body_budget::Permit as BodyPermit;
pub fn responses_fixture(body: &Value) -> Value {
    responses::fixture(body)
}
pub fn responses_stream_fixture(body: &Value) -> Result<Value> {
    responses::stream_fixture(body)
}
use crate::{
    bad,
    engine::{u32_at, u64_at, Engine, Ticket},
    generation::{Prepared, Splitter},
    normalize_path, ApiError, Frontend, Result,
};
use axum::{
    body::{to_bytes, Body, Bytes, HttpBody},
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Router,
};
use justfit_native_client_prototype::{DONE, PROGRESS, START, TOKENS};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    convert::Infallible,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, OwnedSemaphorePermit, Semaphore};
use tokio_stream::wrappers::ReceiverStream;

#[derive(Deserialize)]
pub struct Config {
    pub tokenizer: PathBuf,
    #[serde(default)]
    pub chat_templates: Option<crate::chat_templates::Templates>,
    pub model: String,
    pub native: Vec<String>,
    pub port: u16,
    #[serde(default = "bind_host")]
    pub host: String,
    #[serde(default)]
    pub allowed_hosts: Vec<String>,
    #[serde(default)]
    pub allowed_origins: Vec<String>,
    #[serde(default = "body_limit")]
    pub max_request_bytes: usize,
    #[serde(default)]
    pub vision: Option<crate::vision::Config>,
    #[serde(default)]
    pub thinking_key: Option<PathBuf>,
    #[serde(default)]
    pub request_timeout: Option<f64>,
    #[serde(default = "webui_enabled")]
    pub webui: bool,
    #[serde(default = "queue_size")]
    pub queue_size: usize,
    #[serde(default)]
    pub served_model_names: Vec<String>,
    #[serde(default)]
    pub announce_served_name: bool,
    #[serde(default)]
    pub documents: Option<crate::documents::Config>,
    #[serde(flatten)]
    pub preparation: crate::preparation_options::Options,
}
fn body_limit() -> usize {
    128 * 1024 * 1024
}
fn bind_host() -> String { "127.0.0.1".into() }
fn webui_enabled() -> bool { true }
fn queue_size() -> usize { 32 }
#[derive(Clone)]
struct Service {
    preparation_queue: Arc<crate::preparation_queue::Queue>,
    instance: Value,
    connections: Arc<connections::Gate>,
    front: Arc<Frontend>,
    engine: Arc<Engine>,
    slots: Arc<Semaphore>,
    token_slots: Arc<Semaphore>,
    queue_size: usize,
    body_bytes: Arc<body_budget::Budget>,
    key: Option<String>,
    limit: usize,
    responses: Arc<responses::Store>,
    vision: Option<Arc<crate::vision::Vision>>,
    thinking_path: PathBuf,
    thinking: Arc<std::sync::OnceLock<Result<Arc<crate::thinking::Codec>>>>,
    grammar: Arc<crate::grammar::Factory>,
    access: Arc<crate::access::AccessPolicy>,
    request_timeout: Duration,
    webui: bool,
    shutdown: tokio::sync::watch::Receiver<bool>,
}
impl Service {
    async fn status(&self)->Result<Value>{
        let mut status=self.engine.status().await?;
        status["instance"]=self.instance.clone();
        status["latency"]=self.front.latencies.snapshot();
        status["chat_template"]=self.front.chat_templates.snapshot();
        status["tokenizer_cache"]=self.front.prompt_tokenizer.snapshot();
        status["grammar_cache"]=self.grammar.snapshot();
        status["image_cache"]=self.vision.as_ref().map(|vision|vision.image_snapshot()).unwrap_or_else(||self.front.images.snapshot(0));
        status["frontend"]=self.preparation_queue.snapshot();
        status["response_store"]=self.responses.stats();
        status["http"]=json!({"requests":{"active":self.queue_size-self.slots.available_permits(),"capacity":self.queue_size},"token_counts":{"active":self.queue_size-self.token_slots.available_permits(),"capacity":self.queue_size},"max_request_bytes":self.limit,"request_body_bytes":self.body_bytes.stats(),"connections":self.connections.snapshot()});
        status["vision"]=json!(self.vision.is_some());
        status["input_modalities"]=json!(self.vision.as_ref().map(|vision|vision.input_modalities()).unwrap_or_else(||vec!["text".into()]));
        if let Some(vision)=&self.vision{status["ane_vision"]=vision.snapshot();}
        Ok(status)
    }

    async fn prepare(&self, body: &Value, raw: bool) -> Result<Prepared> {
        let preparation=self.preparation_queue.enter(crate::deadline::request(body)?).await?;
        let id = self.engine.next_id();
        let schema=if raw {None}else{crate::grammar::response_schema(body)?};
        let policy=if raw{None}else{crate::tools::normalize(body)?.1.map(Arc::new)};
        let tool_constrained=policy.as_ref().is_some_and(|p|p.constrained||schema.is_some());
        if (schema.is_some()||tool_constrained)&&body["ignore_eos"]==true{return Err(bad("ignore_eos cannot be combined with constrained tool calls, tool_choice none or structured output"));}
        if (schema.is_some()||policy.as_ref().is_some_and(|p|!p.schemas.is_empty())) && body.get("stop").is_some_and(|value| value.as_str().is_some_and(|s|!s.is_empty()) || value.as_array().is_some_and(|items|!items.is_empty())) {
                return Err(bad("stop cannot be combined with tools or structured output"));
        }
        let input=if schema.is_some() || body["response_format"]==json!({"type":"text"}) {
            let mut input=body.clone();input.as_object_mut().unwrap().remove("response_format");std::borrow::Cow::Owned(input)
        } else {std::borrow::Cow::Borrowed(body)};
        // Keep one public document cache/render queue across generation and
        // token-only requests, before forwarding prepared images to the driver.
        let document_cancel = crate::documents::Cancellation::new();
        let expanded = if !raw && crate::documents::has_documents(input.as_ref()) {
            if self.vision.is_none() {return Err(crate::vision::unavailable())}
            let documents=self.front.documents.clone();let body=input.into_owned();
            let flag=document_cancel.flag.clone();
            let guard=preparation.clone();
            let body=tokio::task::spawn_blocking(move ||{let _guard=guard;documents.expand(&body,&||crate::documents::Cancellation::check(&flag))})
                .await.map_err(|_|bad("document preparation failed"))??;
            std::borrow::Cow::Owned(body)
        } else {input};
        let input=expanded;
        let mut prepared=if !raw && crate::vision::has_media(&input) {
            let _images_timer=self.front.latencies.measure("images");
            self.vision
                .as_ref()
                .ok_or_else(crate::vision::unavailable)?
                .prepare(input.into_owned(), id)
                .await?
        } else {
            let front=self.front.clone();let input=input.into_owned();let guard=preparation.clone();
            tokio::task::spawn_blocking(move||{let _guard=guard;front.generation(&input,id,raw)}).await.map_err(|_|bad("prompt preparation failed"))??
        };
        if tool_constrained {
            let guard=preparation.clone();let grammar=self.grammar.clone();let policy=policy.clone().unwrap();let schema=schema.clone();let thinking=prepared.thinking;let metrics=self.front.latencies.clone();
            prepared.constraint=Some(tokio::spawn(async move{let _guard=guard;let _timer=metrics.measure("grammar");grammar.create_tools(policy,schema,thinking).await}).await.map_err(|_|bad("grammar compiler failed"))??);
            prepared.frame.payload[9]=1;
        }else if let Some(schema)=&schema {
            let guard=preparation.clone();let grammar=self.grammar.clone();let schema=schema.clone();let thinking=prepared.thinking;let metrics=self.front.latencies.clone();
            prepared.constraint=Some(tokio::spawn(async move{let _guard=guard;let _timer=metrics.measure("grammar");grammar.create(schema,thinking).await}).await.map_err(|_|bad("grammar compiler failed"))??);prepared.frame.payload[9]=1;
        }
        prepared.tools=policy;prepared.response_schema=schema;
        Ok(prepared)
    }
}
fn response_error(e: ApiError) -> Response {
    let mut r = (
        StatusCode::from_u16(e.0).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        axum::Json(e.body()),
    )
        .into_response();
    if e.0 == 401 {
        r.headers_mut()
            .insert("www-authenticate", "Bearer".parse().unwrap());
    }
    if e.0 == 503 {
        r.headers_mut().insert("retry-after", "1".parse().unwrap());
    }
    r
}
fn reply(v: Value) -> Response {
    axum::Json(v).into_response()
}
fn head(mut response: Response) -> Response {
    if let Some(length) = response.body().size_hint().exact() {
        response
            .headers_mut()
            .insert("content-length", length.to_string().parse().unwrap());
    }
    *response.body_mut() = Body::empty();
    response
}
fn header<'a>(h: &'a HeaderMap, name: &str) -> Result<&'a str> {
    let all: Vec<_> = h.get_all(name).iter().collect();
    if all.len() != 1 {
        return Err(bad(format!("exactly one {name} header is required")));
    }
    all[0].to_str().map_err(|_| bad(format!("invalid {name}")))
}
async fn handler(State(s): State<Service>, r: Request) -> Response {
    if let Some(info)=r.extensions().get::<axum::extract::ConnectInfo<connections::Info>>() {
        let length=header(r.headers(),"content-length").ok().filter(|text|!text.is_empty()&&text.bytes().all(|byte|byte.is_ascii_digit())).and_then(|text|text.parse::<usize>().ok());
        let unread=length.unwrap_or_else(||if r.headers().contains_key("content-length")||r.headers().contains_key("transfer-encoding"){s.limit}else{0});
        info.0.body(unread,Duration::from_secs_f64(30.+unread.min(s.limit) as f64/(512.*1024.)));
    }
    let messages=normalize_path(r.uri().path()).starts_with("/v1/messages");
    let systemone=normalize_path(r.uri().path())=="/v1/systemone";
    let allowed = s.access.allowed_origin(r.headers());
    let preflight = (r.method() == "OPTIONS").then(|| r.headers().clone());
    let (origin, result) = match allowed {
        Ok(origin) => {
            let mut closing=s.shutdown.clone();
            let result=tokio::select!{biased;
                _=shutdown::cancelled(&mut closing)=>Err(shutdown::error()),
                result=handle(s,r)=>result,
            };
            (origin,result)
        },
        Err(error) => (None, Err(error)),
    };
    let mut response = match result {
        Ok(r) => r,
        Err(e) if messages=>{let payload=anthropic::error_payload(&e);let mut response=response_error(e);*response.body_mut()=Body::from(safe_json(&payload));response},
        Err(mut e) if systemone&&e.0==503=>{e.0=529;response_error(e)},
        Err(e) => response_error(e),
    };
    let preflight = preflight.as_ref().filter(|_| response.status() == StatusCode::NO_CONTENT);
    crate::access::cors_headers(response.headers_mut(), origin.as_deref(), preflight);
    response.headers_mut().insert("connection","close".parse().unwrap());
    response
}
async fn handle(s: Service, r: Request) -> Result<Response> {
    let received = Instant::now();
    let path = normalize_path(r.uri().path());
    let method = r.method().as_str().to_string();
    s.access.authenticate(
        r.headers(),
        &path,
        &method,
        s.key.as_deref(),
    )?;
    if method == "OPTIONS" {
        let mut response = StatusCode::NO_CONTENT.into_response();
        response.headers_mut().insert("allow", "GET, HEAD, POST, DELETE, OPTIONS".parse().unwrap());
        response.headers_mut().insert("content-length", "0".parse().unwrap());
        return Ok(response);
    }
    if let Some(id) = path.strip_prefix("/v1/responses/").filter(|id| {
        id.len() > 5
            && id.starts_with("resp_")
            && id[5..]
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'_')
    }) {
        if method == "GET" || method == "HEAD" {
            let response = reply(s.responses.get(id)?);
            return Ok(if method == "HEAD" {
                head(response)
            } else {
                response
            });
        }
        if method == "DELETE" {
            return Ok(reply(s.responses.delete(id)?));
        }
    }
    if method == "GET" || method == "HEAD" {
        let models = if path == "/v1/models" || path.starts_with("/v1/models/") {
            Some(crate::models::rows(&s.front.model_names, &s.front.response_model, s.front.context,
                s.vision.is_some(), &s.vision.as_ref().map(|v|v.input_modalities()).unwrap_or_else(||vec!["text".into()])))
        } else { None };
        if let Some(name) = path.strip_prefix("/v1/models/") {
            let row = models.as_ref().unwrap().iter().find(|row|row["id"].as_str()==Some(name))
                .ok_or_else(||ApiError(404,"model not found".into(),"model_not_found".into()))?;
            let response = reply(row.clone());
            return Ok(if method == "HEAD" { head(response) } else { response });
        }
        let response = match path.as_str() {
            "/" | "/index.html" if s.webui => (
                [("content-type", "text/html; charset=utf-8")],
                include_bytes!("../../../source/server/chat.html").as_slice(),
            ).into_response(),
            "/favicon.ico" if s.webui => (
                [("content-type", "image/svg+xml")],
                include_bytes!("../../../source/server/favicon.svg").as_slice(),
            ).into_response(),
            "/health" => reply(json!({"status":"ok"})),
            "/ready" => {
                if s.engine.status().await?["ready"] == true {
                    reply(json!({"status":"ready"}))
                } else {
                    (
                        StatusCode::SERVICE_UNAVAILABLE,
                        axum::Json(json!({"status":"unavailable"})),
                    )
                        .into_response()
                }
            }
            "/status" => reply(s.status().await?),
            "/metrics" => ([("content-type","text/plain; version=0.0.4; charset=utf-8")],crate::telemetry::prometheus(&s.status().await?)).into_response(),
            "/v1/models" => {
                let data = models.unwrap();
                let typed: Vec<_> = data.iter().map(|row|json!({"name":row["id"],"description":"JustFit resident model","release_date":""})).collect();
                reply(json!({"object":"list","data":data,"models":typed}))
            }
            _ => return Err(ApiError(404, "not found".into(), "not_found".into())),
        };
        return Ok(if method == "HEAD" {
            head(response)
        } else {
            response
        });
    }
    if method != "POST" {
        return Err(ApiError(404, "not found".into(), "not_found".into()));
    }
    if ![
        "/tokenize",
        "/apply-template",
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/responses",
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/v1/judgments",
        "/v1/systemone",
    ]
    .contains(&path.as_str())
    {
        return Err(ApiError(404,"not found".into(),"not_found".into()));
    }
    let local = matches!(path.as_str(), "/tokenize" | "/apply-template" | "/v1/messages/count_tokens");
    if !local {if let Some(error)=s.engine.refusal(){return Err(error)}}
    let admission = if local {
        &s.token_slots
    } else {
        &s.slots
    };
    let permit = admission.clone().try_acquire_owned().map_err(|_| {
        ApiError(
            503,
            "frontend request capacity is exhausted".into(),
            "frontend_overloaded".into(),
        )
    })?;
    let permit=Arc::new(permit);
    let mut request_timer=s.front.latencies.measure("http_request");
    request_timer.hold(permit.clone());
    if r.headers().contains_key("transfer-encoding") {
        return Err(bad("transfer encoding is not supported"));
    }
    if r.headers().contains_key("content-encoding")
        && header(r.headers(), "content-encoding")?
            .trim()
            .to_lowercase()
            != "identity"
    {
        return Err(ApiError(
            415,
            "content encoding is not supported".into(),
            "invalid_request_error".into(),
        ));
    }
    let mime = header(r.headers(), "content-type")?
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_lowercase();
    if mime != "application/json" && !(mime.starts_with("application/") && mime.ends_with("+json"))
    {
        return Err(ApiError(
            415,
            "Content-Type must be application/json".into(),
            "invalid_request_error".into(),
        ));
    }
    let length = header(r.headers(), "content-length")?;
    if length.is_empty() || !length.bytes().all(|b| b.is_ascii_digit()) {
        return Err(bad("invalid Content-Length header"));
    }
    let length = length
        .parse::<usize>()
        .map_err(|_| bad("invalid Content-Length header"))?;
    if length == 0 {
        return Err(bad("request body must not be empty"));
    }
    if length > s.limit {
        return Err(ApiError(
            413,
            format!("request body is {length} bytes; limit is {} bytes (--max-request-size)", s.limit),
            "request_too_large".into(),
        ));
    }
    let _body_permit = s.body_bytes.reserve(length)?;
    let upload_timer=s.front.latencies.measure("upload");
    let bytes = tokio::time::timeout(
        Duration::from_secs_f64(30. + length as f64 / (512. * 1024.)).min(s.request_timeout),
        to_bytes(r.into_body(), s.limit),
    )
    .await
    .map_err(|_| {
        ApiError(
            408,
            "request upload timed out".into(),
            "timeout_error".into(),
        )
    })?
    .map_err(|_| bad("request body ended before Content-Length"))?;
    if bytes.len() != length {
        return Err(bad("Content-Length mismatch"));
    }
    let mut body = crate::request_json::body(&bytes, &path)?;
    drop(bytes);
    drop(upload_timer);
    // 2.3 validates the shared deadline before endpoint fields; SystemOne
    // reports that validation failure with its field-oriented 422 dialect.
    let requested = match crate::deadline::request(&body) {
        Ok(timeout) => timeout,
        Err(error) if path == "/v1/systemone" => {
            return Ok(scoring::invalid_fields(vec![crate::systemone::error(
                &["timeout"], &error.1, "value_error") ]));
        }
        Err(error) => return Err(error),
    };
    let timeout = requested.min(s.request_timeout);
    if timeout != crate::deadline::UNLIMITED {
        body["timeout"] = crate::deadline::json_seconds(timeout);
    }
    let remaining = crate::deadline::remaining(timeout, received.elapsed())?;
    let mut response=crate::deadline::wait(remaining, route(s, body, path, permit, _body_permit, received)).await??;
    let body=std::mem::replace(response.body_mut(),Body::empty());
    if !response.headers().contains_key("content-length"){if let Some(length)=body.size_hint().exact(){response.headers_mut().insert("content-length",length.to_string().parse().unwrap());}}
    *response.body_mut()=crate::latency::response_body(body,request_timer);
    Ok(response)
}
async fn route(s: Service, body: Value, path: String, permit: Arc<OwnedSemaphorePermit>,
               mut _body_permit: BodyPermit, received: Instant) -> Result<Response> {
    if path=="/v1/systemone"{return scoring::systemone(s,body,permit,_body_permit,received).await;}
    if path=="/v1/judgments"{return scoring::post(s,body,permit,_body_permit,received).await;}
    if path=="/v1/messages"||path=="/v1/messages/count_tokens" {
        return anthropic::post(s,body,permit,_body_permit,received,path.ends_with("/count_tokens")).await;
    }
    if path == "/v1/responses" {
        return responses::post(s, body, permit, _body_permit, received).await;
    }
    if path == "/tokenize" || path == "/apply-template" {
        if path=="/apply-template"&&crate::vision::has_media(&body)&&s.vision.is_none(){return Err(crate::vision::unavailable())}
        let preparation=s.preparation_queue.enter(crate::deadline::request(&body)?).await?;let front=s.front.clone();
        if path=="/apply-template"&&crate::vision::has_media(&body){
            let value=s.vision.as_ref().unwrap().prompt(body,s.engine.next_id(),true,preparation).await?;
            return Ok(reply(value));
        }
        let cancellation=crate::documents::Cancellation::new();let flag=cancellation.flag.clone();
        let value=tokio::task::spawn_blocking(move||->Result<Value>{let _preparation=preparation;
            if path=="/tokenize"{Ok(json!({"tokens":front.tokenize(&body)?}))}else{front.check_model(&body)?;crate::tools::normalize(&body)?;crate::grammar::response_schema(&body)?;Ok(json!({"prompt":front.apply_template_checked(&body,&||crate::documents::Cancellation::check(&flag))?}))}
        }).await.map_err(|_|bad("prompt preparation failed"))??;
        return Ok(reply(value));
    }
    let stream = body
        .get("stream")
        .filter(|v| !v.is_null())
        .map_or(Ok(false), |v| {
            v.as_bool().ok_or_else(|| bad("invalid streaming options"))
        })?;
    let include_usage = match body.get("stream_options").filter(|v| !v.is_null()) {
        None => false,
        Some(v) => {
            if !v.is_object() {
                return Err(bad("invalid streaming options"));
            }
            v.get("include_usage").map_or(Ok(false), |v| {
                v.as_bool().ok_or_else(|| bad("invalid streaming options"))
            })?
        }
    };
    let mut prepared = s.prepare(&body, path == "/v1/completions").await?;
    prepare_deadline(&mut prepared, received)?;
    _body_permit.retain(&prepared,body.get("response_format"),&[])?;
    drop(body);
    let mut ticket = s.engine.submit_prepared(&prepared)?;
    // Native has consumed the wire bytes. Keep only counts used by progress;
    // the driver's owner retains its budget until generation terminates.
    prepared.frame.payload.truncate(94);
    prepared.frame.payload.shrink_to_fit();
    let (sender, receiver) = mpsc::channel(16);
    if stream {
        // Preserve errors before Start as HTTP errors; long waits begin with the
        // same keepalive boundary as Python, after which errors are SSE events.
        let keepalive = stream_ready(&mut ticket, &mut prepared.timeout).await?;
        tokio::spawn(async move {
            let _body_permit = _body_permit;
            if keepalive
                && sender
                    .send(Ok(Bytes::from_static(b": justfit-keepalive\n\n")))
                    .await
                    .is_err()
            {
                return;
            }
            let result = generate(&s, prepared, ticket, Some(&sender), include_usage, permit).await;
            if let Err(e) = result {
                let _ = sender
                    .send(Ok(Bytes::from(format!(
                        "data: {}\n\ndata: [DONE]\n\n",
                        safe_json(&e.body())
                    ))))
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
    Ok(reply(
        generate(&s, prepared, ticket, None, include_usage, permit)
            .await?
            .response,
    ))
}

type Sender = mpsc::Sender<std::result::Result<Bytes, Infallible>>;
fn safe_json(value: &Value) -> String {
    value
        .to_string()
        .replace('\u{85}', "\\u0085")
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029")
}
async fn stream_ready(ticket: &mut Ticket, timeout: &mut Duration) -> Result<bool> {
    let waiting = Instant::now();
    let early = tokio::time::timeout((*timeout).min(Duration::from_secs(2)), ticket.next()).await;
    let keepalive = match early {
        Ok(frame) => {
            ticket.unread_start(frame?)?;
            false
        }
        Err(_) if waiting.elapsed() >= *timeout => {
            return Err(ApiError(
                504,
                "request timed out".into(),
                "request_timeout".into(),
            ))
        }
        Err(_) => true,
    };
    *timeout = crate::deadline::remaining(*timeout, waiting.elapsed())?;
    Ok(keepalive)
}
fn prepare_deadline(prepared: &mut Prepared, received: Instant) -> Result<()> {
    prepared.received_at=Some(received);
    prepared.timeout = crate::deadline::remaining(prepared.timeout, received.elapsed())?;
    let (absolute, micros) = crate::deadline::wire_fields(prepared.timeout);
    prepared.frame.payload[10..18].copy_from_slice(&absolute.to_le_bytes());
    prepared.frame.payload[18..26].copy_from_slice(&micros.to_le_bytes());
    Ok(())
}
async fn emit(sender: Option<&Sender>, v: Value) -> Result<()> {
    if let Some(sender) = sender {
        tokio::time::timeout(
            Duration::from_secs(30),
            sender.send(Ok(Bytes::from(format!("data: {}\n\n", safe_json(&v))))),
        )
        .await
        .map_err(|_| ApiError(408, "stream write timed out".into(), "timeout_error".into()))?
        .map_err(|_| ApiError(499, "client disconnected".into(), "cancelled".into()))?;
    }
    Ok(())
}
fn chunk(id: &str, created: u64, model: &str, raw: bool, delta: Value, reason: Value) -> Value {
    let choice = if raw {
        json!({"index":0,"text":delta["content"].as_str().unwrap_or(""),"logprobs":null,"finish_reason":reason})
    } else {
        json!({"index":0,"delta":delta,"finish_reason":reason})
    };
    json!({"id":id,"object":if raw{"text_completion"}else{"chat.completion.chunk"},"created":created,"model":model,"choices":[choice]})
}
struct Collected {
    response: Value,
    reasoning_open: bool,
    cut_call: Option<String>,
    stop_sequence: Option<String>,
}
async fn generate(
    s: &Service,
    p: Prepared,
    mut ticket: Ticket,
    sender: Option<&Sender>,
    include_usage: bool,
    _permit: Arc<OwnedSemaphorePermit>,
) -> Result<Collected> {
    let created = p.created_at;
    let mut last_output:Option<Instant>=None;
    let id = format!(
        "{}{}",
        if p.raw { "cmpl-" } else { "chatcmpl-" },
        p.public_id
    );
    let mut splitter = Splitter::new(p.thinking);
    splitter.set_tools(p.tools.is_some());
    let mut projector=p.tools.as_ref().map(|policy|crate::tool_output::Projector::new(policy,p.public_id.clone(),p.response_schema.is_some())).transpose()?;
    let mut stream = s.front.tokenizer.decode_stream(true);
    let eos = s.front.tokenizer.token_to_id("<|im_end|>");
    let mut total = 0usize;
    let mut first_batch = 0usize;
    let mut reasoning_tokens = 0usize;
    let mut reasoning_closed = !p.thinking;
    let mut pending = String::new();
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut matched = 0;
    let mut lane = Value::Null;
    let mut stopped = false;
    let mut consumed=0usize;
    let mut stop_sequence=None;
    // return_token_ids: the generated ids, and whether a stream has sent the prompt's.
    let mut output_ids:Option<Vec<u32>>=(p.prompt_token_ids.is_some()||p.rendered_prompt.is_some()).then(Vec::new);
    let mut prompt_ids_sent=false;
    let deadline = crate::deadline::tokio_deadline(p.timeout);
    let mut started = false;
    loop {
        let event = async {
            if let Some(sender) = sender {
                tokio::select! {
                    v=ticket.next()=>v.map(Some),
                    _=sender.closed()=>Err(ApiError(499,"client disconnected".into(), "cancelled".into())),
                    _=tokio::time::sleep(Duration::from_secs(2))=>Ok(None)
                }
            } else {
                ticket.next().await.map(Some)
            }
        };
        let frame = crate::deadline::wait_until(deadline, event).await??;
        let Some(frame) = frame else {
            if started {
                emit(
                    sender,
                    chunk(&id, created, &s.front.response_model, p.raw, json!({}), Value::Null),
                )
                .await?;
            } else if let Some(sender) = sender {
                let _ = sender
                    .send(Ok(Bytes::from_static(b": justfit-keepalive\n\n")))
                    .await;
            }
            continue;
        };
        match frame.kind {
            START => {
                started = true;
                matched = u32_at(&frame.payload, 12)?;
                lane = json!(u32_at(&frame.payload, 8)?);
                if !p.raw {
                    if p.internal_start {
                        let mut value=chunk(&id,created,&s.front.response_model,p.raw,json!({"role":"assistant","content":""}),Value::Null);
                        value["_native_start"]=json!({"prompt_tokens":u32_at(&p.frame.payload,30)?,"matched_tokens":matched});emit(sender,value).await?;
                        continue;
                    }
                    let mut value = chunk(
                        &id,
                        created,
                        &s.front.response_model,
                        p.raw,
                        json!({"role":"assistant","content":""}),
                        Value::Null,
                    );
                    if let Some(prompt) = &p.prompt_token_ids {
                        value["prompt_token_ids"] = json!(prompt);
                        prompt_ids_sent = true;
                    }
                    emit(sender, value).await?;
                }
            }
            PROGRESS => {
                let mut update = chunk(&id, created, &s.front.response_model, p.raw, json!({}), Value::Null);
                update["prompt_progress"] = json!({"total":u32_at(&p.frame.payload,30)?,"cache":matched,"processed":u32_at(&frame.payload,8)?,"time_ms":u64_at(&frame.payload,12)? as f64/1000.});
                emit(sender, update).await?;
            }
            TOKENS => {
                let now=Instant::now();
                if let Some(previous)=last_output{s.front.latencies.observe("output_interval",now.duration_since(previous).as_secs_f64());}else if let Some(received)=p.received_at{s.front.latencies.observe("http_ttft",now.duration_since(received).as_secs_f64());}
                last_output=Some(now);
                let offset = u32_at(&frame.payload, 8)? as usize;
                let count = u32_at(&frame.payload, 12)? as usize;
                if offset != total
                    || count == 0
                    || frame.payload.len() != 16 + count * 4
                    || total + count > p.budget
                {
                    return Err(ApiError(
                        502,
                        "invalid native token sequence".into(),
                        "protocol_error".into(),
                    ));
                }
                if total == 0 {
                    first_batch = count;
                }
                total += count;
                let frame_start = output_ids.as_ref().map_or(0, Vec::len);
                for i in 0..count {
                    let token = u32_at(&frame.payload, 16 + i * 4)?;
                    if stopped{continue;}
                    consumed+=1;
                    if let Some(ids) = &mut output_ids { ids.push(token); }
                    if !reasoning_closed {
                        if Some(token) == s.front.tokenizer.token_to_id("</think>") {
                            reasoning_closed = true;
                        } else {
                            reasoning_tokens += 1;
                        }
                    }
                    if Some(token) == eos {
                        continue;
                    }
                    if let Some(text) = stream.step(token).map_err(|_| {
                        ApiError(
                            502,
                            "output could not be decoded".into(),
                            "engine_error".into(),
                        )
                    })? {
                        pending.push_str(&text);
                    }
                    // Match Python's decoded-token event boundaries even when
                    // speculative decoding delivers several tokens in one frame.
                    let stop = p.stop.iter().filter_map(|m| pending.find(m)).min();
                    let end = if let Some(index) = stop {
                        stop_sequence=p.stop.iter().find(|marker|pending[index..].starts_with(marker.as_str())).cloned();
                        stopped = true;
                        ticket.cancel();
                        index
                    } else {
                        let mut hold = 0;
                        for m in &p.stop {
                            for (n, _) in m.char_indices().skip(1) {
                                if pending.ends_with(&m[..n]) {
                                    hold = hold.max(n);
                                }
                            }
                        }
                        pending.len() - hold
                    };
                    let ready = pending[..end].to_string();
                    pending = if stopped {
                        String::new()
                    } else {
                        pending[end..].to_string()
                    };
                    for (key, text) in splitter.put(&ready) {
                        if key=="content" {
                            if let Some(projector)=&mut projector {
                                for (kind,value) in projector.put(&text) {
                                    let delta=if kind=="tool"{json!({"tool_calls":[value]})}else{content.push_str(value.as_str().unwrap());json!({"content":value})};
                                    emit(sender,chunk(&id,created,&s.front.response_model,p.raw,delta,Value::Null)).await?;
                                }
                                continue;
                            }
                        }
                        if key == "content" {
                            content.push_str(&text);
                        } else {
                            reasoning.push_str(&text);
                        }
                        let mut delta = json!({});
                        delta[key] = json!(text);
                        emit(
                            sender,
                            chunk(&id, created, &s.front.response_model, p.raw, delta, Value::Null),
                        )
                        .await?;
                    }
                }
                if sender.is_some() && p.prompt_token_ids.is_some() {
                    if let Some(ids) = output_ids.as_ref().filter(|ids| ids.len() > frame_start) {
                        let mut value = chunk(&id, created, &s.front.response_model, p.raw, json!({}), Value::Null);
                        value["choices"][0]["token_ids"] = json!(ids[frame_start..]);
                        if !prompt_ids_sent {
                            if let Some(prompt) = &p.prompt_token_ids {
                                if p.raw { value["choices"][0]["prompt_token_ids"] = json!(prompt); } else { value["prompt_token_ids"] = json!(prompt); }
                            }
                            prompt_ids_sent = true;
                        }
                        emit(sender, value).await?;
                    }
                }
            }
            DONE => {
                let prompt = u32_at(&frame.payload, 9)?;
                let generated = u32_at(&frame.payload, 13)?;
                if frame.payload.len() != 45
                    || frame.payload[8] > 2
                    || generated as usize != total
                    || u32_at(&frame.payload, 41)? != 0
                {
                    return Err(ApiError(
                        502,
                        "invalid native Done event".into(),
                        "protocol_error".into(),
                    ));
                }
                // Native may have verified extra draft tokens in the batch
                // that contained a client stop string. Python reports only
                // the tokens its detokenizer consumed through that stop.
                let generated=if stopped{consumed as u32}else{generated};
                let mut tails = splitter.put(&pending);
                tails.extend(splitter.finish());
                for (key, text) in tails {
                    if key=="content" {
                        if let Some(projector)=&mut projector {
                            for (kind,value) in projector.put(&text) {
                                let delta=if kind=="tool"{json!({"tool_calls":[value]})}else{content.push_str(value.as_str().unwrap());json!({"content":value})};
                                emit(sender,chunk(&id,created,&s.front.response_model,p.raw,delta,Value::Null)).await?;
                            }
                            continue;
                        }
                    }
                    if key == "content" {
                        content.push_str(&text);
                    } else {
                        reasoning.push_str(&text);
                    }
                    let mut delta = json!({});
                    delta[key] = json!(text);
                    emit(
                        sender,
                        chunk(&id, created, &s.front.response_model, p.raw, delta, Value::Null),
                    )
                    .await?;
                }
                let mut reason = if stopped || frame.payload[8] == 0 {
                    "stop"
                } else if frame.payload[8] == 1 {
                    "length"
                } else {
                    "cancelled"
                };
                let mut calls=Vec::new();
                if let Some(projector)=&mut projector {
                    let (final_content,final_calls,tail)=projector.finish(reason=="length");content=final_content;calls=final_calls;
                    for (kind,value) in tail {let delta=if kind=="tool"{json!({"tool_calls":[value]})}else{json!({"content":value})};emit(sender,chunk(&id,created,&s.front.response_model,p.raw,delta,Value::Null)).await?;}
                    if !calls.is_empty()&&reason=="stop"{reason="tool_calls";}
                }
                if reason=="stop" {
                    if let (Some(constraint),Some(schema))=(&p.constraint,&p.response_schema) {
                        let validator=if p.tools.is_some(){Arc::new(jsonschema::validator_for(schema).map_err(|_|bad("invalid response schema"))?)}else{constraint.lock().map_err(|_|bad("grammar unavailable"))?.validator.clone()};
                        let structured=content.clone();
                        tokio::task::spawn_blocking(move || {
                            let invalid=||ApiError(500,"model returned invalid structured output".into(),"invalid_model_output".into());
                            let value=crate::strict_json(structured.as_bytes()).map_err(|_|invalid())?;
                            if !validator.is_valid(&value) {return Err(invalid());}Ok(())
                        }).await.map_err(|_|bad("structured output validation failed"))??;
                    }
                }
                let usage = json!({"prompt_tokens":prompt,"completion_tokens":generated,"total_tokens":prompt+generated,"prompt_tokens_details":{"cached_tokens":matched},"completion_tokens_details":{"reasoning_tokens":reasoning_tokens}});
                let mut message = json!({"role":"assistant","content":if !p.raw && content.is_empty(){Value::Null}else{json!(content)}});
                if !calls.is_empty(){message["tool_calls"]=json!(calls);}
                if !reasoning.is_empty() {
                    message["reasoning_content"] = json!(reasoning);
                }
                let mut choice = if p.raw {
                    json!({"index":0,"text":message["content"],"logprobs":null,"finish_reason":reason})
                } else {
                    json!({"index":0,"message":message,"finish_reason":reason})
                };
                if let (Some(prompt), Some(ids)) = (&p.prompt_token_ids, &output_ids) {
                    choice["token_ids"] = json!(ids);
                    if p.raw { choice["prompt_token_ids"] = json!(prompt); }
                }
                let pp = u64_at(&frame.payload, 17)? as f64 / 1000.;
                let tg = u64_at(&frame.payload, 25)? as f64 / 1000.;
                let wall = u64_at(&frame.payload, 33)? as f64 / 1000.;
                let prefill = prompt.saturating_sub(matched);
                let decode = (generated as usize).saturating_sub(first_batch);
                let mut latency = json!({"start_to_first_token_ms":pp,"first_token_to_done_ms":tg,"wall_ms":wall});
                if generated > 0 && wall >= tg {
                    latency["ttft_ms"] = json!(wall - tg);
                    if wall - tg >= pp {
                        latency["queue_to_start_ms"] = json!(wall - tg - pp);
                        s.front.latencies.observe("native_queue",(wall-tg-pp)/1000.);
                    }
                }
                let stream_rate = if decode > 0 && tg > 0. {
                    decode as f64 * 1000. / tg
                } else {
                    0.
                };
                if stream_rate > 0. {
                    latency["stream_tokens_per_second"] = json!(stream_rate);
                }
                let metrics = json!({"prefill":{"tokens":prefill},"decode":{"tokens":decode},"request_latency":latency,
                    "cache":{"status":if matched>0{"hit"}else{"miss"},"matched_tokens":matched,"lane":lane}});
                let timings = json!({"prompt_n":prompt,"prompt_ms":pp,"prompt_per_second":if pp>0.{prefill as f64*1000./pp}else{0.},"predicted_n":generated,"predicted_ms":tg,"predicted_per_second":stream_rate,"cache_n":matched});
                let mut terminal = chunk(
                    &id,
                    created,
                    &s.front.response_model,
                    p.raw,
                    json!({}),
                    json!(reason),
                );
                terminal["timings"] = timings.clone();
                emit(sender, terminal).await?;
                if include_usage {
                    let mut last =
                        chunk(&id, created, &s.front.response_model, p.raw, json!({}), Value::Null);
                    last["choices"] = json!([]);
                    last["usage"] = usage.clone();
                    last["metrics"] = metrics.clone();
                    emit(sender, last).await?;
                }
                if let Some(sender) = sender {
                    let _ = sender
                        .send(Ok(Bytes::from_static(b"data: [DONE]\n\n")))
                        .await;
                }
                if let (Some((text, prompt)), Some(ids)) = (&p.rendered_prompt, &output_ids) {
                    if !stopped && frame.payload[8] == 0 {
                        if let Ok(reply) = s.front.tokenizer.decode(ids, false) {
                            let mut tokens = prompt.clone();
                            tokens.extend_from_slice(ids);
                            s.front.prompt_tokenizer.remember(format!("{text}{reply}"), tokens);
                        }
                    }
                }
                let mut response = json!({"id":id,"object":if p.raw{"text_completion"}else{"chat.completion"},"created":created,"model":s.front.response_model,"choices":[choice],"usage":usage,"metrics":metrics,"timings":timings});
                if let (false, Some(prompt)) = (p.raw, &p.prompt_token_ids) {
                    response["prompt_token_ids"] = json!(prompt);
                }
                return Ok(Collected {
                    response,
                    reasoning_open: splitter.reasoning_open(),
                    cut_call: projector.as_ref().and_then(|p|p.open_call_id()),
                    stop_sequence,
                });
            }
            _ => {
                return Err(ApiError(
                    502,
                    "unexpected native event".into(),
                    "protocol_error".into(),
                ))
            }
        }
    }
}

pub async fn serve(config: Config) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let input_capacity=body_budget::capacity(config.max_request_bytes).map_err(|e|e.1)?;
    let preparation=config.preparation.clone().environment().map_err(|e|e.1)?;
    if config.queue_size == 0 || config.queue_size > Semaphore::MAX_PERMITS - 64 {
        return Err("queue_size must be a positive number of requests within the platform limit".into());
    }
    let access = Arc::new(crate::access::AccessPolicy::new(&config.host, &config.allowed_hosts, &config.allowed_origins).map_err(|e| e.1)?);
    let request_timeout = crate::deadline::seconds(config.request_timeout).map_err(|e| e.1)?;
    let (response_model, model_names) = crate::models::names(&config.model, &config.served_model_names, config.announce_served_name).map_err(|e|e.1)?;
    let key=std::env::var("JUSTFIT_API_KEY").ok();
    if key.as_ref().is_some_and(|v|v.is_empty()||v.bytes().any(|b|!(33..127).contains(&b))){return Err("API key must contain only visible ASCII characters".into())}
    // Reserve the exact endpoint before tokenizer/model loading, matching 2.3.
    connections::descriptor_limit();
    let listener=tokio::net::TcpListener::bind((config.host.as_str(),config.port)).await?;
    let started_at=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs_f64();
    use sha2::{Digest,Sha256};
    let id=format!("{:x}",Sha256::digest(format!("{started_at:?}-{}",std::process::id()).as_bytes()))[..24].to_string();
    let instance=json!({"id":id,"pid":std::process::id(),"model":config.model,"host":listener.local_addr()?.ip().to_string(),"port":listener.local_addr()?.port(),"started_at":started_at});
    let signals=shutdown::Signals::install()?;
    crate::tokenizer_startup::check(&config.tokenizer).map_err(|e|e.1)?;
    let native=config.native;let queue=config.queue_size;let stop=signals.requested.clone();
    let started=tokio::task::spawn_blocking(move||Engine::spawn_cancellable(&native,queue,stop)).await?;
    let engine=match started {
        Ok(engine)=>Arc::new(engine),
        Err(_) if signals.stopping()=>return Ok(()),
        Err(error)=>return Err(error.1.into()),
    };
    signals.register(&engine);
    if signals.stopping(){signals.finish().await;return Ok(())}
    let mut front = Frontend::load_with_templates(&config.tokenizer, config.model, engine.context,config.chat_templates).map_err(|e|e.1)?;
    front.preparation = preparation;
    front.response_model = response_model;
    front.model_names = model_names;
    if let Some(documents)=config.documents {front.documents=Arc::new(crate::documents::Documents::new(documents));}
    let front = Arc::new(front);
    let mut startup_stop=signals.receiver.clone();
    let vision = match config.vision.filter(|_|engine.vision) {
        Some(vision) => Some(Arc::new(
            match tokio::select! {biased;
                _=shutdown::cancelled(&mut startup_stop)=>None,
                result=crate::vision::Vision::start(
                vision,
                engine.clone(),
                config.tokenizer.clone(),
                front.clone(),
                )=>Some(result),
            } {
                Some(result)=>result.map_err(|e|e.1)?,
                None=>{signals.finish().await;return Ok(())},
            },
        )),
        None => None,
    };
    if signals.stopping(){signals.finish().await;return Ok(())}
    let (listener,connections)=connections::Listener::new(listener,config.queue_size+64)?;
    let service = Service {
        connections,
        preparation_queue: crate::preparation_queue::Queue::new(engine.lanes as usize,front.latencies.clone()),
        instance,
        shutdown: signals.receiver.clone(),
        access,
        request_timeout,
        webui: config.webui,
        front,
        engine: engine.clone(),
        slots: Arc::new(Semaphore::new(config.queue_size)),
        token_slots: Arc::new(Semaphore::new(config.queue_size)),
        queue_size: config.queue_size,
        body_bytes: Arc::new(body_budget::Budget::new(input_capacity)),
        key,
        limit: config.max_request_bytes,
        responses: Arc::new(responses::Store::new(64 * 1024 * 1024)),
        vision,
        thinking_path: config.thinking_key.unwrap_or_else(||PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join("Library/Application Support/JustFit/thinking.key")),
        thinking: Arc::new(std::sync::OnceLock::new()),
        grammar: Arc::new(crate::grammar::Factory::new(config.tokenizer.join("tokenizer.json"))),
    };
    eprintln!(
        "{}",
        json!({"rust_http_ready":true,"port":axum::serve::Listener::local_addr(&listener)?.port(),"native_lanes":engine.lanes,"native_context":engine.context,"full_parity":false})
    );
    let app = Router::new().fallback(handler).with_state(service);
    let mut stop_server=signals.receiver.clone();let mut stop_wait=signals.receiver.clone();
    let serving=axum::serve(listener, app.into_make_service_with_connect_info::<connections::Info>()).with_graceful_shutdown(async move{shutdown::cancelled(&mut stop_server).await});
    let mut serving=std::pin::pin!(std::future::IntoFuture::into_future(serving));
    let result=tokio::select!{biased;
        _=shutdown::cancelled(&mut stop_wait)=>{
            signals.finish().await;
            // Match 2.3's bounded connection drain after native cleanup. A
            // slow reader must not keep a supervisor stop hanging forever.
            match tokio::time::timeout(Duration::from_secs(2),&mut serving).await{Ok(result)=>result,Err(_)=>Ok(())}
        },
        result=&mut serving=>result,
    };
    signals.finish().await;
    result?;
    Ok(())
}

#[cfg(test)]
mod deadline_tests {
    use super::*;
    fn prepared(timeout: Duration) -> Prepared {
        Prepared {
            received_at: None,
            frame: justfit_native_client_prototype::text_request(1, &[1], 1, 0, 0).unwrap(),
            thinking: false, stop: vec![], raw: true, budget: 1, timeout,
            public_id: String::new(), created_at: 0, vision_owner: None,
            constraint: None, tools: None, response_schema: None, internal_start: false,
            prompt_token_ids: None, rendered_prompt: None,
        }
    }
    #[test]
    fn native_deadlines_keep_unlimited_sentinel_and_long_explicit_budget() {
        let mut unlimited = prepared(Duration::MAX);
        prepare_deadline(&mut unlimited, Instant::now() - Duration::from_secs(10)).unwrap();
        assert_eq!(unlimited.timeout, Duration::MAX);
        assert_eq!(u64_at(&unlimited.frame.payload, 10).unwrap(), u64::MAX);
        assert!(u64_at(&unlimited.frame.payload, 18).unwrap() < u64::MAX);
        let mut long = prepared(Duration::from_secs(9001));
        prepare_deadline(&mut long, Instant::now() - Duration::from_secs(1)).unwrap();
        let remaining = u64_at(&long.frame.payload, 18).unwrap();
        assert!((8_999_000_000..=9_000_000_000).contains(&remaining));
        let mut expired = prepared(Duration::from_millis(1));
        assert_eq!(prepare_deadline(&mut expired, Instant::now() - Duration::from_secs(1)).unwrap_err().0,504);
    }
    #[tokio::test]
    async fn invalid_queue_capacity_is_rejected_before_native_start() {
        for value in [0, Semaphore::MAX_PERMITS + 1] {
            let config: Config = serde_json::from_value(json!({
                "model":"model","tokenizer":"not-used","native":["/should-not-start"],
                "port":0,"queue_size":value
            })).unwrap();
            assert_eq!(serve(config).await.unwrap_err().to_string(),
                "queue_size must be a positive number of requests within the platform limit");
        }
    }
    #[tokio::test]
    async fn invalid_server_timeout_is_rejected_before_native_start() {
        for value in [json!(0), json!(-1)] {
            let config: Config = serde_json::from_value(json!({
                "model":"model","tokenizer":"not-used","native":["/should-not-start"],
                "port":0,"request_timeout":value
            })).unwrap();
            let error = serve(config).await.unwrap_err();
            assert_eq!(error.to_string(), "timeout must be positive");
        }
    }
}
