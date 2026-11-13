//! Messages HTTP adapter over the same bounded native generation lifecycle.
use super::{
    generate, prepare_deadline, reply, responses, safe_json, stream_ready, Collected, Sender,
    Service,
};
use crate::{bad, ApiError, Result};
use axum::{
    body::{Body, Bytes},
    response::Response,
};
use serde_json::{json, Value};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, OwnedSemaphorePermit};
use tokio_stream::wrappers::ReceiverStream;

pub(super) fn error_payload(error:&ApiError)->Value {
    let kind=match error.0{401=>"authentication_error",403=>"permission_error",404=>"not_found_error",408|504=>"timeout_error",413=>"request_too_large",429=>"rate_limit_error",503=>"overloaded_error",500.. =>"api_error",_=>"invalid_request_error"};
    json!({"type":"error","error":{"type":kind,"message":error.1}})
}

async fn send(sender: &Sender, value: Value) -> Result<()> {
    let bytes = Bytes::from(format!(
        "event: {}\ndata: {}\n\n",
        value["type"].as_str().unwrap_or("error"),
        safe_json(&value)
    ));
    tokio::time::timeout(Duration::from_secs(30), sender.send(Ok(bytes)))
        .await
        .map_err(|_| ApiError(408, "stream write timed out".into(), "timeout_error".into()))?
        .map_err(|_| ApiError(499, "client disconnected".into(), "cancelled".into()))
}
fn usage(chat: &Value) -> Value {
    let usage = &chat["usage"];
    let input = usage["prompt_tokens"].as_u64().unwrap_or(0);
    let cache = usage["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or(0);
    json!({"input_tokens":input.saturating_sub(cache),"cache_read_input_tokens":cache,"output_tokens":usage["completion_tokens"]})
}
fn stop(collected: &Collected, clamped: bool) -> &'static str {
    if collected.response["choices"][0]["finish_reason"] == "tool_calls" {
        "tool_use"
    } else if collected.response["choices"][0]["finish_reason"] == "length" {
        if clamped {
            "model_context_window_exceeded"
        } else {
            "max_tokens"
        }
    } else if collected.stop_sequence.is_some() {
        "stop_sequence"
    } else {
        "end_turn"
    }
}
fn block(item: &Value, omitted: bool, signature: &str) -> Result<Value> {
    match item["type"].as_str() {
        Some("reasoning") => Ok(
            json!({"type":"thinking","thinking":if omitted{""}else{item["summary"][0]["text"].as_str().unwrap_or("")},"signature":signature}),
        ),
        Some("message") => {
            Ok(json!({"type":"text","text":item["content"][0]["text"].as_str().unwrap_or("")}))
        }
        Some("function_call") => Ok(
            json!({"type":"tool_use","id":item["call_id"],"name":item["name"],"input":if item["status"]=="in_progress"{json!({})}else{crate::strict_json(item["arguments"].as_str().ok_or_else(||bad("missing tool arguments"))?.as_bytes())?}}),
        ),
        _ => Err(bad("invalid output item")),
    }
}
async fn forward(
    sender: &Sender,
    events: Vec<Value>,
    omitted: bool,
    codec: &Arc<crate::thinking::Codec>,
) -> Result<()> {
    for event in events {
        let index = event["output_index"].clone();
        match event["type"].as_str() {
            Some("response.output_item.added")=>{
                let item=&event["item"];send(sender,json!({"type":"content_block_start","index":index,"content_block":block(item,omitted,"")?})).await?;
                if item["type"]=="reasoning"&&omitted{send(sender,json!({"type":"content_block_delta","index":index,"delta":{"type":"thinking_delta","thinking":""}})).await?;}
            },
            Some("response.output_text.delta")=>send(sender,json!({"type":"content_block_delta","index":index,"delta":{"type":"text_delta","text":event["delta"]}})).await?,
            Some("response.reasoning_summary_text.delta") if !omitted=>send(sender,json!({"type":"content_block_delta","index":index,"delta":{"type":"thinking_delta","thinking":event["delta"]}})).await?,
            Some("response.function_call_arguments.delta")=>send(sender,json!({"type":"content_block_delta","index":index,"delta":{"type":"input_json_delta","partial_json":event["delta"]}})).await?,
            Some("response.output_item.done")=>{
                if omitted&&event["item"]["type"]=="reasoning" {
                    let text=event["item"]["summary"][0]["text"].as_str().unwrap_or("").to_string();let codec=codec.clone();
                    let signature=tokio::task::spawn_blocking(move||codec.encode(&text)).await.map_err(|_|bad("thinking encoding failed"))??;
                    send(sender,json!({"type":"content_block_delta","index":index,"delta":{"type":"signature_delta","signature":signature}})).await?;
                }
                send(sender,json!({"type":"content_block_stop","index":index})).await?;
            },
            Some("response.in_progress")=>send(sender,if let Some(progress)=event.get("prompt_progress"){json!({"type":"ping","prompt_progress":progress})}else{json!({"type":"ping"})}).await?,
            _=>{},
        }
    }
    Ok(())
}
pub(super) async fn post(
    s: Service,
    body: Value,
    permit: Arc<OwnedSemaphorePermit>,
    mut body_permit: super::BodyPermit,
    received: Instant,
    count: bool,
) -> Result<Response> {
    let service = s.clone();
    let input = body.clone();
    let (mut chat, omitted, codec) = tokio::task::spawn_blocking(move || -> Result<_> {
        let codec = service
            .thinking
            .get_or_init(|| crate::thinking::Codec::load(&service.thinking_path).map(Arc::new))
            .clone()?;
        let converted =
            crate::anthropic::convert(&input, count, &mut |signature| codec.decode(signature))?;
        Ok((
            converted["chat"].clone(),
            converted["thinking_display"] == "omitted",
            codec,
        ))
    })
    .await
    .map_err(|_| bad("Messages preparation failed"))??;
    crate::tools::normalize(&chat)?;
    crate::grammar::response_schema(&chat)?;
    let media = crate::vision::has_media(&chat);
    if media && s.vision.is_none() {
        return Err(crate::vision::unavailable());
    }
    let front = s.front.clone();
    let preparation_cancel = crate::documents::Cancellation::new();
    let flag = preparation_cancel.flag.clone();
    let prompt_body = chat.clone();
    let preparation=s.preparation_queue.enter(crate::deadline::request(&chat)?).await?;
    let token_count=if media{
        s.vision.as_ref().unwrap().prompt(prompt_body,s.engine.next_id(),false,preparation).await?["input_tokens"].as_u64().ok_or_else(||bad("invalid media count"))? as usize
    }else{
        tokio::task::spawn_blocking(move || {
            let _preparation=preparation;
            let _cancel=preparation_cancel;
            crate::documents::Cancellation::check(&flag)?;
            let prompt=front.render(&prompt_body)?;Ok(front.encode(&prompt,false)?.len())
        }).await.map_err(|_|bad("Messages token counting failed"))??
    };
    if count {
        return Ok(reply(json!({"input_tokens":token_count})));
    }
    let left = s.front.context as usize - token_count.min(s.front.context as usize);
    let requested = chat["max_completion_tokens"]
        .as_u64()
        .ok_or_else(|| bad("max_tokens must be a positive integer"))?;
    let clamped = requested > left as u64;
    chat["max_completion_tokens"] = json!(requested.min(left as u64));
    let streaming = body["stream"] == true;
    chat["stream"] = json!(streaming);
    if let Some(progress) = body.get("return_progress") {
        chat["return_progress"] = progress.clone();
    }
    let mut prepared = s.prepare(&chat, false).await?;
    prepared.internal_start = true;
    prepare_deadline(&mut prepared, received)?;
    body_permit.retain(&prepared,chat.get("response_format"),&[])?;
    drop(chat);drop(body);
    let public = prepared.public_id.clone();
    let model = s.front.response_model.clone();
    let created = prepared.created_at;
    let mut ticket = s.engine.submit_prepared(&prepared)?;
    prepared.frame.payload.truncate(94);
    prepared.frame.payload.shrink_to_fit();
    let keepalive = if streaming {
        stream_ready(&mut ticket, &mut prepared.timeout).await?
    } else {
        false
    };
    let (sender, receiver) = mpsc::channel(16);
    let error_sender = sender.clone();
    let run = async move {
        let _body_permit = body_permit;
        if streaming && keepalive {
            send(&sender, json!({"type":"ping"})).await?;
        }
        let mut state =
            responses::StreamState::new(model.clone(), public.clone(), created, false, Value::Null);
        let (chat_sender, mut chat_receiver) = mpsc::channel(16);
        let producer = async {
            let result = generate(&s, prepared, ticket, Some(&chat_sender), false, permit).await;
            drop(chat_sender);
            result
        };
        let output = &sender;
        let reader = &mut state;
        let signer = codec.clone();
        let consumer = async move {
            loop {
                let next = if streaming {
                    tokio::select! {v=chat_receiver.recv()=>v,_=output.closed()=>return Err(ApiError(499,"client disconnected".into(),"cancelled".into()))}
                } else {
                    chat_receiver.recv().await
                };
                let Some(bytes) = next else {
                    break;
                };
                let bytes = bytes.map_err(|_| bad("internal stream failed"))?;
                if streaming {
                    for line in std::str::from_utf8(&bytes)
                        .map_err(|_| bad("invalid internal encoding"))?
                        .lines()
                        .filter_map(|l| l.strip_prefix("data: "))
                        .filter(|l| *l != "[DONE]")
                    {
                        let value = crate::strict_json(line.as_bytes())?;
                        if let Some(start) = value.get("_native_start") {
                            let prompt = start["prompt_tokens"].as_u64().unwrap_or(0);
                            let cache = start["matched_tokens"].as_u64().unwrap_or(0);
                            send(output,json!({"type":"message_start","message":{"id":format!("msg_{public}"),"type":"message","role":"assistant","model":model,"content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":prompt.saturating_sub(cache),"cache_read_input_tokens":cache,"output_tokens":0}}})).await?;
                        }
                    }
                }
                if bytes.starts_with(b":") {
                    if streaming {
                        send(output, json!({"type":"ping"})).await?;
                    }
                    continue;
                }
                let events = responses::chat_events(reader, &bytes)?;
                if streaming {
                    forward(output, events, omitted, &signer).await?;
                }
            }
            Ok::<_, ApiError>(())
        };
        let (collected, drained) = tokio::join!(producer, consumer);
        drained?;
        let collected = collected?;
        let (events, response) = state.finish(&collected.response, collected.reasoning_open);
        if streaming {
            forward(&sender, events, omitted, &codec).await?;
            send(&sender,json!({"type":"message_delta","delta":{"stop_reason":stop(&collected,clamped),"stop_sequence":collected.stop_sequence},"usage":{"output_tokens":collected.response["usage"]["completion_tokens"]}})).await?;
            send(&sender, json!({"type":"message_stop"})).await?;
            Ok::<Option<Value>,ApiError>(None)
        } else {
            let mut blocks = Vec::new();
            for item in response["output"].as_array().unwrap() {
                if item["type"] == "function_call"
                    && collected.cut_call.as_deref() == item["call_id"].as_str()
                {
                    continue;
                }
                let signature = if omitted && item["type"] == "reasoning" {
                    codec.encode(
                        collected.response["choices"][0]["message"]["reasoning_content"]
                            .as_str()
                            .unwrap_or(""),
                    )?
                } else {
                    String::new()
                };
                blocks.push(block(item, omitted, &signature)?);
            }
            if blocks.iter().all(|v| v["type"] == "thinking") {
                blocks.push(json!({"type":"text","text":""}));
            }
            Ok(Some(
                json!({"id":format!("msg_{}",collected.response["id"].as_str().unwrap().strip_prefix("chatcmpl-").unwrap()),"type":"message","role":"assistant","model":collected.response["model"],"content":blocks,"stop_reason":stop(&collected,clamped),"stop_sequence":collected.stop_sequence,"usage":usage(&collected.response)}),
            ))
        }
    };
    if streaming {
        tokio::spawn(async move {
            if let Err(error) = run.await {
                let _ = send(
                    &error_sender,
                    error_payload(&error),
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
        Ok(response)
    } else {
        run.await?
            .map(reply)
            .ok_or_else(|| bad("missing Messages response"))
    }
}
