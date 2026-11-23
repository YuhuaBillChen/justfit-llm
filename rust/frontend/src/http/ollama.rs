//! Ollama-compatible API (`/api/*`) on top of the chat-completions route, so
//! clients that only speak Ollama can use JustFit unchanged.
//!
//! Supported: GET /api/version, /api/tags, /api/ps; POST /api/show,
//! /api/chat, /api/generate (streaming NDJSON by default, as in Ollama).
//! Model management (pull, create, copy, delete, push, embed) is not offered:
//! JustFit serves one resident model.
use super::{BodyPermit, Service};
use crate::{bad, ApiError, Result};
use axum::{body::{Body, Bytes}, response::Response};
use serde_json::{json, Map, Value};
use std::{sync::Arc, time::Instant};
use tokio::sync::{mpsc, OwnedSemaphorePermit};
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

fn json_reply(value: Value) -> Response {
    let mut response = Response::new(Body::from(value.to_string()));
    response
        .headers_mut()
        .insert("content-type", "application/json".parse().unwrap());
    response
}

/// RFC 3339 UTC timestamp for `created_at` (no external time crate).
fn now_rfc3339() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs() as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Civil-from-days (Howard Hinnant).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:09}Z",
        rem / 3_600,
        rem % 3_600 / 60,
        rem % 60,
        now.subsec_nanos()
    )
}

/// Ollama clients append `:latest` to bare names.
fn model_name(s: &Service, body: &Value) -> Result<String> {
    let requested = body["model"]
        .as_str()
        .filter(|m| !m.is_empty())
        .unwrap_or(&s.front.response_model);
    let stripped = requested.strip_suffix(":latest").unwrap_or(requested);
    for candidate in [requested, stripped] {
        if s.front.accepts_model(&json!(candidate)) {
            return Ok(candidate.to_string());
        }
    }
    Err(ApiError(
        404,
        format!("model '{requested}' not found"),
        "model_not_found".into(),
    ))
}

fn details() -> Value {
    json!({"parent_model":"","format":"mlx","family":"","families":null,"parameter_size":"","quantization_level":""})
}

pub fn version() -> Response {
    json_reply(json!({ "version": VERSION }))
}

pub fn tags(s: &Service) -> Response {
    let modified = now_rfc3339();
    let instant = s.front.model_names.iter().map(|name| format!("{name}{}", crate::models::INSTANT_SUFFIX));
    let models: Vec<Value> = s
        .front
        .model_names
        .iter()
        .cloned()
        .chain(instant)
        .map(|name| json!({"name":name,"model":name,"modified_at":modified,"size":0,"digest":"","details":details()}))
        .collect();
    json_reply(json!({ "models": models }))
}

pub fn ps(s: &Service) -> Response {
    let name = &s.front.response_model;
    json_reply(json!({"models":[{"name":name,"model":name,"size":0,"digest":"","details":details(),"expires_at":"0001-01-01T00:00:00Z","size_vram":0,"context_length":s.front.context}]}))
}

fn show(s: &Service, body: &Value) -> Result<Response> {
    let requested = if body["model"].is_null() { body["name"].clone() } else { body["model"].clone() };
    let name = model_name(s, &json!({ "model": requested }))?;
    let mut capabilities = vec!["completion", "tools", "thinking"];
    if s.vision.is_some() {
        capabilities.push("vision");
    }
    Ok(json_reply(json!({
        "modelfile": "",
        "parameters": "",
        "template": "",
        "details": details(),
        "model_info": {"general.architecture":"justfit","general.basename":name,"justfit.context_length":s.front.context},
        "capabilities": capabilities,
        "modified_at": now_rfc3339(),
    })))
}

/// Guess an image MIME type from the start of its base64 text.
fn image_url(data: &str) -> String {
    if data.starts_with("data:") {
        return data.to_string();
    }
    let mime = if data.starts_with("/9j/") {
        "image/jpeg"
    } else if data.starts_with("UklGR") {
        "image/webp"
    } else if data.starts_with("R0lGOD") {
        "image/gif"
    } else {
        "image/png"
    };
    format!("data:{mime};base64,{data}")
}

fn with_images(text: &str, images: &Value) -> Result<Value> {
    let Some(images) = images.as_array().filter(|a| !a.is_empty()) else {
        return Ok(json!(text));
    };
    let mut parts = vec![json!({"type":"text","text":text})];
    for image in images {
        let data = image.as_str().ok_or_else(|| bad("images must be base64 strings"))?;
        parts.push(json!({"type":"image_url","image_url":{"url":image_url(data)}}));
    }
    Ok(json!(parts))
}

/// Ollama messages -> OpenAI chat messages. Ollama tool calls have no ids, so
/// ids are assigned and tool results are matched to them in order.
fn messages(input: &Value) -> Result<Vec<Value>> {
    let input = input
        .as_array()
        .ok_or_else(|| bad("messages must be an array"))?;
    let mut out = Vec::new();
    let mut pending: Vec<(String, String)> = Vec::new();
    let mut counter = 0usize;
    for m in input {
        let role = m["role"].as_str().ok_or_else(|| bad("message role must be a string"))?;
        let text = m["content"].as_str().unwrap_or("");
        match role {
            "tool" => {
                let name = m["tool_name"].as_str().or_else(|| m["name"].as_str());
                let index = name
                    .and_then(|n| pending.iter().position(|(_, pn)| pn == n))
                    .unwrap_or(0);
                let id = if pending.is_empty() {
                    counter += 1;
                    format!("call_{counter}")
                } else {
                    pending.remove(index).0
                };
                out.push(json!({"role":"tool","tool_call_id":id,"content":text}));
            }
            "assistant" => {
                let mut item = json!({"role":"assistant","content":text});
                if let Some(thinking) = m["thinking"].as_str().filter(|t| !t.is_empty()) {
                    item["reasoning_content"] = json!(thinking);
                }
                if let Some(calls) = m["tool_calls"].as_array().filter(|c| !c.is_empty()) {
                    let mut converted = Vec::new();
                    for call in calls {
                        counter += 1;
                        let id = format!("call_{counter}");
                        let name = call["function"]["name"]
                            .as_str()
                            .ok_or_else(|| bad("tool call needs a function name"))?;
                        let arguments = match &call["function"]["arguments"] {
                            Value::String(s) => s.clone(),
                            Value::Null => "{}".into(),
                            other => other.to_string(),
                        };
                        pending.push((id.clone(), name.to_string()));
                        converted.push(json!({"id":id,"type":"function","function":{"name":name,"arguments":arguments}}));
                    }
                    item["tool_calls"] = json!(converted);
                }
                out.push(item);
            }
            "system" | "user" => {
                out.push(json!({"role":role,"content":with_images(text, &m["images"])?}));
            }
            other => return Err(bad(format!("unsupported message role: {other}"))),
        }
    }
    Ok(out)
}

/// Ollama request -> chat-completions request.
fn chat_request(s: &Service, body: &Value, generate: bool) -> Result<(Value, String, bool)> {
    let model = model_name(s, body)?;
    let stream = body.get("stream").map_or(Ok(true), |v| {
        if v.is_null() {
            Ok(true)
        } else {
            v.as_bool().ok_or_else(|| bad("stream must be a boolean"))
        }
    })?;
    let messages = if generate {
        if body["raw"] == true {
            return Err(bad("raw generate requests are not supported; use /v1/completions"));
        }
        if body["suffix"].as_str().is_some_and(|s| !s.is_empty()) {
            return Err(bad("suffix (fill-in-the-middle) is not supported"));
        }
        let mut list = Vec::new();
        if let Some(system) = body["system"].as_str().filter(|s| !s.is_empty()) {
            list.push(json!({"role":"system","content":system}));
        }
        let prompt = body["prompt"].as_str().unwrap_or("");
        list.push(json!({"role":"user","content":with_images(prompt, &body["images"])?}));
        list
    } else {
        messages(&body["messages"])?
    };
    let mut chat = json!({"model":model,"messages":messages,"stream":stream});
    if stream {
        chat["stream_options"] = json!({"include_usage":true});
    }
    if !body["tools"].is_null() && !generate {
        chat["tools"] = body["tools"].clone();
    }
    match &body["format"] {
        Value::Null => {}
        Value::String(f) if f.is_empty() => {}
        Value::String(f) if f == "json" => chat["response_format"] = json!({"type":"json_object"}),
        Value::Object(schema) => {
            chat["response_format"] =
                json!({"type":"json_schema","json_schema":{"name":"response","schema":schema}})
        }
        _ => return Err(bad("format must be \"json\" or a JSON schema object")),
    }
    match &body["think"] {
        Value::Null => {}
        Value::Bool(enabled) => {
            chat["chat_template_kwargs"] = json!({"enable_thinking":enabled});
            if !enabled {
                chat["reasoning_effort"] = json!("none");
            }
        }
        Value::String(level) if ["low", "medium", "high"].contains(&level.as_str()) => {
            chat["reasoning_effort"] = json!(level)
        }
        _ => return Err(bad("think must be a boolean or low, medium, high")),
    }
    if let Some(options) = body["options"].as_object() {
        let mapping = [
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("top_k", "top_k"),
            ("min_p", "min_p"),
            ("seed", "seed"),
            ("stop", "stop"),
            ("num_predict", "max_tokens"),
            ("repeat_penalty", "repetition_penalty"),
            ("presence_penalty", "presence_penalty"),
            ("frequency_penalty", "frequency_penalty"),
        ];
        for (from, to) in mapping {
            if let Some(value) = options.get(from).filter(|v| !v.is_null()) {
                // Ollama uses num_predict -1 (or -2) for "until done".
                if from == "num_predict" && value.as_i64().is_some_and(|n| n < 0) {
                    continue;
                }
                chat[to] = value.clone();
            }
        }
    }
    Ok((chat, model, stream))
}

fn ns(ms: &Value) -> u64 {
    (ms.as_f64().unwrap_or(0.) * 1_000_000.) as u64
}

fn tool_calls_out(calls: &Value) -> Value {
    let Some(calls) = calls.as_array().filter(|c| !c.is_empty()) else {
        return Value::Null;
    };
    json!(calls
        .iter()
        .map(|call| {
            let raw = &call["function"]["arguments"];
            let arguments = match raw {
                Value::String(s) => serde_json::from_str(s).unwrap_or_else(|_| json!({})),
                other => other.clone(),
            };
            json!({"function":{"name":call["function"]["name"],"arguments":arguments}})
        })
        .collect::<Vec<_>>())
}

fn done_fields(out: &mut Value, reason: &str, usage: &Value, timings: &Value, received: Instant) {
    out["done"] = json!(true);
    out["done_reason"] = json!(match reason {
        "length" => "length",
        _ => "stop",
    });
    out["total_duration"] = json!(received.elapsed().as_nanos() as u64);
    out["load_duration"] = json!(0);
    out["prompt_eval_count"] = usage["prompt_tokens"].clone();
    out["prompt_eval_duration"] = json!(ns(&timings["prompt_ms"]));
    out["eval_count"] = usage["completion_tokens"].clone();
    out["eval_duration"] = json!(ns(&timings["predicted_ms"]));
}

/// One Ollama message/response object.
fn piece(model: &str, generate: bool, content: &str, thinking: &str, calls: &Value) -> Value {
    let mut out = json!({"model":model,"created_at":now_rfc3339()});
    if generate {
        out["response"] = json!(content);
        if !thinking.is_empty() {
            out["thinking"] = json!(thinking);
        }
    } else {
        let mut message = json!({"role":"assistant","content":content});
        if !thinking.is_empty() {
            message["thinking"] = json!(thinking);
        }
        if !calls.is_null() {
            message["tool_calls"] = calls.clone();
        }
        out["message"] = message;
    }
    out["done"] = json!(false);
    out
}

fn error_message(value: &Value) -> String {
    value["error"]["message"]
        .as_str()
        .or_else(|| value["error"].as_str())
        .unwrap_or("generation failed")
        .to_string()
}

pub(super) async fn post(
    s: Service,
    body: Value,
    path: String,
    permit: Arc<OwnedSemaphorePermit>,
    body_permit: BodyPermit,
    received: Instant,
) -> Result<Response> {
    if path == "/api/show" {
        return show(&s, &body);
    }
    let generate = match path.as_str() {
        "/api/chat" => false,
        "/api/generate" => true,
        _ => return Err(ApiError(404, "not found".into(), "not_found".into())),
    };
    // A generate request with no prompt only loads the model in Ollama.
    if generate && body["prompt"].as_str().is_none_or(str::is_empty) && body["images"].is_null() {
        let model = model_name(&s, &body)?;
        let mut out = piece(&model, true, "", "", &Value::Null);
        out["done"] = json!(true);
        out["done_reason"] = json!("load");
        return Ok(json_reply(out));
    }
    let (chat, model, stream) = chat_request(&s, &body, generate)?;
    drop(body);
    let response = Box::pin(super::route(
        s,
        chat,
        "/v1/chat/completions".into(),
        permit,
        body_permit,
        received,
    ))
    .await?;
    if !stream {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .map_err(|_| bad("internal response failed"))?;
        let chat: Value = serde_json::from_slice(&bytes).map_err(|_| bad("internal response invalid"))?;
        let message = &chat["choices"][0]["message"];
        let mut out = piece(
            &model,
            generate,
            message["content"].as_str().unwrap_or(""),
            message["reasoning_content"].as_str().unwrap_or(""),
            &tool_calls_out(&message["tool_calls"]),
        );
        done_fields(
            &mut out,
            chat["choices"][0]["finish_reason"].as_str().unwrap_or("stop"),
            &chat["usage"],
            &chat["timings"],
            received,
        );
        return Ok(json_reply(out));
    }
    let (sender, receiver) = mpsc::channel::<std::result::Result<Bytes, std::io::Error>>(16);
    tokio::spawn(async move {
        let mut source = response.into_body().into_data_stream();
        let mut buffer: Vec<u8> = Vec::new();
        let mut calls: Vec<Value> = Vec::new();
        let mut reason = String::from("stop");
        let mut usage = Value::Null;
        let mut timings = Value::Null;
        let send = |value: Value| {
            let sender = sender.clone();
            async move { sender.send(Ok(Bytes::from(value.to_string() + "\n"))).await.is_ok() }
        };
        while let Some(chunk) = source.next().await {
            let Ok(chunk) = chunk else { break };
            // Split on bytes: a character may span two chunks.
            buffer.extend_from_slice(&chunk);
            while let Some(end) = buffer.iter().position(|byte| *byte == b'\n') {
                let bytes: Vec<u8> = buffer.drain(..=end).collect();
                let line = String::from_utf8_lossy(&bytes);
                let Some(data) = line.trim_end().strip_prefix("data: ") else { continue };
                if data == "[DONE]" {
                    continue;
                }
                let Ok(event) = serde_json::from_str::<Value>(data) else { continue };
                if !event["error"].is_null() {
                    let _ = send(json!({"error":error_message(&event)})).await;
                    return;
                }
                if !event["usage"].is_null() {
                    usage = event["usage"].clone();
                }
                if !event["timings"].is_null() {
                    timings = event["timings"].clone();
                }
                let choice = &event["choices"][0];
                if let Some(r) = choice["finish_reason"].as_str() {
                    reason = r.to_string();
                }
                let delta = &choice["delta"];
                for call in delta["tool_calls"].as_array().into_iter().flatten() {
                    let index = call["index"].as_u64().unwrap_or(calls.len() as u64) as usize;
                    while calls.len() <= index {
                        calls.push(json!({"function":{"name":"","arguments":""}}));
                    }
                    let slot = &mut calls[index]["function"];
                    if let Some(name) = call["function"]["name"].as_str() {
                        slot["name"] = json!(slot["name"].as_str().unwrap_or("").to_string() + name);
                    }
                    if let Some(args) = call["function"]["arguments"].as_str() {
                        slot["arguments"] = json!(slot["arguments"].as_str().unwrap_or("").to_string() + args);
                    }
                }
                let content = delta["content"].as_str().unwrap_or("");
                let thinking = delta["reasoning_content"].as_str().unwrap_or("");
                if (!content.is_empty() || !thinking.is_empty())
                    && !send(piece(&model, generate, content, thinking, &Value::Null)).await
                {
                    return;
                }
            }
        }
        if !calls.is_empty() && !send(piece(&model, generate, "", "", &tool_calls_out(&json!(calls)))).await {
            return;
        }
        let mut last = piece(&model, generate, "", "", &Value::Null);
        done_fields(&mut last, &reason, &usage, &timings, received);
        let _ = send(last).await;
    });
    let mut response = Response::new(Body::from_stream(ReceiverStream::new(receiver)));
    response
        .headers_mut()
        .insert("content-type", "application/x-ndjson".parse().unwrap());
    Ok(response)
}

/// Ollama error bodies are `{"error": "<message>"}`.
pub fn error_body(error: &ApiError) -> String {
    let mut map = Map::new();
    map.insert("error".into(), json!(error.1));
    Value::Object(map).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_pair_tool_results_with_generated_ids() {
        let input = json!([
            {"role":"user","content":"weather?"},
            {"role":"assistant","content":"","tool_calls":[{"function":{"name":"weather","arguments":{"city":"Paris"}}}]},
            {"role":"tool","tool_name":"weather","content":"sunny"}
        ]);
        let out = messages(&input).unwrap();
        let id = out[1]["tool_calls"][0]["id"].as_str().unwrap();
        assert_eq!(out[1]["tool_calls"][0]["function"]["arguments"], "{\"city\":\"Paris\"}");
        assert_eq!(out[2]["tool_call_id"], id);
        assert_eq!(out[2]["role"], "tool");
    }

    #[test]
    fn images_become_data_urls_with_sniffed_types() {
        assert!(image_url("/9j/4AAQ").starts_with("data:image/jpeg;base64,"));
        assert!(image_url("iVBORw0").starts_with("data:image/png;base64,"));
        let content = with_images("look", &json!(["iVBORw0"])).unwrap();
        assert_eq!(content[1]["type"], "image_url");
    }

    #[test]
    fn tool_call_arguments_are_objects_in_ollama_output() {
        let calls = json!([{"id":"c1","type":"function","function":{"name":"f","arguments":"{\"a\":1}"}}]);
        assert_eq!(tool_calls_out(&calls), json!([{"function":{"name":"f","arguments":{"a":1}}}]));
    }

    #[test]
    fn timestamps_are_rfc3339() {
        let t = now_rfc3339();
        assert_eq!(t.len(), 30, "{t}");
        assert!(t.ends_with('Z') && &t[4..5] == "-" && &t[10..11] == "T");
    }
}
