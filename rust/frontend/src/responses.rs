//! Responses request adapter, independently qualified before HTTP integration.
use crate::{bad, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

fn text(value: &Value, kinds: &[&str]) -> Result<String> {
    if value.is_null() {
        return Ok(String::new());
    }
    if let Some(s) = value.as_str() {
        return Ok(s.into());
    }
    let parts = value
        .as_array()
        .ok_or_else(|| bad("Responses content must be text"))?;
    let mut output = String::new();
    for part in parts {
        if !part["type"].as_str().is_some_and(|s| kinds.contains(&s)) {
            return Err(bad("only text Responses content is supported"));
        }
        output.push_str(
            part["text"]
                .as_str()
                .ok_or_else(|| bad("only text Responses content is supported"))?,
        );
    }
    Ok(output)
}
fn content(v: &Value) -> Result<Value> {
    let Some(parts) = v.as_array() else {
        return Ok(json!(text(v, &["input_text", "output_text", "text"])?));
    };
    let mut out = Vec::new();
    let mut media = false;
    for part in parts {
        if !part.is_object() {
            return Err(bad("Responses content parts must be objects"));
        }
        match part["type"].as_str() {
            Some("input_image") => {
                let url = if part["image_url"].is_object() {
                    &part["image_url"]["url"]
                } else {
                    &part["image_url"]
                };
                if !url.is_string() {
                    return Err(bad("invalid image content part"));
                }
                out.push(json!({"type":"image_url","image_url":{"url":url}}));
                media = true;
            }
            Some("input_file") => {
                let mut file = part.clone();
                file.as_object_mut().unwrap().remove("type");
                out.push(json!({"type":"file","file":file}));
                media = true;
            }
            Some("input_text" | "output_text" | "text") if part["text"].is_string() => {
                out.push(json!({"type":"text","text":part["text"]}))
            }
            _ => {
                return Err(bad(
                    "only text, image and inline PDF Responses content is supported",
                ))
            }
        }
    }
    if media {
        Ok(json!(out))
    } else {
        Ok(json!(out
            .iter()
            .filter_map(|v| v["text"].as_str())
            .collect::<String>()))
    }
}
fn namespace_alias(namespace: &Value, name: &Value) -> Result<String> {
    fn valid(v: &Value) -> Option<&str> {
        v.as_str().filter(|s| {
            !s.is_empty()
                && s.len() <= 64
                && s.bytes()
                    .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
        })
    }
    let ns = valid(namespace).ok_or_else(|| bad("invalid namespace tool name"))?;
    let name = valid(name).ok_or_else(|| bad("invalid namespace tool name"))?;
    let alias = format!("{ns}__{name}");
    if alias.len() <= 64 {
        return Ok(alias);
    }
    let hash = Sha256::digest(format!("{ns}\0{name}").as_bytes());
    let digest = hash
        .iter()
        .take(8)
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    Ok(format!(
        "{}__{}__{digest}",
        &ns[..20.min(ns.len())],
        &name[..24.min(name.len())]
    ))
}
fn flush(pending: &mut Option<Value>, messages: &mut Vec<Value>) {
    if let Some(v) = pending.take() {
        messages.push(v);
    }
}
fn assistant(pending: &mut Option<Value>) -> &mut Value {
    pending.get_or_insert_with(|| json!({"role":"assistant","content":""}))
}

pub fn normalize_input(instructions: &Value, items: &[Value]) -> Result<Value> {
    let instructions = if instructions.is_null() {
        ""
    } else {
        instructions
            .as_str()
            .ok_or_else(|| bad("instructions must be a string"))?
    };
    if items.is_empty() {
        return Err(bad("input must be a non-empty string or array"));
    }
    let mut messages = Vec::new();
    let mut pending = None;
    if !instructions.is_empty() {
        messages.push(json!({"role":"system","content":instructions}));
    }
    for item in items {
        if !item.is_object() {
            return Err(bad("each input item must be an object"));
        }
        match item.get("type").map_or(Some("message"),Value::as_str).unwrap_or(""){
            "message"=>{
                let role=item["role"].as_str().ok_or_else(||bad("unsupported message role"))?;
                if role=="user"{flush(&mut pending,&mut messages);messages.push(json!({"role":role,"content":content(&item["content"])?}));continue;}
                let text=text(&item["content"],&["input_text","output_text","text"])?;
                if ["system","developer"].contains(&role){flush(&mut pending,&mut messages);messages.push(json!({"role":"system","content":text}));}
                else if role=="assistant"{let a=assistant(&mut pending);a["content"]=json!(a["content"].as_str().unwrap_or("").to_string()+&text);}
                else{return Err(bad(format!("unsupported message role: {role}")));}
            },
            "reasoning"=>{
                let mut s=text(&item["content"],&["reasoning_text","text"])?;
                if s.is_empty(){s=text(item.get("summary").unwrap_or(&json!([])),&["summary_text"])?;}
                if !s.is_empty(){let a=assistant(&mut pending);a["reasoning_content"]=json!(a["reasoning_content"].as_str().unwrap_or("").to_string()+&s);}
            },
            "function_call"=>{
                let call=item["call_id"].as_str().filter(|s|!s.is_empty()).ok_or_else(||bad("function_call requires call_id and name"))?;
                let mut name=item["name"].as_str().filter(|s|!s.is_empty()).ok_or_else(||bad("function_call requires call_id and name"))?.to_string();
                let arguments=item["arguments"].as_str().ok_or_else(||bad("function_call arguments must be JSON text"))?;
                if !item["namespace"].is_null(){name=namespace_alias(&item["namespace"],&item["name"])?;}
                let a=assistant(&mut pending);
                if a.get("tool_calls").is_none(){a["tool_calls"]=json!([]);}
                a["tool_calls"].as_array_mut().unwrap().push(json!({"id":call,"type":"function","function":{"name":name,"arguments":arguments}}));
            },
            "function_call_output"=>{
                let id=item["call_id"].as_str().filter(|s|!s.is_empty()).ok_or_else(||bad("function_call_output requires call_id"))?;
                let output=if item["output"].is_object(){&item["output"]["content"]}else{&item["output"]};
                flush(&mut pending,&mut messages);messages.push(json!({"role":"tool","tool_call_id":id,"content":content(output)?}));
            },
            _=>return Err(bad("only message, reasoning, function_call, and function_call_output input items are supported"))
        }
    }
    flush(&mut pending, &mut messages);
    Ok(json!(messages))
}
fn function(tool: &Value, name: Option<String>) -> Value {
    let mut f = json!({});
    for key in ["name", "description", "parameters", "strict"] {
        if let Some(v) = tool.get(key) {
            f[key] = v.clone();
        }
    }
    if let Some(name) = name {
        f["name"] = json!(name);
    }
    json!({"type":"function","function":f})
}
pub fn convert(body: &Value) -> Result<Value> {
    if !body["conversation"].is_null() {
        return Err(bad("conversation is not supported"));
    }
    if !body["background"].is_null()
        && body["background"] != false
        && body["background"].as_f64() != Some(0.)
    {
        return Err(bad("background responses are not supported"));
    }
    if !body["truncation"].is_null() && body["truncation"] != "disabled" {
        return Err(bad("only disabled truncation is supported"));
    }
    if !body["context_management"].is_null() && body["context_management"] != json!([]) {
        return Err(bad("Responses context_management edits are not supported"));
    }
    if !body["stream"].is_null() && !body["stream"].is_boolean() {
        return Err(bad("stream must be a boolean"));
    }
    let items = match &body["input"] {
        Value::String(s) => vec![json!({"type":"message","role":"user","content":s})],
        Value::Array(a) => a.clone(),
        _ => return Err(bad("input must be a non-empty string or array")),
    };
    let mut chat = json!({"messages":normalize_input(&body["instructions"],&items)?,"parallel_tool_calls":body["parallel_tool_calls"]});
    let mut namespaces = json!({});
    if !body["tools"].is_null() {
        let tools = body["tools"]
            .as_array()
            .ok_or_else(|| bad("tools must be an array"))?;
        let mut out = Vec::new();
        for tool in tools {
            if tool["type"] == "namespace" {
                let children = tool["tools"]
                    .as_array()
                    .ok_or_else(|| bad("namespace tools must be an array"))?;
                for child in children {
                    if child["type"] != "function" {
                        return Err(bad("only function namespace tools are supported"));
                    }
                    let alias = namespace_alias(&tool["name"], &child["name"])?;
                    if namespaces.get(&alias).is_some() {
                        return Err(bad(format!("duplicate tool name: {alias}")));
                    }
                    namespaces[&alias] = json!([tool["name"], child["name"]]);
                    out.push(function(child, Some(alias)));
                }
            } else {
                if tool["type"] != "function" {
                    return Err(bad("only function tools are supported"));
                }
                out.push(function(tool, None));
            }
        }
        chat["tools"] = if out.is_empty() {
            Value::Null
        } else {
            json!(out)
        };
    }
    if !body["tool_choice"].is_null() {
        let mut choice = body["tool_choice"].clone();
        if choice.is_object() {
            if choice["type"] != "function" || !choice["name"].is_string() {
                return Err(bad("invalid named tool_choice"));
            }
            let mut name = choice["name"].as_str().unwrap().to_string();
            if !choice["namespace"].is_null() {
                name = namespace_alias(&choice["namespace"], &choice["name"])?;
                if namespaces[&name] != json!([choice["namespace"], choice["name"]]) {
                    return Err(bad("invalid named tool_choice"));
                }
            }
            choice = json!({"type":"function","function":{"name":name}});
        }
        chat["tool_choice"] = choice;
    }
    if !body["reasoning"].is_null() {
        if !body["reasoning"].is_object() {
            return Err(bad("reasoning must be an object"));
        }
        if !body["reasoning"]["effort"].is_null() {
            chat["reasoning_effort"] = body["reasoning"]["effort"].clone();
        }
    }
    if !body["text"].is_null() {
        if !body["text"].is_object()
            || (!body["text"]["format"].is_null() && !body["text"]["format"].is_object())
        {
            return Err(bad("text.format must be an object"));
        }
        let value = &body["text"]["format"];
        if !value.is_null() && value["type"] != "text" {
            chat["response_format"] = match value["type"].as_str() {
                Some("json_object") => json!({"type":"json_object"}),
                Some("json_schema") => {
                    let mut schema = json!({});
                    for key in ["name", "description", "schema", "strict"] {
                        if let Some(v) = value.get(key) {
                            schema[key] = v.clone();
                        }
                    }
                    json!({"type":"json_schema","json_schema":schema})
                }
                _ => return Err(bad("only text and JSON text formats are supported")),
            };
        }
    }
    for field in [
        "model",
        "temperature",
        "top_p",
        "top_k",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "min_p",
        "logit_bias",
        "seed",
        "timeout",
        "priority",
        "stop",
        "max_output_tokens",
        "thinking_budget",
    ] {
        if let Some(value) = body.get(field).filter(|v| !v.is_null()) {
            chat[if field == "max_output_tokens" {
                "max_completion_tokens"
            } else {
                field
            }] = value.clone();
        }
    }
    Ok(json!({"chat":chat,"namespaces":namespaces}))
}
