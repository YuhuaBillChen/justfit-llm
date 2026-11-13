//! Messages request adapter. HTTP/signature codec qualification is separate.
use crate::{bad, Result};
use serde_json::{json, Value};

/// Validate document metadata without decoding or rendering its PDF. The shared
/// media preparation path owns rendering, deadlines and request-wide budgets.
pub fn document_parts(block: &Value) -> Result<Vec<Value>> {
    let source = &block["source"];
    if !source.is_object()
        || source["type"] != "base64"
        || source["media_type"] != "application/pdf"
    {
        return Err(bad("documents require a base64 application/pdf source"));
    }
    if let Some(citations) = block.get("citations").filter(|v| !v.is_null()) {
        if !citations.is_object()
            || citations.get("enabled").is_some_and(|v| v != &json!(false))
        {
            return Err(bad("document citations are not supported"));
        }
    }
    let mut parts = Vec::new();
    for field in ["title", "context"] {
        if let Some(value) = block.get(field).filter(|v| !v.is_null()) {
            let value = value
                .as_str()
                .ok_or_else(|| bad(format!("document {field} must be a string")))?;
            parts.push(json!({"type":"text","text":format!("{value}\n")}));
        }
    }
    let encoded = source["data"]
        .as_str()
        .ok_or_else(|| bad("PDF data must be a base64 string"))?;
    parts.push(json!({"type":"file","file":{"file_data":format!("data:application/pdf;base64,{encoded}")}}));
    Ok(parts)
}

fn system(value: &Value) -> Result<Option<String>> {
    let parts = match value {
        Value::String(s) => vec![json!({"type":"text","text":s})],
        Value::Array(a) => a.clone(),
        _ => return Err(bad("system must be text content")),
    };
    let mut text = String::new();
    for part in parts {
        let value = part["text"]
            .as_str()
            .filter(|_| part["type"] == "text")
            .ok_or_else(|| bad("system contains unsupported content"))?;
        if !value.starts_with("x-anthropic-billing-header") {
            text.push_str(value);
        }
    }
    Ok((!text.is_empty()).then_some(text))
}
fn content(value: &Value) -> Result<Value> {
    if value.is_string() {
        return Ok(value.clone());
    }
    let mut parts = Vec::new();
    let mut media = false;
    for block in value
        .as_array()
        .ok_or_else(|| bad("content must be text or content blocks"))?
    {
        match block["type"].as_str() {
            Some("text") if block["text"].is_string() => {
                parts.push(json!({"type":"text","text":block["text"]}))
            }
            Some("image") => {
                let source = &block["source"];
                let mime = source["media_type"]
                    .as_str()
                    .filter(|_| source["type"] == "base64")
                    .ok_or_else(|| bad("Anthropic image blocks must be base64 sources"))?;
                let data = source["data"]
                    .as_str()
                    .ok_or_else(|| bad("Anthropic image blocks must be base64 sources"))?;
                parts.push(json!({"type":"image_url","image_url":{"url":format!("data:{mime};base64,{data}")}}));
                media = true;
            }
            Some("document") => {
                parts.extend(document_parts(block)?);
                media = true;
            }
            _ => return Err(bad("content contains unsupported content")),
        }
    }
    if media {
        Ok(json!(parts))
    } else {
        Ok(json!(parts
            .iter()
            .filter_map(|p| p["text"].as_str())
            .collect::<String>()))
    }
}
fn message(value: &Value, resolve: &mut impl FnMut(&str) -> Result<String>) -> Result<Vec<Value>> {
    let role = value["role"]
        .as_str()
        .filter(|r| ["user", "assistant", "system"].contains(r))
        .ok_or_else(|| bad("Anthropic messages require user/assistant/system roles"))?;
    let input = &value["content"];
    if role == "system" {
        return Ok(system(input)?
            .map(|s| vec![json!({"role":"system","content":s})])
            .unwrap_or_default());
    }
    if input.is_string() {
        return Ok(vec![json!({"role":role,"content":input})]);
    }
    let blocks = input
        .as_array()
        .ok_or_else(|| bad("Anthropic message content must be text or blocks"))?;
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut calls = Vec::new();
    let mut results = Vec::new();
    let mut user = Vec::new();
    for block in blocks {
        if !block.is_object() {
            return Err(bad("Anthropic content blocks must be objects"));
        }
        match block["type"].as_str() {
            Some("text") if block["text"].is_string() => {
                text.push_str(block["text"].as_str().unwrap());
                user.push(block.clone());
            }
            Some("image" | "document") if role == "user" => user.push(block.clone()),
            Some("thinking") if role == "assistant" && block["thinking"].is_string() => {
                let visible = block["thinking"].as_str().unwrap();
                let signature = block.get("signature").map_or(Ok(""), |v| {
                    v.as_str().ok_or_else(|| bad("invalid thinking signature"))
                })?;
                if signature.is_empty() {
                    reasoning.push_str(visible);
                } else {
                    match resolve(signature) {
                        Ok(text) => reasoning.push_str(&text),
                        Err(error) if error.2 == "invalid_thinking_signature" => {
                            reasoning.push_str(visible)
                        }
                        Err(error) => return Err(error),
                    }
                }
            }
            Some("redacted_thinking") if role == "assistant" => {
                if !block["data"].as_str().is_some_and(|s| !s.is_empty()) {
                    return Err(bad("invalid redacted_thinking data"));
                }
            }
            Some("tool_use") if role == "assistant" => {
                if !block["id"].is_string()
                    || !block["name"].is_string()
                    || !block["input"].is_object()
                {
                    return Err(bad("invalid Anthropic tool_use block"));
                }
                calls.push(json!({"id":block["id"],"type":"function","function":{"name":block["name"],"arguments":block["input"]}}));
            }
            Some("tool_result") if role == "user" => {
                if !block["tool_use_id"].is_string() {
                    return Err(bad("invalid Anthropic tool_result block"));
                }
                let failed = block.get("is_error").map_or(Ok(false), |v| {
                    v.as_bool()
                        .ok_or_else(|| bad("tool_result.is_error must be a boolean"))
                })?;
                let empty = json!("");
                let mut value = content(block.get("content").unwrap_or(&empty))?;
                if failed {
                    if let Some(text) = value.as_str() {
                        value = json!(format!("Tool execution failed:\n{text}"));
                    } else {
                        value
                            .as_array_mut()
                            .unwrap()
                            .insert(0, json!({"type":"text","text":"Tool execution failed:\n"}));
                    }
                }
                results.push(
                    json!({"role":"tool","tool_call_id":block["tool_use_id"],"content":value}),
                );
            }
            _ => return Err(bad("unsupported Anthropic content block")),
        }
    }
    if role == "assistant" {
        let mut item = json!({"role":role,"content":text});
        if !reasoning.is_empty() {
            item["reasoning_content"] = json!(reasoning);
        }
        if !calls.is_empty() {
            item["tool_calls"] = json!(calls);
        }
        return Ok(vec![item]);
    }
    if !user.is_empty() || results.is_empty() {
        results.push(json!({"role":role,"content":content(&json!(user))?}));
    }
    Ok(results)
}
pub fn convert(
    body: &Value,
    prompt_only: bool,
    resolve: &mut impl FnMut(&str) -> Result<String>,
) -> Result<Value> {
    if !body["model"].as_str().is_some_and(|s| !s.is_empty()) {
        return Err(bad("model must be a non-empty string"));
    }
    if !prompt_only {
        if !body["max_tokens"].as_u64().is_some_and(|n| n > 0) {
            return Err(bad("max_tokens must be a positive integer"));
        }
        if body.get("stream").is_some_and(|v| !v.is_boolean()) {
            return Err(bad("stream must be a boolean"));
        }
    }
    let mut preserve = None;
    if let Some(context) = body.get("context_management").filter(|v| !v.is_null()) {
        if !context.is_object() {
            return Err(bad("context_management must be an object"));
        }
        let empty = json!([]);
        let edits = context
            .get("edits")
            .unwrap_or(&empty)
            .as_array()
            .ok_or_else(|| bad("context_management.edits must be an array"))?;
        if edits
            .iter()
            .any(|v| v["type"] != "clear_thinking_20251015" || v["keep"] != "all")
        {
            return Err(bad("only clear_thinking_20251015 with keep: all is supported in context_management.edits"));
        }
        if !edits.is_empty() {
            preserve = Some(true);
        }
    }
    let empty = json!({});
    let config = body.get("output_config").unwrap_or(&empty);
    if !config.is_object() {
        return Err(bad("output_config must be an object"));
    }
    let effort = config.get("effort").map_or(Ok("high"), |v| {
        v.as_str()
            .ok_or_else(|| bad("output_config.effort is invalid"))
    })?;
    if !["low", "medium", "high", "xhigh", "max"].contains(&effort) {
        return Err(bad("output_config.effort is invalid"));
    }
    if config.get("format").is_some() && body.get("output_format").is_some() {
        return Err(bad(
            "output_config.format and output_format cannot be combined",
        ));
    }
    let format = config.get("format").or_else(|| body.get("output_format"));
    let response = format
        .map(|v| {
            if v["type"] != "json_schema" || !(v["schema"].is_object() || v["schema"].is_boolean())
            {
                Err(bad(
                    "output format must be a json_schema object with a schema",
                ))
            } else {
                Ok(json!({"type":"json_schema","json_schema":{"schema":v["schema"]}}))
            }
        })
        .transpose()?;
    let messages = body["messages"]
        .as_array()
        .filter(|a| !a.is_empty())
        .ok_or_else(|| bad("messages must be a non-empty array"))?;
    let mut translated = Vec::new();
    if let Some(system_) = body.get("system").filter(|v| !v.is_null()) {
        if let Some(text) = system(system_)? {
            translated.push(json!({"role":"system","content":text}));
        }
    }
    for item in messages {
        translated.extend(message(item, resolve)?);
    }
    let mut reasoning = "none";
    let mut display = "summarized";
    if let Some(thinking) = body.get("thinking").filter(|v| !v.is_null()) {
        let kind = thinking["type"]
            .as_str()
            .filter(|k| ["enabled", "disabled", "adaptive"].contains(k))
            .ok_or_else(|| bad("thinking.type must be enabled, disabled, or adaptive"))?;
        if let Some(value) = thinking.get("display") {
            if kind == "disabled" {
                return Err(bad(
                    "thinking.display requires enabled or adaptive thinking",
                ));
            }
            if !value.is_null()
                && !value
                    .as_str()
                    .is_some_and(|s| ["summarized", "omitted", "updates"].contains(&s))
            {
                return Err(bad(
                    "thinking.display must be summarized, omitted, updates, or null",
                ));
            }
            if value == "omitted" || value == "updates" {
                display = "omitted";
            }
        }
        if kind != "disabled" {
            reasoning = effort;
        }
    }
    let mut chat =
        json!({"model":body["model"],"messages":translated,"reasoning_effort":reasoning});
    if let Some(preserve) = preserve {
        chat["preserve_thinking"] = json!(preserve);
    }
    if let Some(response) = response {
        chat["response_format"] = response;
    }
    if let Some(tools) = body.get("tools").filter(|v| !v.is_null()) {
        let tools = tools
            .as_array()
            .ok_or_else(|| bad("tools must be an array"))?;
        let mut out = Vec::new();
        for tool in tools {
            if tool.get("type").is_some_and(|v| v != "custom") {
                return Err(bad("only custom function tools are supported"));
            }
            let default = json!({});
            let schema = tool.get("input_schema").unwrap_or(&default);
            if !tool["name"].is_string() || !(schema.is_object() || schema.is_boolean()) {
                return Err(bad("invalid Anthropic tool"));
            }
            let mut function = json!({"name":tool["name"],"parameters":schema});
            if tool["description"].is_string() {
                function["description"] = tool["description"].clone();
            }
            if let Some(v) = tool.get("strict") {
                function["strict"] = v.clone();
            }
            out.push(json!({"type":"function","function":function}));
        }
        chat["tools"] = json!(out);
    }
    if let Some(choice) = body.get("tool_choice").filter(|v| !v.is_null()) {
        if !choice.is_object() {
            return Err(bad("tool_choice must be an object"));
        }
        let disable = choice
            .get("disable_parallel_tool_use")
            .map_or(Ok(false), |v| {
                v.as_bool()
                    .ok_or_else(|| bad("disable_parallel_tool_use must be a boolean"))
            })?;
        chat["parallel_tool_calls"] = json!(!disable);
        chat["tool_choice"] = match choice["type"].as_str() {
            Some("auto") => json!("auto"),
            Some("any") => json!("required"),
            Some("none") => json!("none"),
            Some("tool") if choice["name"].is_string() => {
                json!({"type":"function","function":{"name":choice["name"]}})
            }
            _ => return Err(bad("invalid Anthropic tool_choice")),
        };
    }
    if !prompt_only {
        if messages.last().unwrap()["role"] == "assistant" {
            return Err(bad("a final assistant message (prefill) is not supported; end messages with a user turn"));
        }
        chat["max_completion_tokens"] = body["max_tokens"].clone();
        for (src, dst) in [
            ("stop_sequences", "stop"),
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("top_k", "top_k"),
            ("seed", "seed"),
            ("timeout", "timeout"),
            ("priority", "priority"),
        ] {
            if let Some(v) = body.get(src).filter(|v| !v.is_null()) {
                chat[dst] = v.clone();
            }
        }
        if let Some(budget) = body["thinking"].get("budget_tokens") {
            chat["thinking_budget"] = budget.clone();
        }
    }
    Ok(json!({"chat":chat,"thinking_display":display}))
}
pub fn fixture(body: &Value) -> Result<Value> {
    convert(&body["request"], body["prompt_only"] == true, &mut |_| {
        Err(crate::ApiError(
            400,
            "invalid signature".into(),
            "invalid_thinking_signature".into(),
        ))
    })
}
