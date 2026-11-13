//! Python accepts lone UTF-16 escapes in fields which endpoint adapters ignore.
//! Rust strings cannot represent those code units. Validate the entire document
//! using a temporary scalar-only view, then discard only ignored root fields.
//! No replacement character from that view may enter a consumed field.
use crate::{bad, json_body, Result};
use serde_json::{Map, Value};

fn unit(bytes: &[u8], at: usize) -> Option<u16> {
    if bytes.get(at..at + 2)? != b"\\u" { return None; }
    let mut value = 0;
    for byte in bytes.get(at + 2..at + 6)? {
        value = value * 16 + (*byte as char).to_digit(16)? as u16;
    }
    Some(value)
}

// Equal-length replacement keeps offsets valid. Escaped backslashes and valid
// surrogate pairs are copied verbatim. The original bytes remain untouched.
fn scalar_view(bytes: &[u8]) -> (Vec<u8>, Vec<usize>) {
    let mut view = bytes.to_vec();
    let mut lone = Vec::new();
    let mut quoted = false;
    let mut at = 0;
    while at < bytes.len() {
        if bytes[at] == b'"' { quoted = !quoted; at += 1; continue; }
        if quoted && bytes[at] == b'\\' {
            if let Some(value) = unit(bytes, at) {
                if (0xd800..=0xdbff).contains(&value)
                    && unit(bytes, at + 6).is_some_and(|v| (0xdc00..=0xdfff).contains(&v))
                { at += 12; continue; }
                if (0xd800..=0xdfff).contains(&value) {
                    view[at + 2..at + 6].copy_from_slice(b"fffd");
                    lone.push(at);
                }
                at += 6;
            } else { at += 2; }
        } else { at += 1; }
    }
    (view, lone)
}

// These are the root fields read by the released 2.3 endpoint adapters and
// their shared deadline, streaming, prompt and generation preparation paths.
fn consumed(path: &str, key: &str) -> bool {
    if key == "timeout" { return true; }
    let sampling = matches!(key, "model" | "priority" | "temperature" | "top_p"
        | "top_k" | "min_p" | "presence_penalty" | "frequency_penalty"
        | "repetition_penalty" | "logit_bias" | "seed" | "stop" | "ignore_eos"
        | "thinking_budget");
    let prompt = matches!(key, "model" | "messages" | "tools" | "tool_choice"
        | "parallel_tool_calls" | "response_format" | "reasoning_effort"
        | "preserve_thinking" | "chat_template_kwargs" | "add_generation_prompt");
    match path {
        "/v1/chat/completions" => sampling || prompt || matches!(key,
            "stream" | "stream_options" | "n" | "logprobs" | "max_tokens" | "max_completion_tokens"
            | "return_token_ids"),
        "/v1/completions" => sampling || matches!(key,
            "prompt" | "stream" | "stream_options" | "max_tokens" | "n" | "best_of" | "suffix" | "echo" | "logprobs"
            | "return_token_ids"),
        "/v1/responses" => matches!(key, "model" | "priority" | "temperature" | "top_p"
            | "top_k" | "min_p" | "presence_penalty" | "frequency_penalty"
            | "repetition_penalty" | "logit_bias" | "seed" | "stop" | "thinking_budget"
            | "input" | "instructions" | "tools" | "tool_choice" | "parallel_tool_calls"
            | "reasoning" | "text" | "max_output_tokens" | "store" | "previous_response_id"
            | "conversation" | "background" | "truncation" | "context_management" | "stream"),
        "/v1/messages" | "/v1/messages/count_tokens" => matches!(key,
            "model" | "messages" | "system" | "tools" | "tool_choice" | "thinking"
            | "context_management" | "output_config" | "output_format")
            || (path == "/v1/messages" && matches!(key, "max_tokens" | "stream"
                | "stop_sequences" | "temperature" | "top_p" | "top_k" | "seed" | "priority")),
        "/tokenize" => matches!(key, "content" | "add_special" | "parse_special" | "with_pieces"),
        "/apply-template" => prompt,
        "/v1/systemone" => matches!(key, "model" | "state" | "questions" | "priority"),
        // Judgments explicitly refuses unknown fields. Leave that contract
        // strict, as for an unrecognized route.
        _ => true,
    }
}

// Python's parse_float/parse_int checks happen before duplicate object keys
// overwrite earlier values. Check every numeric token, including overwritten
// ones, rather than just the resulting object's surviving values.
fn validate_numbers(view: &[u8]) -> Result<()> {
    let mut quoted = false;
    let mut at = 0;
    while at < view.len() {
        match view[at] {
            b'"' => { quoted = !quoted; at += 1; }
            b'\\' if quoted => at += 2,
            b'-' | b'0'..=b'9' if !quoted => {
                let start = at;
                while at < view.len() && !matches!(view[at], b' ' | b'\n' | b'\r' | b'\t' | b',' | b']' | b'}') { at += 1; }
                crate::strict_json(&view[start..at])?;
            }
            _ => at += 1,
        }
    }
    Ok(())
}

pub fn body(bytes: &[u8], path: &str) -> Result<Value> {
    match json_body(bytes) {
        Ok(value) => Ok(value),
        Err(error) => {
            let (view, lone) = scalar_view(bytes);
            if lone.is_empty() { return Err(error); }
            // This view is solely a validator: syntax, numeric range and depth
            // in ignored fields must still be checked before discarding them.
            json_body(&view)?;
            validate_numbers(&view)?;
            let mut fields = Map::new();
            let mut invalid = std::collections::HashSet::new();
            let mut depth = 0usize;
            let mut quoted = false;
            let mut escaped = false;
            let mut start = 0;
            let mut colon = 0;
            for (at, &byte) in view.iter().enumerate() {
                if quoted {
                    if escaped { escaped = false; }
                    else if byte == b'\\' { escaped = true; }
                    else if byte == b'"' { quoted = false; }
                    continue;
                }
                match byte {
                    b'"' => quoted = true,
                    b'{' | b'[' => { depth += 1; if depth == 1 { start = at + 1; } }
                    b':' if depth == 1 => colon = at,
                    b',' | b'}' if depth == 1 => {
                        if colon >= start {
                            let key_bad = lone.get(lone.partition_point(|&p| p < start)).is_some_and(|&p| p < colon);
                            let value_bad = lone.get(lone.partition_point(|&p| p <= colon)).is_some_and(|&p| p < at);
                            if !key_bad {
                                let key: String = serde_json::from_slice(&bytes[start..colon]).map_err(|_| error.clone())?;
                                // Last duplicate wins, including a valid value
                                // overwriting an earlier unrepresentable one.
                                invalid.remove(&key);
                                fields.remove(&key);
                                if value_bad {
                                    if consumed(path, &key) { invalid.insert(key); }
                                } else {
                                    fields.insert(key, crate::strict_json(&bytes[colon + 1..at])?);
                                }
                            } else if consumed(path, "") { return Err(error); }
                        }
                        start = at + 1;
                        if byte == b'}' { depth -= 1; }
                    }
                    b']' | b'}' => depth -= 1,
                    _ => {}
                }
            }
            if invalid.is_empty() { Ok(Value::Object(fields)) } else { Err(bad("invalid JSON request body")) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const CHAT: &str = "/v1/chat/completions";
    #[test]
    fn ignored_utf16_matches_released_acceptance() {
        for path in [CHAT, "/v1/completions", "/v1/responses", "/v1/messages", "/v1/messages/count_tokens", "/tokenize", "/apply-template", "/v1/systemone"] {
            for raw in [br#"{"metadata":{"value":"\ud800"}}"#.as_slice(), br#"{"arbitrary":[{"\udfff":"\ud800"}]}"#, br#"{"\ud800":"anything"}"#] {
                assert_eq!(body(raw, path).unwrap(), serde_json::json!({}), "{path}");
            }
        }
    }
    #[test]
    fn consumed_utf16_and_strict_judgment_extras_are_refused() {
        for raw in [br#"{"messages":[{"content":"\ud800"}]}"#.as_slice(), br#"{"timeout":"\udfff"}"#] {
            assert!(body(raw, CHAT).is_err());
        }
        assert!(body(br#"{"extra":"\ud800"}"#, "/v1/judgments").is_err());
    }
    #[test]
    fn normal_unicode_and_escape_spelling_remain_exact() {
        let raw = br#"{"metadata":"\ud800","messages":[{"content":"\ud83d\ude00 literal \\ud800 \ufffd \ue000"}]}"#;
        let result = body(raw, CHAT).unwrap();
        assert_eq!(result["messages"][0]["content"], "😀 literal \\ud800 � \u{e000}");
        assert!(result.get("metadata").is_none());
    }
    #[test]
    fn duplicate_keys_keep_python_last_value_semantics() {
        assert_eq!(body(br#"{"messages":"\ud800","messages":[]}"#, CHAT).unwrap()["messages"], serde_json::json!([]));
        assert!(body(br#"{"messages":[],"messages":"\ud800"}"#, CHAT).is_err());
        assert_eq!(body(r#"{"\ud800":0,"�":1,"metadata":"\udfff"}"#.as_bytes(), CHAT).unwrap()["�"], 1);
    }
    #[test]
    fn ignored_fields_do_not_bypass_json_guards() {
        for raw in [br#"{"metadata":"\ud800","extra":1e999}"#.as_slice(), br#"{"metadata":"\ud800",}"#, br#"{"metadata":"\ud800","extra":"\uZZZZ"}"#, br#"{"metadata":"\ud800"}false"#] {
            assert!(body(raw, CHAT).is_err());
        }
        for depth in [128usize, 129] {
            let raw = format!("{{\"metadata\":{}\"\\ud800\"{}}}", "[".repeat(depth-1), "]".repeat(depth-1));
            assert_eq!(body(raw.as_bytes(), CHAT).is_ok(), depth == 128);
        }
        let raw = format!("{{\"metadata\":\"\\ud800\",\"extra\":{}}}", "9".repeat(4301));
        assert!(body(raw.as_bytes(), CHAT).is_err());
    }
    #[test]
    fn ignored_number_tokens_are_checked_before_duplicate_keys_overwrite_them() {
        for raw in [br#"{"metadata":"\ud800","x":1e999,"x":0}"#.as_slice(),
            br#"{"metadata":{"x":1e999,"x":0,"s":"\ud800"}}"#,
            br#"{"\ud800":1e999,"\ufffd":0}"#] {
            assert!(body(raw, CHAT).is_err());
        }
        assert!(body(br#"{"metadata":"\ud800","x":1e-7,"x":0}"#, CHAT).is_ok());
    }
}
