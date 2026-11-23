//! Hosted tools: tools a provider runs on its own servers (web search, file
//! search, code execution, ...). A local engine cannot execute them. With the
//! default `ignore` policy they are left out of the prompt, so the model never
//! sees them, and the response names them in `x-justfit-ignored-tools`. With
//! `strict` the request fails and names the tool. Tools the client executes
//! (function, namespace, Anthropic custom) are never dropped, and unknown
//! types are still rejected, so a client-side tool is never silently lost.
use crate::{bad, ApiError, Result};
use serde_json::Value;
use std::collections::BTreeSet;
use std::sync::Mutex;

pub const HEADER: &str = "x-justfit-ignored-tools";
const POLICY_ENV: &str = "JUSTFIT_TOOL_POLICY";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Policy {
    Ignore,
    Strict,
}

pub fn parse_policy(value: &str) -> std::result::Result<Policy, String> {
    match value {
        "ignore" => Ok(Policy::Ignore),
        "strict" => Ok(Policy::Strict),
        _ => Err(format!("invalid tool policy {value:?}: use ignore or strict")),
    }
}

pub fn policy() -> Policy {
    std::env::var(POLICY_ENV)
        .ok()
        .and_then(|v| parse_policy(&v).ok())
        .unwrap_or(Policy::Ignore)
}

/// OpenAI Responses tools executed by the provider.
pub fn openai_hosted(kind: &str) -> bool {
    matches!(
        kind,
        "web_search"
            | "web_search_preview"
            | "web_search_preview_2025_03_11"
            | "file_search"
            | "code_interpreter"
            | "image_generation"
            | "computer_use_preview"
            | "mcp"
    )
}

/// Anthropic server tools (versioned type names such as web_search_20250305).
pub fn anthropic_hosted(kind: &str) -> bool {
    ["web_search_", "web_fetch_", "code_execution_"]
        .iter()
        .any(|prefix| kind.starts_with(prefix))
}

/// Decide one non-client tool: Ok(()) means drop it, recording its type.
pub fn handle(kind: &str, hosted: bool, policy: Policy, ignored: &mut Vec<String>) -> Result<()> {
    if !hosted {
        return Err(bad(format!("tool type {kind} is not supported by this server")));
    }
    if policy == Policy::Strict {
        return Err(ApiError(
            400,
            format!("hosted tool {kind} cannot run on this server (tool policy: strict)"),
            "invalid_request_error".into(),
        ));
    }
    if !ignored.iter().any(|seen| seen == kind) {
        ignored.push(kind.to_string());
    }
    Ok(())
}

/// Types that `handle` would drop for this request's tool list.
pub fn ignored(tools: &Value, hosted: fn(&str) -> bool) -> Vec<String> {
    let mut out = Vec::new();
    for tool in tools.as_array().into_iter().flatten() {
        if let Some(kind) = tool["type"].as_str().filter(|kind| hosted(kind)) {
            if !out.iter().any(|seen| seen == kind) {
                out.push(kind.to_string());
            }
        }
    }
    out
}

/// Add the disclosure header and log each dropped type once per process.
pub fn disclose(response: &mut axum::response::Response, ignored: &[String]) {
    if ignored.is_empty() {
        return;
    }
    static LOGGED: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());
    if let Ok(mut logged) = LOGGED.lock() {
        for kind in ignored {
            if logged.insert(kind.clone()) {
                eprintln!("justfit: ignoring hosted tool {kind}: it cannot run on a local server (set JUSTFIT_TOOL_POLICY=strict or --tool-policy strict to reject it)");
            }
        }
    }
    if let Ok(value) = ignored.join(",").parse() {
        response.headers_mut().insert(HEADER, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn hosted_tools_are_dropped_once_and_client_tools_are_not() {
        let mut ignored = Vec::new();
        handle("web_search", openai_hosted("web_search"), Policy::Ignore, &mut ignored).unwrap();
        handle("web_search", openai_hosted("web_search"), Policy::Ignore, &mut ignored).unwrap();
        assert_eq!(ignored, vec!["web_search"]);
        let error = handle("custom", openai_hosted("custom"), Policy::Ignore, &mut ignored).unwrap_err();
        assert!(error.1.contains("custom"), "{}", error.1);
        let error = handle("local_shell", openai_hosted("local_shell"), Policy::Ignore, &mut ignored).unwrap_err();
        assert!(error.1.contains("local_shell"));
    }

    #[test]
    fn strict_policy_names_the_hosted_tool() {
        let error = handle("web_search", true, Policy::Strict, &mut Vec::new()).unwrap_err();
        assert_eq!(error.0, 400);
        assert!(error.1.contains("web_search") && error.1.contains("strict"));
    }

    #[test]
    fn anthropic_server_tools_are_recognised_by_prefix() {
        assert!(anthropic_hosted("web_search_20250305"));
        assert!(anthropic_hosted("web_fetch_20250910"));
        assert!(anthropic_hosted("code_execution_20250825"));
        assert!(!anthropic_hosted("bash_20250124"));
        assert!(!anthropic_hosted("custom"));
    }

    #[test]
    fn ignored_lists_each_hosted_type_once() {
        let tools = json!([{"type":"function","name":"a"},{"type":"web_search"},{"type":"web_search"},{"type":"file_search"}]);
        assert_eq!(ignored(&tools, openai_hosted), vec!["web_search", "file_search"]);
    }

    #[test]
    fn policy_values() {
        assert_eq!(parse_policy("ignore"), Ok(Policy::Ignore));
        assert_eq!(parse_policy("strict"), Ok(Policy::Strict));
        assert!(parse_policy("drop").is_err());
    }
}
