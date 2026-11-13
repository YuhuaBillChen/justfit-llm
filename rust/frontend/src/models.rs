//! Served aliases preserve loaded-model identity and the 2.3 discovery order.
use crate::{bad, ApiError, Result};
use serde_json::{json, Value};
mod unicode_printable;

pub fn allowed_character(c: char) -> bool {
    let point = c as u32;
    let ranges = unicode_printable::NON_PRINTABLE_OR_WHITESPACE;
    let index = ranges.partition_point(|&(_, high)| high < point);
    ranges.get(index).is_none_or(|&(low, _)| point < low) && !matches!(c, '\\' | '%' | '?' | '#')
}

pub fn names(loaded: &str, aliases: &[String], announce: bool) -> Result<(String, Vec<String>)> {
    for alias in aliases {
        if alias.is_empty() || !alias.chars().all(allowed_character)
            || alias.split('/').any(|part| matches!(part, "" | "." | "..")) {
            return Err(bad("model alias must be a non-empty name without whitespace or URL delimiters"));
        }
    }
    if announce && aliases.is_empty() { return Err(bad("announce_served_name needs a served model name")); }
    let response = if announce { &aliases[0] } else { loaded };
    let mut names = Vec::new();
    for name in std::iter::once(response).chain(std::iter::once(loaded)).chain(aliases.iter().map(String::as_str)) {
        if !names.iter().any(|existing| existing == name) { names.push(name.to_string()); }
    }
    Ok((response.to_string(), names))
}

pub fn rows(names: &[String], response: &str, context: u32, vision: bool, modalities: &[String]) -> Vec<Value> {
    names.iter().map(|name| {
        let mut row = json!({"id":name,"object":"model","created":0,"owned_by":"justfit","max_model_len":context,"context_length":context,"vision":vision,"input_modalities":modalities});
        if name != response { row["root"] = json!(response); }
        row
    }).collect()
}

impl crate::Frontend {
    pub fn accepts_model(&self, model: &Value) -> bool {
        model.as_str().is_some_and(|model| self.model_names.iter().any(|name| name == model))
    }
    pub fn configure_names(&mut self, aliases: &[String], announce: bool) -> Result<()> {
        (self.response_model, self.model_names) = names(&self.model, aliases, announce)?;
        Ok(())
    }
    pub fn check_model(&self, body: &Value) -> Result<()> {
        if let Some(model) = body.get("model").filter(|model| !self.accepts_model(model)) {
            let name = match model {
                Value::String(s) => s.clone(),
                Value::Null => "None".into(),
                Value::Bool(true) => "True".into(),
                Value::Bool(false) => "False".into(),
                other => other.to_string(),
            };
            return Err(ApiError(404, format!("model {name} not found"), "model_not_found".into()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn discovery_deduplicates_and_response_name_is_first() {
        let aliases = vec!["friendly".into(), "owner/model".into(), "friendly".into(), "中文/🙂".into()];
        assert_eq!(names("owner/model", &aliases, false).unwrap(), ("owner/model".into(), vec!["owner/model".into(),"friendly".into(),"中文/🙂".into()]));
        let (response, all) = names("owner/model", &aliases, true).unwrap();
        assert_eq!(response, "friendly");
        assert_eq!(all, vec!["friendly", "owner/model", "中文/🙂"]);
        let items = rows(&all,&response,4096,true,&["text".into(),"image".into()]);
        assert!(items[0].get("root").is_none());
        assert_eq!(items[1]["root"],"friendly");
        assert!(names("owner/model",&[],true).is_err());
    }
    #[test]
    fn url_unsafe_whitespace_and_nonprintable_names_are_rejected() {
        for name in ["", "a b", "a\tb", "a\u{a0}b", "a\u{200b}b", "a\u{13e95}b", "a%2fb", "a#b", "a?b", "a\\b", "/a", "a/", "a//b", "a/./b", "a/../b"] {
            assert!(names("model",&[name.into()],false).is_err(),"{name:?}");
        }
        for name in ["friendly", "owner/model", "中文", "🙂", "a:b@c", "a.b", "a..b", "e\u{301}"] {
            assert!(names("model",&[name.into()],false).is_ok(),"{name:?}");
        }
    }
}
