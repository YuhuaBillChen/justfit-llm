//! Released public server options layered over device presets.
use crate::download::Result;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Default)]
pub struct Options {
    values: BTreeMap<String, String>,
    many: BTreeMap<String, Vec<String>>,
    switches: BTreeSet<String>,
    key: Option<String>,
    pub key_on_command_line: bool,
}
fn size(value: &str, auto: bool, zero: bool) -> Result<Option<u64>> {
    let text = value.trim().to_ascii_uppercase();
    if auto && text == "AUTO" {
        return Ok(None);
    }
    let mut number = text.as_str();
    let mut unit = 1u64;
    for (suffix, n) in [
        ("KIB", 1024),
        ("MIB", 1024 * 1024),
        ("GIB", 1024 * 1024 * 1024),
        ("KB", 1024),
        ("MB", 1024 * 1024),
        ("GB", 1024 * 1024 * 1024),
        ("K", 1024),
        ("M", 1024 * 1024),
        ("G", 1024 * 1024 * 1024),
    ] {
        if let Some(value) = text.strip_suffix(suffix) {
            number = value;
            unit = n;
            break;
        }
    }
    let result = number
        .parse::<u64>()?
        .checked_mul(unit)
        .filter(|n| *n <= i64::MAX as u64 && (*n > 0 || zero))
        .ok_or("invalid byte count")?;
    Ok(Some(result))
}
fn duration(value: &str, off: bool) -> Result<Option<f64>> {
    let text = value.trim().to_ascii_lowercase();
    if off && text == "off" {
        return Ok(None);
    }
    let (value, unit) = match text.as_bytes().last() {
        Some(b's') => (&text[..text.len() - 1], 1.),
        Some(b'm') => (&text[..text.len() - 1], 60.),
        Some(b'h') => (&text[..text.len() - 1], 3600.),
        _ => (text.as_str(), 1.),
    };
    let seconds = value.parse::<f64>()? * unit;
    if !seconds.is_finite() || seconds <= 0. {
        return Err("duration must be positive and finite".into());
    }
    Ok(Some(seconds))
}
fn set_native(native: &mut Vec<String>, flag: &str, value: Option<String>) {
    if let Some(index) = native.iter().position(|s| s == flag) {
        native.drain(index..index + 2);
    }
    if let Some(value) = value {
        native.extend([flag.into(), value]);
    }
}
impl Options {
    pub fn parse(args: &[String]) -> Result<(Vec<String>, Self)> {
        let mut output = vec![];
        let mut result = Self::default();
        let mut index = 0;
        let singles = [
            "--host",
            "--api-key",
            "--max-memory",
            "--host-reserve",
            "--max-context",
            "--kv-format",
            "--idle-release",
            "--max-cache-disk",
            "--cache-dir",
            "--default-reasoning-effort",
            "--max-request-size",
            "--max-image-pixels",
            "--request-timeout",
            "--queue-size",
            "--decode-share",
        ];
        while index < args.len() {
            let flag = args[index].as_str();
            if [
                "--no-webui",
                "--persistent-cache",
                "--announce-served-name",
                "--disable-ane",
                "--allow-idle-sleep",
            ]
            .contains(&flag)
            {
                result.switches.insert(flag.into());
                index += 1;
                continue;
            }
            if singles.contains(&flag)
                || ["--allowed-host", "--allowed-origin", "--served-model-name"].contains(&flag)
            {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| format!("{flag} needs a value"))?
                    .clone();
                if singles.contains(&flag) {
                    if result.values.insert(flag.into(), value).is_some() {
                        return Err(format!("duplicate serve option: {flag}").into());
                    }
                } else {
                    result.many.entry(flag.into()).or_default().push(value);
                }
                index += 2;
            } else {
                output.push(args[index].clone());
                index += 1;
            }
        }
        result.key_on_command_line = result.values.contains_key("--api-key");
        result.key = result
            .values
            .remove("--api-key")
            .or_else(|| std::env::var("JUSTFIT_API_KEY").ok());
        if result
            .key
            .as_ref()
            .is_some_and(|k| k.is_empty() || k.bytes().any(|b| !(33..127).contains(&b)))
        {
            return Err("API key must contain only visible ASCII characters".into());
        }
        if !result.values.contains_key("--default-reasoning-effort") {
            if let Ok(value) = std::env::var("JUSTFIT_DEFAULT_REASONING_EFFORT") {
                result
                    .values
                    .insert("--default-reasoning-effort".into(), value);
            }
        }
        for (flag, value) in &result.values {
            match flag.as_str() {
                "--max-memory" | "--host-reserve" => {
                    size(value, true, false)?;
                }
                "--max-cache-disk" => {
                    size(value, false, true)?;
                }
                "--max-request-size" => {
                    size(value, false, false)?;
                }
                "--max-context" => {
                    context(value)?;
                }
                "--idle-release" => {
                    duration(value, true)?;
                }
                "--request-timeout" => {
                    duration(value, false)?;
                }
                "--kv-format" if !["int8", "bf16", "tq4"].contains(&value.as_str()) => {
                    return Err("invalid KV format".into())
                }
                "--max-image-pixels" => {
                    let n = value.parse::<usize>()?;
                    if !(65536..=crate::media::MAX_PIXELS).contains(&n) {
                        return Err("invalid max image pixels".into());
                    }
                }
                "--queue-size" => {
                    if value.parse::<usize>()? == 0
                        || value.parse::<usize>()? > tokio::sync::Semaphore::MAX_PERMITS - 64
                    {
                        return Err("queue size must be positive".into());
                    }
                }
                "--decode-share" => {
                    let n = value.parse::<f64>()?;
                    if !n.is_finite() || n < 0. {
                        return Err("decode share must be finite and nonnegative".into());
                    }
                }
                "--default-reasoning-effort"
                    if !["none", "minimal", "low", "medium", "high", "xhigh", "max"]
                        .contains(&value.as_str()) =>
                {
                    return Err("invalid default reasoning effort".into())
                }
                "--cache-dir" if value.trim().is_empty() => {
                    return Err("cache directory must not be empty".into())
                }
                _ => (),
            }
        }
        let host = result
            .values
            .get("--host")
            .map(String::as_str)
            .unwrap_or("127.0.0.1");
        crate::access::AccessPolicy::new(
            host,
            result
                .many
                .get("--allowed-host")
                .map(Vec::as_slice)
                .unwrap_or(&[]),
            result
                .many
                .get("--allowed-origin")
                .map(Vec::as_slice)
                .unwrap_or(&[]),
        )
        .map_err(|e| e.1)?;
        crate::models::names(
            "owner/model",
            result
                .many
                .get("--served-model-name")
                .map(Vec::as_slice)
                .unwrap_or(&[]),
            result.switches.contains("--announce-served-name"),
        )
        .map_err(|e| e.1)?;
        Ok((output, result))
    }
    pub fn scrub_key_args(&self, args: &[String]) -> Vec<String> {
        let mut clean = vec![];
        let mut index = 0;
        while index < args.len() {
            if args[index] == "--api-key" {
                index += 2;
            } else {
                clean.push(args[index].clone());
                index += 1;
            }
        }
        clean
    }
    pub fn key(&self) -> Option<&str> {
        self.key.as_deref()
    }
    pub fn apply(&self, plan: &mut Value, redact: bool) -> Result<()> {
        let mut native: Vec<String> = serde_json::from_value(plan["config"]["native"].clone())?;
        let mut image_pixels = None;
        for (flag, value) in &self.values {
            match flag.as_str() {
                "--max-memory" => {
                    native[4] = size(value, true, false)?
                        .map(|n| n.to_string())
                        .unwrap_or("auto".into())
                }
                "--max-context" => native[3] = context(value)?,
                "--host-reserve" => {
                    let bytes = size(value, true, false)?;
                    if plan["config"]["vision"]["profile"].is_string() && bytes.is_none() {
                        return Err("ANE vision requires a positive explicit host reserve".into());
                    }
                    set_native(&mut native, flag, bytes.map(|n| n.to_string()));
                    if let Some(bytes) = bytes {
                        if plan["config"]["vision"].is_object() {
                            plan["config"]["vision"]["host_reserve"] = json!(bytes);
                        }
                    }
                }
                "--kv-format" => set_native(
                    &mut native,
                    flag,
                    (value != "int8").then_some(value.clone()),
                ),
                "--idle-release" => set_native(
                    &mut native,
                    flag,
                    Some(
                        duration(value, true)?
                            .map(|s| s.to_string())
                            .unwrap_or("off".into()),
                    ),
                ),
                "--decode-share" => {
                    set_native(&mut native, flag, Some(value.parse::<f64>()?.to_string()))
                }
                "--max-cache-disk" => {
                    let bytes = size(value, false, true)?.unwrap();
                    if native.get(5).is_some_and(|v| v.parse::<u64>().is_ok()) {
                        native[5] = bytes.to_string();
                    } else {
                        native.insert(5, bytes.to_string());
                    }
                    if bytes == 0 {
                        if self.switches.contains("--persistent-cache")
                            || self.values.contains_key("--cache-dir")
                        {
                            return Err("persistent cache requires a positive disk quota".into());
                        }
                        set_native(&mut native, "--cache-dir", None);
                        plan["environment"]
                            .as_object_mut()
                            .ok_or("environment missing")?
                            .remove("DASH_APC_DIR");
                    }
                }
                "--cache-dir" => {
                    let path = crate::model_install::local_spec(value)?;
                    let path = std::path::absolute(path)?;
                    set_native(&mut native, flag, Some(path.display().to_string()));
                    plan["environment"]["DASH_APC_DIR"] = json!(path);
                }
                "--host" => plan["config"]["host"] = json!(value),
                "--queue-size" => plan["config"]["queue_size"] = json!(value.parse::<usize>()?),
                "--max-request-size" => {
                    plan["config"]["max_request_bytes"] = json!(size(value, false, false)?.unwrap())
                }
                "--request-timeout" => {
                    plan["config"]["request_timeout"] = json!(duration(value, false)?.unwrap())
                }
                "--default-reasoning-effort" => {
                    plan["config"]["default_reasoning_effort"] = json!(value)
                }
                "--max-image-pixels" => {
                    let pixels = value.parse::<usize>()?;
                    plan["config"]["max_image_pixels"] = json!(pixels);
                    image_pixels = Some(pixels);
                }
                _ => (),
            }
        }
        if self.switches.contains("--no-webui") {
            plan["config"]["webui"] = json!(false);
        }
        if self.switches.contains("--announce-served-name") {
            plan["config"]["announce_served_name"] = json!(true);
        }
        if self.switches.contains("--disable-ane") {
            set_native(&mut native, "--ane", Some("off".into()));
        }
        if self.switches.contains("--allow-idle-sleep") {
            set_native(&mut native, "--idle-sleep", Some("allow".into()));
        }
        if let Some(pixels) = image_pixels {
            set_native(
                &mut native,
                "--max-image-patches",
                Some((pixels / 256 / 4 * 4).to_string()),
            );
        }
        for (flag, key) in [
            ("--allowed-host", "allowed_hosts"),
            ("--allowed-origin", "allowed_origins"),
            ("--served-model-name", "served_model_names"),
        ] {
            if let Some(values) = self.many.get(flag) {
                plan["config"][key] = json!(values);
            }
        }
        if let Some(key) = &self.key {
            plan["environment"]["JUSTFIT_API_KEY"] = json!(if redact { "[redacted]" } else { key });
        }
        plan["config"]["native"] = json!(native);
        Ok(())
    }
}
fn context(value: &str) -> Result<String> {
    let value = value.trim().to_ascii_uppercase();
    if value == "AUTO" {
        return Ok("auto".into());
    }
    let tokens = if let Some(n) = value.strip_suffix('K') {
        n.parse::<u64>()?
            .checked_mul(1024)
            .ok_or("context overflow")?
    } else {
        value.parse()?
    };
    if !(1..=262144).contains(&tokens) {
        return Err("context must be a token count up to256K or auto".into());
    }
    Ok(tokens.to_string())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sizes_context_and_durations_follow_public_units() {
        assert_eq!(
            size(" 2gib ", false, false).unwrap(),
            Some(2 * 1024 * 1024 * 1024)
        );
        assert_eq!(context("32K").unwrap(), "32768");
        assert_eq!(duration("1.5m", false).unwrap(), Some(90.));
        for v in ["-1", "nan", "inf", "0"] {
            assert!(duration(v, false).is_err());
        }
        assert!(context("257K").is_err());
    }
    #[test]
    fn secret_and_repeated_access_options_are_preserved_and_redacted_in_plans() {
        let args = [
            "--api-key",
            "test-secret",
            "--allowed-host",
            "one.local",
            "--allowed-host",
            "two.local",
            "--served-model-name",
            "alias",
            "--announce-served-name",
            "--max-cache-disk",
            "0",
        ]
        .map(str::to_string);
        let (remaining, options) = Options::parse(&args).unwrap();
        assert!(remaining.is_empty());
        let mut plan = json!({"config":{"native":["engine","serve-native","model","auto","auto","68719476736","--cache-dir","cache"]},"environment":{"DASH_APC_DIR":"cache"}});
        options.apply(&mut plan, true).unwrap();
        assert_eq!(plan["environment"]["JUSTFIT_API_KEY"], "[redacted]");
        assert!(!plan["config"]["native"]
            .as_array()
            .unwrap()
            .contains(&json!("--cache-dir")));
        assert_eq!(plan["config"]["allowed_hosts"].as_array().unwrap().len(), 2);
    }
}
