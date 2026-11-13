//! Rust service migration. Unsupported capabilities remain explicit parity gaps.
pub mod ane_native;
pub mod ane_bundle;
pub mod ane_distribution;
pub mod vision_identity;
pub mod access;
pub mod coreai;
pub mod engine;
pub mod generation;
pub mod deadline;
pub mod telemetry;
pub mod latency;
pub mod documents;
pub mod sandbox;
pub mod preparation_options;
pub mod preparation_queue;
pub mod prompt_tokenizer;
pub mod image_cache;
pub mod chat_templates;
pub mod cli;
pub mod cli_help;
pub mod replay;
pub mod cli_platform;
pub mod cli_serve;
pub mod serve_options;
pub mod build_helpers;
pub mod cli_build;
pub mod download;
pub mod runtime_bundle;
pub mod model_install;
pub mod model_hold;
pub mod source_mlx;
pub mod gguf_metadata;
pub mod source_gguf;
pub mod source_hub;
mod native_libraries;
pub mod template_startup;
pub mod tokenizer_startup;
pub mod models;
pub mod grammar;
pub mod http;
pub mod media;
pub mod native_driver;
pub mod native_prepare;
pub mod responses;
pub mod tools;
pub mod tool_arguments;
pub mod tool_history;
pub mod tool_output;
pub mod tool_grammar;
pub mod template_values;
pub mod anthropic;
pub mod thinking;
pub mod python_json;
pub mod request_json;
pub mod judgments;
pub mod systemone;
pub mod vision;
pub mod vision_math;
pub mod worker_meter;
use axum::http::HeaderMap;
use minijinja::{Environment, Error, ErrorKind};
use serde_json::{json, Value};
use std::path::Path;
use subtle::ConstantTimeEq;
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct ApiError(pub u16, pub String, pub String);
pub type Result<T> = std::result::Result<T, ApiError>;
pub fn bad(s: impl Into<String>) -> ApiError {
    ApiError(400, s.into(), "invalid_request_error".into())
}
pub fn unsupported(s: &str) -> ApiError {
    ApiError(
        501,
        format!("Rust parity gap: {s}"),
        "not_implemented".into(),
    )
}
impl ApiError {
    pub fn body(&self) -> Value {
        json!({"error":{"message":self.1,"type":if self.0>=500 {"server_error"}else{"invalid_request_error"},"code":self.2}})
    }
}

pub fn normalize_path(raw: &str) -> String {
    let raw = raw.split(['?', '#']).next().unwrap_or("");
    let decoded = percent_encoding::percent_decode_str(raw).decode_utf8_lossy();
    if !decoded.starts_with('/') {
        return decoded.into_owned();
    }
    let mut parts = Vec::new();
    for part in decoded.split('/') {
        match part {
            "" | "." => (),
            ".." => {
                parts.pop();
            }
            _ => parts.push(part),
        }
    }
    let mut result = format!("/{}", parts.join("/"));
    if decoded.ends_with('/') && result != "/" {
        result.push('/');
    }
    result
}

pub(crate) fn authority(value: &str) -> Result<(String, Option<u16>)> {
    if value.is_empty()
        || value.bytes().any(|c| !(33..127).contains(&c))
        || value.contains(['@', '/', '?', '#', '\\'])
    {
        return Err(ApiError(
            403,
            "invalid Host header".into(),
            "forbidden".into(),
        ));
    }
    let a: axum::http::uri::Authority = value
        .parse()
        .map_err(|_| ApiError(403, "invalid Host header".into(), "forbidden".into()))?;
    if a.port().is_some() && a.port_u16().is_none() {
        return Err(ApiError(
            403,
            "invalid Host header".into(),
            "forbidden".into(),
        ));
    }
    Ok((
        a.host()
            .trim_matches(['[', ']'])
            .to_lowercase()
            .trim_end_matches('.')
            .to_string(),
        a.port_u16(),
    ))
}

pub fn guard(
    headers: &HeaderMap,
    path: &str,
    method: &str,
    key: Option<&str>,
    hosts: &[String],
) -> Result<()> {
    let hosts_given: Vec<_> = headers.get_all("host").iter().collect();
    let origins: Vec<_> = headers.get_all("origin").iter().collect();
    if hosts_given.len() != 1 || origins.len() > 1 {
        return Err(ApiError(
            403,
            "expected one Host header and at most one Origin header".into(),
            "forbidden".into(),
        ));
    }
    let (host, port) = authority(
        hosts_given[0]
            .to_str()
            .map_err(|_| bad("invalid Host header"))?,
    )?;
    if !hosts.contains(&host) {
        return Err(ApiError(
            403,
            format!("Host {host} is not allowed"),
            "forbidden".into(),
        ));
    }
    if let Some(origin) = origins.first() {
        let origin = origin
            .to_str()
            .map_err(|_| ApiError(403, "invalid Origin header".into(), "forbidden".into()))?;
        let same = ["http", "https"].iter().any(|scheme| {
            let Some(rest) = origin.strip_prefix(&format!("{scheme}://")) else {
                return false;
            };
            authority(rest).is_ok_and(|(h, p)| {
                h == host
                    && p.unwrap_or(if *scheme == "http" { 80 } else { 443 })
                        == port.unwrap_or(if *scheme == "http" { 80 } else { 443 })
            })
        });
        if !same {
            return Err(ApiError(
                403,
                "Origin is not allowed".into(),
                "forbidden".into(),
            ));
        }
    }
    let public = method == "OPTIONS"
        || (["GET", "HEAD"].contains(&method)
            && ["/", "/index.html", "/favicon.ico", "/health", "/ready"].contains(&path));
    if !public {
        if let Some(key) = key {
            let auth: Vec<_> = headers.get_all("authorization").iter().collect();
            let api: Vec<_> = headers.get_all("x-api-key").iter().collect();
            // Require at least one credential; validate every supplied credential.
            let mut valid =
                auth.len() <= 1 && api.len() <= 1 && (!auth.is_empty() || !api.is_empty());
            for a in auth {
                valid &=
                    a.to_str()
                        .ok()
                        .and_then(|s| s.split_once(' '))
                        .is_some_and(|(scheme, v)| {
                            scheme.eq_ignore_ascii_case("bearer")
                                && bool::from(v.as_bytes().ct_eq(key.as_bytes()))
                        });
            }
            for a in api {
                valid &= bool::from(a.as_bytes().ct_eq(key.as_bytes()));
            }
            if !valid {
                return Err(ApiError(
                    401,
                    "invalid or missing API key".into(),
                    "authentication_error".into(),
                ));
            }
        }
    }
    Ok(())
}

/// Preserve Python's arbitrary-size JSON integers while retaining finite-float
/// and integer-digit guards. Arbitrary precision must not enable 1e999 inputs.
pub fn strict_json(bytes: &[u8]) -> Result<Value> {
    // Match the released 128-container limit before enabling the decoder's
    // bounded recursion. Brackets inside JSON strings do not add depth.
    let mut depth=0usize;let mut quoted=false;let mut escaped=false;
    for byte in bytes {
        if quoted {
            if escaped {escaped=false;}
            else if *byte==b'\\' {escaped=true;}
            else if *byte==b'"' {quoted=false;}
        } else {
            match *byte {
                b'"'=>quoted=true,
                b'['|b'{'=>{depth+=1;if depth>128{return Err(bad("JSON nests deeper than 128 levels"));}},
                b']'|b'}'=>depth=depth.saturating_sub(1),
                _=>{},
            }
        }
    }
    let mut decoder=serde_json::Deserializer::from_slice(bytes);
    decoder.disable_recursion_limit();
    let v:Value=serde::Deserialize::deserialize(&mut decoder).map_err(|_|bad("invalid JSON request body"))?;
    decoder.end().map_err(|_|bad("invalid JSON request body"))?;
    let mut pending=vec![&v];
    while let Some(value)=pending.pop() {
        match value {
            Value::Number(number)=> {
                let text=number.to_string();
                if text.contains(['.','e','E']) {
                    if !number.as_f64().is_some_and(f64::is_finite){return Err(bad("JSON number is outside the supported range"));}
                } else if text.trim_start_matches('-').len()>4300 {
                    return Err(bad("JSON integer exceeds supported digit limit"));
                }
            },
            Value::Object(map)=>pending.extend(map.values()),
            Value::Array(array)=>pending.extend(array),
            _=>{}
        }
    }
    Ok(v)
}
pub fn json_body(bytes: &[u8]) -> Result<Value> {
    let v=strict_json(bytes)?;
    if !v.is_object() {
        return Err(bad("request body must be an object"));
    }
    Ok(v)
}

fn content(v: &Value) -> Result<String> {
    if v.is_null() {
        return Ok(String::new());
    }
    if let Some(s) = v.as_str() {
        return Ok(s.into());
    }
    if let Some(parts) = v.as_array() {
        let mut text = String::new();
        for p in parts {
            if ![Some("text"), Some("input_text")].contains(&p["type"].as_str()) {
                return Err(unsupported("image/PDF content preprocessing"));
            }
            text.push_str(
                p.get("text")
                    .unwrap_or(&Value::String(String::new()))
                    .as_str()
                    .ok_or_else(|| bad("text message content must be a string"))?,
            );
        }
        return Ok(text);
    }
    Err(bad("message content must be text"))
}

pub fn messages(body: &Value) -> Result<Value> {
    let input = body["messages"]
        .as_array()
        .filter(|v| !v.is_empty())
        .ok_or_else(|| bad("messages must be a non-empty array"))?;
    let mut normalized: Vec<Value> = Vec::new();
    let mut has_user = false;
    for m in input {
        if !m.is_object() {
            return Err(bad("each message must be an object"));
        }
        let role = m["role"]
            .as_str()
            .ok_or_else(|| bad("unsupported message role"))?;
        let role = if role == "developer" { "system" } else { role };
        if !["system", "user", "assistant", "tool"].contains(&role) {
            return Err(bad(format!("unsupported message role: {role}")));
        }
        has_user |= role == "user";
        let text = content(&m["content"])?;
        if role == "system" && normalized.len() == 1 && normalized[0]["role"] == "system" {
            let old = normalized[0]["content"].as_str().unwrap();
            normalized[0]["content"] = json!([old, &text]
                .into_iter()
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join("\n\n"));
            continue;
        }
        let mut item = json!({"role":role,"content":text});
        if role == "assistant" && !m["reasoning_content"].is_null() {
            item["reasoning_content"] = json!(content(&m["reasoning_content"])?);
        }
        if role == "assistant" && !m["tool_calls"].is_null() {
            let calls = m["tool_calls"]
                .as_array()
                .ok_or_else(|| bad("assistant tool_calls must be an array"))?;
            let mut output = Vec::new();
            for call in calls {
                let f = &call["function"];
                let name = f["name"]
                    .as_str()
                    .ok_or_else(|| bad("invalid assistant tool call"))?;
                let (args,unfinished) = match f.get("arguments") {
                    None => (json!({}),false),
                    Some(Value::String(text)) => match strict_json(if text.trim().is_empty(){b"{}"}else{text.as_bytes()}) {
                        Ok(value)=>(value,false),
                        Err(_) if crate::tool_history::unfinished(text)=>(json!(text),true),
                        Err(_)=>return Err(bad("tool call arguments must be valid JSON")),
                    },
                    Some(value)=>(value.clone(),false),
                };
                if !args.is_object()&&!unfinished{return Err(bad("invalid assistant tool call"));}
                let mut c = json!({"type":"function","function":{"name":name,"arguments":args}});
                if let Some(id) = call.get("id") {
                    if !id.is_string() {
                        return Err(bad("tool call id must be a string"));
                    }
                    c["id"] = id.clone();
                }
                output.push(c);
            }
            item["tool_calls"] = json!(output);
        }
        if role == "tool" {
            if let Some(id) = m.get("tool_call_id") {
                if !id.is_string() {
                    return Err(bad("tool_call_id must be a string"));
                }
                item["tool_call_id"] = id.clone();
            }
        }
        normalized.push(item);
    }
    if !has_user {
        return Err(bad("messages must include a user message"));
    }
    Ok(crate::tool_history::template_messages(normalized))
}

pub struct Frontend {
    pub latencies: std::sync::Arc<crate::latency::Metrics>,
    pub images: std::sync::Arc<crate::image_cache::Cache>,
    pub tokenizer: Tokenizer,
    pub prompt_tokenizer: crate::prompt_tokenizer::Cache,
    template: String,
    pub chat_templates: crate::chat_templates::Templates,
    pub model: String,
    pub response_model: String,
    pub model_names: Vec<String>,
    pub context: u32,
    pub preparation: crate::preparation_options::Options,
    pub documents: std::sync::Arc<crate::documents::Documents>,
    pub(crate) score_labels: std::sync::OnceLock<Result<Vec<String>>>,
}
impl Frontend {
    pub fn load(path:&Path,model:String,context:u32)->Result<Self>{Self::load_with_templates(path,model,context,None)}
    pub fn load_with_templates(path:&Path,model:String,context:u32,templates:Option<crate::chat_templates::Templates>)->Result<Self>{
        if let Some(templates)=&templates{templates.validate()?;}
        let tokenizer =
            Tokenizer::from_file(path.join("tokenizer.json")).map_err(|e| bad(e.to_string()))?;
        let defined=match &templates{Some(templates)=>templates.clone(),None=>crate::template_startup::defined(path)?};
        defined.validate()?;
        let template=defined.select(&json!({})).source.clone();
        let mut front=Self {
            chat_templates:crate::chat_templates::Templates::raw(template.clone()),
            latencies: std::sync::Arc::new(crate::latency::Metrics::default()),
            prompt_tokenizer: crate::prompt_tokenizer::Cache::new(&tokenizer),
            images: std::sync::Arc::new(crate::image_cache::Cache::default()),
            tokenizer,
            template,
            response_model: model.clone(),
            model_names: vec![model.clone()],
            model,
            context,
            preparation: crate::preparation_options::Options::default(),
            documents: std::sync::Arc::new(crate::documents::Documents::new(crate::documents::Config::default())),
            score_labels: std::sync::OnceLock::new(),
        };
        if let Some(templates)=templates{front.install_templates(templates)?;}
        else{
            let prepared=crate::template_startup::prepare(&front,defined)?;
            front.install_templates(prepared)?;
            front.latencies=std::sync::Arc::new(crate::latency::Metrics::default());
        }
        Ok(front)
    }
    pub fn install_templates(&mut self,templates:crate::chat_templates::Templates)->Result<()>{
        templates.validate()?;self.template=templates.select(&json!({})).source.clone();self.chat_templates=templates;Ok(())
    }
    pub fn encode(&self, text: &str, special: bool) -> Result<Vec<u32>> {
        let _timer=self.latencies.measure("tokenization");
        self.tokenizer
            .encode(text, special)
            .map(|e| e.get_ids().to_vec())
            .map_err(|_| bad("content could not be tokenized"))
    }
    pub fn encode_prompt(&self,text:&str)->Result<Vec<u32>>{
        let _timer=self.latencies.measure("tokenization");
        self.prompt_tokenizer.encode(&self.tokenizer,text)
    }
    pub fn tokenize(&self, body: &Value) -> Result<Vec<u32>> {
        let text = body["content"]
            .as_str()
            .ok_or_else(|| bad("content must be a string"))?;
        let special = body.get("add_special").map_or(Ok(false), |v| {
            v.as_bool()
                .ok_or_else(|| bad("add_special must be a boolean"))
        })?;
        for (k, default) in [("parse_special", true), ("with_pieces", false)] {
            if body.get(k).is_some_and(|v| v.as_bool() != Some(default)) {
                return Err(bad(format!("only {k}={default} is supported")));
            }
        }
        self.encode(text, special)
    }
    pub fn apply_template_checked(&self,body:&Value,check:&dyn Fn()->Result<()>)->Result<String>{
        let prompt=self.render_checked(body,check)?;
        if body["add_generation_prompt"]!=false&&!crate::vision::has_media(body){
            check()?;let mut history=body.clone();history["add_generation_prompt"]=json!(false);
            let prefix=self.render_checked(&history,check)?;
            if !prompt.strip_prefix(&prefix).is_some_and(|suffix|!suffix.is_empty()){return Err(bad("chat template must end with an assistant generation prefix"))}
        }Ok(prompt)
    }
    pub fn render(&self, body: &Value) -> Result<String> {self.render_checked(body,&||Ok(()))}
    pub fn render_checked(&self,body:&Value,check:&dyn Fn()->Result<()>)->Result<String>{
        check()?;
        if crate::vision::has_media(body){return self.media_render_checked(body,check)}
        let messages=messages(body)?;let template=self.chat_templates.select(body);
        crate::chat_templates::accepts(template,&messages)?;
        self.render_messages(body,messages,&template.source)
    }
    pub(crate) fn render_messages(
        &self,
        body: &Value,
        messages: Value,
        source: &str,
    ) -> Result<String> {
        let _timer=self.latencies.measure("template");
        self.check_model(body)?;
        let mut context = json!({"messages":messages,"add_generation_prompt":true});
        if let Some(v) = body.get("add_generation_prompt") {
            if !v.is_boolean() {
                return Err(bad("add_generation_prompt must be a boolean"));
            }
            context["add_generation_prompt"] = v.clone();
        }
        let kwargs = body
            .get("chat_template_kwargs")
            .filter(|v| !v.is_null())
            .cloned()
            .unwrap_or(json!({}));
        let kwargs = kwargs
            .as_object()
            .ok_or_else(|| bad("chat_template_kwargs must be an object"))?;
        for (k, v) in kwargs {
            if [
                "add_generation_prompt",
                "chat_template",
                "continue_final_message",
                "conversation",
                "documents",
                "messages",
                "return_dict",
                "tokenize",
                "tools",
            ]
            .contains(&k.as_str())
            {
                return Err(bad(format!("chat_template_kwargs cannot set {k}")));
            }
            context[k] = v.clone();
        }
        let default_effort=self.preparation.default_reasoning_effort.as_ref().map(|s|json!(s));
        let effort = body.get("reasoning_effort").filter(|v| !v.is_null()).or(default_effort.as_ref());
        if effort.is_some_and(|v| {
            !v.as_str().is_some_and(|s| {
                ["none", "minimal", "low", "medium", "high", "xhigh", "max"].contains(&s)
            })
        }) {
            return Err(bad("invalid reasoning_effort"));
        }
        let thinking = kwargs.get("enable_thinking").filter(|v| !v.is_null());
        if thinking.is_some_and(|v| !v.is_boolean()) {
            return Err(bad(
                "chat_template_kwargs enable_thinking must be a boolean",
            ));
        }
        let enable = thinking
            .and_then(Value::as_bool)
            .or_else(|| effort.map(|e| e != "none"));
        if let Some(enable) = enable {
            context["enable_thinking"] = json!(enable);
            if !enable {
                context["reasoning_effort"] = json!("none");
            } else if let Some(e) = effort.filter(|e| *e != "none") {
                context["reasoning_effort"] = e.clone();
            }
        }
        if let Some(p) = body.get("preserve_thinking").filter(|v| !v.is_null()) {
            if !p.is_boolean() {
                return Err(bad("preserve_thinking must be a boolean"));
            }
            context["preserve_thinking"] = p.clone();
        }
        if let Some(tools) = body
            .get("tools")
            .filter(|v| v.as_array().is_some_and(|a| !a.is_empty()))
        {
            context["tools"] = tools.clone();
        }
        let mut env = Environment::new();
        env.add_filter("tojson",template_values::tojson);
        env.set_fuel(Some(10_000_000));
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function(
            "raise_exception",
            |s: String| -> std::result::Result<String, Error> {
                Err(Error::new(ErrorKind::InvalidOperation, s))
            },
        );
        env.add_template("chat", source)
            .map_err(|_| bad("messages could not be rendered"))?;
        let template = env.get_template("chat").unwrap();
        match template.render(template_values::value(&context)) {
            Ok(s) => Ok(s),
            Err(_) => {
                let alias = match context["reasoning_effort"].as_str() {
                    Some("high" | "max") => Some("xhigh"),
                    Some("minimal") => Some("low"),
                    _ => None,
                };
                if let Some(alias) = alias {
                    context["reasoning_effort"] = json!(alias);
                } else if context["reasoning_effort"] == "none"
                    && context["enable_thinking"] == false
                {
                    context.as_object_mut().unwrap().remove("reasoning_effort");
                } else {
                    return Err(bad("messages could not be rendered"));
                }
                template
                    .render(template_values::value(&context))
                    .map_err(|_| bad("messages could not be rendered"))
            }
        }
    }
    pub fn dispatch(&self, op: &str, body: &Value) -> Result<Value> {
        match op {
            // JSONL acceptance oracle: template inputs before HTTP/media normalization.
            "template-probe"=>{
                let template=self.chat_templates.select(body);
                let messages=body.get("messages").filter(|v|v.is_array()).ok_or_else(||bad("messages must be an array"))?;
                crate::chat_templates::accepts(template,messages)?;
                Ok(json!({"prompt":self.render_messages(body,messages.clone(),&template.source)?}))
            }
            "systemone-prepare"=>systemone::prepare_fixture(self,body),
            "systemone-validate"=>Ok(systemone::fixture(body)),
            "judgment-prepare"=>judgments::fixture(self,body),
            "python-json"=>Ok(json!(python_json::dumps(&body["value"]))),
            "thinking-codec"=>thinking::fixture(body),
            "anthropic-adapter"=>anthropic::fixture(body),
            "tools-project" => tool_output::fixture(body),
            "tools-frame" => tool_arguments::fixture(body),
            "tools-convert" => tools::conversion_fixture(body),
            "tools-types" => Ok(tools::types_fixture(body)),
            "tools-normalize" => tools::fixture(body),
            "media-count" => Ok(json!(self.media_token_count(body)?)),
            "media-wire" => {
                let prepared = self.media_generation(body, 1)?;
                let mut bytes = Vec::new();
                justfit_native_client_prototype::write_frame(&mut bytes, &prepared.frame)
                    .map_err(|e| bad(e.to_string()))?;
                use sha2::{Digest, Sha256};
                Ok(
                    json!({"sha256":format!("{:x}",Sha256::digest(&bytes)),"bytes":bytes.len(),"thinking":prepared.thinking,"stop":prepared.stop}),
                )
            }
            "responses-stream" => http::responses_stream_fixture(body),
            "responses-output" => Ok(http::responses_fixture(body)),
            "responses-adapter" => responses::convert(body),
            "generation-wire" | "completion-wire" => {
                let prepared = self.generation(body, 1, op == "completion-wire")?;
                let mut bytes = Vec::new();
                justfit_native_client_prototype::write_frame(&mut bytes, &prepared.frame)
                    .map_err(|e| bad(e.to_string()))?;
                Ok(json!({"bytes":bytes}))
            }
            "guard" => {
                let mut headers = HeaderMap::new();
                for h in body["headers"]
                    .as_array()
                    .ok_or_else(|| bad("missing headers"))?
                {
                    let name: axum::http::HeaderName = h[0]
                        .as_str()
                        .ok_or_else(|| bad("header name"))?
                        .parse()
                        .map_err(|_| bad("header name"))?;
                    let value: axum::http::HeaderValue = h[1]
                        .as_str()
                        .ok_or_else(|| bad("header value"))?
                        .parse()
                        .map_err(|_| bad("header value"))?;
                    headers.append(name, value);
                }
                let path = normalize_path(body["path"].as_str().unwrap_or("/v1/chat/completions"));
                guard(
                    &headers,
                    &path,
                    body["method"].as_str().unwrap_or("POST"),
                    body["key"].as_str(),
                    &["localhost".into(), "127.0.0.1".into(), "::1".into()],
                )?;
                Ok(json!({"admitted":true}))
            }
            "parse-json" => json_body(
                body["raw"]
                    .as_str()
                    .ok_or_else(|| bad("missing raw"))?
                    .as_bytes(),
            ),
            "tokenize" => Ok(json!({"tokens":self.tokenize(body)?})),
            "apply-template" => Ok(json!({"prompt":self.render(body)?})),
            "prepare" => {
                let prompt = self.render(body)?;
                Ok(json!({"prompt":prompt,"tokens":self.encode(&prompt,false)?}))
            }
            "normalize-messages" => messages(body),
            _ => Err(unsupported(op)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn paths_do_not_bypass_authentication() {
        assert_eq!(
            normalize_path("/public/../v1/%63hat/completions?x"),
            "/v1/chat/completions"
        );
    }
    #[test]
    fn json_depth_matches_released_128_container_limit() {
        for depth in [1usize,64,126,127,128,129,256] {
            let raw=[r#"{"value":"#, &"[".repeat(depth-1), "0", &"]".repeat(depth-1), "}"].concat();
            assert_eq!(json_body(raw.as_bytes()).is_ok(),depth<=128,"depth {depth}");
        }
        let quoted=serde_json::to_vec(&json!({"value":"[".repeat(1024)+&char::from(34u8).to_string()+"}"})).unwrap();
        assert!(json_body(&quoted).is_ok());
    }
    #[test]
    fn unsafe_json_is_rejected() {
        for b in [b"{\"x\":NaN}".as_slice(), b"{\"x\":1e999}", b"[]", b"{"] {
            assert!(json_body(b).is_err());
        }
    }
    #[test]
    fn json_integers_roundtrip_without_float_loss_and_float_guards_remain() {
        for number in ["18446744073709551617","-9223372036854775809","10000000000000000000000000000000000000001"] {
            let raw=format!("{{\"value\":{number}}}");let value=json_body(raw.as_bytes()).unwrap();
            assert_eq!(value["value"].to_string(),number);
        }
        assert!(strict_json(b"1e999").is_err());
        assert!(strict_json("9".repeat(4301).as_bytes()).is_err());
        assert!(strict_json("9".repeat(4300).as_bytes()).is_ok());
    }
    #[test]
    fn auth_requires_each_credential_once() {
        let mut h = HeaderMap::new();
        h.insert("host", "localhost:8000".parse().unwrap());
        let hosts = vec!["localhost".into()];
        assert!(guard(&h, "/v1/chat/completions", "POST", Some("secret"), &hosts).is_err());
        h.insert("authorization", "Bearer secret".parse().unwrap());
        assert!(guard(&h, "/v1/chat/completions", "POST", Some("secret"), &hosts).is_ok());
        h.append("authorization", "Bearer secret".parse().unwrap());
        assert_eq!(
            guard(&h, "/v1/chat/completions", "POST", Some("secret"), &hosts)
                .unwrap_err()
                .0,
            401
        );
    }
}
