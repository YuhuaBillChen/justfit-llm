use crate::{bad, unsupported, Frontend, Result};
use justfit_native_client_prototype::{text_request, Frame};
use serde_json::{json, Value};

pub struct Prepared {
    pub received_at: Option<std::time::Instant>,
    pub frame: Frame,
    pub thinking: bool,
    pub stop: Vec<String>,
    pub raw: bool,
    pub budget: usize,
    pub timeout: std::time::Duration,
    pub public_id: String,
    pub created_at: u64,
    // Retain media/embedding budget until the native request has terminated.
    pub vision_owner: Option<crate::vision::Owner>,
    pub constraint: Option<crate::grammar::Shared>,
    pub tools: Option<std::sync::Arc<crate::tools::Policy>>,
    pub response_schema: Option<Value>,
    pub internal_start: bool,
    // The prompt's token ids, kept only when the request asks for
    // return_token_ids (vLLM/SGLang token-in/token-out tracing).
    pub prompt_token_ids: Option<Vec<u32>>,
    // The rendered chat prompt and its tokens, kept when finished replies are
    // remembered for reuse (prompt_tokenizer::Cache::remember).
    pub rendered_prompt: Option<(String, Vec<u32>)>,
}

fn number(body: &Value, key: &str, default: f64, min: f64, max: f64) -> Result<f32> {
    let requirement=match key{"temperature"=>"a number in [0, 2]","top_p"=>"a number in (0, 1]","presence_penalty"|"frequency_penalty"=>"a number in [-2, 2]","repetition_penalty"=>"a positive number","min_p"=>"a number in [0, 1]",_=>unreachable!()};
    let error=||bad(format!("{key} must be {requirement}"));
    let n = match body.get(key).filter(|value| !value.is_null()) {
        None => default,
        Some(v) => v.as_f64().ok_or_else(error)?,
    };
    if !n.is_finite() || n < min || n > max {
        return Err(error());
    }
    // Clamp in f64 before narrowing: tiny positive values can underflow to zero.
    let n = if key == "temperature" && n > 0. && n < 0.01 { 0.01 } else { n };
    Ok(n as f32)
}

impl Frontend {
    pub fn generation(&self, body: &Value, id: u64, raw: bool) -> Result<Prepared> {
        self.generation_inner(body, id, raw, None)
    }
    pub(crate) fn generation_media(
        &self,
        body: &Value,
        id: u64,
        tokens: Vec<u32>,
        thinking: bool,
        generation: u32,
    ) -> Result<Prepared> {
        self.generation_inner(body, id, false, Some((tokens, thinking, generation)))
    }
    fn generation_inner(
        &self,
        body: &Value,
        id: u64,
        raw: bool,
        media: Option<(Vec<u32>, bool, u32)>,
    ) -> Result<Prepared> {
        let timeout = crate::deadline::request(body)?;
        if raw {
            for field in ["suffix", "logprobs"] {
                if body.get(field).is_some_and(|v| !v.is_null()) {
                    return Err(bad(format!("{field} is not supported")));
                }
            }
            if body
                .get("echo")
                .filter(|v| !v.is_null())
                .is_some_and(|v| v.as_bool() != Some(false))
            {
                return Err(bad("echo is not supported"));
            }
        }
        self.check_model(body)?;
        if !raw && crate::grammar::response_schema(body)?.is_some(){return Err(unsupported("tool/grammar generation"));}
        if body
            .get("logit_bias")
            .is_some_and(|v| !v.is_null() && v != &json!({}))
        {
            return Err(bad("logit_bias is not supported"));
        }
        if raw {
            for key in ["best_of","n"]{
                if body.get(key).filter(|value|!value.is_null()).is_some_and(|value|value.as_u64()!=Some(1)){return Err(bad(format!("{key} must be 1")));}
            }
        }else if body.get("n").filter(|value|!value.is_null()).is_some_and(|value|value.as_u64()!=Some(1))||body.get("logprobs").filter(|value|!value.is_null()).is_some_and(|value|value.as_bool()!=Some(false)){
            return Err(bad("n and logprobs are not currently supported"));
        }
        let mut generation_tokens = 0;
        let mut rendered = None;
        let (tokens, thinking) = if let Some((tokens, thinking, generation)) = media {
            generation_tokens = generation;
            (tokens, thinking)
        } else if raw {
            let tokens = match &body["prompt"] {
                Value::String(s) => self.encode(s, true).map_err(|_|bad("prompt could not be tokenized"))?,
                Value::Array(a) => {
                    if !a.iter().all(|value|value.as_number().is_some_and(|number|!number.to_string().contains(['.','e','E']))){return Err(bad("prompt must be one string or one array of token ids"));}
                    let vocab = self.tokenizer.get_vocab_size(true) as u64;
                    a.iter()
                        .map(|v| {
                            v.as_u64()
                                .filter(|n| *n < vocab)
                                .map(|n| n as u32)
                                .ok_or_else(|| bad("prompt token ids must be in the vocabulary"))
                        })
                        .collect::<Result<Vec<_>>>()?
                }
                _ => return Err(bad("prompt must be one string or one array of token ids")),
            };
            (tokens, false)
        } else {
            let mut request = body.clone();
            request["add_generation_prompt"] = json!(true);
            let prompt = self.render(&request)?;
            request["add_generation_prompt"] = json!(false);
            let history = self.render(&request)?;
            let suffix = prompt
                .strip_prefix(&history)
                .ok_or_else(|| bad("chat template must end with an assistant generation prefix"))?;
            if suffix.is_empty(){return Err(bad("chat template must end with an assistant generation prefix"));}
            let ids = self.encode(suffix, false)?;
            let tokens = self.encode_prompt(&prompt)?;
            if self.prompt_tokenizer.reuses_generated() {
                rendered = Some(prompt.clone());
            }
            if !ids.is_empty() && ids.len() < tokens.len() && tokens.ends_with(&ids) {
                generation_tokens = ids.len() as u32;
            }
            (tokens, suffix.rfind("<think>") > suffix.rfind("</think>"))
        };
        if tokens.is_empty() {
            return Err(bad("prompt must not be empty"));
        }
        if tokens.len()>=self.context as usize{return Err(crate::ApiError(400,"prompt exceeds the context window".into(),"context_length_exceeded".into()));}
        let left=self.context as usize-tokens.len();
        let requested=if raw{body.get("max_tokens").filter(|v|!v.is_null())}else{body.get("max_completion_tokens").filter(|v|!v.is_null()).or_else(||body.get("max_tokens").filter(|v|!v.is_null()))};
        let field=if !raw&&body.get("max_completion_tokens").is_some_and(|v|!v.is_null()){"max_completion_tokens"}else{"max_tokens"};
        let budget=match requested{
            None=>if raw{left.min(16)}else{left},
            Some(value)=>{
                let text=value.as_number().map(ToString::to_string).filter(|text|!text.contains(['.','e','E'])&&!text.starts_with('-')&&text!="0").ok_or_else(||bad(format!("{field} must be a positive integer")))?;
                let amount=text.parse::<usize>().ok();
                if amount.is_none_or(|amount|amount>left){return Err(crate::ApiError(400,format!("prompt and {field} exceed the context window: {} + {text} > {} tokens",tokens.len(),self.context),"context_length_exceeded".into()));}
                amount.unwrap()
            },
        };
        let thinking_budget = body
            .get("thinking_budget")
            .filter(|v| !v.is_null())
            .map_or(Ok(0), |v| {
                v.as_u64()
                    .filter(|v| *v < u32::MAX as u64 + 1)
                    .ok_or_else(|| bad("thinking_budget must be a non-negative uint32 integer"))
            })? as u32;
        if thinking_budget > 0 && (!thinking || thinking_budget as usize + 1 >= budget) {
            return Err(bad(
                "thinking_budget requires thinking and room for a final answer",
            ));
        }
        let end = if thinking_budget == 0 {
            0
        } else {
            self.tokenizer
                .token_to_id("</think>")
                .ok_or_else(|| bad("thinking end token is unavailable"))?
        };
        let return_token_ids = match body.get("return_token_ids").filter(|v| !v.is_null()) {
            None => false,
            Some(v) => v.as_bool().ok_or_else(|| bad("return_token_ids must be a boolean"))?,
        };
        let prompt_token_ids = return_token_ids.then(|| tokens.clone());
        let rendered_prompt = rendered.map(|text| (text, tokens.clone()));
        let mut frame = text_request(id, &tokens, budget as u32, thinking_budget, end)
            .map_err(|e| bad(e.to_string()))?;
        let temperature = number(body, "temperature", 1., 0., 2.)?;
        let top_k=body.get("top_k").filter(|value| !value.is_null()).map_or(Ok(20),|value|{
            let invalid=||bad("top_k must be 0 or -1 (disabled) or a positive integer");
            if value.as_i64()==Some(-1)||value.as_i64()==Some(0){return Ok(0)}
            let text=value.as_number().map(ToString::to_string).filter(|text|!text.contains(['.','e','E'])&&!text.starts_with('-')).ok_or_else(invalid)?;
            Ok(text.parse::<u32>().unwrap_or(u32::MAX))
        })?;
        let sampling = [
            if temperature > 0. && temperature < 0.01 {
                0.01f32.to_bits()
            } else {
                temperature.to_bits()
            },
            number(body, "top_p", 0.95, f32::from_bits(1) as f64, 1.)?.to_bits(),
            top_k,
            number(body, "presence_penalty", 0., -2., 2.)?.to_bits(),
            number(body, "frequency_penalty", 0., -2., 2.)?.to_bits(),
            number(
                body,
                "repetition_penalty",
                1.,
                f32::from_bits(1) as f64,
                f32::MAX as f64,
            )?
            .to_bits(),
            number(body, "min_p", 0., 0., 1.)?.to_bits(),
        ];
        for (i, n) in sampling.iter().enumerate() {
            frame.payload[38 + i * 4..42 + i * 4].copy_from_slice(&n.to_le_bytes());
        }
        // Public IDs are not the predictable internal routing counter.
        use std::io::Read;
        let mut entropy = [0u8; 24];
        std::fs::File::open("/dev/urandom")
            .and_then(|mut f| f.read_exact(&mut entropy))
            .map_err(|_| {
                crate::ApiError(
                    500,
                    "request entropy unavailable".into(),
                    "internal_server_error".into(),
                )
            })?;
        let public_id = entropy[..16].iter().map(|b| format!("{b:02x}")).collect();
        let seed = match body.get("seed").filter(|v| !v.is_null()) {
            Some(v) => v
                .as_u64()
                .ok_or_else(|| bad("seed must be an unsigned 64-bit integer"))?,
            None => u64::from_le_bytes(entropy[16..].try_into().unwrap()),
        };
        frame.payload[66..74].copy_from_slice(&seed.to_le_bytes());
        frame.payload[78..82].copy_from_slice(&generation_tokens.to_le_bytes());
        let ignore = body
            .get("ignore_eos")
            .map_or(Ok(false), |v| {
                v.as_bool()
                    .ok_or_else(|| bad("ignore_eos must be a boolean"))
            })?;
        let progress = body.get("return_progress").map_or(Ok(false), |v| {
            v.as_bool()
                .ok_or_else(|| bad("return_progress must be a boolean"))
        })?;
        if progress && body["stream"] != true {
            return Err(bad(
                "return_progress requires stream: true and must be a boolean",
            ));
        }
        let flags = ignore as u32 | ((progress as u32) << 1);
        frame.payload[82..86].copy_from_slice(&flags.to_le_bytes());
        frame.payload[8] = match body.get("priority").map_or(Ok("normal"), |v| {
            v.as_str().ok_or_else(|| bad("priority must be foreground, normal, or background"))
        })? {
            "foreground" => 0,
            "normal" => 1,
            "background" => 2,
            _ => return Err(bad("priority must be foreground, normal, or background")),
        };
        let stop = match body.get("stop").filter(|v| !v.is_null()) {
            None => vec![],
            Some(Value::String(s)) if !s.is_empty() => vec![s.clone()],
            Some(Value::Array(a)) if a.len() <= 4 => a
                .iter()
                .map(|v| {
                    v.as_str()
                        .filter(|s| !s.is_empty())
                        .map(String::from)
                        .ok_or_else(|| bad("stop must be a string or up to four strings"))
                })
                .collect::<Result<Vec<_>>>()?,
            _ => return Err(bad("stop must be a string or up to four strings")),
        };
        Ok(Prepared {
                received_at: None,
            frame,
            thinking,
            stop,
            raw,
            budget,
            timeout,
            public_id,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            vision_owner: None,
            constraint: None,
            tools: None,
            response_schema: None,
            internal_start: false,
            prompt_token_ids,
            rendered_prompt,
        })
    }
}

/// Incremental reasoning parser preserves split delimiters and separator state.
pub struct Splitter {
    pending: String,
    thinking: bool,
    separator: bool,
    tools: bool,
}
impl Splitter {
    pub fn new(thinking: bool) -> Self {
        Self {
            pending: String::new(),
            thinking,
            separator: false,
            tools: false,
        }
    }
    pub fn set_tools(&mut self,tools:bool){self.tools=tools;}
    pub fn put(&mut self, text: &str) -> Vec<(&'static str, String)> {
        if text.is_empty() {return vec![]}
        if !self.thinking {
            let text = if self.separator {
                text.trim_start_matches('\n')
            } else {
                text
            };
            if self.separator && text.is_empty() {
                return vec![];
            }
            self.separator = false;
            return vec![("content", text.into())];
        }
        self.pending.push_str(text);
        let call="<tool_call>\n<function=";
        let end=self.pending.find("</think>").map(|i|(i,false)).into_iter().chain(if self.tools{self.pending.find(call).map(|i|(i,true))}else{None}).min();
        if let Some((index,is_call)) = end {
            let reasoning = self.pending[..index].to_string();
            let content = self.pending[index + if is_call{0}else{8}..].to_string();
            self.pending.clear();
            self.thinking = false;
            self.separator = !is_call;
            let mut out = if reasoning.is_empty() {
                vec![]
            } else {
                vec![("reasoning_content", reasoning)]
            };
            out.extend(self.put(&content));
            return out;
        }
        let mut held = 0;
        for n in 1..8 {
            if self.pending.ends_with(&"</think>"[..n]) {
                held = n;
            }
        }
        if self.tools{for n in 1..call.len(){if self.pending.ends_with(&call[..n]){held=held.max(n);}}}
        let emit = self.pending.len() - held;
        let ready = self.pending[..emit].to_string();
        self.pending = self.pending[emit..].to_string();
        if ready.is_empty() {
            vec![]
        } else {
            vec![("reasoning_content", ready)]
        }
    }
    pub fn reasoning_open(&self) -> bool {
        self.thinking
    }
    pub fn finish(&mut self) -> Vec<(&'static str, String)> {
        if self.pending.is_empty() {
            return vec![];
        }
        vec![(
            if self.thinking {
                "reasoning_content"
            } else {
                "content"
            },
            std::mem::take(&mut self.pending),
        )]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_decoder_and_done_tails_do_not_emit_content_events() {
        let mut plain=Splitter::new(false);
        assert!(plain.put("").is_empty());
        assert_eq!(plain.put("323"),vec![("content","323".into())]);
        assert!(plain.put("").is_empty());assert!(plain.finish().is_empty());
        let mut thinking=Splitter::new(true);
        assert!(thinking.put("").is_empty());
        assert_eq!(thinking.put("reason</think>\n\n323"),vec![("reasoning_content","reason".into()),("content","323".into())]);
        assert!(thinking.put("").is_empty());assert!(thinking.finish().is_empty());
    }
    #[test]
    fn reasoning_delimiter_survives_every_split() {
        let input = "careful</think>\n\nanswer";
        for split in 0..=input.len() {
            let mut s = Splitter::new(true);
            let mut parts = s.put(&input[..split]);
            parts.extend(s.put(&input[split..]));
            parts.extend(s.finish());
            let reasoning = parts
                .iter()
                .filter(|(k, _)| *k == "reasoning_content")
                .map(|(_, v)| v.as_str())
                .collect::<String>();
            let content = parts
                .iter()
                .filter(|(k, _)| *k == "content")
                .map(|(_, v)| v.as_str())
                .collect::<String>();
            assert_eq!(reasoning, "careful");
            assert_eq!(content, "answer");
        }
    }
}
