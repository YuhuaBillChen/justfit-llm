//! Finite-option scoring contracts; native raw logits, never generated guesses.
//! Direct-options prompt derived from SemIf (MIT), TheoLeeCJ, 2026.
//! Full attribution is retained in SEMIF_LICENSE.txt.
use crate::{bad, ApiError, Frontend, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
pub const DIRECT:&str="Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning.";
pub const READOUT: &str =
    "native full-vocabulary last-position logits restricted to declared answer slots";
pub const STATUS: &str = "conditional option score; uncalibrated as decision confidence";
pub struct Encoded {
    pub tokens: Vec<u32>,
    pub slots: Vec<u32>,
    pub prompt: String,
    pub digest: String,
}
fn nonempty(value: &Value) -> bool {
    match value {
        Value::String(v) => !v.is_empty(),
        Value::Array(v) => !v.is_empty(),
        Value::Object(v) => !v.is_empty(),
        _ => false,
    }
}
pub fn validate(body: &Value) -> Result<()> {
    let allowed = [
        "id", "state", "question", "options", "model", "timeout", "priority",
    ];
    if let Some(map) = body.as_object() {
        let extra = map
            .keys()
            .filter(|k| !allowed.contains(&k.as_str()))
            .collect::<Vec<_>>();
        if !extra.is_empty() {
            return Err(bad(format!(
                "unsupported fields: {}",
                extra.into_iter().cloned().collect::<Vec<_>>().join(", ")
            )));
        }
    }
    if !body["id"].as_str().is_some_and(|s| !s.is_empty())
        || !body["question"].as_str().is_some_and(|s| !s.is_empty())
    {
        return Err(bad("id and question must be nonempty strings"));
    }
    if !nonempty(&body["state"]) {
        return Err(bad("state must be a nonempty string, object, or array"));
    }
    let options = body["options"]
        .as_array()
        .filter(|a| (2..=16).contains(&a.len()))
        .ok_or_else(|| bad("options must contain 2-16 entries"))?;
    let mut names = BTreeSet::new();
    for option in options {
        let id = option["id"]
            .as_str()
            .filter(|_| option["description"].is_string())
            .ok_or_else(|| bad("Each option needs string id and description fields"))?;
        if !names.insert(id) {
            return Err(bad("Option IDs must be unique"));
        }
    }
    Ok(())
}
pub fn messages(body: &Value) -> Result<Value> {
    validate(body)?;
    let options=body["options"].as_array().unwrap().iter().enumerate().map(|(i,v)|json!({"letter":((b'A'+i as u8)as char).to_string(),"description":v["description"]})).collect::<Vec<_>>();
    Ok(
        json!([{"role":"system","content":DIRECT},{"role":"user","content":crate::python_json::dumps(&json!({"evidence":body["state"],"criterion":body["question"],"options":options}))}]),
    )
}
pub fn encode(front: &Frontend, messages: Value, labels: &[String]) -> Result<Encoded> {
    let unsupported = |s: String| ApiError(500, s, "scoring_unsupported".into());
    let prompt = front.render(&json!({"messages":messages,"reasoning_effort":"none"}))?;
    let tokens = front.encode(&prompt, false)?;
    if tokens.is_empty() {
        return Err(unsupported("the tokenizer produced an empty prompt".into()));
    }
    if tokens.len() > front.context as usize {
        return Err(ApiError(
            400,
            "prompt exceeds the context window".into(),
            "context_length_exceeded".into(),
        ));
    }
    let mut slots = Vec::new();
    for label in labels {
        let encoded = front.encode(label, false)?;
        if encoded.len() != 1
            || front
                .tokenizer
                .decode(&encoded, false)
                .map_err(|_| unsupported("answer slot could not be decoded".into()))?
                != *label
        {
            return Err(unsupported(format!(
                "answer slot {label:?} is not one exact round-trip token"
            )));
        }
        if slots.contains(&encoded[0]) {
            return Err(unsupported("answer-slot tokens collide".into()));
        }
        slots.push(encoded[0]);
        let mut expected = tokens.clone();
        expected.push(encoded[0]);
        if front.encode(&(prompt.clone() + label), false)? != expected {
            return Err(unsupported(format!(
                "answer boundary changes tokenization for slot {label:?}"
            )));
        }
    }
    let digest = format!("{:x}", Sha256::digest(prompt.as_bytes()));
    Ok(Encoded {
        tokens,
        slots,
        prompt,
        digest,
    })
}
pub fn compensated(values: impl IntoIterator<Item = f64>) -> f64 {
    let (mut high, mut low) = (0f64, 0f64);
    for value in values {
        let total = high + value;
        low += if high.abs() >= value.abs() {
            (high - total) + value
        } else {
            (value - total) + high
        };
        high = total;
    }
    high + low
}
pub fn softmax(logits: &[f64]) -> Result<Vec<f64>> {
    if logits.len() < 2 || logits.iter().any(|v| !v.is_finite()) {
        return Err(bad("Need at least two finite scores"));
    }
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights = logits
        .iter()
        .map(|v| (v - maximum).exp())
        .collect::<Vec<_>>();
    let sum = compensated(weights.iter().copied());
    Ok(weights.into_iter().map(|v| v / sum).collect())
}
pub fn fixture(front: &Frontend, body: &Value) -> Result<Value> {
    let messages = messages(body)?;
    let labels = (0..body["options"].as_array().unwrap().len())
        .map(|i| ((b'A' + i as u8) as char).to_string())
        .collect::<Vec<_>>();
    let encoded = encode(front, messages.clone(), &labels)?;
    let frame = justfit_native_client_prototype::score_request(1, &encoded.tokens, &encoded.slots)
        .map_err(|_| bad("invalid score wire"))?;
    let mut bytes = Vec::new();
    justfit_native_client_prototype::write_frame(&mut bytes, &frame)
        .map_err(|_| bad("score serialization failed"))?;
    Ok(
        json!({"messages":messages,"prompt":encoded.prompt,"tokens":encoded.tokens,"slots":encoded.slots,"prompt_sha256":encoded.digest,"wire":bytes}),
    )
}
