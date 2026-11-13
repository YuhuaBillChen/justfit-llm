//! The native model's released tokenizer contract, before engine admission.
use crate::{bad,Result};
use serde_json::Value;
use std::path::Path;
use tokenizers::Tokenizer;

pub fn validate(tokenizer: &Tokenizer, config: &Value) -> Result<()> {
    let vocabulary=tokenizer.get_vocab(true);
    if vocabulary.is_empty() || vocabulary.values().any(|id|*id>=248320) {
        return Err(bad("tokenizer vocabulary does not fit the native model"));
    }
    for (token,id) in [("<|endoftext|>",248044),("<|im_end|>",248046),("</think>",248069),("<tool_call>",248058)] {
        if vocabulary.get(token)!=Some(&id) || !tokenizer.encode(token,false).is_ok_and(|encoded|encoded.get_ids()==[id]) {
            return Err(bad(format!("tokenizer must encode '{token}' as native token {id}")));
        }
    }
    let eos=&config["eos_token"];
    let eos=eos.as_str().or_else(||eos["content"].as_str());
    if !eos.and_then(|eos|vocabulary.get(eos)).is_some_and(|id|[248044,248046].contains(id)) {
        return Err(bad("tokenizer EOS token does not match the native model"));
    }
    Ok(())
}
pub fn check(path: &Path) -> Result<()> {
    let tokenizer=Tokenizer::from_file(path.join("tokenizer.json")).map_err(|e|bad(e.to_string()))?;
    let config:Value=serde_json::from_slice(&std::fs::read(path.join("tokenizer_config.json")).map_err(|e|bad(e.to_string()))?).map_err(|e|bad(e.to_string()))?;
    validate(&tokenizer,&config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn tokenizer(case: &str) -> Tokenizer {
        let mut vocabulary=vec![("unknown".to_string(),0),("<|endoftext|>".to_string(),248044),("<|im_end|>".to_string(),248046),("</think>".to_string(),248069),("<tool_call>".to_string(),248058)];
        if case=="outside" {vocabulary.push(("outside".into(),248320));}
        if case=="marker" {vocabulary[2].1=3;}
        if case=="empty" {vocabulary.clear();}
        Tokenizer::new(tokenizers::models::wordlevel::WordLevel::builder().vocab(vocabulary.into_iter().collect()).unk_token("unknown".into()).build().unwrap())
    }
    #[test]
    fn native_marker_and_eos_errors_match_released_contract() {
        for (case,eos,expected) in [
            ("valid",json!("<|im_end|>"),None),
            ("valid",json!({"content":"<|endoftext|>"}),None),
            ("empty",json!("<|im_end|>"),Some("tokenizer vocabulary does not fit the native model")),
            ("outside",json!("<|im_end|>"),Some("tokenizer vocabulary does not fit the native model")),
            ("marker",json!("<|im_end|>"),Some("tokenizer must encode '<|im_end|>' as native token 248046")),
            ("valid",json!("unknown"),Some("tokenizer EOS token does not match the native model")),
        ] {
            let actual=validate(&tokenizer(case),&json!({"eos_token":eos})).err().map(|e|e.1);
            assert_eq!(actual.as_deref(),expected,"{case}");
        }
    }
}
