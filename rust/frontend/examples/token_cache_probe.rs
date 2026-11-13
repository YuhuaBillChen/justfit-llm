use justfit_rust_frontend::{Frontend,prompt_tokenizer::Cache};
use serde_json::{json,Value};
use std::{io::{self,BufRead},path::Path};
fn main()->Result<(),Box<dyn std::error::Error>>{
    let path=std::env::args().nth(1).unwrap();let front=Frontend::load(Path::new(&path),"model".into(),262144).map_err(|e|e.1)?;let mut tokenizer=front.tokenizer.clone();let mut cache=Cache::new(&tokenizer);
    for line in io::stdin().lock().lines(){let body:Value=serde_json::from_str(&line?)?;
        if let Some(config)=body.get("replace_tokenizer"){tokenizer=tokenizers::Tokenizer::from_bytes(config.to_string().as_bytes()).map_err(|e|e.to_string())?;cache=Cache::new(&tokenizer);}
        if body["reset"]==true{cache=Cache::new(&tokenizer);}
        let value=if let Some(text)=body["text"].as_str(){json!({"tokens":cache.encode(&tokenizer,text).map_err(|e|e.1)?,"stats":cache.snapshot()})}else{json!({"stats":cache.snapshot()})};println!("{value}");
    }Ok(())
}
