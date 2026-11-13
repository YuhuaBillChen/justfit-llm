use justfit_rust_frontend::{Frontend,documents};
use serde_json::{json,Value};
use std::{io::{self,BufRead},path::Path};
fn main()->Result<(),Box<dyn std::error::Error>>{
    let args:Vec<_>=std::env::args().skip(1).collect();
    if args.first().map(String::as_str)==Some("pdf-worker"){return documents::worker(&args[1],&args[2],&args[3]).map_err(|e|e.1.into())}
    let front=Frontend::load(Path::new(&args[0]),"model".into(),262144).map_err(|e|e.1)?;
    for line in io::stdin().lock().lines(){let body:Value=serde_json::from_str(&line?)?;let result=front.render(&body);let value=match result{Ok(prompt)=>json!({"status":200,"body":{"prompt":prompt}}),Err(error)=>json!({"status":error.0,"body":error.body()})};println!("{value}");}
    Ok(())
}
