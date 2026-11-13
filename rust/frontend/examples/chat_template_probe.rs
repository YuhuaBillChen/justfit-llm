use justfit_rust_frontend::Frontend;
use serde_json::{json,Value};
use std::{io::{self,BufRead},path::Path};
fn main()->Result<(),Box<dyn std::error::Error>>{
    let path=std::env::args().nth(1).unwrap();let mut front=Frontend::load(Path::new(&path),"model".into(),262144).map_err(|e|e.1)?;
    for line in io::stdin().lock().lines(){let body:Value=serde_json::from_str(&line?)?;
        if body["templates"].is_object(){front.install_templates(serde_json::from_value(body["templates"].clone())?).map_err(|e|e.1)?;}
        let value=match front.apply_template_checked(&body["body"],&||Ok(())){Ok(prompt)=>json!({"status":200,"body":{"prompt":prompt}}),Err(error)=>json!({"status":error.0,"body":error.body()})};println!("{}",json!({"result":value,"chat_template":front.chat_templates.snapshot()}));
    }Ok(())
}
