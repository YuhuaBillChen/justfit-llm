use justfit_rust_frontend::{Frontend,preparation_options::Options};
use serde_json::{json,Value};
use std::{io::{self,BufRead},path::Path};
use sha2::{Sha256,Digest};
fn main() {
    let args:Vec<_>=std::env::args().skip(1).collect();let mut front=Frontend::load(Path::new(&args[0]),"model".into(),262144).unwrap();
    for line in io::stdin().lock().lines() {
        let input:Value=serde_json::from_str(&line.unwrap()).unwrap();
        let result:justfit_rust_frontend::Result<Value>=(||{
            let options:Options=serde_json::from_value(input["options"].clone()).map_err(|_|justfit_rust_frontend::bad("invalid preparation options"))?;
            options.validate()?;front.preparation=options;
            match input["op"].as_str().unwrap() {
                "render"=>Ok(json!({"prompt":front.render(&input["body"])?})),
                "count"=>Ok(json!({"input_tokens":front.media_token_count(&input["body"])?})),
                "image"=>{let bytes=justfit_rust_frontend::media::data_url(&input["url"])?;let image=justfit_rust_frontend::media::prepare(&bytes,front.preparation.max_image_pixels)?;
                    Ok(json!({"grid":[image.h,image.w],"bytes":image.pixels.len(),"sha256":format!("{:x}",Sha256::digest(&image.pixels)),"lo":image.lo,"hi":image.hi}))},
                _=>unreachable!()
            }
        })();
        println!("{}",match result {Ok(v)=>json!({"status":200,"body":v}),Err(e)=>json!({"status":e.0,"body":e.body()})});
    }
}
