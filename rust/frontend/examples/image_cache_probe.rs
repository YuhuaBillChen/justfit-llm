use justfit_rust_frontend::image_cache::Cache;
use serde_json::{json,Value};
use base64::{Engine as _,engine::general_purpose::STANDARD};
use sha2::{Digest,Sha256};
use std::io::{self,BufRead};
fn main()->Result<(),Box<dyn std::error::Error>>{
    let cache=Cache::default();for line in io::stdin().lock().lines(){let body:Value=serde_json::from_str(&line?)?;let raw=STANDARD.decode(body["image"].as_str().unwrap())?;
        let result=cache.prepare(&raw,body["max_pixels"].as_u64().unwrap()as usize);let value=match result{Ok(image)=>json!({"status":200,"grid":[image.h,image.w],"digest":[image.lo,image.hi],"pixel_sha256":format!("{:x}",Sha256::digest(&image.pixels)),"stats":cache.snapshot(0)}),Err(error)=>json!({"status":error.0,"error":error.body(),"stats":cache.snapshot(0)})};println!("{value}");
    }Ok(())
}
