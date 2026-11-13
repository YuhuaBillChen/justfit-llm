use justfit_rust_frontend::deadline;
use serde_json::{json,Value};
use std::io::{self,BufRead};
fn main() {
    for line in io::stdin().lock().lines() {
        let value:Value=serde_json::from_str(&line.unwrap()).unwrap();
        let result=(|| {
            let request=deadline::duration(value.get("timeout"))?;
            let server=deadline::duration(value.get("server_timeout"))?;
            let (absolute,remaining)=deadline::wire_fields_at(request.min(server),value["wall_micros"].as_u64().unwrap());
            Ok::<_,justfit_rust_frontend::ApiError>(json!({"status":200,"absolute":absolute,"remaining":remaining}))
        })();
        println!("{}",match result {Ok(v)=>v,Err(e)=>json!({"status":e.0,"message":e.1})});
    }
}
