use justfit_rust_frontend::models;
use serde_json::{json,Value};
use std::io::{self,BufRead,Write};
fn main() {
    if std::env::args().nth(1).as_deref() == Some("--unicode-bitmap") {
        let mut bits = vec![0u8; 0x110000 / 8];
        for point in 0..0x110000 {
            if char::from_u32(point).is_some_and(models::allowed_character) {
                bits[point as usize / 8] |= 1 << (point % 8);
            }
        }
        io::stdout().write_all(&bits).unwrap();
        return;
    }
    for line in io::stdin().lock().lines() {
        let v:Value=serde_json::from_str(&line.unwrap()).unwrap();
        let aliases:Vec<String>=serde_json::from_value(v["aliases"].clone()).unwrap();
        let output=match models::names(v["loaded"].as_str().unwrap(),&aliases,v["announce"].as_bool().unwrap()) {
            Ok((response,names))=>json!({"response_model":response,"model_names":names}),
            Err(e)=>json!({"error":e.1}),
        };
        println!("{output}");
    }
}
