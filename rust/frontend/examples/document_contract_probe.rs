use justfit_rust_frontend::anthropic::document_parts;
use serde_json::{json, Value};
use std::io::{self, BufRead};

fn main() {
    for line in io::stdin().lock().lines() {
        let block: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let result = match document_parts(&block) {
            Ok(parts) => json!({"status":200,"parts":parts}),
            Err(error) => json!({"status":error.0,"message":error.1,"code":error.2}),
        };
        println!("{result}");
    }
}
