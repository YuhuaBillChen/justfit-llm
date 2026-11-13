use std::io::{self,BufRead};
fn main(){for line in io::stdin().lock().lines(){let value:serde_json::Value=serde_json::from_str(&line.unwrap()).unwrap();println!("{}",justfit_rust_frontend::python_json::compact_size(&value).unwrap())}}
