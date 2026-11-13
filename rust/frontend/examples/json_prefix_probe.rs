use std::io::{self,BufRead};
fn main(){for line in io::stdin().lock().lines(){let text:String=serde_json::from_str(&line.unwrap()).unwrap();println!("{}",justfit_rust_frontend::tool_history::unfinished(&text));}}
