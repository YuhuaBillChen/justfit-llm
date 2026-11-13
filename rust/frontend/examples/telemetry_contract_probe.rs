use serde_json::{json,Value};
use std::io::{self,BufRead};
fn main(){
    for line in io::stdin().lock().lines(){
        let input:Value=serde_json::from_str(&line.unwrap()).unwrap();
        let output=if input["op"]=="latency"{
            let metrics=justfit_rust_frontend::latency::Metrics::default();
            for row in input["observations"].as_array().unwrap(){if let(Some(stage),Some(seconds))=(row[0].as_str(),row[1].as_f64()){metrics.observe(stage,seconds)}}
            metrics.snapshot()
        }else{json!({"text":justfit_rust_frontend::telemetry::prometheus(&input["status"])})};
        println!("{output}");
    }
}
