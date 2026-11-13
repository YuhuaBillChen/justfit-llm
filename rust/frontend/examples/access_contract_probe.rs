use axum::http::HeaderMap;
use justfit_rust_frontend::{
    access::{cors_headers, AccessPolicy},
    ApiError,
};
use serde_json::{json, Value};
use std::io::{self, BufRead};

fn run(value: &Value) -> Result<Value, ApiError> {
    let hosts: Vec<String> = serde_json::from_value(value["hosts"].clone()).unwrap();
    let origins: Vec<String> = serde_json::from_value(value["origins"].clone()).unwrap();
    let policy = AccessPolicy::new("127.0.0.1", &hosts, &origins)?;
    let mut headers = HeaderMap::new();
    for pair in value["headers"].as_array().unwrap() {
        headers.append(
            pair[0]
                .as_str()
                .unwrap()
                .parse::<axum::http::HeaderName>()
                .unwrap(),
            pair[1].as_str().unwrap().parse().unwrap(),
        );
    }
    let origin = policy.allowed_origin(&headers)?;
    let method = value["method"].as_str().unwrap_or("POST");
    let auth = policy.authenticate(&headers, "/v1/messages", method, value["key"].as_str());
    let mut response = HeaderMap::new();
    cors_headers(
        &mut response,
        origin.as_deref(),
        (method == "OPTIONS" && auth.is_ok()).then_some(&headers),
    );
    Ok(
        json!({"status":auth.as_ref().map(|_|200).unwrap_or_else(|e|e.0),
        "origin":origin,"headers":response.iter().map(|(k,v)|(k.as_str(),v.to_str().unwrap())).collect::<std::collections::BTreeMap<_,_>>()}),
    )
}
fn main() {
    for line in io::stdin().lock().lines() {
        let input: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let value = match run(&input) {
            Ok(v) => v,
            Err(e) => json!({"status":e.0,"origin":null,"headers":{}}),
        };
        println!("{value}");
    }
}
