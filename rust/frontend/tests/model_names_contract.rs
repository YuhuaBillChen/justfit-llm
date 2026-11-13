use justfit_rust_frontend::http::{serve,Config};
use serde_json::json;
#[tokio::test]
async fn invalid_aliases_are_rejected_before_native_start() {
    for name in ["", "a b", "/a", "a//b", "a/../b", "a%2fb", "a\u{200b}b"] {
        let config: Config=serde_json::from_value(json!({"tokenizer":"unused","model":"loaded","native":["/should-not-start"],"port":0,"served_model_names":[name]})).unwrap();
        assert_eq!(serve(config).await.unwrap_err().to_string(),"model alias must be a non-empty name without whitespace or URL delimiters");
    }
}
#[tokio::test]
async fn announcing_requires_an_explicit_alias() {
    let config: Config=serde_json::from_value(json!({"tokenizer":"unused","model":"loaded","native":["/should-not-start"],"port":0,"announce_served_name":true})).unwrap();
    assert_eq!(serve(config).await.unwrap_err().to_string(),"announce_served_name needs a served model name");
}
