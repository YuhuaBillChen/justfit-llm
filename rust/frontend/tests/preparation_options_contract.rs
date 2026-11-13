use justfit_rust_frontend::http::{serve,Config};
use serde_json::json;
#[tokio::test]
async fn invalid_preparation_options_fail_before_native_start() {
    for (field,value,message) in [("default_reasoning_effort",json!("invalid"),"invalid default_reasoning_effort"),("max_image_pixels",json!(0),"max_image_pixels must be between 65536 and 4194304 pixels"),("max_image_pixels",json!(4194305),"max_image_pixels must be between 65536 and 4194304 pixels")] {
        let mut config=json!({"tokenizer":"unused","model":"loaded","native":["/should-not-start"],"port":0});config[field]=value;
        assert_eq!(serve(serde_json::from_value::<Config>(config).unwrap()).await.unwrap_err().to_string(),message);
    }
}
#[tokio::test]
async fn request_body_limit_above_128m_is_not_rejected_by_development_cap() {
    let config=serde_json::from_value::<Config>(json!({"tokenizer":"unused","model":"loaded","native":["/should-not-start"],"port":0,"max_request_bytes":268435456})).unwrap();
    let error=serve(config).await.unwrap_err().to_string();
    assert!(!error.contains("128MiB"),"{error}");
    assert!(error.contains("No such file"),"{error}");
}
