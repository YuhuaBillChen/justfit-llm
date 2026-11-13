use justfit_rust_frontend::Frontend;
use serde_json::json;
fn front()->Frontend {
    Frontend::load(std::path::Path::new(&std::env::var("JUSTFIT_TEST_TOKENIZER").unwrap()),"model".into(),4096).unwrap()
}
#[test]
fn pdf_invalid_base64_matches_released_error_before_rendering() {
    let error=front().media_token_count(&json!({"messages":[{"role":"user","content":[{"type":"file","file":{"file_data":"!"}}]}]})).unwrap_err();
    assert_eq!(error.0,400);assert_eq!(error.1,"PDF data is not valid base64");
}
#[test]
fn pdf_file_schema_is_validated_before_rendering() {
    for (file,message) in [(json!({"file_url":"https://example.com/x.pdf"}),"file_id and file_url are not supported; use inline file_data"),(json!({"file_data":"data:image/png;base64,AA=="}),"only application/pdf file data is supported"),(json!({"filename":1,"file_data":"AA=="}),"filename must be a string")] {
        let error=front().media_token_count(&json!({"messages":[{"role":"user","content":[{"type":"file","file":file}]}]})).unwrap_err();
        assert_eq!(error.0,400);assert_eq!(error.1,message);
    }
}
