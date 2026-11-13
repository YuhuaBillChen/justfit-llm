use justfit_rust_frontend::anthropic;
use serde_json::{json, Value};

fn convert(content: Value) -> justfit_rust_frontend::Result<Value> {
    anthropic::convert(
        &json!({"model":"model","max_tokens":64,"messages":[{"role":"user","content":content}]}),
        false,
        &mut |_| panic!("documents must not resolve thinking signatures"),
    )
}

#[test]
fn document_preserves_title_context_and_surrounding_text() {
    let converted = convert(json!([
        {"type":"text","text":"before"},
        {"type":"document","title":"标题","context":"context","source":{"type":"base64","media_type":"application/pdf","data":"JVBERi0="}},
        {"type":"text","text":"after"}
    ])).unwrap();
    assert_eq!(
        converted["chat"]["messages"][0]["content"],
        json!([
            {"type":"text","text":"before"},
            {"type":"text","text":"标题\n"},
            {"type":"text","text":"context\n"},
            {"type":"file","file":{"file_data":"data:application/pdf;base64,JVBERi0="}},
            {"type":"text","text":"after"}
        ])
    );
}

#[test]
fn tool_result_document_keeps_error_prefix_and_call_id() {
    let converted=convert(json!([{"type":"tool_result","tool_use_id":"call_one","is_error":true,"content":[
        {"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"JVBERi0="}}
    ]}])).unwrap();
    assert_eq!(
        converted["chat"]["messages"][0],
        json!({"role":"tool","tool_call_id":"call_one","content":[
            {"type":"text","text":"Tool execution failed:\n"},
            {"type":"file","file":{"file_data":"data:application/pdf;base64,JVBERi0="}}
        ]})
    );
}

#[test]
fn invalid_documents_keep_python_validation_errors() {
    let base = json!({"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"JVBERi0="}});
    for (field, value, message) in [
        (
            "source",
            json!({"type":"url","url":"https://example.com/a.pdf"}),
            "documents require a base64 application/pdf source",
        ),
        (
            "source",
            json!({"type":"base64","media_type":"application/pdf","data":null}),
            "PDF data must be a base64 string",
        ),
        ("title", json!(false), "document title must be a string"),
        ("context", json!(7), "document context must be a string"),
        (
            "citations",
            json!({"enabled":true}),
            "document citations are not supported",
        ),
        (
            "citations",
            json!({"enabled":0}),
            "document citations are not supported",
        ),
        (
            "citations",
            json!([]),
            "document citations are not supported",
        ),
    ] {
        let mut block = base.clone();
        block[field] = value;
        let error = convert(json!([block])).unwrap_err();
        assert_eq!(error.0, 400);
        assert_eq!(error.1, message);
    }
    for citations in [Value::Null, json!({}), json!({"enabled":false})] {
        let mut block = base.clone();
        block["citations"] = citations;
        assert!(convert(json!([block])).is_ok());
    }
}
