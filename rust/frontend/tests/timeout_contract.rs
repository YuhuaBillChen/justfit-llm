use justfit_rust_frontend::Frontend;
use serde_json::json;
use std::{path::PathBuf, sync::atomic::{AtomicU64, Ordering}, time::Duration};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64=AtomicU64::new(0);
        let p=std::env::temp_dir().join(format!("justfit-timeout-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
        std::fs::create_dir(&p).unwrap();
        std::fs::write(p.join("tokenizer.json"),r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1},"unk_token":"[UNK]"}}"#).unwrap();
        std::fs::write(p.join("chat_template.jinja"),"{% for message in messages %}{{ message.content }}{% endfor %}").unwrap();
        Self(p)
    }
    fn front(&self)->Frontend { Frontend::load(&self.0,"model".into(),1024).unwrap() }
}
impl Drop for Fixture { fn drop(&mut self){ let _=std::fs::remove_dir_all(&self.0); } }

#[test]
fn omitted_and_null_request_deadlines_are_unlimited_like_23() {
    let fixture=Fixture::new();let front=fixture.front();
    for body in [json!({"prompt":"hello","seed":0}),json!({"prompt":"hello","timeout":null,"seed":0})] {
        assert_eq!(front.generation(&body,1,true).unwrap().timeout,Duration::MAX);
    }
}

#[test]
fn explicit_deadlines_above_two_hours_are_not_silently_clamped() {
    let fixture=Fixture::new();let front=fixture.front();
    for seconds in [7201.,9000.,86400.] {
        let result=front.generation(&json!({"prompt":"hello","timeout":seconds,"seed":0}),1,true).unwrap();
        assert_eq!(result.timeout,Duration::from_secs_f64(seconds));
    }
}

#[test]
fn very_large_finite_deadline_does_not_panic() {
    let fixture=Fixture::new();let front=fixture.front();
    assert_eq!(front.generation(&json!({"prompt":"hello","timeout":1e300,"seed":0}),1,true).unwrap().timeout,Duration::MAX);
}

#[test]
fn invalid_deadlines_remain_rejected() {
    let fixture=Fixture::new();let front=fixture.front();
    for value in [json!(false),json!(true),json!(0),json!(-1),json!("30"),json!([]),json!({})] {
        let error=match front.generation(&json!({"prompt":"hello","timeout":value,"seed":0}),1,true) {
            Ok(_)=>panic!("invalid timeout accepted"),Err(error)=>error,
        };
        assert_eq!(error.0,400);assert_eq!(error.1,"timeout must be positive");
    }
}
