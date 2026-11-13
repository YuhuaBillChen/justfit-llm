//! Startup behavior, independent of the Python launcher.
use justfit_rust_frontend::Frontend;
use serde_json::json;
use std::{path::PathBuf, sync::atomic::{AtomicU64, Ordering}};
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Fixture(PathBuf);
impl Fixture {
    fn new(source: &str, named: bool) -> Self {
        let base = PathBuf::from(std::env::var_os("JUSTFIT_TEST_TOKENIZER").expect("M4 tokenizer fixture required"));
        let dir = std::env::temp_dir().join(format!("justfit-template-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        std::fs::create_dir(&dir).unwrap();
        std::os::unix::fs::symlink(base.join("tokenizer.json"), dir.join("tokenizer.json")).unwrap();
        let template = if named { json!([{"name":"default","template":source},{"name":"tool_use","template":source.replace("<s>","<tool>")}]) } else {json!(source)};
        std::fs::write(dir.join("tokenizer_config.json"), json!({"chat_template":template}).to_string()).unwrap();
        if !named {std::fs::write(dir.join("chat_template.jinja"), source).unwrap();}
        Self(dir)
    }
    fn front(&self) -> Frontend {Frontend::load(&self.0,"test".into(),8192).unwrap()}
}
impl Drop for Fixture {fn drop(&mut self) {std::fs::remove_dir_all(&self.0).unwrap();}}
const REJECT: &str = "{%- for message in messages -%}{%- if message.role == 'system' -%}{%- if loop.first -%}{{- '<s>' + message.role + ':' + (message.content|trim) + '</s>' -}}{%- else -%}{{- raise_exception('later system') -}}{%- endif -%}{%- else -%}{{- '<s>' + message.role + ':' + (message.content|trim) + '</s>' -}}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}{{- '<s>assistant:' -}}{%- endif -%}";
const SKIP: &str = "{%- for message in messages -%}{%- if message['role'] != 'system' or loop.first -%}{{- '<s>' + message.role + ':' + (message.content|trim) + '</s>' -}}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}{{- '<s>assistant:' -}}{%- endif -%}";
fn body() -> serde_json::Value {json!({"messages":[{"role":"user","content":"one"},{"role":"assistant","content":"two"},{"role":"system","content":"  later  "},{"role":"user","content":"three"}]})}
#[test]
fn rejecting_template_patched_without_python() {
    let fixture=Fixture::new(REJECT,false); let front=fixture.front();
    assert_eq!(front.chat_templates.snapshot(),json!({"later_system":"patched"}));
    assert_eq!(front.render(&body()).unwrap(),"<s>user:one</s><s>assistant:two</s><s>system:later</s><s>user:three</s><s>assistant:");
}
#[test]
fn unsupported_skip_construct_is_preserved() {
    // Released structural matcher only patches AND exclusions; OR is unsupported.
    let fixture=Fixture::new(SKIP,false); let front=fixture.front();
    assert_eq!(front.chat_templates.snapshot(),json!({"later_system":"unsupported"}));
    assert!(front.render(&body()).is_err());
}
#[test]
fn named_default_and_tool_template_loaded_without_jinja_file() {
    let fixture=Fixture::new(REJECT,true); let front=fixture.front();
    let mut request=body();request["tools"]=json!([{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]);
    assert!(front.render(&request).unwrap().starts_with("<tool>user:one"));
    assert_eq!(front.chat_templates.snapshot(),json!({"later_system":{"default":"patched","tool_use":"patched"}}));
}
