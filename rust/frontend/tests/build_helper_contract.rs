use std::{path::PathBuf,process::Command,sync::atomic::{AtomicU64,Ordering}};
static NEXT:AtomicU64=AtomicU64::new(0);
struct Fixture(PathBuf);
impl Fixture {
    fn new()->Self{let root=std::env::temp_dir().join(format!("justfit-build-helper-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));std::fs::create_dir(&root).unwrap();Self(root)}
    fn command(&self)->Command{let mut command=Command::new(env!("CARGO_BIN_EXE_justfit-rs"));command.env("PATH","").current_dir(&self.0).args(["build-helper","dev/tools/build_config.py"]);command}
}
impl Drop for Fixture{fn drop(&mut self){std::fs::remove_dir_all(&self.0).unwrap();}}
#[test]
fn failed_compiler_invalidates_config_and_preserves_exit_code() {
    let fixture=Fixture::new();std::fs::write(fixture.0.join("object.config"),"old\n").unwrap();
    let output=fixture.command().args(["record","--output","object","--config","new","--","/bin/sh","-c","exit 7"]).output().unwrap();
    assert_eq!(output.status.code(),Some(7));assert!(!fixture.0.join("object.config").exists());
}
#[test]
fn successful_compiler_records_config_and_stale_query_matches() {
    let fixture=Fixture::new();
    let output=fixture.command().args(["record","--output","object","--config","current","--","/usr/bin/touch","object"]).output().unwrap();assert!(output.status.success(),"{}",String::from_utf8_lossy(&output.stderr));
    assert_eq!(std::fs::read(fixture.0.join("object.config")).unwrap(),b"current\n");
    let output=fixture.command().args(["stale","--config","current","object","missing"]).output().unwrap();assert!(output.status.success());assert_eq!(output.stdout,b"missing\n");
}
