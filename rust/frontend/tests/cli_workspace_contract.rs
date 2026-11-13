//! Exercise the real Rust CLI with no Python or shell on PATH.
use sha2::{Digest,Sha256};
use std::{path::PathBuf,process::{Command,Output},sync::atomic::{AtomicU64,Ordering}};
static NEXT: AtomicU64=AtomicU64::new(0);
struct Fixture(PathBuf);
impl Fixture {
    fn new()->Self {
        let root=std::env::temp_dir().join(format!("justfit-cli-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
        std::fs::create_dir(&root).unwrap();
        std::fs::create_dir(root.join("source")).unwrap();
        std::fs::write(root.join("source/justfit"),"source bytes\n").unwrap();
        let hash=format!("{:x}",Sha256::digest(b"source bytes\n"));
        std::fs::write(root.join("package-manifest.json"),serde_json::json!({"source/justfit":hash}).to_string()).unwrap();
        Self(root)
    }
    fn command(&self,action:&str)->Command {
        let mut command=Command::new(env!("CARGO_BIN_EXE_justfit-rs"));command.env("PATH","").arg(action);command
    }
    fn verify(&self)->Output {self.command("verify").arg("--project").arg(&self.0).output().unwrap()}
}
impl Drop for Fixture {fn drop(&mut self){std::fs::remove_dir_all(&self.0).unwrap();}}
#[test]
fn verify_and_exclusive_init_work_without_an_interpreter() {
    let fixture=Fixture::new();assert!(fixture.verify().status.success());
    let destination=fixture.0.join("new/workspace");
    let run=||fixture.command("init").arg(&destination).arg("--source").arg(&fixture.0).output().unwrap();
    assert!(run().status.success());
    use std::os::unix::fs::PermissionsExt;
    assert_ne!(std::fs::metadata(destination.join("source/justfit")).unwrap().permissions().mode()&0o111,0);
    std::fs::write(destination.join("owner-data"),"preserve").unwrap();
    assert!(!run().status.success());
    assert_eq!(std::fs::read_to_string(destination.join("owner-data")).unwrap(),"preserve");
}
#[test]
fn changed_source_and_symlink_sources_are_rejected() {
    let fixture=Fixture::new();
    std::fs::write(fixture.0.join("source/justfit"),"changed").unwrap();
    let result=fixture.verify();assert!(!result.status.success());assert!(String::from_utf8_lossy(&result.stderr).contains("source digest mismatch"));
    std::fs::rename(fixture.0.join("source/justfit"),fixture.0.join("owner-file")).unwrap();
    std::os::unix::fs::symlink(fixture.0.join("owner-file"),fixture.0.join("source/justfit")).unwrap();
    let result=fixture.verify();assert!(!result.status.success());assert!(String::from_utf8_lossy(&result.stderr).contains("missing or unsafe source"));
    assert_eq!(std::fs::read_to_string(fixture.0.join("owner-file")).unwrap(),"changed");
}
#[test]
fn escaping_manifest_and_dangling_destination_are_rejected() {
    let fixture=Fixture::new();
    let destination=fixture.0.join("destination");
    std::os::unix::fs::symlink(fixture.0.join("absent"),&destination).unwrap();
    let result=fixture.command("init").arg(&destination).arg("--source").arg(&fixture.0).output().unwrap();
    assert!(!result.status.success());assert!(destination.symlink_metadata().unwrap().file_type().is_symlink());
    std::fs::write(fixture.0.join("package-manifest.json"),"{\"../owner-file\":\"unused\"}").unwrap();
    let result=fixture.verify();assert!(!result.status.success());assert!(String::from_utf8_lossy(&result.stderr).contains("unsafe manifest path"));
}
