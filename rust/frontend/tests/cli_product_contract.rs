use std::{path::PathBuf,process::Command,sync::atomic::{AtomicU64,Ordering}};
use sha2::{Digest,Sha256};
static NEXT:AtomicU64=AtomicU64::new(0);
struct Fixture(PathBuf);
impl Fixture{
 fn new()->Self{let path=std::env::temp_dir().join(format!("justfit-product-cli-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));std::fs::create_dir(&path).unwrap();Self(path)}
 fn project(&self)->PathBuf{let root=self.0.join("project");std::fs::create_dir_all(root.join("config/nodes")).unwrap();let source=PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf();let name="config/nodes/m4.json";let bytes=std::fs::read(source.join(name)).unwrap();std::fs::write(root.join(name),&bytes).unwrap();std::fs::write(root.join("package-manifest.json"),serde_json::to_vec(&serde_json::json!({(name):format!("{:x}",Sha256::digest(bytes))})).unwrap()).unwrap();root}
}
impl Drop for Fixture{fn drop(&mut self){std::fs::remove_dir_all(&self.0).unwrap();}}
fn cli()->Command{let binary=std::env::var_os("JUSTFIT_TEST_PRODUCT_CLI").unwrap_or_else(||env!("CARGO_BIN_EXE_justfit-rs").into());let mut command=Command::new(binary);command.env("PATH","");command}
#[test]
fn doctor_is_native_and_checks_optional_metal_without_python(){
 let fixture=Fixture::new();let tools=fixture.0.join("Metal tools");std::fs::create_dir(&tools).unwrap();for name in ["metal","metallib"]{std::fs::write(tools.join(name),b"existence fixture, never executed").unwrap();}
 let output=cli().arg("doctor").output().unwrap();assert!(output.status.success(),"{}",String::from_utf8_lossy(&output.stderr));let value:serde_json::Value=serde_json::from_slice(&output.stdout).unwrap();assert_eq!(value["native_platform_supported"],true);assert_eq!(value["dependencies_ok"],true);assert!(value["metal_tools"].as_object().unwrap().is_empty());
 let output=cli().args(["doctor","--metal-bin"]).arg(&tools).output().unwrap();assert!(output.status.success());let value:serde_json::Value=serde_json::from_slice(&output.stdout).unwrap();assert_eq!(value["metal_tools"]["metal"],true);assert_eq!(value["metal_tools"]["metallib"],true);
 std::fs::remove_file(tools.join("metallib")).unwrap();let output=cli().args(["doctor","--metal-bin"]).arg(&tools).output().unwrap();assert_eq!(output.status.code(),Some(1));let value:serde_json::Value=serde_json::from_slice(&output.stdout).unwrap();assert_eq!(value["metal_tools"]["metallib"],false);
 assert_eq!(cli().args(["doctor","--unknown"]).output().unwrap().status.code(),Some(2));
}
#[test]
fn serve_accepts_build_options_without_dry_run_side_effects(){
 let fixture=Fixture::new();let root=fixture.project();let output=cli().args(["serve","--project"]).arg(&root).args(["--node","m4","--model","/unavailable-model","--jobs","2","--metal-bin","/not-needed-for-dry-run","--dry-run","--no-ane-vision"]).output().unwrap();assert!(output.status.success(),"{}",String::from_utf8_lossy(&output.stderr));assert!(!root.join("build-receipt.json").exists());assert!(!root.join("local-state").exists());
 for jobs in ["0","-1","bad"]{let absent=fixture.0.join("must-not-create");let output=cli().args(["serve","--project"]).arg(&absent).args(["--node","m4","--jobs",jobs]).output().unwrap();assert_eq!(output.status.code(),Some(2));assert!(!absent.exists());}
}
#[test]
fn missing_runtime_enters_rust_build_before_model_installation(){
 let fixture=Fixture::new();let root=fixture.project();let absent=fixture.0.join("missing-metal-tools");for stale in [false,true]{if stale{std::fs::write(root.join("build-receipt.json"),b"{broken old receipt").unwrap();}let output=cli().args(["serve","--project"]).arg(&root).args(["--node","m4","--model","owner/not-downloaded","--jobs","2","--metal-bin"]).arg(&absent).output().unwrap();assert_eq!(output.status.code(),Some(2));assert!(String::from_utf8_lossy(&output.stderr).contains("Metal toolchain directory unavailable"),"{}",String::from_utf8_lossy(&output.stderr));assert!(!root.join("local-state/models").exists());}
}
#[test]
fn build_defaults_to_reference_user_workspace(){
 let fixture=Fixture::new();let root=fixture.0.join("Library/Application Support/JustFit").join("2.5.0.dev1");std::fs::create_dir_all(&root).unwrap();std::fs::write(root.join("guard"),b"preserve owner source").unwrap();std::fs::write(root.join("package-manifest.json"),serde_json::to_vec(&serde_json::json!({"guard":"0".repeat(64)})).unwrap()).unwrap();
 let output=cli().env("HOME",&fixture.0).args(["build","--metal-bin","/not-reached"]).output().unwrap();assert_eq!(output.status.code(),Some(2));assert!(String::from_utf8_lossy(&output.stderr).contains("source digest mismatch: guard"),"{}",String::from_utf8_lossy(&output.stderr));assert_eq!(std::fs::read(root.join("guard")).unwrap(),b"preserve owner source");
}
