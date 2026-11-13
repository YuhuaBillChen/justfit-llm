//! Public SDK contains native inputs, never a Python build identity oracle.
use std::{path::{Path,PathBuf},process::{Command,Output},sync::atomic::{AtomicU64,Ordering}};
static NEXT:AtomicU64=AtomicU64::new(0);
const QUALIFIED:&str="src-157bacee619c7ae8d47681b62531885f1bd127015333620322a0bf9eaca6b198";
struct Fixture(PathBuf);
impl Fixture{
 fn new()->Self{
  let path=std::env::temp_dir().join(format!("justfit-native-sdk-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));std::fs::create_dir(&path).unwrap();
  fn copy(from:&Path,to:&Path){std::fs::create_dir(to).unwrap();for entry in std::fs::read_dir(from).unwrap(){let entry=entry.unwrap();assert!(!entry.file_type().unwrap().is_symlink());if entry.file_type().unwrap().is_dir(){copy(&entry.path(),&to.join(entry.file_name()));}else{std::fs::copy(entry.path(),to.join(entry.file_name())).unwrap();}}}
  let root=Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();copy(&root.join("source/runtime"),&path.join("runtime"));Self(path)
 }
 fn command(&self)->Command{let exe=std::env::var_os("JUSTFIT_TEST_SDK_CLI").unwrap_or_else(||env!("CARGO_BIN_EXE_justfit-rs").into());let mut command=Command::new(exe);command.env("PATH","").current_dir(&self.0);command}
 fn identity(&self,extra:&str)->Output{
  let flags="-std=c++20 -O3 -Wall -Wextra -Werror -Iruntime -mmacosx-version-min=26.4";
  self.command().args(["build-helper","build-identity","write","--root",".","--header","identity.hpp","--stamp","identity.json","--constant","runtime=qwen-hybrid-dflash8","--constant",&format!("engine_cxxflags={flags}{extra}"),"--constant",&format!("engine_objcxxflags={flags} -fobjc-arc"),"--constant","engine_linkflags=-framework Foundation -framework Metal -framework IOKit -framework IOSurface","--constant","production_metalflags=-std=metal4.0 -O3 -Wall -Wextra -Werror -Iruntime -mmacosx-version-min=26.4"]).output().unwrap()
 }
 fn id(&self,extra:&str)->String{let output=self.identity(extra);assert!(output.status.success(),"{}",String::from_utf8_lossy(&output.stderr));serde_json::from_slice::<serde_json::Value>(&output.stdout).unwrap()["build_id"].as_str().unwrap().to_string()}
}
impl Drop for Fixture{fn drop(&mut self){std::fs::remove_dir_all(&self.0).unwrap();}}
#[test]
fn stripped_sdk_keeps_only_exact_qualified_native_identity(){
 let fixture=Fixture::new();assert!(!fixture.0.join("dev").exists());assert_eq!(fixture.id(""),QUALIFIED);
 let changed_flags=fixture.id(" -DJUSTFIT_TEST_NEW_CONFIG=1");assert_ne!(changed_flags,QUALIFIED);assert_eq!(changed_flags,fixture.id(" -DJUSTFIT_TEST_NEW_CONFIG=1"));
 let source=fixture.0.join("runtime/main.mm");let original=std::fs::read(&source).unwrap();let mut changed=original.clone();changed.extend_from_slice(b"\n// identity contract mutation\n");std::fs::write(&source,changed).unwrap();let changed_source=fixture.id("");assert_ne!(changed_source,QUALIFIED);assert_ne!(changed_source,changed_flags);
 std::fs::write(source,original).unwrap();assert_eq!(fixture.id(""),QUALIFIED);
}
#[test]
fn native_config_helper_records_and_detects_stale_without_python(){
 let fixture=Fixture::new();let output=fixture.command().args(["build-helper","build-config","record","--output","native.o","--config","native","--","/usr/bin/touch","native.o"]).output().unwrap();assert!(output.status.success(),"{}",String::from_utf8_lossy(&output.stderr));assert_eq!(std::fs::read(fixture.0.join("native.o.config")).unwrap(),b"native\n");
 let output=fixture.command().args(["build-helper","build-config","stale","--config","native","native.o","missing.o"]).output().unwrap();assert!(output.status.success());assert_eq!(output.stdout,b"missing.o\n");
}
#[test]
fn identity_rejects_symlink_oracle_and_native_sources(){
 let fixture=Fixture::new();std::fs::create_dir_all(fixture.0.join("dev/tools")).unwrap();std::fs::write(fixture.0.join("owner"),b"preserve owner").unwrap();std::os::unix::fs::symlink(fixture.0.join("owner"),fixture.0.join("dev/tools/build_identity.py")).unwrap();assert!(!fixture.identity("").status.success());assert_eq!(std::fs::read(fixture.0.join("owner")).unwrap(),b"preserve owner");
 std::fs::remove_file(fixture.0.join("dev/tools/build_identity.py")).unwrap();let source=fixture.0.join("runtime/main.mm");std::fs::rename(&source,fixture.0.join("owner-native")).unwrap();std::os::unix::fs::symlink(fixture.0.join("owner-native"),source).unwrap();assert!(!fixture.identity("").status.success());
}
