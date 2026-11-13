use std::{fs,path::PathBuf,process::Command,sync::atomic::{AtomicU64,Ordering}};
static NEXT:AtomicU64=AtomicU64::new(0);
struct Fixture(PathBuf);
impl Fixture{fn new()->Self{let path=std::env::temp_dir().join(format!("justfit-help-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));fs::create_dir(&path).unwrap();Self(path)}fn command(&self)->Command{let binary=std::env::var_os("JUSTFIT_TEST_HELP_CLI").unwrap_or_else(||env!("CARGO_BIN_EXE_justfit-rs").into());let mut command=Command::new(binary);command.env("PATH","").env("HOME",self.0.join("must-not-create")).current_dir(&self.0);command}}
impl Drop for Fixture{fn drop(&mut self){fs::remove_dir_all(&self.0).unwrap();}}
#[test]
fn main_and_public_command_help_work_without_workspace_or_python(){
 let fixture=Fixture::new();
 for flag in ["--help","-h"]{
  let output=fixture.command().arg(flag).output().unwrap();assert!(output.status.success(),"{}",String::from_utf8_lossy(&output.stderr));let text=String::from_utf8(output.stdout).unwrap();for command in ["init","verify","doctor","build","serve","model","runtime","ane","replay"]{assert!(text.contains(command));}
  for command in ["init","verify","doctor","build","serve","model","runtime","ane","replay"]{let output=fixture.command().args([command,flag]).output().unwrap();assert!(output.status.success(),"{command}: {}",String::from_utf8_lossy(&output.stderr));let text=String::from_utf8(output.stdout).unwrap();assert!(text.contains(&format!("usage: justfit {command}")));}
  for group in ["model","runtime","ane"]{assert!(fixture.command().args([group,"install",flag]).output().unwrap().status.success());}
 }
 assert!(!fixture.0.join("must-not-create").exists());assert_eq!(fs::read_dir(&fixture.0).unwrap().count(),0);
}
#[test]
fn required_command_and_unknown_command_use_reference_exit_two(){
 let fixture=Fixture::new();for args in [vec![],vec!["--unknown"],vec!["unknown"],vec!["unknown","--help"]]{let output=fixture.command().args(args).output().unwrap();assert_eq!(output.status.code(),Some(2),"{}",String::from_utf8_lossy(&output.stderr));}assert!(!fixture.0.join("must-not-create").exists());
}
#[test]
fn help_lists_current_serve_build_runtime_and_ane_options(){
 let fixture=Fixture::new();let output=fixture.command().args(["serve","--help"]).output().unwrap();assert!(output.status.success());let text=String::from_utf8(output.stdout).unwrap();for name in ["--project","--node","--model","--revision","--port","--jobs","--metal-bin","--runtime-bundle","--runtime-manifest","--runtime-sha256","--ane-bundle","--ane-sha256","--ane-manifest","--ane-manifest-sha256","--vision-profile","--no-ane-vision","--dry-run"]{assert!(text.contains(name),"{name}");}
}
