use std::process::Command;
#[test]
fn public_model_prepare_help_exposes_native_source_workflow(){
 let binary=std::env::var_os("JUSTFIT_TEST_SOURCE_CLI").unwrap_or_else(||env!("CARGO_BIN_EXE_justfit-rs").into());
 let output=Command::new(binary).args(["model","prepare","--help"]).env("PATH","").output().unwrap();
 assert!(output.status.success(),"{}",String::from_utf8_lossy(&output.stderr));let text=String::from_utf8(output.stdout).unwrap();for flag in ["--target","--draft-model","--language-only"]{assert!(text.contains(flag));}
}
