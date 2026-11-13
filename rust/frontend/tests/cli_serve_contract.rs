use std::{path::PathBuf,process::Command,sync::atomic::{AtomicU64,Ordering}};
use sha2::{Digest,Sha256};
static NEXT:AtomicU64=AtomicU64::new(0);
struct Fixture(PathBuf);
impl Fixture {
    fn new()->Self {
        let root=std::env::temp_dir().join(format!("justfit-serve-plan-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
        std::fs::create_dir_all(root.join("config/nodes")).unwrap();
        let project=PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf();
        let mut manifest=serde_json::Map::new();
        for node in ["m4","m5","m6"] {
            let name=format!("config/nodes/{node}.json");let data=std::fs::read(project.join(&name)).unwrap();
            manifest.insert(name.clone(),serde_json::json!(format!("{:x}",Sha256::digest(&data))));std::fs::write(root.join(name),data).unwrap();
        }
        std::fs::write(root.join("package-manifest.json"),serde_json::to_vec(&manifest).unwrap()).unwrap();Self(root)
    }
    fn command(&self,node:&str)->Command {
        let mut command=Command::new(env!("CARGO_BIN_EXE_justfit-rs"));
        command.env("PATH","").args(["serve","--project"]).arg(&self.0).args(["--node",node,"--model","/local-model","--state-dir","/local-state","--port","18140","--dry-run"]);command
    }
}
impl Drop for Fixture{fn drop(&mut self){std::fs::remove_dir_all(&self.0).unwrap();}}
#[test]
fn three_presets_plan_in_rust_without_python() {
    let fixture=Fixture::new();
    for node in ["m4","m5","m6"] {
        let result=fixture.command(node).arg("--no-ane-vision").output().unwrap();assert!(result.status.success(),"{}",String::from_utf8_lossy(&result.stderr));
        let plan:serde_json::Value=serde_json::from_slice(&result.stdout).unwrap();
        assert_eq!(plan["environment"]["JUSTFIT_MIXED_DECODE_MAX_LANES"],"2");
        assert_eq!(plan["config"]["native"][5],(64u64*1024*1024*1024).to_string());
        assert_eq!(plan["config"]["port"],18140);
        assert_eq!(plan["config"]["vision"]["command"][1],"media-driver");
        let native=plan["config"]["native"].as_array().unwrap();
        assert_eq!(native.iter().any(|part|part=="--ane"),node!="m6");
        assert!(!fixture.0.join("local-state").exists());
    }
}
#[test]
fn invalid_port_and_conflicting_vision_options_fail_before_launch() {
    let fixture=Fixture::new();
    for args in [vec!["--no-ane-vision","--vision-profile","/missing"],vec!["--unexpected"],vec!["--port","0"]] {
        assert!(!fixture.command("m4").args(args).output().unwrap().status.success());
    }
}

#[test]
fn runtime_setup_requires_one_source_and_validates_pins_before_dry_run() {
    let fixture=Fixture::new();
    for args in [vec!["--runtime-bundle","/missing"],vec!["--runtime-sha256","bad"],vec!["--runtime-bundle","/missing","--runtime-manifest","https://invalid","--runtime-sha256","bad"],vec!["--runtime-bundle","/missing","--runtime-sha256","bad"]] {
        let result=fixture.command("m4").args(args).output().unwrap();
        assert!(!result.status.success());
        let error=String::from_utf8_lossy(&result.stderr);
        assert!(error.contains("choose one")||error.contains("SHA256"),"{error}");
        assert!(!fixture.0.join("build-receipt.json").exists());
        assert!(!fixture.0.join(".runtime-install.lock").exists());
    }
}

#[test]
fn dry_run_with_pinned_runtime_plans_without_installing() {
    let fixture=Fixture::new();
    for source in ["--runtime-bundle","--runtime-manifest"] {
        let result=fixture.command("m4").args([source,"/not-downloaded","--runtime-sha256",&"a".repeat(64)]).output().unwrap();
        assert!(result.status.success(),"{}",String::from_utf8_lossy(&result.stderr));
        let plan:serde_json::Value=serde_json::from_slice(&result.stdout).unwrap();assert_eq!(plan["config"]["port"],18140);
        assert!(!fixture.0.join("build-receipt.json").exists());assert!(!fixture.0.join(".runtime-install.lock").exists());
    }
}

#[test]
fn native_chip_detection_does_not_require_path() {
    let fixture=Fixture::new();
    let result=Command::new(env!("CARGO_BIN_EXE_justfit-rs")).env("PATH","").args(["serve","--project"]).arg(&fixture.0).args(["--model","/local-model","--dry-run","--no-ane-vision"]).output().unwrap();
    assert!(result.status.success(),"{}",String::from_utf8_lossy(&result.stderr));
    let plan:serde_json::Value=serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(plan["environment"]["JUSTFIT_MIXED_DECODE_MAX_LANES"],"2");
    assert!(plan["config"]["native"].as_array().unwrap().iter().any(|v|v=="--ane"));
}
#[test]
fn invalid_options_use_reference_public_cli_exit_code() {
    let fixture=Fixture::new();
    for args in [vec!["--unexpected"],vec!["--port","0"],vec!["--runtime-sha256","bad"]] {
        let result=fixture.command("m4").args(args).output().unwrap();assert_eq!(result.status.code(),Some(2));
    }
}
#[test]
fn default_workspace_dry_run_does_not_initialize_or_require_python() {
    let fixture=Fixture::new();let home=fixture.0.join("new-home");std::fs::create_dir(&home).unwrap();
    let result=Command::new(env!("CARGO_BIN_EXE_justfit-rs")).env("PATH","").env("HOME",&home).args(["serve","--node","m4","--model","/local-model","--dry-run"]).output().unwrap();
    assert_eq!(result.status.code(),Some(2));assert!(String::from_utf8_lossy(&result.stderr).contains("run without --dry-run to create it"));assert!(!home.join("Library").exists());
}

#[test]
fn presets_are_language_independent_and_pdf_uses_system_zlib() {
    let fixture=Fixture::new();
    for node in ["m4","m5","m6"] {
        let path=fixture.0.join(format!("config/nodes/{node}.json"));
        let preset:serde_json::Value=serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(preset["schema"],2);assert!(preset.get("command").is_none());
        let text=serde_json::to_string(&preset).unwrap();assert!(!text.contains("python"));assert!(!text.contains("server.server"));
    }
}

#[test]
fn default_model_and_revision_plan_without_model_installation() {
    let fixture=Fixture::new();let binary=std::env::var_os("JUSTFIT_TEST_SERVE_CLI").unwrap_or_else(||env!("CARGO_BIN_EXE_justfit-rs").into());
    for (model,revision) in [(None,None),(Some("owner/toy-model"),Some("release/stable")),(Some("/local-model"),Some("a"))] {
        let mut command=Command::new(&binary);command.env("PATH","").env("HOME",fixture.0.join("no-home")).env("HF_HUB_CACHE",fixture.0.join("empty-cache")).env("HF_HUB_OFFLINE","1")
            .args(["serve","--project"]).arg(&fixture.0).args(["--node","m4","--dry-run","--no-ane-vision"]);
        if let Some(model)=model{command.args(["--model",model]);}
        if let Some(revision)=revision{command.args(["--revision",revision]);}
        let result=command.output().unwrap();assert!(result.status.success(),"{}",String::from_utf8_lossy(&result.stderr));
        let value:serde_json::Value=serde_json::from_slice(&result.stdout).unwrap();let expected=std::path::absolute(model.unwrap_or("mlx-community/Qwen3.8-27B-4bit")).unwrap();
        assert_eq!(value["config"]["native"][2],expected.to_str().unwrap());assert_eq!(value["config"]["model"],"mlx-community/Qwen3.8-27B-4bit");assert!(value["config"]["served_model_names"].as_array().unwrap().iter().any(|v|v=="incoai/Qwen3.8-27B-Splash"));assert!(!fixture.0.join("local-state").exists());assert!(!fixture.0.join("empty-cache").exists());assert!(!fixture.0.join("no-home").exists());
    }
}
#[test]
fn unsafe_revision_rejected_before_creating_workspace() {
    let fixture=Fixture::new();let project=fixture.0.join("missing-project");
    let result=Command::new(env!("CARGO_BIN_EXE_justfit-rs")).env("PATH","").args(["serve","--project"]).arg(&project).args(["--node","m4","--revision","../owner"]).output().unwrap();
    assert_eq!(result.status.code(),Some(2));assert!(String::from_utf8_lossy(&result.stderr).contains("invalid model revision"));assert!(!project.exists());
}

#[test]
fn ane_package_options_plan_without_download_or_profile_writes() {
    let fixture=Fixture::new();let pin="a".repeat(64);let binary=std::env::var_os("JUSTFIT_TEST_SERVE_CLI").unwrap_or_else(||env!("CARGO_BIN_EXE_justfit-rs").into());
    for flags in [vec!["--ane-bundle","/not-downloaded-4k.zip","--ane-sha256",&pin,"--ane-bundle","/not-downloaded-8k.zip","--ane-sha256",&pin],vec!["--ane-manifest","https://not-downloaded/index.json","--ane-manifest-sha256",&pin]] {
        let result=Command::new(&binary).env("PATH","").args(["serve","--project"]).arg(&fixture.0).args(["--node","m4","--model","/local-model","--dry-run"]).args(flags).output().unwrap();
        assert!(result.status.success(),"{}",String::from_utf8_lossy(&result.stderr));let plan:serde_json::Value=serde_json::from_slice(&result.stdout).unwrap();assert!(plan["config"]["vision"]["profile"].is_null());assert!(!fixture.0.join("local-state").exists());
    }
}
#[test]
fn ane_pins_and_conflicts_are_checked_before_workspace_creation() {
    let fixture=Fixture::new();let root=fixture.0.join("missing-workspace");let pin="a".repeat(64);
    for flags in [vec!["--ane-bundle","/missing"],vec!["--ane-sha256",&pin],vec!["--ane-manifest","https://invalid"],vec!["--ane-bundle","/missing","--ane-sha256","bad"],vec!["--no-ane-vision","--ane-bundle","/missing","--ane-sha256",&pin],vec!["--vision-profile","/missing","--ane-bundle","/missing","--ane-sha256",&pin],vec!["--ane-manifest","https://invalid","--ane-manifest-sha256",&pin,"--ane-bundle","/missing","--ane-sha256",&pin]] {
        let result=Command::new(env!("CARGO_BIN_EXE_justfit-rs")).env("PATH","").args(["serve","--project"]).arg(&root).args(["--node","m4"]).args(flags).output().unwrap();assert_eq!(result.status.code(),Some(2));assert!(!root.exists());
    }
}
