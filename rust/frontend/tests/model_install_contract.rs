//! Packed manifest/filesystem contracts with toy weights; no inference claims.
use justfit_rust_frontend::{download::Stage,model_install::{manifest,verify}};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{fs,path::{Path,PathBuf},process::Command};
fn fixture(moe:bool)->Value {
    let (schema,target,draft,magic)=if moe{(4,40,6,"MDFM0001")}else{(3,64,5,"MDFL0006")};
    let bytes=vec![42u8;16384];let sha=format!("{:x}",Sha256::digest(&bytes));
    let mut names=vec!["target/embedding.bin".into(),"target/head.bin".into(),"draft/model.bin".into(),"vision/model.bin".into()];
    names.extend((0..target).map(|n|format!("target/layer-{n}.bin")));names.extend((0..draft).map(|n|format!("draft/layer-{n}.bin")));
    names.extend(["chat_template.jinja","config.json","tokenizer.json","tokenizer_config.json","vocab.json"].map(|n|format!("tokenizer/{n}")));
    json!({"schema_version":schema,"model":"toy weights: never executed","execution_geometry":{},"format":{"name":if moe{"splash-packed-q4-moe"}else{"splash-packed-q4"},"section_alignment_bytes":16384,"target_layer_magic":magic,"draft_layer_magic":"MDFD0004","vision_magic":"MDFV0001"},"target":{"architecture":"qwen3_5_moe"},"draft":{"architecture":"DFlash2DraftModel"},"artifacts":names.into_iter().map(|path:String|json!({"path":path,"size":16384,"sha256":sha})).collect::<Vec<_>>()})
}
fn write(root:&Path,value:&Value){fs::write(root.join("manifest.json"),serde_json::to_vec(value).unwrap()).unwrap();}
#[test]
fn dense_and_moe_records_match_supported_legacy_layouts() {
    for (moe,count) in [(false,78),(true,55)]{assert_eq!(manifest(&fixture(moe)).unwrap().len(),count);}
    let mut value=fixture(false);value["artifacts"][0]["sha256"]=json!("ABCDEF".repeat(10)+"ABCD");assert!(manifest(&value).is_ok());
}
#[test]
fn rejects_manifest_paths_sizes_duplicates_overlap_and_incomplete_formats() {
    for path in ["","../owner","/owner","target//owner","target/./owner","target/../owner","target\\owner","target/*","target/[owner]","target/owner\n","manifest.json"] {
        let mut value=fixture(false);value["artifacts"][0]["path"]=json!(path);assert!(manifest(&value).is_err(),"{path:?}");
    }
    for size in [json!(0),json!(-1),json!(true),json!(1.0),json!(16385)]{let mut v=fixture(false);v["artifacts"][0]["size"]=size;assert!(manifest(&v).is_err());}
    for change in 0..8 {
        let mut v=fixture(false);
        match change {
            0=>{v["artifacts"][0]["extra"]=json!(1);},
            1=>{let item=v["artifacts"][0].clone();v["artifacts"].as_array_mut().unwrap().push(item);},
            2=>{v["artifacts"].as_array_mut().unwrap().push(json!({"path":"target","size":16384,"sha256":"a".repeat(64)}));},
            3=>{v["artifacts"].as_array_mut().unwrap().pop();},
            4=>{v["format"]["section_alignment_bytes"]=json!(true);},
            5=>{v["format"]["vision_magic"]=json!("bad");},
            6=>{v["schema_version"]=json!(3.0);},
            _=>{v["model"]=json!("  ");},
        }
        assert!(manifest(&v).is_err(),"change {change}");
    }
}
#[test]
fn full_verification_is_read_only_and_works_through_hub_snapshot_symlinks() {
    let stage=Stage::new(&std::env::temp_dir(),"justfit-model-verify-test").unwrap();let root=stage.path();let value=fixture(false);write(root,&value);
    for record in manifest(&value).unwrap(){let path=root.join(record.path);fs::create_dir_all(path.parent().unwrap()).unwrap();fs::write(path,vec![42u8;16384]).unwrap();}
    assert_eq!(verify(root,true).unwrap(),78);
    let path=root.join("target/head.bin");let external=root.join("owner-weights");fs::rename(&path,&external).unwrap();std::os::unix::fs::symlink(&external,&path).unwrap();assert_eq!(verify(root,true).unwrap(),78);
    fs::write(&external,vec![43u8;16384]).unwrap();assert!(verify(root,true).is_err());assert!(verify(root,false).is_ok());assert_eq!(fs::read(external).unwrap(),vec![43u8;16384]);
}
#[test]
fn public_cli_invalid_model_and_offline_miss_do_not_create_project_data() {
    let stage=Stage::new(&std::env::temp_dir(),"justfit-model-cli-test").unwrap();let project=stage.path().join("owner-project");
    for model in ["../missing-owner-data","not-a-repo","owner/../repo","owner/bad--repo","owner/valid-repo"] {
        let result=Command::new(env!("CARGO_BIN_EXE_justfit-rs")).env("PATH","").env("HOME",stage.path()).env("HF_HUB_CACHE",stage.path().join("empty-hub")).env("HF_HUB_OFFLINE","1").args(["model","install","--project"]).arg(&project).args(["--model",model,"--revision",&"a".repeat(40)]).output().unwrap();
        assert_eq!(result.status.code(),Some(2),"{model}: {}",String::from_utf8_lossy(&result.stderr));assert!(!project.exists());
    }
}
#[test]
fn default_source_rejects_invalid_revision_before_runtime_bootstrap() {
    let stage=Stage::new(&std::env::temp_dir(),"justfit-source-bootstrap-invalid").unwrap();
    let project=stage.path().join("new-project");
    let result=Command::new(env!("CARGO_BIN_EXE_justfit-rs")).env("PATH","").env("HOME",stage.path())
        .args(["model","install","--project"]).arg(&project)
        .args(["--revision","../owner-reference"]).output().unwrap();
    assert_eq!(result.status.code(),Some(2));
    assert!(String::from_utf8_lossy(&result.stderr).contains("invalid model revision"));
    assert!(!project.exists());
}
#[test]
#[ignore = "requires the M4 qualified real model; full hashing reads all weights"]
fn actual_qualified_model_is_verified_by_the_rust_cli_without_python() {
    let model=PathBuf::from(std::env::var_os("JUSTFIT_TEST_PACKED_MODEL").expect("real M4 model"));
    let result=Command::new(env!("CARGO_BIN_EXE_justfit-rs")).env("PATH","").args(["model","verify"]).arg(model).output().unwrap();
    assert!(result.status.success(),"{}",String::from_utf8_lossy(&result.stderr));let result:Value=serde_json::from_slice(&result.stdout).unwrap();assert_eq!(result["files"],78);
}

fn install_offline(root:&Path,project:&Path,hub:&Path,revision:&str)->std::process::Output {
    Command::new(std::env::var_os("JUSTFIT_TEST_MODEL_CLI").unwrap_or_else(||env!("CARGO_BIN_EXE_justfit-rs").into()))
        .env("PATH","").env("HOME",root).env("HF_HUB_CACHE",hub).env("HF_HUB_OFFLINE","1")
        .args(["model","install","--project"]).arg(project).args(["--model","owner/toy-model","--revision",revision]).output().unwrap()
}
fn weights(root:&Path) {
    fs::create_dir_all(root).unwrap();let value=fixture(false);write(root,&value);
    for record in manifest(&value).unwrap(){let path=root.join(record.path);fs::create_dir_all(path.parent().unwrap()).unwrap();fs::write(path,vec![42u8;16384]).unwrap();}
}
#[test]
fn offline_reuses_project_snapshots_and_hub_branch_refs_without_writes() {
    let stage=Stage::new(&std::env::temp_dir(),"justfit-model-offline-test").unwrap();let root=stage.path();
    let commit="a".repeat(40);let project=root.join("project");let hub=root.join("hub");
    for owned in [true,false] {
        let repo=if owned{project.join("local-state/models/models--owner--toy-model")}else{hub.join("models--owner--toy-model")};
        let snapshot=repo.join("snapshots").join(&commit);weights(&snapshot);
        fs::create_dir_all(repo.join("refs/release")).unwrap();fs::write(repo.join("refs/release/stable"),format!("{commit}\n")).unwrap();
        let owner=snapshot.join("owner-marker");fs::write(&owner,b"preserve owner data").unwrap();
        for revision in [commit.as_str(),"release/stable"] {
            let result=install_offline(root,&project,&hub,revision);
            assert!(result.status.success(),"owned={owned}, revision={revision}: {}",String::from_utf8_lossy(&result.stderr));
            let value:Value=serde_json::from_slice(&result.stdout).unwrap();assert_eq!(value["model"],snapshot.canonicalize().unwrap().to_str().unwrap());
        }
        assert_eq!(fs::read(owner).unwrap(),b"preserve owner data");
        // Remove only this test's fixture to exercise the other cache next.
        fs::remove_dir_all(repo).unwrap();
    }
}
#[test]
fn offline_rejects_bad_refs_and_corrupt_snapshots_without_replacing_owner_files() {
    let stage=Stage::new(&std::env::temp_dir(),"justfit-model-offline-negative").unwrap();let root=stage.path();
    let project=root.join("project");let hub=root.join("hub");let repo=project.join("local-state/models/models--owner--toy-model");
    let commit="b".repeat(40);let snapshot=repo.join("snapshots").join(&commit);weights(&snapshot);
    fs::create_dir_all(repo.join("refs")).unwrap();let reference=repo.join("refs/main");fs::write(&reference,"../../owner").unwrap();
    let result=install_offline(root,&project,&hub,"main");assert_eq!(result.status.code(),Some(2));assert_eq!(fs::read(&reference).unwrap(),b"../../owner");
    let artifact=snapshot.join("target/head.bin");fs::write(&artifact,vec![43u8;16384]).unwrap();
    let result=install_offline(root,&project,&hub,&commit);assert_eq!(result.status.code(),Some(2));assert!(String::from_utf8_lossy(&result.stderr).contains("checksum"));assert_eq!(fs::read(artifact).unwrap(),vec![43u8;16384]);
    for revision in ["../owner","/owner","release//stable"] {let result=install_offline(root,&project,&hub,revision);assert_eq!(result.status.code(),Some(2));}
}

#[test]
fn compatibility_python_whitespace_and_hidden_bin_filename() {
    let stage=Stage::new(&std::env::temp_dir(),"justfit-model-metadata-compatibility").unwrap();let root=stage.path();weights(root);
    let binary=std::env::var_os("JUSTFIT_TEST_MODEL_CLI").unwrap_or_else(||env!("CARGO_BIN_EXE_justfit-rs").into());
    for code in 0x1c..=0x1f {
        let mut value=fixture(false);value["model"]=json!(char::from_u32(code).unwrap().to_string());write(root,&value);
        let result=Command::new(&binary).env("PATH","").args(["model","verify"]).arg(root).output().unwrap();
        assert_eq!(result.status.code(),Some(2),"Python strip treats U+{code:04X} as blank");
    }
    let mut value=fixture(false);let bytes=b"x";let sha=format!("{:x}",Sha256::digest(bytes));
    value["artifacts"].as_array_mut().unwrap().push(json!({"path":"tokenizer/.bin","size":1,"sha256":sha}));write(root,&value);fs::write(root.join("tokenizer/.bin"),bytes).unwrap();
    let result=Command::new(&binary).env("PATH","").args(["model","verify"]).arg(root).output().unwrap();
    assert!(result.status.success(),"Python pathlib does not give hidden .bin an extension: {}",String::from_utf8_lossy(&result.stderr));
    let value:Value=serde_json::from_slice(&result.stdout).unwrap();assert_eq!(value["files"],79);
}
#[test]
fn compatibility_xdg_cache_does_not_require_home() {
    let stage=Stage::new(&std::env::temp_dir(),"justfit-model-xdg-compatibility").unwrap();let root=stage.path();let project=root.join("project-must-not-exist");
    let commit="c".repeat(40);let xdg=root.join("xdg");let snapshot=xdg.join("huggingface/hub/models--owner--toy-model/snapshots").join(&commit);weights(&snapshot);
    let binary=std::env::var_os("JUSTFIT_TEST_MODEL_CLI").unwrap_or_else(||env!("CARGO_BIN_EXE_justfit-rs").into());
    let result=Command::new(binary).env("PATH","").env_remove("HOME").env_remove("HF_HOME").env_remove("HF_HUB_CACHE").env("XDG_CACHE_HOME",&xdg).env("HF_HUB_OFFLINE","1")
        .args(["model","install","--project"]).arg(&project).args(["--model","owner/toy-model","--revision",&commit]).output().unwrap();
    assert!(result.status.success(),"{}",String::from_utf8_lossy(&result.stderr));assert!(!project.exists());
    let value:Value=serde_json::from_slice(&result.stdout).unwrap();assert_eq!(value["model"],snapshot.canonicalize().unwrap().to_str().unwrap());
}

#[test]
fn compatibility_hub_cache_env_paths_and_offline_aliases() {
    let stage=Stage::new(&std::env::temp_dir(),"justfit-model-hf-env").unwrap();let root=stage.path();let project=root.join("project-must-not-exist");
    let commit="d".repeat(40);let hub=root.join("hub");let repo=hub.join("models--owner--toy-model");let snapshot=repo.join("snapshots").join(&commit);weights(&snapshot);
    fs::create_dir_all(repo.join("refs")).unwrap();fs::write(repo.join("refs/main"),&commit).unwrap();
    let binary=std::env::var_os("JUSTFIT_TEST_MODEL_CLI").unwrap_or_else(||env!("CARGO_BIN_EXE_justfit-rs").into());
    for (key,path) in [("HUGGINGFACE_HUB_CACHE","~/hub"),("HF_HUB_CACHE","$JUSTFIT_TEST_OWNER/hub"),("HF_HUB_CACHE","${JUSTFIT_TEST_OWNER}/hub")] {
        let result=Command::new(&binary).env("PATH","").env("HOME",root).env_remove("HF_HOME").env_remove("HF_HUB_CACHE").env_remove("HUGGINGFACE_HUB_CACHE").env("JUSTFIT_TEST_OWNER",root).env(key,path).env("HF_HUB_OFFLINE","1")
            .args(["model","install","--project"]).arg(&project).args(["--model","owner/toy-model","--revision",&commit]).output().unwrap();
        assert!(result.status.success(),"{key} {path}: {}",String::from_utf8_lossy(&result.stderr));
    }
    for (key,value) in [("HF_HUB_OFFLINE","YES"),("HF_HUB_OFFLINE","on"),("TRANSFORMERS_OFFLINE","TRUE")] {
        let result=Command::new(env!("CARGO_BIN_EXE_justfit-rs")).env("PATH","").env("HOME",root).env("HF_HUB_CACHE",&hub).env_remove("HF_HUB_OFFLINE").env_remove("TRANSFORMERS_OFFLINE").env(key,value)
            .args(["model","install","--project"]).arg(&project).args(["--model","owner/toy-model","--revision","main"]).output().unwrap();
        assert!(result.status.success(),"{key}={value}: {}",String::from_utf8_lossy(&result.stderr));
    }
    assert!(!project.exists());
}
