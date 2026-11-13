//! Installer contracts; the bootstrap test executes only a fixture shell marker.
use justfit_rust_frontend::{cli,cli_build::OUTPUTS};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{fs,io::Write,path::PathBuf,process::Command,sync::atomic::{AtomicU64,Ordering}};
static NEXT:AtomicU64=AtomicU64::new(0);
fn payload(mode:&str,name:&str)->&'static [u8] {
    if mode=="self-exec" && name=="source/build/justfit-rs" {b"#!/bin/sh\nprintf '%s\\n' 'runtime-selected' \"$$\" \"$@\"\n"}else{b"runtime-test-data"}
}
fn sha(bytes:&[u8])->String{format!("{:x}",Sha256::digest(bytes))}
struct Fixture{base:PathBuf,root:PathBuf}
impl Fixture {
    fn new()->Self {
        let base=std::env::temp_dir().join(format!("justfit-runtime-install-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));let root=base.join("project");fs::create_dir_all(&root).unwrap();fs::write(root.join("package-manifest.json"),"{}").unwrap();Self{base,root}
    }
    fn archive(&self,mode:&str)->PathBuf {
        let path=self.base.join(format!("{mode}.zip"));let mut zip=zip::ZipWriter::new(fs::File::create(&path).unwrap());
        let options=zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored).unix_permissions(0o600);
        let mut outputs=serde_json::Map::new();for name in OUTPUTS{outputs.insert(name.into(),json!(sha(payload(mode,name))));}
        if mode=="bad-member-hash"{outputs.insert(OUTPUTS[0].into(),json!("a".repeat(64)));}
        let meta=json!({"schema":1,"version":cli::VERSION,"source_manifest_sha256":sha(b"{}"),"architecture":"arm64","minimum_macos":[27,0],"outputs":outputs});
        zip.start_file("runtime.json",options).unwrap();zip.write_all(serde_json::to_string(&meta).unwrap().as_bytes()).unwrap();
        for (i,name) in OUTPUTS.into_iter().enumerate(){zip.start_file(if mode=="escape" && i==0{"../outside"}else{name},options).unwrap();zip.write_all(payload(mode,name)).unwrap();}
        zip.finish().unwrap();path
    }
    fn install(&self,path:&std::path::Path,pin:&str)->std::process::Output {
        Command::new(env!("CARGO_BIN_EXE_justfit-rs")).env("PATH","").args(["runtime","install","--project"]).arg(&self.root).arg("--bundle").arg(path).args(["--sha256",pin]).output().unwrap()
    }
    fn owner_file(&self)->PathBuf {let path=self.root.join(OUTPUTS[0]);fs::create_dir_all(path.parent().unwrap()).unwrap();fs::write(&path,b"owner data").unwrap();path}
}
impl Drop for Fixture{fn drop(&mut self){fs::remove_dir_all(&self.base).unwrap();}}
#[test]
fn installs_all_outputs_and_receipt_without_python() {
    let fixture=Fixture::new();let archive=fixture.archive("valid");let result=fixture.install(&archive,&sha(&fs::read(&archive).unwrap()));
    assert!(result.status.success(),"{}",String::from_utf8_lossy(&result.stderr));
    let result:Value=serde_json::from_slice(&result.stdout).unwrap();assert_eq!(result["outputs"],11);assert_eq!(result["installer"],"rust");
    let receipt:Value=serde_json::from_slice(&fs::read(fixture.root.join("build-receipt.json")).unwrap()).unwrap();assert_eq!(receipt["outputs"].as_object().unwrap().len(),11);
    for name in OUTPUTS{assert_eq!(fs::read(fixture.root.join(name)).unwrap(),b"runtime-test-data");}
    use std::os::unix::fs::PermissionsExt;
    assert_eq!(fs::metadata(fixture.root.join("source/build/justfit-rs")).unwrap().permissions().mode()&0o777,0o755);
}
#[test]
fn wrong_pin_escape_and_bad_member_hash_preserve_owner_outputs() {
    for mode in ["wrong-pin","escape","bad-member-hash"] {
        let fixture=Fixture::new();let owner=fixture.owner_file();let archive=fixture.archive(mode);
        let pin=if mode=="wrong-pin"{"0".repeat(64)}else{sha(&fs::read(&archive).unwrap())};let result=fixture.install(&archive,&pin);
        assert!(!result.status.success(),"{mode}");assert_eq!(fs::read(owner).unwrap(),b"owner data");assert!(!fixture.base.join("outside").exists());assert!(!fixture.root.join("build-receipt.json").exists());
    }
}
#[test]
fn corrupt_crc_and_symlink_destination_cannot_publish() {
    let fixture=Fixture::new();let owner=fixture.owner_file();let archive=fixture.archive("crc");let mut bytes=fs::read(&archive).unwrap();let at=bytes.windows(b"runtime-test-data".len()).position(|p|p==b"runtime-test-data").unwrap();bytes[at]^=1;fs::write(&archive,&bytes).unwrap();
    assert!(!fixture.install(&archive,&sha(&bytes)).status.success());assert_eq!(fs::read(owner).unwrap(),b"owner data");
    let fixture=Fixture::new();let outside=fixture.base.join("owner-directory");fs::create_dir(&outside).unwrap();fs::create_dir_all(fixture.root.join("source")).unwrap();std::os::unix::fs::symlink(&outside,fixture.root.join("source/build")).unwrap();let archive=fixture.archive("symlink-destination");
    let result=fixture.install(&archive,&sha(&fs::read(&archive).unwrap()));assert!(!result.status.success());assert!(String::from_utf8_lossy(&result.stderr).contains("unsafe runtime destination"));assert!(fs::read_dir(outside).unwrap().next().is_none());
}

#[test]
fn serve_installs_runtime_then_replaces_bootstrap_without_repeating_install() {
    let fixture=Fixture::new();
    let project=PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf();
    fs::create_dir_all(fixture.root.join("config/nodes")).unwrap();
    fs::copy(project.join("config/nodes/m4.json"),fixture.root.join("config/nodes/m4.json")).unwrap();
    let model=fixture.base.join("model");for name in ["target","draft","tokenizer"]{fs::create_dir_all(model.join(name)).unwrap();}
    let archive=fixture.archive("self-exec");
    let child=Command::new(env!("CARGO_BIN_EXE_justfit-rs")).env("PATH","").args(["serve","--project"]).arg(&fixture.root).args(["--node","m4","--model"]).arg(&model).arg("--no-ane-vision").arg("--runtime-bundle").arg(&archive).arg("--runtime-sha256").arg(sha(&fs::read(&archive).unwrap())).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).spawn().unwrap();
    let pid=child.id();let result=child.wait_with_output().unwrap();assert!(result.status.success(),"{}",String::from_utf8_lossy(&result.stderr));
    let output=String::from_utf8(result.stdout).unwrap();let lines:Vec<_>=output.lines().collect();
    assert_eq!(lines[0],"runtime-selected");assert_eq!(lines[1],pid.to_string());assert_eq!(lines[2],"serve");
    assert!(!output.contains("--runtime-"));assert!(output.contains("--no-ane-vision"));assert!(output.contains(model.to_str().unwrap()));
    assert!(fixture.root.join("build-receipt.json").is_file());
}

#[test]
fn runtime_outputs_include_license_and_do_not_redistribute_system_zlib() {
    assert!(OUTPUTS.contains(&"source/build/PDFIUM-LICENSES.txt"));
    assert!(!OUTPUTS.iter().any(|name|name.contains("libjustfit-z")||name.contains("libz.")));
}

fn bundled(fixture:&Fixture,archive:&std::path::Path)->PathBuf {
    let payload=fixture.base.join("payload");fs::create_dir_all(payload.join("runtime")).unwrap();
    fs::write(payload.join("package-manifest.json"),b"{}").unwrap();
    fs::copy(archive,payload.join("runtime/runtime.zip")).unwrap();
    fs::write(payload.join("bundled-runtime.json"),serde_json::to_vec(&json!({"schema":1,"version":cli::VERSION,"source_manifest_sha256":sha(b"{}"),"archive":"runtime/runtime.zip","archive_sha256":sha(&fs::read(archive).unwrap())})).unwrap()).unwrap();payload
}
#[test]
fn bundled_descriptor_is_optional_but_invalid_metadata_never_mutates_runtime(){
    use justfit_rust_frontend::runtime_bundle::install_bundled;
    let fixture=Fixture::new();let archive=fixture.archive("valid");let payload=bundled(&fixture,&archive);let owner=fixture.owner_file();
    let descriptor=payload.join("bundled-runtime.json");let original=fs::read(&descriptor).unwrap();
    fs::remove_file(&descriptor).unwrap();assert!(!install_bundled(&fixture.root,&payload).unwrap());
    for (field,value) in [("schema",json!(2)),("version",json!("old")),("source_manifest_sha256",json!("a".repeat(64))),("archive",json!("../owner")),("archive_sha256",json!("0".repeat(64)))]{
        let mut meta:Value=serde_json::from_slice(&original).unwrap();meta[field]=value;fs::write(&descriptor,serde_json::to_vec(&meta).unwrap()).unwrap();assert!(install_bundled(&fixture.root,&payload).is_err());assert_eq!(fs::read(&owner).unwrap(),b"owner data");assert!(!fixture.root.join("build-receipt.json").exists());
    }
    fs::write(&descriptor,&original).unwrap();fs::rename(&descriptor,payload.join("owner-descriptor")).unwrap();std::os::unix::fs::symlink(payload.join("owner-descriptor"),&descriptor).unwrap();assert!(install_bundled(&fixture.root,&payload).is_err());fs::remove_file(&descriptor).unwrap();fs::write(&descriptor,&original).unwrap();
    let packed=payload.join("runtime/runtime.zip");fs::rename(&packed,payload.join("owner-runtime.zip")).unwrap();std::os::unix::fs::symlink(payload.join("owner-runtime.zip"),&packed).unwrap();assert!(install_bundled(&fixture.root,&payload).is_err());assert_eq!(fs::read(owner).unwrap(),b"owner data");
}
#[test]
fn serve_uses_bundled_runtime_without_compilers_or_install_flags(){
    let fixture=Fixture::new();let archive=fixture.archive("self-exec");let payload=bundled(&fixture,&archive);fs::create_dir(payload.join("bin")).unwrap();
    let source=std::env::var_os("JUSTFIT_TEST_BUNDLED_CLI").unwrap_or_else(||env!("CARGO_BIN_EXE_justfit-rs").into());let exe=payload.join("bin/justfit");fs::copy(source,&exe).unwrap();
    let project=PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf();fs::create_dir_all(fixture.root.join("config/nodes")).unwrap();fs::copy(project.join("config/nodes/m4.json"),fixture.root.join("config/nodes/m4.json")).unwrap();
    let child=Command::new(&exe).env("PATH","").args(["serve","--project"]).arg(&fixture.root).args(["--node","m4","--model","/missing-model-not-used-until-runtime-exec","--no-ane-vision"]).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).spawn().unwrap();let pid=child.id();let output=child.wait_with_output().unwrap();assert!(output.status.success(),"{}",String::from_utf8_lossy(&output.stderr));let text=String::from_utf8(output.stdout).unwrap();let lines:Vec<_>=text.lines().collect();assert_eq!(lines[0],"runtime-selected");assert_eq!(lines[1],pid.to_string());assert!(!text.contains("--runtime-"));assert!(fixture.root.join("build-receipt.json").is_file());
}
