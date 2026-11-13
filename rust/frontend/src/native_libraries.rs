//! Source builds obtain a pinned upstream PDFium asset without Python wheels.
use crate::{cli,download::{self,Result,Stage}};
use serde_json::Value;
use std::{fs,path::{Path,PathBuf},process::{Command,Stdio}};

pub(crate) fn pdfium(root:&Path,explicit:Option<&str>)->Result<(PathBuf,Option<Stage>)> {
    let manifest:Value=serde_json::from_slice(&fs::read(root.join("config/native-libraries.json"))?)?;
    let record=&manifest["libraries"]["source/build/libpdfium.dylib"];
    let expected=record["sha256"].as_str().ok_or("PDFium library SHA256 missing")?;download::digest(expected)?;
    let path=explicit.map(PathBuf::from).unwrap_or_else(||root.join("source/build/libpdfium.dylib"));
    if explicit.is_some() || path.exists() {
        if !path.is_file() || cli::digest(&path)?!=expected{return Err("PDFium upstream SHA256 mismatch".into());}
        return Ok((path.canonicalize()?,None));
    }
    let archive_size=record["archive_bytes"].as_u64().filter(|n|*n>0&&*n<=64*1024*1024).ok_or("invalid PDFium archive size")?;
    let archive_sha=record["archive_sha256"].as_str().ok_or("PDFium archive SHA256 missing")?;
    let url=record["url"].as_str().ok_or("PDFium upstream URL missing")?;
    if record["member"]!="lib/libpdfium.dylib" {return Err("unsupported PDFium archive layout".into());}
    let stage=Stage::new(root,"pdfium-upstream")?;let archive=stage.path().join("upstream.tgz");
    if download::fetch(url,&archive,archive_size,archive_sha)?!=archive_size{return Err("PDFium archive size mismatch".into());}
    // Parse only the authenticated immutable upstream asset. Stream a fixed
    // member to an owned file; archive names never select destination paths.
    let path=stage.path().join("libpdfium.dylib");
    let output=fs::OpenOptions::new().write(true).create_new(true).open(&path)?;
    let status=Command::new("/usr/bin/tar").arg("-xOf").arg(&archive).arg("lib/libpdfium.dylib").stdin(Stdio::null()).stdout(Stdio::from(output)).stderr(Stdio::null()).status()?;
    if !status.success() || record["bytes"].as_u64()!=Some(fs::metadata(&path)?.len()) || cli::digest(&path)?!=expected{return Err("PDFium upstream library verification failed".into());}
    Ok((path,Some(stage)))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_library_is_pinned_and_owner_files_are_preserved() {
        let stage=Stage::new(&std::env::temp_dir(),"justfit-native-library-test").unwrap();let root=stage.path();
        fs::create_dir(root.join("config")).unwrap();let library=root.join("owner.dylib");fs::write(&library,b"owner-library").unwrap();
        let record=serde_json::json!({"libraries":{"source/build/libpdfium.dylib":{"sha256":cli::digest(&library).unwrap()}}});
        fs::write(root.join("config/native-libraries.json"),serde_json::to_vec(&record).unwrap()).unwrap();
        let (actual,owned)=pdfium(root,library.to_str()).unwrap();assert_eq!(actual,library.canonicalize().unwrap());assert!(owned.is_none());
        fs::write(&library,b"different-owner-library").unwrap();assert!(pdfium(root,library.to_str()).is_err());assert_eq!(fs::read(library).unwrap(),b"different-owner-library");
    }
}
#[cfg(test)]
mod upstream_download_contract {
    use super::*;
    #[test]
    #[ignore = "requires M4 HTTPS access to the pinned upstream release"]
    fn pinned_upstream_pdfium_download_has_no_python_wheel_input() {
        let stage=Stage::new(&std::env::temp_dir(),"justfit-pdfium-download-test").unwrap();let root=stage.path();
        fs::create_dir(root.join("config")).unwrap();
        fs::write(root.join("config/native-libraries.json"),include_bytes!("../../../config/native-libraries.json")).unwrap();
        let (path,owned)=pdfium(root,None).unwrap();assert!(owned.is_some());
        assert_eq!(fs::metadata(&path).unwrap().len(),7388864);
        assert_eq!(cli::digest(&path).unwrap(),"a45c44645523968525c710462691dd974f4efb1013ce1c249919a0939cfb373a");
        assert!(!root.join("source/build/libpdfium.dylib").exists());
    }
}
