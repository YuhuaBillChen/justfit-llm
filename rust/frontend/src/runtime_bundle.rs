//! Pinned runtime installation. Archive names never select filesystem paths.
use crate::{cli,cli_build::OUTPUTS,download::{self,Result,Stage}};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{collections::{BTreeMap,BTreeSet},fs,io::{Read,Write},os::{fd::AsRawFd,unix::fs::{OpenOptionsExt,PermissionsExt}},path::{Path,PathBuf}};
const LIMIT:u64=256*1024*1024;
struct Lock(fs::File);
impl Lock {
    fn take(root:&Path)->Result<Self> {
        let file=fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).mode(0o600).custom_flags(libc::O_NOFOLLOW).open(root.join(".runtime-install.lock"))?;
        if !file.metadata()?.is_file(){return Err("unsafe runtime installer lock".into());}
        if unsafe{libc::flock(file.as_raw_fd(),libc::LOCK_EX)}!=0{return Err(std::io::Error::last_os_error().into());}Ok(Self(file))
    }
}
impl Drop for Lock{fn drop(&mut self){let _=unsafe{libc::flock(self.0.as_raw_fd(),libc::LOCK_UN)};}}
fn macos_version()->Result<(u64,u64)> {
    let result=std::process::Command::new("/usr/bin/sw_vers").arg("-productVersion").output()?;
    if !result.status.success(){return Err("cannot identify macOS release".into());}
    let value=String::from_utf8(result.stdout)?;let mut parts=value.trim().split('.');
    Ok((parts.next().ok_or("invalid macOS release")?.parse()?,parts.next().unwrap_or("0").parse()?))
}
fn metadata(meta:&Value,root:&Path)->Result<BTreeMap<String,String>> {
    if meta["schema"]!=1 || meta["version"]!=cli::VERSION{return Err("runtime version/schema mismatch".into());}
    if meta["source_manifest_sha256"]!=cli::digest(&root.join("package-manifest.json"))?{return Err("runtime does not match this source package".into());}
    if !cfg!(all(target_os="macos",target_arch="aarch64")) || meta["architecture"]!="arm64"{return Err("runtime requires macOS arm64".into());}
    let floor=meta["minimum_macos"].as_array().filter(|a|a.len()==2).ok_or("invalid runtime macOS floor")?;
    let floor=(floor[0].as_u64().ok_or("invalid runtime macOS floor")?,floor[1].as_u64().ok_or("invalid runtime macOS floor")?);
    if macos_version()?<floor{return Err("runtime requires a newer macOS release".into());}
    let outputs: BTreeMap<String,String>=serde_json::from_value(meta["outputs"].clone()).map_err(|_|"runtime output list mismatch")?;
    if outputs.keys().map(String::as_str).collect::<BTreeSet<_>>()!=OUTPUTS.into_iter().collect(){return Err("runtime output list mismatch".into());}
    for value in outputs.values(){download::digest(value)?;}Ok(outputs)
}
fn safe_destination(root:&Path,name:&str)->Result<PathBuf> {
    let path=root.join(name);let mut current=path.as_path();
    while current!=root {
        match fs::symlink_metadata(current) {
            Ok(meta) if meta.file_type().is_symlink() || (current==path && !meta.is_file()) || (current!=path && !meta.is_dir())=>return Err("unsafe runtime destination".into()),
            Err(error) if error.kind()!=std::io::ErrorKind::NotFound=>return Err(error.into()),_=>(),
        }
        current=current.parent().ok_or("unsafe runtime destination")?;
    }Ok(path)
}
fn snapshot(archive:&Path,target:&Path,expected:&str)->Result<()> {
    // Copy once into private staging before hashing/parsing, so replacing the
    // caller's ZIP cannot replace already authenticated bytes during install.
    let mut input=fs::OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW|libc::O_NONBLOCK).open(archive.canonicalize()?)?;
    if !input.metadata()?.is_file() || input.metadata()?.len()>LIMIT{return Err("runtime archive exceeds size limit or is not a regular file".into());}
    let mut output=fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(target)?;
    let mut hash=Sha256::new();let mut bytes=[0u8;65536];let mut total=0u64;
    loop {let n=input.read(&mut bytes)?;if n==0{break;}total=total.checked_add(n as u64).ok_or("runtime size overflow")?;if total>LIMIT{return Err("runtime archive exceeds size limit".into());}hash.update(&bytes[..n]);output.write_all(&bytes[..n])?;}
    if format!("{:x}",hash.finalize())!=expected{return Err("runtime archive SHA256 mismatch".into());}output.sync_all()?;Ok(())
}
pub fn install(root:&Path,archive:&Path,expected:&str)->Result<Value> {
    download::digest(expected)?;let root=root.canonicalize()?;cli::verify(&root)?;
    let _lock=Lock::take(&root)?;let stage=Stage::new(&root,"runtime")?;
    let frozen=stage.path().join("archive.zip");snapshot(archive,&frozen,expected)?;
    let mut archive=zip::ZipArchive::new(fs::File::open(frozen)?)?;
    if archive.len()!=OUTPUTS.len()+1{return Err("unexpected or duplicate runtime archive member".into());}
    let allowed:BTreeSet<_>=OUTPUTS.into_iter().chain(std::iter::once("runtime.json")).collect();
    let mut entries=BTreeMap::new();let mut expanded=0u64;
    for index in 0..archive.len() {
        let entry=archive.by_index(index)?;let name=entry.name()?.into_owned();
        if !allowed.contains(name.as_str()) || entries.insert(name.clone(),index).is_some(){return Err("unexpected or duplicate runtime archive member".into());}
        if !entry.unix_mode().is_some_and(|mode|mode&0o170000==0o100000){return Err("runtime archive must contain regular files only".into());}
        expanded=expanded.checked_add(entry.size()).ok_or("runtime archive size overflow")?;
        if expanded>LIMIT || (name=="runtime.json" && entry.size()>65536){return Err("runtime archive exceeds size limit".into());}
    }
    if entries.keys().map(String::as_str).collect::<BTreeSet<_>>()!=allowed{return Err("unexpected runtime archive member".into());}
    let mut bytes=vec![];archive.by_index(entries["runtime.json"])?.take(65537).read_to_end(&mut bytes)?;
    if bytes.len()>65536{return Err("runtime metadata exceeds size limit".into());}
    let meta:Value=serde_json::from_slice(&bytes)?;let outputs=metadata(&meta,&root)?;
    for name in OUTPUTS {safe_destination(&root,name)?;}
    safe_destination(&root,"build-receipt.json")?;
    for (index,name) in OUTPUTS.into_iter().enumerate() {
        let mut entry=archive.by_index(entries[name])?;
        let target=stage.path().join(index.to_string());let mut output=fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&target)?;
        let mut total=0u64;let mut hash=Sha256::new();let mut bytes=[0u8;65536];
        loop {let n=entry.read(&mut bytes)?;if n==0{break;}total=total.checked_add(n as u64).ok_or("runtime member size overflow")?;if total>LIMIT || total>entry.size(){return Err("runtime member exceeds size limit".into());}hash.update(&bytes[..n]);output.write_all(&bytes[..n])?;}
        if total!=entry.size() || format!("{:x}",hash.finalize())!=outputs[name]{return Err(format!("runtime output SHA256 mismatch: {name}").into());}
        output.sync_all()?;drop(output);
        let executable=name.ends_with(".dylib")||name.ends_with("/justfit-engine")||name.ends_with("/justfit-rs")||name.ends_with("/justfit-ane-worker");
        fs::set_permissions(&target,fs::Permissions::from_mode(if executable{0o755}else{0o644}))?;
    }
    // No runtime output is published until every member has passed CRC/hash.
    cli::verify(&root)?;
    for (index,name) in OUTPUTS.into_iter().enumerate() {
        let target=safe_destination(&root,name)?;fs::create_dir_all(target.parent().ok_or("invalid runtime output")?)?;fs::rename(stage.path().join(index.to_string()),target)?;
    }
    let mut receipt=meta.clone();receipt["status"]=json!("prebuilt_installed_runtime_unqualified");receipt["archive_sha256"]=json!(expected);receipt["installer"]=json!("rust");
    let path=stage.path().join("receipt.json");fs::write(&path,serde_json::to_vec_pretty(&receipt)?)?;fs::rename(path,root.join("build-receipt.json"))?;
    Ok(meta)
}
/// A binary distribution carries a verified runtime beside its source SDK.
/// Absence permits a source build; a present but invalid descriptor is an error.
pub fn install_bundled(root:&Path,payload:&Path)->Result<bool> {
    let payload=payload.canonicalize()?;
    let path=payload.join("bundled-runtime.json");
    let mut input=match fs::OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW|libc::O_NONBLOCK).open(&path) {
        Ok(input)=>input,
        Err(error) if error.kind()==std::io::ErrorKind::NotFound=>return Ok(false),
        Err(error)=>return Err(error.into()),
    };
    if !input.metadata()?.is_file()||input.metadata()?.len()>65536{return Err("invalid bundled runtime descriptor".into());}
    let mut bytes=vec![];std::io::Read::by_ref(&mut input).take(65537).read_to_end(&mut bytes)?;
    if bytes.len()>65536{return Err("bundled runtime descriptor exceeds limit".into());}
    let meta:Value=serde_json::from_slice(&bytes)?;
    if meta["schema"]!=1||meta["version"]!=cli::VERSION||meta["source_manifest_sha256"]!=cli::digest(&root.join("package-manifest.json"))?
        ||meta["source_manifest_sha256"]!=cli::digest(&payload.join("package-manifest.json"))?
        ||meta["archive"]!="runtime/runtime.zip"{return Err("bundled runtime descriptor identity/path mismatch".into());}
    let pin=meta["archive_sha256"].as_str().ok_or("bundled runtime archive SHA256 missing")?;
    download::digest(pin)?;
    let archive=safe_destination(&payload,"runtime/runtime.zip")?;
    install(root,&archive,pin)?;Ok(true)
}

pub fn fetch_and_install(root:&Path,url:&str,expected:&str)->Result<Value> {
    download::digest(expected)?;let root=root.canonicalize()?;cli::verify(&root)?;let stage=Stage::new(&root,"runtime-download")?;
    let manifest=stage.path().join("runtime.json");download::fetch(url,&manifest,65536,expected)?;
    let meta:Value=serde_json::from_slice(&fs::read(manifest)?)?;metadata(&meta,&root)?;
    let archive=stage.path().join("runtime.zip");let archive_sha=meta["archive_sha256"].as_str().ok_or("runtime archive SHA256 missing")?;
    download::fetch(meta["url"].as_str().ok_or("runtime archive URL missing")?,&archive,LIMIT,archive_sha)?;
    install(&root,&archive,archive_sha)
}
pub fn run_cli(args:&[String])->Result<()> {
    if args.first().map(String::as_str)!=Some("install"){return Err("usage: justfit runtime install --project PATH (--bundle ZIP|--manifest HTTPS_URL) --sha256 SHA".into());}
    let mut flags=BTreeMap::new();let mut index=1;
    while index<args.len(){let flag=args[index].as_str();if !["--project","--bundle","--manifest","--sha256"].contains(&flag){return Err(format!("unknown runtime option: {flag}").into());}let value=args.get(index+1).ok_or("runtime option needs a value")?;if flags.insert(flag,value.as_str()).is_some(){return Err(format!("duplicate runtime option: {flag}").into());}index+=2;}
    let root=PathBuf::from(flags.get("--project").ok_or("runtime --project is required")?);let expected=flags.get("--sha256").ok_or("runtime --sha256 is required")?;
    let meta=match (flags.get("--bundle"),flags.get("--manifest")){(Some(path),None)=>install(&root,Path::new(path),expected)?,(None,Some(url))=>fetch_and_install(&root,url,expected)?,_=>return Err("choose one runtime bundle or HTTPS manifest".into())};
    println!("{}",json!({"installed":true,"outputs":meta["outputs"].as_object().map_or(0,|m|m.len()),"installer":"rust"}));Ok(())
}
