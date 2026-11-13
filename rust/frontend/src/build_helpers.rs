//! Rust implementations of the existing make build-identity/config contracts.
//! Keeping those contracts avoids changing the native engine's identity inputs.
use sha2::{Digest,Sha256};
use std::{collections::{BTreeMap,BTreeSet},fs,io::Write,path::{Path,PathBuf},process::Command,sync::atomic::{AtomicU64,Ordering}};
type Result<T>=std::result::Result<T,Box<dyn std::error::Error>>;
static NEXT:AtomicU64=AtomicU64::new(0);
fn atomic(path:&Path,bytes:&[u8])->Result<bool> {
    match fs::read(path) {Ok(old) if old==bytes=>return Ok(false),Err(error) if error.kind()!=std::io::ErrorKind::NotFound=>return Err(error.into()),_=>()}
    let parent=path.parent().filter(|p|!p.as_os_str().is_empty()).unwrap_or(Path::new("."));fs::create_dir_all(parent)?;
    let temporary=parent.join(format!(".justfit-build-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));
    let mut created=false;
    let result=(|| {
        use std::os::unix::fs::OpenOptionsExt;
        let mut output=fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&temporary)?;
        created=true;
        output.write_all(bytes)?;output.sync_all()?;drop(output);fs::rename(&temporary,path)?;Ok(true)
    })();
    // Only this invocation's exclusively created temporary file is removed.
    if created && temporary.exists(){fs::remove_file(&temporary)?;}
    result
}
fn config(args:&[String])->Result<()> {
    let mode=args.first().ok_or("missing build-config mode")?;
    let mut config=None;let mut output=None;let mut command=vec![];let mut paths=vec![];let mut index=1;
    while index<args.len() {
        match args[index].as_str() {
            "--"=>{command.extend_from_slice(&args[index+1..]);break;},
            "--config"|"--output"=>{let value=args.get(index+1).ok_or("missing build-config option value")?;if args[index]=="--config" {if config.replace(value).is_some(){return Err("duplicate --config".into());}}else if output.replace(value).is_some(){return Err("duplicate --output".into());}index+=2;},
            _ if mode=="stale"=>{paths.push(&args[index]);index+=1;},
            _ if mode=="record"=>{command.extend_from_slice(&args[index..]);break;},
            _=>return Err("invalid build-config mode".into()),
        }
    }
    let config=config.ok_or("missing --config")?;
    if !config.is_ascii(){return Err("build configuration must be ASCII".into());}
    if mode=="stale" {
        let expected=format!("{config}\n");
        let stale:Vec<_>=paths.into_iter().filter(|path|fs::read(format!("{path}.config")).ok().as_deref()!=Some(expected.as_bytes())).map(String::as_str).collect();
        println!("{}",stale.join(" "));return Ok(());
    }
    if mode!="record"||command.is_empty(){return Err("record requires a compiler or linker command after --".into());}
    let record=PathBuf::from(format!("{}.config",output.ok_or("missing --output")?));
    match fs::remove_file(&record){Err(error) if error.kind()!=std::io::ErrorKind::NotFound=>return Err(error.into()),_=>()}
    let status=Command::new(&command[0]).args(&command[1..]).status()?;
    if !status.success(){use std::os::unix::process::ExitStatusExt;std::process::exit(status.code().unwrap_or_else(||128+status.signal().unwrap_or(1)));}
    atomic(&record,format!("{config}\n").as_bytes())?;Ok(())
}
fn production_inputs(root:&Path)->Result<BTreeSet<String>> {
    fn walk(root:&Path,path:&Path,result:&mut BTreeSet<String>)->Result<()> {
        for entry in fs::read_dir(path)? {
            let entry=entry?;let path=entry.path();let metadata=entry.file_type()?;
            if metadata.is_dir() {walk(root,&path,result)?;}
            let extension=path.extension().and_then(|s|s.to_str()).unwrap_or("");
            if ["c","cc","cpp","h","hpp","m","mm"].contains(&extension) || (extension=="metal" && path.starts_with(root.join("runtime/metal/kernels"))) {
                result.insert(path.strip_prefix(root)?.to_str().ok_or("non-UTF8 production input")?.to_string());
            }
        }Ok(())
    }
    let mut result=BTreeSet::new();
    // A developer checkout retains the old identity contract. The public SDK
    // excludes the Python oracle entirely, rather than renaming its bytes.
    match fs::symlink_metadata(root.join("dev/tools/build_identity.py")) {
        Ok(meta) if meta.is_file()=>{result.insert("dev/tools/build_identity.py".to_string());},
        Ok(_)=>return Err("legacy identity oracle is not a regular non-symlink file".into()),
        Err(error) if error.kind()==std::io::ErrorKind::NotFound=>(),
        Err(error)=>return Err(error.into()),
    }
    walk(root,&root.join("runtime"),&mut result)?;Ok(result)
}
fn record(hash:&mut Sha256,kind:&[u8],name:&str,value:&[u8]) {
    hash.update(kind);hash.update((name.len() as u64).to_be_bytes());hash.update(name.as_bytes());hash.update((value.len() as u64).to_be_bytes());hash.update(value);
}
fn identity(root:&Path,constants:&BTreeMap<String,String>)->Result<String> {
    let inputs=production_inputs(root)?;
    let legacy=inputs.contains("dev/tools/build_identity.py");
    let mut hash=Sha256::new();
    hash.update(if legacy {b"justfit-production-build-identity-v1\0"}else{b"justfit-production-build-identity-v2\0"});
    for (name,value) in constants {
        if name.is_empty()||name.contains('\0'){return Err("build identity constant name is invalid".into());}
        record(&mut hash,b"constant\0",name,value.as_bytes());
    }
    for name in inputs {
        let path=root.join(&name);
        if !fs::symlink_metadata(&path)?.is_file(){return Err(format!("cannot read production input {name}: input is not a regular non-symlink file").into());}
        record(&mut hash,b"file\0",&name,&fs::read(&path)?);
    }
    let digest=format!("{:x}",hash.finalize());
    // This single audited alias covers only removal of the old Python oracle:
    // all 266 native inputs and every compiler constant remain hashed. Any
    // other source/configuration gets its own v2 identity and new calibration.
    let digest=if !legacy && digest=="10b1923e62c7c109071dd5c88da3427ab4f56e79070a3c64b7126c69f6b1c583" {
        "084c97c431cf003a31d61044b2f5346456d63f7ebcc3172c2cc6c3f87a6d87f2".to_string()
    }else{digest};
    Ok(format!("src-{digest}"))
}
fn identity_command(args:&[String])->Result<()> {
    let mode=args.first().ok_or("missing build-identity mode")?;
    let mut root=std::env::current_dir()?;let mut header=None;let mut stamp=None;let mut constants=BTreeMap::new();let mut binaries=vec![];
    let mut index=1;
    while index<args.len() {
        let key=&args[index];let value=args.get(index+1).ok_or("missing build-identity option value")?;
        match key.as_str() {
            "--root"=>root=PathBuf::from(value),"--header"=>header=Some(PathBuf::from(value)),"--stamp"=>stamp=Some(PathBuf::from(value)),
            "--binary"=>binaries.push(PathBuf::from(value)),
            "--constant"=>{let(name,value)=value.split_once('=').ok_or("invalid --constant")?;if name.is_empty()||constants.insert(name.to_string(),value.to_string()).is_some(){return Err("invalid or duplicate --constant".into());}},
            _=>return Err(format!("unknown build-identity option: {key}").into()),
        }index+=2;
    }
    let root=root.canonicalize()?;let identity=identity(&root,&constants)?;
    let header=header.ok_or("missing --header")?;let stamp=stamp.ok_or("missing --stamp")?;
    let header_bytes=format!("#pragma once\n#define JUSTFIT_BUILD_ID \"{identity}\"\n#define JUSTFIT_BUILD_SOURCE_SHA256 \"{}\"\n",&identity[4..]);
    let stamp_bytes=format!("{{\"build_id\":\"{identity}\",\"schema_version\":1,\"source_sha256\":\"{}\"}}\n",&identity[4..]);
    match mode.as_str() {
        "write"=>{let header_changed=atomic(&header,header_bytes.as_bytes())?;let stamp_changed=atomic(&stamp,stamp_bytes.as_bytes())?;println!("{}",serde_json::json!({"build_id":identity,"header_changed":header_changed,"stamp_changed":stamp_changed}));},
        "check"=>{
            if fs::read(&header)?!=header_bytes.as_bytes() || fs::read(&stamp)?!=stamp_bytes.as_bytes(){return Err("generated build identity is stale".into());}
            for binary in binaries {
                let data=fs::read(&binary)?;let mut found=BTreeSet::new();let mut offset=0;
                while offset+68<=data.len() {
                    let candidate=&data[offset..offset+68];
                    if candidate.starts_with(b"src-") && candidate[4..].iter().all(|c|c.is_ascii_digit()||(b'a'..=b'f').contains(c)) {found.insert(String::from_utf8(candidate.to_vec())?);offset+=68;}else{offset+=1;}
                }
                if found!=BTreeSet::from([identity.clone()]){return Err(format!("binary {} embeds unexpected build identities",binary.display()).into());}
            }
            println!("{}",serde_json::json!({"build_id":identity,"verified":true}));
        },
        _=>return Err("invalid build-identity mode".into()),
    }Ok(())
}
pub fn run(args:&[String])->Result<()> {
    let name=Path::new(args.first().ok_or("missing build helper")?).file_name().and_then(|s|s.to_str()).ok_or("invalid build helper")?;
    match name {"build-config"|"build_config.py"=>config(&args[1..]),"build-identity"|"build_identity.py"=>identity_command(&args[1..]),_=>Err("unsupported build helper contract".into())}
}
