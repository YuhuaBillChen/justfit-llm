//! Public workspace and lifecycle commands implemented in Rust.
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, io::Read, path::{Component, Path, PathBuf}};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const VERSION:&str="2.6.0";
pub(crate) fn digest(path: &Path) -> Result<String> {
    let mut input=fs::File::open(path)?;
    let mut hash=Sha256::new();let mut bytes=[0u8;65536];
    loop {let n=input.read(&mut bytes)?;if n==0 {break;}hash.update(&bytes[..n]);}
    Ok(format!("{:x}",hash.finalize()))
}
pub fn verify(root: &Path) -> Result<BTreeMap<String,String>> {
    let root=root.canonicalize()?;
    let manifest: BTreeMap<String,String>=serde_json::from_slice(&fs::read(root.join("package-manifest.json"))?)?;
    for (name,want) in &manifest {
        let relative=Path::new(name);
        if relative.is_absolute() || relative.components().any(|v|matches!(v,Component::ParentDir|Component::RootDir|Component::Prefix(_))) {
            return Err(format!("unsafe manifest path: {name}").into());
        }
        let path=root.join(relative);
        let safe=fs::symlink_metadata(&path).is_ok_and(|meta|meta.is_file() && !meta.file_type().is_symlink()) && path.canonicalize().is_ok_and(|path|path.starts_with(&root));
        if !safe {return Err(format!("missing or unsafe source: {name}").into());}
        if digest(&path)?!=*want {return Err(format!("source digest mismatch: {name}").into());}
    }
    Ok(manifest)
}
pub(crate) fn payload_root() -> Result<PathBuf> {
    let exe=std::env::current_exe()?;
    exe.ancestors().skip(1).find(|path|path.join("package-manifest.json").is_file()).map(Path::to_path_buf).ok_or_else(||"cannot locate bundled source manifest; pass --project/--source".into())
}
pub(crate) fn initialize(source: &Path, destination: &Path) -> Result<PathBuf> {
    let source=source.canonicalize()?;
    let manifest=verify(&source)?;
    let destination=std::path::absolute(destination)?;
    if let Some(parent)=destination.parent() {fs::create_dir_all(parent)?;}
    // Exclusive creation preserves existing work and rejects dangling symlinks.
    fs::create_dir(&destination)?;
    for name in manifest.keys().map(String::as_str).chain(std::iter::once("package-manifest.json")) {
        let target=destination.join(name);
        if let Some(parent)=target.parent() {fs::create_dir_all(parent)?;}
        fs::copy(source.join(name),&target)?;
        if name=="source/justfit" || name.ends_with(".sh") {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions=fs::metadata(&target)?.permissions();permissions.set_mode(permissions.mode()|0o111);fs::set_permissions(target,permissions)?;
        }
    }
    verify(&destination)?;
    if source.join("config/ane-distribution.json").is_file(){
        use std::io::Write;
        let mut locator=fs::OpenOptions::new().write(true).create_new(true).open(destination.join("distribution-locator.json"))?;
        locator.write_all(&serde_json::to_vec(&serde_json::json!({"payload":source,"source_manifest_sha256":digest(&destination.join("package-manifest.json"))?}))?)?;locator.sync_all()?;
    }
    Ok(destination)
}

/// Public user commands share this executable with native process helpers.
pub fn dispatch(args: &[String]) -> Option<Result<()>> {
    if args.is_empty(){return Some(Err("usage: justfit COMMAND; run justfit --help".into()));}
    if let Some(help)=crate::cli_help::text(args){println!("{help}");return Some(Ok(()));}
    let first=args.first()?.as_str();
    match first {
        "replay"=>Some(crate::replay::run_cli(&args[1..])),
        "doctor"=>Some(crate::cli_platform::doctor(&args[1..])),
        "ane-bootstrap"=>Some(crate::ane_distribution::run_cli(&args[1..])),
        "ane"=>Some(crate::ane_bundle::run_cli(&args[1..])),
        "model"=>Some(crate::model_install::run_cli(&args[1..])),
        "runtime"=>Some(crate::runtime_bundle::run_cli(&args[1..])),
        "build"=>Some(crate::cli_build::run_cli(&args[1..])),
        "serve" if args.len()==1 || args.get(1).is_some_and(|v|v.starts_with("--"))=>Some(crate::cli_serve::run(&args[1..])),
        "--version" if args.len()==1=>Some({println!("{VERSION}");Ok(())}),
        "verify"=>Some((|| {
            let root=match args {[_]=>payload_root()?,[_,flag,root] if flag=="--project"=>PathBuf::from(root),_=>return Err("usage: justfit verify [--project PATH]".into())};
            let manifest=verify(&root)?;
            println!("{}",serde_json::json!({"passed":true,"files":manifest.len()}));Ok(())
        })()),
        "init"=>Some((|| {
            let (destination,source)=match args { [_,destination]=>(PathBuf::from(destination),payload_root()?),[_,destination,flag,source] if flag=="--source"=>(PathBuf::from(destination),PathBuf::from(source)),_=>return Err("usage: justfit init DESTINATION [--source PATH]".into())};
            println!("{}",initialize(&source,&destination)?.display());Ok(())
        })()),
        _ if matches!(first,"build-helper"|"tokenizer-check"|"templates"|"pdf-worker"|"media-driver"|"native-cache-prepare"|"coreai-probe"|"ane-worker"|"geometry-probe"|"image-probe"|"serve"|"serve-fd"|"jsonl")=>None,
        _=>Some(Err(format!("unknown command: {first}; run justfit --help").into())),
    }
}

pub fn select_preset(brand:&str,memory_bytes:u64)->Result<&'static str> {
    let invalid=||format!("unrecognized chip {brand:?}; pass --node m4, m5 or m6");
    let suffix=brand.strip_prefix("Apple M").ok_or_else(invalid)?;
    let end=suffix.find(|c:char|!c.is_ascii_digit()).unwrap_or(suffix.len());
    if end==0 || suffix[end..].chars().next().is_some_and(|c|c.is_alphanumeric()||c=='_') {return Err(invalid().into());}
    let generation:u64=suffix[..end].parse()?;
    if generation<4 {return Err(format!("{brand} is older than the tested M4 baseline; pass --node explicitly at your own risk").into());}
    if generation>6 {return Err(format!("{brand} has no tested automatic preset; pass --node explicitly").into());}
    if memory_bytes<24*1024*1024*1024 {return Err("memory is below the 24 GiB the presets require".into());}
    Ok(match generation{4=>"m4",5=>"m5",_=>"m6"})
}
pub(crate) fn detect_preset()->Result<&'static str> {
    let read=|name:&str|->Result<String>{
        let output=std::process::Command::new("/usr/sbin/sysctl").args(["-n",name]).output()?;
        if !output.status.success(){return Err(format!("cannot read {name}").into());}
        Ok(String::from_utf8(output.stdout)?.trim().to_string())
    };
    select_preset(&read("machdep.cpu.brand_string")?,read("hw.memsize")?.parse()?)
}
pub(crate) fn default_project()->Result<PathBuf> {
    Ok(PathBuf::from(std::env::var_os("HOME").ok_or("HOME is missing")?).join("Library/Application Support/JustFit").join(VERSION))
}
#[cfg(test)]
mod preset_tests {
    use super::*;
    #[test]
    fn same_qualified_chip_and_memory_boundaries_as_reference() {
        for (brand,want) in [("Apple M4","m4"),("Apple M4 Pro","m4"),("Apple M5 Max","m5"),("Apple M6 Ultra","m6")] {
            assert_eq!(select_preset(brand,24<<30).unwrap(),want);
            assert!(select_preset(brand,(24<<30)-1).is_err());
        }
        for brand in ["Intel","Apple M3 Max","Apple M7","Apple M4x","Apple M41"] {assert!(select_preset(brand,128<<30).is_err());}
    }
}
