//! Keep prepared source-model assemblies alive using the released flock contract.
use crate::download::Result;
use serde_json::Value;
use std::{collections::BTreeSet,fs,io::Read,os::{fd::AsRawFd,unix::fs::{MetadataExt,OpenOptionsExt}},path::{Path,PathBuf}};
const LIMIT:u64=64*1024*1024;
pub struct HeldModel{pub path:PathBuf,_record:Option<fs::File>}
fn keys(value:&Value,expected:&[&str])->bool{value.as_object().is_some_and(|v|v.keys().map(String::as_str).collect::<BTreeSet<_>>()==expected.iter().copied().collect())}
fn hex(value:&Value,length:usize)->bool{value.as_str().is_some_and(|v|v.len()==length&&v.bytes().all(|c|c.is_ascii_hexdigit()))}
fn integer(value:&Value)->bool{value.as_number().is_some_and(|n|!n.to_string().contains(['.','e','E']))}
fn nanoseconds(meta:&fs::Metadata,modified:bool)->i128{if modified{meta.mtime() as i128*1_000_000_000+meta.mtime_nsec() as i128}else{meta.ctime() as i128*1_000_000_000+meta.ctime_nsec() as i128}}
fn matches_integer(value:&Value,expected:i128)->bool{integer(value)&&value.to_string().parse::<i128>().ok()==Some(expected)}
pub(crate) fn verify(root:&Path,record:&Value)->Result<()>{
    let gguf=record["target_format"]=="gguf";let mut names=vec!["version","model","family","target_format","vision_format","sources","files"];if gguf{names.push("metadata");}
    if !keys(record,&names)||record["version"].as_u64()!=Some(1)||record["model"].as_str().is_none()||record["family"].as_str().is_none()
        ||!matches!(record["target_format"].as_str(),Some("mlx-affine"|"gguf"))||!matches!(record["vision_format"].as_str(),Some("none"|"safetensors"|"gguf"))||(gguf&&!hex(&record["metadata"],64))
        ||!keys(&record["sources"],&["target","draft"]){return Err("invalid resolved model record".into());}
    for source in record["sources"].as_object().unwrap().values(){if !keys(source,&["repo","revision"])||source["repo"].as_str().is_none()||(!source["revision"].is_null()&&!hex(&source["revision"],40)){return Err("invalid resolved model source".into());}}
    for (name,entry) in record["files"].as_object().ok_or("invalid resolved model files")?{
        if !crate::model_install::safe_path(name)||!keys(entry,&["path","bytes","mtime_ns","ctime_ns","digest"])||entry["path"].as_str().is_none()
            ||!["bytes","mtime_ns","ctime_ns"].iter().all(|key|integer(&entry[*key]))||(!hex(&entry["digest"],40)&&!hex(&entry["digest"],64)){return Err("invalid resolved model file record".into());}
        let path=root.join(name);let meta=fs::metadata(&path)?;
        if !path.is_symlink()||fs::read_link(&path)?.as_os_str()!=Path::new(entry["path"].as_str().unwrap()).as_os_str()
            ||!meta.is_file()||!matches_integer(&entry["bytes"],meta.len() as i128)||!matches_integer(&entry["mtime_ns"],nanoseconds(&meta,true))||!matches_integer(&entry["ctime_ns"],nanoseconds(&meta,false)){
            return Err(format!("resolved model file changed: {name}").into());
        }
    }Ok(())
}
pub fn hold(selection:&Path)->Result<HeldModel>{
    let initial=selection.canonicalize()?;
    if !selection.join("model.json").exists(){
        // Preserve local legacy-package behavior, including unmanifested local
        // directories. An orphan in a managed assembly store is never served.
        if selection.join("manifest.json").exists()||initial.parent().and_then(Path::file_name)!=Some(std::ffi::OsStr::new(".resolved")){return Ok(HeldModel{path:initial,_record:None});}
        return Err("model selection names no prepared assembly; prepare it again".into());
    }
    let resolved=initial.parent().filter(|p|p.file_name()==Some(std::ffi::OsStr::new(".resolved"))).ok_or("model directory is not an assembly JustFit prepared")?;
    let store=resolved.parent().ok_or("model assembly store missing")?;
    let installation=fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).mode(0o600).custom_flags(libc::O_NOFOLLOW|libc::O_NONBLOCK).open(store.join(".install.lock"))?;
    if !installation.metadata()?.is_file(){return Err("unsafe model installation lock".into());}
    if unsafe{libc::flock(installation.as_raw_fd(),libc::LOCK_EX)}!=0{return Err(std::io::Error::last_os_error().into());}
    let path=selection.canonicalize()?;if path.parent()!=Some(resolved){return Err("model selection changed assembly store during startup".into());}
    let record=fs::OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW|libc::O_NONBLOCK).open(path.join("model.json"))?;
    if !record.metadata()?.is_file()||record.metadata()?.len()>LIMIT{return Err("invalid resolved model record file".into());}
    if unsafe{libc::flock(record.as_raw_fd(),libc::LOCK_SH)}!=0{return Err(std::io::Error::last_os_error().into());}
    let mut bytes=vec![];(&record).take(LIMIT+1).read_to_end(&mut bytes)?;if bytes.len() as u64>LIMIT{return Err("resolved model record too large".into());}
    verify(&path,&serde_json::from_slice(&bytes)?)?;
    let descriptor=record.as_raw_fd();let flags=unsafe{libc::fcntl(descriptor,libc::F_GETFD)};
    if flags<0||unsafe{libc::fcntl(descriptor,libc::F_SETFD,flags&!libc::FD_CLOEXEC)}<0{return Err(std::io::Error::last_os_error().into());}
    // Native children keep the same shared hold if the HTTP owner dies.
    Ok(HeldModel{path,_record:Some(record)})
}
#[cfg(test)]
mod tests{
    use super::*;
    use serde_json::json;
    use std::{process::{Child,Command},sync::atomic::{AtomicU64,Ordering}};
    static NEXT:AtomicU64=AtomicU64::new(0);
    // Inheritable descriptors must not leak into a sibling fixture child.
    static SERIAL:std::sync::Mutex<()>=std::sync::Mutex::new(());
    // A shared hold intentionally reaches EVERY child. Run each fixture in
    // its own test process so unrelated parallel replay/compiler children
    // cannot inherit this fixture's descriptor and outlive its owned child.
    fn isolated_case(name:&str)->bool{
        if std::env::var("JUSTFIT_MODEL_HOLD_TEST_CHILD").ok().as_deref()==Some(name){return false;}
        let output=Command::new(std::env::current_exe().unwrap()).args(["--exact",&format!("model_hold::tests::{name}"),"--test-threads=1"]).env("JUSTFIT_MODEL_HOLD_TEST_CHILD",name).output().unwrap();
        assert!(output.status.success(),"isolated hold case {name}: {}\n{}",String::from_utf8_lossy(&output.stdout),String::from_utf8_lossy(&output.stderr));true
    }

    struct Fixture{root:PathBuf,assembly:PathBuf,source:PathBuf,selection:PathBuf}
    impl Fixture{
        fn new()->Self{
            let root=std::env::temp_dir().join(format!("justfit-assembly-hold-{}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));fs::create_dir_all(&root).unwrap();let root=root.canonicalize().unwrap();let assembly=root.join("models/.resolved/fixture");fs::create_dir_all(assembly.join("target")).unwrap();let source=root.join("source.bin");fs::write(&source,b"source model bytes").unwrap();std::os::unix::fs::symlink(&source,assembly.join("target/model.safetensors")).unwrap();let meta=fs::metadata(&source).unwrap();
            let record=json!({"version":1,"model":"owner/source","family":"qwen3_8","target_format":"mlx-affine","vision_format":"none","sources":{"target":{"repo":"owner/source","revision":null},"draft":{"repo":"owner/draft","revision":null}},"files":{"target/model.safetensors":{"path":source,"bytes":meta.len(),"mtime_ns":nanoseconds(&meta,true),"ctime_ns":nanoseconds(&meta,false),"digest":"a".repeat(64)}}});fs::write(assembly.join("model.json"),serde_json::to_vec(&record).unwrap()).unwrap();let selection=root.join("selection");std::os::unix::fs::symlink(&assembly,&selection).unwrap();Self{root,assembly,source,selection}
        }
        fn gc_can_lock(&self)->bool{let file=fs::File::open(self.assembly.join("model.json")).unwrap();unsafe{libc::flock(file.as_raw_fd(),libc::LOCK_EX|libc::LOCK_NB)==0}}
    }
    impl Drop for Fixture{fn drop(&mut self){fs::remove_dir_all(&self.root).unwrap();}}
    struct Owned(Child);
    impl Drop for Owned{fn drop(&mut self){let _=self.0.kill();let _=self.0.wait();}}
    #[test]
    fn source_assembly_is_held_against_collection_until_drop(){
        if isolated_case("source_assembly_is_held_against_collection_until_drop"){return;}
        let _serial=SERIAL.lock().unwrap_or_else(|poisoned|poisoned.into_inner());
        let fixture=Fixture::new();assert!(fixture.gc_can_lock());let held=hold(&fixture.selection).unwrap();assert_eq!(held.path,fixture.assembly);assert!(!fixture.gc_can_lock());drop(held);assert!(fixture.gc_can_lock());assert_eq!(fs::read(&fixture.source).unwrap(),b"source model bytes");
    }
    #[test]
    fn native_child_inherits_hold_after_parent_releases_it(){
        if isolated_case("native_child_inherits_hold_after_parent_releases_it"){return;}
        let _serial=SERIAL.lock().unwrap_or_else(|poisoned|poisoned.into_inner());
        let fixture=Fixture::new();let held=hold(&fixture.selection).unwrap();let mut child=Owned(Command::new("/bin/sleep").arg("30").spawn().unwrap());drop(held);assert!(!fixture.gc_can_lock());child.0.kill().unwrap();child.0.wait().unwrap();assert!(fixture.gc_can_lock());
    }
    #[test]
    fn changed_source_or_foreign_record_is_rejected_without_rewriting_owner_files(){
        if isolated_case("changed_source_or_foreign_record_is_rejected_without_rewriting_owner_files"){return;}
        let _serial=SERIAL.lock().unwrap_or_else(|poisoned|poisoned.into_inner());
        let fixture=Fixture::new();fs::write(&fixture.source,b"changed bytes").unwrap();assert!(hold(&fixture.selection).is_err());assert_eq!(fs::read(&fixture.source).unwrap(),b"changed bytes");let owner=fixture.root.join("owner");fs::create_dir(&owner).unwrap();fs::copy(fixture.assembly.join("model.json"),owner.join("model.json")).unwrap();assert!(hold(&owner).is_err());assert!(fixture.gc_can_lock());
    }
    #[test]
    fn legacy_directories_stay_unheld_and_orphan_managed_assembly_is_refused(){
        if isolated_case("legacy_directories_stay_unheld_and_orphan_managed_assembly_is_refused"){return;}
        let _serial=SERIAL.lock().unwrap_or_else(|poisoned|poisoned.into_inner());
        let fixture=Fixture::new();let legacy=fixture.root.join("legacy");fs::create_dir(&legacy).unwrap();let held=hold(&legacy).unwrap();assert_eq!(held.path,legacy);assert!(held._record.is_none());let orphan=fixture.root.join("models/.resolved/orphan");fs::create_dir(&orphan).unwrap();assert!(hold(&orphan).is_err());
    }
}
