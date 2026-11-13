//! Pinned native-worker ANE packages. No exporter, interpreter or compiler runs here.
use crate::{cli,download::{self,Result,Stage}};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{collections::{BTreeMap,BTreeSet},fs,io::{Read,Write},os::{fd::AsRawFd,unix::fs::OpenOptionsExt},path::{Path,PathBuf}};
pub const NAMESPACE:&str="org.justfit.vision.aaa0bdc9ed86239e";
const ARCHIVE_LIMIT:u64=2*1024*1024*1024;
const EXPANDED_LIMIT:u64=4*1024*1024*1024;
#[derive(Clone)]
pub struct Identity{pub chip:String,pub os_build:String}
pub(crate) fn identity()->Result<Identity>{
    if !cfg!(all(target_os="macos",target_arch="aarch64")){return Err("native ANE packages require macOS arm64".into());}
    let read=|program:&str,args:&[&str]|->Result<String>{let out=std::process::Command::new(program).args(args).output()?;if !out.status.success(){return Err("cannot identify this Mac".into());}Ok(String::from_utf8(out.stdout)?.trim().into())};
    Ok(Identity{chip:read("/usr/sbin/sysctl",&["-n","machdep.cpu.brand_string"] )?,os_build:read("/usr/bin/sw_vers",&["-buildVersion"])?})
}
fn path_name(name:&str)->bool{crate::model_install::safe_path(name)}
fn safe(path:&Path)->Result<()> {
    for parent in path.ancestors(){match fs::symlink_metadata(parent){Ok(meta) if meta.file_type().is_symlink()=>return Err("symlink in ANE installation destination".into()),Err(error) if error.kind()!=std::io::ErrorKind::NotFound=>return Err(error.into()),_=>()}}
    Ok(())
}
fn regular_sha(path:&Path)->Result<String>{safe(path)?;let mut file=fs::OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW|libc::O_NONBLOCK).open(path)?;if !file.metadata()?.is_file(){return Err("ANE input is not a regular file".into());}let mut hash=Sha256::new();let mut buf=[0u8;65536];loop{let n=file.read(&mut buf)?;if n==0{break;}hash.update(&buf[..n]);}Ok(format!("{:x}",hash.finalize()))}
fn json_file(path:&Path,limit:u64)->Result<Value>{safe(path)?;let file=fs::OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW|libc::O_NONBLOCK).open(path)?;if !file.metadata()?.is_file()||file.metadata()?.len()>limit{return Err("invalid ANE JSON metadata size or type".into());}let mut bytes=vec![];file.take(limit+1).read_to_end(&mut bytes)?;if bytes.len() as u64>limit{return Err("ANE JSON metadata exceeds size limit".into());}Ok(serde_json::from_slice(&bytes)?)}
fn file_set(root:&Path)->Result<BTreeMap<String,String>>{
    fn walk(root:&Path,current:&Path,files:&mut BTreeMap<String,String>)->Result<()>{safe(current)?;for entry in fs::read_dir(current)?{let path=entry?.path();let meta=fs::symlink_metadata(&path)?;if meta.is_dir(){walk(root,&path,files)?;}else{files.insert(path.strip_prefix(root)?.to_str().ok_or("non-UTF8 ANE path")?.into(),regular_sha(&path)?);}}Ok(())}
    let mut files=BTreeMap::new();walk(root,root,&mut files)?;Ok(files)
}
fn exclusive(source:&Path,target:&Path)->Result<()> {
    use std::os::unix::ffi::OsStrExt;let a=std::ffi::CString::new(source.as_os_str().as_bytes())?;let b=std::ffi::CString::new(target.as_os_str().as_bytes())?;
    #[cfg(target_os="macos")]
    let code=unsafe{libc::renamex_np(a.as_ptr(),b.as_ptr(),libc::RENAME_EXCL)};
    #[cfg(target_os="linux")]
    let code=unsafe{libc::renameat2(libc::AT_FDCWD,a.as_ptr(),libc::AT_FDCWD,b.as_ptr(),libc::RENAME_NOREPLACE)};
    #[cfg(any(target_os="macos",target_os="linux"))]
    if code!=0{return Err(std::io::Error::last_os_error().into());}
    #[cfg(not(any(target_os="macos",target_os="linux")))]
    return Err("exclusive ANE installation unsupported".into());
    Ok(())
}
struct Lock(fs::File);
impl Lock{fn take(root:&Path)->Result<Self>{let path=root.join(".justfit-install.lock");safe(&path)?;let file=fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).mode(0o600).custom_flags(libc::O_NOFOLLOW).open(path)?;if !file.metadata()?.is_file(){return Err("invalid ANE installation lock".into());}
        if unsafe{libc::flock(file.as_raw_fd(),libc::LOCK_EX)}!=0{return Err(std::io::Error::last_os_error().into());}Ok(Self(file))}}
impl Drop for Lock{fn drop(&mut self){let _=unsafe{libc::flock(self.0.as_raw_fd(),libc::LOCK_UN)};}}
fn groups(meta:&Value)->Result<&Vec<Value>>{meta["groups"].as_array().filter(|rows|rows.len()==9).ok_or_else(||"ANE package requires nine groups".into())}
fn manifest_files(meta:&Value)->Result<BTreeMap<String,String>>{Ok(serde_json::from_value(meta["files"].clone())?)}
pub fn validate(meta:&Value,node:&str,current:&Identity,vision_sha:&str)->Result<()> {
    let graphs=meta["schema"].as_u64()==Some(3)&&meta["kind"]=="justfit-ane-graphs";
    if !["m4","m5","m6"].contains(&node)||(!graphs&&(meta["schema"].as_u64()!=Some(2)||meta["kind"]!="justfit-ane-vision"))||meta["architecture"]!="arm64"{return Err("unsupported native ANE package format or node".into());}
    if meta["node"]!=node||meta["chip"]!=current.chip||(!graphs&&meta["os_build"]!=current.os_build){return Err("ANE package does not match this chip and macOS build".into());}
    if !matches!(meta["capacity"].as_u64(),Some(4096|8192))||meta["vision_weights_sha256"]!=vision_sha{return Err("ANE package capacity or vision weights mismatch".into());}
    if meta["worker_kind"]!="native"||meta["native_cache_abi"].as_u64()!=Some(1)||meta["cache_namespace"]!=NAMESPACE{return Err("native ANE worker ABI or namespace mismatch".into());}
    let files=manifest_files(meta)?;if !files.contains_key("positions.npy")||!files.contains_key("LICENSE")||files.len()>4096{return Err("invalid ANE package file manifest".into());}
    for (name,sha) in &files {
        if !path_name(name)||(graphs&&name.starts_with("compiled/")){return Err("unsafe ANE package path".into());}download::digest(sha)?;
        let parts:Vec<_>=name.split('/').collect();let grouped=parts.len()>=3&&["assets","compiled"].contains(&parts[0])&&(0..9).any(|i|parts[1]==format!("group-{i:02}"));
        if !["positions.npy","LICENSE","NOTICE"].contains(&name.as_str())&&!grouped{return Err("unexpected ANE package path".into());}
        let mut parent=Path::new(name);
        while let Some(path)=parent.parent(){if path.as_os_str().is_empty(){break;}
        if files.contains_key(path.to_str().ok_or("invalid ANE path")?){return Err("overlapping ANE package paths".into());}parent=path;}
    }
    for (index,row) in groups(meta)?.iter().enumerate(){
        let asset=row["asset_hash"].as_str().ok_or("missing ANE asset hash")?;download::digest(asset)?;
        if graphs{if row.as_object().is_none_or(|r|r.len()!=1){return Err("graph-only package must not claim compiled cache guards".into());}}
        else{let cache=row["cache_relative"].as_str().ok_or("missing ANE cache path")?;let parts:Vec<_>=cache.split('/').collect();
        if !path_name(cache)||parts.len()!=5||parts[0]!=current.os_build||parts[1]!=NAMESPACE||parts[2]!=asset||parts[3].len()!=64||!parts[3].bytes().all(|b|b.is_ascii_hexdigit())||parts[4]!="model.aimodelx"{return Err("unsafe or incompatible native ANE cache namespace".into());}}
        let mut prefixes=vec![format!("assets/group-{index:02}/model.aimodel/")];if !graphs{prefixes.push(format!("compiled/group-{index:02}/"));}
        for prefix in prefixes{if !files.keys().any(|n|n.starts_with(&prefix)){return Err("incomplete ANE package group".into());}}
    }
    download::digest(meta["reference_qualification_sha256"].as_str().ok_or("missing ANE qualification digest")?)?;
    let proof=&meta["reference_qualification"];
    if proof["passed"]!=true||proof["node"]!=node||!proof["worker_upper_bytes"].as_u64().is_some_and(|n|n>0&&n<=1127*1024*1024)||proof["phase_qualification"]!=json!({"prefill":node!="m6","decode":true}){return Err("missing device-class ANE qualification".into());}Ok(())
}
fn model_sha(root:&Path,model:&Path)->Result<String>{crate::vision_identity::sha(&root.join("source/build/justfit-engine"),model)}
fn cache_root()->Result<PathBuf>{Ok(PathBuf::from(std::env::var_os("HOME").ok_or("HOME missing")?).join("Library/Application Support/JustFit/native-home/Library/Application Support/coreai-cache"))}
fn subset(files:&BTreeMap<String,String>,prefix:&str)->BTreeMap<String,String>{files.iter().filter_map(|(n,v)|n.strip_prefix(prefix).map(|n|(n.into(),v.clone()))).collect()}
fn verify_graph_assets(destination:&Path,meta:&Value)->Result<()> {
    let files=manifest_files(meta)?;if file_set(&destination.join("assets"))?!=subset(&files,"assets/"){return Err("installed ANE assets changed; no files overwritten".into());}
    for (name,sha) in &files{if !name.starts_with("assets/")&&!name.starts_with("compiled/")&&regular_sha(&destination.join(name))?!=*sha{return Err("installed ANE metadata changed".into());}}
    for (index,row) in groups(meta)?.iter().enumerate(){let hash=fs::read(destination.join(format!("assets/group-{index:02}/model.aimodel/main.hash")))?.iter().map(|b|format!("{b:02x}")).collect::<String>();if row["asset_hash"]!=hash{return Err("installed ANE graph identity mismatch".into());}}
    Ok(())
}
fn verify_installed(destination:&Path,meta:&Value,cache:&Path)->Result<()> {
    verify_graph_assets(destination,meta)?;if meta["schema"]==3{return Ok(());}let files=manifest_files(meta)?;
    for (i,row) in groups(meta)?.iter().enumerate(){let path=cache.join(row["cache_relative"].as_str().ok_or("missing cache path")?);if file_set(&path)?!=subset(&files,&format!("compiled/group-{i:02}/")){return Err("installed ANE cache changed; no files overwritten".into());}}Ok(())
}
fn snapshot(source:&Path,target:&Path,sha:&str)->Result<()> {
    let mut input=fs::OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW|libc::O_NONBLOCK).open(source.canonicalize()?)?;
    if !input.metadata()?.is_file()||input.metadata()?.len()>ARCHIVE_LIMIT{return Err("ANE archive exceeds size limit or is not a regular file".into());}
    // CoW keeps a private immutable snapshot without another multi-GB disk copy.
    // Hash the snapshot itself, never trust a mutable source pathname or hard link.
    #[cfg(target_os="macos")]
    {
        use std::os::unix::ffi::OsStrExt;
        let parent=fs::File::open(target.parent().ok_or("ANE snapshot parent missing")?)?;
        let name=std::ffi::CString::new(target.file_name().ok_or("ANE snapshot filename missing")?.as_bytes())?;
        if unsafe{libc::fclonefileat(input.as_raw_fd(),parent.as_raw_fd(),name.as_ptr(),0)}==0{
            let output=fs::OpenOptions::new().read(true).write(true).custom_flags(libc::O_NOFOLLOW).open(target)?;
            if output.metadata()?.len()>ARCHIVE_LIMIT||regular_sha(target)?!=sha{return Err("ANE archive SHA256 mismatch".into());}
            output.set_permissions(std::os::unix::fs::PermissionsExt::from_mode(0o600))?;output.sync_all()?;return Ok(());
        }
        let error=std::io::Error::last_os_error();
        if !matches!(error.raw_os_error(),Some(libc::ENOTSUP|libc::EXDEV|libc::EINVAL|libc::ENOSYS)){return Err(error.into());}
    }
    let mut output=fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(target)?;let mut hash=Sha256::new();let mut total=0u64;let mut buf=[0u8;65536];
    loop{let n=input.read(&mut buf)?;if n==0{break;}total+=n as u64;if total>ARCHIVE_LIMIT{return Err("ANE archive exceeds size limit".into());}hash.update(&buf[..n]);output.write_all(&buf[..n])?;}
    if format!("{:x}",hash.finalize())!=sha{return Err("ANE archive SHA256 mismatch".into());}output.sync_all()?;Ok(())
}
fn copy_tree(source:&Path,target:&Path)->Result<()>{fs::create_dir(target)?;for entry in fs::read_dir(source)?{let path=entry?.path();safe(&path)?;let destination=target.join(path.file_name().ok_or("invalid ANE cache member")?);if path.is_dir(){copy_tree(&path,&destination)?;}else{regular_sha(&path)?;fs::copy(path,destination)?;}}Ok(())}
fn publish_cache(source:&Path,target:&Path)->Result<()> {
    safe(target)?;fs::create_dir_all(target.parent().ok_or("invalid native cache destination")?)?;
    if target.exists(){if file_set(target)?!=file_set(source)?{return Err("existing native ANE cache differs; no cache overwritten".into());}return Ok(());}
    match exclusive(source,target){Ok(())=>Ok(()),Err(error) if error.downcast_ref::<std::io::Error>().is_some_and(|e|e.raw_os_error()==Some(libc::EXDEV))=>{
        let stage=Stage::new(target.parent().ok_or("missing cache parent")?,"ane-cache-copy")?;let copied=stage.path().join("model.aimodelx");copy_tree(source,&copied)?;
        if file_set(&copied)?!=file_set(source)?{return Err("native ANE cache copy changed".into());}exclusive(&copied,target)
    },Err(error)=>Err(error)}
}
fn clean_dead_stages(assets:&Path)->Result<()>{
    for entry in fs::read_dir(assets)?{let path=entry?.path();let name=path.file_name().and_then(|s|s.to_str()).unwrap_or("");if !name.starts_with(".ane-install-")||!path.symlink_metadata()?.is_dir(){continue;}
        let owner=match json_file(&path.join("stage-owner.json"),8192){Ok(v)=>v,Err(_)=>continue};let Some(pid)=owner["pid"].as_i64().filter(|p|*p>0&&*p<=i32::MAX as i64)else{continue;};
        if owner["kind"]!="justfit-ane-stage"||owner["assets"]!=assets.to_str().unwrap_or("")||!name.starts_with(&format!(".ane-install-{pid}-")){continue;}
        if unsafe{libc::kill(pid as i32,0)}<0&&std::io::Error::last_os_error().raw_os_error()==Some(libc::ESRCH){fs::remove_dir_all(path)?;}
    }Ok(())
}
fn install_at(root:&Path,archive:&Path,sha:&str,node:&str,model:&Path,cache:&Path,current:&Identity)->Result<Value>{install_at_mode(&root.join("local-state/ane").join(node),archive,sha,node,model,InstallationDevice{root,cache,current},false)}
struct InstallationDevice<'a>{root:&'a Path,cache:&'a Path,current:&'a Identity}
fn install_at_mode(assets:&Path,archive:&Path,sha:&str,node:&str,model:&Path,device:InstallationDevice<'_>,graph_only:bool)->Result<Value>{
    let InstallationDevice{root,cache,current}=device;
    if !["m4","m5","m6"].contains(&node){return Err("invalid ANE node".into());}
    download::digest(sha)?;safe(assets)?;fs::create_dir_all(assets)?;let _lock=Lock::take(assets)?;
    clean_dead_stages(assets)?;let owned=Stage::new(assets,"ane-install")?;
    let owner=owned.path().join("stage-owner.json");let mut marker=fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(owner)?;marker.write_all(&serde_json::to_vec(&json!({"kind":"justfit-ane-stage","pid":std::process::id(),"assets":assets,"archive_sha256":sha}))?)?;marker.sync_all()?;let frozen=owned.path().join("archive.zip");snapshot(archive,&frozen,sha)?;let mut zip=zip::ZipArchive::new(fs::File::open(frozen)?)?;
    if zip.len()>4097{return Err("too many ANE archive members".into());}let mut members=BTreeMap::new();let mut expanded=0u64;
    for i in 0..zip.len(){let entry=zip.by_index(i)?;let name=entry.name()?.into_owned();if !path_name(&name)||members.insert(name.clone(),i).is_some()||!entry.unix_mode().is_some_and(|m|m&0o170000==0o100000){return Err("invalid, duplicate or non-regular ANE archive member".into());}expanded=expanded.checked_add(entry.size()).ok_or("ANE archive size overflow")?;if expanded>EXPANDED_LIMIT||(name=="bundle.json"&&entry.size()>512*1024){return Err("ANE archive exceeds expanded size limit".into());}}
    let index=*members.get("bundle.json").ok_or("ANE bundle metadata missing")?;let mut bytes=vec![];zip.by_index(index)?.take(512*1024+1).read_to_end(&mut bytes)?;if bytes.len()>512*1024{return Err("ANE metadata exceeds size limit".into());}let mut meta:Value=serde_json::from_slice(&bytes)?;
    let archive_files=manifest_files(&meta)?;
    if graph_only&&meta["schema"]==2{
        let source_identity=Identity{chip:current.chip.clone(),os_build:meta["os_build"].as_str().ok_or("source ANE OS missing")?.into()};validate(&meta,node,&source_identity,&model_sha(root,model)?)?;
        meta["schema"]=json!(3);meta["kind"]=json!("justfit-ane-graphs");meta["source_native_bundle_sha256"]=json!(sha);
        meta["files"].as_object_mut().ok_or("source graph manifest missing")?.retain(|name,_|!name.starts_with("compiled/"));
        for group in meta["groups"].as_array_mut().ok_or("source graph groups missing")?{let asset=group["asset_hash"].clone();*group=json!({"asset_hash":asset});}
    }
    validate(&meta,node,current,&model_sha(root,model)?)?;let files=manifest_files(&meta)?;
    if members.keys().map(String::as_str).collect::<BTreeSet<_>>()!=archive_files.keys().map(String::as_str).chain(std::iter::once("bundle.json")).collect(){return Err("unexpected ANE archive member".into());}
    safe(cache)?;fs::create_dir_all(cache)?;let destination=assets.join(meta["capacity"].as_u64().ok_or("missing capacity")?.to_string());safe(&destination)?;
    if destination.exists(){if json_file(&destination.join("bundle.json"),512*1024)?!=meta{return Err("existing ANE tier differs; use a new workspace".into());}verify_installed(&destination,&meta,cache)?;return Ok(meta);}
    let stage=owned.path().join("stage");fs::create_dir(&stage)?;
    for (name,sha) in &files {
        let target=stage.join(name);fs::create_dir_all(target.parent().ok_or("invalid ANE member")?)?;let mut input=zip.by_index(members[name])?;let declared=input.size();let mut output=fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&target)?;let mut total=0;let mut hash=Sha256::new();let mut buffer=[0u8;65536];
        loop{let n=input.read(&mut buffer)?;if n==0{break;}total+=n as u64;if total>declared{return Err("ANE archive member exceeds declared size".into());}hash.update(&buffer[..n]);output.write_all(&buffer[..n])?;}
        if total!=declared||format!("{:x}",hash.finalize())!=*sha{return Err("ANE archive member SHA256 mismatch".into());}output.sync_all()?;
    }
    for (i,row) in groups(&meta)?.iter().enumerate(){let asset=stage.join(format!("assets/group-{i:02}/model.aimodel/main.hash"));let hash=fs::read(asset)?.iter().map(|v|format!("{v:02x}")).collect::<String>();if row["asset_hash"]!=hash{return Err("ANE graph identity mismatch".into());}
        if meta["schema"]!=3{let source=stage.join(format!("compiled/group-{i:02}"));publish_cache(&source,&cache.join(row["cache_relative"].as_str().ok_or("cache path missing")?))?;}}
    if meta["schema"]!=3{fs::remove_dir_all(stage.join("compiled"))?;}fs::write(stage.join("bundle.json"),serde_json::to_vec_pretty(&meta)?)?;exclusive(&stage,&destination)?;verify_installed(&destination,&meta,cache)?;Ok(meta)
}
pub fn install(root:&Path,archive:&Path,sha:&str,node:&str,model:&Path)->Result<Value>{if !["m4","m5","m6"].contains(&node){return Err("invalid ANE node".into());}let root=root.canonicalize()?;cli::verify(&root)?;install_at(&root,archive,sha,node,&model.canonicalize()?,&cache_root()?,&identity()?)}
fn immutable_json(path:&Path,value:&Value)->Result<()> {
    safe(path)?;if path.exists(){if json_file(path,4*1024*1024)?!=*value{return Err("installed ANE evidence changed; use a new workspace".into());}return Ok(());}
    let parent=path.parent().ok_or("missing ANE metadata parent")?;fs::create_dir_all(parent)?;let stage=Stage::new(parent,"ane-binding")?;let temp=stage.path().join("metadata");let mut file=fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&temp)?;file.write_all(&serde_json::to_vec_pretty(value)?)?;file.sync_all()?;drop(file);exclusive(&temp,path)
}
fn placement(compiled:&Path)->Result<Value>{
    let mut manifests=vec![];for entry in fs::read_dir(compiled.join("main-this-delegates/MPSGraph"))?{let path=entry?.path().join("manifest.plist");if path.exists(){manifests.push(path);}}
    if manifests.len()!=1{return Err("native ANE placement absent or ambiguous".into());}safe(&manifests[0])?;
    crate::ane_native::placement_manifest(&manifests[0]).map_err(|error|error.1.into())
}
fn bind_at(root:&Path,node:&str,model:&Path,cache:&Path,current:&Identity)->Result<Option<PathBuf>>{bind_at_location(root,&root.join("local-state/ane").join(node),node,model,cache,current)}
fn bind_at_location(root:&Path,assets:&Path,node:&str,model:&Path,cache:&Path,current:&Identity)->Result<Option<PathBuf>> {
    if !["m4","m5","m6"].contains(&node){return Err("invalid ANE node".into());}
    if ![4096,8192].iter().all(|n|assets.join(n.to_string()).join("bundle.json").is_file()){return Ok(None);}
    safe(assets)?;let _lock=Lock::take(assets)?;let library=root.join("vision-tools/libcoreai_bridge.dylib");let library_sha=regular_sha(&library)?;let mut records=serde_json::Map::new();let mut leaves=vec![];
    for capacity in [4096,8192]{
        let tier=assets.join(capacity.to_string());let meta=json_file(&tier.join("bundle.json"),512*1024)?;validate(&meta,node,current,&model_sha(root,model)?)?;verify_graph_assets(&tier,&meta)?;
        let expected=manifest_files(&meta)?;
        if meta["schema"]!=3{for (index,row) in groups(&meta)?.iter().enumerate(){let compiled=cache.join(row["cache_relative"].as_str().ok_or("missing cache path")?);if compiled.exists()&&file_set(&compiled)?!=subset(&expected,&format!("compiled/group-{index:02}/")){return Err("installed ANE cache changed; no files overwritten".into());}}}
        let mut rows=vec![];
        for row in groups(&meta)?{if meta["schema"]==3{rows.push(json!({"asset_hash":row["asset_hash"]}));continue;}let compiled=cache.join(row["cache_relative"].as_str().ok_or("missing cache path")?);
            // Only validated graph pins are required to start GPU service and
            // prepare the native cache in the background. Missing programs
            // are never represented as a successful cached-load qualification.
            if !compiled.exists(){rows.push(json!({"asset_hash":row["asset_hash"]}));continue;}
            let guard=placement(&compiled)?;let mut files=vec![];
            for name in file_set(&compiled)?.keys(){let path=compiled.join(name);let stat=fs::metadata(&path)?;let ns=stat.modified()?.duration_since(std::time::UNIX_EPOCH)?.as_nanos();files.push(json!([path,stat.len(),u64::try_from(ns)?]));}
            rows.push(json!({"asset_hash":row["asset_hash"],"guard":{"path":guard["manifest_path"],"manifest_sha256":guard["manifest_sha256"]},"files":files}));
        }
        let catalog=tier.join("native-assets-catalog.json");if catalog.exists(){let record=json_file(&catalog,4*1024*1024)?;if record["build"]!=current.os_build||groups(&record)?.iter().zip(groups(&meta)?).any(|(a,b)|a["asset_hash"]!=b["asset_hash"]){return Err("installed ANE graph catalog differs; no metadata overwritten".into());}}else{immutable_json(&catalog,&json!({"build":current.os_build,"groups":rows}))?;}
        records.insert(capacity.to_string(),json!({"reference_qualification":meta["reference_qualification"],"reference_qualification_sha256":meta["reference_qualification_sha256"],"bundle_sha256":regular_sha(&tier.join("bundle.json"))?,"catalog_sha256":regular_sha(&catalog)?}));
        let mut pins=serde_json::Map::new();for (name,sha) in file_set(&tier.join("assets"))?{pins.insert(tier.join("assets").join(name).to_str().ok_or("invalid ANE asset path")?.into(),json!(sha));}
        for path in [tier.join("positions.npy"),catalog,library.clone()]{pins.insert(path.to_str().ok_or("invalid ANE pin path")?.into(),json!(regular_sha(&path)?));}
        if pins.get(library.to_str().ok_or("invalid library path")?)!=Some(&json!(library_sha)){return Err("CoreAI library changed during profile binding".into());}
        leaves.push((capacity.to_string()+".json",json!({"schema":2,"kind":"grouped-vision","native_cache_abi":1,"group_count":9,"capacity_raw_patches":capacity,"worker_memory_bytes":1127u64*1024*1024,"worker_timeout_seconds":600,"startup_timeout_seconds":120,"vision_weights_sha256":meta["vision_weights_sha256"],"assets":tier.join("assets"),"position_table":tier.join("positions.npy"),"cache_record":tier.join("native-assets-catalog.json"),"route_policy":"gpu_relief","phase_qualification":meta["reference_qualification"]["phase_qualification"],"qualification_status":"operator_reviewed","files":pins})));
    }
    let evidence=assets.join("installed-native-qualification.json");immutable_json(&evidence,&json!({"schema":1,"node":node,"native_cache_abi":1,"vision_weights_sha256":model_sha(root,model)?,"tiers":records,"scope":"Pinned graph qualification and file identity; actual worker-owned cache compilation and placement are validated separately"}))?;
    let destination=assets.join("qualified");let mut tiers=vec![];
    for (name,mut leaf) in leaves{leaf["files"][evidence.to_str().ok_or("invalid qualification path")?]=json!(regular_sha(&evidence)?);let path=destination.join(&name);immutable_json(&path,&leaf)?;tiers.push(json!({"profile":name,"sha256":regular_sha(&path)?}));}
    let top=destination.join("tiered.json");immutable_json(&top,&json!({"schema":3,"kind":"tiered-vision","tiers":tiers,"gpu_only_capacities":[12288,14400,16384]}))?;Ok(Some(top))
}
pub fn prepare_profile(root:&Path,node:&str,model:&Path)->Result<Option<PathBuf>>{bind_at(&root.canonicalize()?,node,&model.canonicalize()?,&cache_root()?,&identity()?)}
pub(crate) fn install_delivery(root:&Path,assets:&Path,archive:&Path,sha:&str,node:&str,model:&Path,graphs:bool)->Result<Value>{cli::verify(root)?;if !assets.starts_with(root.join("local-state/ane-default")){return Err("default ANE assets outside owned workspace".into());}install_at_mode(assets,archive,sha,node,model,InstallationDevice{root,cache:&cache_root()?,current:&identity()?},graphs)}
pub(crate) fn bind_delivery(root:&Path,assets:&Path,node:&str,model:&Path)->Result<Option<PathBuf>>{bind_at_location(root,assets,node,model,&cache_root()?,&identity()?)}
pub(crate) fn publish_delivery(path:&Path,value:&Value)->Result<()>{immutable_json(path,value)}
struct DownloadTarget<'a>{root:&'a Path,node:&'a str,model:&'a Path,cache:&'a Path,current:&'a Identity,assets:Option<&'a Path>,graph_only:bool}
fn resumable_archive(root:&Path,row:&Value,agent:&ureq::Agent)->Result<PathBuf>{
    let sha=row["sha256"].as_str().ok_or("default ANE archive digest missing")?;download::digest(sha)?;let bytes=row["bytes"].as_u64().filter(|v|*v>0&&*v<=ARCHIVE_LIMIT).ok_or("default ANE download requires a bounded byte count")?;
    let cache=root.join("local-state/ane-downloads").join(sha);safe(&cache)?;fs::create_dir_all(&cache)?;let _lock=Lock::take(&cache)?;let archive=cache.join("archive.zip");let receipt=cache.join("owner.json");let expected=json!({"kind":"justfit-ane-download","sha256":sha,"bytes":bytes});
    if receipt.exists(){if json_file(&receipt,8192)?!=expected{return Err("foreign ANE download owner receipt; no data overwritten".into());}}else{if archive.exists(){return Err("ANE partial has no owner receipt; preserved".into());}immutable_json(&receipt,&expected)?;}
    download::fetch_resumable_with(agent,row["url"].as_str().ok_or("ANE archive URL missing")?,&archive,bytes,sha,None)?;Ok(archive)
}
fn fetch_at(target:&DownloadTarget<'_>,url:&str,sha:&str,agent:&ureq::Agent)->Result<Option<PathBuf>>{
    let DownloadTarget{root,node,model,cache,current,assets,graph_only}=*target;
    download::digest(sha)?;let stage=Stage::new(root,"ane-download")?;let index=stage.path().join("index.json");download::fetch_with(agent,url,&index,512*1024,sha)?;let data=json_file(&index,512*1024)?;if data["schema"].as_u64()!=Some(1)||data["kind"]!="justfit-ane-downloads"{return Err("unknown native ANE download index".into());}
    let mut rows:Vec<_>=data["artifacts"].as_array().ok_or("missing ANE downloads")?.iter().filter(|row|row["node"]==node&&row["chip"]==current.chip&&(graph_only||row["delivery"]=="graphs"||row["os_build"]==current.os_build)).collect();
    // An index may carry the same chip's tiers for several macOS builds: take
    // this build's set when it has one, else (graphs only) the first build's.
    if graph_only{let builds:BTreeSet<_>=rows.iter().filter_map(|row|row["os_build"].as_str()).collect();let pick=builds.get(current.os_build.as_str()).copied().or_else(||builds.iter().next().copied());if let Some(build)=pick{rows.retain(|row|row["os_build"].as_str().is_none_or(|b|b==build));}}
    if rows.len()!=2||rows.iter().filter_map(|row|row["capacity"].as_u64()).collect::<BTreeSet<_>>()!=BTreeSet::from([4096,8192]){return Err("no complete compatible native 4K/8K ANE download set".into());}
    // Validate both download declarations before opening a connection or installing.
    for row in &rows{download::digest(row["sha256"].as_str().ok_or("ANE archive pin missing")?)?;if !row["url"].as_str().is_some_and(|url|url.starts_with("https://")){return Err("ANE downloads require HTTPS".into());}}
    let mut archives=vec![];
    for row in rows{
        let pin=row["sha256"].as_str().ok_or("ANE archive pin missing")?;let archive=if assets.is_some(){resumable_archive(root,row,agent)?}else{let archive=stage.path().join(format!("{}.zip",row["capacity"]));download::fetch_with(agent,row["url"].as_str().ok_or("ANE download URL missing")?,&archive,ARCHIVE_LIMIT,pin)?;archive};
        let mut zip=zip::ZipArchive::new(fs::File::open(&archive)?)?;let mut bytes=vec![];zip.by_name("bundle.json")?.take(512*1024+1).read_to_end(&mut bytes)?;if bytes.len()>512*1024{return Err("ANE metadata exceeds size limit".into());}let meta:Value=serde_json::from_slice(&bytes)?;
        let source_identity=Identity{chip:current.chip.clone(),os_build:if graph_only&&meta["schema"]==2{meta["os_build"].as_str().ok_or("source OS missing")?.into()}else{current.os_build.clone()}};validate(&meta,node,&source_identity,&model_sha(root,model)?)?;if meta["capacity"]!=row["capacity"]{return Err("ANE download tier mismatch".into());}
        archives.push((archive,pin.to_string()));
    }
    // No tier is installed until both transfers and metadata checks have succeeded.
    for (archive,pin) in &archives{if let Some(assets)=assets{install_at_mode(assets,archive,pin,node,model,InstallationDevice{root,cache,current},graph_only)?;}else{install_at(root,archive,pin,node,model,cache,current)?;}}
    let result=if let Some(assets)=assets{bind_at_location(root,assets,node,model,cache,current)}else{bind_at(root,node,model,cache,current)}?;
    if assets.is_some(){for (archive,_) in archives{if archive.starts_with(root.join("local-state/ane-downloads")){fs::remove_file(archive)?;}}}Ok(result)
}
pub fn fetch(root:&Path,url:&str,sha:&str,node:&str,model:&Path)->Result<Option<PathBuf>>{
    if !["m4","m5","m6"].contains(&node){return Err("invalid ANE node".into());}
    let root=root.canonicalize()?;cli::verify(&root)?;let current=identity()?;let cache=cache_root()?;
    fetch_at(&DownloadTarget{root:&root,node,model,cache:&cache,current:&current,assets:None,graph_only:false},url,sha,&download::agent())
}
pub(crate) fn fetch_delivery(root:&Path,assets:&Path,url:&str,sha:&str,node:&str,model:&Path,graphs:bool)->Result<Option<PathBuf>>{cli::verify(root)?;if !assets.starts_with(root.join("local-state/ane-default")){return Err("default ANE delivery destination outside workspace".into());}let current=identity()?;let cache=cache_root()?;fetch_at(&DownloadTarget{root,node,model,cache:&cache,current:&current,assets:Some(assets),graph_only:graphs},url,sha,&download::agent())}
pub fn validate_options(bundles:&[String],pins:&[String],manifest:Option<&str>,manifest_pin:Option<&str>)->Result<()> {
    if bundles.len()!=pins.len()||manifest.is_some()!=manifest_pin.is_some()||manifest.is_some()&&!bundles.is_empty(){return Err("supply matching ANE bundle pins or one pinned manifest".into());}
    for pin in pins{download::digest(pin)?;}
    if let Some(pin)=manifest_pin{download::digest(pin)?;}Ok(())
}
pub fn install_options(root:&Path,node:&str,model:&Path,bundles:&[String],pins:&[String],manifest:Option<&str>,manifest_pin:Option<&str>)->Result<Option<PathBuf>> {
    validate_options(bundles,pins,manifest,manifest_pin)?;
    if let Some(url)=manifest{return fetch(root,url,manifest_pin.ok_or("missing ANE manifest pin")?,node,model);}
    for (archive,pin) in bundles.iter().zip(pins){install(root,Path::new(archive),pin,node,model)?;}
    prepare_profile(root,node,model)
}
pub fn run_cli(args:&[String])->Result<()> {
    if args.first().map(String::as_str)!=Some("install"){return Err("usage: justfit ane install [--project PATH] [--node m4|m5|m6] [--model MODEL] (--ane-bundle ZIP --ane-sha256 SHA ... | --ane-manifest HTTPS_URL --ane-manifest-sha256 SHA)".into());}
    let mut flags=BTreeMap::new();let mut bundles=vec![];let mut pins=vec![];let mut index=1;
    while index<args.len(){let flag=args[index].as_str();let value=args.get(index+1).ok_or("ANE option needs a value")?;match flag{"--ane-bundle"=>bundles.push(value.clone()),"--ane-sha256"=>pins.push(value.clone()),"--project"|"--node"|"--model"|"--revision"|"--ane-manifest"|"--ane-manifest-sha256"=>{if flags.insert(flag,value.as_str()).is_some(){return Err("duplicate ANE option".into());}},_=>return Err(format!("unknown ANE option: {flag}").into())}index+=2;}
    let manifest=flags.get("--ane-manifest").copied();let manifest_pin=flags.get("--ane-manifest-sha256").copied();validate_options(&bundles,&pins,manifest,manifest_pin)?;
    identity()?;
    let root=match flags.get("--project"){Some(path)=>std::path::absolute(path)?,None=>crate::cli::default_project()?};
    if !root.exists(){cli::initialize(&cli::payload_root()?,&root)?;}
    let root=root.canonicalize()?;cli::verify(&root)?;
    let node=match flags.get("--node"){Some(node)=>*node,None=>crate::cli::detect_preset()?};if !["m4","m5","m6"].contains(&node){return Err("invalid ANE node".into());}
    let held=crate::model_install::resolve_for_serve_options(&root,flags.get("--model").copied().unwrap_or(crate::model_install::DEFAULT_MODEL),flags.get("--revision").copied(),None,None,false)?;let model=held.path.clone();
    let profile=install_options(&root,node,&model,&bundles,&pins,manifest,manifest_pin)?;
    println!("{}",json!({"installed":true,"vision_profile":profile,"note":"both 4K and 8K tiers are required for the automatic profile","installer":"rust"}));Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    const PLIST:&str=r#"<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd"><plist version="1.0"><dict><key>Package Version</key><dict><key>1</key><dict><key>Optimized Modules</key><dict><key>group-module</key><dict><key>Entry Function Attributes</key><dict><key>group</key><array><string>mps.fullyPlacedOnANE</string><string>mps.noGPUActivity</string><dict><key>mps.ane.calleeName</key><string>toy-opaque-never-loaded</string></dict></array></dict></dict></dict></dict></dict></dict></plist>"#;
    struct Fixture{_owned:Stage,root:PathBuf,project:PathBuf,model:PathBuf,cache:PathBuf,current:Identity}
    impl Fixture{fn new()->Self{
        let owned=Stage::new(&std::env::temp_dir(),"justfit-native-ane-package-test").unwrap();let root=owned.path().canonicalize().unwrap();let project=root.join("project");let model=root.join("model");let cache=root.join("Library/Application Support/JustFit/native-home/Library/Application Support/coreai-cache");
        fs::create_dir_all(project.join("vision-tools")).unwrap();fs::write(project.join("vision-tools/libcoreai_bridge.dylib"),b"never loaded: fixture library pin only").unwrap();fs::write(project.join("package-manifest.json"),serde_json::to_vec(&json!({"vision-tools/libcoreai_bridge.dylib":regular_sha(&project.join("vision-tools/libcoreai_bridge.dylib")).unwrap()})).unwrap()).unwrap();fs::create_dir_all(model.join("vision")).unwrap();fs::write(model.join("vision/model.bin"),b"fixture vision weights: never executed").unwrap();
        Self{_owned:owned,root,project,model,cache,current:identity().unwrap()}
    }
    fn archive(&self,capacity:u64)->(PathBuf,String,Value){
        let mut files=BTreeMap::<String,Vec<u8>>::from([("positions.npy".into(),b"fixture positions never executed".to_vec()),("LICENSE".into(),b"fixture license".to_vec())]);let mut groups=vec![];
        for i in 0..9{let signature=Sha256::digest(format!("fixture-{capacity}-{i}"));let asset=format!("{signature:x}");let specialization="a".repeat(64);
            files.insert(format!("assets/group-{i:02}/model.aimodel/main.hash"),signature.to_vec());files.insert(format!("assets/group-{i:02}/model.aimodel/program.mil"),b"opaque fixture, never compiled".to_vec());
            files.insert(format!("compiled/group-{i:02}/main-this-delegates/MPSGraph/fixture/manifest.plist"),PLIST.as_bytes().to_vec());files.insert(format!("compiled/group-{i:02}/program.bin"),b"opaque fixture, never loaded".to_vec());
            groups.push(json!({"asset_hash":asset,"cache_relative":format!("{}/{NAMESPACE}/{asset}/{specialization}/model.aimodelx",self.current.os_build)}));
        }
        let manifest:BTreeMap<_,_>=files.iter().map(|(n,bytes)|(n.clone(),format!("{:x}",Sha256::digest(bytes)))).collect();
        let meta=json!({"schema":2,"kind":"justfit-ane-vision","architecture":"arm64","node":"m4","chip":self.current.chip,"os_build":self.current.os_build,"capacity":capacity,"vision_weights_sha256":cli::digest(&self.model.join("vision/model.bin")).unwrap(),"worker_kind":"native","native_cache_abi":1,"cache_namespace":NAMESPACE,"files":manifest,"groups":groups,"reference_qualification_sha256":"b".repeat(64),"reference_qualification":{"passed":true,"node":"m4","worker_upper_bytes":1127u64*1024*1024,"phase_qualification":{"prefill":true,"decode":true}}});
        let archive=self.root.join(format!("{capacity}.zip"));let mut zip=zip::ZipWriter::new(fs::File::create(&archive).unwrap());let options=zip::write::SimpleFileOptions::default().unix_permissions(0o600).compression_method(zip::CompressionMethod::Deflated);
        zip.start_file("bundle.json",options).unwrap();zip.write_all(&serde_json::to_vec(&meta).unwrap()).unwrap();for (name,bytes) in files{zip.start_file(name,options).unwrap();zip.write_all(&bytes).unwrap();}zip.finish().unwrap();let sha=regular_sha(&archive).unwrap();(archive,sha,meta)
    }}
    #[test]
    fn native_metadata_rejects_wrong_device_namespace_paths_and_qualification(){
        let fixture=Fixture::new();let (_,_,meta)=fixture.archive(4096);let vision=regular_sha(&fixture.model.join("vision/model.bin")).unwrap();validate(&meta,"m4",&fixture.current,&vision).unwrap();
        for n in 0..12{let mut bad=meta.clone();match n{
            0=>bad["schema"]=json!(1),1=>bad["capacity"]=json!(true),2=>bad["os_build"]=json!("different"),3=>bad["chip"]=json!("different"),4=>bad["cache_namespace"]=json!("org.python.python"),5=>bad["native_cache_abi"]=json!(true),6=>bad["vision_weights_sha256"]=json!("f".repeat(64)),7=>bad["groups"][0]["cache_relative"]=json!("../../owner"),8=>{bad["files"]["../owner"]=json!("a".repeat(64));},9=>bad["reference_qualification"]["phase_qualification"]["prefill"]=json!(false),10=>bad["reference_qualification"]["worker_upper_bytes"]=json!(1128u64*1024*1024),_=>{bad["groups"].as_array_mut().unwrap().pop();}}
            assert!(validate(&bad,"m4",&fixture.current,&vision).is_err(),"mutation {n}");
        }
    }
    #[test]
    fn two_native_tiers_install_reuse_bind_and_keep_compiled_files_unchanged(){
        let fixture=Fixture::new();let (archive,pin,meta)=fixture.archive(4096);assert!(bind_at(&fixture.project,"m4",&fixture.model,&fixture.cache,&fixture.current).unwrap().is_none());
        install_at(&fixture.project,&archive,&pin,"m4",&fixture.model,&fixture.cache,&fixture.current).unwrap();assert!(bind_at(&fixture.project,"m4",&fixture.model,&fixture.cache,&fixture.current).unwrap().is_none());
        let before=file_set(&fixture.cache).unwrap();install_at(&fixture.project,&archive,&pin,"m4",&fixture.model,&fixture.cache,&fixture.current).unwrap();assert_eq!(before,file_set(&fixture.cache).unwrap());
        let first=fixture.cache.join(meta["groups"][0]["cache_relative"].as_str().unwrap()).join("program.bin");let modified=fs::metadata(&first).unwrap().modified().unwrap();
        let (archive,pin,_)=fixture.archive(8192);install_at(&fixture.project,&archive,&pin,"m4",&fixture.model,&fixture.cache,&fixture.current).unwrap();let before=file_set(&fixture.cache).unwrap();
        let top=bind_at(&fixture.project,"m4",&fixture.model,&fixture.cache,&fixture.current).unwrap().unwrap();let profile=json_file(&top,512*1024).unwrap();assert_eq!(profile["tiers"].as_array().unwrap().len(),2);
        for row in profile["tiers"].as_array().unwrap(){let leaf=top.parent().unwrap().join(row["profile"].as_str().unwrap());assert_eq!(regular_sha(&leaf).unwrap(),row["sha256"]);let loaded=crate::ane_native::Profile::open_for_preparation(&leaf).map_err(|error|error.1).unwrap();assert_eq!(loaded.data["native_cache_abi"],1);}
        assert_eq!(before,file_set(&fixture.cache).unwrap());assert_eq!(modified,fs::metadata(first).unwrap().modified().unwrap());assert_eq!(bind_at(&fixture.project,"m4",&fixture.model,&fixture.cache,&fixture.current).unwrap(),Some(top));
    }
    #[test]
    fn bad_archive_pin_and_existing_cache_conflict_do_not_replace_owner_files(){
        let fixture=Fixture::new();let (archive,pin,meta)=fixture.archive(4096);assert!(install_at(&fixture.project,&archive,&"0".repeat(64),"m4",&fixture.model,&fixture.cache,&fixture.current).is_err());assert!(!fixture.cache.exists());
        let target=fixture.cache.join(meta["groups"][0]["cache_relative"].as_str().unwrap());fs::create_dir_all(&target).unwrap();fs::write(target.join("owner"),b"preserve").unwrap();
        assert!(install_at(&fixture.project,&archive,&pin,"m4",&fixture.model,&fixture.cache,&fixture.current).is_err());assert_eq!(fs::read(target.join("owner")).unwrap(),b"preserve");assert_eq!(fs::read_dir(target).unwrap().count(),1);assert!(!fixture.project.join("local-state/ane/m4/4096").exists());
        fs::remove_dir_all(&fixture.cache).unwrap();let owner=fixture.root.join("owner-cache");fs::create_dir(&owner).unwrap();fs::write(owner.join("marker"),b"owner").unwrap();fs::create_dir_all(fixture.cache.parent().unwrap()).unwrap();std::os::unix::fs::symlink(&owner,&fixture.cache).unwrap();
        assert!(install_at(&fixture.project,&archive,&pin,"m4",&fixture.model,&fixture.cache,&fixture.current).is_err());assert_eq!(fs::read(owner.join("marker")).unwrap(),b"owner");assert_eq!(fs::read_dir(owner).unwrap().count(),1);
    }
    #[test]
    #[ignore = "requires the current built Rust CLI on M4"]
    fn public_cli_installs_both_native_tiers_without_python() {
        let fixture=Fixture::new();let binary=std::env::var_os("JUSTFIT_TEST_ANE_CLI").expect("current M4 CLI");
        let (first,pin1,_)=fixture.archive(4096);let (second,pin2,_)=fixture.archive(8192);
        let run=|bundles:&[(&Path,&str)]|{
            let mut command=std::process::Command::new(&binary);command.env("PATH","").env("HOME",&fixture.root).env("HF_HUB_OFFLINE","1")
                .args(["ane","install","--project"]).arg(&fixture.project).args(["--node","m4","--model"]).arg(&fixture.model);
            for (path,pin) in bundles{command.arg("--ane-bundle").arg(path).arg("--ane-sha256").arg(pin);}
            let output=command.output().unwrap();assert!(output.status.success(),"{}",String::from_utf8_lossy(&output.stderr));serde_json::from_slice::<Value>(&output.stdout).unwrap()
        };
        let no_assets=run(&[]);assert_eq!(no_assets["installed"],true);assert!(no_assets["vision_profile"].is_null());assert!(!fixture.cache.exists());
        let first_result=run(&[(&first,&pin1)]);assert_eq!(first_result["installed"],true);assert!(first_result["vision_profile"].is_null());
        let second_result=run(&[(&first,&pin1),(&second,&pin2)]);let top=PathBuf::from(second_result["vision_profile"].as_str().unwrap());assert!(top.is_file());
        let before=file_set(&fixture.cache).unwrap();let again=run(&[(&first,&pin1),(&second,&pin2)]);assert_eq!(again["vision_profile"],second_result["vision_profile"]);assert_eq!(before,file_set(&fixture.cache).unwrap());
        fs::remove_dir_all(&fixture.cache).unwrap();let graph_only=run(&[]);assert_eq!(graph_only["vision_profile"],second_result["vision_profile"]);assert!(!fixture.cache.exists());
    }

    #[test]
    fn readonly_hub_model_blob_symlinks_are_valid_inputs() {
        let fixture=Fixture::new();let (archive,pin,_)=fixture.archive(4096);
        let path=fixture.model.join("vision/model.bin");let blob=fixture.root.join("owner-vision-blob");fs::rename(&path,&blob).unwrap();std::os::unix::fs::symlink(&blob,&path).unwrap();
        let before=fs::read(&blob).unwrap();install_at(&fixture.project,&archive,&pin,"m4",&fixture.model,&fixture.cache,&fixture.current).unwrap();assert_eq!(fs::read(&blob).unwrap(),before);assert!(path.is_symlink());
        let (second,pin2,_)=fixture.archive(8192);install_at(&fixture.project,&second,&pin2,"m4",&fixture.model,&fixture.cache,&fixture.current).unwrap();
        assert!(bind_at(&fixture.project,"m4",&fixture.model,&fixture.cache,&fixture.current).unwrap().unwrap().is_file());assert_eq!(fs::read(blob).unwrap(),before);
    }

    #[test]
    #[ignore = "requires the M4 controlled TLS file fixture"]
    fn pinned_https_index_installs_two_tiers_and_rejects_bad_sets_before_install() {
        let fixture=Fixture::new();let url=std::env::var("JUSTFIT_TEST_HTTPS_URL").unwrap();let public=PathBuf::from(std::env::var_os("JUSTFIT_TEST_ANE_HTTPS_FILES").unwrap());
        let cert=fs::read(std::env::var_os("JUSTFIT_TEST_HTTPS_CERT").unwrap()).unwrap();let cert=ureq::tls::Certificate::from_pem(&cert).unwrap().to_owned();
        let tls=ureq::tls::TlsConfig::builder().root_certs(ureq::tls::RootCerts::Specific(vec![cert].into())).build();
        let agent:ureq::Agent=ureq::Agent::config_builder().https_only(true).tls_config(tls).timeout_per_read(Some(std::time::Duration::from_secs(5))).build().into();
        let target=DownloadTarget{root:&fixture.project,node:"m4",model:&fixture.model,cache:&fixture.cache,current:&fixture.current,assets:None,graph_only:false};
        let mut rows=vec![];
        for capacity in [4096,8192]{let (archive,pin,_)=fixture.archive(capacity);let name=format!("{capacity}.zip");fs::copy(archive,public.join(&name)).unwrap();rows.push(json!({"node":"m4","chip":fixture.current.chip,"os_build":fixture.current.os_build,"capacity":capacity,"sha256":pin,"url":format!("{url}/ane/{name}")}));}
        let index=json!({"schema":1,"kind":"justfit-ane-downloads","artifacts":rows});
        let publish=|name:&str,data:&Value|{let bytes=serde_json::to_vec(data).unwrap();fs::write(public.join(name),&bytes).unwrap();(format!("{url}/ane/{name}"),format!("{:x}",Sha256::digest(&bytes)))};
        let (endpoint,pin)=publish("index.json",&index);
        assert!(fetch_at(&target,&endpoint,&"a".repeat(64),&agent).unwrap_err().to_string().contains("SHA256"));assert!(!fixture.cache.exists());
        assert!(fetch_at(&target,&endpoint,&pin,&download::agent()).is_err());assert!(!fixture.cache.exists());
        for mutation in 0..7{
            let mut bad=index.clone();match mutation{
                0=>bad["artifacts"][1]["sha256"]=json!("0".repeat(64)),
                1=>bad["artifacts"][1]["chip"]=json!("another chip"),
                2=>{bad["artifacts"].as_array_mut().unwrap().pop();},
                3=>{bad["artifacts"][1]["url"]=index["artifacts"][0]["url"].clone();bad["artifacts"][1]["sha256"]=index["artifacts"][0]["sha256"].clone();},
                4=>bad["artifacts"][1]["url"]=json!("http://127.0.0.1:1/no-http"),
                5=>bad["artifacts"][1]["os_build"]=json!("another build"),
                _=>bad["artifacts"][1]["sha256"]=json!("not-a-pin"),
            }
            let (endpoint,pin)=publish(&format!("bad-{mutation}.json"),&bad);assert!(fetch_at(&target,&endpoint,&pin,&agent).is_err(),"mutation {mutation}");assert!(!fixture.cache.exists(),"cache mutation {mutation}");assert!(!fixture.project.join("local-state/ane").exists());
        }
        let top=fetch_at(&target,&endpoint,&pin,&agent).unwrap().unwrap();assert!(top.is_file());let before=file_set(&fixture.cache).unwrap();
        let times:BTreeMap<_,_>=before.keys().map(|name|(name.clone(),fs::metadata(fixture.cache.join(name)).unwrap().modified().unwrap())).collect();
        assert_eq!(fetch_at(&target,&endpoint,&pin,&agent).unwrap(),Some(top));assert_eq!(before,file_set(&fixture.cache).unwrap());
        for (name,time) in times{assert_eq!(fs::metadata(fixture.cache.join(name)).unwrap().modified().unwrap(),time);}
        assert!(!fs::read_dir(&fixture.project).unwrap().any(|entry|entry.unwrap().file_name().to_string_lossy().starts_with(".ane-download-")));
    }

    #[test]
    fn graph_only_binding_survives_missing_compiled_cache_without_fake_guards(){
        let fixture=Fixture::new();for capacity in [4096,8192]{let (archive,pin,_)=fixture.archive(capacity);install_at(&fixture.project,&archive,&pin,"m4",&fixture.model,&fixture.cache,&fixture.current).unwrap();}
        fs::remove_dir_all(&fixture.cache).unwrap();let top=bind_at(&fixture.project,"m4",&fixture.model,&fixture.cache,&fixture.current).unwrap().unwrap();assert!(!fixture.cache.exists());
        let profile=json_file(&top,512*1024).unwrap();for row in profile["tiers"].as_array().unwrap(){let leaf=top.parent().unwrap().join(row["profile"].as_str().unwrap());let profile=crate::ane_native::Profile::open_for_preparation(&leaf).map_err(|e|e.1).unwrap();for group in profile.catalog["groups"].as_array().unwrap(){assert_eq!(group.as_object().unwrap().len(),1);assert!(group["asset_hash"].is_string());}}
        assert_eq!(bind_at(&fixture.project,"m4",&fixture.model,&fixture.cache,&fixture.current).unwrap(),Some(top));
    }
    #[test]
    fn existing_bound_profile_keeps_graph_identity_after_cache_disappears(){
        let fixture=Fixture::new();for capacity in [4096,8192]{let (archive,pin,_)=fixture.archive(capacity);install_at(&fixture.project,&archive,&pin,"m4",&fixture.model,&fixture.cache,&fixture.current).unwrap();}
        let top=bind_at(&fixture.project,"m4",&fixture.model,&fixture.cache,&fixture.current).unwrap().unwrap();let before=regular_sha(&top).unwrap();fs::remove_dir_all(&fixture.cache).unwrap();assert_eq!(bind_at(&fixture.project,"m4",&fixture.model,&fixture.cache,&fixture.current).unwrap(),Some(top.clone()));assert_eq!(regular_sha(&top).unwrap(),before);assert!(!fixture.cache.exists());
        let graph=fixture.project.join("local-state/ane/m4/4096/assets/group-00/model.aimodel/program.mil");fs::write(&graph,b"changed owner bytes").unwrap();assert!(bind_at(&fixture.project,"m4",&fixture.model,&fixture.cache,&fixture.current).is_err());assert_eq!(fs::read(graph).unwrap(),b"changed owner bytes");assert_eq!(regular_sha(&top).unwrap(),before);
    }

    #[test]
    fn cross_os_graph_delivery_extracts_no_compiled_programs_and_binds_current_build(){
        let fixture=Fixture::new();let current=Identity{chip:fixture.current.chip.clone(),os_build:"future-build".into()};let assets=fixture.project.join("local-state/ane-default/test/graphs");
        for capacity in [4096,8192]{let (archive,pin,_)=fixture.archive(capacity);
            assert!(install_at_mode(&assets,&archive,&pin,"m4",&fixture.model,InstallationDevice{root:&fixture.project,cache:&fixture.cache,current:&current},false).is_err());
            let meta=install_at_mode(&assets,&archive,&pin,"m4",&fixture.model,InstallationDevice{root:&fixture.project,cache:&fixture.cache,current:&current},true).unwrap();
            assert_eq!(meta["schema"],3);assert_eq!(meta["source_native_bundle_sha256"],pin);
            assert!(!assets.join(capacity.to_string()).join("compiled").exists());
            for group in groups(&meta).unwrap(){assert_eq!(group.as_object().unwrap().len(),1);}
            let mut forged=meta.clone();forged["groups"][0]["cache_relative"]=json!("fake");
            assert!(validate(&forged,"m4",&current,&model_sha(&fixture.project,&fixture.model).unwrap()).is_err());
        }
        assert!(file_set(&fixture.cache).unwrap().is_empty());
        let top=bind_at_location(&fixture.project,&assets,"m4",&fixture.model,&fixture.cache,&current).unwrap().unwrap();
        let before=regular_sha(&top).unwrap();assert_eq!(bind_at_location(&fixture.project,&assets,"m4",&fixture.model,&fixture.cache,&current).unwrap(),Some(top.clone()));assert_eq!(regular_sha(&top).unwrap(),before);
        let catalog=json_file(&assets.join("4096/native-assets-catalog.json"),512*1024).unwrap();assert_eq!(catalog["build"],"future-build");
        for group in groups(&catalog).unwrap(){assert_eq!(group.as_object().unwrap().len(),1);}
        assert!(file_set(&fixture.cache).unwrap().is_empty());
    }
    #[test]
    fn archive_snapshot_is_independent_of_source_and_never_replaces_an_owner(){
        let fixture=Fixture::new();let source=fixture.root.join("source.zip");fs::write(&source,b"before").unwrap();let pin=regular_sha(&source).unwrap();let target=fixture.root.join("snapshot.zip");
        snapshot(&source,&target,&pin).unwrap();fs::write(&source,b"after").unwrap();assert_eq!(fs::read(&target).unwrap(),b"before");
        assert!(snapshot(&source,&target,&regular_sha(&source).unwrap()).is_err());assert_eq!(fs::read(&target).unwrap(),b"before");
        let bad=fixture.root.join("wrong-pin.zip");assert!(snapshot(&source,&bad,&pin).is_err());
    }
    #[test]
    fn stale_stage_cleanup_only_removes_marked_dead_owner(){
        let fixture=Fixture::new();let assets=fixture.root.join("assets");fs::create_dir(&assets).unwrap();
        let mut child=std::process::Command::new("/usr/bin/true").spawn().unwrap();let pid=child.id();child.wait().unwrap();
        let dead=assets.join(format!(".ane-install-{pid}-1"));let foreign=assets.join(format!(".ane-install-{pid}-2"));let live=assets.join(format!(".ane-install-{}-1",std::process::id()));
        for path in [&dead,&foreign,&live]{fs::create_dir(path).unwrap();fs::write(path.join("owner-data"),b"keep unless owned dead stage").unwrap();}
        for (path,owner) in [(&dead,pid),(&live,std::process::id())]{fs::write(path.join("stage-owner.json"),serde_json::to_vec(&json!({"kind":"justfit-ane-stage","pid":owner,"assets":assets})).unwrap()).unwrap();}
        clean_dead_stages(&assets).unwrap();assert!(!dead.exists());assert!(foreign.join("owner-data").is_file());assert!(live.join("owner-data").is_file());
    }

    #[test]
    fn default_delivery_installs_two_pinned_payload_tiers_and_reuses_private_profile(){
        const CASE:&str="ane_bundle::tests::default_delivery_installs_two_pinned_payload_tiers_and_reuses_private_profile";
        if std::env::var("JUSTFIT_DELIVERY_TEST_CHILD").as_deref()!=Ok("1"){
            let output=std::process::Command::new(std::env::current_exe().unwrap()).args(["--exact",CASE,"--nocapture"]).env("JUSTFIT_DELIVERY_TEST_CHILD","1").output().unwrap();
            assert!(output.status.success(),"{}{}",String::from_utf8_lossy(&output.stdout),String::from_utf8_lossy(&output.stderr));return;
        }
        let fixture=Fixture::new();std::env::set_var("HOME",&fixture.root);
        let mut bundles=vec![];
        for capacity in [4096,8192]{let (archive,pin,_)=fixture.archive(capacity);let name=format!("runtime/ane/m4/{capacity}.zip");let path=fixture.project.join(&name);fs::create_dir_all(path.parent().unwrap()).unwrap();fs::copy(archive,&path).unwrap();bundles.push(json!({"capacity":capacity,"path":name,"sha256":pin}));}
        let descriptor=fixture.project.join("config/ane-distribution.json");fs::create_dir_all(descriptor.parent().unwrap()).unwrap();
        let data=json!({"schema":1,"kind":"justfit-ane-defaults","version":cli::VERSION,"native_cache_abi":1,"cache_namespace":NAMESPACE,"vision_weights_sha256":model_sha(&fixture.project,&fixture.model).unwrap(),"precompiled":[{"node":"m4","chip":fixture.current.chip,"os_build":fixture.current.os_build,"bundles":bundles}],"graphs":[]});
        fs::write(&descriptor,serde_json::to_vec(&data).unwrap()).unwrap();let manifest=fixture.project.join("package-manifest.json");let mut pins:BTreeMap<String,String>=serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();pins.insert("config/ane-distribution.json".into(),regular_sha(&descriptor).unwrap());fs::write(&manifest,serde_json::to_vec(&pins).unwrap()).unwrap();
        let config=crate::ane_distribution::Delivery{project:fixture.project.clone(),node:"m4".into(),model:fixture.model.clone(),descriptor:descriptor.clone(),sha256:regular_sha(&descriptor).unwrap(),payload:fixture.project.clone()};
        assert!(!crate::ane_distribution::profile(&config).unwrap().exists());
        let top=crate::ane_distribution::install(&config).unwrap();assert_eq!(top,crate::ane_distribution::profile(&config).unwrap());let before=regular_sha(&top).unwrap();let cached=file_set(&fixture.cache).unwrap();assert!(!cached.is_empty());
        assert_eq!(crate::ane_distribution::install(&config).unwrap(),top);assert_eq!(regular_sha(&top).unwrap(),before);assert_eq!(file_set(&fixture.cache).unwrap(),cached);
        let profile=json_file(&top,65536).unwrap();for tier in profile["tiers"].as_array().unwrap(){let leaf=top.parent().unwrap().join(tier["profile"].as_str().unwrap());assert_eq!(regular_sha(&leaf).unwrap(),tier["sha256"]);crate::ane_native::Profile::open_for_preparation(&leaf).map_err(|e|e.1).unwrap();}
        fs::write(&descriptor,b"changed descriptor").unwrap();assert!(crate::ane_distribution::install(&config).is_err());assert_eq!(regular_sha(&top).unwrap(),before);assert_eq!(file_set(&fixture.cache).unwrap(),cached);
    }

}
