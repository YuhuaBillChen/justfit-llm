//! Packed-model installation: commit-pinned HTTPS, per-artifact verification,
//! existing Hub-cache reuse and resumable completed artifacts, without Python.
use crate::{cli,download::{self,Result,Stage}};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{collections::{BTreeMap,BTreeSet},fs,io::Read,os::{fd::AsRawFd,unix::fs::{DirBuilderExt,OpenOptionsExt}},path::{Path,PathBuf}};
pub const DEFAULT_MODEL:&str="mlx-community/Qwen3.8-27B-4bit";
pub const DEFAULT_REVISION:&str="3e6447f082e89cc7f0bc6e5441afd38dfce760ff";
pub const DEFAULT_DRAFT:&str="incoai/Qwen3.8-27B-DFlash2";
pub const DEFAULT_DRAFT_REVISION:&str="dedf8df68adfb1afeaf7b7480c0a0243108177b4";
pub const PACKED_MODEL:&str="incoai/Qwen3.8-27B-Splash";
const PACKED_REVISION:&str="9d27070b71f7142c6b6025f03ac011d70a73cb48";
pub(crate) fn source_revision<'a>(spec:&str,revision:Option<&'a str>)->&'a str{revision.unwrap_or(if spec==DEFAULT_MODEL{DEFAULT_REVISION}else{"main"})}
const DEFAULT_MANIFEST:&str="40121d0d933fb27a7206248a592451f9ebbd73671f19ce3162cf27a7db0968d5";
const MAX_JSON:u64=64*1024*1024;
#[derive(Clone,Debug)]
pub struct Artifact {pub path:String,pub size:u64,pub sha256:String}
pub fn safe_path(name:&str)->bool {
    !name.is_empty() && !name.starts_with('/') && !name.split('/').any(|part|part.is_empty()||part=="."||part=="..") && !name.chars().any(|c|c<' '||"\\*?[]".contains(c))
}
fn hex(value:&str,n:usize)->bool{value.len()==n && value.bytes().all(|c|c.is_ascii_hexdigit())}
pub fn manifest(value:&Value)->Result<Vec<Artifact>> {
    let format=&value["format"];
    let (schema,magic,target,draft)=match format["name"].as_str(){Some("splash-packed-q4")=>(3,"MDFL0006",64,5),Some("splash-packed-q4-moe")=>(4,"MDFM0001",40,6),_=>return Err("repository is not a supported JustFit runtime package".into())};
    if value["schema_version"].as_u64()!=Some(schema)||value["model"].as_str().is_none_or(|s|s.trim_matches(|c:char|c.is_whitespace()||('\u{1c}'..='\u{1f}').contains(&c)).is_empty())||!value["execution_geometry"].is_object(){return Err("repository is not a supported JustFit runtime package".into());}
    if format["section_alignment_bytes"].as_u64()!=Some(16384)||format["target_layer_magic"]!=magic||format["draft_layer_magic"]!="MDFD0004"||format["vision_magic"]!="MDFV0001"{return Err("runtime package has an unsupported weight format".into());}
    if schema==4 && (value["target"]["architecture"]!="qwen3_5_moe"||value["draft"]["architecture"]!="DFlash2DraftModel"){return Err("runtime package has an unsupported architecture".into());}
    let records=value["artifacts"].as_array().filter(|v|!v.is_empty()).ok_or("runtime package manifest has no artifact list")?;
    let mut result=vec![];let mut names=BTreeSet::new();
    for record in records {
        let record=record.as_object().ok_or("invalid runtime package artifact")?;
        if record.keys().map(String::as_str).collect::<BTreeSet<_>>()!=BTreeSet::from(["path","size","sha256"]){return Err("invalid runtime package artifact".into());}
        let path=record["path"].as_str().filter(|s|safe_path(s)&&*s!="manifest.json").ok_or("invalid runtime package artifact path")?;
        let size=record["size"].as_u64().filter(|n|*n>0).ok_or("invalid runtime package artifact size")?;
        let sha=record["sha256"].as_str().filter(|s|hex(s,64)).ok_or("invalid runtime package artifact SHA256")?.to_ascii_lowercase();
        if !names.insert(path.to_string()){return Err("runtime package artifact paths are not unique".into());}
        if Path::new(path).extension().is_some_and(|suffix|suffix=="bin") && size%16384!=0{return Err("runtime package weight file is unaligned".into());}
        result.push(Artifact{path:path.into(),size,sha256:sha});
    }
    for name in &names {
        let mut path=Path::new(name);
        while let Some(parent)=path.parent(){if parent.as_os_str().is_empty(){break;}
        if names.contains(parent.to_str().ok_or("invalid artifact path")?){return Err("runtime package artifact paths overlap".into());}path=parent;}
    }
    let mut required=vec!["target/embedding.bin".to_string(),"target/head.bin".into(),"draft/model.bin".into(),"vision/model.bin".into()];
    required.extend((0..target).map(|n|format!("target/layer-{n}.bin")));required.extend((0..draft).map(|n|format!("draft/layer-{n}.bin")));
    required.extend(["chat_template.jinja","config.json","tokenizer.json","tokenizer_config.json","vocab.json"].map(|n|format!("tokenizer/{n}")));
    if required.iter().any(|name|!names.contains(name)){return Err("runtime package artifact list is missing required files".into());}Ok(result)
}
fn read_json(path:&Path)->Result<(Value,Vec<u8>)> {
    let input=fs::File::open(path)?;if !input.metadata()?.is_file()||input.metadata()?.len()>MAX_JSON{return Err("model JSON metadata is too large or not a regular file".into());}
    let mut bytes=vec![];input.take(MAX_JSON+1).read_to_end(&mut bytes)?;if bytes.len() as u64>MAX_JSON{return Err("model JSON metadata is too large".into());}Ok((serde_json::from_slice(&bytes)?,bytes))
}
pub fn verify(root:&Path,full:bool)->Result<usize> {
    let (value,_)=read_json(&root.join("manifest.json"))?;let records=manifest(&value)?;
    for record in &records {
        let path=root.join(&record.path);let meta=fs::metadata(&path)?;
        if !meta.is_file()||meta.len()!=record.size{return Err(format!("installed artifact has the wrong size: {}",record.path).into());}
        if full && cli::digest(&path)?!=record.sha256{return Err(format!("installed artifact checksum changed: {}",record.path).into());}
    }Ok(records.len())
}
pub(crate) fn repository(value:&str)->Result<()> {
    let parts:Vec<_>=value.split('/').collect();
    let part=|s:&str|!s.is_empty()&&s.bytes().all(|c|c.is_ascii_alphanumeric()||b"_.-".contains(&c))&&s.as_bytes().first().is_some_and(|c|c.is_ascii_alphanumeric()||*c==b'_')&&s.as_bytes().last().is_some_and(|c|c.is_ascii_alphanumeric()||*c==b'_');
    if parts.len()!=2||!part(parts[0])||!part(parts[1])||parts[1].len()>96||value.contains("--")||value.contains("..")||value.ends_with(".git"){return Err("model must be a full Hugging Face repository ID (owner/repo)".into());}Ok(())
}
fn env_true(value:&str)->bool{["1","ON","YES","TRUE"].iter().any(|word|value.eq_ignore_ascii_case(word))}
pub(crate) fn offline()->bool {
    let value=std::env::var("HF_HUB_OFFLINE").ok().filter(|s|!s.is_empty()).or_else(||std::env::var("TRANSFORMERS_OFFLINE").ok()).unwrap_or_default();env_true(&value)
}
fn expanded(path:PathBuf)->PathBuf {
    // Python expanduser runs before expandvars. Keep non-UTF8 owner paths intact.
    let Some(text)=path.to_str() else{return path};
    let home=std::env::var_os("HOME");
    let text=if text=="~"{home.clone().map(PathBuf::from).unwrap_or(path.clone())}else if let Some(tail)=text.strip_prefix("~/"){home.map(|home|PathBuf::from(home).join(tail)).unwrap_or(path.clone())}else{path.clone()};
    let Some(text)=text.to_str() else{return text};let mut output=std::ffi::OsString::new();let mut position=0;
    while let Some(offset)=text[position..].find('$') {
        let start=position+offset;output.push(&text[position..start]);let tail=&text[start+1..];
        let variable=if let Some(tail)=tail.strip_prefix('{'){tail.find('}').map(|end|(&tail[..end],start+end+3))}else{
            let end=tail.find(|c:char|!c.is_alphanumeric()&&c!='_').unwrap_or(tail.len());(end>0).then_some((&tail[..end],start+1+end))
        };
        if let Some((name,end))=variable {if let Some(value)=std::env::var_os(name){output.push(value)}else{output.push(&text[start..end]);}position=end;}
        else{output.push("$");position=start+1;}
    }
    output.push(&text[position..]);PathBuf::from(output)
}
fn hf_home()->Result<PathBuf> {
    if let Some(root)=std::env::var_os("HF_HOME"){return Ok(expanded(PathBuf::from(root)));}
    let base=match std::env::var_os("XDG_CACHE_HOME"){Some(path)=>PathBuf::from(path),None=>PathBuf::from(std::env::var_os("HOME").ok_or("HOME missing")?).join(".cache")};
    Ok(expanded(base.join("huggingface")))
}
pub(crate) fn hub_cache()->Result<PathBuf> {
    match std::env::var_os("HF_HUB_CACHE").or_else(||std::env::var_os("HUGGINGFACE_HUB_CACHE")){Some(path)=>Ok(expanded(PathBuf::from(path))),None=>Ok(hf_home()?.join("hub"))}
}
pub fn cached_source_draft(family:&str)->Result<PathBuf>{
    let repo=match family{"Qwen3.8-27B"=>"incoai/Qwen3.8-27B-DFlash2","Qwen3.6-35B-A3B"=>"incoai/Qwen3.6-35B-A3B-DFlash2",_=>return Err("no known source draft for model family".into())};
    let root=hub_cache()?.join(format!("models--{}",repo.replace('/',"--")));let commit=fs::read_to_string(root.join("refs/main"))?;let commit=commit.trim();if !hex(commit,40){return Err("invalid cached draft commit".into());}let path=root.join("snapshots").join(commit);if !path.is_dir(){return Err("default draft is not cached; supply --draft-model".into());}Ok(path.canonicalize()?)
}
pub(crate) fn repository_cache(project:&Path,repo:&str)->PathBuf {
    project.join("local-state/models").join(format!("models--{}",repo.replace('/',"--")))
}
fn cached_in(root:&Path,repo:&str,revision:&str)->Result<Option<PathBuf>> {
    let commit=if hex(revision,40){revision.to_ascii_lowercase()}else{
        let reference=root.join("refs").join(revision);
        let input=match fs::File::open(reference){Ok(file)=>file,Err(error) if error.kind()==std::io::ErrorKind::NotFound=>return Ok(None),Err(error)=>return Err(error.into())};
        if !input.metadata()?.is_file()||input.metadata()?.len()>4096{return Err("invalid cached model revision reference".into());}
        let mut value=String::new();input.take(4097).read_to_string(&mut value)?;
        let value=value.trim();if !hex(value,40){return Err("invalid cached model revision reference".into());}value.to_ascii_lowercase()
    };
    let path=root.join("snapshots").join(&commit);
    if !path.is_dir(){return Ok(None);}
    if repo==PACKED_MODEL&&commit==PACKED_REVISION&&cli::digest(&path.join("manifest.json"))?!=DEFAULT_MANIFEST{return Err("cached qualified model manifest SHA256 mismatch".into());}
    verify(&path,true)?;Ok(Some(path.canonicalize()?))
}
fn cached(project:&Path,repo:&str,revision:&str)->Result<Option<PathBuf>> {
    if let Some(path)=cached_in(&repository_cache(project,repo),repo,revision)?{return Ok(Some(path));}
    cached_in(&hub_cache()?.join(format!("models--{}",repo.replace('/',"--"))),repo,revision)
}
pub(crate) fn remember_revision(root:&Path,revision:&str,commit:&str)->Result<()> {
    if hex(revision,40){return Ok(());}
    // The per-repository lock covers publication. Reject staging symlinks and
    // publish a whole reference atomically, including branch names with slashes.
    let path=fresh_path(root,&format!("refs/{revision}"))?;
    fs::create_dir_all(path.parent().ok_or("invalid revision reference path")?)?;
    let stage=Stage::new(root,"model-ref")?;let temp=stage.path().join("commit");
    use std::io::Write;
    let mut file=fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&temp)?;
    file.write_all(commit.as_bytes())?;file.sync_all()?;drop(file);
    fs::rename(temp,path)?;Ok(())
}
pub(crate) struct Lock(fs::File);
impl Lock {pub(crate) fn take(root:&Path)->Result<Self>{let file=fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).mode(0o600).custom_flags(libc::O_NOFOLLOW).open(root.join(".install.lock"))?;if !file.metadata()?.is_file(){return Err("unsafe model install lock".into());}
        if unsafe{libc::flock(file.as_raw_fd(),libc::LOCK_EX)}!=0{return Err(std::io::Error::last_os_error().into());}Ok(Self(file))}}
impl Drop for Lock{fn drop(&mut self){let _=unsafe{libc::flock(self.0.as_raw_fd(),libc::LOCK_UN)};}}
pub(crate) fn token()->Option<String> {
    if std::env::var("HF_HUB_DISABLE_IMPLICIT_TOKEN").is_ok_and(|s|env_true(&s)){return None;}
    std::env::var("HF_TOKEN").ok().filter(|s|!s.trim().is_empty()).or_else(||std::env::var("HUGGING_FACE_HUB_TOKEN").ok().filter(|s|!s.trim().is_empty())).or_else(||{
        let path=std::env::var_os("HF_TOKEN_PATH").map(|p|expanded(PathBuf::from(p))).or_else(||hf_home().ok().map(|p|p.join("token")))?;
        if fs::metadata(&path).ok()?.len()>16384{return None;}fs::read_to_string(path).ok()
    }).map(|s|s.trim().to_string()).filter(|s|!s.is_empty())
}
fn fresh_path(root:&Path,name:&str)->Result<PathBuf> {
    if !safe_path(name){return Err("unsafe model destination".into());}
    let path=root.join(name);let mut parent=path.as_path();
    while parent!=root {
        if fs::symlink_metadata(parent).is_ok_and(|m|m.file_type().is_symlink()){return Err("model staging contains a symlink".into());}
        parent=parent.parent().ok_or("unsafe model destination")?;
    }Ok(path)
}
pub(crate) fn rename_exclusive(source:&Path,destination:&Path)->Result<()> {
    use std::os::unix::ffi::OsStrExt;
    let source=std::ffi::CString::new(source.as_os_str().as_bytes())?;
    let destination=std::ffi::CString::new(destination.as_os_str().as_bytes())?;
    #[cfg(target_os="macos")]
    let code=unsafe{libc::renamex_np(source.as_ptr(),destination.as_ptr(),libc::RENAME_EXCL)};
    #[cfg(target_os="linux")]
    let code=unsafe{libc::renameat2(libc::AT_FDCWD,source.as_ptr(),libc::AT_FDCWD,destination.as_ptr(),libc::RENAME_NOREPLACE)};
    #[cfg(not(any(target_os="macos",target_os="linux")))]
    return Err("exclusive model publication is unsupported on this platform".into());
    #[cfg(any(target_os="macos",target_os="linux"))]
    if code!=0{return Err(std::io::Error::last_os_error().into());}
    Ok(())
}
fn atomic_metadata(root:&Path,destination:&Path,bytes:&[u8])->Result<()> {
    use std::io::Write;
    let stage=Stage::new(root,"model-metadata")?;let temp=stage.path().join("metadata");
    let mut file=fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&temp)?;
    file.write_all(bytes)?;file.sync_all()?;drop(file);rename_exclusive(&temp,destination)
}
pub(crate) fn local_spec(spec:&str)->Result<PathBuf>{Ok(if let Some(tail)=spec.strip_prefix("~/"){PathBuf::from(std::env::var_os("HOME").ok_or("HOME missing")?).join(tail)}else{PathBuf::from(spec)})}
/// Preserve a local selection link until its actual assembly store is locked.
pub fn resolve_for_serve(project:&Path,spec:&str,revision:Option<&str>)->Result<crate::model_hold::HeldModel>{
    let local=local_spec(spec)?;
    if local.is_dir(){return crate::model_hold::hold(&local);}
    crate::model_hold::hold(&resolve(project,spec,revision)?)
}
/// Binary first-use model/ANE installation shares the verified bundled runtime.
/// Existing projects and source-only SDKs are never overwritten or silently built.
pub(crate) fn source_runtime(project:&Path)->Result<()> {
    if crate::cli_build::is_built(project){return Ok(());}
    if !project.exists(){cli::initialize(&cli::payload_root()?,project)?;}
    cli::verify(project)?;
    if let Ok(payload)=cli::payload_root(){crate::runtime_bundle::install_bundled(project,&payload)?;}
    if !crate::cli_build::is_built(project){return Err("source model preparation requires the matching native runtime; run justfit build or install its runtime bundle".into());}
    Ok(())
}
/// Raw local checkpoints are prepared by Rust before the service holds them.
pub fn resolve_for_serve_options(project:&Path,spec:&str,revision:Option<&str>,draft:Option<&str>,mmproj:Option<&str>,language_only:bool)->Result<crate::model_hold::HeldModel>{
    let local=local_spec(spec)?;
    if !local.exists()&&!spec.starts_with(['/', '.', '~'])&&spec!=PACKED_MODEL{
        if mmproj.is_some(){return Err("Hub models choose their projector; --mmproj is for a local GGUF".into());}
        if let Some(revision)=revision{if !safe_path(revision){return Err("invalid model revision".into());}}
        if spec==DEFAULT_MODEL{source_runtime(project)?;}
        let prepared=crate::source_hub::resolve(project,spec,Some(source_revision(spec,revision)),draft,language_only)?;return crate::model_hold::hold(&prepared);
    }
    let raw=(local.is_dir()&&!local.join("target").is_dir()&&!local.join("model.json").is_file()&&!local.join("manifest.json").is_file())||(local.is_file()&&local.extension().is_some_and(|s|s=="gguf"));
    if raw{
        source_runtime(project)?;
        let draft=draft.map(local_spec).transpose()?;
        let projector=mmproj.map(local_spec).transpose()?;let prepared=crate::source_mlx::prepare_local(project,&local,draft.as_deref(),projector.as_deref(),language_only)?;
        return crate::model_hold::hold(&prepared);
    }
    if mmproj.is_some(){return Err("--mmproj requires a raw GGUF source target".into());}
    if draft.is_some(){return Err("--draft-model requires a raw source model; prepared packages already pin their draft".into());}
    if language_only&&local.join("model.json").is_file(){
        let record=crate::source_mlx::read_json(&local.join("model.json"))?;
        if record["target_format"]=="mlx-affine"{
            let target=record["sources"]["target"]["repo"].as_str().ok_or("source target missing")?;let draft=record["sources"]["draft"]["repo"].as_str().ok_or("source draft missing")?;
            let target=local_spec(target)?;let draft=local_spec(draft)?;
            let prepared=crate::source_mlx::prepare(project,&target,Some(&draft),true)?;
            return crate::model_hold::hold(&prepared);
        }
    }
    resolve_for_serve(project,spec,revision)
}
pub fn resolve(project:&Path,spec:&str,revision:Option<&str>)->Result<PathBuf> {
    let local=local_spec(spec)?;
    if local.is_dir(){return Ok(local.canonicalize()?);}
    if spec.starts_with('/')||spec.starts_with('.')||spec.starts_with('~'){return Err(format!("model directory does not exist: {spec}").into());}
    repository(spec)?;
    let revision=revision.unwrap_or(if spec==PACKED_MODEL{PACKED_REVISION}else{"main"});
    if !safe_path(revision){return Err("invalid model revision".into());}
    let offline=offline();
    // Branches must be resolved online unless offline was explicitly requested.
    if hex(revision,40)||offline {if let Some(path)=cached(project,spec,revision)?{return Ok(path);}}
    if offline{return Err("model is not present in the offline model caches".into());}
    let credential=token();
    resolve_remote(project,spec,revision,|url,limit|download::metadata(url,limit,credential.as_deref()),|url,path,size,sha|download::fetch_resumable_authenticated(url,path,size,sha,credential.as_deref()))
}
fn resolve_remote(project:&Path,spec:&str,revision:&str,mut metadata:impl FnMut(&str,u64)->Result<Vec<u8>>,transfer:impl FnMut(&str,&Path,u64,&str)->Result<u64>)->Result<PathBuf> {
    repository(spec)?;if !safe_path(revision){return Err("invalid model revision".into());}
    let encoded=percent_encoding::utf8_percent_encode(revision,percent_encoding::NON_ALPHANUMERIC).to_string();
    let info:Value=serde_json::from_slice(&metadata(&format!("https://huggingface.co/api/models/{spec}/revision/{encoded}?blobs=true"),MAX_JSON)?)?;
    let commit=info["sha"].as_str().filter(|s|hex(s,40)).ok_or("Hub did not resolve model to a snapshot commit")?.to_ascii_lowercase();
    if hex(revision,40)&&!revision.eq_ignore_ascii_case(&commit){return Err("Hub resolved a different requested commit".into());}
    let siblings=info["siblings"].as_array().ok_or("Hub file metadata missing")?;
    let manifest_size=siblings.iter().find(|row|row["rfilename"]=="manifest.json").and_then(|r|r["size"].as_u64()).filter(|n|*n>0&&*n<=MAX_JSON).ok_or("repository has no valid JustFit runtime package manifest")?;
    let bytes=metadata(&format!("https://huggingface.co/{spec}/resolve/{commit}/manifest.json"),manifest_size)?;
    if bytes.len() as u64!=manifest_size{return Err("Hub model manifest size mismatch".into());}
    install_snapshot(project,Snapshot{spec,revision,info:&info,bytes:&bytes},transfer)
}

struct Snapshot<'a>{spec:&'a str,revision:&'a str,info:&'a Value,bytes:&'a [u8]}
fn install_snapshot(project:&Path,input:Snapshot<'_>,mut transfer:impl FnMut(&str,&Path,u64,&str)->Result<u64>)->Result<PathBuf> {
    let Snapshot{spec,revision,info,bytes}=input;repository(spec)?;
    if !safe_path(revision){return Err("invalid model revision".into());}
    let commit=info["sha"].as_str().filter(|s|hex(s,40)).ok_or("invalid snapshot commit")?.to_ascii_lowercase();
    if hex(revision,40)&&!revision.eq_ignore_ascii_case(&commit){return Err("snapshot commit mismatch".into());}
    let siblings=info["siblings"].as_array().ok_or("Hub file metadata missing")?;
    let published:BTreeMap<_,_>=siblings.iter().filter_map(|row|Some((row["rfilename"].as_str()?,row))).collect();
    if published.get("manifest.json").and_then(|row|row["size"].as_u64())!=Some(bytes.len() as u64)||bytes.len() as u64>MAX_JSON{return Err("snapshot manifest size mismatch".into());}
    let base=format!("https://huggingface.co/{spec}/resolve/{commit}");
    let manifest_sha=format!("{:x}",Sha256::digest(bytes));
    if spec==PACKED_MODEL&&commit==PACKED_REVISION&&manifest_sha!=DEFAULT_MANIFEST{return Err("qualified model manifest SHA256 mismatch".into());}
    let value:Value=serde_json::from_slice(bytes)?;let records=manifest(&value)?;
    for record in &records {
        let row=published.get(record.path.as_str()).ok_or("Hub artifact missing")?;
        if row["size"].as_u64()!=Some(record.size)||row["lfs"]["sha256"].as_str().is_some_and(|s|!s.eq_ignore_ascii_case(&record.sha256)){return Err(format!("Hub artifact does not match manifest: {}",record.path).into());}
    }
    fs::create_dir_all(project)?;let project=project.canonicalize()?;
    let root=fresh_path(&project,&format!("local-state/models/models--{}",spec.replace('/',"--")))?;
    fs::create_dir_all(&root)?;let _lock=Lock::take(&root)?;
    let snapshots=fresh_path(&root,"snapshots")?;fs::create_dir_all(&snapshots)?;let destination=fresh_path(&snapshots,&commit)?;
    if destination.exists(){if cli::digest(&destination.join("manifest.json"))?!=manifest_sha{return Err("existing model snapshot manifest changed".into());}verify(&destination,true)?;remember_revision(&root,revision,&commit)?;return Ok(destination.canonicalize()?);}
    let partial=fresh_path(&root,&format!(".downloading-{commit}"))?;let receipt=fresh_path(&root,&format!(".downloading-{commit}.json"))?;
    let transfers=fresh_path(&root,&format!(".transfers-{commit}"))?;
    let owner=json!({"installer":"rust","repo":spec,"revision":commit,"manifest_sha256":manifest_sha});
    // The receipt is published before any persistent partial directory. Both
    // metadata files are atomic, so a crash can leave either absent, never half written.
    if (partial.exists()||transfers.exists())&&!receipt.exists(){return Err("model staging has no ownership receipt".into());}
    if receipt.exists(){if read_json(&receipt)?.0!=owner{return Err("model staging is not owned by this installation".into());}}
    else{atomic_metadata(&root,&receipt,serde_json::to_string(&owner)?.as_bytes())?;}
    match fs::DirBuilder::new().mode(0o700).create(&partial) {
        Ok(())=>(),
        Err(error) if error.kind()==std::io::ErrorKind::AlreadyExists=>{if fs::symlink_metadata(&partial)?.file_type().is_symlink()||!partial.is_dir(){return Err("unsafe model staging directory".into());}},
        Err(error)=>return Err(error.into()),
    }
    match fs::DirBuilder::new().mode(0o700).create(&transfers){Ok(())=>(),Err(error) if error.kind()==std::io::ErrorKind::AlreadyExists=>{if !transfers.is_dir(){return Err("invalid owned transfer directory".into());}},Err(error)=>return Err(error.into())}
    let manifest_path=fresh_path(&partial,"manifest.json")?;
    if !manifest_path.exists(){atomic_metadata(&partial,&manifest_path,bytes)?;}
    if cli::digest(&manifest_path)?!=manifest_sha{return Err("model staging manifest changed".into());}
    for record in &records {
        let path=fresh_path(&partial,&record.path)?;
        if path.exists(){if fs::metadata(&path)?.len()==record.size&&cli::digest(&path)?==record.sha256{continue;}return Err(format!("completed staged artifact checksum changed: {}",record.path).into());}
        let temp=fresh_path(&transfers,&format!("{}.part",record.sha256))?;
        let encoded=record.path.split('/').map(|p|percent_encoding::utf8_percent_encode(p,percent_encoding::NON_ALPHANUMERIC).to_string()).collect::<Vec<_>>().join("/");
        if transfer(&format!("{base}/{encoded}"),&temp,record.size,&record.sha256)?!=record.size{return Err("model artifact size mismatch".into());}
        fs::create_dir_all(path.parent().ok_or("invalid model path")?)?;rename_exclusive(&temp,&path)?;
    }
    verify(&partial,true)?;
    if fs::read_dir(&transfers)?.next().is_none(){fs::remove_dir(&transfers)?;}
    rename_exclusive(&partial,&destination)?;remember_revision(&root,revision,&commit)?;Ok(destination.canonicalize()?)
}
pub fn run_cli(args:&[String])->Result<()> {
    if args.first().map(String::as_str)==Some("prepare"){return crate::source_mlx::run_cli(&args[1..]);}
    if let [action,path]=args {if action=="verify"{let root=Path::new(path);let files=if root.join("model.json").is_file(){crate::source_hub::verify_assembly(root)?}else{verify(root,true)?};println!("{}",json!({"passed":true,"files":files,"verifier":"rust"}));return Ok(());}}
    if args.first().map(String::as_str)!=Some("install"){return Err("usage: justfit model verify LOCAL_PACKAGE | justfit model install --project PATH --model MODEL [--revision REV]".into());}
    let mut flags=BTreeMap::new();let mut index=1;let mut language_only=false;let mut offline=false;
    while index<args.len(){let flag=args[index].as_str();if flag=="--language-only"{language_only=true;index+=1;continue;}
        if flag=="--offline"{offline=true;index+=1;continue;}
        if !["--project","--model","--revision","--draft-model","--mmproj"].contains(&flag){return Err(format!("unknown model option: {flag}").into());}let value=args.get(index+1).ok_or("model option needs a value")?;if flags.insert(flag,value.as_str()).is_some(){return Err(format!("duplicate model option: {flag}").into());}index+=2;}
    let project=flags.get("--project").map(PathBuf::from).map(Ok).unwrap_or_else(cli::default_project)?;let model=flags.get("--model").copied().unwrap_or(DEFAULT_MODEL);
    if offline{std::env::set_var("HF_HUB_OFFLINE","1");}
    let held=resolve_for_serve_options(&project,model,flags.get("--revision").copied(),flags.get("--draft-model").copied(),flags.get("--mmproj").copied(),language_only)?;println!("{}",json!({"model":held.path,"installer":"rust"}));Ok(())
}

#[cfg(test)]
mod snapshot_contract {
    use super::*;
    #[test]
    fn original_mlx_default_is_pinned_without_changing_explicit_packed_selection() {
        assert_eq!(DEFAULT_MODEL,"mlx-community/Qwen3.8-27B-4bit");
        assert_eq!(source_revision(DEFAULT_MODEL,None),"3e6447f082e89cc7f0bc6e5441afd38dfce760ff");
        assert_eq!(source_revision(DEFAULT_MODEL,Some("owner-revision")),"owner-revision");
        assert_eq!(source_revision("owner/source",None),"main");
        assert_eq!(PACKED_MODEL,"incoai/Qwen3.8-27B-Splash");
        assert_eq!(DEFAULT_DRAFT_REVISION,"dedf8df68adfb1afeaf7b7480c0a0243108177b4");
    }
    #[test]
    fn hf_boolean_values_match_reference_constants() {
        for value in ["1","ON","on","YES","yes","TRUE","True"]{assert!(env_true(value));}
        for value in ["","0","OFF","false"," true ","auto"]{assert!(!env_true(value));}
    }

    fn fixture()->(Vec<u8>,Value) {
        let mut names=vec!["target/embedding.bin".to_string(),"target/head.bin".into(),"draft/model.bin".into(),"vision/model.bin".into()];
        names.extend((0..64).map(|n|format!("target/layer-{n}.bin")));names.extend((0..5).map(|n|format!("draft/layer-{n}.bin")));
        names.extend(["chat_template.jinja","config.json","tokenizer.json","tokenizer_config.json","vocab.json"].map(|n|format!("tokenizer/{n}")));
        let sha=format!("{:x}",Sha256::digest(vec![42u8;16384]));
        let artifacts:Vec<_>=names.iter().map(|path|json!({"path":path,"size":16384,"sha256":sha})).collect();
        let bytes=serde_json::to_vec(&json!({"schema_version":3,"model":"toy data, never inference","execution_geometry":{},"format":{"name":"splash-packed-q4","section_alignment_bytes":16384,"target_layer_magic":"MDFL0006","draft_layer_magic":"MDFD0004","vision_magic":"MDFV0001"},"artifacts":artifacts})).unwrap();
        let mut siblings=vec![json!({"rfilename":"manifest.json","size":bytes.len()})];
        siblings.extend(names.iter().map(|name|json!({"rfilename":name,"size":16384,"lfs":{"sha256":sha}})));
        (bytes,json!({"sha":"a".repeat(40),"siblings":siblings}))
    }
    fn input<'a>(bytes:&'a [u8],info:&'a Value)->Snapshot<'a>{Snapshot{spec:"owner/toy-model",revision:"stable",bytes,info}}
    fn copy(_: &str,path:&Path,size:u64,sha:&str)->Result<u64>{fs::write(path,vec![42u8;16384])?;assert_eq!(cli::digest(path)?,sha);assert_eq!(size,16384);Ok(size)}
    #[test]
    fn interrupted_install_reuses_completed_artifacts_and_recovers_owned_manifest() {
        let stage=Stage::new(&std::env::temp_dir(),"justfit-snapshot-resume-test").unwrap();let project=stage.path().join("project");let (bytes,info)=fixture();let mut count=0;
        assert!(install_snapshot(&project,input(&bytes,&info),|url,path,size,sha|{count+=1;if count==5{return Err("injected transfer interruption".into());}copy(url,path,size,sha)}).is_err());
        let root=repository_cache(&project,"owner/toy-model");let partial=root.join(format!(".downloading-{}","a".repeat(40)));
        assert!(!root.join("refs/stable").exists());assert!(!root.join("snapshots").join("a".repeat(40)).exists());
        let first=partial.join("target/embedding.bin");let modified=fs::metadata(&first).unwrap().modified().unwrap();
        // A matching owner receipt permits recovery if a crash left no manifest.
        fs::remove_file(partial.join("manifest.json")).unwrap();let mut resumed=0;
        let snapshot=install_snapshot(&project,input(&bytes,&info),|url,path,size,sha|{resumed+=1;copy(url,path,size,sha)}).unwrap();
        assert_eq!(resumed,74);assert_eq!(verify(&snapshot,true).unwrap(),78);
        assert_eq!(fs::metadata(snapshot.join("target/embedding.bin")).unwrap().modified().unwrap(),modified);
        assert_eq!(cached_in(&root,"owner/toy-model","stable").unwrap(),Some(snapshot.clone()));
        assert_eq!(install_snapshot(&project,input(&bytes,&info),|_,_,_,_|panic!("completed snapshot must not download again")).unwrap(),snapshot);
    }
    #[test]
    fn invalid_remote_metadata_and_owned_path_collisions_preserve_owner_data() {
        let stage=Stage::new(&std::env::temp_dir(),"justfit-snapshot-owner-test").unwrap();let (bytes,info)=fixture();
        let project=stage.path().join("bad-metadata");let mut bad=info.clone();bad["siblings"][1]["lfs"]["sha256"]=json!("b".repeat(64));
        assert!(install_snapshot(&project,input(&bytes,&bad),copy).is_err());assert!(!project.exists());
        let root=repository_cache(&project,"owner/toy-model");fs::create_dir_all(&root).unwrap();let owner=stage.path().join("owner");fs::create_dir(&owner).unwrap();fs::write(owner.join("marker"),b"owner bytes").unwrap();
        std::os::unix::fs::symlink(&owner,root.join("snapshots")).unwrap();assert!(install_snapshot(&project,input(&bytes,&info),copy).is_err());
        assert_eq!(fs::read(owner.join("marker")).unwrap(),b"owner bytes");assert_eq!(fs::read_dir(&owner).unwrap().count(),1);
        fs::remove_file(root.join("snapshots")).unwrap();let partial=root.join(format!(".downloading-{}","a".repeat(40)));fs::create_dir(&partial).unwrap();fs::write(partial.join("marker"),b"foreign staging").unwrap();
        assert!(install_snapshot(&project,input(&bytes,&info),copy).is_err());assert_eq!(fs::read(partial.join("marker")).unwrap(),b"foreign staging");assert!(!root.join(format!(".downloading-{}.json","a".repeat(40))).exists());
        let source=stage.path().join("source");let target=stage.path().join("target");fs::write(&source,b"source").unwrap();fs::write(&target,b"owner").unwrap();
        assert!(rename_exclusive(&source,&target).is_err());assert_eq!(fs::read(source).unwrap(),b"source");assert_eq!(fs::read(target).unwrap(),b"owner");
    }
    #[test]
    #[ignore = "requires the M4 controlled HTTPS Hub fixture"]
    fn https_api_manifest_and_all_artifacts_resume_without_python() {
        let base=std::env::var("JUSTFIT_TEST_HTTPS_URL").unwrap();let cert=fs::read(std::env::var("JUSTFIT_TEST_HTTPS_CERT").unwrap()).unwrap();
        let cert=ureq::tls::Certificate::from_pem(&cert).unwrap().to_owned();
        let tls=ureq::tls::TlsConfig::builder().root_certs(ureq::tls::RootCerts::Specific(vec![cert].into())).build();
        let agent:ureq::Agent=ureq::Agent::config_builder().https_only(true).tls_config(tls).build().into();
        let stage=Stage::new(&std::env::temp_dir(),"justfit-https-snapshot-test").unwrap();let project=stage.path().join("project");let mut requests=0;
        let rewrite=|url:&str|format!("{base}/hub/{}",url.strip_prefix("https://huggingface.co/").expect("immutable production Hub URL"));
        let result=resolve_remote(&project,"owner/toy-model","stable",|url,limit|download::metadata_with(&agent,&rewrite(url),limit,None),|url,path,size,sha|{requests+=1;let endpoint=if requests==4{format!("{}?drop=1",rewrite(url))}else{rewrite(url)};download::fetch_resumable_with(&agent,&endpoint,path,size,sha,None)});
        assert!(result.is_err());
        let transfer_root=repository_cache(&project,"owner/toy-model").join(format!(".transfers-{}","a".repeat(40)));
        let parts:Vec<_>=fs::read_dir(&transfer_root).unwrap().map(|p|p.unwrap().path()).collect();assert_eq!(parts.len(),1);
        assert!(fs::metadata(&parts[0]).unwrap().len()>0&&fs::metadata(&parts[0]).unwrap().len()<16384);
        let mut resumed=0;
        let snapshot=resolve_remote(&project,"owner/toy-model","stable",|url,limit|download::metadata_with(&agent,&rewrite(url),limit,None),|url,path,size,sha|{resumed+=1;download::fetch_resumable_with(&agent,&rewrite(url),path,size,sha,None)}).unwrap();
        assert_eq!(resumed,75);assert_eq!(verify(&snapshot,true).unwrap(),78);assert!(!transfer_root.exists());
        assert_eq!(fs::read_to_string(repository_cache(&project,"owner/toy-model").join("refs/stable")).unwrap(),"a".repeat(40));
        let failed=stage.path().join("bad-sha");
        let result=resolve_remote(&failed,"owner/toy-model","stable",|url,limit|download::metadata_with(&agent,&rewrite(url),limit,None),|_,path,size,sha|download::fetch_resumable_with(&agent,&format!("{base}/hub/corrupt"),path,size,sha,None));
        assert!(result.unwrap_err().to_string().contains("SHA256"));assert!(!repository_cache(&failed,"owner/toy-model").join("snapshots").join("a".repeat(40)).exists());
    }
}
