//! Prepare local MLX source checkpoints without an interpreter or weight copying.
use crate::{cli,download::{Result,Stage}};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{collections::{BTreeMap,BTreeSet},fs,io::{Read,Write},os::{fd::AsRawFd,unix::fs::{MetadataExt,OpenOptionsExt}},path::{Path,PathBuf}};
const JSON_LIMIT:u64=64*1024*1024;
const TOKENIZER:&[&str]=&["tokenizer.json","tokenizer_config.json","chat_template.jinja","vocab.json","merges.txt","added_tokens.json","special_tokens_map.json"];
const VISION:&[&str]=&["vision_tower.","model.visual."];
pub fn read_json(path:&Path)->Result<Value>{
    let mut input=fs::File::open(path)?;if !input.metadata()?.is_file()||input.metadata()?.len()>JSON_LIMIT{return Err("source JSON exceeds size limit or is not a file".into());}
    let mut bytes=vec![];(&mut input).take(JSON_LIMIT+1).read_to_end(&mut bytes)?;if bytes.len()as u64>JSON_LIMIT{return Err("source JSON exceeds size limit".into());}Ok(serde_json::from_slice(&bytes)?)
}
fn canonical(value:Value)->Value{match value{Value::Object(map)=>Value::Object(map.into_iter().collect::<BTreeMap<_,_>>().into_iter().map(|(k,v)|(k,canonical(v))).collect()),Value::Array(items)=>Value::Array(items.into_iter().map(canonical).collect()),v=>v}}
pub fn json_bytes(value:Value)->Result<Vec<u8>>{let mut bytes=serde_json::to_vec_pretty(&canonical(value))?;bytes.push(b'\n');Ok(bytes)}
fn source_path(path:&Path)->Result<PathBuf>{let absolute=std::path::absolute(path)?;Ok(absolute.parent().ok_or("source has no parent")?.canonicalize()?.join(absolute.file_name().ok_or("source has no name")?))}
pub(crate) fn safe_destination(path:&Path)->Result<()>{for p in path.ancestors(){match fs::symlink_metadata(p){Ok(m) if m.is_symlink()=>return Err("symlink in model store destination".into()),Err(e) if e.kind()!=std::io::ErrorKind::NotFound=>return Err(e.into()),_=>()}}Ok(())}
fn family(engine:&Path,target:&Path,vision:bool,draft:Option<&Path>)->Result<String>{
    let mut command=std::process::Command::new(engine);command.args(["model-check","mlx-affine",if vision{"safetensors"}else{"none"}]).arg(target);
    if let Some(draft)=draft{command.arg(draft);}
    let output=command.output()?;if !output.status.success(){return Err(format!("source model check: {}",String::from_utf8_lossy(&output.stderr)).into());}
    let value:Value=serde_json::from_slice(&output.stdout)?;let family=value["family"].as_str().ok_or("native model check omitted family")?;
    if !matches!(family,"Qwen3.8-27B"|"Qwen3.6-35B-A3B"){return Err(format!("no DFlash2 draft is known for the {family} family").into());}Ok(family.into())
}
pub(crate) fn shards(root:&Path,vision:bool,exclude_vision:bool)->Result<BTreeSet<String>>{
    let mut names=BTreeSet::new();
    if root.join("model.safetensors.index.json").is_file(){
        let index=read_json(&root.join("model.safetensors.index.json"))?;let map=index["weight_map"].as_object().filter(|v|!v.is_empty()).ok_or("invalid safetensors shard index")?;
        for (tensor,file) in map{
            let file=file.as_str().ok_or("invalid safetensors shard filename")?;let tower=VISION.iter().any(|prefix|tensor.starts_with(prefix));
            if (!vision||tower)&&(!exclude_vision||!tower){names.insert(file.to_string());}
        }
    }else{
        let path=root.join("model.safetensors");let mut input=fs::File::open(path)?;let mut size=[0u8;8];input.read_exact(&mut size)?;let size=u64::from_le_bytes(size);
        if !(2..=100_000_000).contains(&size){return Err("invalid safetensors header".into());}let mut bytes=vec![0;size as usize];input.read_exact(&mut bytes)?;let header:Value=serde_json::from_slice(&bytes)?;let map=header.as_object().ok_or("invalid safetensors header")?;
        if !vision||map.keys().filter(|name|name.as_str()!="__metadata__").any(|name|VISION.iter().any(|prefix|name.starts_with(prefix))){names.insert("model.safetensors".into());}
    }
    for name in &names{if name.contains('/')||!crate::model_install::safe_path(name)||!name.ends_with(".safetensors")||!root.join(name).is_file(){return Err(format!("checkpoint has missing or unsupported shard: {name}").into());}}
    Ok(names)
}
pub(crate) fn processor(root:&Path)->Result<()> {
    let config=if root.join("preprocessor_config.json").is_file(){read_json(&root.join("preprocessor_config.json"))?}else{read_json(&root.join("processor_config.json"))?["image_processor"].clone()};
    if config["patch_size"]!=16||config["temporal_patch_size"]!=2||config["merge_size"]!=2||config["image_mean"]!=json!([0.5,0.5,0.5])||config["image_std"]!=json!([0.5,0.5,0.5]){return Err("unsupported image processor geometry".into());}
    Ok(())
}
fn record(path:&Path,previous:Option<&Value>)->Result<Value>{
    let meta=fs::metadata(path)?;if !meta.is_file(){return Err("source model entry is not a regular file".into());}
    let mut value=json!({"path":path,"bytes":meta.len(),"mtime_ns":meta.mtime()as i128*1_000_000_000+meta.mtime_nsec()as i128,"ctime_ns":meta.ctime()as i128*1_000_000_000+meta.ctime_nsec()as i128});
    let unchanged=previous.is_some_and(|p|["path","bytes","mtime_ns","ctime_ns"].iter().all(|k|p[*k]==value[*k]));
    let digest=if unchanged{previous.unwrap()["digest"].as_str().filter(|v|matches!(v.len(),40|64)&&v.bytes().all(|c|c.is_ascii_hexdigit())).ok_or("invalid prior source digest")?.to_string()}else{
        let resolved=path.canonicalize()?;let name=resolved.file_name().and_then(|s|s.to_str()).unwrap_or("");
        if resolved.parent().and_then(Path::file_name).is_some_and(|p|p=="blobs")&&matches!(name.len(),40|64)&&name.bytes().all(|c|c.is_ascii_hexdigit()){name.to_string()}else{cli::digest(path)?}
    };value["digest"]=json!(digest);Ok(value)
}
fn write_file(path:&Path,bytes:&[u8])->Result<()>{let mut output=fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(path)?;output.write_all(bytes)?;output.sync_all()?;Ok(())}
pub fn is_mlx(root:&Path)->bool{root.is_dir()&&(root.join("model.safetensors").is_file()||root.join("model.safetensors.index.json").is_file()||fs::read_dir(root).is_ok_and(|mut entries|entries.any(|e|e.is_ok_and(|e|e.path().is_file()&&e.path().extension().is_some_and(|v|v=="safetensors")))))}
pub fn prepare_local(project:&Path,target:&Path,draft:Option<&Path>,mmproj:Option<&Path>,language_only:bool)->Result<PathBuf>{
    if is_mlx(target){if mmproj.is_some(){return Err("an MLX target keeps its vision tower in its checkpoint; --mmproj is for GGUF".into());}prepare(project,target,draft,language_only)}else{crate::source_gguf::prepare(project,target,draft,mmproj,language_only)}
}
pub fn prepare(project:&Path,target:&Path,draft:Option<&Path>,language_only:bool)->Result<PathBuf>{
    prepare_with(project,target,draft,language_only,|target,vision,draft|family(&project.join("source/build/justfit-engine"),target,vision,draft))
}
fn prepare_with(project:&Path,target:&Path,draft:Option<&Path>,language_only:bool,mut check:impl FnMut(&Path,bool,Option<&Path>)->Result<String>)->Result<PathBuf>{
    let target=source_path(target)?;if !target.is_dir(){return Err("MLX source target must be a directory".into());}
    for name in ["config.json","tokenizer.json","tokenizer_config.json"]{if !target.join(name).is_file(){return Err(format!("target is missing {name}").into());}}
    let config=target.join("config.json");let name=check(&config,false,None)?;
    let draft=if let Some(path)=draft{source_path(path)?}else{crate::model_install::cached_source_draft(&name)?};
    if !draft.is_dir(){return Err("DFlash2 draft must be a directory".into());}let draft_config=draft.join("config.json");
    let mut files=BTreeMap::new();for path in ["config.json","target/config.json","tokenizer/config.json"]{files.insert(path.to_string(),config.clone());}
    for path in TOKENIZER{if target.join(path).is_file(){files.insert(format!("tokenizer/{path}"),target.join(path));}}
    for path in shards(&target,false,true)?{files.insert(format!("target/{path}"),target.join(path));}
    if !language_only{processor(&target)?;let tower=shards(&target,true,false)?;if tower.is_empty(){return Err("target has no vision tower; use --language-only for text".into());}check(&config,true,None)?;files.insert("vision/config.json".into(),config.clone());for path in tower{files.insert(format!("vision/{path}"),target.join(path));}}
    check(&config,!language_only,Some(&draft_config))?;files.insert("draft/config.json".into(),draft_config);
    for path in shards(&draft,false,false)?{files.insert(format!("draft/{path}"),draft.join(path));}
    if draft.join("mask_embedding.safetensors").is_file(){files.insert("draft/mask_embedding.safetensors".into(),draft.join("mask_embedding.safetensors"));}
    let selection=json!({"target":target,"draft":draft,"mmproj":null,"language_only":language_only});
    let base=json!({"version":1,"model":target,"family":name,"target_format":"mlx-affine","vision_format":if language_only{"none"}else{"safetensors"},"sources":{"target":{"repo":target,"revision":null},"draft":{"repo":draft,"revision":null}}});
    publish(&project.join("local-state/models"),selection,base,files)
}
pub(crate) fn publish(store:&Path,selection:Value,mut base:Value,files:BTreeMap<String,PathBuf>)->Result<PathBuf>{
    safe_destination(store)?;fs::create_dir_all(store)?;let store=store.canonicalize()?;let lock_path=store.join(".install.lock");safe_destination(&lock_path)?;let lock=fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).mode(0o600).custom_flags(libc::O_NOFOLLOW|libc::O_NONBLOCK).open(lock_path)?;
    if !lock.metadata()?.is_file()||unsafe{libc::flock(lock.as_raw_fd(),libc::LOCK_EX)}!=0{return Err("cannot lock model store".into());}
    let resolved=store.join(".resolved");let selections=store.join(".selections");safe_destination(&resolved)?;safe_destination(&selections)?;fs::create_dir_all(&resolved)?;fs::create_dir_all(&selections)?;
    let link=selections.join(format!("{:x}",Sha256::digest(serde_json::to_vec(&selection)?)));
    let previous=if link.is_symlink(){let path=link.canonicalize()?;if path.parent()!=Some(resolved.as_path()){return Err("foreign model selection; no files overwritten".into());}Some(read_json(&path.join("model.json"))?)}else if link.exists(){return Err("model selection is an owner file; no files overwritten".into());}else{None};
    let mut unique:BTreeMap<PathBuf,Value>=BTreeMap::new();let mut entries=serde_json::Map::new();
    for (name,path) in &files{
        if !crate::model_install::safe_path(name){return Err("unsafe assembly path".into());}
        let value=if let Some(v)=unique.get(path){v.clone()}else{let old=previous.as_ref().or(Some(&base)).and_then(|p|p["files"].as_object()).and_then(|map|map.values().find(|v|v["path"].as_str()==path.to_str()));let v=record(path,old)?;unique.insert(path.clone(),v.clone());v};entries.insert(name.clone(),value);
    }
    base["files"]=Value::Object(entries);let bytes=json_bytes(base)?;let destination=resolved.join(format!("{:x}",Sha256::digest(&bytes)));safe_destination(&destination)?;
    if destination.exists(){crate::model_hold::verify(&destination,&read_json(&destination.join("model.json"))?)?;if fs::read(destination.join("model.json"))?!=bytes{return Err("existing source assembly differs; no data overwritten".into());}}else{
        let stage=Stage::new(&resolved,"source-assembly")?;
        for (name,path) in &files{let link=stage.path().join(name);fs::create_dir_all(link.parent().ok_or("invalid assembly path")?)?;std::os::unix::fs::symlink(path,&link)?;}
        write_file(&stage.path().join("model.json"),&bytes)?;crate::model_hold::verify(stage.path(),&serde_json::from_slice(&bytes)?)?;
        // The store lock serializes all compliant writers; reject collisions.
        if destination.exists(){return Err("source assembly destination collision".into());}fs::rename(stage.path(),&destination)?;fs::File::open(&resolved)?.sync_all()?;
    }
    let stage=Stage::new(&selections,"source-selection")?;let temp=stage.path().join("link");std::os::unix::fs::symlink(&destination,&temp)?;fs::rename(temp,&link)?;fs::File::open(&selections)?.sync_all()?;Ok(destination)
}
pub fn run_cli(args:&[String])->Result<()> {
    let mut flags=BTreeMap::new();let mut language_only=false;let mut index=0;
    while index<args.len(){let flag=args[index].as_str();if flag=="--language-only"{language_only=true;index+=1;continue;}
        if !["--project","--target","--draft-model","--mmproj"].contains(&flag){return Err(format!("unknown model prepare option: {flag}").into());}let value=args.get(index+1).ok_or("model prepare option needs a value")?;if flags.insert(flag,value).is_some(){return Err("duplicate model prepare option".into());}index+=2;}
    let project=flags.get("--project").map(PathBuf::from).map(Ok).unwrap_or_else(cli::default_project)?;cli::verify(&project)?;let target=flags.get("--target").map(PathBuf::from).ok_or("--target is required")?;let draft=flags.get("--draft-model").map(PathBuf::from);let projector=flags.get("--mmproj").map(PathBuf::from);let path=prepare_local(&project,&target,draft.as_deref(),projector.as_deref(),language_only)?;println!("{}",json!({"model":path,"installer":"rust"}));Ok(())
}
#[cfg(test)]mod tests{
    use super::*;
    fn fixture()->(Stage,PathBuf,PathBuf){let stage=Stage::new(&std::env::temp_dir().canonicalize().unwrap(),"source-mlx-test").unwrap();let target=stage.path().join("target");let draft=stage.path().join("draft");for root in [&target,&draft]{fs::create_dir(root).unwrap();fs::write(root.join("config.json"),b"{}").unwrap();fs::write(root.join("model.safetensors.index.json"),br#"{"weight_map":{"weight":"model-1.safetensors","vision_tower.weight":"model-2.safetensors"}}"#).unwrap();for n in ["model-1.safetensors","model-2.safetensors"]{fs::write(root.join(n),b"testdata-not-inference").unwrap();}}for n in ["tokenizer.json","tokenizer_config.json"]{fs::write(target.join(n),b"{}").unwrap();}fs::write(target.join("preprocessor_config.json"),br#"{"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5]}"#).unwrap();(stage,target,draft)}
    fn check(_: &Path,_:bool,_:Option<&Path>)->Result<String>{Ok("Qwen3.8-27B".into())}
    #[test]fn publication_reuses_records_and_preserves_source_weights(){let (stage,target,draft)=fixture();let project=stage.path().join("workspace");let a=prepare_with(&project,&target,Some(&draft),false,check).unwrap();let bytes=fs::read(a.join("model.json")).unwrap();assert!(a.join("target/model-1.safetensors").is_symlink());assert!(!a.join("target/model-2.safetensors").exists());assert!(a.join("vision/model-2.safetensors").is_symlink());assert_eq!(prepare_with(&project,&target,Some(&draft),false,check).unwrap(),a);assert_eq!(bytes,fs::read(a.join("model.json")).unwrap());assert_eq!(crate::source_hub::verify_assembly(&a).unwrap(),read_json(&a.join("model.json")).unwrap()["files"].as_object().unwrap().len());let held=crate::model_hold::hold(&a).unwrap();drop(held);assert_eq!(fs::read(target.join("model-1.safetensors")).unwrap(),b"testdata-not-inference");}
    #[test]fn textonly_skips_only_vision_and_source_change_creates_new_identity(){let (stage,target,draft)=fixture();fs::remove_file(target.join("preprocessor_config.json")).unwrap();assert!(prepare_with(stage.path(),&target,Some(&draft),false,check).is_err());let a=prepare_with(stage.path(),&target,Some(&draft),true,check).unwrap();assert!(!a.join("vision").exists());fs::write(target.join("model-1.safetensors"),b"changed model bytes").unwrap();let b=prepare_with(stage.path(),&target,Some(&draft),true,check).unwrap();assert_ne!(a,b);assert!(crate::model_hold::hold(&a).is_err());assert!(crate::model_hold::hold(&b).is_ok());}
    #[test]fn unsafe_shards_and_foreign_store_refused_without_owner_changes(){let (stage,target,draft)=fixture();fs::write(target.join("model.safetensors.index.json"),br#"{"weight_map":{"a":"../owner.safetensors"}}"#).unwrap();assert!(prepare_with(stage.path(),&target,Some(&draft),true,check).is_err());fs::write(target.join("model.safetensors.index.json"),br#"{"weight_map":{"weight":"model-1.safetensors"}}"#).unwrap();let project=stage.path().join("project");fs::create_dir_all(&project).unwrap();let owner=stage.path().join("owner");fs::create_dir(&owner).unwrap();std::os::unix::fs::symlink(&owner,project.join("local-state")).unwrap();assert!(prepare_with(&project,&target,Some(&draft),true,check).is_err());assert!(fs::read_dir(owner).unwrap().next().is_none());}
}
