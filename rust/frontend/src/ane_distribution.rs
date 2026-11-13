//! Release-pinned ANE delivery, performed by an owned Rust background child.
use crate::{ane_bundle,cli,download::Result};
use serde::{Deserialize,Serialize};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{fs,io::Read,os::unix::fs::OpenOptionsExt,path::{Path,PathBuf}};
#[derive(Clone,Deserialize,Serialize)]pub struct Delivery{pub project:PathBuf,pub node:String,pub model:PathBuf,pub descriptor:PathBuf,pub sha256:String,pub payload:PathBuf}
fn bounded(path:&Path,limit:u64)->Result<Value>{let input=fs::OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW|libc::O_NONBLOCK).open(path)?;if !input.metadata()?.is_file()||input.metadata()?.len()>limit{return Err("ANE delivery metadata is not a bounded regular file".into());}let mut bytes=vec![];input.take(limit+1).read_to_end(&mut bytes)?;if bytes.len()as u64>limit{return Err("ANE delivery metadata exceeds size limit".into());}Ok(serde_json::from_slice(&bytes)?)}
fn descriptor(config:&Delivery)->Result<Value>{
 if !["m4","m5","m6"].contains(&config.node.as_str())||config.project.canonicalize()?!=config.project{return Err("invalid owned ANE delivery workspace or node".into());}crate::download::digest(&config.sha256)?;let pins=cli::verify(&config.project)?;let expected=pins.get("config/ane-distribution.json").ok_or("ANE distribution is not pinned by this workspace")?;
 if expected!=&config.sha256||config.descriptor!=config.project.join("config/ane-distribution.json")||cli::digest(&config.descriptor)?!=config.sha256{return Err("ANE distribution pin mismatch".into());}
 let data=bounded(&config.descriptor,64*1024)?;if data["schema"]!=1||data["kind"]!="justfit-ane-defaults"||data["version"]!=cli::VERSION||data["native_cache_abi"]!=1||data["cache_namespace"]!=ane_bundle::NAMESPACE{return Err("incompatible default ANE distribution".into());}
 if data["vision_weights_sha256"]!=crate::vision_identity::sha(&config.project.join("source/build/justfit-engine"),&config.model)?{return Err("default ANE distribution does not match this vision model".into());}Ok(data)
}
pub fn discover(project:&Path,node:&str,model:&Path)->Result<Option<Delivery>>{
 let path=project.join("config/ane-distribution.json");if !path.exists()||!model.join("vision/model.bin").is_file()&&!model.join("model.json").is_file(){return Ok(None);}let pins=cli::verify(project)?;let sha256=pins.get("config/ane-distribution.json").ok_or("default ANE descriptor is not source-pinned")?.clone();let source_sha=cli::digest(&project.join("package-manifest.json"))?;
 let payload=bounded(&project.join("distribution-locator.json"),16384).ok().filter(|v|v["source_manifest_sha256"]==source_sha).and_then(|v|v["payload"].as_str().map(PathBuf::from)).filter(|p|cli::digest(&p.join("package-manifest.json")).is_ok_and(|sha|sha==source_sha)).unwrap_or_else(||cli::payload_root().unwrap_or_else(|_|project.to_path_buf()));
 let config=Delivery{project:project.canonicalize()?,node:node.into(),model:model.canonicalize()?,descriptor:path.canonicalize()?,sha256,payload:payload.canonicalize()?};
 // An unsupported model uses GPU vision; a malformed pinned descriptor fails
 // verification rather than silently installing a different model's assets.
 let data=bounded(&config.descriptor,64*1024)?;if data["vision_weights_sha256"]!=crate::vision_identity::sha(&project.join("source/build/justfit-engine"),model)?{return Ok(None);}let data=descriptor(&config)?;let current=ane_bundle::identity()?;
 if select(&data,node,&current).is_empty(){return Ok(None);}Ok(Some(config))
}
fn select<'a>(data:&'a Value,node:&str,current:&ane_bundle::Identity)->Vec<(&'a Value,bool)>{
 let matches=|kind:&str|data[kind].as_array().into_iter().flatten().filter(|row|row["node"]==node&&row["chip"]==current.chip&&(kind=="graphs"||row["os_build"]==current.os_build)).collect::<Vec<_>>();let mut selected=matches("precompiled").into_iter().map(|row|(row,false)).collect::<Vec<_>>();selected.extend(matches("graphs").into_iter().map(|row|(row,true)));selected
}
fn location(config:&Delivery)->Result<PathBuf>{let os=ane_bundle::identity()?.os_build;let key=format!("{:x}",Sha256::digest(format!("{}:{os}",config.sha256).as_bytes()));Ok(config.project.join("local-state/ane-default").join(&config.node).join(key))}
pub fn profile(config:&Delivery)->Result<PathBuf>{Ok(location(config)?.join("tiered.json"))}
fn install_row(config:&Delivery,common:&Path,row:&Value,graphs:bool)->Result<PathBuf>{
 let class=if graphs{"graphs"}else{"compiled"};let assets=common.join(class);let result;
 if let Some(index)=row.get("manifest"){let url=index["url"].as_str().ok_or("default ANE index URL missing")?;let pin=index["sha256"].as_str().ok_or("default ANE index pin missing")?;result=ane_bundle::fetch_delivery(&config.project,&assets,url,pin,&config.node,&config.model,graphs)?;}
 else{
  let bundles=row["bundles"].as_array().filter(|v|v.len()==2).ok_or("default ANE delivery requires two tier bundles")?;let mut paths=vec![];let mut capacities=std::collections::BTreeSet::new();
  for bundle in bundles{let capacity=bundle["capacity"].as_u64().filter(|n|matches!(n,4096|8192)).ok_or("invalid default ANE tier")?;if !capacities.insert(capacity){return Err("duplicate default ANE tier".into());}let name=bundle["path"].as_str().filter(|n|crate::model_install::safe_path(n)&&n.starts_with("runtime/ane/")).ok_or("unsafe default ANE bundle path")?;let pin=bundle["sha256"].as_str().ok_or("default ANE bundle pin missing")?;crate::download::digest(pin)?;let path=config.payload.join(name).canonicalize()?;if !path.starts_with(&config.payload)||!path.is_file()||path.metadata()?.len()>2*1024*1024*1024{return Err("default ANE bundle is not a bounded payload file".into());}
   if cli::digest(&path)?!=pin{return Err("default ANE bundle SHA256 mismatch".into());}paths.push((path,pin.to_string()));}
  for (path,pin) in paths{ane_bundle::install_delivery(&config.project,&assets,&path,&pin,&config.node,&config.model,graphs)?;}result=ane_bundle::bind_delivery(&config.project,&assets,&config.node,&config.model)?;
 }
 let source=result.ok_or("default ANE delivery did not provide both tiers")?;let mut top=bounded(&source,64*1024)?;if top["schema"]!=3||top["kind"]!="tiered-vision"{return Err("default ANE profile is invalid".into());}
 for tier in top["tiers"].as_array_mut().filter(|v|v.len()==2).ok_or("default ANE tiers missing")?{let name=tier["profile"].as_str().filter(|n|matches!(*n,"4096.json"|"8192.json")).ok_or("unexpected default ANE tier path")?;tier["profile"]=json!(format!("{class}/qualified/{name}"));}
 let destination=common.join("tiered.json");ane_bundle::publish_delivery(&destination,&top)?;Ok(destination)
}
pub fn install(config:&Delivery)->Result<PathBuf>{
 let data=descriptor(config)?;let common=location(config)?;let ready=common.join("tiered.json");
 if ready.is_file(){let top=bounded(&ready,64*1024)?;if top["schema"]!=3{return Err("default ANE ready profile is invalid".into());}for tier in top["tiers"].as_array().filter(|v|v.len()==2).ok_or("default ANE ready tiers missing")?{let name=tier["profile"].as_str().filter(|n|crate::model_install::safe_path(n)).ok_or("unsafe ready ANE tier")?;let path=common.join(name);if tier["sha256"]!=cli::digest(&path)?{return Err("default ANE ready tier pin mismatch".into());}crate::ane_native::Profile::open_for_preparation(&path).map_err(|e|e.1)?;}return Ok(ready);}
 let current=ane_bundle::identity()?;let selected=select(&data,&config.node,&current);if selected.len()>2{return Err("ambiguous default ANE delivery choices".into());}let mut failures=vec![];
 for (row,graphs) in selected{match install_row(config,&common,row,graphs){Ok(profile)=>return Ok(profile),Err(error)=>failures.push(error.to_string())}}
 Err(format!("default ANE assets unavailable; GPU service remains active: {}",failures.join("; ")).into())
}
pub fn catalog(config:&Delivery,worker:&Path)->Result<PathBuf>{let os=ane_bundle::identity()?.os_build;let hash=format!("{:x}",Sha256::digest(format!("{}:{}:{}:{}",config.project.display(),config.sha256,cli::digest(worker)?,os).as_bytes()));Ok(PathBuf::from(std::env::var_os("HOME").ok_or("HOME missing")?).join("Library/Application Support/JustFit/native-vision").join(hash).join("native-catalog.json"))}
pub fn run_cli(args:&[String])->Result<()>{if args.len()!=1{return Err("ANE bootstrap requires one owned delivery configuration".into());}let data=bounded(Path::new(&args[0]),64*1024)?;let config:Delivery=serde_json::from_value(data)?;let profile=install(&config)?;println!("{}",profile.display());Ok(())}
#[cfg(test)]mod tests{use super::*;
 #[test]fn defaults_prefer_exact_compiled_build_and_keep_graph_fallback(){let current=ane_bundle::Identity{chip:"Apple M4 Pro".into(),os_build:"build".into()};let data=json!({"precompiled":[{"node":"m4","chip":"Apple M4 Pro","os_build":"wrong"},{"node":"m4","chip":"Apple M4 Pro","os_build":"build"}],"graphs":[{"node":"m4","chip":"Apple M4 Pro"},{"node":"m6","chip":"Apple M6 Pro"}]});let rows=select(&data,"m4",&current);assert_eq!(rows.len(),2);assert_eq!(rows[0].0["os_build"],"build");assert!(!rows[0].1);assert!(rows[1].1);}
}
