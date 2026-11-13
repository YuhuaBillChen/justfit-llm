//! Match source vision tensors to precompiled ANE weights without repacking a model.
use crate::{cli,download::Result};
use serde_json::Value;
use std::{collections::BTreeMap,path::Path,sync::{Mutex,OnceLock}};
type Key=(String,String);
static IDENTITIES:OnceLock<Mutex<BTreeMap<Key,String>>>=OnceLock::new();
pub fn sha(engine:&Path,model:&Path)->Result<String>{
    if model.join("vision/model.bin").is_file(){return cli::digest(&model.join("vision/model.bin"));}
    let held=crate::model_hold::hold(model)?;
    let key=(cli::digest(engine)?,cli::digest(&held.path.join("model.json"))?);
    let mut identities=IDENTITIES.get_or_init(||Mutex::new(BTreeMap::new())).lock().map_err(|_|"vision identity lock poisoned")?;
    if let Some(value)=identities.get(&key){return Ok(value.clone());}
    let output=std::process::Command::new(engine).arg("vision-identity").arg(&held.path).output()?;
    if !output.status.success(){return Err(format!("source vision identity: {}",String::from_utf8_lossy(&output.stderr)).into());}
    let value:Value=serde_json::from_slice(&output.stdout)?;
    if value["schema"]!=1||value["source_identity"]!=key.1||!value["bytes"].as_u64().is_some_and(|n|n>0&&n<=2*1024*1024*1024){return Err("native source vision identity mismatch".into());}
    let digest=value["sha256"].as_str().ok_or("native vision digest missing")?;crate::download::digest(digest)?;
    if identities.len()>=16{identities.clear();}identities.insert(key,digest.into());Ok(digest.into())
}
