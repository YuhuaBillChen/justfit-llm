//! Startup-prepared templates; literal rendering remains in the Rust frontend.
use crate::{bad,Frontend,Result};
use serde::{Deserialize,Serialize};
use serde_json::{json,Value};
use std::collections::BTreeMap;
#[derive(Clone,Deserialize,Serialize)]
pub struct Template{pub source:String,pub later_system:String}
#[derive(Clone,Deserialize,Serialize)]
pub struct Templates{pub named:bool,pub entries:BTreeMap<String,Template>}
impl Templates{
    pub fn validate(&self)->Result<()>{
        if self.entries.is_empty()||!self.entries.contains_key(if self.named{"default"}else{""})||self.entries.values().any(|template|template.source.is_empty()||!["native","patched","unsupported"].contains(&template.later_system.as_str())){return Err(bad("invalid prepared chat templates"))}Ok(())
    }
    pub fn select(&self,body:&Value)->&Template{
        let key=if !self.named{""}else if body["tools"].as_array().is_some_and(|tools|!tools.is_empty())&&self.entries.contains_key("tool_use"){"tool_use"}else{"default"};&self.entries[key]
    }
    pub fn snapshot(&self)->Value{
        if self.named{json!({"later_system":self.entries.iter().map(|(name,template)|(name.clone(),json!(template.later_system))).collect::<serde_json::Map<_,_>>()})}
        else{json!({"later_system":self.entries[""].later_system})}
    }
    pub fn raw(source:String)->Self{Self{named:false,entries:BTreeMap::from([(String::new(),Template{source,later_system:"unsupported".into()})])}}
}
pub fn accepts(template:&Template,messages:&Value)->Result<()>{
    if template.later_system=="unsupported"&&messages.as_array().is_some_and(|messages|messages.iter().skip(1).any(|message|message["role"]=="system")){return Err(bad("this model's chat template does not accept system messages after the first message"))}Ok(())
}
pub fn probe(front:&Frontend,source:&str)->bool{
    const MARKER:&str="JustFit later-system canary 5d0c8e";
    for leading in [true,false]{
        let mut messages=vec![];if leading{messages.push(json!({"role":"system","content":"Leading instructions"}))}
        messages.extend([json!({"role":"user","content":"First question"}),json!({"role":"assistant","content":"First answer"}),json!({"role":"system","content":MARKER}),json!({"role":"user","content":"Next question"})]);
        let Ok(rendered)=front.render_messages(&json!({"add_generation_prompt":true}),json!(messages),source)else{return false};
        let(before,marker,after)=(rendered.find("First answer"),rendered.find(MARKER),rendered.find("Next question"));
        if rendered.matches(MARKER).count()!=1||!matches!((before,marker,after),(Some(before),Some(marker),Some(after)) if before+"First answer".len()<=marker&&marker+MARKER.len()<=after){return false}
    }true
}
