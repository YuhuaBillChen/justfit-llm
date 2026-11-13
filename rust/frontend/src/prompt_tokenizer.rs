//! Reuse immutable tokens only at a verified literal message-end boundary.
use crate::{bad,Result};
use serde_json::{json,Value};
use std::{collections::VecDeque,sync::Mutex};
use tokenizers::Tokenizer;
const MARKER:&str="<|im_end|>";
const BUDGET:usize=8*1024*1024;
const CAPACITY:usize=4;
struct Entry{key:String,tokens:Vec<u32>,bytes:usize}
#[derive(Default)]
struct State{entries:VecDeque<Entry>,bytes:usize,hits:u64,reused:u64}
pub struct Cache{marker:Option<u32>,state:Mutex<State>,generated:bool}
fn supported(config:&Value)->bool{
    if config["model"]["type"]!="BPE"||config["model"].get("dropout").is_some_and(|value|!value.is_null()&&value.as_f64()!=Some(0.)){return false}
    let pre=&config["pre_tokenizer"];let parts=if pre["type"]=="Sequence"{pre["pretokenizers"].as_array().cloned().unwrap_or_default()}else{vec![pre.clone()]};
    let kinds:Vec<_>=parts.iter().map(|part|part["type"].as_str().unwrap_or("")).collect();
    if kinds!=["ByteLevel"]&&kinds!=["Split","ByteLevel"]{return false}
    if parts.last().is_some_and(|part|part["add_prefix_space"]==true){return false}
    for(key,kind)in [("normalizer","NFC"),("post_processor","ByteLevel")]{if !config[key].is_null()&&config[key]["type"]!=kind{return false}}
    let Some(tokens)=config["added_tokens"].as_array()else{return false};
    let markers:Vec<_>=tokens.iter().filter(|token|token["content"].as_str().is_some_and(|s|s.contains(MARKER))).collect();
    markers.len()==1&&markers[0]["content"]==MARKER&&["normalized","lstrip","rstrip","single_word"].iter().all(|key|markers[0][*key]==false)
}
// Match the released Python 3.12 quota unit, including compact Unicode width.
fn string_charge(text:&str)->usize{
    let max=text.chars().map(u32::from).max().unwrap_or(0);let count=text.chars().count();
    if max<128{41+count}else if max<256{57+count+if count==1{3}else{0}}else if max<65536{56+2*(count+1)}else{56+4*(count+1)}
}
impl Cache{
    pub fn new(tokenizer:&Tokenizer)->Self{
        let config=tokenizer.to_string(false).ok().and_then(|text|serde_json::from_str::<Value>(&text).ok());
        let marker=config.as_ref().filter(|value|supported(value)).and_then(|_|tokenizer.token_to_id(MARKER));
        // JUSTFIT_REUSE_GENERATED_TOKENS=1: a finished reply's own token ids
        // stand for its text when a later prompt repeats it (remember).
        let generated=marker.is_some()&&std::env::var("JUSTFIT_REUSE_GENERATED_TOKENS").is_ok_and(|value|value=="1");
        Self{marker,state:Mutex::new(State::default()),generated}
    }
    pub fn reuses_generated(&self)->bool{self.generated}
    // A prompt and the reply generated from it, as the sampled token ids: a
    // later prompt that repeats this text verbatim reuses them instead of
    // re-tokenizing it, whose canonical split may differ from the sampled one.
    pub fn remember(&self,key:String,tokens:Vec<u32>){
        if !self.generated||tokens.last()!=self.marker.as_ref()||!key.ends_with(MARKER){return}
        let bytes=string_charge(&key)+33+tokens.len()*4;
        if bytes>BUDGET{return}
        let mut state=self.state.lock().unwrap();
        if let Some(index)=state.entries.iter().position(|entry|entry.key==key){let old=state.entries.remove(index).unwrap();state.bytes-=old.bytes;}
        state.bytes+=bytes;state.entries.push_back(Entry{key,tokens,bytes});
        while state.entries.len()>CAPACITY||state.bytes>BUDGET{let old=state.entries.pop_front().unwrap();state.bytes-=old.bytes;}
    }
    pub fn snapshot(&self)->Value{let state=self.state.lock().unwrap();json!({"enabled":self.marker.is_some(),"reuse_generated":self.generated,"entries":state.entries.len(),"bytes":state.bytes,"budget_bytes":BUDGET,"capacity":CAPACITY,"hits":state.hits,"reused_tokens":state.reused})}
    pub fn encode(&self,tokenizer:&Tokenizer,text:&str)->Result<Vec<u32>>{
        let encode=|text:&str|tokenizer.encode(text,false).map(|encoded|encoded.get_ids().to_vec()).map_err(|_|bad("content could not be tokenized"));
        let Some(marker)=self.marker else{return encode(text)};
        let Some(boundary)=text.rfind(MARKER).map(|index|index+MARKER.len())else{return encode(text)};
        let prefix=&text[..boundary];if prefix.chars().count()<4096{return encode(text)}
        let(key,mut tokens)={
            let mut state=self.state.lock().unwrap();let index=state.entries.iter().enumerate().filter(|(_,entry)|prefix.starts_with(&entry.key)).max_by_key(|(_,entry)|entry.key.len()).map(|(index,_)|index);
            match index{
                Some(index)=>{let entry=state.entries.remove(index).unwrap();state.hits+=1;state.reused+=entry.tokens.len() as u64;let result=(entry.key.clone(),entry.tokens.clone());state.entries.push_back(entry);result},
                None=>(String::new(),Vec::new()),
            }
        };
        if key!=prefix{
            tokens.extend(encode(&prefix[key.len()..])?);
            if tokens.last()!=Some(&marker){return encode(text)}
            let bytes=string_charge(prefix)+33+tokens.len()*4;
            if bytes<=BUDGET{
                let mut state=self.state.lock().unwrap();
                for remove in [&key,prefix]{if let Some(index)=state.entries.iter().position(|entry|entry.key==remove){let old=state.entries.remove(index).unwrap();state.bytes-=old.bytes;}}
                state.bytes+=bytes;state.entries.push_back(Entry{key:prefix.into(),tokens:tokens.clone(),bytes});
                while state.entries.len()>CAPACITY||state.bytes>BUDGET{let old=state.entries.pop_front().unwrap();state.bytes-=old.bytes;}
            }
        }
        tokens.extend(encode(&text[boundary..])?);Ok(tokens)
    }
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn python_compact_unicode_quota_units(){assert_eq!(string_charge(""),41);assert_eq!(string_charge("a"),42);assert_eq!(string_charge("é"),61);assert_eq!(string_charge("汉"),60);assert_eq!(string_charge("😀"),64);}
}
