//! Prepared RGB pixels shared within a strict byte-budgeted LRU.
use crate::{media,Result};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{collections::VecDeque,sync::{Arc,Mutex}};
const BUDGET:usize=256*1024*1024;
struct Entry{key:[u8;32],image:Arc<media::Image>}
#[derive(Default)]
struct State{entries:VecDeque<Entry>,bytes:usize}
#[derive(Default)]
pub struct Cache(Mutex<State>);
impl Cache{
    pub fn prepare(&self,payload:&[u8],pixels:usize)->Result<Arc<media::Image>>{
        let mut digest=Sha256::new();digest.update((pixels as u32).to_le_bytes());digest.update(payload);let key:[u8;32]=digest.finalize().into();
        {
            let mut state=self.0.lock().unwrap();if let Some(index)=state.entries.iter().position(|entry|entry.key==key){let entry=state.entries.remove(index).unwrap();let image=entry.image.clone();state.entries.push_back(entry);return Ok(image)}
        }
        let image=Arc::new(media::prepare(payload,pixels)?);let size=image.pixels.len();
        if size<=BUDGET{
            let mut state=self.0.lock().unwrap();
            if let Some(index)=state.entries.iter().position(|entry|entry.key==key){let old=state.entries.remove(index).unwrap();state.bytes-=old.image.pixels.len();}
            while state.bytes+size>BUDGET{let old=state.entries.pop_front().unwrap();state.bytes-=old.image.pixels.len();}
            state.bytes+=size;state.entries.push_back(Entry{key,image:image.clone()});
        }
        Ok(image)
    }
    pub fn clear(&self){*self.0.lock().unwrap()=State::default();}
    pub fn snapshot(&self,request_bytes:usize)->Value{let state=self.0.lock().unwrap();json!({"entries":state.entries.len(),"bytes":state.bytes,"budget_bytes":BUDGET,"request_bytes":request_bytes,"request_budget_bytes":BUDGET})}
}
