//! Opt-in, bounded generation-local traces compatible with the 2.3 replay tool.
use base64::{engine::general_purpose::STANDARD,Engine as _};
use justfit_native_client_prototype::{Frame,VERSION};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{collections::VecDeque,fs,io::Write,os::unix::fs::{OpenOptionsExt,PermissionsExt},path::PathBuf,sync::Mutex,time::{Instant,SystemTime,UNIX_EPOCH}};
const BYTES:usize=16*1024*1024;
const ENTRIES:usize=512;
struct Entry{offset:u128,direction:&'static str,bytes:Vec<u8>,omitted:usize,digest:String}
#[derive(Default)]
struct State{command:Vec<String>,generation:u64,pid:u32,started:Option<Instant>,entries:VecDeque<Entry>,bytes:usize,dumped:u64,last:Option<PathBuf>}
pub(super) struct Trace{enabled:bool,state:Mutex<State>}
impl Trace {
    pub fn new()->Self{Self{enabled:std::env::var("JUSTFIT_CRASH_TRACE").as_deref()==Ok("1"),state:Mutex::new(State::default())}}
    pub fn command(&self,command:&[String]){self.state.lock().unwrap().command=command.to_vec();}
    pub fn start(&self,generation:u64,pid:u32){if !self.enabled{return}let mut state=self.state.lock().unwrap();state.generation=generation;state.pid=pid;state.started=Some(Instant::now());state.entries.clear();state.bytes=0;}
    pub fn last(&self)->Option<PathBuf>{self.state.lock().unwrap().last.clone()}
    pub fn record(&self,generation:u64,direction:&'static str,frame:&Frame){
        if !self.enabled{return}
        let mut header=Vec::with_capacity(16);header.extend(b"SPLH");header.extend(VERSION.to_le_bytes());header.extend(frame.kind.to_le_bytes());header.extend((frame.payload.len() as u64).to_le_bytes());
        let size=16+frame.payload.len();let omitted=if size>BYTES{size}else{0};
        let digest=if omitted>0 {let mut sha=Sha256::new();sha.update(&header);sha.update(&frame.payload);format!("{:x}",sha.finalize())}else{header.extend(&frame.payload);String::new()};
        let mut state=self.state.lock().unwrap();if generation!=state.generation||state.started.is_none(){return}
        let offset=state.started.unwrap().elapsed().as_micros();state.bytes+=header.len();state.entries.push_back(Entry{offset,direction,bytes:header,omitted,digest});
        while state.entries.len()>ENTRIES||state.bytes>BYTES{let entry=state.entries.pop_front().unwrap();state.bytes-=entry.bytes.len();}
    }
    pub fn dump(&self,generation:u64,error:&str,returncode:Option<i32>,status:Option<&Value>)->Option<PathBuf>{
        if !self.enabled{return None}
        let document={
            let mut state=self.state.lock().unwrap();
            if generation!=state.generation||state.command.is_empty()||state.dumped==generation{return state.last.clone()}
            state.dumped=generation;
            let frames=state.entries.iter().map(|entry|{
                let mut row=json!({"offset_micros":entry.offset,"direction":entry.direction});
                if entry.omitted>0{row["header_base64"]=json!(STANDARD.encode(&entry.bytes));row["omitted_bytes"]=json!(entry.omitted);row["sha256"]=json!(entry.digest)}else{row["frame_base64"]=json!(STANDARD.encode(&entry.bytes))}row
            }).collect::<Vec<_>>();
            json!({"schema_version":1,"created_at":timestamp(),"generation":generation,"pid":state.pid,"process_returncode":returncode,"command":state.command,"error":{"type":"EngineUnhealthy","message":error},"last_status_json":status.map(Value::to_string),"frames":frames,"omitted_frames":state.entries.iter().filter(|e|e.omitted>0).count()})
        };
        let path=(||->std::io::Result<PathBuf>{
            let directory=PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join("Library/Logs/JustFit/crash");
            if !directory.exists(){fs::create_dir_all(&directory)?;fs::set_permissions(&directory,fs::Permissions::from_mode(0o700))?;}
            let now=SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
            let name=format!("justfit-crash-g{generation}-{}.{:09}-{}.json",now.as_secs(),now.subsec_nanos(),std::process::id());let target=directory.join(name);let temporary=target.with_extension("partial");
            let result=(||{let mut file=fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&temporary)?;serde_json::to_writer(&mut file,&document)?;file.write_all(b"\n")?;file.sync_all()?;fs::rename(&temporary,&target)?;Ok::<_,std::io::Error>(target)})();
            if result.is_err(){let _=fs::remove_file(temporary);}result
        })().ok()?;
        self.state.lock().unwrap().last=Some(path.clone());
        // Only trace files in this explicit opt-in directory are candidates.
        let mut files=fs::read_dir(path.parent()?).ok()?.filter_map(|entry|entry.ok()).filter_map(|entry|{
            let name=entry.file_name();let name=name.to_str()?;
            if !name.starts_with("justfit-crash-g")||!name.ends_with(".json"){return None}
            let metadata=entry.metadata().ok()?;if !metadata.is_file()||entry.file_type().ok()?.is_symlink(){return None}Some((metadata.modified().ok()?,entry.path(),metadata.len()))
        }).collect::<Vec<_>>();files.sort_by_key(|entry|std::cmp::Reverse(entry.0));let mut total=0;
        for(index,(_,file,bytes))in files.into_iter().enumerate(){total+=bytes;if index>=4||total>64*1024*1024{let _=fs::remove_file(file);}}
        Some(path)
    }
}
fn timestamp()->String{
    let now=SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();let seconds=now.as_secs() as libc::time_t;let mut tm=unsafe{std::mem::zeroed::<libc::tm>()};
    if unsafe{libc::gmtime_r(&seconds,&mut tm)}.is_null(){return now.as_secs().to_string()}
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:06}+00:00",tm.tm_year+1900,tm.tm_mon+1,tm.tm_mday,tm.tm_hour,tm.tm_min,tm.tm_sec,now.subsec_micros())
}

#[cfg(test)]
mod tests{
    use super::*;
    fn active()->Trace{Trace{enabled:true,state:Mutex::new(State::default())}}
    #[test]
    fn trace_is_generation_local_and_retains_only_newest_512_frames(){
        let trace=active();trace.start(1,42);
        for index in 0u64..600{trace.record(1,"client_to_engine",&Frame{kind:4,payload:index.to_le_bytes().to_vec()})}
        let state=trace.state.lock().unwrap();assert_eq!(state.entries.len(),512);assert_eq!(state.bytes,512*24);assert_eq!(&state.entries.front().unwrap().bytes[16..],88u64.to_le_bytes());drop(state);
        trace.start(2,43);trace.record(1,"engine_to_client",&Frame{kind:4,payload:vec![0;8]});assert!(trace.state.lock().unwrap().entries.is_empty());
    }
    #[test]
    fn oversized_wire_frame_keeps_only_header_digest_and_cannot_evict_the_whole_ring(){
        let trace=active();trace.start(1,42);let frame=Frame{kind:1,payload:vec![7;BYTES]};trace.record(1,"client_to_engine",&frame);
        let state=trace.state.lock().unwrap();let entry=&state.entries[0];assert_eq!(entry.bytes.len(),16);assert_eq!(entry.omitted,BYTES+16);
        let mut sha=Sha256::new();sha.update(&entry.bytes);sha.update(&frame.payload);assert_eq!(entry.digest,format!("{:x}",sha.finalize()));
    }
    #[test]
    fn ring_byte_limit_applies_even_before_entry_count_limit(){
        let trace=active();trace.start(1,42);let frame=Frame{kind:1,payload:vec![0;1024*1024]};
        for _ in 0..20{trace.record(1,"client_to_engine",&frame)}let state=trace.state.lock().unwrap();assert!(state.bytes<=BYTES);assert_eq!(state.entries.len(),15);
    }
}
