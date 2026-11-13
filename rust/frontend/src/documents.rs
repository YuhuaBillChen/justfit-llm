//! Bounded inline-PDF normalization. PDFium runs only in a disposable process.
mod pdfium;
mod png;
mod sandbox;
use crate::{bad,ApiError,Result};
use base64::{engine::general_purpose::GeneralPurpose,alphabet,engine::general_purpose::GeneralPurposeConfig,Engine as _};
use serde::{Deserialize,Serialize};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{collections::VecDeque,fs::File,io::{Read,Write,Seek,SeekFrom},os::fd::FromRawFd,path::{Path,PathBuf},process::{Child,Stdio},sync::{Arc,Condvar,Mutex},time::{Duration,Instant}};
pub fn has_documents(body:&Value)->bool {
    body["messages"].as_array().is_some_and(|ms|ms.iter().any(|m|m["content"].as_array().is_some_and(|parts|parts.iter().any(|p|p["type"]=="file"))))
}
const MIB:usize=1024*1024;
const PDF_BYTES:usize=64*MIB;
const MAX_PAGES:usize=64;
const OUTPUT_BYTES:usize=64*MIB;
const MEMORY_BYTES:u64=1024*1024*1024;
pub struct Cancellation {pub flag:Arc<std::sync::atomic::AtomicBool>}
impl Default for Cancellation {fn default()->Self{Self::new()}}
impl Cancellation {
    pub fn new()->Self{Self{flag:Arc::new(std::sync::atomic::AtomicBool::new(false))}}
    pub fn check(flag:&std::sync::atomic::AtomicBool)->Result<()> {
        if flag.load(std::sync::atomic::Ordering::Acquire){Err(ApiError(499,"request cancelled".into(),"request_cancelled".into()))}else{Ok(())}
    }
}
impl Drop for Cancellation {fn drop(&mut self){self.flag.store(true,std::sync::atomic::Ordering::Release)}}
const PREFIX:&str="data:application/pdf;base64,";
#[derive(Clone,Deserialize,Serialize)]
pub struct Config {pub library:PathBuf,pub zlib:PathBuf}
impl Default for Config {
    fn default()->Self {
        let parent=std::env::current_exe().ok().and_then(|p|p.parent().map(Path::to_path_buf)).unwrap_or_default();
        Self{library:std::env::var_os("JUSTFIT_PDFIUM_LIBRARY").map(PathBuf::from).unwrap_or_else(||parent.join("libpdfium.dylib")),
            zlib:std::env::var_os("JUSTFIT_PDF_ZLIB_LIBRARY").map(PathBuf::from).unwrap_or_else(||PathBuf::from("/usr/lib/libz.dylib"))}
    }
}
#[derive(Clone,Deserialize,Serialize)]
pub struct Page {pub text:String,pub image_url:String}
impl Page {fn size(&self)->usize{self.text.chars().count()*4+self.image_url.len()}}
#[derive(Serialize,Deserialize)]
pub struct Limits {pub pages:usize,pub page_pixels:usize,pub text_characters:usize,pub rendered_bytes:usize,pub request_bytes:usize}
pub struct Budget {deadline:Option<Instant>,bytes:usize,pages:usize}
impl Budget {
    fn new(timeout:Duration)->Self {Self{deadline:crate::deadline::mono_deadline(timeout),bytes:PDF_BYTES,pages:MAX_PAGES}}
    fn check(&self)->Result<()> {if self.deadline.is_some_and(|d|Instant::now()>=d){Err(crate::deadline::expired())}else{Ok(())}}
    fn remaining(&self)->Result<Duration>{self.check()?;Ok(self.deadline.map_or(Duration::MAX,|d|d.saturating_duration_since(Instant::now())))}
    fn charge(&mut self,n:usize)->Result<()> {self.check()?;self.bytes=self.bytes.checked_sub(n).ok_or_else(||bad("PDF sources and rendered pages exceed the shared request size limit"))?;Ok(())}
    fn charge_pages(&mut self,n:usize)->Result<()> {self.pages=self.pages.checked_sub(n).ok_or_else(||bad("documents exceed the request page limit"))?;Ok(())}
}
type Cached=([u8;32],Arc<Vec<Page>>,usize);
#[derive(Default)]
struct State {rendering:bool,cache:VecDeque<Cached>,bytes:usize,renders:usize,hits:usize}
pub struct Documents {pub config:Config,state:Mutex<State>,ready:Condvar}
struct RenderLock<'a>(&'a Documents);
impl Drop for RenderLock<'_>{fn drop(&mut self){self.0.state.lock().unwrap().rendering=false;self.0.ready.notify_all()}}
impl Documents {
    pub fn new(config:Config)->Self {Self{config,state:Mutex::new(State::default()),ready:Condvar::new()}}
    pub fn available(&self)->bool{self.config.library.is_file()&&png::available(&self.config.zlib)}
    pub fn stats(&self)->Value {let s=self.state.lock().unwrap();json!({"cache_entries":s.cache.len(),"cache_bytes":s.bytes,"renders":s.renders,"cache_hits":s.hits,"rendering":s.rendering})}
    fn cached(&self,key:&[u8;32],budget:&mut Budget)->Result<Option<Arc<Vec<Page>>>> {
        let mut s=self.state.lock().unwrap();let Some(index)=s.cache.iter().position(|v|&v.0==key)else{return Ok(None)};
        let entry=s.cache.remove(index).unwrap();let pages=entry.1.clone();s.cache.push_back(entry);s.hits+=1;drop(s);
        budget.charge_pages(pages.len())?;budget.charge(pages.iter().map(Page::size).sum())?;Ok(Some(pages))
    }
    pub fn parts(&self,file:&Value,budget:&mut Budget,check:&dyn Fn()->Result<()>)->Result<Vec<Value>> {
        if !file.is_object(){return Err(bad("file must be an object"))}
        if file.get("file_id").is_some_and(|v|!v.is_null())||file.get("file_url").is_some_and(|v|!v.is_null()){return Err(bad("file_id and file_url are not supported; use inline file_data"))}
        if file.get("filename").is_some_and(|v|!v.is_null()&&!v.is_string()){return Err(bad("filename must be a string"))}
        let mut data=file["file_data"].as_str().ok_or_else(||bad("file_data must contain a base64 PDF"))?;
        if data.starts_with("data:"){data=data.strip_prefix(PREFIX).ok_or_else(||bad("only application/pdf file data is supported"))?}
        if data.len()>4*PDF_BYTES.div_ceil(3){return Err(bad("PDF document exceeds the size limit"))}
        budget.check()?;check()?;
        // Python validate=True accepts noncanonical trailing pad bits.
        let decoder=GeneralPurpose::new(&alphabet::STANDARD,GeneralPurposeConfig::new().with_decode_allow_trailing_bits(true));
        let payload=decoder.decode(data).map_err(|_|bad("PDF data is not valid base64"))?;
        if payload.len()>PDF_BYTES{return Err(bad("PDF document exceeds the size limit"))}
        budget.charge(payload.len())?;let key:[u8;32]=Sha256::digest(&payload).into();
        let pages=if let Some(pages)=self.cached(&key,budget)?{pages}else{
            let mut s=self.state.lock().unwrap();while s.rendering {budget.check()?;check()?;s=self.ready.wait_timeout(s,Duration::from_millis(10)).unwrap().0}
            s.rendering=true;drop(s);let _lock=RenderLock(self);
            if let Some(pages)=self.cached(&key,budget)?{pages}else{
                let pages=supervised(&self.config,&payload,budget,check)?;budget.charge_pages(pages.len())?;
                budget.charge(pages.iter().map(Page::size).sum())?;
                let size=pages.iter().map(Page::size).sum();let pages=Arc::new(pages);
                let mut s=self.state.lock().unwrap();s.renders+=1;
                if size<=64*MIB {
                    while s.cache.len()>=16||s.bytes+size>64*MIB{if let Some((_,_,n))=s.cache.pop_front(){s.bytes-=n}else{break}}
                    s.bytes+=size;s.cache.push_back((key,pages.clone(),size));
                }pages
            }
        };
        Ok(pages.iter().flat_map(|p|[json!({"type":"text","text":p.text}),json!({"type":"image_url","image_url":{"url":p.image_url}})]).collect())
    }
    /// Expand all documents in request order against one shared byte/page budget.
    pub fn expand(&self,body:&Value,check:&dyn Fn()->Result<()>)->Result<Value> {
        let mut body=body.clone();let mut budget=Budget::new(crate::deadline::request(&body)?);
        let messages=body["messages"].as_array_mut().ok_or_else(||bad("messages must be an array"))?;
        for message in messages {
            let Some(parts)=message["content"].as_array()else{continue};
            if !parts.iter().any(|v|v["type"]=="file"){continue}
            if !matches!(message["role"].as_str(),Some("user"|"tool")){return Err(bad("only user/tool messages support images"))}
            let mut expanded=Vec::new();for part in parts {
                check()?;budget.check()?;
                if part["type"]=="file" {expanded.extend(self.parts(&part["file"],&mut budget,check)?)}else{expanded.push(part.clone())}
            }message["content"]=json!(expanded);
        }Ok(body)
    }
}
fn unavailable(message:&str)->ApiError{ApiError(503,message.into(),"document_unavailable".into())}
fn tempfile()->Result<File> {
    let mut path=std::env::temp_dir().join("justfit-pdf-XXXXXX").as_os_str().as_encoded_bytes().to_vec();path.push(0);
    let fd=unsafe{libc::mkstemp(path.as_mut_ptr().cast())};if fd<0{return Err(unavailable("document worker could not start"))}
    if unsafe{libc::unlink(path.as_ptr().cast())}!=0 || unsafe{libc::fcntl(fd,libc::F_SETFD,libc::FD_CLOEXEC)}<0 {
        unsafe{libc::close(fd)};return Err(unavailable("document worker could not start"))
    }
    Ok(unsafe{File::from_raw_fd(fd)})
}
struct OwnedChild(Child);
impl Drop for OwnedChild {fn drop(&mut self){if self.0.try_wait().ok().flatten().is_none(){let _=self.0.kill();}let _=self.0.wait();}}
#[cfg(target_os="macos")]
fn memory(pid:u32)->Result<u64> {
    #[repr(C)]struct Usage {uuid:[u8;16],values:[u64;32]}
    unsafe extern "C" {fn proc_pid_rusage(pid:i32,flavor:i32,buffer:*mut c_void)->i32;}
    use std::ffi::c_void;let mut usage=Usage{uuid:[0;16],values:[0;32]};
    if unsafe{proc_pid_rusage(pid as i32,2,(&mut usage as *mut Usage).cast())}!=0{return Err(unavailable("document worker memory measurement failed"))}
    Ok(usage.values[6].max(usage.values[7]))
}
#[cfg(not(target_os="macos"))]
fn memory(_pid:u32)->Result<u64>{Err(unavailable("document worker memory measurement failed"))}
fn supervised(config:&Config,payload:&[u8],budget:&mut Budget,check:&dyn Fn()->Result<()>)->Result<Vec<Page>> {
    let remaining=budget.remaining()?;let duration=remaining.min(Duration::from_secs(30));let until=Instant::now()+duration;
    if !config.library.is_file()||!png::available(&config.zlib){return Err(unavailable("document worker could not start"))}
    let limits=Limits{pages:budget.pages.min(MAX_PAGES),page_pixels:1024*1024,text_characters:1_000_000,rendered_bytes:32*MIB,request_bytes:budget.bytes};
    let mut source=tempfile()?;source.write_all(payload).map_err(|_|unavailable("document worker could not start"))?;source.rewind().map_err(|_|unavailable("document worker could not start"))?;
    let mut output=tempfile()?;
    let child=sandbox::command(&std::env::current_exe().map_err(|_|unavailable("document worker could not start"))?,config)?
        .args(["pdf-worker",&serde_json::to_string(config).unwrap(),&serde_json::to_string(&limits).unwrap(),&duration.as_secs_f64().to_string()])
        .stdin(Stdio::from(source)).stdout(Stdio::from(output.try_clone().map_err(|_|unavailable("document worker could not start"))?)).stderr(Stdio::null()).spawn().map_err(|_|unavailable("document worker could not start"))?;
    let mut child=OwnedChild(child);
    let status=loop {
        if let Some(status)=child.0.try_wait().map_err(|_|unavailable("document worker memory measurement failed"))?{break status}
        check()?;
        if Instant::now()>=until {return Err(if remaining<=Duration::from_secs(30){crate::deadline::expired()}else{bad("PDF processing exceeded the time limit")})}
        match memory(child.0.id()) {
            Ok(n) if n>MEMORY_BYTES=>return Err(bad("PDF processing exceeded the memory limit")),
            Err(error)=>{if let Some(status)=child.0.try_wait().map_err(|_|unavailable("document worker memory measurement failed"))?{break status}else{return Err(error)}},_=>()
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    if status.code()==Some(71){return Err(unavailable("PDF sandbox could not start"))}
    if !status.success(){return Err(bad("PDF processing failed or exceeded its resource limit"))}
    output.seek(SeekFrom::Start(0)).map_err(|_|unavailable("document worker returned an invalid result"))?;
    let mut bytes=Vec::new();output.take((OUTPUT_BYTES+1)as u64).read_to_end(&mut bytes).map_err(|_|unavailable("document worker returned an invalid result"))?;
    if bytes.len()>OUTPUT_BYTES{return Err(bad("rendered PDF exceeds the size limit"))}
    let result:Value=serde_json::from_slice(&bytes).map_err(|_|ApiError(500,"document worker returned an invalid result".into(),"document_worker_failed".into()))?;
    if let Some(e)=result.get("error"){return Err(ApiError(e["status"].as_u64().unwrap_or(500)as u16,e["message"].as_str().unwrap_or("document worker returned an invalid result").into(),e["code"].as_str().unwrap_or("document_worker_failed").into()))}
    serde_json::from_value(result["pages"].clone()).map_err(|_|ApiError(500,"document worker returned an invalid result".into(),"document_worker_failed".into()))
}
pub fn worker(config:&str,limits:&str,duration:&str)->Result<()> {
    let config:Config=serde_json::from_str(config).map_err(|_|bad("invalid PDF worker config"))?;
    let limits:Limits=serde_json::from_str(limits).map_err(|_|bad("invalid PDF worker limits"))?;
    let duration:f64=duration.parse().map_err(|_|bad("invalid PDF worker duration"))?;
    if !duration.is_finite()||duration<=0.||duration>30.||limits.pages>64||limits.page_pixels>1024*1024||limits.text_characters>1_000_000||limits.rendered_bytes>32*MIB||limits.request_bytes>PDF_BYTES{return Err(bad("invalid PDF worker limits"))}
    unsafe{
        let core=libc::rlimit{rlim_cur:0,rlim_max:0};libc::setrlimit(libc::RLIMIT_CORE,&core);
        let file=libc::rlimit{rlim_cur:OUTPUT_BYTES as _,rlim_max:OUTPUT_BYTES as _};
        let cpu=libc::rlimit{rlim_cur:duration.ceil().max(1.) as _,rlim_max:duration.ceil().max(1.) as _};
        if libc::setrlimit(libc::RLIMIT_FSIZE,&file)!=0||libc::setrlimit(libc::RLIMIT_CPU,&cpu)!=0{return Err(bad("PDF worker resource limits unavailable"))}
    }
    let mut payload=Vec::new();std::io::stdin().take((PDF_BYTES+1)as u64).read_to_end(&mut payload).map_err(|_|bad("PDF document could not be decoded"))?;
    let result=if payload.len()>PDF_BYTES{Err(bad("PDF document exceeds the size limit"))}else{
        let mut budget=Budget::new(Duration::from_secs_f64(duration));budget.bytes=limits.request_bytes;
        pdfium::render(&payload,&limits,&mut budget,config.library.to_str().ok_or_else(||bad("invalid PDFium library path"))?,config.zlib.to_str().ok_or_else(||bad("invalid PNG compression library path"))?)
    };
    let response=match result{Ok(pages)=>json!({"pages":pages}),Err(e)=>json!({"error":{"status":e.0,"message":e.1,"code":e.2}})};
    serde_json::to_writer(std::io::stdout().lock(),&response).map_err(|_|bad("rendered PDF exceeds the size limit"))?;Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cache_hits_charge_shared_pages_and_bytes() {
        let docs=Documents::new(Config::default());
        let page=Page{text:"page\n".into(),image_url:"data:image/png;base64,AA==".into()};
        let size=page.size();let mut state=docs.state.lock().unwrap();
        state.cache.push_back(([1;32],Arc::new(vec![page]),size));state.bytes=size;drop(state);
        let mut budget=Budget::new(Duration::MAX);budget.pages=1;budget.bytes=size;
        assert!(docs.cached(&[1;32],&mut budget).unwrap().is_some());assert_eq!((budget.pages,budget.bytes),(0,0));
        assert_eq!(docs.cached(&[1;32],&mut budget).err().unwrap().1,"documents exceed the request page limit");
        let mut budget=Budget::new(Duration::MAX);budget.bytes=size-1;
        assert_eq!(docs.cached(&[1;32],&mut budget).err().unwrap().1,"PDF sources and rendered pages exceed the shared request size limit");
    }
    #[test]
    fn cancellation_and_expired_request_precede_worker_launch() {
        let docs=Documents::new(Config::default());
        let input=json!({"timeout":1e-9,"messages":[{"role":"user","content":[{"type":"file","file":{"file_data":"AA=="}}]}]});
        assert_eq!(docs.expand(&input,&||Ok(())).err().unwrap().0,504);
        let mut input=input;input.as_object_mut().unwrap().remove("timeout");
        let flag=std::sync::atomic::AtomicBool::new(true);
        assert_eq!(docs.expand(&input,&||Cancellation::check(&flag)).err().unwrap().0,499);
    }
}
