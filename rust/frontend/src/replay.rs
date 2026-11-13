//! Replay schema-1 crash traces without a Python transport or interpreter.
use base64::{Engine as _};
use justfit_native_client_prototype::{read_frame,Frame,VERSION,READY,START,TOKENS,MASK,DONE,ERROR,STATUS,PROGRESS};
use serde_json::Value;
use std::{fmt,fs,io::{self,Read,Write},os::{fd::AsRawFd,unix::process::ExitStatusExt},path::Path,process::{Child,ChildStdout,Command,Stdio},sync::{Arc,Condvar,Mutex,atomic::{AtomicBool,AtomicI32,Ordering}},thread::{self,JoinHandle},time::{Duration,Instant,SystemTime,UNIX_EPOCH}};
type Result<T>=std::result::Result<T,Box<dyn std::error::Error>>;
#[derive(Debug)]
struct EngineFailure(String);
impl fmt::Display for EngineFailure{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{self.0.fmt(f)}}
impl std::error::Error for EngineFailure{}
fn engine_error(message:impl Into<String>)->Box<dyn std::error::Error>{Box::new(EngineFailure(message.into()))}
static INTERRUPTED:AtomicI32=AtomicI32::new(0);
extern "C" fn interrupt(signal:libc::c_int){INTERRUPTED.store(signal,Ordering::Relaxed);}
struct Signals{int:libc::sigaction,term:libc::sigaction}
impl Signals{
    fn install()->io::Result<Self>{
        INTERRUPTED.store(0,Ordering::Relaxed);
        let mut action=unsafe{std::mem::zeroed::<libc::sigaction>()};let mut old=Self{int:unsafe{std::mem::zeroed()},term:unsafe{std::mem::zeroed()}};
        action.sa_sigaction=interrupt as *const () as usize;
        unsafe{libc::sigemptyset(&mut action.sa_mask);}
        if unsafe{libc::sigaction(libc::SIGINT,&action,&mut old.int)}!=0{return Err(io::Error::last_os_error());}
        if unsafe{libc::sigaction(libc::SIGTERM,&action,&mut old.term)}!=0{let error=io::Error::last_os_error();unsafe{libc::sigaction(libc::SIGINT,&old.int,std::ptr::null_mut());}return Err(error);}
        Ok(old)
    }
}
impl Drop for Signals{fn drop(&mut self){unsafe{libc::sigaction(libc::SIGINT,&self.int,std::ptr::null_mut());libc::sigaction(libc::SIGTERM,&self.term,std::ptr::null_mut());}}}
fn check_interrupt()->Result<()>{if INTERRUPTED.load(Ordering::Relaxed)!=0{Err(engine_error("replay interrupted"))}else{Ok(())}}

#[derive(Clone,Copy)]
struct Policy{timeout:Duration,shutdown:Duration}
const POLICY:Policy=Policy{timeout:Duration::from_secs(600),shutdown:Duration::from_secs(15)};
struct Input{command:Vec<String>,frames:Vec<(u64,Vec<u8>)>}
fn input(document:&Value)->Result<Input>{
    // Python's schema comparison accepts JSON true as equal to integer one.
    if document["schema_version"]!=1&&document["schema_version"]!=true{return Err("unsupported JustFit crash trace".into());}
    let command=document["command"].as_array().filter(|a|!a.is_empty()).ok_or("crash trace is missing its replay command or frames")?.iter().map(|v|v.as_str().filter(|s|!s.is_empty()).map(str::to_string).ok_or("crash trace has an invalid command")).collect::<std::result::Result<Vec<_>,_>>()?;
    let entries=document["frames"].as_array().ok_or("crash trace is missing its replay command or frames")?;let mut frames=vec![];
    for entry in entries{
        if entry["direction"]!="client_to_engine"{continue;}
        if entry.get("omitted_bytes").is_some(){return Err("crash trace omits a client-to-engine frame too large to record; it cannot be replayed".into());}
        let offset=entry["offset_micros"].as_u64().ok_or("crash trace contains an invalid outgoing frame")?;
        let bytes=entry["frame_base64"].as_str().ok_or("crash trace contains an invalid outgoing frame")?;
        let decoder=base64::engine::GeneralPurpose::new(&base64::alphabet::STANDARD,base64::engine::GeneralPurposeConfig::new().with_decode_allow_trailing_bits(true));
        frames.push((offset,decoder.decode(bytes)?));
    }
    if frames.is_empty(){return Err("crash trace contains no client-to-engine frames".into());}Ok(Input{command,frames})
}
fn refresh_deadline(bytes:&mut [u8],now:u64){
    if bytes.len()<110||u16::from_le_bytes([bytes[6],bytes[7]])!=1{return;}
    let remaining=u64::from_le_bytes(bytes[34..42].try_into().unwrap());bytes[26..34].copy_from_slice(&now.saturating_add(remaining).to_le_bytes());
}
fn nonblocking(fd:libc::c_int)->io::Result<()>{
    let flags=unsafe{libc::fcntl(fd,libc::F_GETFL)};
    if flags<0||unsafe{libc::fcntl(fd,libc::F_SETFL,flags|libc::O_NONBLOCK)}<0{return Err(io::Error::last_os_error());}Ok(())
}
fn poll(fd:libc::c_int,events:libc::c_short)->io::Result<()>{
    let mut descriptor=libc::pollfd{fd,events,revents:0};
    if unsafe{libc::poll(&mut descriptor,1,100)}<0{let error=io::Error::last_os_error();if error.kind()!=io::ErrorKind::Interrupted{return Err(error);}}Ok(())
}
#[derive(Default)]
struct ReaderState{ready:bool,done:bool,error:Option<String>}
type Shared=Arc<(Mutex<ReaderState>,Condvar)>;
struct Drain{stdout:ChildStdout,stop:Arc<AtomicBool>}
impl Read for Drain{
    fn read(&mut self,bytes:&mut [u8])->io::Result<usize>{
        loop{
            if self.stop.load(Ordering::Acquire){return Err(io::Error::other("replay reader stopped"));}
            match self.stdout.read(bytes){
                Err(error) if error.kind()==io::ErrorKind::WouldBlock=>poll(self.stdout.as_raw_fd(),libc::POLLIN)?,
                Err(error) if error.kind()==io::ErrorKind::Interrupted=>(),
                result=>return result,
            }
        }
    }
}
fn engine_frame(reader:&mut impl Read)->io::Result<Option<Frame>> {
    let mut header=[0u8;16];
    loop{match reader.read(&mut header[..1]){Ok(0)=>return Ok(None),Ok(_)=>break,Err(error) if error.kind()==io::ErrorKind::Interrupted=>(),Err(error)=>return Err(error)}}
    reader.read_exact(&mut header[1..])?;
    let kind=u16::from_le_bytes([header[6],header[7]]);
    if &header[..4]==b"SPLH"&&u16::from_le_bytes([header[4],header[5]])==VERSION
        &&![READY,START,TOKENS,MASK,DONE,ERROR,STATUS,PROGRESS].contains(&kind){
        return Err(io::Error::new(io::ErrorKind::InvalidData,"frame type is not a native protocol event"));
    }
    read_frame(&mut io::Cursor::new(header).chain(reader))
}
struct Owned{child:Child,stop:Arc<AtomicBool>,reader:Option<JoinHandle<()>>,shutdown:Duration}
impl Drop for Owned{
    fn drop(&mut self){
        self.child.stdin.take();
        if !matches!(self.child.try_wait(),Ok(Some(_))){
            let _=unsafe{libc::kill(self.child.id() as libc::pid_t,libc::SIGTERM)};
            let deadline=Instant::now()+self.shutdown;
            while Instant::now()<deadline{
                if matches!(self.child.try_wait(),Ok(Some(_))){break;}
                thread::sleep(Duration::from_millis(10));
            }
            if !matches!(self.child.try_wait(),Ok(Some(_))){let _=self.child.kill();let _=self.child.wait();}
        }
        self.stop.store(true,Ordering::Release);
        if let Some(reader)=self.reader.take(){let _=reader.join();}
    }
}
fn check(state:&ReaderState,need_input:bool)->Result<()>{
    if let Some(error)=&state.error{return Err(engine_error(error.clone()));}
    if need_input&&state.done{return Err(engine_error("replay engine reached EOF before all input was sent"));}Ok(())
}
fn replay(parsed:Input,policy:Policy)->Result<i32>{
    let child=Command::new(&parsed.command[0]).args(&parsed.command[1..]).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit()).spawn()?;
    let stop=Arc::new(AtomicBool::new(false));let shared:Shared=Arc::new((Mutex::new(ReaderState::default()),Condvar::new()));
    let mut owned=Owned{child,stop:stop.clone(),reader:None,shutdown:policy.shutdown};
    let stdout=owned.child.stdout.take().ok_or("replay engine stdout missing")?;nonblocking(stdout.as_raw_fd())?;
    let changed=shared.clone();owned.reader=Some(thread::spawn(move||{
        let mut output=Drain{stdout,stop};
        let result=(||->io::Result<()>{
            while let Some(frame)=engine_frame(&mut output)?{
                if frame.kind==READY{let mut state=changed.0.lock().unwrap();state.ready=true;changed.1.notify_all();}
            }Ok(())
        })();
        let mut state=changed.0.lock().unwrap();if let Err(error)=result{state.error=Some(error.to_string());}state.done=true;changed.1.notify_all();
    }));
    let deadline=Instant::now()+policy.timeout;
    {
        let mut state=shared.0.lock().unwrap();
        while !state.ready&&!state.done{
            check_interrupt()?;
            let remaining=deadline.saturating_duration_since(Instant::now());if remaining.is_zero(){return Err(io::Error::new(io::ErrorKind::TimedOut,"replay engine did not become ready").into());}
            state=shared.1.wait_timeout(state,remaining.min(Duration::from_millis(100))).unwrap().0;
        }
        if !state.ready&&state.error.is_none(){return Err(engine_error("replay engine reached EOF before ReadyEvent"));}check(&state,false)?;
    }
    let started=Instant::now();let first=parsed.frames[0].0;
    let stdin=owned.child.stdin.as_mut().ok_or("replay engine stdin missing")?;nonblocking(stdin.as_raw_fd())?;
    for (offset,mut bytes) in parsed.frames{
        let target=started.checked_add(Duration::from_micros(offset.saturating_sub(first))).ok_or("crash trace offset exceeds clock range")?;
        {
            let mut state=shared.0.lock().unwrap();
            loop{check_interrupt()?;check(&state,true)?;let remaining=target.saturating_duration_since(Instant::now());if remaining.is_zero(){break;}state=shared.1.wait_timeout(state,remaining.min(Duration::from_millis(100))).unwrap().0;}
        }
        let now=SystemTime::now().duration_since(UNIX_EPOCH)?.as_micros().min(u64::MAX as u128) as u64;refresh_deadline(&mut bytes,now);
        let deadline=Instant::now()+policy.timeout;let mut position=0;
        while position<bytes.len(){
            check_interrupt()?;
            check(&shared.0.lock().unwrap(),true)?;
            if Instant::now()>=deadline{return Err(io::Error::new(io::ErrorKind::TimedOut,"replay engine input write timed out").into());}
            match stdin.write(&bytes[position..]){
                Ok(0)=>return Err(io::Error::new(io::ErrorKind::BrokenPipe,"replay engine input accepted zero bytes").into()),
                Ok(written)=>position+=written,
                Err(error) if error.kind()==io::ErrorKind::Interrupted=>(),
                Err(error) if error.kind()==io::ErrorKind::WouldBlock=>poll(stdin.as_raw_fd(),libc::POLLOUT)?,
                Err(error)=>return Err(error.into()),
            }
        }
    }
    owned.child.stdin.take();let deadline=Instant::now()+policy.timeout;
    let status=loop{
        check_interrupt()?;
        if let Some(status)=owned.child.try_wait()?{break status;}
        check(&shared.0.lock().unwrap(),false)?;
        if Instant::now()>=deadline{return Err(engine_error("replay engine did not finish before timeout"));}
        let state=shared.0.lock().unwrap();let _guard=shared.1.wait_timeout(state,Duration::from_millis(100)).unwrap();
    };
    let state=shared.0.lock().unwrap();let(state,_)=shared.1.wait_timeout_while(state,Duration::from_secs(5),|state|!state.done).unwrap();check(&state,false)?;drop(state);
    Ok(status.code().unwrap_or_else(||-status.signal().unwrap_or(1)))
}
pub fn run_cli(args:&[String])->Result<()>{
    let [path]=args else{return Err("usage: justfit replay TRACE".into());};
    let document:Value=serde_json::from_slice(&fs::read(Path::new(path))?)?;let parsed=input(&document)?;
    let signals=Signals::install()?;
    let result=replay(parsed,POLICY);let signal=INTERRUPTED.load(Ordering::Relaxed);drop(signals);
    if signal!=0{std::process::exit(128+signal);}
    match result{
        Ok(code)=>std::process::exit(code),
        Err(error) if error.downcast_ref::<EngineFailure>().is_some()=>{eprintln!("justfit: error: {error}");std::process::exit(1);},
        Err(error)=>Err(error),
    }
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn only_deadline_changes_even_for_nan_or_invalid_wire_header(){
        let mut bytes=vec![0x7f;114];bytes[54..58].copy_from_slice(&0x7fa00001u32.to_le_bytes());bytes[6..8].copy_from_slice(&1u16.to_le_bytes());bytes[34..42].copy_from_slice(&9u64.to_le_bytes());let original=bytes.clone();refresh_deadline(&mut bytes,100);assert_eq!(&bytes[26..34],109u64.to_le_bytes());assert_eq!(&bytes[..26],&original[..26]);assert_eq!(&bytes[34..],&original[34..]);refresh_deadline(&mut bytes,u64::MAX);assert_eq!(&bytes[26..34],u64::MAX.to_le_bytes());
        let mut short=original[..109].to_vec();let original_short=short.clone();refresh_deadline(&mut short,1);assert_eq!(short,original_short);
        let mut cancel=original;cancel[6..8].copy_from_slice(&2u16.to_le_bytes());let original_cancel=cancel.clone();refresh_deadline(&mut cancel,1);assert_eq!(cancel,original_cancel);
    }
    #[test]
    fn client_frame_headers_reject_before_reading_large_payload(){
        let mut header=b"SPLH".to_vec();header.extend(VERSION.to_le_bytes());header.extend(1u16.to_le_bytes());header.extend((256u64*1024*1024).to_le_bytes());
        let error=engine_frame(&mut io::Cursor::new(header)).unwrap_err();assert!(error.to_string().contains("not a native protocol event"));
    }
    #[test]
    fn input_matches_python_noncanonical_base64_tail_bits(){
        let parsed=input(&serde_json::json!({"schema_version":true,"command":["/usr/bin/true"],"frames":[{"direction":"client_to_engine","offset_micros":0,"frame_base64":"Zh=="}]})).unwrap();assert_eq!(parsed.frames[0].1,b"f");
    }
    #[test]
    fn ready_timeout_reaps_owned_child(){
        let start=Instant::now();let error=replay(Input{command:vec!["/bin/sleep".into(),"30".into()],frames:vec![(0,vec![1])]},Policy{timeout:Duration::from_millis(100),shutdown:Duration::from_millis(50)}).unwrap_err();assert!(error.to_string().contains("did not become ready"));assert!(start.elapsed()<Duration::from_secs(3));
    }
}
