//! Owned native generations, recovery and stale-callback fencing.
use super::{failed,u32_at,u64_at,Admission};
use crate::{ApiError,Result};
use justfit_native_client_prototype::{cancel,read_frame,write_frame,Frame,READY,START,DONE,ERROR,STATUS};
use super::quarantine::{self, Fingerprint, Quarantine};
use serde_json::{json,Value};
use std::{collections::HashMap,io::{self,Write},os::fd::AsRawFd,process::{Child,ChildStdin,ChildStdout,Stdio},sync::{atomic::{AtomicBool,AtomicU64,AtomicU8,Ordering},Arc,Mutex,Condvar,mpsc as sync_mpsc},time::{Duration,Instant}};
use tokio::sync::{mpsc,OwnedSemaphorePermit};
const STARTUP:Duration=Duration::from_secs(600);
const STATUS_WAIT:Duration=Duration::from_secs(30);
const CANCEL_WAIT:Duration=Duration::from_secs(150);
// A blocked native pipe must not pin the writer mutex forever. Each actual
// write resets the same five-second no-progress allowance used by 2.3.
struct Pipe(ChildStdin);
impl Pipe {
    fn new(stdin:ChildStdin)->io::Result<Self>{
        let fd=stdin.as_raw_fd();let flags=unsafe{libc::fcntl(fd,libc::F_GETFL)};
        if flags<0||unsafe{libc::fcntl(fd,libc::F_SETFL,flags|libc::O_NONBLOCK)}<0{return Err(io::Error::last_os_error())}Ok(Self(stdin))
    }
}
impl Write for Pipe {
    fn write(&mut self,bytes:&[u8])->io::Result<usize>{
        let deadline=Instant::now()+Duration::from_secs(5);
        loop {
            match self.0.write(bytes){
                Err(e) if e.kind()==io::ErrorKind::WouldBlock=>(),
                Err(e) if e.kind()==io::ErrorKind::Interrupted=>continue,
                result=>return result,
            }
            let left=deadline.saturating_duration_since(Instant::now());
            if left.is_zero(){return Err(io::Error::new(io::ErrorKind::TimedOut,"native pipe write made no progress for 5 s"))}
            let mut fd=libc::pollfd{fd:self.0.as_raw_fd(),events:libc::POLLOUT,revents:0};
            let result=unsafe{libc::poll(&mut fd,1,left.as_millis().clamp(1,5000) as i32)};
            if result<0 {let error=io::Error::last_os_error();if error.kind()!=io::ErrorKind::Interrupted{return Err(error)}}
        }
    }
    fn flush(&mut self)->io::Result<()>{Ok(())}
}
#[derive(Clone,Copy,PartialEq,Eq)]
pub(super) struct Ready {pub lanes:u32,pub context:u32,pub vision:bool}
#[derive(Default)]
pub(super) struct Terminal {pub state:AtomicU8,pub error:Mutex<Option<ApiError>>}
struct Pending {_permit:OwnedSemaphorePermit,kind:u16,terminal:Arc<Terminal>,fingerprint:Option<Fingerprint>,started:bool}
type Hook=Box<dyn Fn()+Send+Sync>;
#[derive(Default)]
struct Recovery {failures:u32,due:Option<Instant>,fatal:Option<String>,last_error:Option<String>,ready_at:Option<Instant>,restarts:u64}
pub(super) struct Inner {
    trace:super::crash_trace::Trace,
    writer:Mutex<Option<Pipe>>,
    fencing:Mutex<()>,
    pub routes:Mutex<HashMap<u64,mpsc::Sender<Result<Frame>>>>,
    pending:Mutex<HashMap<u64,Pending>>,
    cancelled:Mutex<HashMap<u64,Instant>>,
    child:Mutex<Option<Child>>,
    pub healthy:AtomicBool,
    closed:AtomicBool,
    stop_requested:Arc<AtomicBool>,
    pub epoch:AtomicU64,
    pub next:AtomicU64,
    pub admission:Admission,
    recovery:Mutex<Recovery>,
    quarantine:Mutex<Quarantine>,
    changed:Condvar,
    hooks:Mutex<Vec<(u64,Hook)>>,
    snapshot:Mutex<Option<(Instant,Value)>>,
    unanswered:Mutex<Option<Instant>>,
    pending_limit:usize,
}
impl Inner {
    fn new(limit:usize,stop_requested:Arc<AtomicBool>)->Arc<Self> {Arc::new(Self {
        trace:super::crash_trace::Trace::new(),writer:Mutex::new(None),fencing:Mutex::new(()),routes:Mutex::new(HashMap::new()),pending:Mutex::new(HashMap::new()),cancelled:Mutex::new(HashMap::new()),child:Mutex::new(None),
        healthy:AtomicBool::new(false),closed:AtomicBool::new(false),stop_requested,epoch:AtomicU64::new(1),next:AtomicU64::new(1),admission:Admission::new(limit),recovery:Mutex::new(Recovery::default()),quarantine:Mutex::new(Quarantine::default()),changed:Condvar::new(),hooks:Mutex::new(Vec::new()),snapshot:Mutex::new(None),unanswered:Mutex::new(None),pending_limit:limit,
    })}
    #[cfg(test)]
    pub fn for_test()->Arc<Self>{let core=Self::new(32,Arc::new(AtomicBool::new(false)));core.healthy.store(true,Ordering::Release);core}
    pub fn epoch(&self)->u64{self.epoch.load(Ordering::Acquire)}
    pub fn pid(&self)->u32{self.child.lock().unwrap().as_ref().map_or(0,Child::id)}
    pub fn terminal_error(&self)->ApiError {
        if self.stopping(){return ApiError(503,"server is shutting down".into(),"server_shutdown".into())}
        if let Some(message)=&self.recovery.lock().unwrap().fatal{return ApiError(500,message.clone(),"engine_failed".into())}
        ApiError(503,"the inference engine stopped unexpectedly and is restarting; retry the request".into(),"runtime_unavailable".into())
    }
    pub fn refusal(&self)->Option<ApiError> {
        if self.stopping(){return Some(self.terminal_error())}
        if self.healthy.load(Ordering::Acquire){return None}
        let error=self.terminal_error();if error.2!="runtime_unavailable"{return Some(error)}
        let recovery=self.recovery.lock().unwrap();
        Some(ApiError(503,format!("engine is recovering; retry shortly{}",recovery.last_error.as_ref().map(|s|format!(" (last failure: {s})")).unwrap_or_default()),"engine_recovering".into()))
    }
    pub fn send(&self,epoch:u64,frame:&Frame)->Result<()> {
        let result={
            let mut writer=self.writer.lock().map_err(|_|failed("native writer poisoned"))?;
            if self.stopping()||epoch!=self.epoch()||!self.healthy.load(Ordering::Acquire){return Err(self.terminal_error())}
            write_frame(writer.as_mut().ok_or_else(||self.terminal_error())?,frame).map_err(|e|failed(e.to_string())).map(|()|self.trace.record(epoch,"client_to_engine",frame))
        };
        if result.is_err(){self.fail(epoch,"native write failed",false);return Err(self.terminal_error())}result
    }
    pub fn register(&self,epoch:u64,frame:&Frame,sender:mpsc::Sender<Result<Frame>>,permit:OwnedSemaphorePermit,terminal:Arc<Terminal>)->Result<()> {
        let id=u64_at(&frame.payload,0)?;
        let fingerprint=quarantine::fingerprint(frame);
        let result={
            let mut writer=self.writer.lock().unwrap();
            if self.stopping()||epoch!=self.epoch()||!self.healthy.load(Ordering::Acquire){return Err(self.terminal_error())}
            if fingerprint.is_some_and(|key|self.quarantine.lock().unwrap().refuses(&key)) {
                return Err(ApiError(400,"this request repeatedly crashed the inference engine".into(),"request_quarantined".into()));
            }
            let mut routes=self.routes.lock().unwrap();
            if routes.contains_key(&id){return Err(ApiError(503,"native request capacity exhausted".into(),"frontend_overloaded".into()))}
            routes.insert(id,sender);self.pending.lock().unwrap().insert(id,Pending{_permit:permit,kind:frame.kind,terminal,fingerprint,started:false});drop(routes);
            write_frame(writer.as_mut().ok_or_else(||self.terminal_error())?,frame).map_err(|e|failed(e.to_string())).map(|()|self.trace.record(epoch,"client_to_engine",frame))
        };
        if result.is_err(){self.fail(epoch,"native write failed",false);return Err(self.terminal_error())}result
    }
    fn finish(&self,id:u64) {
        if let Some(p)=self.pending.lock().unwrap().remove(&id){p.terminal.state.store(1,Ordering::Release)}
        self.cancelled.lock().unwrap().remove(&id);
    }
    pub fn abandon(&self,epoch:u64,id:u64,control:bool) {
        self.routes.lock().unwrap().remove(&id);
        if control {self.pending.lock().unwrap().remove(&id);}else{self.cancel(epoch,id)}
    }
    pub fn cancel(&self,epoch:u64,id:u64) {
        if self.stopping()||epoch!=self.epoch()||!self.healthy.load(Ordering::Acquire){return}
        if !self.pending.lock().unwrap().get(&id).is_some_and(|p|p.kind==1){return}
        self.cancelled.lock().unwrap().entry(id).or_insert_with(Instant::now);
        if let Ok(frame)=cancel(id){let _=self.send(epoch,&frame);}
    }
    pub fn on_failure(&self,epoch:u64,hook:Hook) {
        // The writer mutex is the admission/failure gate, including hook registration.
        let guard=self.writer.lock().unwrap();
        if !self.stopping()&&epoch==self.epoch()&&self.healthy.load(Ordering::Acquire){self.hooks.lock().unwrap().push((epoch,hook));drop(guard)}else{drop(guard);hook()}
    }
    pub fn stopping(&self)->bool{self.closed.load(Ordering::Acquire)||self.stop_requested.load(Ordering::Acquire)}
    pub fn kill_owned(&self){if let Some(child)=self.child.lock().unwrap().as_mut(){let _=child.kill();}}
    fn stop_child(&self,grace:bool) {
        let deadline=Instant::now()+if grace{Duration::from_secs(15)}else{Duration::ZERO};let mut signalled=false;
        loop {
            let mut slot=self.child.lock().unwrap();let Some(child)=slot.as_mut()else{return};
            if child.try_wait().ok().flatten().is_some(){slot.take();return}
            if !signalled&&grace{let _=unsafe{libc::kill(child.id() as i32,libc::SIGTERM)};signalled=true;}
            if Instant::now()>=deadline{let _=child.kill();let _=child.wait();slot.take();return}
            // Leave the exact Child reachable so a second Ctrl+C can cut a
            // graceful wait short without ever signalling a saved/stale PID.
            drop(slot);std::thread::sleep(Duration::from_millis(20));
        }
    }
    fn capture_trace(&self,epoch:u64,message:&str){
        if self.stopping(){return}
        use std::os::unix::process::ExitStatusExt;
        let code=self.child.lock().unwrap().as_mut().and_then(|child|child.try_wait().ok().flatten()).and_then(|status|status.code().or_else(||status.signal().map(|signal|-signal)));
        let status=self.snapshot.lock().unwrap().as_ref().map(|(_,v)|v.clone());
        let _=self.trace.dump(epoch,message,code,status.as_ref());
    }
    pub fn fail(&self,epoch:u64,message:&str,immutable:bool) {
        let _fencing=self.fencing.lock().unwrap();
        if epoch!=self.epoch()||!self.healthy.swap(false,Ordering::AcqRel){return}
        // Attribute only a real fault-signal exit observed before our cleanup
        // kills the child. SIGKILL/SIGTERM, timeouts and memory refusals do not
        // count. On EOF allow a dying child a bounded moment to become waitable.
        if !self.stopping()&&!immutable {
            use std::os::unix::process::ExitStatusExt;
            let until=Instant::now()+Duration::from_millis(20);
            let signal=loop {
                let status=self.child.lock().unwrap().as_mut().and_then(|child|child.try_wait().ok().flatten());
                if let Some(status)=status {break status.signal()}
                if message!="native protocol reached EOF"||Instant::now()>=until {break None}
                std::thread::sleep(Duration::from_millis(1));
            };
            if quarantine::fault_signal(signal) {
                let started=self.pending.lock().unwrap().values().filter(|p|p.started).filter_map(|p|p.fingerprint).collect::<Vec<_>>();
                self.quarantine.lock().unwrap().after_fault(&started);
            }
        }
        self.capture_trace(epoch,message);
        self.epoch.fetch_add(1,Ordering::AcqRel);
        if let Some((_,status))=self.snapshot.lock().unwrap().as_mut(){status["ready"]=json!(false)}
        self.record_failure(message,immutable);
        // Kill the exact child before waiting for the writer mutex: a hung
        // child may have stopped draining stdin, leaving a large write blocked.
        for (_,hook) in self.hooks.lock().unwrap().drain(..){hook()}
        let graceful=self.closed.load(Ordering::Acquire);
        if graceful {if let Ok(mut writer)=self.writer.try_lock(){writer.take();}}
        self.stop_child(graceful);self.writer.lock().unwrap().take();
        let error=self.terminal_error();
        for (_,pending) in self.pending.lock().unwrap().drain(){*pending.terminal.error.lock().unwrap()=Some(error.clone());pending.terminal.state.store(2,Ordering::Release)}
        self.cancelled.lock().unwrap().clear();
        self.unanswered.lock().unwrap().take();
        for (_,sender) in self.routes.lock().unwrap().drain(){let _=sender.try_send(Err(error.clone()));}
        self.changed.notify_all();
    }
    fn record_failure(&self,message:&str,immutable:bool) {
        if self.stopping(){return}
        let mut state=self.recovery.lock().unwrap();
        if state.ready_at.take().is_some_and(|t|t.elapsed()>=Duration::from_secs(60)){state.failures=0}
        state.failures+=1;state.last_error=Some(message.into());
        if immutable {state.fatal=Some(message.into());state.due=None;}
        else if state.failures>=3 {
            state.fatal=Some(format!("the inference engine failed {} times in a row, each within 60 s of starting ({message}); JustFit stopped restarting it. {}Restart the JustFit server after fixing the cause.",state.failures,self.trace.last().map(|path|format!("Crash trace: {}. ",path.display())).unwrap_or_else(||"Set JUSTFIT_CRASH_TRACE=1 to record a crash trace. ".into())));state.due=None;
        } else {state.due=Some(Instant::now()+if state.failures==1{Duration::ZERO}else{Duration::from_secs(5)});}
    }
    pub fn shutdown(&self) {
        if self.closed.swap(true,Ordering::AcqRel){return}
        let epoch=self.epoch();self.fail(epoch,"service shutdown",false);
        self.writer.lock().unwrap().take();self.stop_child(false);self.changed.notify_all();
    }
    fn start(&self,command:&[String],expected:Option<Ready>)->Result<(ChildStdout,Ready)> {
        let fencing=self.fencing.lock().unwrap();
        if self.stopping(){return Err(self.terminal_error())}
        let program=command.first().ok_or_else(||failed("missing native command"))?;
        let mut child=crate::sandbox::command(program,crate::sandbox::SERVICE_PROFILE).args(&command[1..]).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit()).spawn().map_err(|e|failed(e.to_string()))?;
        let writer=child.stdin.take().map(Pipe::new).transpose();
        let writer=match writer {Ok(writer)=>writer,Err(e)=>{let _=child.kill();let _=child.wait();return Err(failed(e.to_string()))}};let mut output=child.stdout.take().ok_or_else(||failed("missing native stdout"))?;
        let mut target=self.child.lock().unwrap();
        if self.stopping(){let _=child.kill();let _=child.wait();return Err(self.terminal_error())}
        self.trace.start(self.epoch(),child.id());*target=Some(child);drop(target);drop(fencing);
        let (sender,receiver)=sync_mpsc::channel();
        std::thread::spawn(move||{let frame=read_frame(&mut output);let _=sender.send((output,frame));});
        let deadline=Instant::now()+STARTUP;
        let (output,frame)=loop {
            if self.stopping()||Instant::now()>=deadline {self.stop_child(false);return Err(failed("native ReadyEvent timed out"))}
            match receiver.recv_timeout(Duration::from_millis(10)){Ok(pair)=>break pair,Err(sync_mpsc::RecvTimeoutError::Timeout)=>(),Err(_)=>{self.stop_child(false);return Err(failed("native did not announce Ready"))}}
        };
        if let Ok(Some(frame))=&frame{self.trace.record(self.epoch(),"engine_to_client",frame);}
        let parsed=(||{
            let frame=frame.map_err(|_|failed("native did not announce Ready"))?.filter(|f|f.kind==READY).ok_or_else(||failed("native did not announce Ready"))?;
            let lanes=u32_at(&frame.payload,0)?;let context=u32_at(&frame.payload,4)?;
            if lanes==0||context==0||context>262144||frame.payload[8]>1{return Err(failed("invalid Ready"))}
            let ready=Ready{lanes,context,vision:frame.payload[8]!=0};
            if expected.is_some_and(|v|v!=ready){return Err(failed("native context window, concurrency or vision changed; restart the JustFit server"))}
            Ok(ready)
        })();
        let ready=match parsed{Ok(r)=>r,Err(e)=>{self.stop_child(false);return Err(e)}};
        let mut gate=self.writer.lock().unwrap();
        if self.stopping(){drop(gate);self.stop_child(false);return Err(self.terminal_error())}
        *gate=writer;self.healthy.store(true,Ordering::Release);
        let mut state=self.recovery.lock().unwrap();state.ready_at=Some(Instant::now());state.due=None;drop(state);drop(gate);
        Ok((output,ready))
    }
    fn read(&self,output:&mut ChildStdout,epoch:u64) {
        loop {
            let frame=match read_frame(output){
                Ok(Some(frame))=>frame,
                Ok(None)=>{self.fail(epoch,"native protocol reached EOF",false);break},
                Err(error)=>{self.fail(epoch,&format!("native protocol failed: {error}"),false);break},
            };
            if epoch!=self.epoch(){break}
            self.trace.record(epoch,"engine_to_client",&frame);
            let id=if frame.kind==ERROR{u64_at(&frame.payload,2)}else{u64_at(&frame.payload,0)};
            let Ok(id)=id else{self.fail(epoch,"native event has no request id",false);break};
            if frame.kind==START {
                if let Some(p)=self.pending.lock().unwrap().get_mut(&id) {p.started=p.kind==1;}
            }
            if frame.kind==ERROR&&(id==0||super::request_error(&frame.payload).0==502){self.fail(epoch,"native reported a fatal or malformed error",false);break}
            if [DONE,STATUS,ERROR].contains(&frame.kind){self.finish(id)}
            let sender=self.routes.lock().unwrap().get(&id).cloned();
            if let Some(sender)=sender {
                if sender.try_send(Ok(frame)).is_err(){self.routes.lock().unwrap().remove(&id);self.cancel(epoch,id);}
            }
        }
    }
    pub fn cached_status(&self)->Value {
        self.snapshot.lock().unwrap().as_ref().map(|(_,v)|v.clone()).unwrap_or_else(||json!({"schema_version":6,"ready":false}))
    }
    pub fn status_timeout(&self){self.unanswered.lock().unwrap().get_or_insert_with(Instant::now);self.changed.notify_all();}
    pub fn cached_ready(&self)->bool{self.healthy.load(Ordering::Acquire)&&self.unanswered.lock().unwrap().is_none_or(|t|t.elapsed()<STATUS_WAIT)}
    pub fn remember_status(&self,epoch:u64,value:&Value){
        // Fence the snapshot too: a status callback can resume after its native
        // generation has failed and must not mark the replacement ready.
        let _gate=self.writer.lock().unwrap();
        if epoch==self.epoch()&&self.healthy.load(Ordering::Acquire){
            *self.snapshot.lock().unwrap()=Some((Instant::now(),value.clone()));
            self.unanswered.lock().unwrap().take();
            self.recovery.lock().unwrap().last_error=None;
        }
    }
    pub fn transport(&self,stale:bool)->Value {
        let recovery=self.recovery.lock().unwrap();let age=self.snapshot.lock().unwrap().as_ref().map_or(0.,|(when,_)|when.elapsed().as_secs_f64()*1000.);
        let mut value=json!({"ready":self.healthy.load(Ordering::Acquire),"recovering":!self.closed.load(Ordering::Acquire)&&recovery.fatal.is_none()&&!self.healthy.load(Ordering::Acquire),"stopped":recovery.fatal.is_some(),"pending":self.pending.lock().unwrap().len(),"pending_limit":self.pending_limit,"restarts":recovery.restarts,"last_crash_trace":self.trace.last(),"status_stale":stale,"status_age_ms":if stale{age}else{0.}});
        if stale {if let Some(error)=&recovery.last_error{value["error"]=json!(error)}}value
    }
    fn maintenance(core:Arc<Self>) {
        let mut probe:Option<(u64,u64,Instant,mpsc::Receiver<Result<Frame>>)>=None;let mut next_probe=Instant::now()+Duration::from_secs(10);
        while !core.stopping() {
            let epoch=core.epoch();
            if core.healthy.load(Ordering::Acquire) {
                if core.cancelled.lock().unwrap().values().any(|t|t.elapsed()>=CANCEL_WAIT){core.fail(epoch,"native cancellation was not acknowledged",false);}
                if let Some((id,gen,until,receiver))=&mut probe {
                    match receiver.try_recv(){
                        Ok(Ok(frame)) if frame.kind==STATUS=>{
                            let status:Option<Value>=serde_json::from_slice(&frame.payload[8..]).ok();
                            if let Some(value)=status.filter(|v|v.is_object()&&v["schema_version"]==6){core.remember_status(*gen,&value)}else{core.fail(*gen,"native status does not match the current schema",false)}
                            core.routes.lock().unwrap().remove(id);probe=None;next_probe=Instant::now()+Duration::from_secs(10);
                        },
                        Ok(_)|Err(mpsc::error::TryRecvError::Disconnected)=>{if *gen==epoch{core.fail(epoch,"native liveness status failed",false)}probe=None;},
                        Err(mpsc::error::TryRecvError::Empty)=>{if *gen!=epoch{probe=None}else if Instant::now()>=*until{core.fail(epoch,"native status request was not answered within 30 s",false);probe=None;}}
                    }
                }else if core.unanswered.lock().unwrap().is_some()||(Instant::now()>=next_probe&&core.pending.lock().unwrap().values().any(|p|p.kind==1)){
                    if let Ok(permit)=core.admission.acquire(4){
                        let id=core.next.fetch_add(1,Ordering::Relaxed);let(sender,receiver)=mpsc::channel(1);
                        let frame=Frame{kind:4,payload:id.to_le_bytes().to_vec()};
                        if core.register(epoch,&frame,sender,permit,Arc::new(Terminal::default())).is_ok(){probe=Some((id,epoch,Instant::now()+STATUS_WAIT,receiver));}
                    }
                }
            }else{probe=None;next_probe=Instant::now()+Duration::from_secs(10);}
            let state=core.recovery.lock().unwrap();drop(core.changed.wait_timeout(state,Duration::from_millis(20)).unwrap());
        }
    }
}
pub(super) fn launch(command:&[String],limit:usize)->Result<(Arc<Inner>,Ready)> {
    launch_cancellable(command,limit,Arc::new(AtomicBool::new(false)))
}
pub(super) fn launch_cancellable(command:&[String],limit:usize,stop_requested:Arc<AtomicBool>)->Result<(Arc<Inner>,Ready)> {
    let core=Inner::new(limit,stop_requested);core.trace.command(command);
    let (mut output,ready)=match core.start(command,None){Ok(pair)=>pair,Err(e)=>{core.capture_trace(core.epoch(),&e.1);core.shutdown();return Err(e)}};
    let maintenance=core.clone();std::thread::spawn(move||Inner::maintenance(maintenance));
    let reader=core.clone();let command=command.to_vec();
    std::thread::spawn(move||{
        loop {
            let epoch=reader.epoch();reader.read(&mut output,epoch);
            loop {
                if reader.stopping(){return}
                let mut state=reader.recovery.lock().unwrap();
                let Some(due)=state.due else{return};
                if Instant::now()<due {state=reader.changed.wait_timeout(state,due.saturating_duration_since(Instant::now()).min(Duration::from_millis(20))).unwrap().0;drop(state);continue}
                state.restarts+=1;drop(state);
                match reader.start(&command,Some(ready)) {
                    Ok((next,_))=>{output=next;break},
                    Err(e)=>{
                        reader.capture_trace(reader.epoch(),&e.1);
                        reader.epoch.fetch_add(1,Ordering::AcqRel);
                        reader.record_failure(&e.1,e.1.contains("context window, concurrency or vision changed"));
                    }
                }
            }
        }
    });
    Ok((core,ready))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn old_ticket_retains_its_generation_failure() {
        let core=Inner::for_test();
        let terminal=Arc::new(Terminal::default());
        core.pending.lock().unwrap().insert(1,Pending{_permit:core.admission.acquire(1).unwrap(),kind:1,terminal:terminal.clone(),fingerprint:None,started:false});
        core.fail(1,"first",false);assert_eq!(terminal.error.lock().unwrap().as_ref().unwrap().0,503);
        core.record_failure("second",false);core.record_failure("third",false);
        assert_eq!(core.terminal_error().0,500);assert_eq!(terminal.error.lock().unwrap().as_ref().unwrap().0,503);
    }
    #[test]
    fn recovery_policy_matches_released_three_failure_window() {
        let core=Inner::new(1,Arc::new(AtomicBool::new(false)));
        core.record_failure("first",false);
        {let r=core.recovery.lock().unwrap();assert_eq!(r.failures,1);assert!(r.due.unwrap()<=Instant::now());}
        core.record_failure("second",false);
        {let r=core.recovery.lock().unwrap();assert_eq!(r.failures,2);let left=r.due.unwrap().saturating_duration_since(Instant::now());assert!(left>Duration::from_millis(4900)&&left<=Duration::from_secs(5));}
        core.record_failure("third",false);
        let r=core.recovery.lock().unwrap();assert!(r.due.is_none()&&r.fatal.is_some());
    }
    #[test]
    fn full_service_window_resets_consecutive_failures() {
        let core=Inner::new(1,Arc::new(AtomicBool::new(false)));core.record_failure("first",false);
        core.recovery.lock().unwrap().ready_at=Some(Instant::now()-Duration::from_secs(60));
        core.record_failure("after service",false);
        let r=core.recovery.lock().unwrap();assert_eq!(r.failures,1);assert!(r.fatal.is_none());
    }
    #[test]
    fn immutable_ready_change_stops_and_old_failure_cannot_affect_new_generation() {
        let core=Inner::for_test();core.fail(0,"stale callback",false);assert!(core.healthy.load(Ordering::Acquire));
        core.remember_status(0,&json!({"ready":true}));assert_eq!(core.cached_status()["ready"],false);
        core.fail(1,"changed Ready",true);assert_eq!(core.terminal_error().0,500);assert_eq!(core.epoch(),2);
    }
    #[test]
    fn busy_status_retains_readiness_only_inside_answer_window() {
        let core=Inner::for_test();assert!(core.cached_ready());core.status_timeout();assert!(core.cached_ready());
        *core.unanswered.lock().unwrap()=Some(Instant::now()-STATUS_WAIT);assert!(!core.cached_ready());
        core.remember_status(1,&json!({"schema_version":6,"ready":true}));assert!(core.cached_ready());
    }
}
