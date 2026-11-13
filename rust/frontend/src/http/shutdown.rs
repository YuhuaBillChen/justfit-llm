//! The service owns native cleanup; signal receipt must not wait for HTTP drain.
use crate::{engine::Engine,ApiError};
use std::{io,sync::{Arc,Mutex,OnceLock,Weak,atomic::{AtomicBool,Ordering}}};
use tokio::{signal::unix::{signal,SignalKind},sync::watch,task::JoinHandle};
type Cleanup=Arc<Mutex<Option<JoinHandle<()>>>>;
fn begin(engine:&OnceLock<Weak<Engine>>,cleanup:&Cleanup){
    let mut job=cleanup.lock().unwrap();
    if job.is_none(){if let Some(engine)=engine.get().and_then(Weak::upgrade){*job=Some(tokio::task::spawn_blocking(move||engine.shutdown()));}}
}
pub(super) fn error()->ApiError{ApiError(503,"server is shutting down".into(),"server_shutdown".into())}
pub(super) async fn cancelled(receiver:&mut watch::Receiver<bool>){let _=receiver.wait_for(|closed|*closed).await;}
pub(super) struct Signals {
    pub requested:Arc<AtomicBool>,
    pub receiver:watch::Receiver<bool>,
    engine:Arc<OnceLock<Weak<Engine>>>,
    cleanup:Cleanup,
    task:JoinHandle<()>,
}
impl Signals {
    pub fn install()->io::Result<Self>{
        // Register before unblocking inherited launcher masks, as 2.3 does.
        let mut interrupt=signal(SignalKind::interrupt())?;let mut terminate=signal(SignalKind::terminate())?;
        let mut mask=unsafe{std::mem::zeroed::<libc::sigset_t>()};
        let result=unsafe{libc::sigemptyset(&mut mask);libc::sigaddset(&mut mask,libc::SIGINT);libc::sigaddset(&mut mask,libc::SIGTERM);libc::pthread_sigmask(libc::SIG_UNBLOCK,&mask,std::ptr::null_mut())};
        if result!=0{return Err(io::Error::from_raw_os_error(result))}
        let requested=Arc::new(AtomicBool::new(false));let engine=Arc::new(OnceLock::<Weak<Engine>>::new());let cleanup:Cleanup=Arc::new(Mutex::new(None));let(sender,receiver)=watch::channel(false);
        let(stop,target,jobs)=(requested.clone(),engine.clone(),cleanup.clone());
        let task=tokio::spawn(async move{
            loop {
                let is_interrupt=tokio::select!{value=interrupt.recv()=>{if value.is_none(){break}true},value=terminate.recv()=>{if value.is_none(){break}false}};
                if !stop.swap(true,Ordering::AcqRel){begin(&target,&jobs);let _=sender.send(true);}
                else if is_interrupt {if let Some(engine)=target.get().and_then(Weak::upgrade){engine.kill_owned();}}
            }
        });
        Ok(Self{requested,receiver,engine,cleanup,task})
    }
    pub fn register(&self,engine:&Arc<Engine>){let _=self.engine.set(Arc::downgrade(engine));if self.stopping(){begin(&self.engine,&self.cleanup);}}
    pub fn stopping(&self)->bool{self.requested.load(Ordering::Acquire)}
    pub async fn finish(&self){
        begin(&self.engine,&self.cleanup);
        let job=self.cleanup.lock().unwrap().take();if let Some(job)=job{let _=job.await;}
    }
}
impl Drop for Signals{fn drop(&mut self){self.task.abort();}}
