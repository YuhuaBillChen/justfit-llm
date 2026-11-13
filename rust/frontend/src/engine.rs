//! Multiplexed, bounded native protocol transport. No Python request execution.
use crate::{ApiError, Result};
use justfit_native_client_prototype::{
    Frame, DONE, ERROR, STATUS,
};
use serde_json::Value;
use std::{
    sync::{
        atomic::Ordering,
        Arc,
    },
    time::Duration,
};
use tokio::sync::{mpsc, OwnedSemaphorePermit, Semaphore};

fn failed(s: impl Into<String>) -> ApiError {
    ApiError(502, s.into(), "engine_unavailable".into())
}
fn request_error(p: &[u8]) -> ApiError {
    let parsed = || -> Result<ApiError> {
        if p.len() < 18 || p[0] != 1 || p[1] > 1 || u64_at(p, 2)? == 0 {
            return Err(failed("invalid native error classification"));
        }
        let codes = u32_at(p, 10)? as usize;
        let messages = u32_at(p, 14)? as usize;
        if codes == 0
            || codes > 1024 * 1024
            || messages > 1024 * 1024
            || p.len() != 18 + codes + messages
        {
            return Err(failed("invalid native error lengths"));
        }
        let code = String::from_utf8_lossy(&p[18..18 + codes]).into_owned();
        let message = String::from_utf8_lossy(&p[18 + codes..]).into_owned();
        if code == "deadline_exceeded" {
            return Ok(ApiError(504, message, "request_timeout".into()));
        }
        if code == "capacity_exhausted" {
            return Ok(ApiError(400,format!("the request does not fit in the memory this server may use, even after every cached prefix was evicted; restart the server with a larger --max-memory or a smaller --max-context ({message})"),code));
        }
        let status = if p[1] != 0 {
            503
        } else if [
            "invalid_constraint",
            "invalid_count",
            "invalid_deadline",
            "invalid_enum_value",
            "invalid_request",
            "invalid_sampling",
            "limit_exceeded",
        ]
        .contains(&code.as_str())
        {
            400
        } else {
            500
        };
        Ok(ApiError(status, message, code))
    };
    parsed().unwrap_or_else(|e| e)
}
pub fn u32_at(p: &[u8], offset: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        p.get(offset..offset + 4)
            .ok_or_else(|| failed("short native uint32"))?
            .try_into()
            .unwrap(),
    ))
}
pub fn u64_at(p: &[u8], offset: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(
        p.get(offset..offset + 8)
            .ok_or_else(|| failed("short native uint64"))?
            .try_into()
            .unwrap(),
    ))
}

mod lifecycle;
mod crash_trace;
mod quarantine;
use lifecycle::{Inner,Terminal};
// Keep native control/lease traffic available when the generation queue is full.
struct Admission {
    requests: Arc<Semaphore>,
    controls: Arc<Semaphore>,
}
impl Admission {
    fn new(requests: usize) -> Self {
        Self { requests: Arc::new(Semaphore::new(requests)), controls: Arc::new(Semaphore::new(64)) }
    }
    fn acquire(&self, kind: u16) -> Result<OwnedSemaphorePermit> {
        let control = matches!(kind, 4 | 5);
        let slots = if control { &self.controls } else { &self.requests };
        slots.clone().try_acquire_owned().map_err(|_| ApiError(503,
            if control { "native control request capacity exhausted" } else { "request queue is full" }.into(),
            "frontend_overloaded".into()))
    }
}
pub struct Engine {
    inner: Arc<Inner>,

    pub context: u32,
    pub lanes: u32,
    pub vision: bool,
}
pub struct Ticket {
    inner: Arc<Inner>,
    pub id: u64,
    receiver: mpsc::Receiver<Result<Frame>>,
    done: bool,
    first: Option<Frame>,
    grammar: Option<crate::grammar::Shared>,
    grammar_offset: usize,
    pending_mask: Option<PendingMask>,
    epoch:u64,
    terminal:Arc<Terminal>,
    control:bool,
}
struct PendingMask(tokio::task::JoinHandle<Result<Frame>>);
impl PendingMask {
    async fn reply(&mut self) -> Result<Frame> {
        (&mut self.0).await.map_err(|_|failed("grammar task failed"))?
    }
}
impl Engine {
    pub fn spawn(command: &[String], pending_limit: usize) -> Result<Self> {
        let (inner,ready)=lifecycle::launch(command,pending_limit)?;
        Ok(Self {inner,context:ready.context,lanes:ready.lanes,vision:ready.vision})
    }
    pub fn spawn_cancellable(command:&[String],pending_limit:usize,stop:Arc<std::sync::atomic::AtomicBool>)->Result<Self>{
        let(inner,ready)=lifecycle::launch_cancellable(command,pending_limit,stop)?;
        Ok(Self{inner,context:ready.context,lanes:ready.lanes,vision:ready.vision})
    }
    pub fn kill_owned(&self){self.inner.kill_owned();}
    pub fn healthy(&self)->bool {!self.inner.stopping()&&self.inner.healthy.load(Ordering::Acquire)}
    pub fn owned_pid(&self)->u32 {self.inner.pid()}
    pub fn generation(&self)->u64 {self.inner.epoch()}
    pub fn refusal(&self)->Option<ApiError>{self.inner.refusal()}
    pub fn fail_generation(&self,generation:u64,message:&str){self.inner.fail(generation,message,false)}
    pub fn on_generation_failure(&self,generation:u64,hook:impl Fn()+Send+Sync+'static){self.inner.on_failure(generation,Box::new(hook));}
    pub fn next_id(&self) -> u64 {
        self.inner.next.fetch_add(1, Ordering::Relaxed)
    }
    pub fn submit(&self, frame: Frame) -> Result<Ticket> {
        self.submit_constrained(frame, None)
    }
    pub fn submit_constrained(&self, frame: Frame, grammar: Option<crate::grammar::Shared>) -> Result<Ticket> {
        self.submit_generation(frame,grammar,self.generation())
    }
    pub fn submit_prepared(&self,prepared:&crate::generation::Prepared)->Result<Ticket> {
        let epoch=prepared.vision_owner.as_ref().map_or_else(||self.generation(),|owner|owner.generation());
        self.submit_generation(Frame{kind:prepared.frame.kind,payload:prepared.frame.payload.clone()},prepared.constraint.clone(),epoch)
    }
    fn submit_generation(&self,frame:Frame,grammar:Option<crate::grammar::Shared>,epoch:u64)->Result<Ticket> {
        if let Some(error)=self.refusal(){return Err(error)}
        let id=u64_at(&frame.payload,0)?;
        let admission=self.inner.admission.acquire(frame.kind)?;
        let (sender,receiver)=mpsc::channel(128);
        let terminal=Arc::new(Terminal::default());
        self.inner.register(epoch,&frame,sender,admission,terminal.clone())?;
        Ok(Ticket{inner:self.inner.clone(),id,receiver,done:false,first:None,grammar,grammar_offset:0,pending_mask:None,epoch,terminal,control:matches!(frame.kind,4|5)})
    }
    pub async fn status(&self) -> Result<Value> {
        let reply=if self.healthy(){Some(self.control(0,0,Duration::from_millis(50)).await)}else{None};
        let busy=matches!(&reply,Some(Err(error)) if error.1=="status timeout");
        let live=reply.and_then(|reply|reply.ok());
        let stale=live.is_none();let mut status=live.unwrap_or_else(||self.inner.cached_status());
        if busy&&self.healthy(){self.inner.status_timeout();}
        if (stale&&!busy)||!self.inner.cached_ready(){status["ready"]=serde_json::json!(false)}
        status["transport"]=self.inner.transport(stale);Ok(status)
    }
    /// Status and governed external-memory leases share native status replies.
    /// Lease IDs are driver-owned; correlation IDs always come from this host.
    pub async fn control(&self,lease:u64,bytes:u64,timeout:Duration)->Result<Value> {
        self.control_generation(self.generation(),lease,bytes,timeout).await
    }
    pub async fn control_generation(&self,epoch:u64, lease: u64, bytes: u64, timeout: Duration) -> Result<Value> {
        let id = self.next_id();
        let mut payload = id.to_le_bytes().to_vec();
        if lease != 0 {
            if bytes > 2 * 1024 * 1024 * 1024 {
                return Err(failed("external memory budget exceeds 2GiB"));
            }
            payload.extend(lease.to_le_bytes());
            payload.extend(bytes.to_le_bytes());
        }
        let mut ticket = self.submit_generation(Frame {
            kind: if lease == 0 { 4 } else { 5 },
            payload,
        },None,epoch)?;
        let frame = tokio::time::timeout(timeout, ticket.next())
            .await
            .map_err(|_| failed("status timeout"))??;
        if frame.kind != STATUS {
            return Err(failed("expected native status"));
        }
        let value:Value=serde_json::from_slice(&frame.payload[8..]).map_err(|_|failed("invalid native status JSON"))?;
        if !value.is_object()||value["schema_version"]!=6{return Err(failed("native status does not match the current schema"))}
        if epoch!=self.generation()||!self.healthy(){return Err(self.inner.terminal_error())}
        self.inner.remember_status(epoch,&value);Ok(value)
    }
    pub fn shutdown(&self) {self.inner.shutdown();}

}
impl Ticket {
    fn generation_error(&self)->Option<ApiError>{
        if self.terminal.state.load(Ordering::Acquire)!=2{return None}
        self.terminal.error.lock().unwrap().clone().or_else(||Some(self.inner.terminal_error()))
    }
    pub fn unread_start(&mut self, frame: Frame) -> Result<()> {
        if self.first.is_some()
            || frame.kind != justfit_native_client_prototype::START
            || u64_at(&frame.payload, 0)? != self.id
        {
            return Err(failed("invalid deferred Start event"));
        }
        self.first = Some(frame);
        Ok(())
    }
    pub async fn next(&mut self) -> Result<Frame> {
        if let Some(error)=self.generation_error(){return Err(error)}
        if let Some(frame) = self.first.take() {
            return Ok(frame);
        }
        loop {
        if let Some(job)=self.pending_mask.as_mut() {
            // Await through a mutable reference: idle/keepalive timeouts must
            // not discard a mask request whose blocking work is still running.
            let frame=job.reply().await?;
            self.pending_mask.take();
            if let Some(error)=self.generation_error(){return Err(error)}
            if frame.kind==3 { self.inner.send(self.epoch,&frame)?;continue; }
            return Ok(frame);
        }
        let frame = self
            .receiver
            .recv()
            .await
            .ok_or_else(|| failed("native event channel closed"))??;
        if let Some(error)=self.generation_error(){return Err(error)}
        if [DONE, STATUS, ERROR].contains(&frame.kind) {
            self.done = true;
            self.terminal.state.store(1,Ordering::Release);
            self.inner.routes.lock().unwrap().remove(&self.id);
        }
        if frame.kind == ERROR {
            return Err(request_error(&frame.payload));
        }
        if frame.kind==justfit_native_client_prototype::MASK {
            let grammar=self.grammar.clone().ok_or_else(||failed("native requested an unconfigured grammar"))?;
            let id=self.id;let mask_id=u64_at(&frame.payload,8)?;let count=u32_at(&frame.payload,20)? as usize;
            if u64_at(&frame.payload,0)?!=id || u32_at(&frame.payload,16)?!=7760 || count>=9 || frame.payload.len()!=24+count*4 { return Err(failed("invalid native mask request")); }
            let simulation=(0..count).map(|i|u32_at(&frame.payload,24+i*4)).collect::<Result<Vec<_>>>()?;
            self.pending_mask=Some(PendingMask(tokio::task::spawn_blocking(move || {
                let payload=grammar.lock().map_err(|_|failed("grammar unavailable"))?.masks(id,mask_id,&simulation)?;
                Ok(Frame{kind:3,payload})
            })));
            continue;
        }
        if frame.kind==justfit_native_client_prototype::MASK_TREE {
            // A draft tree's rows and their parents (masks_tree): the masks before the anchor and after each row's path.
            let grammar=self.grammar.clone().ok_or_else(||failed("native requested an unconfigured grammar"))?;
            let id=self.id;let mask_id=u64_at(&frame.payload,8)?;let rows=u32_at(&frame.payload,20)? as usize;
            if u64_at(&frame.payload,0)?!=id || u32_at(&frame.payload,16)?!=7760 || rows==0 || rows>16 || frame.payload.len()!=24+rows*8 { return Err(failed("invalid native tree mask request")); }
            let tokens=(0..rows).map(|i|u32_at(&frame.payload,24+i*4)).collect::<Result<Vec<_>>>()?;
            let parents=(0..rows).map(|i|u32_at(&frame.payload,24+(rows+i)*4)).collect::<Result<Vec<_>>>()?;
            self.pending_mask=Some(PendingMask(tokio::task::spawn_blocking(move || {
                let payload=grammar.lock().map_err(|_|failed("grammar unavailable"))?.masks_tree(id,mask_id,&tokens,&parents)?;
                Ok(Frame{kind:3,payload})
            })));
            continue;
        }
        if frame.kind==justfit_native_client_prototype::TOKENS {
            if let Some(grammar)=self.grammar.clone() {
                let count=u32_at(&frame.payload,12)? as usize;
                if count==0 || count>4096 || frame.payload.len()!=16+count*4 || u32_at(&frame.payload,8)? as usize!=self.grammar_offset {return Err(failed("invalid constrained token sequence"));}
                self.grammar_offset+=count;
                let tokens=(0..count).map(|i|u32_at(&frame.payload,16+i*4)).collect::<Result<Vec<_>>>()?;
                self.pending_mask=Some(PendingMask(tokio::task::spawn_blocking(move || {grammar.lock().map_err(|_|failed("grammar unavailable"))?.commit(&tokens)?;Ok(frame)})));
                continue;
            }
        }
        return Ok(frame);
        }
    }
    pub fn cancel(&self) {self.inner.cancel(self.epoch,self.id);}
}
impl Drop for Ticket {
    fn drop(&mut self) {self.inner.abandon(self.epoch,self.id,self.control);}
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn configured_queue_above_64_keeps_controls_available_and_recovers() {
        let slots = Admission::new(65);
        let requests: Vec<_> = (0..65).map(|_| slots.acquire(1).unwrap()).collect();
        let error = slots.acquire(1).unwrap_err();
        assert_eq!((error.0,error.1,error.2), (503,"request queue is full".into(),"frontend_overloaded".into()));
        let status = slots.acquire(4).unwrap();
        let lease = slots.acquire(5).unwrap();
        drop(requests);
        assert!(slots.acquire(1).is_ok());
        drop((status,lease));
    }
    #[tokio::test]
    async fn keepalive_timeout_preserves_the_inflight_grammar_job() {
        let count=Arc::new(std::sync::atomic::AtomicUsize::new(0));let observed=count.clone();
        let mut pending=PendingMask(tokio::spawn(async move {
            observed.fetch_add(1,Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(30)).await;
            Ok(Frame{kind:3,payload:vec![7]})
        }));
        assert!(tokio::time::timeout(Duration::from_millis(1),pending.reply()).await.is_err());
        assert_eq!(pending.reply().await.unwrap(),Frame{kind:3,payload:vec![7]});
        assert_eq!(count.load(Ordering::SeqCst),1);
    }
    fn error(code: &str, retry: bool) -> Vec<u8> {
        let mut p = vec![1, retry as u8];
        p.extend(7u64.to_le_bytes());
        p.extend((code.len() as u32).to_le_bytes());
        p.extend(6u32.to_le_bytes());
        p.extend(code.as_bytes());
        p.extend(b"detail");
        p
    }
    #[test]
    fn deadline_and_retry_errors_keep_the_python_classification() {
        let e = request_error(&error("deadline_exceeded", false));
        assert_eq!((e.0, e.2), (504, "request_timeout".into()));
        let e = request_error(&error("arbitrary_native_retry_code", true));
        assert_eq!((e.0, e.2), (503, "arbitrary_native_retry_code".into()));
        let e = request_error(&error("invalid_count", false));
        assert_eq!(e.0, 400);
    }
    #[test]
    fn error_lengths_are_validated() {
        let mut p = error("invalid_count", false);
        p.pop();
        assert_eq!(request_error(&p).0, 502);
    }
    #[tokio::test]
    async fn deferred_start_preserves_following_events_and_rejects_wrong_ids() {
        let (sender, receiver) = mpsc::channel(4);
        let inner = Inner::for_test();
        inner.routes.lock().unwrap().insert(7, sender.clone());
        let mut ticket = Ticket {
            inner,
            id: 7,
            receiver,
            done: false,
            first: None,
            grammar: None,
            grammar_offset: 0,
            pending_mask: None,
            epoch:1,terminal:Arc::new(Terminal::default()),control:false,
        };
        let start = Frame {
            kind: justfit_native_client_prototype::START,
            payload: [7u64.to_le_bytes().as_slice(), &[0u8; 8]].concat(),
        };
        let wrong = Frame {
            kind: justfit_native_client_prototype::START,
            payload: [8u64.to_le_bytes().as_slice(), &[0u8; 8]].concat(),
        };
        assert!(ticket.unread_start(wrong).is_err());
        sender.send(Ok(start)).await.unwrap();
        sender
            .send(Ok(Frame {
                kind: DONE,
                payload: [7u64.to_le_bytes().as_slice(), &[0u8; 37]].concat(),
            }))
            .await
            .unwrap();
        let first = ticket.next().await.unwrap();
        ticket.unread_start(first).unwrap();
        assert_eq!(
            ticket.next().await.unwrap().kind,
            justfit_native_client_prototype::START
        );
        assert_eq!(ticket.next().await.unwrap().kind, DONE);
        assert!(ticket.done && ticket.inner.routes.lock().unwrap().is_empty());
    }
}
