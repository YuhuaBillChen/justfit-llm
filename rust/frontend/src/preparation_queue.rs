//! The preparation bound is native concurrency, separate from HTTP admission.
use crate::{ApiError,Result};
use serde_json::{json,Value};
use std::{sync::{Arc,atomic::{AtomicUsize,Ordering}},time::{Duration,Instant}};
use tokio::sync::{Semaphore,OwnedSemaphorePermit};
pub struct Queue{slots:Arc<Semaphore>,capacity:usize,waiting:AtomicUsize,metrics:Arc<crate::latency::Metrics>}
pub struct Guard{_permit:OwnedSemaphorePermit,_timer:crate::latency::Timer}
struct Waiting{queue:Arc<Queue>,_timer:crate::latency::Timer}
impl Drop for Waiting{fn drop(&mut self){self.queue.waiting.fetch_sub(1,Ordering::AcqRel);}}
impl Queue{
    pub fn new(capacity:usize,metrics:Arc<crate::latency::Metrics>)->Arc<Self>{assert!(capacity>0);Arc::new(Self{slots:Arc::new(Semaphore::new(capacity)),capacity,waiting:AtomicUsize::new(0),metrics})}
    pub fn snapshot(&self)->Value{json!({"preparation_capacity":self.capacity,"active":self.capacity-self.slots.available_permits(),"waiting":self.waiting.load(Ordering::Acquire)})}
    pub async fn enter(self:&Arc<Self>,timeout:Duration)->Result<Arc<Guard>>{self.enter_bounded(timeout,Duration::from_secs(30)).await}
    async fn enter_bounded(self:&Arc<Self>,timeout:Duration,ceiling:Duration)->Result<Arc<Guard>>{
        let started=Instant::now();self.waiting.fetch_add(1,Ordering::AcqRel);
        let waiting=Waiting{queue:self.clone(),_timer:self.metrics.measure("preparation_queue")};
        let acquired=tokio::time::timeout(timeout.min(ceiling),self.slots.clone().acquire_owned()).await;drop(waiting);
        crate::deadline::remaining(timeout,started.elapsed())?;
        let permit=acquired.ok().and_then(std::result::Result::ok).ok_or_else(||ApiError(503,"frontend preparation capacity is exhausted".into(),"frontend_overloaded".into()))?;
        Ok(Arc::new(Guard{_permit:permit,_timer:self.metrics.measure("preparation")}))
    }
}
#[cfg(test)]
mod tests{
    use super::*;
    #[tokio::test]
    async fn waiting_cancel_timeout_and_owned_work_release_the_exact_slot(){
        let metrics=Arc::new(crate::latency::Metrics::default());let queue=Queue::new(1,metrics.clone());let held=queue.enter(Duration::from_secs(1)).await.unwrap();let work=held.clone();drop(held);
        let q=queue.clone();let task=tokio::spawn(async move{q.enter(Duration::from_secs(10)).await});
        for _ in 0..100{if queue.snapshot()["waiting"]==1{break}tokio::task::yield_now().await;}
        assert_eq!(queue.snapshot(),json!({"preparation_capacity":1,"active":1,"waiting":1}));task.abort();assert!(matches!(task.await,Err(error) if error.is_cancelled()));assert_eq!(queue.snapshot()["waiting"],0);
        let error=queue.enter_bounded(Duration::from_secs(1),Duration::from_millis(10)).await.err().unwrap();assert_eq!((error.0,error.2),(503,"frontend_overloaded".into()));
        let error=queue.enter(Duration::from_millis(10)).await.err().unwrap();assert_eq!((error.0,error.2),(504,"request_timeout".into()));
        assert_eq!(queue.snapshot()["active"],1);drop(work);assert_eq!(queue.snapshot()["active"],0);let next=queue.enter(Duration::from_secs(1)).await.unwrap();drop(next);
        assert_eq!(metrics.snapshot()["preparation_queue"]["count"],5);assert_eq!(metrics.snapshot()["preparation"]["count"],2);
    }
}
