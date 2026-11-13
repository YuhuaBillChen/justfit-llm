//! Fixed-memory histograms at the same public stages as the 2.3 service.
use serde_json::{json,Value};
use std::{sync::{Arc,Mutex},time::Instant};
pub const STAGES:[&str;11]=["http_request","upload","preparation_queue","preparation","template","tokenization","grammar","images","native_queue","http_ttft","output_interval"];
pub const BUCKETS:[(&str,f64);18]=[("0.001",0.001),("0.005",0.005),("0.01",0.01),("0.025",0.025),("0.05",0.05),("0.1",0.1),("0.25",0.25),("0.5",0.5),("1",1.),("2.5",2.5),("5",5.),("10",10.),("30",30.),("60",60.),("120",120.),("300",300.),("900",900.),("1800",1800.)];
#[derive(Default)]
struct State{counts:[[u64;19];11],sums:[f64;11]}
#[derive(Default)]
pub struct Metrics(Mutex<State>);
pub struct Timer{metrics:Arc<Metrics>,stage:&'static str,began:Instant,held:Vec<Box<dyn Send+Sync>>}
impl Timer{pub fn hold<T:Send+Sync+'static>(&mut self,value:T){self.held.push(Box::new(value));}}
impl Drop for Timer{fn drop(&mut self){self.metrics.observe(self.stage,self.began.elapsed().as_secs_f64())}}
impl Metrics{
    pub fn measure(self:&Arc<Self>,stage:&'static str)->Timer{Timer{metrics:self.clone(),stage,began:Instant::now(),held:Vec::new()}}
    pub fn observe(&self,stage:&str,seconds:f64){
        if !seconds.is_finite()||seconds<0.{return}
        let Some(index)=STAGES.iter().position(|s|*s==stage)else{return};let bucket=BUCKETS.iter().position(|(_,limit)|seconds<=*limit).unwrap_or(18);
        let mut state=self.0.lock().unwrap();state.counts[index][bucket]=state.counts[index][bucket].saturating_add(1);state.sums[index]=(state.sums[index]+seconds).min(f64::MAX);
    }
    pub fn snapshot(&self)->Value{
        let state=self.0.lock().unwrap();let mut result=json!({});
        for(index,stage)in STAGES.iter().enumerate(){
            let mut buckets=serde_json::Map::new();let mut count=0u64;
            for(bucket,(label,_))in BUCKETS.iter().enumerate(){count=count.saturating_add(state.counts[index][bucket]);buckets.insert((*label).into(),json!(count));}
            count=count.saturating_add(state.counts[index][18]);buckets.insert("+Inf".into(),json!(count));result[*stage]=json!({"buckets":buckets,"count":count,"sum":state.sums[index]});
        }result
    }
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn boundaries_are_inclusive_and_invalid_measurements_do_not_count(){
        let metrics=Metrics::default();for seconds in [0.,0.001,0.002,1800.,1801.,-1.,f64::NAN,f64::INFINITY]{metrics.observe("http_request",seconds)}
        let snapshot=metrics.snapshot();let h=&snapshot["http_request"];assert_eq!(h["count"],5);assert_eq!(h["buckets"]["0.001"],2);assert_eq!(h["buckets"]["0.005"],3);assert_eq!(h["buckets"]["1800"],4);assert_eq!(h["buckets"]["+Inf"],5);assert_eq!(snapshot["grammar"]["count"],0);
    }
}

// A body can reach EOF while Hyper still owns its final frame in a pending
// socket write. Share the receipt with every Bytes owner, so admission and
// timing outlive both the producer and every transport-buffered data frame.
struct Chunk{bytes:axum::body::Bytes,_receipt:Arc<Timer>}
impl AsRef<[u8]> for Chunk{fn as_ref(&self)->&[u8]{self.bytes.as_ref()}}
struct ResponseStream<S>{stream:std::pin::Pin<Box<S>>,timer:Option<Arc<Timer>>}
impl<S,E> tokio_stream::Stream for ResponseStream<S> where S:tokio_stream::Stream<Item=std::result::Result<axum::body::Bytes,E>>{
    type Item=std::result::Result<axum::body::Bytes,E>;
    fn poll_next(self:std::pin::Pin<&mut Self>,cx:&mut std::task::Context<'_>)->std::task::Poll<Option<Self::Item>>{
        let this=self.get_mut();match this.stream.as_mut().poll_next(cx){
            std::task::Poll::Ready(Some(Ok(bytes)))=>std::task::Poll::Ready(Some(Ok(axum::body::Bytes::from_owner(Chunk{bytes,_receipt:this.timer.as_ref().expect("body receipt before EOF").clone()})))),
            std::task::Poll::Ready(None)=>{this.timer.take();std::task::Poll::Ready(None)},
            result=>result,
        }
    }
}
pub fn response_body(body:axum::body::Body,timer:Timer)->axum::body::Body{
    axum::body::Body::from_stream(ResponseStream{stream:Box::pin(body.into_data_stream()),timer:Some(Arc::new(timer))})
}
#[cfg(test)]
mod response_tests{
    use super::*;
    use tokio_stream::StreamExt;
    use axum::body::Bytes;
    use std::convert::Infallible;
    #[tokio::test]
    async fn response_receipt_survives_body_eof_until_all_transport_bytes_drop(){
        let metrics=Arc::new(Metrics::default());let slots=Arc::new(tokio::sync::Semaphore::new(1));let mut timer=metrics.measure("http_request");timer.hold(slots.clone().acquire_owned().await.unwrap());
        let mut stream=ResponseStream{stream:Box::pin(tokio_stream::iter([Ok::<_,Infallible>(Bytes::from_static(b"a"))])),timer:Some(Arc::new(timer))};
        let frame=stream.next().await.unwrap().unwrap();let copy=frame.clone();assert_eq!(frame.as_ref(),b"a");
        assert!(stream.next().await.is_none());drop(stream);assert_eq!(metrics.snapshot()["http_request"]["count"],0);assert_eq!(slots.available_permits(),0);
        drop(frame);assert_eq!(metrics.snapshot()["http_request"]["count"],0);drop(copy);assert_eq!(metrics.snapshot()["http_request"]["count"],1);assert_eq!(slots.available_permits(),1);
        let stream=ResponseStream{stream:Box::pin(tokio_stream::pending::<std::result::Result<Bytes,Infallible>>()),timer:Some(Arc::new(metrics.measure("http_request")))};
        drop(stream);assert_eq!(metrics.snapshot()["http_request"]["count"],2);
    }
}
