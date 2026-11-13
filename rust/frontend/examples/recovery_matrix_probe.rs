use justfit_rust_frontend::engine::Engine;
use justfit_native_client_prototype::{text_request,Frame,DONE};
use serde_json::json;
use std::{sync::{Arc,atomic::{AtomicBool,Ordering}},time::{Duration,Instant}};
fn kill(engine:&Engine)->u32 {let pid=engine.owned_pid();assert!(pid>0);assert_eq!(unsafe{libc::kill(pid as i32,libc::SIGKILL)},0);pid}
async fn recovered(engine:&Engine,old:u32,timeout:Duration){
    let until=Instant::now()+timeout;
    loop {if engine.healthy()&&engine.owned_pid()!=old {let s=engine.control(0,0,Duration::from_secs(1)).await.unwrap();assert_eq!(s["test_pid"],engine.owned_pid());return}assert!(Instant::now()<until,"recovery timeout");tokio::time::sleep(Duration::from_millis(10)).await;}
}
async fn wait_failure(engine:&Engine){let until=Instant::now()+Duration::from_secs(3);while engine.refusal().is_none_or(|e|e.0!=500){assert!(Instant::now()<until,"fatal timeout");tokio::time::sleep(Duration::from_millis(10)).await;}}
#[tokio::main]
async fn main(){
    let args:Vec<String>=std::env::args().skip(1).collect();assert_eq!(args.len(),5);
    let scenario=&args[4];let engine=Arc::new(Engine::spawn(&args[..4],1).unwrap());let before=engine.control(0,0,Duration::from_secs(1)).await.unwrap();assert_eq!(before["test_pid"],engine.owned_pid());
    let began=Instant::now();
    match scenario.as_str(){
        "shutdown-flush"=>{
            let pid=engine.owned_pid();let start=Instant::now();engine.shutdown();assert!(start.elapsed()>=Duration::from_millis(750));
            let marker=std::path::Path::new(&args[2]).join(format!("graceful-{pid}.json"));assert!(marker.exists(),"native shutdown flush lost");
        }
        "write-stall"=>{
            let mut payload=vec![0u8;16*1024*1024];payload[..8].copy_from_slice(&engine.next_id().to_le_bytes());
            let core=engine.clone();let start=Instant::now();let old=engine.owned_pid();
            let reply=tokio::task::spawn_blocking(move||core.submit(Frame{kind:1,payload})).await.unwrap();
            assert_eq!(reply.err().unwrap().0,503);assert!(start.elapsed()>=Duration::from_millis(4900)&&start.elapsed()<Duration::from_secs(8));
            recovered(&engine,old,Duration::from_secs(3)).await;
        }
        "liveness-timeout"=>{
            let old=engine.owned_pid();let status=engine.status().await.unwrap();assert_eq!(status["ready"],true);assert_eq!(status["transport"]["status_stale"],true);
            recovered(&engine,old,Duration::from_secs(34)).await;assert!(began.elapsed()>=Duration::from_millis(29900));
        }
        "pending-fencing"=>{
            let mut ticket=engine.submit(text_request(engine.next_id(),&[1],8,0,0).unwrap()).unwrap();let epoch=engine.generation();let cleanup=Arc::new(AtomicBool::new(false));let observed=cleanup.clone();
            engine.on_generation_failure(epoch,move||{std::thread::sleep(Duration::from_millis(40));observed.store(true,Ordering::Release)});
            let old=kill(&engine);recovered(&engine,old,Duration::from_secs(3)).await;assert!(cleanup.load(Ordering::Acquire));
            let error=ticket.next().await.unwrap_err();assert_eq!((error.0,error.2.as_str()),(503,"runtime_unavailable"));
            assert_eq!(engine.control_generation(epoch,11,1234,Duration::from_secs(1)).await.unwrap_err().0,503);
            engine.fail_generation(epoch,"late media callback");assert!(engine.healthy());drop(ticket);
            assert_eq!(engine.status().await.unwrap()["transport"]["pending"],0);
        }
        "backoff-fatal"=>{
            let old=kill(&engine);recovered(&engine,old,Duration::from_secs(3)).await;
            let second=Instant::now();let old=kill(&engine);recovered(&engine,old,Duration::from_secs(8)).await;assert!(second.elapsed()>=Duration::from_millis(4900));
            kill(&engine);wait_failure(&engine).await;assert_eq!(engine.status().await.unwrap()["transport"]["restarts"],2);
            assert_eq!(engine.status().await.unwrap()["ready"],false);
        }
        "changed-ready"=>{kill(&engine);wait_failure(&engine).await;assert!(engine.refusal().unwrap().1.contains("context window, concurrency or vision changed"));}
        "shutdown-restart"=>{
            let old=kill(&engine);let until=Instant::now()+Duration::from_secs(3);
            while engine.owned_pid()==0||engine.owned_pid()==old {assert!(Instant::now()<until);tokio::time::sleep(Duration::from_millis(10)).await;}
            let pid=engine.owned_pid();let start=Instant::now();engine.shutdown();assert!(start.elapsed()<Duration::from_secs(2));assert_ne!(unsafe{libc::kill(pid as i32,0)},0);assert_eq!(engine.refusal().unwrap().2,"server_shutdown");
        }
        "cancel-reservation"=>{
            let ticket=engine.submit(text_request(engine.next_id(),&[1],8,0,0).unwrap()).unwrap();drop(ticket);
            assert_eq!(engine.submit(text_request(engine.next_id(),&[1],8,0,0).unwrap()).err().unwrap().1,"request queue is full");
            assert_eq!(engine.status().await.unwrap()["transport"]["pending"],1);
            let old=kill(&engine);recovered(&engine,old,Duration::from_secs(3)).await;assert_eq!(engine.status().await.unwrap()["transport"]["pending"],0);
        }
        "cancel-ack"=>{
            let ticket=engine.submit(text_request(engine.next_id(),&[1],8,0,0).unwrap()).unwrap();drop(ticket);tokio::time::sleep(Duration::from_millis(60)).await;
            let ticket=engine.submit(text_request(engine.next_id(),&[1],8,0,0).unwrap()).unwrap();drop(ticket);
        }
        "busy-status"=>{
            let status=engine.status().await.unwrap();assert_eq!(status["ready"],true);assert_eq!(status["transport"]["status_stale"],true);
            tokio::time::sleep(Duration::from_millis(400)).await;
            let live=engine.control(0,0,Duration::from_secs(1)).await.unwrap();assert_eq!(live["ready"],true);assert_eq!(engine.generation(),1);
        }
        "buffered-done"=>{
            let mut ticket=engine.submit(text_request(engine.next_id(),&[1],8,0,0).unwrap()).unwrap();tokio::time::sleep(Duration::from_millis(80)).await;
            let old=kill(&engine);recovered(&engine,old,Duration::from_secs(3)).await;
            loop {if ticket.next().await.unwrap().kind==DONE{break}}
            // This old completed consumer may drop after the next engine is ready.
            drop(ticket);assert!(engine.healthy());
        }
        "malformed"=>{
            // Test process mode sends an invalid frame after the second status.
            let _=engine.control(0,0,Duration::from_secs(1)).await;
            let until=Instant::now()+Duration::from_secs(3);
            while engine.generation()==1{assert!(Instant::now()<until);tokio::time::sleep(Duration::from_millis(10)).await;}
        }
        _=>panic!("unknown scenario"),
    }
    let final_status=engine.status().await.unwrap();engine.shutdown();assert_eq!(engine.owned_pid(),0);
    println!("{}",json!({"scenario":scenario,"passed":true,"seconds":began.elapsed().as_secs_f64(),"status":final_status}));
}
