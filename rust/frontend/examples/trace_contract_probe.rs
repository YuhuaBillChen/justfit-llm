use justfit_rust_frontend::engine::Engine;
use justfit_native_client_prototype::text_request;
use std::time::{Duration,Instant};
#[tokio::main]
async fn main(){
    let command:Vec<String>=std::env::args().skip(1).collect();let engine=Engine::spawn(&command,1).unwrap();
    engine.control(0,0,Duration::from_secs(1)).await.unwrap();
    let ticket=engine.submit(text_request(engine.next_id(),&[1,2,3],8,0,0).unwrap()).unwrap();
    let old=engine.owned_pid();assert_eq!(unsafe{libc::kill(old as i32,libc::SIGKILL)},0);
    let until=Instant::now()+Duration::from_secs(3);let mut trace=serde_json::Value::Null;
    while Instant::now()<until{
        let status=engine.status().await.unwrap();
        if !status["transport"]["last_crash_trace"].is_null(){trace=status["transport"]["last_crash_trace"].clone();break}
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    drop(ticket);engine.shutdown();println!("{}",serde_json::json!({"trace":trace,"old_pid":old}));assert_eq!(trace.is_string(),std::env::var("JUSTFIT_CRASH_TRACE").as_deref()==Ok("1"),"native crash trace enablement mismatch");
}
