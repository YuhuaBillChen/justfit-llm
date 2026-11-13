use justfit_rust_frontend::engine::Engine;
use serde_json::json;
use std::time::{Duration,Instant};
#[tokio::main]
async fn main() {
    let command:Vec<String>=std::env::args().skip(1).collect();
    let engine=Engine::spawn(&command,32).unwrap();
    let before=engine.status().await.unwrap();let old=engine.owned_pid();
    assert_eq!(before["test_pid"],old);
    // This PID is the exact fake child returned by our Engine object.
    assert_eq!(unsafe{libc::kill(old as i32,libc::SIGKILL)},0);
    let until=Instant::now()+Duration::from_secs(3);let mut recovered=false;let mut after=serde_json::Value::Null;
    while Instant::now()<until {
        if engine.healthy()&&engine.owned_pid()!=old {
            if let Ok(value)=engine.status().await{after=value;recovered=true;break}
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    println!("{}",json!({"recovered":recovered,"before_pid":old,"after_pid":engine.owned_pid(),"before":before,"after":after}));
    engine.shutdown();assert!(recovered,"native crash did not recover automatically");
}
