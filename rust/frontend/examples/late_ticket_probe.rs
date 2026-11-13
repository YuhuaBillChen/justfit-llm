use justfit_rust_frontend::engine::Engine;
use justfit_native_client_prototype::text_request;
use std::time::{Duration,Instant};
#[tokio::main]
async fn main(){
    let command:Vec<String>=std::env::args().skip(1).collect();let engine=Engine::spawn(&command,1).unwrap();
    let mut ticket=engine.submit(text_request(engine.next_id(),&[1],8,0,0).unwrap()).unwrap();
    for index in 0..3 {
        let old=engine.owned_pid();assert!(old>0);assert_eq!(unsafe{libc::kill(old as i32,libc::SIGKILL)},0);
        let until=Instant::now()+Duration::from_secs(8);
        while if index<2 {!(engine.healthy()&&engine.owned_pid()!=old)}else{engine.refusal().is_none_or(|e|e.0!=500)} {
            assert!(Instant::now()<until);tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
    let error=ticket.next().await.unwrap_err();println!("{}",serde_json::json!({"old_ticket_status":error.0,"code":error.2}));engine.shutdown();
    assert_eq!(error.0,503,"a later crash loop overwrote an old generation's terminal error");
}
