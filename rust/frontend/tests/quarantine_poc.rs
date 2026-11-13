use justfit_rust_frontend::engine::Engine;
use justfit_native_client_prototype::{text_request, START, DONE};
use std::time::{Duration, Instant};

// Test oracle only: speaks native protocol on pipes; no HTTP, models or GPU.
const FAKE: &str = r#"
import os, signal, struct, sys, resource
resource.setrlimit(resource.RLIMIT_CORE, (0,0))
def emit(kind, payload):
    sys.stdout.buffer.write(struct.pack('<4sHHQ', b'SPLH', 9, kind, len(payload))+payload)
    sys.stdout.buffer.flush()
emit(0x100, struct.pack('<IIB',2,262144,0))
mode=sys.argv[1] if len(sys.argv)>1 else 'fault'
started=0
while True:
    h=sys.stdin.buffer.read(16)
    if not h: break
    _,_,kind,n=struct.unpack('<4sHHQ',h)
    p=sys.stdin.buffer.read(n)
    if kind==1:
        if mode!='queued': emit(0x101,p[:8]+bytes(8))
        started+=1
        if mode=='hold': continue
        if mode=='batch' and started<2: continue
        if struct.unpack_from('<I',p,94)[0]==17:
            os.kill(os.getpid(), signal.SIGABRT)
        emit(0x104,p[:8]+bytes(37))
"#;

async fn ready(engine: &Engine) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !engine.healthy() {
        assert!(Instant::now() < deadline, "engine failed to recover");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

#[tokio::test]
async fn same_started_request_after_two_faults_is_refused_before_third_engine_crash() {
    let command = vec!["/usr/bin/python3".into(), "-c".into(), FAKE.into()];
    let engine = Engine::spawn(&command, 8).unwrap();
    for id in [1, 2] {
        let mut ticket = engine.submit(text_request(id, &[17, 18], 16, 0, 0).unwrap()).unwrap();
        // A fast crash may make next() return the terminal error before Start.
        let event = tokio::time::timeout(Duration::from_secs(5), ticket.next()).await.unwrap();
        if let Ok(frame) = event {
            assert_eq!(frame.kind, START);
            assert!(tokio::time::timeout(Duration::from_secs(5), ticket.next()).await.unwrap().is_err());
        }
        ready(&engine).await;
    }
    let pid = engine.owned_pid();
    let result = engine.submit(text_request(3, &[17, 18], 16, 0, 0).unwrap());
    let error = match result {
        Err(error) => error,
        Ok(_) => {engine.shutdown(); panic!("a twice crashing request was admitted again")},
    };
    assert_eq!((error.0, error.2.as_str()), (400, "request_quarantined"));
    assert!(engine.healthy());
    assert_eq!(engine.owned_pid(), pid);
    let mut healthy = engine.submit(text_request(4, &[99, 100], 16, 0, 0).unwrap()).unwrap();
    assert_eq!(healthy.next().await.unwrap().kind, START);
    assert_eq!(healthy.next().await.unwrap().kind, DONE);
    assert_eq!(engine.owned_pid(), pid);
    engine.shutdown();
}

#[tokio::test]
async fn manual_kills_do_not_quarantine_a_started_request() {
    let command = vec!["/usr/bin/python3".into(), "-c".into(), FAKE.into(), "hold".into()];
    let engine = Engine::spawn(&command, 8).unwrap();
    for id in [1, 2] {
        let mut ticket = engine.submit(text_request(id, &[17], 16, 0, 0).unwrap()).unwrap();
        assert_eq!(ticket.next().await.unwrap().kind, START);
        engine.kill_owned();
        assert!(tokio::time::timeout(Duration::from_secs(5), ticket.next()).await.unwrap().is_err());
        ready(&engine).await;
    }
    let mut ticket = engine.submit(text_request(3, &[17], 16, 0, 0).unwrap()).unwrap();
    assert_eq!(ticket.next().await.unwrap().kind, START);
    engine.shutdown();
}

#[tokio::test]
async fn request_without_start_event_is_not_implicated_in_a_fault() {
    let command = vec!["/usr/bin/python3".into(), "-c".into(), FAKE.into(), "queued".into()];
    let engine = Engine::spawn(&command, 8).unwrap();
    for id in [1, 2] {
        let mut ticket = engine.submit(text_request(id, &[17], 16, 0, 0).unwrap()).unwrap();
        assert!(tokio::time::timeout(Duration::from_secs(5), ticket.next()).await.unwrap().is_err());
        ready(&engine).await;
    }
    let result = engine.submit(text_request(3, &[17], 16, 0, 0).unwrap());
    assert!(result.is_ok(), "an unstarted request was quarantined");
    engine.shutdown();
}

#[tokio::test]
async fn ambiguous_concurrent_faults_do_not_quarantine_either_request() {
    let command = vec!["/usr/bin/python3".into(), "-c".into(), FAKE.into(), "batch".into()];
    let engine = Engine::spawn(&command, 8).unwrap();
    for base in [1, 3] {
        let mut first = engine.submit(text_request(base, &[17], 16, 0, 0).unwrap()).unwrap();
        assert_eq!(first.next().await.unwrap().kind, START);
        let mut second = engine.submit(text_request(base+1, &[17,18], 16, 0, 0).unwrap()).unwrap();
        let _ = tokio::time::timeout(Duration::from_secs(5), second.next()).await.unwrap();
        assert!(tokio::time::timeout(Duration::from_secs(5), first.next()).await.unwrap().is_err());
        ready(&engine).await;
    }
    let mut first = engine.submit(text_request(5, &[17], 16, 0, 0).unwrap()).unwrap();
    assert_eq!(first.next().await.unwrap().kind, START);
    let second = engine.submit(text_request(6, &[17,18], 16, 0, 0).unwrap());
    assert!(second.is_ok(), "a concurrent peer was quarantined");
    engine.shutdown();
}
