use justfit_rust_frontend::{documents::{self,Documents,Config}};
use serde_json::{json,Value};
use std::{io::{self,BufRead},path::PathBuf};
fn main()->Result<(),Box<dyn std::error::Error>> {
    let args:Vec<_>=std::env::args().skip(1).collect();
    if args.first().map(String::as_str)==Some("pdf-worker") {
        // Fault injection exists only in this acceptance probe executable.
        match std::env::var("JUSTFIT_PROBE_PDF_FAULT").as_deref() {
            Ok("crash")=>std::process::exit(71),
            Ok("malformed")=>{println!("not-json");return Ok(())},
            Ok("sleep")=>{std::thread::sleep(std::time::Duration::from_secs(35));return Ok(())},
            Ok("memory")=>{let bytes=vec![7u8;1100*1024*1024];std::hint::black_box(&bytes);std::thread::sleep(std::time::Duration::from_secs(5));return Ok(())},
            Ok("output")=>{use std::io::Write;io::stdout().lock().write_all(&vec![b'x';65*1024*1024])?;return Ok(())},
            _=>()
        }
        return documents::worker(&args[1],&args[2],&args[3]).map_err(|e|e.1.into());
    }
    let docs=Documents::new(Config{library:PathBuf::from(&args[0]),zlib:PathBuf::from(&args[1])});
    for line in io::stdin().lock().lines() {
        let body:Value=serde_json::from_str(&line?)?;
        let cancel=body.get("cancel_after_ms").and_then(Value::as_u64).map(|n|std::time::Instant::now()+std::time::Duration::from_millis(n));
        let check=||if cancel.is_some_and(|end|std::time::Instant::now()>=end){Err(justfit_rust_frontend::ApiError(499,"request cancelled".into(),"request_cancelled".into()))}else{Ok(())};
        let result=docs.expand(&body,&check);
        let row=match result {Ok(v)=>json!({"status":200,"body":v,"stats":docs.stats()}),Err(e)=>json!({"status":e.0,"body":e.body(),"stats":docs.stats()})};
        println!("{row}");
    }
    Ok(())
}
