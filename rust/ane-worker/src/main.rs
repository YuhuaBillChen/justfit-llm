//! Stable native ANE executable: independent of HTTP/tokenizer/codec builds.
#[path = "../../frontend/src/ane_native.rs"]
#[allow(dead_code)] // Shared profile fields and probes are also used by the host.
mod ane_native;
#[path = "../../frontend/src/coreai.rs"]
mod coreai;
#[path = "../../frontend/src/vision_math.rs"]
#[allow(dead_code)] // Keep identical geometry with the separately tested host.
mod vision_math;
#[path = "../../frontend/src/worker_meter.rs"]
mod worker_meter;
use std::path::Path;

#[derive(Debug)]
pub struct ApiError(pub u16, pub String, pub String);
pub type Result<T> = std::result::Result<T, ApiError>;
pub fn bad(message: impl Into<String>) -> ApiError {
    ApiError(400, message.into(), "invalid_request_error".into())
}
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let result=match args.first().map(String::as_str){
  Some("prepare-cache") if args.len()==3=>ane_native::prepare_native_cache(Path::new(&args[1]),Path::new(&args[2])),
  Some("probe") if args.len()==2=>ane_native::probe(Path::new(&args[1])),
  Some("encode") if args.len()==5=>{
   let grids:Vec<(usize,usize)>=serde_json::from_str(&args[4])?;
   let p=ane_native::Profile::open(Path::new(&args[1])).map_err(|e|e.1)?;
   let _meter=worker_meter::Meter::start(&p.library.parent().ok_or("bridge has no parent")?.join("attributed_memory.dylib")).map_err(|e|e.1)?;
   ane_native::encode(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),&grids)
  },
  _=>return Err("usage: justfit-ane-worker prepare-cache TOP_PROFILE RECORD | probe PROFILE | encode PROFILE SOURCE TARGET GRIDS".into()),
 }.map_err(|e|e.1)?;
    println!("{result}");
    Ok(())
}
