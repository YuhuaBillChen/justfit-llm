use base64::{engine::general_purpose::STANDARD,Engine as _};
use serde_json::{json,Value};
use std::{fs,path::PathBuf,process::{Command,Output,Stdio},sync::atomic::{AtomicU64,Ordering},time::{Duration,Instant,SystemTime,UNIX_EPOCH}};
static NEXT:AtomicU64=AtomicU64::new(0);
struct Fixture(PathBuf);
impl Fixture{
 fn new()->Self{let path=std::env::temp_dir().join(format!("justfit replay {}-{}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)));fs::create_dir(&path).unwrap();Self(path)}
 fn command(&self)->Command{let binary=std::env::var_os("JUSTFIT_TEST_REPLAY_CLI").unwrap_or_else(||env!("CARGO_BIN_EXE_justfit-rs").into());let mut command=Command::new(binary);command.env("PATH","").current_dir(&self.0);command}
 fn trace(&self,command:Value,frames:Value)->PathBuf{let path=self.0.join("trace.json");fs::write(&path,serde_json::to_vec(&json!({"schema_version":1,"command":command,"frames":frames})).unwrap()).unwrap();path}
 fn run(&self,path:&std::path::Path)->Output{self.command().arg("replay").arg(path).output().unwrap()}
}
impl Drop for Fixture{fn drop(&mut self){fs::remove_dir_all(&self.0).unwrap();}}
fn ready()->String{let mut bytes=b"SPLH".to_vec();bytes.extend(9u16.to_le_bytes());bytes.extend(0x100u16.to_le_bytes());bytes.extend(9u64.to_le_bytes());bytes.extend(2u32.to_le_bytes());bytes.extend(262144u32.to_le_bytes());bytes.push(1);bytes.iter().map(|b|format!("\\{b:03o}")).collect()}
fn outgoing(offset:u64,bytes:&[u8])->Value{json!({"direction":"client_to_engine","offset_micros":offset,"frame_base64":STANDARD.encode(bytes)})}
#[test]
fn replay_preserves_wire_bytes_refreshes_deadline_and_propagates_exit(){
 let fixture=Fixture::new();let captured=fixture.0.join("captured frames");let mut request=vec![0x7f;114];request[54..58].copy_from_slice(&0x7fa00001u32.to_le_bytes());request[6..8].copy_from_slice(&1u16.to_le_bytes());request[34..42].copy_from_slice(&123456u64.to_le_bytes());
 let script=format!("printf '{}'; /bin/cat > \"$1\"; exit 7",ready());let trace=fixture.trace(json!(["/bin/sh","-c",script,"engine-fixture",captured]),json!([outgoing(9000,&request),{"direction":"engine_to_client","omitted_bytes":999},outgoing(29000,b"short invalid frame")]));
 let before=SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as u64;let started=Instant::now();let result=fixture.run(&trace);assert_eq!(result.status.code(),Some(7),"{}",String::from_utf8_lossy(&result.stderr));assert!(started.elapsed()>=Duration::from_millis(18));let after=SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as u64;
 let bytes=fs::read(captured).unwrap();assert_eq!(&bytes[..26],&request[..26]);assert_eq!(&bytes[34..114],&request[34..]);assert_eq!(&bytes[114..],b"short invalid frame");let deadline=u64::from_le_bytes(bytes[26..34].try_into().unwrap());assert!((before+123456..=after+123456).contains(&deadline));
}
#[test]
fn invalid_trace_rejected_before_spawning_command(){
 let fixture=Fixture::new();let marker=fixture.0.join("must not exist");
 for frames in [json!([]),json!([{"direction":"engine_to_client","frame_base64":""}]),json!([{"direction":"client_to_engine","offset_micros":0,"frame_base64":"!"}]),json!([{"direction":"client_to_engine","offset_micros":true,"frame_base64":"AA=="}]),json!([{"direction":"client_to_engine","offset_micros":-1,"frame_base64":"AA=="}]),json!([{"direction":"client_to_engine","offset_micros":0,"frame_base64":"AA==","omitted_bytes":0}])]{let trace=fixture.trace(json!(["/usr/bin/touch",marker]),frames);assert_eq!(fixture.run(&trace).status.code(),Some(2));assert!(!marker.exists());}
}
#[test]
fn early_eof_and_bad_output_fail_without_hanging(){
 let fixture=Fixture::new();for script in ["exit 7","printf 'bad'"]{let trace=fixture.trace(json!(["/bin/sh","-c",script]),json!([outgoing(0,b"input")]));let started=Instant::now();let output=fixture.run(&trace);assert_eq!(output.status.code(),Some(1),"{}",String::from_utf8_lossy(&output.stderr));assert!(started.elapsed()<Duration::from_secs(3));}
}
#[test]
fn stdout_eof_ends_backpressured_input_and_reaps_engine(){
 let fixture=Fixture::new();let pid_file=fixture.0.join("engine.pid");let script=format!("printf '%s' \"$$\" > \"$1\"; printf '{}'; exec 1>&-; exec /bin/sleep 30",ready());let trace=fixture.trace(json!(["/bin/sh","-c",script,"engine-fixture",pid_file]),json!([outgoing(0,&vec![42;4*1024*1024])]));let started=Instant::now();let output=fixture.run(&trace);assert_eq!(output.status.code(),Some(1),"{}",String::from_utf8_lossy(&output.stderr));assert!(started.elapsed()<Duration::from_secs(5));let pid:i32=fs::read_to_string(pid_file).unwrap().parse().unwrap();assert_eq!(unsafe{libc::kill(pid,0)},-1);
}
#[test]
fn interrupt_replay_reaps_its_own_engine(){
 let fixture=Fixture::new();let pid_file=fixture.0.join("engine.pid");let script=format!("printf '%s' \"$$\" > \"$1\"; printf '{}'; exec /bin/sleep 30",ready());let trace=fixture.trace(json!(["/bin/sh","-c",script,"engine-fixture",pid_file]),json!([outgoing(0,b"input"),outgoing(60_000_000,b"later")]));let mut child=fixture.command().arg("replay").arg(trace).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();let deadline=Instant::now()+Duration::from_secs(5);
 let pid:i32=loop{if let Ok(text)=fs::read_to_string(&pid_file){if let Ok(pid)=text.parse(){break pid;}}assert!(child.try_wait().unwrap().is_none());assert!(Instant::now()<deadline);std::thread::sleep(Duration::from_millis(10));};assert_eq!(unsafe{libc::kill(child.id() as i32,libc::SIGINT)},0);let output=child.wait_with_output().unwrap();assert_eq!(output.status.code(),Some(130),"{}",String::from_utf8_lossy(&output.stderr));assert_eq!(unsafe{libc::kill(pid,0)},-1);
}
