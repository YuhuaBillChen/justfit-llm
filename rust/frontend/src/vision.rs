//! Bounded connection to the existing profile-pinned media/ANE driver.
//! Rust owns HTTP and the native generation. The driver owns image validation,
//! preprocessing and ANE child lifetimes, and cannot spawn a native engine.
use crate::{bad, engine::Engine, generation::Prepared, ApiError, Result};
use justfit_native_client_prototype::{Frame, VERSION};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    io::{BufReader, Read, Write},
    os::unix::process::CommandExt,
    path::PathBuf,
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};
use tokio::sync::Semaphore;

#[derive(Clone, Deserialize, Serialize)]
pub struct Config {
    pub command: Vec<String>,
    pub model_root: PathBuf,
    pub profile: Option<PathBuf>,
    pub host_reserve: u64,
    #[serde(default)]
    pub native_catalog: Option<PathBuf>,
    #[serde(default)]
    pub native_worker: Option<PathBuf>,
    #[serde(default)]
    pub native_cache_home: Option<PathBuf>,
    #[serde(default)]
    pub native_prepare: bool,
    #[serde(default)]
    pub native_distribution: Option<crate::ane_distribution::Delivery>,
}

pub fn unavailable()->ApiError{crate::bad("this model is serving without vision (started with --language-only)")}
fn failed(message: impl Into<String>) -> ApiError {
    ApiError(502, message.into(), "vision_driver_unavailable".into())
}
fn remaining_timeout(timeout: Duration, elapsed: Duration) -> Result<Duration> {
    crate::deadline::remaining(timeout, elapsed).map_err(|_| ApiError(
        504, "vision preparation timed out".into(), "request_timeout".into()))
}
fn packet(writer: &mut impl Write, value: &Value) -> std::io::Result<()> {
    let bytes = serde_json::to_vec(value)?;
    if bytes.len() > 128 * 1024 * 1024 {
        return Err(std::io::Error::other("vision JSON too large"));
    }
    writer.write_all(&(bytes.len() as u32).to_le_bytes())?;
    writer.write_all(&bytes)?;
    writer.write_all(&0u32.to_le_bytes())?;
    writer.flush()
}
fn receive(reader: &mut impl Read) -> std::io::Result<(Value, Vec<u8>)> {
    let mut size = [0u8; 4];
    reader.read_exact(&mut size)?;
    let size_json = u32::from_le_bytes(size) as usize;
    if !(2..=1024 * 1024).contains(&size_json) {
        return Err(std::io::Error::other("vision metadata length invalid"));
    }
    let mut bytes = vec![0; size_json];
    reader.read_exact(&mut bytes)?;
    let value = serde_json::from_slice(&bytes)?;
    reader.read_exact(&mut size)?;
    let size_blob = u32::from_le_bytes(size) as usize;
    if size_blob > 256 * 1024 * 1024 + 16 {
        return Err(std::io::Error::other("vision payload length invalid"));
    }
    let mut blob = Vec::new();
    blob.try_reserve_exact(size_blob)
        .map_err(|_| std::io::Error::other("vision allocation refused"))?;
    blob.resize(size_blob, 0);
    reader.read_exact(&mut blob)?;
    Ok((value, blob))
}
fn native_request(mut bytes: Vec<u8>, id: u64) -> Result<Frame> {
    if bytes.len() < 110
        || &bytes[..4] != b"SPLH"
        || u16::from_le_bytes(bytes[4..6].try_into().unwrap()) != VERSION
        || u16::from_le_bytes(bytes[6..8].try_into().unwrap()) != 1
        || u64::from_le_bytes(bytes[8..16].try_into().unwrap()) != (bytes.len() - 16) as u64
        || u64::from_le_bytes(bytes[16..24].try_into().unwrap()) != id
    {
        return Err(failed("invalid prepared media frame"));
    }
    // Reuse the large IPC allocation instead of copying pixels/embeddings into
    // a second payload allocation. Header removal shifts within this buffer.
    bytes.drain(..16);
    Ok(Frame {
        kind: 1,
        payload: bytes,
    })
}

struct Driver {
    child: Mutex<Child>,
    stop_lock: Mutex<()>,
    epoch:u64,
    writer: Mutex<ChildStdin>,
    reader: Mutex<BufReader<ChildStdout>>,
    engine: Arc<Engine>,
    metrics: Mutex<Value>,
    image_cache: Mutex<Value>,
    rgb_bytes: AtomicU64,
    stopped: AtomicBool,
    payload_bytes: AtomicU64,
}
impl Driver {
    fn send(&self, value: Value) -> Result<()> {
        let local=matches!(value["op"].as_str(),Some("render"|"count"|"release"|"cancel"));
        if self.stopped.load(Ordering::Acquire)||(!local&&(self.epoch!=self.engine.generation()||!self.engine.healthy())){return Err(self.engine.refusal().unwrap_or_else(||ApiError(503,"the inference engine stopped unexpectedly and is restarting; retry the request".into(),"runtime_unavailable".into())))}
        packet(&mut *self.writer.lock().unwrap(), &value).map_err(|e| failed(e.to_string()))
    }
    // This process group was created by us, and contains only this driver and
    // its worker children. Reap it before native reservations can be discarded.
    fn stop(&self) {
        let _stop=self.stop_lock.lock().unwrap();
        if self.stopped.swap(true, Ordering::AcqRel) {
            return;
        }
        let mut child = self.child.lock().unwrap();
        if matches!(child.try_wait(), Ok(Some(_))) {
            // A crashed parent can leave a worker in its original process group.
            let _ = Command::new("/bin/kill")
                .args(["-KILL", &format!("-{}", child.id())])
                .stderr(Stdio::null())
                .status();
            return;
        }
        let group = format!("-{}", child.id());
        let _ = Command::new("/bin/kill")
            .args(["-TERM", &group])
            .stderr(Stdio::null())
            .status();
        for _ in 0..50 {
            if matches!(child.try_wait(), Ok(Some(_))) {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let _ = Command::new("/bin/kill")
            .args(["-KILL", &group])
            .stderr(Stdio::null())
            .status();
        let _ = child.wait();
    }
    fn pump(&self, handle: &tokio::runtime::Handle) -> Result<(Value, Vec<u8>)> {
        let mut reader = self.reader.lock().unwrap();
        loop {
            let (value, blob) = receive(&mut *reader).map_err(|e| failed(e.to_string()))?;
            if value["op"] == "fatal" {
                return Err(failed("vision driver could not release external memory"));
            }
            if value["op"] != "native" {
                return Ok((value, blob));
            }
            let lease = value["lease"]
                .as_u64()
                .ok_or_else(|| failed("invalid lease ID"))?;
            let size = value["size"]
                .as_u64()
                .ok_or_else(|| failed("invalid lease size"))?;
            let timeout = value["timeout"]
                .as_f64()
                .filter(|n| n.is_finite() && *n > 0. && *n <= 30.)
                .ok_or_else(|| failed("invalid control deadline"))?;
            let result = handle.block_on(self.engine.control_generation(
                self.epoch, lease,
                size,
                Duration::from_secs_f64(timeout),
            ));
            self.send(match result {
                Ok(status) => json!({"op":"native_result","status":status}),
                Err(e) => json!({"op":"native_result","error":e.1}),
            })?;
        }
    }
    fn transaction(&self, handle: &tokio::runtime::Handle) -> Result<(Value, Vec<u8>)> {
        let result = self.pump(handle);
        if result.is_err() {
            let planned=self.stopped.load(Ordering::Acquire);
            self.stop();
            // A dead driver cannot prove lease release. Fail this owned engine
            // generation closed, rather than silently leaking its reservation.
            if !planned {self.engine.fail_generation(self.epoch,"vision driver could not release external memory");}
            return Err(self.engine.refusal().unwrap_or_else(||ApiError(503,"the inference engine stopped unexpectedly and is restarting; retry the request".into(),"runtime_unavailable".into())));
        }
        result
    }
}
impl Drop for Driver {
    fn drop(&mut self) {
        self.stop();
    }
}

#[derive(Clone)]
struct Spec {command:Vec<String>,packet:Value}
pub struct Vision {
    driver: Mutex<Arc<Driver>>,
    engine:Arc<Engine>,
    spec:Spec,
    serial: Arc<Semaphore>,
    modalities: Vec<String>,
    fallback: Arc<crate::Frontend>,
    preparation: Mutex<Option<Arc<crate::native_prepare::Preparation>>>,
    prepare_config: Option<crate::native_prepare::Config>,
    activated: AtomicBool,
}
pub struct Owner {
    driver: Arc<Driver>,
    id: u64,
    bytes: u64,
    rgb: u64,
}
impl Drop for Vision {
    fn drop(&mut self) {
        if let Some(preparation)=self.preparation.get_mut().unwrap() {preparation.stop();}
    }
}
impl Owner {pub fn generation(&self)->u64{self.driver.epoch}}
impl Drop for Owner {
    fn drop(&mut self) {
        self.driver.rgb_bytes.fetch_sub(self.rgb,Ordering::AcqRel);
        self.driver
            .payload_bytes
            .fetch_sub(self.bytes, Ordering::AcqRel);
        let _ = self.driver.send(json!({"op":"release","id":self.id}));
    }
}
struct Cancellation {
    driver: Arc<Driver>,
    id: u64,
    armed: bool,
}
impl Drop for Cancellation {
    fn drop(&mut self) {
        if self.armed {
            let _ = self.driver.send(json!({"op":"cancel","id":self.id}));
        }
    }
}

impl Vision {
    pub async fn start(
        config: Config,
        engine: Arc<Engine>,
        tokenizer: PathBuf,
        fallback: Arc<crate::Frontend>,
    ) -> Result<Self> {
        if config.command.is_empty() || config.host_reserve == 0 || !engine.vision {
            return Err(bad(
                "vision driver requires an executable, explicit host reserve and vision model",
            ));
        }
        let prepare = if config.native_prepare {
            Some(crate::native_prepare::Config {
                profile:config.profile.clone().ok_or_else(||bad("background ANE profile missing"))?,
                worker:config.native_worker.clone().ok_or_else(||bad("background ANE worker missing"))?,
                catalog:config.native_catalog.clone().ok_or_else(||bad("background ANE catalog missing"))?,
                cache_home:config.native_cache_home.clone().ok_or_else(||bad("background ANE private home missing"))?,
                model_root:config.model_root.clone(),
                distribution:config.native_distribution.clone(),
            })
        } else {None};
        let spec=Spec {command:config.command,packet:json!({"tokenizer":tokenizer,"model":fallback.model,"served_model_names":fallback.model_names,"documents":fallback.documents.config,"preparation":fallback.preparation,"chat_templates":{"named":fallback.chat_templates.named,"entries":fallback.chat_templates.entries.iter().map(|(key,template)|(key.clone(),json!({"source":template.source,"later_system":template.later_system}))).collect::<serde_json::Map<_,_>>()},"model_root":config.model_root,
            "profile":config.profile,"host_reserve":config.host_reserve,"context":engine.context,
            "native_catalog":config.native_catalog,"native_worker":config.native_worker,"native_cache_home":config.native_cache_home})};
        let mut initial=spec.clone();
        if prepare.is_some(){initial.packet["profile"]=Value::Null;}
        let (driver,modalities)=Self::launch(&initial,engine.clone()).await?;
        let preparation=prepare.clone().map(|config|crate::native_prepare::Preparation::start(config,engine.clone()));
        Ok(Self{driver:Mutex::new(driver),engine,spec,serial:Arc::new(Semaphore::new(1)),modalities,fallback,preparation:Mutex::new(preparation),prepare_config:prepare,activated:AtomicBool::new(!config.native_prepare)})
    }
    async fn launch(spec:&Spec,engine:Arc<Engine>)->Result<(Arc<Driver>,Vec<String>)> {
        if let Some(error)=engine.refusal(){return Err(error)}
        let epoch=engine.generation();
        let mut child=crate::sandbox::command(&spec.command[0],crate::sandbox::SERVICE_PROFILE).args(&spec.command[1..])
            .env("JUSTFIT_RESEARCH_NATIVE_PID",engine.owned_pid().to_string()).process_group(0)
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit()).spawn().map_err(|e|failed(e.to_string()))?;
        let writer=child.stdin.take().unwrap();let reader=BufReader::new(child.stdout.take().unwrap());
        let driver=Arc::new(Driver{child:Mutex::new(child),stop_lock:Mutex::new(()),epoch,writer:Mutex::new(writer),reader:Mutex::new(reader),engine:engine.clone(),metrics:Mutex::new(Value::Null),image_cache:Mutex::new(Value::Null),rgb_bytes:AtomicU64::new(0),stopped:AtomicBool::new(false),payload_bytes:AtomicU64::new(0)});
        let weak=Arc::downgrade(&driver);
        engine.on_generation_failure(epoch,move||{if let Some(driver)=weak.upgrade(){driver.stop()}});
        // Dropping a timed-out startup must reap its child even while the
        // blocking reader is still unwinding. It has not admitted a request.
        struct Startup {driver:Arc<Driver>,armed:bool}
        impl Drop for Startup {fn drop(&mut self){if self.armed{self.driver.stop()}}}
        let mut startup=Startup{driver:driver.clone(),armed:true};
        driver.send(spec.packet.clone())?;
        let handle = tokio::runtime::Handle::current();
        let cloned = driver.clone();
        let job = tokio::task::spawn_blocking(move || cloned.transaction(&handle));
        let mut modalities = vec!["text".to_string(), "image".to_string(), "pdf".to_string()];
        match tokio::time::timeout(Duration::from_secs(360), job).await {
            Ok(Ok(Ok((value, blob)))) if value["op"] == "ready" && blob.is_empty() => {
                *driver.metrics.lock().unwrap() = value["metrics"].clone();
                *driver.image_cache.lock().unwrap() = value["image_cache"].clone();
                if let Some(values) = value["input_modalities"].as_array() {
                    modalities = values
                        .iter()
                        .map(|v| {
                            v.as_str()
                                .filter(|s| ["text", "image", "pdf"].contains(s))
                                .map(str::to_string)
                                .ok_or_else(|| failed("invalid media driver modalities"))
                        })
                        .collect::<Result<Vec<_>>>()?;
                }
            }
            other => {
                driver.stop();
                engine.fail_generation(epoch,"vision driver startup failed");
                return Err(failed(format!("vision startup failed: {other:?}")));
            }
        }
        if driver.epoch!=engine.generation()||!engine.healthy(){return Err(engine.refusal().unwrap_or_else(||failed("vision generation changed during startup")))}
        startup.armed=false;
        Ok((driver,modalities))
    }
    async fn current(&self)->Result<Arc<Driver>> {
        let existing=self.driver.lock().unwrap().clone();
        let preparation=self.preparation.lock().unwrap().clone();
        if preparation.as_ref().is_some_and(|p|p.ready())
            && !self.activated.load(Ordering::Acquire)
            && existing.epoch==self.engine.generation() && self.engine.healthy()
            && existing.payload_bytes.load(Ordering::Acquire)==0 && existing.rgb_bytes.load(Ordering::Acquire)==0 {
            let (driver,modalities)=Self::launch(&self.spec,self.engine.clone()).await?;
            if modalities!=self.modalities {driver.stop();return Err(failed("media modalities changed during ANE activation"))}
            existing.stop();
            *self.driver.lock().unwrap()=driver.clone();
            self.activated.store(true,Ordering::Release);
            if let Some(preparation)=&preparation {preparation.activated();}
            return Ok(driver)
        }
        if existing.epoch==self.engine.generation()&&!existing.stopped.load(Ordering::Acquire)&&self.engine.healthy(){return Ok(existing)}
        existing.stop();
        self.fallback.images.clear();
        let mut spec=self.spec.clone();
        let preparing=self.prepare_config.is_some()&&!self.activated.load(Ordering::Acquire);
        if preparing {spec.packet["profile"]=Value::Null;}
        let (driver,modalities)=Self::launch(&spec,self.engine.clone()).await?;
        if modalities!=self.modalities {driver.stop();return Err(failed("media modalities changed after native recovery"))}
        if preparing {
            if let Some(preparation)=preparation {preparation.stop();}
            *self.preparation.lock().unwrap()=self.prepare_config.clone()
                .map(|config|crate::native_prepare::Preparation::start(config,self.engine.clone()));
        }
        *self.driver.lock().unwrap()=driver.clone();Ok(driver)
    }
    pub fn input_modalities(&self) -> Vec<String> {
        self.modalities.clone()
    }
    pub fn snapshot(&self) -> Value {
        let driver=self.driver.lock().unwrap().clone();
        let mut metrics = driver.metrics.lock().unwrap().clone();
        if let Some(preparation)=self.preparation.lock().unwrap().clone() {
            if !metrics.is_object(){metrics=json!({});}
            metrics["native_preparation"]=preparation.snapshot();
        }
        if !metrics.is_null() && metrics.get("shared_payload").is_some() {
            metrics["shared_payload"]["reserved_bytes"] =
                json!(driver.payload_bytes.load(Ordering::Acquire));
        }
        metrics
    }
    pub fn image_snapshot(&self)->Value{
        let driver=self.driver.lock().unwrap().clone();
        if driver.stopped.load(Ordering::Acquire){return self.fallback.images.snapshot(0)}
        let mut value=driver.image_cache.lock().unwrap().clone();
        if value.is_object(){value["request_bytes"]=json!(driver.rgb_bytes.load(Ordering::Acquire));}value
    }
    pub async fn prompt(&self,body:Value,id:u64,render:bool,preparation:Arc<crate::preparation_queue::Guard>)->Result<Value>{
        let began=Instant::now();let timeout=crate::deadline::request(&body)?;
        let permit=crate::deadline::wait(timeout,self.serial.clone().acquire_owned()).await?.map_err(|_|failed("vision driver closed"))?;
        // Prompt-only APIs need no native-engine controls during recovery.
        let driver=self.driver.lock().unwrap().clone();
        if driver.stopped.load(Ordering::Acquire)||!self.engine.healthy(){
            let front=self.fallback.clone();
            let cancellation=crate::documents::Cancellation::new();let flag=cancellation.flag.clone();
            let job=tokio::task::spawn_blocking(move||->Result<Value>{
                let _preparation=preparation;let _permit=permit;
                // Finish owned driver cleanup before allocating a replacement cache.
                driver.stop();
                let check=||{crate::documents::Cancellation::check(&flag)?;crate::deadline::remaining(timeout,began.elapsed()).map(|_|())};
                front.check_model(&body)?;crate::tools::normalize(&body)?;crate::grammar::response_schema(&body)?;
                if render{Ok(json!({"prompt":front.media_render_checked(&body,&check)?}))}
                else{Ok(json!({"input_tokens":front.media_token_count_checked(&body,&check)?}))}
            });
            return crate::deadline::wait(timeout,job).await?.map_err(|_|failed("media fallback worker failed"))?;
        }
        let mut cancel=Cancellation{driver:driver.clone(),id,armed:true};
        let handle=tokio::runtime::Handle::current();
        let job=tokio::task::spawn_blocking(move||{
            let _preparation=preparation;
            let _permit=permit;driver.send(json!({"op":if render{"render"}else{"count"},"id":id,"body":body,"timeout":crate::deadline::json_seconds(remaining_timeout(timeout,began.elapsed())?)}))?;
            let(value,blob)=driver.transaction(&handle)?;
            if value["image_cache"].is_object(){*driver.image_cache.lock().unwrap()=value["image_cache"].clone();}
            if value["op"]=="error"{return Err(ApiError(value["status"].as_u64().unwrap_or(502)as u16,value["message"].as_str().unwrap_or("media preparation failed").into(),value["code"].as_str().unwrap_or("vision_failed").into()))}
            if value["op"]!="prompt"||value["id"]!=id||!blob.is_empty(){return Err(failed("invalid prompt response"))}
            Ok(value["value"].clone())
        });
        let result=crate::deadline::wait(timeout,job).await?.map_err(|_|failed("media prompt worker failed"))?;
        cancel.armed=false;result
    }
    pub async fn prepare(&self, body: Value, id: u64) -> Result<Prepared> {
        let began = Instant::now();
        let timeout = crate::deadline::request(&body)?;
        let _permit = crate::deadline::wait(timeout, self.serial.clone().acquire_owned())
            .await
            .map_err(|_| {
                ApiError(
                    504,
                    "vision queue timed out".into(),
                    "request_timeout".into(),
                )
            })?
            .map_err(|_| failed("vision driver closed"))?;
        let driver=crate::deadline::wait(remaining_timeout(timeout,began.elapsed())?,self.current()).await??;
        let remaining = remaining_timeout(timeout, began.elapsed())?;
        let mut cancel = Cancellation {
            driver: driver.clone(),
            id,
            armed: true,
        };
        let handle = tokio::runtime::Handle::current();
        let job = tokio::task::spawn_blocking(move || {
            let _permit = _permit;
            driver.send(
                json!({"op":"prepare","id":id,"body":body,"timeout":crate::deadline::json_seconds(remaining)}),
            )?;
            let (value, blob) = driver.transaction(&handle)?;
            if value["image_cache"].is_object(){*driver.image_cache.lock().unwrap()=value["image_cache"].clone();}
            if value["op"] == "error" {
                return Err(ApiError(
                    value["status"].as_u64().unwrap_or(502) as u16,
                    value["message"].as_str().unwrap_or("vision failed").into(),
                    value["code"].as_str().unwrap_or("vision_failed").into(),
                ));
            }
            let owner = Owner {
                driver: driver.clone(),
                id,
                bytes: 0,
                rgb: 0,
            };
            if driver.epoch!=driver.engine.generation()||!driver.engine.healthy(){return Err(driver.engine.refusal().unwrap_or_else(||failed("media generation changed")))}
            let frame = native_request(blob, id)?;
            if value["op"] != "prepared"
                || value["id"].as_u64() != Some(id)
                || frame.kind != 1
                || crate::engine::u64_at(&frame.payload, 0)? != id
            {
                return Err(failed("invalid prepared media response"));
            }
            let mut owner = owner;
            owner.rgb=value["rgb_bytes"].as_u64().ok_or_else(||failed("invalid image byte accounting"))?;
            driver.rgb_bytes.fetch_add(owner.rgb,Ordering::AcqRel);
            if value["ane"] == true {
                let tokens = crate::engine::u32_at(&frame.payload, 30)? as usize;
                let spans = crate::engine::u32_at(&frame.payload, 34)? as usize;
                let data_offset = 94 + tokens * 4 + spans * 32 + 4;
                owner.bytes = frame
                    .payload
                    .len()
                    .checked_sub(data_offset)
                    .ok_or_else(|| failed("invalid embedding size"))?
                    as u64
                    * 3;
                driver
                    .payload_bytes
                    .fetch_add(owner.bytes, Ordering::AcqRel);
            }
            let mut metrics = value["metrics"].clone();
            if metrics.is_object() {
                metrics["last_request_id"] = json!(id);
                metrics["last_ane"] = value["ane"].clone();
            }
            *driver.metrics.lock().unwrap() = metrics;
            let stop = serde_json::from_value(value["stop"].clone())
                .map_err(|_| failed("invalid stop list"))?;
            Ok(Prepared {
                received_at: None,
                frame,
                thinking: value["thinking"]
                    .as_bool()
                    .ok_or_else(|| failed("invalid thinking state"))?,
                stop,
                raw: false,
                budget: value["budget"]
                    .as_u64()
                    .ok_or_else(|| failed("invalid output budget"))?
                    as usize,
                timeout,
                public_id: value["public_id"]
                    .as_str()
                    .ok_or_else(|| failed("invalid public ID"))?
                    .into(),
                created_at: value["created_at"]
                    .as_u64()
                    .ok_or_else(|| failed("invalid created time"))?,
                vision_owner: Some(owner),
                constraint: None,
            tools: None,
            response_schema: None,
            internal_start: false,
            prompt_token_ids: None,
            rendered_prompt: None,
            })
        });
        let result = crate::deadline::wait(remaining, job)
            .await
            .map_err(|_| {
                ApiError(
                    504,
                    "vision preparation timed out".into(),
                    "request_timeout".into(),
                )
            })?
            .map_err(|e| failed(e.to_string()))?;
        cancel.armed = false;
        result
    }
}

pub fn has_media(body: &Value) -> bool {
    body["messages"].as_array().is_some_and(|messages| {
        messages.iter().any(|m| {
            m["content"].as_array().is_some_and(|parts| {
                parts.iter().any(|p| {
                    matches!(
                        p["type"].as_str(),
                        Some("image_url" | "input_image" | "file" | "input_file")
                    )
                })
            })
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn expired_positive_timeout_is_504_before_driver_dispatch() {
        assert_eq!(remaining_timeout(Duration::MAX, Duration::from_secs(9001)).unwrap(), Duration::MAX);
        let timeout = Duration::from_micros(1);
        for elapsed in [timeout, Duration::from_micros(2)] {
            let error = remaining_timeout(timeout, elapsed).unwrap_err();
            assert_eq!(error.0, 504);
            assert_eq!(error.2, "request_timeout");
        }
        assert_eq!(remaining_timeout(timeout, Duration::ZERO).unwrap(), timeout);
    }
    #[test]
    fn media_detection_is_explicit_and_packet_bounds_precede_allocation() {
        assert!(has_media(
            &json!({"messages":[{"content":[{"type":"image_url"}]}]})
        ));
        assert!(!has_media(&json!({"messages":[{"content":"image_url"}]})));
        assert!(receive(&mut u32::MAX.to_le_bytes().as_slice()).is_err());
        let mut bytes = vec![];
        packet(&mut bytes, &json!({"op":"cancel","id":1})).unwrap();
        let (value, blob) = receive(&mut bytes.as_slice()).unwrap();
        assert_eq!(value["id"], 1);
        assert!(blob.is_empty());
    }
    #[test]
    fn native_payload_reuses_the_ipc_allocation_and_rejects_wrong_ids() {
        let frame = justfit_native_client_prototype::text_request(7, &[1, 2], 16, 0, 0).unwrap();
        let mut bytes = Vec::new();
        justfit_native_client_prototype::write_frame(&mut bytes, &frame).unwrap();
        let ptr = bytes.as_ptr();
        assert!(native_request(bytes.clone(), 8).is_err());
        let decoded = native_request(bytes, 7).unwrap();
        assert_eq!(decoded.payload.as_ptr(), ptr);
        assert_eq!(decoded, frame);
    }
}
