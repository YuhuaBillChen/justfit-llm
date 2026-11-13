//! Python-free media request preparation and governed ANE orchestration.
//! This private driver has no HTTP listener or native-engine child. Native
//! status/lease controls are relayed to the same Rust-owned engine generation.
use crate::{
    ane_native::{self, Profile},
    bad,
    engine::u32_at,
    generation::Prepared,
    ApiError, Frontend, Result,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    io::{Read, Write},
    os::unix::fs::DirBuilderExt,
    path::{Path, PathBuf},
    process::{Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

fn send(writer: &Mutex<std::io::Stdout>, value: Value, blob: &[u8]) -> Result<()> {
    let data = serde_json::to_vec(&value).map_err(|_| bad("driver JSON encoding failed"))?;
    if data.len() > 1024 * 1024 || blob.len() > 256 * 1024 * 1024 + 16 {
        return Err(bad("driver packet too large"));
    }
    let mut out = writer.lock().unwrap();
    out.write_all(&(data.len() as u32).to_le_bytes())
        .and_then(|_| out.write_all(&data))
        .and_then(|_| out.write_all(&(blob.len() as u32).to_le_bytes()))
        .and_then(|_| out.write_all(blob))
        .and_then(|_| out.flush())
        .map_err(|_| bad("Rust host disconnected"))
}
fn receive(reader: &mut impl Read) -> Result<Value> {
    let mut len = [0u8; 4];
    reader
        .read_exact(&mut len)
        .map_err(|_| bad("Rust host disconnected"))?;
    let n = u32::from_le_bytes(len) as usize;
    if !(2..=128 * 1024 * 1024).contains(&n) {
        return Err(bad("invalid driver request length"));
    }
    let mut data = vec![0; n];
    reader
        .read_exact(&mut data)
        .map_err(|_| bad("truncated driver JSON"))?;
    reader
        .read_exact(&mut len)
        .map_err(|_| bad("missing driver binary length"))?;
    if u32::from_le_bytes(len) != 0 {
        return Err(bad("driver input binary must be empty"));
    }
    serde_json::from_slice(&data).map_err(|_| bad("invalid driver JSON"))
}
struct Owner {
    rgb: usize,
    embedding: usize,
    tier: Option<usize>,
}
struct State {
    cancel: HashMap<u64, Arc<AtomicBool>>,
    pending: HashSet<u64>,
    owners: HashMap<u64, Owner>,
    metrics: Value,
}
impl State {
    fn snapshot(&self) -> Value {
        let mut metrics = self.metrics.clone();
        let reserved = self.owners.values().map(|o| o.embedding).sum::<usize>();
        metrics["shared_payload"]["reserved_bytes"] = json!(reserved);
        for (cap, entry) in metrics.as_object_mut().unwrap().iter_mut() {
            if let Ok(cap) = cap.parse::<usize>() {
                entry["payload_reserved_bytes"] = json!(self
                    .owners
                    .values()
                    .filter(|o| o.tier == Some(cap))
                    .map(|o| o.embedding)
                    .sum::<usize>())
            }
        }
        metrics
    }
    fn count(&mut self, capacity: usize, key: &str) {
        let value = &mut self.metrics[capacity.to_string()][key];
        *value = json!(value.as_u64().unwrap_or(0) + 1);
    }
}
struct Runtime {
    writer: Arc<Mutex<std::io::Stdout>>,
    answers: Receiver<Value>,
    closed: Arc<AtomicBool>,
    next_lease: u64,
}
impl Runtime {
    fn control(&self, lease: u64, size: u64, timeout: f64) -> Result<Value> {
        send(
            &self.writer,
            json!({"op":"native","lease":lease,"size":size,"timeout":timeout}),
            &[],
        )?;
        let result = self
            .answers
            .recv_timeout(Duration::from_secs_f64(timeout + 5.))
            .map_err(|_| {
                ApiError(
                    504,
                    "native vision control timed out".into(),
                    "request_timeout".into(),
                )
            })?;
        if let Some(error) = result["error"].as_str() {
            return Err(ApiError(504, error.into(), "request_timeout".into()));
        }
        Ok(result["status"].clone())
    }
    fn status(&self) -> Result<Value> {
        self.control(0, 0, 0.5)
    }
    fn release(&self, id: u64) -> Result<()> {
        let mut last = None;
        for timeout in [2., 30.] {
            match self.control(id, 0, timeout) {
                Ok(result) => {
                    let answer = &result["external_memory_lease"];
                    if answer["id"].as_u64() == Some(id) && answer["granted"] == true {
                        return Ok(());
                    }
                    last = Some(bad("invalid native release acknowledgment"));
                    break;
                }
                Err(e) => last = Some(e),
            }
        }
        self.closed.store(true, Ordering::Release);
        let _ = send(
            &self.writer,
            json!({"op":"fatal","message":"external memory release failed"}),
            &[],
        );
        Err(last.unwrap_or_else(|| bad("external release failed")))
    }
    fn acquire(&mut self, bytes: u64, timeout: f64) -> Result<Option<u64>> {
        let id = self.next_lease;
        self.next_lease = self
            .next_lease
            .checked_add(1)
            .ok_or_else(|| bad("lease IDs exhausted"))?;
        match self.control(id, bytes, timeout) {
            Ok(status) => {
                let answer = &status["external_memory_lease"];
                if answer["id"].as_u64() != Some(id) || !answer["granted"].is_boolean() {
                    self.release(id)?;
                    return Err(bad("invalid native lease acknowledgment"));
                }
                Ok((answer["granted"] == true).then_some(id))
            }
            Err(e) => {
                self.release(id)?;
                Err(e)
            }
        }
    }
}
struct Tier {
    path: PathBuf,
    memory: u64,
    timeout: Duration,
    prefill: bool,
    decode: bool,
}
fn metrics() -> Value {
    json!({"encoded":0,"fallback":0,"cancelled":0,"failed":0,"rejected":0,"worker_starts":0,"worker_reuses":0,"worker_stops":0,"routed_idle":0,"routed_unavailable":0,"memory_unavailable":0,"queued_images":0,"payload_reserved_bytes":0,"last_timings":{}})
}
pub fn relief(status: &Value, images: usize, prefill: bool, decode: bool) -> &'static str {
    if status["ready"] != true
        || ["prefilling", "decoding", "queued"]
            .iter()
            .any(|k| status["scheduler"][k].as_u64().is_none())
    {
        return "unavailable";
    }
    if images != 1 {
        return "idle";
    }
    let ffn = status["ane_ffn"]["state"].as_str().unwrap_or("");
    if !["off", "split", "stopped"].contains(&ffn) {
        return "unavailable";
    }
    if status["scheduler"]["prefilling"].as_u64().unwrap_or(0) > 0 {
        return if ffn == "off" && prefill {
            "ane"
        } else {
            "idle"
        };
    }
    if status["scheduler"]["decoding"].as_u64().unwrap_or(0) > 0 && decode {
        return "ane";
    }
    "idle"
}
fn admitted(status: &Value, reserve: u64) -> bool {
    let g = &status["memory_governor"];
    status["ready"] == true
        && matches!(g["system_pressure"].as_str(), Some("normal" | "warning"))
        && g["growth_allowed"] == true
        && g["host_measurement_valid"] == true
        && g["host_reserve_bytes"].as_u64().unwrap_or(0) >= reserve
        && g["host_available_bytes"].as_u64().unwrap_or(0) >= reserve
}
fn retry(status: &Value, required: u64) -> bool {
    let g = &status["memory_governor"];
    if status["ready"] != true || g["system_pressure"] == "critical" {
        return false;
    }
    let reserved = g["reserved_bytes"].as_u64().or_else(|| {
        Some(
            g["limit_bytes"]
                .as_u64()?
                .saturating_sub(g["charged_bytes"].as_u64()? + g["headroom_bytes"].as_u64()?),
        )
    });
    let Some(reserved) = reserved else {
        return false;
    };
    if reserved > 0 {
        return true;
    }
    let Some(mut available) = g["headroom_bytes"].as_u64() else {
        return false;
    };
    let memory = &status["memory_plan"]["model"]["memory"];
    if status["phase_residency"]["prefill_arena_resident"] == true {
        available = available.saturating_add(memory["shared_prefill_bytes"].as_u64().unwrap_or(0));
    }
    available = available.saturating_add(
        status["kv"]["reclaimable_backing_bytes"]
            .as_u64()
            .or_else(|| status["kv"]["reclaimable_bytes"].as_u64())
            .unwrap_or(0),
    );
    available = available.saturating_add(
        status["state"]["warm_idle_cells"]
            .as_u64()
            .unwrap_or(0)
            .saturating_mul(memory["active_state_cell_bytes"].as_u64().unwrap_or(0)),
    );
    available >= required
}
struct Temporary(PathBuf);
impl Temporary {
    fn create() -> Result<Self> {
        let mut entropy = [0u8; 16];
        fs::File::open("/dev/urandom")
            .and_then(|mut f| f.read_exact(&mut entropy))
            .map_err(|_| bad("temporary entropy unavailable"))?;
        let path = std::env::temp_dir().join(format!(
            "justfit-native-vision-{}-{}",
            std::process::id(),
            entropy
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        ));
        fs::DirBuilder::new()
            .mode(0o700)
            .create(&path)
            .map_err(|_| bad("private vision directory unavailable"))?;
        Ok(Self(path))
    }
}
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn checkpoint(cancel: &AtomicBool, closed: &AtomicBool, deadline: Option<Instant>) -> Result<()> {
    if cancel.load(Ordering::Acquire) || closed.load(Ordering::Acquire) {
        return Err(ApiError(
            499,
            "request cancelled".into(),
            "request_cancelled".into(),
        ));
    }
    if deadline.is_some_and(|end| Instant::now() >= end) {
        return Err(ApiError(
            504,
            "request timed out".into(),
            "request_timeout".into(),
        ));
    }
    Ok(())
}

struct Driver {
    front: Frontend,
    tiers: BTreeMap<usize, Tier>,
    gpu_only: Vec<usize>,
    catalog: Option<PathBuf>,
    worker: Option<PathBuf>,
    cache_id: String,
    cache_home: Option<PathBuf>,
    reserve: u64,
    state: Arc<Mutex<State>>,
    runtime: Runtime,
}
impl Driver {
    fn initialize(config: &Value, state: Arc<Mutex<State>>, mut runtime: Runtime) -> Result<Self> {
        let mut front = Frontend::load_with_templates(
            Path::new(
                config["tokenizer"]
                    .as_str()
                    .ok_or_else(|| bad("tokenizer path missing"))?,
            ),
            config["model"]
                .as_str()
                .ok_or_else(|| bad("model missing"))?
                .into(),
            config["context"]
                .as_u64()
                .ok_or_else(|| bad("context missing"))? as u32,
            config.get("chat_templates").filter(|value|value.is_object()).map(|value|serde_json::from_value(value.clone()).map_err(|_|bad("invalid driver templates"))).transpose()?,
        )?;

        let aliases: Vec<String> = serde_json::from_value(config.get("served_model_names").cloned().unwrap_or_else(||json!([])))
            .map_err(|_|bad("invalid served_model_names"))?;
        front.configure_names(&aliases, false)?;
        if let Some(config)=config.get("preparation") {
            front.preparation=serde_json::from_value(config.clone()).map_err(|_|bad("invalid preparation options"))?;
            front.preparation.validate()?;
        }
        if let Some(config)=config.get("documents") {
            let config=serde_json::from_value(config.clone()).map_err(|_|bad("invalid document config"))?;
            front.documents=Arc::new(crate::documents::Documents::new(config));
        }
        let reserve = config["host_reserve"]
            .as_u64()
            .filter(|n| *n > 0)
            .ok_or_else(|| bad("explicit host reserve missing"))?;
        let mut tiers = BTreeMap::new();
        let mut gpu_only = vec![];
        let mut catalog = None;
        let mut worker = None;
        let mut cache_id = String::new();
        if let Some(top) = config["profile"].as_str() {
            let path = Path::new(top);
            let top = ane_native::read_json(path)?;
            if top["schema"] != 3 || top["kind"] != "tiered-vision" {
                return Err(bad("native driver requires qualified tiered profile"));
            }
            let cat = PathBuf::from(
                config["native_catalog"]
                    .as_str()
                    .ok_or_else(|| bad("native vision cache catalog missing"))?,
            );
            let record = ane_native::read_json(&cat)?;
            let executable = PathBuf::from(
                config["native_worker"]
                    .as_str()
                    .ok_or_else(|| bad("native worker missing"))?,
            );
            if record["schema"] != 1
                || record["worker_sha256"] != ane_native::sha(&executable)?
                || record["tiered_profile_sha256"] != ane_native::sha(path)?
            {
                return Err(bad("native cache worker/profile identity mismatch"));
            }
            cache_id = record["host"]
                .as_str()
                .filter(|s| s.starts_with("org.justfit.vision."))
                .ok_or_else(|| bad("native cache namespace invalid"))?
                .into();
            for tier in top["tiers"]
                .as_array()
                .filter(|t| !t.is_empty() && t.len() <= 4)
                .ok_or_else(|| bad("invalid vision tiers"))?
            {
                let leaf = path
                    .parent()
                    .ok_or_else(|| bad("profile parent missing"))?
                    .join(
                        tier["profile"]
                            .as_str()
                            .ok_or_else(|| bad("leaf profile missing"))?,
                    );
                if ane_native::sha(&leaf)? != tier["sha256"].as_str().unwrap_or("") {
                    return Err(bad("leaf profile digest mismatch"));
                }
                let profile = Profile::open(&leaf)?;
                let native = &record["tiers"][profile.capacity.to_string()];
                if native["profile_sha256"] != ane_native::sha(&leaf)?
                    || native["library_sha256"] != ane_native::sha(&profile.library)?
                    || record["build"] != profile.catalog["build"]
                {
                    return Err(bad("native cache asset/OS mismatch"));
                }
                for row in native["groups"]
                    .as_array()
                    .filter(|a| a.len() == 9)
                    .ok_or_else(|| bad("native cache groups incomplete"))?
                {
                    ane_native::check_native_group(row)?
                }
                let model_root = Path::new(
                    config["model_root"]
                        .as_str()
                        .ok_or_else(|| bad("model root missing"))?,
                );
                let engine = executable.ancestors().nth(4)
                    .ok_or_else(|| bad("native worker bundle layout invalid"))?
                    .join("justfit-engine");
                if crate::vision_identity::sha(&engine, model_root)
                    .map_err(|error| bad(error.to_string()))?
                    != profile.data["vision_weights_sha256"].as_str().unwrap_or("")
                {
                    return Err(bad("vision weights do not match profile"));
                }
                let memory = profile.data["worker_memory_bytes"]
                    .as_u64()
                    .filter(|n| (512 * 1024 * 1024..=1127 * 1024 * 1024).contains(n))
                    .ok_or_else(|| bad("invalid native worker memory budget"))?;
                let timeout = profile.data["worker_timeout_seconds"]
                    .as_f64()
                    .filter(|v| v.is_finite() && (10.0..=600.0).contains(v))
                    .ok_or_else(|| bad("invalid worker timeout"))?;
                let phases = &profile.data["phase_qualification"];
                if profile.data["route_policy"] != "gpu_relief"
                    || !phases["prefill"].is_boolean()
                    || !phases["decode"].is_boolean()
                {
                    return Err(bad("native driver requires pinned gpu_relief policy"));
                }
                let capacity = profile.capacity;
                let prefill = phases["prefill"] == true;
                let decode = phases["decode"] == true;
                if tiers
                    .insert(
                        capacity,
                        Tier {
                            path: leaf,
                            memory,
                            timeout: Duration::from_secs_f64(timeout),
                            prefill,
                            decode,
                        },
                    )
                    .is_some()
                {
                    return Err(bad("duplicate native tier"));
                }
                state.lock().unwrap().metrics[capacity.to_string()] = metrics();
            }
            gpu_only = serde_json::from_value(top["gpu_only_capacities"].clone())
                .map_err(|_| bad("invalid GPU-only tiers"))?;
            if gpu_only
                .iter()
                .any(|n| !matches!(n, 12288 | 14400 | 16384) || tiers.contains_key(n))
            {
                return Err(bad("invalid or ambiguous GPU-only tiers"));
            }
            catalog = Some(cat);
            worker = Some(executable);
        }
        // The released startup prewarm asks the native governor for an
        // external lease before serving. This is also a real phase boundary:
        // the native engine can release unused warmup scratch safely.
        // Cached native placement was already checked above; no worker is
        // started here and a denial leaves ordinary governed GPU fallback.
        for (capacity,tier) in &tiers {
            match runtime.acquire(tier.memory,30.) {
                Ok(Some(id))=>runtime.release(id)?,
                Ok(None)=>state.lock().unwrap().count(*capacity,"memory_unavailable"),
                Err(error)=>return Err(error),
            }
        }
        state.lock().unwrap().metrics["gpu_only_capacities"] = json!(gpu_only);
        Ok(Self {
            front,
            tiers,
            gpu_only,
            catalog,
            worker,
            cache_id,
            cache_home: config["native_cache_home"].as_str().map(PathBuf::from),
            reserve,
            state,
            runtime,
        })
    }
    fn prepare(
        &mut self,
        body: &Value,
        id: u64,
        cancel: &AtomicBool,
        deadline: Option<Instant>,
    ) -> Result<Prepared> {
        checkpoint(cancel, &self.runtime.closed, deadline)?;
        let mut prepared = self.front.media_generation_checked(body, id, &||checkpoint(cancel,&self.runtime.closed,deadline))?;
        checkpoint(cancel, &self.runtime.closed, deadline)?;
        let count = u32_at(&prepared.frame.payload, 34)? as usize;
        let prompt = u32_at(&prepared.frame.payload, 30)? as usize;
        let spans_offset = 94 + prompt * 4;
        let pixel_offset = spans_offset + count * 32;
        let rgb_bytes = prepared
            .frame
            .payload
            .len()
            .checked_sub(pixel_offset)
            .ok_or_else(|| bad("invalid native image frame"))?;
        let state = self.state.lock().unwrap();
        let used = state.owners.values().map(|o| o.rgb).sum::<usize>();
        if rgb_bytes > 256 * 1024 * 1024 - used.min(256 * 1024 * 1024) {
            return Err(ApiError(
                503,
                "in-flight image memory budget is full".into(),
                "frontend_overloaded".into(),
            ));
        }
        drop(state);
        let mut required = 0;
        let mut grids = Vec::new();
        for i in 0..count {
            let h = u32_at(&prepared.frame.payload, spans_offset + i * 32 + 8)? as usize;
            let w = u32_at(&prepared.frame.payload, spans_offset + i * 32 + 12)? as usize;
            required = required.max(h * w);
            grids.push((h, w));
        }
        let chosen = self
            .tiers
            .keys()
            .copied()
            .chain(self.gpu_only.iter().copied())
            .filter(|n| *n >= required)
            .min();
        let mut embedding = 0;
        let mut owner_tier = None;
        if let Some(capacity) = chosen.filter(|c| self.tiers.contains_key(c)) {
            let tier = &self.tiers[&capacity];
            let status = self.runtime.status().unwrap_or(Value::Null);
            let route = relief(&status, count, tier.prefill, tier.decode);
            if route != "ane" || grids.iter().any(|(h, w)| *h > 128 || *w > 128) {
                self.state.lock().unwrap().count(
                    capacity,
                    if route == "unavailable" {
                        "routed_unavailable"
                    } else {
                        "routed_idle"
                    },
                );
            } else if !admitted(&status, self.reserve) {
                self.state.lock().unwrap().count(capacity, "fallback");
            } else {
                let cap = Instant::now() + Duration::from_secs(60);
                let until = deadline.map_or(cap, |end| end.min(cap));
                let mut lease = None;
                while Instant::now() < until {
                    checkpoint(cancel, &self.runtime.closed, deadline)?;
                    // Keep the governed acquire alive until the request's bounded
                    // admission deadline. Cancelling it every 500ms can erase
                    // native reclaim demand before a long GPU command drains.
                    let remaining=until.saturating_duration_since(Instant::now());
                    if remaining.is_zero(){break;}
                    match self.runtime.acquire(tier.memory,remaining.as_secs_f64().min(30.)) {
                        Ok(Some(id)) => {
                            lease = Some(id);
                            break;
                        }
                        Ok(None) => {}
                        Err(e) => {
                            if self.runtime.closed.load(Ordering::Acquire) {
                                return Err(e);
                            }
                        }
                    }
                    if !retry(&self.runtime.status().unwrap_or(Value::Null), tier.memory) {
                        self.state
                            .lock()
                            .unwrap()
                            .count(capacity, "memory_unavailable");
                        break;
                    }
                    thread::sleep(Duration::from_millis(50));
                }
                if let Some(lease) = lease {
                    let status = self.runtime.status().unwrap_or(Value::Null);
                    let still_ane = relief(&status, count, tier.prefill, tier.decode) == "ane";
                    let charge = grids
                        .iter()
                        .map(|(h, w)| h * w / 4 * 5120 * 2 * 3)
                        .sum::<usize>();
                    let used = self
                        .state
                        .lock()
                        .unwrap()
                        .owners
                        .values()
                        .map(|o| o.embedding)
                        .sum::<usize>();
                    let result =
                        if still_ane && charge <= 512 * 1024 * 1024 - used.min(512 * 1024 * 1024) {
                            self.run_worker(
                                capacity,
                                &prepared.frame.payload[pixel_offset..],
                                &grids,
                                cancel,
                                deadline,
                            )
                        } else {
                            Err(bad("ANE route or payload capacity changed while waiting"))
                        };
                    self.runtime.release(lease)?;
                    checkpoint(cancel, &self.runtime.closed, deadline)?;
                    match result {
                        Ok((rows, worker)) => {
                            let leaf = &self.tiers[&capacity].path;
                            let profile_hash = Sha256::digest(
                                fs::read(leaf).map_err(|_| bad("profile disappeared"))?,
                            );
                            let mut payload = prepared.frame.payload[..pixel_offset].to_vec();
                            let flags = u32_at(&payload, 82)? | 4;
                            payload[82..86].copy_from_slice(&flags.to_le_bytes());
                            let mut offset = 0;
                            for (i, (h, w)) in grids.iter().enumerate() {
                                let size = h * w / 4 * 5120 * 2;
                                let mut hash = Sha256::new();
                                hash.update(b"prepared-vision-bf16-v1\0");
                                hash.update(profile_hash);
                                for v in [*h as u32, *w as u32, 5120] {
                                    hash.update(v.to_le_bytes())
                                }
                                hash.update(&rows[offset..offset + size]);
                                let digest = hash.finalize();
                                payload[spans_offset + i * 32 + 16..spans_offset + i * 32 + 32]
                                    .copy_from_slice(&digest[..16]);
                                offset += size;
                            }
                            payload.extend(5120u32.to_le_bytes());
                            payload.extend(rows);
                            prepared.frame.payload = payload;
                            embedding = charge;
                            owner_tier = Some(capacity);
                            let mut state = self.state.lock().unwrap();
                            state.count(capacity, "encoded");
                            state.metrics[capacity.to_string()]["last_timings"] =
                                json!({"tier_capacity":capacity,"worker":worker});
                        }
                        Err(e) => {
                            let mut state = self.state.lock().unwrap();
                            state.count(capacity, "failed");
                            state.count(capacity, "fallback");
                            eprintln!("NATIVE_VISION_FALLBACK {}", e.1);
                        }
                    }
                } else {
                    self.state.lock().unwrap().count(capacity, "fallback");
                }
            }
        }
        checkpoint(cancel, &self.runtime.closed, deadline)?;
        self.state.lock().unwrap().owners.insert(
            id,
            Owner {
                rgb: rgb_bytes,
                embedding,
                tier: owner_tier,
            },
        );
        Ok(prepared)
    }
    fn run_worker(
        &self,
        capacity: usize,
        rgb: &[u8],
        grids: &[(usize, usize)],
        cancel: &AtomicBool,
        deadline: Option<Instant>,
    ) -> Result<(Vec<u8>, Value)> {
        let tier = &self.tiers[&capacity];
        let temp = Temporary::create()?;
        let source = temp.0.join("pixels.rgb");
        let target = temp.0.join("rows.bf16");
        fs::write(&source, rgb).map_err(|_| bad("RGB write failed"))?;
        let stdout = temp.0.join("worker.json");
        let errors = temp.0.join("worker.log");
        let mut child = crate::sandbox::command(
            self.worker
                .as_ref()
                .ok_or_else(|| bad("native worker unavailable"))?,
            crate::sandbox::SERVICE_PROFILE,
        )
        .args([
            "encode",
            tier.path
                .to_str()
                .ok_or_else(|| bad("invalid profile path"))?,
            source.to_str().unwrap(),
            target.to_str().unwrap(),
            &serde_json::to_string(grids).map_err(|_| bad("invalid grids"))?,
        ])
        .env(
            "JUSTFIT_NATIVE_VISION_CATALOG",
            self.catalog
                .as_ref()
                .ok_or_else(|| bad("native catalog missing"))?,
        )
        .env("JUSTFIT_NATIVE_CACHE_ID", &self.cache_id)
        .envs(self.cache_home.as_ref().map(|home|("CFFIXED_USER_HOME",home)))
        .stdin(Stdio::null())
        .stdout(fs::File::create(&stdout).map_err(|_| bad("worker output unavailable"))?)
        .stderr(fs::File::create(&errors).map_err(|_| bad("worker log unavailable"))?)
        .spawn()
        .map_err(|_| bad("native worker failed to start"))?;
        let expected = grids
            .iter()
            .map(|(h, w)| h * w / 4 * 5120 * 2)
            .sum::<usize>();
        let cap = Instant::now() + tier.timeout;
        let until = deadline.map_or(cap, |end| end.min(cap));
        let result = (|| {
            loop {
                checkpoint(cancel, &self.runtime.closed, Some(until))?;
                if target
                    .metadata()
                    .map(|m| m.len() > expected as u64)
                    .unwrap_or(false)
                    || stdout
                        .metadata()
                        .map(|m| m.len() > 1024 * 1024)
                        .unwrap_or(false)
                    || errors
                        .metadata()
                        .map(|m| m.len() > 8 * 1024 * 1024)
                        .unwrap_or(false)
                {
                    return Err(bad("native worker exceeded output limit"));
                }
                if let Some(exit) = child
                    .try_wait()
                    .map_err(|_| bad("native worker wait failed"))?
                {
                    if !exit.success() {
                        let detail = fs::read_to_string(&errors).unwrap_or_default();
                        return Err(bad(format!(
                            "native worker failed: {}",
                            detail.chars().take(2048).collect::<String>()
                        )));
                    }
                    break;
                }
                thread::sleep(Duration::from_millis(50));
            }
            if target.metadata().map(|m| m.len()).unwrap_or(0) != expected as u64 {
                return Err(bad("native worker returned incorrect row count"));
            }
            let rows = fs::read(&target).map_err(|_| bad("native embeddings unavailable"))?;
            if rows
                .as_chunks::<2>()
                .0
                .iter()
                .any(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7f80 == 0x7f80)
            {
                return Err(bad("native worker returned non-finite rows"));
            }
            Ok((rows, ane_native::read_json(&stdout)?))
        })();
        if child.try_wait().ok().flatten().is_none() {
            let _ = child.kill();
        }
        let _ = child.wait();
        result
    }
}

pub fn run() -> Result<()> {
    let mut input = std::io::stdin();
    let config = receive(&mut input)?;
    let writer = Arc::new(Mutex::new(std::io::stdout()));
    let closed = Arc::new(AtomicBool::new(false));
    let state = Arc::new(Mutex::new(State {
        cancel: HashMap::new(),
        pending: HashSet::new(),
        owners: HashMap::new(),
        metrics: json!({"shared_payload":{"limit_bytes":512*1024*1024,"reserved_bytes":0}}),
    }));
    let (jobs_tx, jobs) = mpsc::sync_channel::<Value>(2);
    let (answers_tx, answers) = mpsc::sync_channel::<Value>(2);
    let shared = state.clone();
    let done = closed.clone();
    thread::spawn(move || {
        let result = (|| -> Result<()> {
            loop {
                let packet = receive(&mut input)?;
                let id = packet["id"].as_u64().unwrap_or(0);
                let mut state = shared.lock().unwrap();
                match packet["op"].as_str() {
                    Some("native_result") => answers_tx
                        .try_send(packet)
                        .map_err(|_| bad("unexpected native reply"))?,
                    Some("cancel") => {
                        if state.cancel.len() >= 64 && !state.cancel.contains_key(&id) {
                            return Err(bad("excessive cancelled requests"));
                        }
                        state
                            .cancel
                            .entry(id)
                            .or_insert_with(|| Arc::new(AtomicBool::new(false)))
                            .store(true, Ordering::Release);
                    }
                    Some("release") => {
                        state.owners.remove(&id);
                    }
                    Some("prepare"|"render"|"count") => {
                        if id == 0
                            || state.pending.contains(&id)
                            || state.owners.contains_key(&id)
                            || state.owners.len() >= 32
                        {
                            return Err(bad("duplicate/excessive media request"));
                        }
                        state.pending.insert(id);
                        state
                            .cancel
                            .entry(id)
                            .or_insert_with(|| Arc::new(AtomicBool::new(false)));
                        jobs_tx
                            .try_send(packet)
                            .map_err(|_| bad("media queue overflow"))?;
                    }
                    _ => return Err(bad("unknown driver operation")),
                }
            }
        })();
        if let Err(e) = result {
            eprintln!("NATIVE_DRIVER_INPUT_CLOSED {}", e.1)
        }
        done.store(true, Ordering::Release);
        for event in shared.lock().unwrap().cancel.values() {
            event.store(true, Ordering::Release);
        }
    });
    let runtime = Runtime {
        writer: writer.clone(),
        answers,
        closed: closed.clone(),
        next_lease: 1 << 63,
    };
    let mut driver = Driver::initialize(&config, state.clone(), runtime)?;
    send(
        &writer,
        json!({"op":"ready","image_cache":driver.front.images.snapshot(0),"ane":!driver.tiers.is_empty(),"runtime_python":false,"input_modalities":if driver.front.documents.available(){vec!["text","image","pdf"]}else{vec!["text","image"]},"metrics":state.lock().unwrap().snapshot()}),
        &[],
    )?;
    while !closed.load(Ordering::Acquire) {
        let packet = match jobs.recv_timeout(Duration::from_millis(100)) {
            Ok(p) => p,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(_) => break,
        };
        let id = packet["id"]
            .as_u64()
            .ok_or_else(|| bad("request id missing"))?;
        let event = state.lock().unwrap().cancel[&id].clone();
        let timeout=crate::deadline::duration(packet.get("timeout"))?;
        if matches!(packet["op"].as_str(),Some("render"|"count")){
            let deadline=crate::deadline::mono_deadline(timeout);
            let check=||checkpoint(&event,&driver.runtime.closed,deadline);
            let body=&packet["body"];
            let result=(||->Result<Value>{
                driver.front.check_model(body)?;crate::tools::normalize(body)?;crate::grammar::response_schema(body)?;
                if packet["op"]=="render"{Ok(json!({"prompt":driver.front.media_render_checked(body,&check)?}))}
                else{Ok(json!({"input_tokens":driver.front.media_token_count_checked(body,&check)?}))}
            })();
            let cache=driver.front.images.snapshot(0);
            let metadata=match result{Ok(value)=>json!({"op":"prompt","id":id,"value":value,"image_cache":cache}),Err(error)=>json!({"op":"error","status":error.0,"message":error.1,"code":error.2,"image_cache":cache})};
            send(&writer,metadata,&[])?;
            let mut state=state.lock().unwrap();state.cancel.remove(&id);state.pending.remove(&id);continue
        }
        let result = driver.prepare(
            &packet["body"],id,&event,crate::deadline::mono_deadline(timeout),
        );
        if closed.load(Ordering::Acquire) {
            break;
        }
        match result {
            Ok(prepared) => {
                let mut blob = Vec::new();
                justfit_native_client_prototype::write_frame(&mut blob, &prepared.frame)
                    .map_err(|_| bad("native frame serialization failed"))?;
                let image_cache=driver.front.images.snapshot(0);
                let rgb_bytes=state.lock().unwrap().owners.get(&id).map_or(0,|owner|owner.rgb);
                let metadata = json!({"op":"prepared","id":id,"image_cache":image_cache,"rgb_bytes":rgb_bytes,"thinking":prepared.thinking,"stop":prepared.stop,"budget":prepared.budget,"public_id":prepared.public_id,"created_at":prepared.created_at,"ane":u32_at(&prepared.frame.payload,82)?&4!=0,"runtime_python":false,"metrics":state.lock().unwrap().snapshot()});
                send(&writer, metadata, &blob)?;
            }
            Err(e) => {
                state.lock().unwrap().owners.remove(&id);
                send(
                    &writer,
                    json!({"op":"error","status":e.0,"message":e.1,"code":e.2,"image_cache":driver.front.images.snapshot(0)}),
                    &[],
                )?;
            }
        }
        let mut state = state.lock().unwrap();
        state.cancel.remove(&id);
        state.pending.remove(&id);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unlimited_preparation_still_obeys_cancellation_and_worker_checkpoints() {
        let cancel = AtomicBool::new(false);
        let closed = AtomicBool::new(false);
        assert!(checkpoint(&cancel, &closed, None).is_ok());
        let error = checkpoint(&cancel, &closed, Some(Instant::now() - Duration::from_secs(1))).unwrap_err();
        assert_eq!(error.0, 504);
        cancel.store(true, Ordering::Release);
        assert_eq!(checkpoint(&cancel, &closed, None).unwrap_err().0, 499);
        cancel.store(false, Ordering::Release);
        closed.store(true, Ordering::Release);
        assert_eq!(checkpoint(&cancel, &closed, None).unwrap_err().0, 499);
    }
    #[test]
    fn relief_obeys_phase_and_actual_lane_state() {
        let mut status = json!({"ready":true,"scheduler":{"prefilling":0,"decoding":2,"queued":0},"ane_ffn":{"state":"off"}});
        assert_eq!(relief(&status, 1, true, true), "ane");
        assert_eq!(relief(&status, 2, true, true), "idle");
        status["scheduler"]["prefilling"] = json!(1);
        status["ane_ffn"]["state"] = json!("split");
        assert_eq!(relief(&status, 1, true, true), "idle");
        status["scheduler"]["queued"] = Value::Null;
        assert_eq!(relief(&status, 1, true, true), "unavailable");
    }
}
