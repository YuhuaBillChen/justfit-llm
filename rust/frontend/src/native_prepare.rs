//! Owned, governed native cache preparation. HTTP starts on GPU while this runs.
use crate::{ane_native::{self, Profile}, bad, engine::Engine, Result};
use serde_json::{json, Value};
use std::{fs, os::fd::AsRawFd, os::unix::{fs::OpenOptionsExt, process::CommandExt},
    path::{Path, PathBuf}, process::{Child,Command,Stdio},
    sync::{Arc, Mutex, atomic::{AtomicBool, AtomicU64, Ordering}}, time::Duration};

const CACHE_ID: &str = "org.justfit.vision.aaa0bdc9ed86239e";
const MEMORY: u64 = 1127 * 1024 * 1024;
const LEASE: u64 = 1 << 62;
static ATTEMPT: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
pub struct Config {
    pub profile: PathBuf,
    pub worker: PathBuf,
    pub catalog: PathBuf,
    pub cache_home: PathBuf,
    pub model_root: PathBuf,
    pub distribution: Option<crate::ane_distribution::Delivery>,
}

pub struct Preparation {
    child: Mutex<Option<Child>>,
    stopped: AtomicBool,
    ready: AtomicBool,
    status: Mutex<Value>,
}

pub fn validate(config: &Config, path: &Path) -> Result<()> {
    let top = ane_native::read_json(&config.profile)?;
    let record = ane_native::read_json(path)?;
    if top["schema"] != 3 || top["kind"] != "tiered-vision"
        || record["schema"] != 1 || record["host"] != CACHE_ID
        || record["worker_sha256"] != ane_native::sha(&config.worker)?
        || record["tiered_profile_sha256"] != ane_native::sha(&config.profile)? {
        return Err(bad("native prepared catalog identity mismatch"));
    }
    let root = config.profile.parent().ok_or_else(|| bad("profile parent missing"))?;
    for tier in top["tiers"].as_array().filter(|v| !v.is_empty() && v.len() <= 4)
        .ok_or_else(|| bad("native tiers missing"))? {
        let leaf = root.join(tier["profile"].as_str().ok_or_else(|| bad("tier path missing"))?);
        if tier["sha256"] != ane_native::sha(&leaf)? {
            return Err(bad("native prepared tier identity mismatch"));
        }
        let profile = Profile::open_for_preparation(&leaf)?;
        let phases=&profile.data["phase_qualification"];
        if profile.data["route_policy"]!="gpu_relief" || !phases["prefill"].is_boolean()
            || !phases["decode"].is_boolean()
            || !profile.data["worker_memory_bytes"].as_u64().is_some_and(|v|(512*1024*1024..=MEMORY).contains(&v))
            || !profile.data["worker_timeout_seconds"].as_f64().is_some_and(|v|v.is_finite()&&(10.0..=600.0).contains(&v))
            || profile.data["vision_weights_sha256"]!=crate::vision_identity::sha(&config.worker.ancestors().nth(4).ok_or_else(||bad("native worker bundle layout invalid"))?.join("justfit-engine"),&config.model_root).map_err(|e|bad(e.to_string()))? {
            return Err(bad("native prepared policy/weights invalid"));
        }
        let row = &record["tiers"][profile.capacity.to_string()];
        if row["profile_sha256"] != ane_native::sha(&leaf)?
            || row["library_sha256"] != ane_native::sha(&profile.library)?
            || record["build"] != profile.catalog["build"] {
            return Err(bad("native prepared assets/OS mismatch"));
        }
        let cache=config.cache_home.join("Library/Application Support/coreai-cache")
            .join(record["build"].as_str().ok_or_else(||bad("native OS missing"))?).join(CACHE_ID);
        for (index,group) in row["groups"].as_array().filter(|v| v.len() == 9)
            .ok_or_else(|| bad("native prepared groups incomplete"))?.iter().enumerate() {
            if group["asset_hash"]!=profile.catalog["groups"][index]["asset_hash"] {
                return Err(bad("native prepared group identity mismatch"));
            }
            let expected=cache.join(group["asset_hash"].as_str().ok_or_else(||bad("native asset ID missing"))?)
                .canonicalize().map_err(|_|bad("native private cache missing"))?;
            let guard=Path::new(group["guard"]["path"].as_str().ok_or_else(||bad("native placement path missing"))?)
                .canonicalize().map_err(|_|bad("native placement path unavailable"))?;
            if !guard.starts_with(&expected) {return Err(bad("native placement belongs to another cache home"))}
            for file in group["files"].as_array().ok_or_else(||bad("native cache files missing"))? {
                let file=Path::new(file[0].as_str().ok_or_else(||bad("native cached file path invalid"))?)
                    .canonicalize().map_err(|_|bad("native cached file unavailable"))?;
                if !file.starts_with(&expected) {return Err(bad("native file belongs to another cache home"))}
            }
            ane_native::check_native_group(group)?;
        }
    }
    Ok(())
}

// Called only with the preparation lock held. Keep the failed catalog for
// diagnosis; a matching producer identity authorizes rebuilding its metadata,
// never deleting or rewriting any compiled program or an unrelated catalog.
fn reuse_or_retire_catalog(config:&Config, verify:impl FnOnce()->Result<()>)->Result<bool> {
    use std::io::Read;
    use std::os::unix::fs::MetadataExt;
    let input=match fs::OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW|libc::O_NONBLOCK).open(&config.catalog) {
        Ok(file)=>file,
        Err(error) if error.kind()==std::io::ErrorKind::NotFound=>return Ok(false),
        Err(_)=>return Err(bad("native catalog is unavailable or unsafe")),
    };
    let before=input.metadata().map_err(|_|bad("native catalog metadata unavailable"))?;
    if !before.is_file()||before.len()>8*1024*1024{return Err(bad("native catalog is not a bounded regular file"));}
    if verify().is_ok(){return Ok(true);}
    let mut bytes=vec![];input.take(8*1024*1024+1).read_to_end(&mut bytes).map_err(|_|bad("native catalog read failed"))?;
    if bytes.len()>8*1024*1024{return Err(bad("native catalog exceeds repair limit"));}
    let record:Value=serde_json::from_slice(&bytes).map_err(|_|bad("unrecognized native catalog; preserving original"))?;
    if record["schema"]!=1||record["host"]!=CACHE_ID||record["worker_sha256"]!=ane_native::sha(&config.worker)?
        ||record["tiered_profile_sha256"]!=ane_native::sha(&config.profile)? {
        return Err(bad("foreign native catalog; preserving original"));
    }
    let same=|meta:&fs::Metadata|meta.is_file()&&meta.dev()==before.dev()&&meta.ino()==before.ino()
        &&meta.len()==before.len()&&meta.mtime()==before.mtime()&&meta.mtime_nsec()==before.mtime_nsec();
    let current=fs::symlink_metadata(&config.catalog).map_err(|_|bad("native catalog changed during repair"))?;
    if !same(&current){return Err(bad("native catalog changed during repair"));}
    let backup=config.catalog.with_extension(format!("stale-{}-{}.json",std::process::id(),ATTEMPT.fetch_add(1,Ordering::Relaxed)));
    fs::hard_link(&config.catalog,&backup).map_err(|_|bad("native catalog quarantine unavailable"))?;
    // Check the exclusive backup before removing only the matching original.
    if !same(&fs::symlink_metadata(&backup).map_err(|_|bad("native catalog backup unavailable"))?)
        ||!same(&fs::symlink_metadata(&config.catalog).map_err(|_|bad("native catalog changed during repair"))?) {
        return Err(bad("native catalog changed during repair; backup preserved"));
    }
    fs::remove_file(&config.catalog).map_err(|_|bad("native stale catalog retirement failed; backup preserved"))?;
    Ok(false)
}

impl Preparation {
    pub fn start(config: Config, engine: Arc<Engine>) -> Arc<Self> {
        let task = Arc::new(Self { child: Mutex::new(None), stopped: AtomicBool::new(false),
            ready: AtomicBool::new(false), status: Mutex::new(json!({"state":"preparing","serving":"gpu"})) });
        let epoch = engine.generation();
        let weak = Arc::downgrade(&task);
        engine.on_generation_failure(epoch, move || {if let Some(task) = weak.upgrade() {task.stop()}});
        let running = task.clone();
        tokio::spawn(async move {
            let result = running.run(&config, &engine, epoch).await;
            if let Err(error) = result {
                *running.status.lock().unwrap() = json!({"state":"failed","serving":"gpu","error":error.1});
            } else if !running.stopped.load(Ordering::Acquire) {
                running.ready.store(true, Ordering::Release);
                *running.status.lock().unwrap() = json!({"state":"ready","serving":"gpu","activation":"next_idle_media_request"});
            }
        });
        task
    }
    pub fn ready(&self) -> bool {self.ready.load(Ordering::Acquire)}
    pub fn snapshot(&self) -> Value {self.status.lock().unwrap().clone()}
    pub fn activated(&self) {*self.status.lock().unwrap() = json!({"state":"ready","serving":"gpu_relief_ane"});}
    pub fn stop(&self) {
        self.stopped.store(true, Ordering::Release);
        self.reap_compiler();
    }
    fn reap_compiler(&self) {
        if let Some(mut child) = self.child.lock().unwrap().take() {
            // This group contains only the compiler we spawned. Reap it before
            // releasing its governor lease or stopping the owning generation.
            // Avoid platform-dependent parsing of a negative CLI argument.
            // Never signal a shared/inherited group, even if a future caller
            // forgets process_group(0); the exact child remains the fallback.
            if let Ok(pid) = libc::pid_t::try_from(child.id()) {
                if pid > 0 && unsafe { libc::getpgid(pid) } == pid {
                    let _ = unsafe { libc::killpg(pid, libc::SIGKILL) };
                }
            }
            let _ = child.kill();
            let _ = child.wait();
        }
    }
    async fn deliver(&self,delivery:&crate::ane_distribution::Delivery,config:&Config,engine:&Arc<Engine>,epoch:u64)->Result<()>{
        if config.profile.is_file(){return Ok(());}
        let path=config.catalog.with_extension(format!("delivery-{}-{}.json",std::process::id(),ATTEMPT.fetch_add(1,Ordering::Relaxed)));
        let mut file=fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&path).map_err(|_|bad("ANE delivery configuration cannot be created"))?;
        use std::io::Write;file.write_all(&serde_json::to_vec(delivery).map_err(|_|bad("ANE delivery configuration cannot be encoded"))?).map_err(|_|bad("ANE delivery configuration write failed"))?;file.sync_all().map_err(|_|bad("ANE delivery configuration sync failed"))?;
        let log=path.with_extension("log");let errors=fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&log).map_err(|_|bad("ANE delivery log unavailable"))?;
        *self.status.lock().unwrap()=json!({"state":"preparing","phase":"assets","serving":"gpu"});
        // ANE delivery downloads its pinned assets over HTTPS: it is not given the
        // service profile, which denies outbound network.
        let child=Command::new(std::env::current_exe().map_err(|_|bad("Rust delivery executable unavailable"))?).arg("ane-bootstrap").arg(&path).process_group(0).stdin(Stdio::null()).stdout(Stdio::null()).stderr(errors).spawn().map_err(|_|bad("Rust ANE delivery could not start"))?;
        *self.child.lock().unwrap()=Some(child);
        loop{
            if self.stopped.load(Ordering::Acquire)||engine.generation()!=epoch{self.stop();return Err(bad("ANE delivery cancelled"));}
            if log.metadata().map(|m|m.len()>8*1024*1024).unwrap_or(false){self.stop();return Err(bad("ANE delivery exceeded diagnostic output limit"));}
            let exit=self.child.lock().unwrap().as_mut().map(|child|child.try_wait()).transpose().map_err(|_|bad("ANE delivery wait failed"))?.flatten();
            if let Some(exit)=exit{self.child.lock().unwrap().take();if !exit.success(){return Err(bad(format!("ANE assets unavailable ({exit}); GPU service remains active; see {}",log.display())));}break;}
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        if !config.profile.is_file(){return Err(bad("ANE delivery produced no qualified graph profile"));}Ok(())
    }
    async fn run(&self, config: &Config, engine: &Arc<Engine>, epoch: u64) -> Result<()> {
        let parent = config.catalog.parent().ok_or_else(|| bad("catalog parent missing"))?;
        fs::create_dir_all(parent).map_err(|_| bad("native catalog directory unavailable"))?;
        let lock = fs::OpenOptions::new().read(true).write(true).create(true).truncate(false)
            .mode(0o600).custom_flags(libc::O_NOFOLLOW).open(config.catalog.with_extension("lock"))
            .map_err(|_| bad("native preparation lock unavailable"))?;
        loop {
            if self.stopped.load(Ordering::Acquire) {return Err(bad("native preparation stopped"))}
            // SAFETY: the live file owns this descriptor; flock does not retain pointers.
            if unsafe {libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB)} == 0 {break}
            let error = std::io::Error::last_os_error();
            if !matches!(error.raw_os_error(), Some(libc::EWOULDBLOCK)) {return Err(bad("native preparation lock failed"))}
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        // File identity cannot prove that the OS can still load the underlying
        // ANE specialization. Recheck hardware in the owned worker under lease.
        // Wait for an idle admission point. Once admitted, the compiler has
        // the same hard physical/neural ceiling as the governed worker; its
        // meter aborts on excess instead of exceeding its reservation.
        let mut leased = false;
        let result = async {
            loop {
                if self.stopped.load(Ordering::Acquire) || engine.generation() != epoch {
                    return Err(bad("native preparation generation stopped"));
                }
                let status = engine.status().await?;
                if status["ready"] == true && ["prefilling", "decoding", "queued"].iter()
                    .all(|key| status["scheduler"][key].as_u64() == Some(0)) {
                    leased = true; // Reconcile even a timed-out or malformed acknowledgment.
                    let answer = engine.control_generation(epoch, LEASE, MEMORY, Duration::from_secs(30)).await?;
                    if answer["external_memory_lease"]["id"].as_u64() != Some(LEASE) {
                        return Err(bad("invalid compiler lease acknowledgment"));
                    }
                    if answer["external_memory_lease"]["granted"] == true {break}
                    let _ = engine.control_generation(epoch, LEASE, 0, Duration::from_secs(30)).await?;
                    leased = false;
                }
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
            if let Some(delivery)=&config.distribution{
                self.deliver(delivery,config,engine,epoch).await?;
            }
            // Refuse foreign metadata before touching hardware. A valid owned
            // catalog still needs the hardware recheck below.
            reuse_or_retire_catalog(config, ||validate(config, &config.catalog))?;
            *self.status.lock().unwrap()=json!({"state":"preparing","phase":"compile","serving":"gpu"});
            let pending = config.catalog.with_extension(format!("pending-{}-{}.json", std::process::id(), ATTEMPT.fetch_add(1, Ordering::Relaxed)));
            let log = pending.with_extension("log");
            let errors = fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&log)
                .map_err(|_| bad("native preparation log already exists or unavailable"))?;
            if pending.exists() {return Err(bad("native pending catalog already exists"))}
            let child = crate::sandbox::command(&config.worker, crate::sandbox::SERVICE_PROFILE).arg("prepare-cache").arg(&config.profile).arg(&pending)
                .env("JUSTFIT_NATIVE_CACHE_ID", CACHE_ID).env("CFFIXED_USER_HOME", &config.cache_home)
                .env("JUSTFIT_ANE_CACHE_REPAIR", "1")
                .env("JUSTFIT_RESEARCH_NATIVE_PID", engine.owned_pid().to_string())
                .process_group(0).stdin(Stdio::null()).stdout(Stdio::null()).stderr(errors).spawn()
                .map_err(|_| bad("native compiler could not start"))?;
            {
                let mut owned = self.child.lock().unwrap();
                *owned = Some(child);
            }
            loop {
                if self.stopped.load(Ordering::Acquire) || engine.generation() != epoch {
                    self.stop(); return Err(bad("native compilation cancelled"));
                }
                if log.metadata().map(|m| m.len() > 8 * 1024 * 1024).unwrap_or(false) {
                    self.stop(); return Err(bad("native compiler exceeded diagnostic output limit"));
                }
                let exit = self.child.lock().unwrap().as_mut().map(|child| child.try_wait())
                    .transpose().map_err(|_| bad("native compiler wait failed"))?.flatten();
                if let Some(exit) = exit {
                    self.child.lock().unwrap().take();
                    if !exit.success() {return Err(bad(format!("native compiler failed ({exit}); see {}", log.display())))}
                    break;
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            validate(config, &pending)?;
            // Keep the previous producer-bound metadata for diagnostics, then
            // publish the record of this successful hardware recheck.
            reuse_or_retire_catalog(config, ||Err(bad("refresh after hardware recheck")))?;
            // Same-filesystem exclusive publication; never replace an existing catalog.
            fs::hard_link(&pending, &config.catalog).map_err(|_| bad("native ready catalog publication failed"))?;
            Ok(())
        }.await;
        // Every error path must reap a still-live compiler before returning
        // its reservation, including OS wait/inspection failures.
        self.reap_compiler();
        if leased {
            let release = engine.control_generation(epoch, LEASE, 0, Duration::from_secs(30)).await;
            if !matches!(release, Ok(ref status) if status["external_memory_lease"]["id"].as_u64()==Some(LEASE) && status["external_memory_lease"]["granted"]==true) {
                engine.fail_generation(epoch, "compiler external memory release failed");
                return Err(bad("compiler external memory release failed"));
            }
        }
        result
    }
}

impl Drop for Preparation {fn drop(&mut self) {self.stop()}}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;
    struct CatalogFixture {root:PathBuf,config:Config}
    impl CatalogFixture {
        fn new()->Self {
            let root=std::env::temp_dir().join(format!("justfit-catalog-repair-{}-{}",std::process::id(),ATTEMPT.fetch_add(1,Ordering::Relaxed)));
            fs::create_dir(&root).unwrap();let config=Config{profile:root.join("profile.json"),worker:root.join("worker"),catalog:root.join("catalog.json"),cache_home:root.join("cache-home"),model_root:root.join("model"),distribution:None};
            fs::write(&config.profile,b"fixture profile").unwrap();fs::write(&config.worker,b"fixture worker").unwrap();
            fs::create_dir(&config.cache_home).unwrap();fs::write(config.cache_home.join("program.bin"),b"preserve compiled bytes").unwrap();Self{root,config}
        }
        fn owned(&self)->Vec<u8>{serde_json::to_vec(&json!({"schema":1,"host":CACHE_ID,"worker_sha256":ane_native::sha(&self.config.worker).unwrap(),"tiered_profile_sha256":ane_native::sha(&self.config.profile).unwrap(),"tiers":{"4096":{"stale":"missing guard"}}})).unwrap()}
    }
    impl Drop for CatalogFixture{fn drop(&mut self){fs::remove_dir_all(&self.root).unwrap();}}
    #[test]
    fn stale_owned_catalog_is_preserved_for_reprepare_without_touching_programs(){
        let fixture=CatalogFixture::new();let bytes=fixture.owned();fs::write(&fixture.config.catalog,&bytes).unwrap();
        assert!(!reuse_or_retire_catalog(&fixture.config,||Err(bad("missing compiled guard"))).unwrap());assert!(!fixture.config.catalog.exists());
        let backups=fs::read_dir(&fixture.root).unwrap().map(|e|e.unwrap().path()).filter(|p|p.file_name().unwrap().to_string_lossy().contains("stale-")).collect::<Vec<_>>();assert_eq!(backups.len(),1);assert_eq!(fs::read(&backups[0]).unwrap(),bytes);assert_eq!(fs::read(fixture.config.cache_home.join("program.bin")).unwrap(),b"preserve compiled bytes");
    }
    #[test]
    fn ready_catalog_is_reused_and_foreign_or_symlink_catalog_is_preserved(){
        let fixture=CatalogFixture::new();assert!(!reuse_or_retire_catalog(&fixture.config,||panic!("absent catalog is not verified")).unwrap());let bytes=fixture.owned();fs::write(&fixture.config.catalog,&bytes).unwrap();assert!(reuse_or_retire_catalog(&fixture.config,||Ok(())).unwrap());assert_eq!(fs::read(&fixture.config.catalog).unwrap(),bytes);
        for bytes in [b"owner document".to_vec(),serde_json::to_vec(&json!({"schema":1,"host":"foreign"})).unwrap()]{fs::write(&fixture.config.catalog,&bytes).unwrap();assert!(reuse_or_retire_catalog(&fixture.config,||Err(bad("invalid"))).is_err());assert_eq!(fs::read(&fixture.config.catalog).unwrap(),bytes);}
        fs::rename(&fixture.config.catalog,fixture.root.join("owner")).unwrap();std::os::unix::fs::symlink(fixture.root.join("owner"),&fixture.config.catalog).unwrap();assert!(reuse_or_retire_catalog(&fixture.config,||Ok(())).is_err());assert!(fixture.config.catalog.is_symlink());
    }
    struct Owned(Child);
    impl Drop for Owned {fn drop(&mut self) {let _=self.0.kill();let _=self.0.wait();}}
    #[test]
    fn stopping_preparation_reaps_its_group_and_preserves_other_children() {
        let mut sentinel=Owned(Command::new("/bin/sleep").arg("30").spawn().unwrap());
        let child=Command::new("/bin/sleep").arg("30").process_group(0).spawn().unwrap();
        let task=Preparation{child:Mutex::new(Some(child)),stopped:AtomicBool::new(false),
            ready:AtomicBool::new(false),status:Mutex::new(json!({"state":"preparing"}))};
        let started=std::time::Instant::now();
        task.stop();
        assert!(started.elapsed()<Duration::from_secs(2),"stop waited for the child to exit naturally");
        assert!(task.child.lock().unwrap().is_none());
        assert!(task.stopped.load(Ordering::Acquire));
        assert!(sentinel.0.try_wait().unwrap().is_none());
        task.stop(); // Idempotence must not signal a reused PID.
        assert!(sentinel.0.try_wait().unwrap().is_none());
    }
    #[test]
    fn stopping_preparation_with_inherited_group_only_kills_its_child() {
        let mut sentinel=Owned(Command::new("/bin/sleep").arg("30").spawn().unwrap());
        let child=Command::new("/bin/sleep").arg("30").spawn().unwrap();
        let task=Preparation{child:Mutex::new(Some(child)),stopped:AtomicBool::new(false),
            ready:AtomicBool::new(false),status:Mutex::new(json!({"state":"preparing"}))};
        let started=std::time::Instant::now();task.stop();
        assert!(started.elapsed()<Duration::from_secs(2));
        assert!(sentinel.0.try_wait().unwrap().is_none());
        assert!(task.child.lock().unwrap().is_none());
    }
    #[test]
    fn stopping_preparation_closes_descendants_in_its_own_group() {
        use std::io::Read;
        let mut child=Command::new("/bin/sh").args(["-c","/bin/sleep 30 & printf r; wait"])
            .process_group(0).stdout(Stdio::piped()).spawn().unwrap();
        let group=child.id() as libc::pid_t;
        let mut output=child.stdout.take().unwrap();let mut ready=[0];output.read_exact(&mut ready).unwrap();
        assert_eq!(ready,[b'r']);
        let task=Preparation{child:Mutex::new(Some(child)),stopped:AtomicBool::new(false),
            ready:AtomicBool::new(false),status:Mutex::new(json!({"state":"preparing"}))};
        let (send,recv)=std::sync::mpsc::channel();
        std::thread::spawn(move||{let result=output.read_to_end(&mut Vec::new());let _=send.send(result);});
        let started=std::time::Instant::now();task.stop();
        let eof=recv.recv_timeout(Duration::from_secs(2));
        if eof.is_err() {let _=unsafe{libc::killpg(group,libc::SIGKILL)};}
        assert!(eof.unwrap().is_ok(),"a compiler descendant kept its pipe open");
        assert!(started.elapsed()<Duration::from_secs(2));
    }
}
