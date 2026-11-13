//! Reuse the qualified physical/neural accounting ABI and original ceilings.
use crate::{bad, coreai, Result};
use serde_json::json;
use std::{
    fs,
    io::Write,
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};

pub struct Meter {
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}
impl Meter {
    pub fn start(library: &Path) -> Result<Self> {
        let (own, pid) = coreai::meter_functions(library)?;
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let folder = std::env::var_os("ANE_METER_DIR").map(std::path::PathBuf::from);
        let mut log = if let Some(path) = &folder {
            Some(
                fs::OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .open(path.join(format!("worker-memory-{}.jsonl", std::process::id())))
                    .map_err(|_| bad("worker memory log unavailable"))?,
            )
        } else {
            None
        };
        let native_pid = std::env::var("JUSTFIT_RESEARCH_NATIVE_PID")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(0);
        let thread = thread::spawn(move || {
            loop {
                let mut ledger = [0u64; 7];
                let mut rusage = [0u64; 4];
                // SAFETY: the process-lived ABI writes exactly7/4 uint64 values; these
                // arrays have matching size and no other thread accesses their storage.
                let ok = unsafe { own(ledger.as_mut_ptr()) } != 0;
                let pid_ok = unsafe { pid(std::process::id() as i32, rusage.as_mut_ptr()) } != 0;
                if let Some(log) = log.as_mut() {
                    let row = json!({"pid":std::process::id(),"time":std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs_f64(),"clock":"unix","ok":ok,"pid_ok":pid_ok,"ledger":{
     "phys_footprint":ledger[0],"graphics_footprint":ledger[1],"graphics_nofootprint":ledger[2],"neural_footprint":ledger[3],"neural_nofootprint":ledger[4],"neural_nofootprint_peak":ledger[5],"neural_nofootprint_total":ledger[6]},"rusage":rusage});
                    if writeln!(log, "{row}").and_then(|_| log.flush()).is_err() {
                        eprintln!("ANE memory log write failed");
                        std::process::exit(70)
                    }
                }
                if !ok || !pid_ok {
                    eprintln!("ANE memory attribution failed closed");
                    std::process::exit(70)
                }
                let bound = ledger[0].max(rusage[3]) + ledger[4].max(ledger[5]).max(rusage[2]);
                let mut native = [0u64; 4];
                let native_bytes =
                    if native_pid > 0 && unsafe { pid(native_pid, native.as_mut_ptr()) } != 0 {
                        native[0]
                    } else {
                        0
                    };
                if bound > 1127 * 1024 * 1024
                    || native_bytes + ledger[0] + ledger[4] > 22015 * 1024 * 1024
                {
                    let row = json!({"tripped":true,"worker_conservative_bound":bound,"native_physical":native_bytes,"worker_physical":ledger[0],"worker_neural_nonfootprint":ledger[4],"native_pid":native_pid});
                    if let Some(folder) = &folder {
                        let _ = fs::write(
                            folder.join(format!("worker-guard-{}.json", std::process::id())),
                            row.to_string(),
                        );
                    }
                    eprintln!("ANE_WORKER_GUARD {row}");
                    std::process::exit(70)
                }
                if flag.load(Ordering::Acquire) {
                    break;
                }
                thread::sleep(Duration::from_millis(50));
            }
        });
        Ok(Self {
            stop,
            thread: Some(thread),
        })
    }
}
impl Drop for Meter {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
