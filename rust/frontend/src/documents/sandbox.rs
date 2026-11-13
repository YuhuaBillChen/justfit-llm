//! Native PDF worker launch policy: a deny-default profile of its own,
//! applied where crate::sandbox can apply one.
use super::{unavailable, Config};
use crate::Result;
use std::{collections::BTreeSet, path::Path, process::Command};

// Arguments are sandbox parameters, never interpolated into policy source.
// sysctl reads stay open: the Rust runtime sizes its stack guard page from
// hw.pagesize at start-up and aborts when that read is refused.
const PROFILE: &str = r#"(version 1)
(deny default)
(allow sysctl-read)
(allow file-read*
  (literal "/")
  (subpath "/System")
  (subpath "/usr/lib")
  (subpath "/Library/Fonts")
  (literal (param "WORKER"))
  (literal (param "PDFIUM"))
  (literal (param "ZLIB")))
(allow process-exec (literal (param "WORKER")))
(deny process-info*)
(allow process-info* (target self))
"#;

fn plan(sandbox: &Path, worker: &Path, config: &Config) -> Result<Command> {
    let worker = worker.canonicalize().map_err(|_| unavailable("PDF worker path unavailable"))?;
    let pdfium = config.library.canonicalize().map_err(|_| unavailable("PDFium path unavailable"))?;
    // macOS system zlib lives in the dyld cache and need not exist on disk.
    let zlib = if config.zlib == Path::new("/usr/lib/libz.dylib") {
        config.zlib.clone()
    } else {
        config.zlib.canonicalize().map_err(|_| unavailable("zlib path unavailable"))?
    };
    let mut command = Command::new(sandbox);
    command.env_clear().current_dir("/");
    for (name, path) in [("WORKER", &worker), ("PDFIUM", &pdfium), ("ZLIB", &zlib)] {
        let mut value = std::ffi::OsString::from(format!("{name}="));
        value.push(path);
        command.arg("-D").arg(value);
    }
    let mut profile = PROFILE.to_owned();
    // Canonical files can live under a user directory. Permit traversal
    // metadata on their ancestors without granting those directories' data.
    let parents: BTreeSet<_> = [&worker, &pdfium, &zlib].into_iter()
        .flat_map(|p| p.ancestors().skip(1)).collect();
    for (index, path) in parents.iter().enumerate() {
        let name = format!("PARENT{index}");
        let mut value = std::ffi::OsString::from(format!("{name}="));
        value.push(path);
        command.arg("-D").arg(value);
        profile.push_str(&format!("(allow file-read-metadata (literal (param \"{name}\")))\n"));
    }
    command.args(["-p", &profile, "--"]).arg(worker);
    Ok(command)
}

pub(super) fn command(worker: &Path, config: &Config) -> Result<Command> {
    if !cfg!(target_os = "macos") || !Path::new("/usr/bin/sandbox-exec").is_file() {
        return Err(unavailable("PDF sandbox unavailable"));
    }
    // The host already sandboxes the service (crate::sandbox): the worker
    // keeps that sandbox and the cleared environment only.
    if !crate::sandbox::available() {
        let worker = worker.canonicalize().map_err(|_| unavailable("PDF worker path unavailable"))?;
        let mut command = Command::new(worker);
        command.env_clear().current_dir("/");
        return Ok(command);
    }
    plan(Path::new("/usr/bin/sandbox-exec"), worker, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, os::unix::fs::PermissionsExt, sync::atomic::{AtomicU64, Ordering}};
    static NEXT: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn unsupported_platform_refuses_before_launch() {
        if cfg!(target_os = "macos") { return; }
        let error = command(Path::new("/must-not-start"), &Config::default()).unwrap_err();
        assert_eq!((error.0, error.2.as_str()), (503, "document_unavailable"));
    }

    #[test]
    fn command_does_not_inherit_parent_environment_and_paths_are_parameters() {
        // This shim observes argv/env only; it does NOT emulate a Mac sandbox.
        let root = std::env::temp_dir().join(format!("justfit-pdf-policy-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        fs::create_dir(&root).unwrap();
        let shim = root.join("sandbox observer");
        fs::write(&shim, "#!/bin/sh\nexec /usr/bin/env\n").unwrap();
        fs::set_permissions(&shim, fs::Permissions::from_mode(0o700)).unwrap();
        let library = root.join("library with spaces and \"quotes\".dylib");
        fs::write(&library, b"fixture").unwrap();
        let config = Config { library, zlib: "/usr/lib/libz.dylib".into() };
        let mut command = plan(&shim, Path::new("/bin/true"), &config).unwrap();
        let args: Vec<_> = command.get_args().map(|s| s.to_string_lossy().into_owned()).collect();
        assert!(args.iter().any(|s| s == &format!("PDFIUM={}", config.library.display())));
        assert!(args.iter().any(|s| s == "ZLIB=/usr/lib/libz.dylib"));
        assert!(args.iter().any(|s| s.starts_with(PROFILE) && s.contains("file-read-metadata")));
        let output = command.output().unwrap();
        assert!(output.status.success());
        let environment = String::from_utf8(output.stdout).unwrap();
        assert!(!environment.lines().any(|s| s.starts_with("HOME=") || s.starts_with("PATH=") || s.starts_with("JUSTFIT_")));
        fs::remove_dir_all(root).unwrap();
    }
}
