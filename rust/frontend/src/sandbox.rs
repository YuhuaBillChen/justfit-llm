//! One sandbox scheme for the native processes the service starts. The
//! engine, the media driver, the vision worker and the ANE delivery run under
//! SERVICE_PROFILE; the PDF worker under its own stricter policy
//! (documents::sandbox). Applied here, from Rust, so a plain `justfit serve`
//! has it too; a host that already sandboxes the service (macOS refuses a
//! nested sandbox_apply: sandbox-exec exits 71) keeps that outer sandbox, and
//! the children run in it as before. Probed once per process.
use std::{ffi::OsStr, path::Path, process::Command, sync::OnceLock};

/// What the service's children may do: everything the service may, except
/// talk out. Metal, the Neural Engine and CoreML go through XPC, not sockets.
pub const SERVICE_PROFILE: &str = "(version 1)\n(allow default)\n(deny network-outbound)\n";
const SANDBOX_EXEC: &str = "/usr/bin/sandbox-exec";

/// Whether this process can apply a sandbox profile to a child.
pub fn available() -> bool {
    static STATE: OnceLock<bool> = OnceLock::new();
    *STATE.get_or_init(|| {
        if !cfg!(target_os = "macos") || !Path::new(SANDBOX_EXEC).is_file() {
            return false;
        }
        let refused = Command::new(SANDBOX_EXEC)
            .args(["-p", "(version 1)(allow default)", "--", "/usr/bin/true"])
            .env_clear().current_dir("/")
            .output().map(|o| o.status.code() == Some(71)).unwrap_or(false);
        eprintln!("JUSTFIT_SANDBOX {}", if refused {
            "the host already sandboxes this service; its children keep that sandbox"
        } else {
            "service profile applied to the engine and the native workers"
        });
        !refused
    })
}

/// A command for `program` under `profile`, or `program` itself where no
/// second sandbox can be applied.
pub fn command(program: impl AsRef<OsStr>, profile: &str) -> Command {
    if available() { plan(Path::new(SANDBOX_EXEC), program, profile) } else { Command::new(program) }
}

/// The sandboxed command, through `sandbox`; tests pass a shim.
pub fn plan(sandbox: &Path, program: impl AsRef<OsStr>, profile: &str) -> Command {
    let mut command = Command::new(sandbox);
    command.args(["-p", profile, "--"]).arg(program);
    command
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, os::unix::fs::PermissionsExt};

    #[test]
    fn plan_passes_the_profile_and_program_as_arguments() {
        let root = std::env::temp_dir().join(format!("justfit-sandbox-plan-{}", std::process::id()));
        fs::create_dir(&root).unwrap();
        let shim = root.join("sandbox shim");
        fs::write(&shim, "#!/bin/sh\nfor a in \"$@\"; do printf '%s\\0' \"$a\"; done\n").unwrap();
        fs::set_permissions(&shim, fs::Permissions::from_mode(0o700)).unwrap();
        let output = plan(&shim, "/bin/true", SERVICE_PROFILE).output().unwrap();
        let args: Vec<_> = String::from_utf8(output.stdout).unwrap().split('\0').filter(|s| !s.is_empty()).map(str::to_owned).collect();
        assert_eq!(args, ["-p", SERVICE_PROFILE, "--", "/bin/true"].map(|s| s.to_owned()));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn without_a_host_sandbox_the_program_runs_as_is() {
        if cfg!(target_os = "macos") { return; }
        assert!(!available());
        assert_eq!(command("/bin/true", SERVICE_PROFILE).get_program(), OsStr::new("/bin/true"));
    }
}
