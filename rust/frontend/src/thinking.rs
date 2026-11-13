//! Python-compatible Fernet hidden-reasoning transport and guarded key files.
use crate::{bad, ApiError, Result};
use base64::Engine;
use fernet::Fernet;
use serde_json::{json, Value};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};
use std::{
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};
const MAX: usize = 16 * 1024 * 1024;
// Match Python's urlsafe_b64decode(validate=False) text acceptance. This only
// decodes framing; the Fernet library still authenticates all decoded bytes.
fn python_base64(text: &str) -> Option<Vec<u8>> {
    if !text.is_ascii() {
        return None;
    }
    let mut output = Vec::with_capacity(text.len() * 3 / 4);
    let mut quad = 0;
    let mut bits = 0u32;
    let mut padding = 0;
    let mut ended = false;
    for ch in text.bytes() {
        if ch == b'=' {
            if quad >= 2 {
                padding += 1;
                if quad + padding >= 4 {
                    ended = true;
                    break;
                }
            }
            continue;
        }
        let value = match ch {
            b'A'..=b'Z' => ch - b'A',
            b'a'..=b'z' => ch - b'a' + 26,
            b'0'..=b'9' => ch - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            _ => continue,
        };
        padding = 0;
        bits = (bits << 6) | value as u32;
        quad += 1;
        match quad {
            2 => output.push((bits >> 4) as u8),
            3 => output.push((bits >> 2) as u8),
            4 => {
                output.push(bits as u8);
                quad = 0;
                bits = 0;
            }
            _ => {}
        }
    }
    if !ended && quad != 0 {
        return None;
    }
    Some(output)
}
pub struct Codec(Fernet);
impl Codec {
    pub fn from_key(key: &str) -> Result<Self> {
        let decoded = python_base64(key).ok_or_else(|| bad("invalid thinking key"))?;
        Fernet::new(&base64::engine::general_purpose::URL_SAFE.encode(decoded))
            .map(Self)
            .ok_or_else(|| bad("invalid thinking key"))
    }
    pub fn load(path: &Path) -> Result<Self> {
        let error = |e: std::io::Error| {
            ApiError(
                500,
                format!("cannot use thinking key: {}", e.kind()),
                "thinking_key_error".into(),
            )
        };
        fn read(path: &Path) -> std::io::Result<String> {
            let source = OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
                .open(path)?;
            let stat = source.metadata()?;
            // SAFETY: getuid takes no pointers and has no preconditions.
            if !stat.is_file()
                || stat.uid() != unsafe { libc::getuid() }
                || ![0o400, 0o600].contains(&(stat.mode() & 0o7777))
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "thinking key must be user-owned regular file,0400 or0600",
                ));
            }
            let mut bytes = Vec::new();
            source.take(45).read_to_end(&mut bytes)?;
            if bytes.len() != 44 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "invalid thinking key length",
                ));
            }
            String::from_utf8(bytes).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "invalid thinking key encoding",
                )
            })
        }
        let key = match read(path) {
            Ok(key) => key,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let parent = path
                    .parent()
                    .ok_or_else(|| bad("thinking key has no parent"))?;
                fs::DirBuilder::new()
                    .recursive(true)
                    .mode(0o700)
                    .create(parent)
                    .map_err(error)?;
                static SERIAL: AtomicU64 = AtomicU64::new(0);
                let mut created = None;
                for _ in 0..100 {
                    let temporary = parent.join(format!(
                        ".thinking-rust-{}-{}",
                        std::process::id(),
                        SERIAL.fetch_add(1, Ordering::Relaxed)
                    ));
                    match OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .mode(0o600)
                        .open(&temporary)
                    {
                        Ok(file) => {
                            created = Some((temporary, file));
                            break;
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                        Err(e) => return Err(error(e)),
                    }
                }
                let (temporary, mut file) = created.ok_or_else(|| {
                    ApiError(
                        500,
                        "cannot allocate thinking key temporary".into(),
                        "thinking_key_error".into(),
                    )
                })?;
                let publish = (|| -> std::io::Result<()> {
                    file.write_all(Fernet::generate_key().as_bytes())?;
                    file.sync_all()?;
                    match fs::hard_link(&temporary, path) {
                        Ok(()) => Ok(()),
                        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
                        Err(e) => Err(e),
                    }
                })();
                let cleanup = fs::remove_file(&temporary);
                publish.map_err(error)?;
                cleanup.map_err(error)?;
                read(path).map_err(error)?
            }
            Err(e) => return Err(error(e)),
        };
        Self::from_key(&key).map_err(|_| {
            ApiError(
                500,
                "invalid thinking key".into(),
                "thinking_key_error".into(),
            )
        })
    }
    pub fn encode(&self, text: &str) -> Result<String> {
        let size = 4 * (57 + 16 * (text.len() / 16 + 1)).div_ceil(3);
        if size > MAX {
            return Err(ApiError(
                500,
                "thinking exceeds the signature size limit".into(),
                "thinking_too_large".into(),
            ));
        }
        Ok(self.0.encrypt(text.as_bytes()))
    }
    pub fn decode(&self, text: &str) -> Result<String> {
        let invalid = || {
            ApiError(400,"invalid signature in thinking block; hidden thinking must be returned to a server using the key that generated it".into(),"invalid_thinking_signature".into())
        };
        if text.len() > MAX || !text.is_ascii() {
            return Err(invalid());
        }
        let canonical = base64::engine::general_purpose::URL_SAFE
            .encode(python_base64(text).ok_or_else(invalid)?);
        let text = canonical.as_str();
        // Python decrypt(token, ttl=None) deliberately checks authentication but
        // not expiry/future skew. fernet 0.2.2's default decrypt always enforces
        // a 60-second future bound. Feed its no-TTL path a synthetic time from
        // the header to disable only that bound; HMAC and padding are still
        // verified by the library before any plaintext is returned.
        let header = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(text.get(..12).ok_or_else(invalid)?)
            .map_err(|_| invalid())?;
        let timestamp = u64::from_be_bytes(
            header
                .get(1..9)
                .ok_or_else(invalid)?
                .try_into()
                .map_err(|_| invalid())?,
        );
        String::from_utf8(
            self.0
                .decrypt_at_time(text, None, timestamp.saturating_sub(60))
                .map_err(|_| invalid())?,
        )
        .map_err(|_| invalid())
    }
}
pub fn fixture(body: &Value) -> Result<Value> {
    let codec = if let Some(path) = body["path"].as_str() {
        Codec::load(Path::new(path))?
    } else {
        Codec::from_key(body["key"].as_str().ok_or_else(|| bad("key required"))?)?
    };
    match body["action"].as_str() {
        Some("encode") => Ok(json!(
            codec.encode(body["text"].as_str().ok_or_else(|| bad("text required"))?)?
        )),
        Some("decode") => Ok(json!(codec.decode(
            body["signature"]
                .as_str()
                .ok_or_else(|| bad("signature required"))?
        )?)),
        _ => Err(bad("codec action required")),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tampering_wrong_key_and_invalid_utf8_are_refused() {
        let a = Codec::from_key(&Fernet::generate_key()).unwrap();
        let b = Codec::from_key(&Fernet::generate_key()).unwrap();
        let token = a.encode("你好🌱").unwrap();
        assert_eq!(a.decode(&token).unwrap(), "你好🌱");
        assert_eq!(
            b.decode(&token).unwrap_err().2,
            "invalid_thinking_signature"
        );
        let mut broken = token.into_bytes();
        broken[20] = if broken[20] == b'A' { b'B' } else { b'A' };
        assert!(a.decode(std::str::from_utf8(&broken).unwrap()).is_err());
        assert!(a.decode(&a.0.encrypt(&[0xff])).is_err());
    }
}
