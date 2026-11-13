//! The 2.3 Host, Origin and CORS contract, including non-HTTP webview origins.
use crate::{authority, guard, ApiError, Result};
use axum::http::HeaderMap;

const METHODS: &str = "GET, HEAD, POST, DELETE, OPTIONS";
type Origin = (String, String, Option<u16>);

fn forbidden(message: impl Into<String>) -> ApiError {
    ApiError(403, message.into(), "forbidden".into())
}

fn origin(value: &str) -> Result<Origin> {
    let (scheme, rest) = value
        .split_once("://")
        .ok_or_else(|| forbidden("invalid Origin header"))?;
    if value.contains('*')
        || !scheme
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphabetic)
        || !scheme
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"+.-".contains(&b))
    {
        return Err(forbidden("invalid Origin header"));
    }
    let (host, port) = authority(rest).map_err(|_| forbidden("invalid Origin header"))?;
    let scheme = scheme.to_ascii_lowercase();
    let default = match scheme.as_str() {
        "http" => Some(80),
        "https" => Some(443),
        _ => None,
    };
    Ok((scheme, host, port.or(default)))
}

pub struct AccessPolicy {
    hosts: Vec<String>,
    origins: Vec<Origin>,
    any_origin: bool,
}

impl AccessPolicy {
    pub fn new(bind: &str, hosts: &[String], origins: &[String]) -> Result<Self> {
        let mut allowed = vec!["localhost".into(), "127.0.0.1".into(), "::1".into()];
        allowed.extend(
            std::iter::once(bind)
                .chain(hosts.iter().map(String::as_str))
                .map(|h| {
                    h.trim_matches(['[', ']'])
                        .to_ascii_lowercase()
                        .trim_end_matches('.')
                        .to_owned()
                }),
        );
        let parsed = origins
            .iter()
            .filter(|s| *s != "*")
            .map(|s| origin(s))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            hosts: allowed,
            origins: parsed,
            any_origin: origins.iter().any(|s| s == "*"),
        })
    }

    fn without_origin(headers: &HeaderMap) -> HeaderMap {
        let mut clean = headers.clone();
        clean.remove("origin");
        clean
    }

    /// Header admission precedes authentication. Its result also applies to
    /// error responses and SSE, so browsers can read authentication failures.
    pub fn allowed_origin(&self, headers: &HeaderMap) -> Result<Option<String>> {
        if headers.get_all("host").iter().count() != 1
            || headers.get_all("origin").iter().count() > 1
        {
            return Err(forbidden(
                "expected one Host header and at most one Origin header",
            ));
        }
        guard(
            &Self::without_origin(headers),
            "/",
            "OPTIONS",
            None,
            &self.hosts,
        )?;
        let Some(value) = headers.get("origin") else {
            return Ok(None);
        };
        if self.any_origin {
            return Ok(Some("*".into()));
        }
        let value = value
            .to_str()
            .map_err(|_| forbidden("invalid Origin header"))?;
        let parsed = origin(value)?;
        let (host, port) = authority(
            headers["host"]
                .to_str()
                .map_err(|_| forbidden("invalid Host header"))?,
        )?;
        let default = match parsed.0.as_str() {
            "http" => Some(80),
            "https" => Some(443),
            _ => None,
        };
        if default.is_some() && parsed.1 == host && parsed.2 == port.or(default) {
            return Ok(None);
        }
        if !self.origins.contains(&parsed) {
            return Err(forbidden(format!("Origin {value} is not allowed; restart the server with --allowed-origin {value} to accept it")));
        }
        Ok(Some(value.into()))
    }

    pub fn authenticate(
        &self,
        headers: &HeaderMap,
        path: &str,
        method: &str,
        key: Option<&str>,
    ) -> Result<()> {
        guard(
            &Self::without_origin(headers),
            path,
            method,
            key,
            &self.hosts,
        )
    }
}

pub fn cors_headers(headers: &mut HeaderMap, allowed: Option<&str>, preflight: Option<&HeaderMap>) {
    let Some(allowed) = allowed else { return };
    headers.insert(
        "access-control-allow-origin",
        allowed.parse().expect("validated Origin"),
    );
    if allowed != "*" {
        headers.append("vary", "Origin".parse().unwrap());
    }
    headers.insert(
        "access-control-expose-headers",
        "Retry-After, WWW-Authenticate".parse().unwrap(),
    );
    if let Some(request) = preflight.filter(|h| h.contains_key("access-control-request-method")) {
        headers.insert("access-control-allow-methods", METHODS.parse().unwrap());
        if let Some(value) = request.get("access-control-request-headers").filter(|v| {
            v.as_bytes()
                .iter()
                .all(|&b| b == b' ' || b.is_ascii_graphic())
        }) {
            headers.insert("access-control-allow-headers", value.clone());
        }
        headers.insert("access-control-max-age", "600".parse().unwrap());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn headers(host: &str, origin: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert("host", host.parse().unwrap());
        h.insert("origin", origin.parse().unwrap());
        h
    }
    #[test]
    fn named_hosts_and_webviews_are_explicit() {
        let policy = AccessPolicy::new(
            "0.0.0.0",
            &["Mac.Local.".into()],
            &["tauri://localhost".into(), "https://ui.example:443".into()],
        )
        .unwrap();
        assert_eq!(
            policy
                .allowed_origin(&headers("mac.local:8000", "tauri://localhost"))
                .unwrap(),
            Some("tauri://localhost".into())
        );
        assert!(policy
            .allowed_origin(&headers("evil.example", "tauri://localhost"))
            .is_err());
        assert!(policy
            .allowed_origin(&headers("localhost:8000", "https://evil.example"))
            .is_err());
        assert_eq!(
            policy
                .allowed_origin(&headers("localhost:8000", "https://UI.EXAMPLE"))
                .unwrap(),
            Some("https://UI.EXAMPLE".into())
        );
        assert_eq!(
            policy
                .allowed_origin(&headers("localhost:8000", "http://LOCALHOST:8000"))
                .unwrap(),
            None
        );
    }
    #[test]
    fn cors_survives_authentication_failure_and_preflight_is_public() {
        let policy = AccessPolicy::new("127.0.0.1", &[], &["tauri://localhost".into()]).unwrap();
        let mut request = headers("localhost:8000", "tauri://localhost");
        let admitted = policy.allowed_origin(&request).unwrap();
        assert_eq!(
            policy
                .authenticate(&request, "/v1/messages", "POST", Some("secret"))
                .unwrap_err()
                .0,
            401
        );
        assert!(policy
            .authenticate(&request, "/v1/messages", "OPTIONS", Some("secret"))
            .is_ok());
        request.insert("access-control-request-method", "POST".parse().unwrap());
        request.insert(
            "access-control-request-headers",
            "content-type, x-api-key".parse().unwrap(),
        );
        let mut response = HeaderMap::new();
        cors_headers(&mut response, admitted.as_deref(), Some(&request));
        assert_eq!(response["access-control-allow-origin"], "tauri://localhost");
        assert_eq!(response["vary"], "Origin");
        assert_eq!(response["access-control-max-age"], "600");
        assert_eq!(
            response["access-control-allow-headers"],
            "content-type, x-api-key"
        );
    }
    #[test]
    fn wildcard_matches_python_including_opaque_origins_but_not_duplicate_headers() {
        let policy = AccessPolicy::new("127.0.0.1", &[], &["*".into()]).unwrap();
        let mut request = headers("localhost", "null");
        assert_eq!(policy.allowed_origin(&request).unwrap(), Some("*".into()));
        request.append("origin", "https://ui.example".parse().unwrap());
        assert!(policy.allowed_origin(&request).is_err());
        for value in [
            "http://*.example.com",
            "tauri://*",
            "http://localhost/",
            "null",
        ] {
            assert!(AccessPolicy::new("localhost", &[], &[value.into()]).is_err());
        }
    }
}
