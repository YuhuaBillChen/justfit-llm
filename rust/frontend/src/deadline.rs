//! Unbounded HTTP requests use Duration::MAX. Native deadlines saturate like
//! 2.3; upload, stream-write, lease and worker safety limits stay separate.
use crate::{bad, ApiError, Result};
use serde_json::{json, Value};
use std::{future::Future, time::{Duration, Instant, SystemTime, UNIX_EPOCH}};

pub const UNLIMITED: Duration = Duration::MAX;

pub fn expired() -> ApiError {
    ApiError(504, "request timed out".into(), "request_timeout".into())
}

pub fn duration(value: Option<&Value>) -> Result<Duration> {
    let Some(value) = value.filter(|v| !v.is_null()) else { return Ok(UNLIMITED) };
    seconds(Some(value.as_f64().ok_or_else(|| bad("timeout must be positive"))?))
}

pub fn seconds(value: Option<f64>) -> Result<Duration> {
    let Some(seconds) = value else { return Ok(UNLIMITED) };
    if !seconds.is_finite() || seconds <= 0. { return Err(bad("timeout must be positive")) }
    // A positive deadline beyond the protocol or platform clock's range is
    // indistinguishable from its maximum sentinel, as in the 2.3 wire encoder.
    Ok(Duration::try_from_secs_f64(seconds).unwrap_or(UNLIMITED))
}

pub fn request(body: &Value) -> Result<Duration> { duration(body.get("timeout")) }

pub fn remaining(timeout: Duration, elapsed: Duration) -> Result<Duration> {
    if timeout == UNLIMITED { return Ok(UNLIMITED) }
    timeout.checked_sub(elapsed).filter(|d| !d.is_zero()).ok_or_else(expired)
}

pub fn json_seconds(timeout: Duration) -> Value {
    if timeout == UNLIMITED { Value::Null } else { json!(timeout.as_secs_f64()) }
}

pub fn mono_deadline(timeout: Duration) -> Option<Instant> {
    if timeout == UNLIMITED { None } else { Instant::now().checked_add(timeout) }
}

pub fn tokio_deadline(timeout: Duration) -> Option<tokio::time::Instant> {
    if timeout == UNLIMITED { None } else { tokio::time::Instant::now().checked_add(timeout) }
}

pub fn wire_fields(timeout: Duration) -> (u64, u64) {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default()
        .as_micros().min(u64::MAX as u128) as u64;
    wire_fields_at(timeout, now)
}

pub fn wire_fields_at(timeout: Duration, now: u64) -> (u64, u64) {
    let limit = u64::MAX - now;
    let micros = if timeout == UNLIMITED { limit } else {
        timeout.as_micros().max(1).min(limit as u128) as u64
    };
    (now + micros, micros)
}

pub async fn wait<F: Future>(timeout: Duration, future: F) -> Result<F::Output> {
    wait_until(tokio_deadline(timeout), future).await
}

pub async fn wait_until<F: Future>(deadline: Option<tokio::time::Instant>, future: F) -> Result<F::Output> {
    match deadline {
        None => Ok(future.await),
        Some(deadline) => tokio::time::timeout_at(deadline, future).await.map_err(|_| expired()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unlimited_survives_elapsed_time_and_protocol_serialization() {
        assert_eq!(remaining(UNLIMITED,Duration::from_secs(9001)).unwrap(),UNLIMITED);
        assert_eq!(wire_fields_at(UNLIMITED,1234),(u64::MAX,u64::MAX-1234));
        assert!(mono_deadline(UNLIMITED).is_none());
        assert!(tokio_deadline(UNLIMITED).is_none());
        assert_eq!(json_seconds(UNLIMITED),Value::Null);
        let duration=remaining(Duration::from_secs(9001),Duration::from_secs(1)).unwrap();
        assert_eq!(wire_fields(duration).1,9_000_000_000);
    }
    #[tokio::test]
    async fn unbounded_wait_is_awaited_without_clock_overflow() {
        assert_eq!(wait(UNLIMITED,async { 17 }).await.unwrap(),17);
        let error=wait(Duration::ZERO,std::future::pending::<()>()).await.unwrap_err();
        assert_eq!(error.0,504);assert_eq!(error.2,"request_timeout");
    }
}
