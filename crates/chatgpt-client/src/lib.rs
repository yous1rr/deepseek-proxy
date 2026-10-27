//! Native ChatGPT Web client using the project's reqwest/rustls transport.
//!
//! This deliberately does not claim to reproduce Chrome's TLS ClientHello or
//! HTTP/2 fingerprints. Turnstile uses a bounded, allowlisted bytecode VM and
//! never evaluates arbitrary JavaScript.
use std::{
    fmt,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

mod client;
mod pow;
mod turnstile;

pub use client::ChatGptClient;

const DEFAULT_BASE_URL: &str = "https://chatgpt.com";

/// Configuration for a ChatGPT Web upstream client.
#[derive(Clone)]
pub struct ClientConfig {
    pub base_url: String,
    pub access_token: String,
    pub proxy_url: Option<String>,
    pub timeout: std::time::Duration,
}
impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.into(),
            access_token: String::new(),
            proxy_url: None,
            timeout: std::time::Duration::from_secs(300),
        }
    }
}
impl fmt::Debug for ClientConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let base = safe_origin(&self.base_url);
        let proxy = self.proxy_url.as_ref().map(|_| "[REDACTED]");
        f.debug_struct("ClientConfig")
            .field("base_url", &base)
            .field("access_token", &"[REDACTED]")
            .field("proxy_url", &proxy)
            .field("timeout", &self.timeout)
            .finish()
    }
}
fn safe_origin(raw: &str) -> String {
    url::Url::parse(raw)
        .map(|u| u.origin().ascii_serialization())
        .unwrap_or_else(|_| "[REDACTED URL]".into())
}

/// Safe error categories. Network and protocol errors intentionally carry no
/// request, response, URL, token, or raw response-body strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    Configuration,
    InvalidRequest,
    Authentication,
    RateLimit,
    Timeout,
    Network,
    Upstream,
    InvalidResponse,
    ProofOfWork,
    Unsupported,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    kind: ErrorKind,
    status_code: Option<u16>,
    retry_after: Option<u64>,
}
impl Error {
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }
    pub fn status_code(&self) -> Option<u16> {
        self.status_code
    }
    pub fn retry_after(&self) -> Option<u64> {
        self.retry_after
    }
    pub fn is_auth_error(&self) -> bool {
        self.kind == ErrorKind::Authentication
    }
    pub fn is_rate_limited(&self) -> bool {
        self.kind == ErrorKind::RateLimit
    }
    pub fn is_timeout(&self) -> bool {
        self.kind == ErrorKind::Timeout
    }
    pub(crate) fn new(kind: ErrorKind) -> Self {
        Self {
            kind,
            status_code: None,
            retry_after: None,
        }
    }
    pub(crate) fn http(status: u16, retry_after: Option<u64>) -> Self {
        let kind = match status {
            401 | 403 => ErrorKind::Authentication,
            429 => ErrorKind::RateLimit,
            _ => ErrorKind::Upstream,
        };
        Self {
            kind,
            status_code: Some(status),
            retry_after,
        }
    }
    pub(crate) fn from_reqwest(error: &reqwest::Error) -> Self {
        Self::new(if error.is_timeout() {
            ErrorKind::Timeout
        } else {
            ErrorKind::Network
        })
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            ErrorKind::Configuration => f.write_str("invalid client configuration"),
            ErrorKind::InvalidRequest => f.write_str("invalid client request"),
            ErrorKind::Authentication => write!(
                f,
                "upstream authentication failed{}",
                self.status_code
                    .map(|s| format!(" (HTTP {s})"))
                    .unwrap_or_default()
            ),
            ErrorKind::RateLimit => write!(
                f,
                "upstream rate limit reached{}",
                self.status_code
                    .map(|s| format!(" (HTTP {s})"))
                    .unwrap_or_default()
            ),
            ErrorKind::Timeout => f.write_str("upstream operation timed out"),
            ErrorKind::Network => f.write_str("upstream network request failed"),
            ErrorKind::Upstream => write!(
                f,
                "upstream returned HTTP {}",
                self.status_code.unwrap_or(0)
            ),
            ErrorKind::InvalidResponse => f.write_str("upstream response was invalid"),
            ErrorKind::ProofOfWork => {
                f.write_str("upstream proof of work could not be solved within its bound")
            }
            ErrorKind::Unsupported => f.write_str("upstream requested an unsupported challenge"),
        }
    }
}
impl std::error::Error for Error {}

/// A web-shaped message request. `messages` are passed through after structural
/// validation; IDs, routing fields, and history policy are constructed here.
#[derive(Debug, Clone, Default)]
pub struct ConversationRequest {
    pub messages: Vec<serde_json::Value>,
    pub model: String,
    pub thinking_effort: Option<String>,
    pub system_hints: Vec<String>,
}

static ID_COUNTER: AtomicU64 = AtomicU64::new(1);
pub(crate) fn random_u64() -> u64 {
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let mut x = time
        ^ ID_COUNTER
            .fetch_add(1, Ordering::Relaxed)
            .wrapping_mul(0x9e3779b97f4a7c15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
pub(crate) fn new_id() -> String {
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&random_u64().to_be_bytes());
    bytes[8..].copy_from_slice(&random_u64().to_be_bytes());
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    format!("{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}", bytes[0],bytes[1],bytes[2],bytes[3],bytes[4],bytes[5],bytes[6],bytes[7],bytes[8],bytes[9],bytes[10],bytes[11],bytes[12],bytes[13],bytes[14],bytes[15])
}

#[cfg(test)]
mod tests;
