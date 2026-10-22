//! Configuration layering: environment > `config.toml` > built-in defaults.
//!
//! Only values a human would want to tune live here. Protocol constants (API
//! paths, header names, the PoW algorithm) stay in code — putting them in a
//! config file invites silent divergence from the server.

use serde::Deserialize;
use std::path::{Path, PathBuf};

/// File-backed configuration. Every field is optional; unset means "use the
/// default", so a partial `config.toml` is valid.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct FileConfig {
    pub proxy: ProxySection,
    pub storage: StorageSection,
    pub deepseek: DeepseekSection,
    pub limits: LimitsSection,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct ProxySection {
    pub host: String,
    pub port: u16,
}

impl Default for ProxySection {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".into(),
            port: 9174,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct StorageSection {
    pub db_path: String,
}

impl Default for StorageSection {
    fn default() -> Self {
        Self {
            db_path: "deepseek-rs.db".into(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct DeepseekSection {
    /// Model id exposed when the client does not send one.
    pub model: String,
    /// Default thinking flag when the request does not specify one.
    pub thinking: bool,
    /// Send the web-search flag. Off by default: thinking+search is throttled
    /// hard by the backend and turns transient limits into hard failures.
    pub search: bool,
}

impl Default for DeepseekSection {
    fn default() -> Self {
        Self {
            model: "deepseek-chat".into(),
            thinking: false,
            search: false,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct LimitsSection {
    /// Maximum simultaneous upstream requests for one account.
    pub max_inflight_per_account: usize,
    /// Seconds to wait after one request finishes before starting the next on
    /// the same account (completion-to-start, not start-to-start).
    pub request_delay_secs: f64,
    /// Requests allowed to queue behind a busy account before rejecting.
    pub queue_limit: usize,
    /// Max time to wait for a busy account's slot, in seconds. Python:
    /// `QUEUE_TIMEOUT` (default 120).
    pub queue_timeout_secs: f64,
    /// Cooldown for an account that fails to log in, in seconds.
    pub login_cooldown_secs: u64,
    /// Retries for a throttled turn, before the first byte. Python:
    /// `RATE_LIMIT_MAX_RETRIES`.
    pub rate_limit_max_retries: u32,
    /// Base backoff in seconds, multiplied by the attempt number. Python:
    /// `RATE_LIMIT_BACKOFF_S` (and it floors at 0.5s).
    pub rate_limit_backoff_secs: f64,
    /// Correction turns for malformed tool calls. Python:
    /// `TOOL_REPAIR_MAX_RETRIES`.
    pub tool_repair_max_retries: u32,
    /// Wall-clock budget for one turn, in seconds. Python:
    /// `REQUEST_TIMEOUT` (default 300), applied by `asyncio.timeout` around the
    /// whole turn. Without it a stalled upstream socket hangs the request
    /// forever and nothing is ever logged.
    pub request_timeout_secs: f64,
    /// SSE keep-alive interval in seconds. Python: `HEARTBEAT_INTERVAL`
    /// (default 10). Clients and proxies drop an idle stream otherwise.
    pub heartbeat_interval_secs: f64,
    pub image_max_bytes: usize,
}

impl Default for LimitsSection {
    fn default() -> Self {
        Self {
            max_inflight_per_account: 1,
            // Measured: sequential requests with no gap do not trip the upstream
            // rate limiter, so the safe default is no delay. Raise it if you see
            // `rate_limit_reached` / `Messages too frequent`.
            request_delay_secs: 0.0,
            queue_limit: 64,
            queue_timeout_secs: 120.0,
            login_cooldown_secs: 300,
            rate_limit_max_retries: 3,
            rate_limit_backoff_secs: 10.0,
            tool_repair_max_retries: 3,
            request_timeout_secs: 300.0,
            heartbeat_interval_secs: 10.0,
            image_max_bytes: 10 * 1024 * 1024,
        }
    }
}

/// Fully resolved configuration.
#[derive(Debug, Clone)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub db_path: String,
    pub model: String,
    pub thinking: bool,
    pub search: bool,
    pub max_inflight_per_account: usize,
    pub request_delay_secs: f64,
    pub queue_limit: usize,
    pub queue_timeout_secs: f64,
    pub login_cooldown_secs: u64,
    pub rate_limit_max_retries: u32,
    pub rate_limit_backoff_secs: f64,
    pub tool_repair_max_retries: u32,
    pub request_timeout_secs: f64,
    pub heartbeat_interval_secs: f64,
    pub image_max_bytes: usize,
}

impl Config {
    /// Resolve config: `config.toml` (if present) then environment overrides.
    ///
    /// `CONFIG_PATH` selects a different file; a missing file is not an error.
    pub fn load() -> Result<Self, String> {
        let path = std::env::var("CONFIG_PATH").unwrap_or_else(|_| default_config_path());
        let file = Self::read_file(&path)?;

        Ok(Self {
            host: env_str("PROXY_HOST").unwrap_or(file.proxy.host),
            port: env_num("PROXY_PORT").unwrap_or(file.proxy.port),
            db_path: env_str("DB_PATH")
                .map(PathBuf::from)
                .unwrap_or_else(|| anchor_path(&path, &file.storage.db_path))
                .to_string_lossy()
                .into_owned(),
            model: env_str("MODEL_NAME").unwrap_or(file.deepseek.model),
            thinking: env_flag("THINKING_ENABLED", file.deepseek.thinking),
            search: env_flag("SEARCH_ENABLED", file.deepseek.search),
            max_inflight_per_account: resolve_max_inflight_per_account(
                file.limits.max_inflight_per_account,
                env_str("MAX_INFLIGHT_PER_ACCOUNT").as_deref(),
            ),
            request_delay_secs: env_num("REQUEST_DELAY")
                .unwrap_or(file.limits.request_delay_secs)
                .max(0.0),
            queue_limit: env_num("QUEUE_LIMIT").unwrap_or(file.limits.queue_limit),
            queue_timeout_secs: env_num("QUEUE_TIMEOUT")
                .unwrap_or(file.limits.queue_timeout_secs)
                .max(1.0),
            login_cooldown_secs: env_num("LOGIN_COOLDOWN")
                .unwrap_or(file.limits.login_cooldown_secs),
            rate_limit_max_retries: env_num("RATE_LIMIT_MAX_RETRIES")
                .unwrap_or(file.limits.rate_limit_max_retries),
            rate_limit_backoff_secs: env_num("RATE_LIMIT_BACKOFF_S")
                .unwrap_or(file.limits.rate_limit_backoff_secs)
                .max(0.0),
            tool_repair_max_retries: env_num("TOOL_REPAIR_MAX_RETRIES")
                .unwrap_or(file.limits.tool_repair_max_retries),
            request_timeout_secs: env_num("REQUEST_TIMEOUT")
                .unwrap_or(file.limits.request_timeout_secs)
                .max(1.0),
            heartbeat_interval_secs: env_num("HEARTBEAT_INTERVAL")
                .unwrap_or(file.limits.heartbeat_interval_secs)
                .max(1.0),
            image_max_bytes: env_num("IMAGE_MAX_BYTES").unwrap_or(file.limits.image_max_bytes),
        })
    }

    fn read_file(path: &str) -> Result<FileConfig, String> {
        if !Path::new(path).exists() {
            return Ok(FileConfig::default());
        }
        let raw = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
        toml::from_str(&raw).map_err(|e| format!("parse {path}: {e}"))
    }
}

/// Pure resolver for `max_inflight_per_account`: defaults to file config,
/// overridden by environment variable, and clamped to at least 1.
pub fn resolve_max_inflight_per_account(file_val: usize, env_val: Option<&str>) -> usize {
    env_val
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(file_val)
        .max(1)
}

/// Resolve a relative path against the config file's directory, so a relative
/// `db_path` means "next to config.toml" rather than "wherever the process was
/// started". Absolute paths pass through untouched.
fn anchor_path(config_path: &str, value: &str) -> PathBuf {
    let p = PathBuf::from(value);
    if p.is_absolute() {
        return p;
    }
    match Path::new(config_path).parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir.join(p),
        _ => p,
    }
}

/// Default `config.toml` location: the repo root this binary was built from,
/// not the process working directory. Running the server from `/` or from a
/// systemd unit must still pick up the checked-in config.
fn default_config_path() -> String {
    // Compiled-in repo root; `config.rs` lives at <root>/crates/proxy/src/.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let candidate = root.join("config.toml");
    if candidate.exists() {
        return candidate.to_string_lossy().into_owned();
    }
    // Packaged/moved binary: fall back to the working directory.
    "config.toml".into()
}

fn env_str(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

fn env_num<T: std::str::FromStr>(key: &str) -> Option<T> {
    env_str(key).and_then(|v| v.trim().parse().ok())
}

fn env_flag(key: &str, default: bool) -> bool {
    match env_str(key) {
        Some(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        None => default,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_toml_keeps_defaults() {
        let f: FileConfig = toml::from_str("[proxy]\nport = 9999\n").unwrap();
        assert_eq!(f.proxy.port, 9999);
        assert_eq!(f.proxy.host, "0.0.0.0");
        assert_eq!(f.limits.request_delay_secs, 0.0);
        assert!(!f.deepseek.search);
    }

    #[test]
    fn empty_toml_is_all_defaults() {
        let f: FileConfig = toml::from_str("").unwrap();
        assert_eq!(f.proxy.port, 9174);
        assert_eq!(f.storage.db_path, "deepseek-rs.db");
        assert_eq!(f.limits.queue_limit, 64);
    }

    #[test]
    fn relative_db_path_is_anchored_to_the_config_dir() {
        let got = anchor_path("/srv/app/config.toml", "data/deepseek-rs.db");
        assert_eq!(got, PathBuf::from("/srv/app/data/deepseek-rs.db"));
        // Absolute values are left alone, whatever the config location.
        let abs = anchor_path("/srv/app/config.toml", "/var/lib/ds.db");
        assert_eq!(abs, PathBuf::from("/var/lib/ds.db"));
        // Bare filename with no directory component: never becomes absolute.
        assert_eq!(
            anchor_path("config.toml", "deepseek-rs.db"),
            PathBuf::from("deepseek-rs.db")
        );
    }

    #[test]
    fn missing_file_is_not_an_error() {
        let f = Config::read_file("/nonexistent/config.toml").unwrap();
        assert_eq!(f.proxy.port, 9174);
    }

    #[test]
    fn toml_max_inflight_per_account() {
        let f: FileConfig = toml::from_str("[limits]\nmax_inflight_per_account = 4\n").unwrap();
        assert_eq!(f.limits.max_inflight_per_account, 4);
    }

    #[test]
    fn default_max_inflight_per_account_is_one() {
        let f = FileConfig::default();
        assert_eq!(f.limits.max_inflight_per_account, 1);
    }

    #[test]
    fn test_resolve_max_inflight_per_account() {
        assert_eq!(resolve_max_inflight_per_account(1, None), 1);
        assert_eq!(resolve_max_inflight_per_account(3, None), 3);
        assert_eq!(resolve_max_inflight_per_account(3, Some("5")), 5);
        assert_eq!(resolve_max_inflight_per_account(0, None), 1);
        assert_eq!(resolve_max_inflight_per_account(3, Some("0")), 1);
        assert_eq!(resolve_max_inflight_per_account(4, Some("not-a-number")), 4);
        assert_eq!(resolve_max_inflight_per_account(0, Some("bad")), 1);
    }
}
