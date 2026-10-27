//! Configuration layering: environment > `config.toml` > built-in defaults.
//!
//! Only values a human would want to tune live here. Protocol constants (API
//! paths, header names, the PoW algorithm) stay in code — putting them in a
//! config file invites silent divergence from the server.

use serde::Deserialize;
use std::collections::HashMap;
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
    /// Keyed by channel id, e.g. `[channels.deepseek_web]`.
    pub channels: HashMap<String, ChannelSection>,
    /// Optional ChatGPT Web channel configuration.
    pub chatgpt: ChatgptSection,
}

/// Per-channel tuning for the `/v1/responses` adapters.
///
/// `max_output_tokens` is the output budget the proxy itself enforces. An
/// upstream that has no such parameter (the DeepSeek web API) cannot apply it,
/// so the proxy truncates instead. A request above this value is rejected;
/// a request within it is honoured by local truncation.
#[derive(Debug, Default, Deserialize, Clone)]
#[serde(default)]
pub struct ChannelSection {
    pub max_output_tokens: Option<u32>,
}

/// Non-secret configuration for the optional ChatGPT Web upstream channel.
///
/// Credentials (access tokens) are strictly loaded from environment variables
/// (`CHATGPT_TOKEN_<n>` / `CHATGPT_ACCESS_TOKEN_<n>`), never from `config.toml`.
#[derive(Debug, Default, Deserialize, Clone, PartialEq)]
#[serde(default)]
pub struct ChatgptSection {
    /// Whether the ChatGPT Web channel is enabled.
    pub enabled: bool,
    /// Upstream ChatGPT base URL (defaults to `https://chatgpt.com`).
    pub base_url: Option<String>,
    /// HTTP/HTTPS/SOCKS5 proxy URL for upstream requests.
    #[serde(alias = "proxy")]
    pub proxy_url: Option<String>,
    /// Request timeout in seconds (defaults to 60.0s).
    #[serde(alias = "timeout")]
    pub timeout_secs: Option<f64>,
}

/// Fully resolved ChatGPT Web channel configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatGptConfig {
    /// Whether the channel is explicitly enabled.
    pub enabled: bool,
    /// Verified non-blank account tokens loaded from environment variables.
    pub tokens: Vec<String>,
    /// Validated base URL for upstream ChatGPT Web requests.
    pub base_url: String,
    /// Optional validated proxy URL (HTTP, HTTPS, SOCKS5).
    pub proxy_url: Option<String>,
    /// Validated request timeout in seconds.
    pub timeout_secs: f64,
}

#[allow(dead_code)]
pub type ChatgptConfig = ChatGptConfig;

pub const DEFAULT_CHATGPT_BASE_URL: &str = "https://chatgpt.com";
pub const DEFAULT_CHATGPT_TIMEOUT_SECS: f64 = 60.0;

/// All ChatGPT Web model IDs known to the proxy from chatgpt2api.
pub const CHATGPT_MODELS: &[&str] = &[
    "auto",
    "gpt-5",
    "gpt-5-1",
    "gpt-5-2",
    "gpt-5-3",
    "gpt-5-3-mini",
    "gpt-5-mini",
    "gpt-image-2",
    "codex-gpt-image-2",
];

/// Expose actual ChatGPT model IDs from chatgpt2api.
pub fn chatgpt_models() -> &'static [&'static str] {
    CHATGPT_MODELS
}

impl Default for ChatGptConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            tokens: Vec::new(),
            base_url: DEFAULT_CHATGPT_BASE_URL.to_string(),
            proxy_url: None,
            timeout_secs: DEFAULT_CHATGPT_TIMEOUT_SECS,
        }
    }
}

#[allow(dead_code)]
impl ChatGptConfig {
    /// Returns whether ChatGPT models should be dynamically advertised in `/v1/models`.
    ///
    /// Requires the channel to be enabled and at least one non-blank token
    /// configured. Upstream ChatGPT Web channel does not support anonymous access.
    pub fn should_advertise_models(&self) -> bool {
        self.enabled && !self.tokens.is_empty()
    }

    /// Access the list of configured tokens.
    pub fn token_list(&self) -> &[String] {
        &self.tokens
    }

    /// Validated base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Validated proxy URL if configured.
    pub fn proxy_url(&self) -> Option<&str> {
        self.proxy_url.as_deref()
    }

    /// Timeout in seconds.
    pub fn timeout_secs(&self) -> f64 {
        self.timeout_secs
    }
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
    /// Create a brand-new remote session for every request and send only that
    /// request's own input, carrying no conversation history.
    ///
    /// Intended for stateless decision calls (e.g. a fast classifier) where a
    /// contaminated context is worse than a cold start. Off by default; while
    /// off, session reuse and persistence behave exactly as before.
    pub fresh_session_per_request: bool,
}

impl Default for DeepseekSection {
    fn default() -> Self {
        Self {
            model: "deepseek-chat".into(),
            thinking: false,
            search: false,
            fresh_session_per_request: false,
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
    /// Output budget per channel id, resolved from `[channels.<id>]` and
    /// `CHANNEL_MAX_OUTPUT_TOKENS`. A channel absent here has no limit.
    pub channel_max_output_tokens: HashMap<String, u32>,
    /// When true, every request gets a brand-new remote session and carries
    /// only its own input. See `DeepseekSection::fresh_session_per_request`.
    pub fresh_session_per_request: bool,
    /// ChatGPT Web channel configuration.
    pub chatgpt: ChatGptConfig,
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
            fresh_session_per_request: env_flag(
                "FRESH_SESSION_PER_REQUEST",
                file.deepseek.fresh_session_per_request,
            ),
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
            channel_max_output_tokens: resolve_channel_limits(
                &file.channels,
                env_str("CHANNEL_MAX_OUTPUT_TOKENS").as_deref(),
            ),
            chatgpt: resolve_chatgpt_config(&file.chatgpt, &std::env::vars().collect())?,
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

/// Validate an HTTP or HTTPS base URL. Strips trailing slashes on success.
pub fn validate_base_url(raw: &str) -> Result<String, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("base_url cannot be empty".into());
    }
    let parsed = reqwest::Url::parse(trimmed)
        .map_err(|e| format!("invalid ChatGPT base URL '{trimmed}': {e}"))?;
    match parsed.scheme() {
        "http" | "https" => {}
        other => {
            return Err(format!(
                "unsupported scheme '{other}' in base URL '{trimmed}'; expected http or https"
            ));
        }
    }
    if !parsed.has_host() {
        return Err(format!("ChatGPT base URL '{trimmed}' is missing a host"));
    }
    Ok(trimmed.trim_end_matches('/').to_string())
}

/// Validate a proxy URL (http, https, socks5, socks5h).
pub fn validate_proxy_url(raw: &str) -> Result<String, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("proxy_url cannot be empty".into());
    }
    let parsed = reqwest::Url::parse(trimmed)
        .map_err(|e| format!("invalid ChatGPT proxy URL '{trimmed}': {e}"))?;
    match parsed.scheme() {
        "http" | "https" | "socks5" | "socks5h" => {}
        other => {
            return Err(format!(
                "unsupported proxy scheme '{other}' in '{trimmed}'; expected http, https, socks5, or socks5h"
            ));
        }
    }
    if !parsed.has_host() {
        return Err(format!("ChatGPT proxy URL '{trimmed}' is missing a host"));
    }
    Ok(trimmed.to_string())
}

/// Validate a timeout in seconds: must be finite and > 0.0.
pub fn validate_timeout_secs(secs: f64) -> Result<f64, String> {
    if secs <= 0.0 || secs.is_nan() || secs.is_infinite() {
        return Err(format!(
            "invalid timeout {secs}; must be a positive finite number of seconds"
        ));
    }
    Ok(secs)
}

/// Clean and validate a single token string.
///
/// Trims whitespace, strips optional `Bearer ` prefix, and returns `None`
/// if the token is blank or contains invalid characters (such as whitespace).
pub fn clean_token(raw: &str) -> Option<String> {
    let t = raw.trim();
    if t.is_empty() || t.eq_ignore_ascii_case("bearer") {
        return None;
    }
    let stripped = if t.to_ascii_lowercase().starts_with("bearer ") {
        &t[7..]
    } else {
        t
    }
    .trim();
    if stripped.is_empty() || stripped.contains(char::is_whitespace) {
        None
    } else {
        Some(stripped.to_string())
    }
}

/// Extract and normalize ChatGPT tokens from environment variable pairs.
///
/// Supports indexed forms `CHATGPT_TOKEN_<n>` and `CHATGPT_ACCESS_TOKEN_<n>`,
/// as well as unindexed `CHATGPT_TOKEN` / `CHATGPT_ACCESS_TOKEN`.
/// Empty, whitespace-only, and malformed tokens are omitted; duplicates are removed
/// while preserving stable index order.
pub fn resolve_chatgpt_tokens<I, K, V>(vars: I) -> Vec<String>
where
    I: IntoIterator<Item = (K, V)>,
    K: AsRef<str>,
    V: AsRef<str>,
{
    let mut indexed: std::collections::BTreeMap<usize, Vec<String>> =
        std::collections::BTreeMap::new();
    let mut unindexed: Vec<String> = Vec::new();

    for (k, v) in vars {
        let key = k.as_ref();
        let val = v.as_ref();

        let Some(cleaned) = clean_token(val) else {
            continue;
        };

        if key.eq_ignore_ascii_case("CHATGPT_TOKEN")
            || key.eq_ignore_ascii_case("CHATGPT_ACCESS_TOKEN")
        {
            unindexed.push(cleaned);
        } else if let Some(idx_str) = key
            .strip_prefix("CHATGPT_TOKEN_")
            .or_else(|| key.strip_prefix("chatgpt_token_"))
            .or_else(|| key.strip_prefix("CHATGPT_ACCESS_TOKEN_"))
            .or_else(|| key.strip_prefix("chatgpt_access_token_"))
        {
            if let Ok(idx) = idx_str.parse::<usize>() {
                indexed.entry(idx).or_default().push(cleaned);
            }
        }
    }

    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for tok in unindexed {
        if seen.insert(tok.clone()) {
            out.push(tok);
        }
    }

    for (_idx, tokens) in indexed {
        for tok in tokens {
            if seen.insert(tok.clone()) {
                out.push(tok);
            }
        }
    }

    out
}

/// Resolve the ChatGPT Web configuration from file section and environment variables.
pub fn resolve_chatgpt_config(
    file: &ChatgptSection,
    env_vars: &HashMap<String, String>,
) -> Result<ChatGptConfig, String> {
    // 1. enabled: env var CHATGPT_ENABLED overrides file
    let enabled = match env_vars.get("CHATGPT_ENABLED").map(|s| s.trim()) {
        Some(v) => matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        None => file.enabled,
    };

    // 2. tokens: purely from env, never from file
    let tokens = resolve_chatgpt_tokens(env_vars);

    // 3. base_url: env var CHATGPT_BASE_URL overrides file, defaults to https://chatgpt.com
    let raw_base_url = env_vars
        .get("CHATGPT_BASE_URL")
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .or_else(|| {
            file.base_url
                .as_deref()
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
        })
        .unwrap_or(DEFAULT_CHATGPT_BASE_URL);
    let base_url = validate_base_url(raw_base_url)?;

    // 4. proxy_url: env var CHATGPT_PROXY / CHATGPT_PROXY_URL overrides file
    let raw_proxy_url = env_vars
        .get("CHATGPT_PROXY")
        .or_else(|| env_vars.get("CHATGPT_PROXY_URL"))
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .or_else(|| {
            file.proxy_url
                .as_deref()
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
        });
    let proxy_url = match raw_proxy_url {
        Some(raw) => Some(validate_proxy_url(raw)?),
        None => None,
    };

    // 5. timeout_secs: env var CHATGPT_TIMEOUT / CHATGPT_TIMEOUT_SECS overrides file
    let timeout_secs = match env_vars
        .get("CHATGPT_TIMEOUT")
        .or_else(|| env_vars.get("CHATGPT_TIMEOUT_SECS"))
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
    {
        Some(val_str) => {
            let parsed = val_str
                .parse::<f64>()
                .map_err(|e| format!("invalid CHATGPT_TIMEOUT '{val_str}': {e}"))?;
            validate_timeout_secs(parsed)?
        }
        None => match file.timeout_secs {
            Some(file_val) => validate_timeout_secs(file_val)?,
            None => DEFAULT_CHATGPT_TIMEOUT_SECS,
        },
    };

    Ok(ChatGptConfig {
        enabled,
        tokens,
        base_url,
        proxy_url,
        timeout_secs,
    })
}

/// Pure resolver for `max_inflight_per_account`: defaults to file config,
/// overridden by environment variable, and clamped to at least 1.
pub fn resolve_max_inflight_per_account(file_val: usize, env_val: Option<&str>) -> usize {
    env_val
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(file_val)
        .max(1)
}

/// Resolve per-channel output budgets from the file section plus the optional
/// `CHANNEL_MAX_OUTPUT_TOKENS` override (`deepseek_web=8192,other=4096`).
///
/// A limit below 1 is meaningless and dropped, so a typo cannot silently make
/// every request fail. Channels with no configured limit stay absent, which
/// keeps the fail-closed behaviour for `max_output_tokens`.
pub fn resolve_channel_limits(
    file: &HashMap<String, ChannelSection>,
    env_val: Option<&str>,
) -> HashMap<String, u32> {
    let mut limits: HashMap<String, u32> = file
        .iter()
        .filter_map(|(id, section)| {
            section
                .max_output_tokens
                .filter(|limit| *limit >= 1)
                .map(|limit| (id.clone(), limit))
        })
        .collect();
    for entry in env_val.unwrap_or_default().split(',') {
        let Some((id, value)) = entry.split_once('=') else {
            continue;
        };
        let (id, value) = (id.trim(), value.trim());
        if id.is_empty() {
            continue;
        }
        match value.parse::<u32>() {
            Ok(limit) if limit >= 1 => {
                limits.insert(id.to_string(), limit);
            }
            _ => {
                limits.remove(id);
            }
        }
    }
    limits
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
    fn fresh_session_defaults_off_and_parses_from_toml() {
        let default: FileConfig = toml::from_str("").unwrap();
        assert!(!default.deepseek.fresh_session_per_request);
        let on: FileConfig =
            toml::from_str("[deepseek]\nfresh_session_per_request = true\n").unwrap();
        assert!(on.deepseek.fresh_session_per_request);
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

    #[test]
    fn chatgpt_section_defaults() {
        let f: FileConfig = toml::from_str("").unwrap();
        assert!(!f.chatgpt.enabled);
        assert_eq!(f.chatgpt.base_url, None);
        assert_eq!(f.chatgpt.proxy_url, None);
        assert_eq!(f.chatgpt.timeout_secs, None);

        let default_cfg = ChatGptConfig::default();
        assert!(!default_cfg.enabled);
        assert!(default_cfg.tokens.is_empty());
        assert_eq!(default_cfg.base_url, "https://chatgpt.com");
        assert_eq!(default_cfg.proxy_url, None);
        assert_eq!(default_cfg.timeout_secs, 60.0);
    }

    #[test]
    fn chatgpt_section_from_toml() {
        let toml_str = r#"
[chatgpt]
enabled = true
base_url = "https://custom-chatgpt.example.com"
proxy = "socks5://127.0.0.1:1080"
timeout = 45.0
"#;
        let f: FileConfig = toml::from_str(toml_str).unwrap();
        assert!(f.chatgpt.enabled);
        assert_eq!(
            f.chatgpt.base_url.as_deref(),
            Some("https://custom-chatgpt.example.com")
        );
        assert_eq!(
            f.chatgpt.proxy_url.as_deref(),
            Some("socks5://127.0.0.1:1080")
        );
        assert_eq!(f.chatgpt.timeout_secs, Some(45.0));

        let res = resolve_chatgpt_config(&f.chatgpt, &HashMap::new()).unwrap();
        assert!(res.enabled);
        assert_eq!(res.base_url, "https://custom-chatgpt.example.com");
        assert_eq!(res.proxy_url.as_deref(), Some("socks5://127.0.0.1:1080"));
        assert_eq!(res.timeout_secs, 45.0);
        assert!(res.tokens.is_empty());
    }

    #[test]
    fn chatgpt_toml_reads_no_secrets() {
        let toml_str = r#"
[chatgpt]
enabled = true
token = "secret_in_toml"
tokens = ["secret1", "secret2"]
"#;
        let f: FileConfig = toml::from_str(toml_str).unwrap();
        let res = resolve_chatgpt_config(&f.chatgpt, &HashMap::new()).unwrap();
        assert!(res.enabled);
        // Secrets in TOML are ignored; tokens strictly from env
        assert!(res.tokens.is_empty());
    }

    #[test]
    fn resolve_chatgpt_tokens_filtering_and_indexing() {
        let mut env_vars: HashMap<String, String> = HashMap::new();
        env_vars.insert("CHATGPT_TOKEN_0".into(), "".into()); // blank -> omitted
        env_vars.insert("CHATGPT_TOKEN_1".into(), "   ".into()); // whitespace -> omitted
        env_vars.insert("CHATGPT_TOKEN_2".into(), "tok_two".into());
        env_vars.insert("CHATGPT_TOKEN_3".into(), "Bearer ".into()); // blank after prefix -> omitted
        env_vars.insert("CHATGPT_TOKEN_4".into(), "tok with spaces".into()); // whitespace inside -> omitted
        env_vars.insert("CHATGPT_TOKEN_5".into(), "Bearer tok_five".into()); // trimmed
        env_vars.insert("CHATGPT_TOKEN_6".into(), "tok_two".into()); // duplicate -> omitted
        env_vars.insert("CHATGPT_ACCESS_TOKEN_7".into(), "tok_seven".into()); // alternative prefix
        env_vars.insert("CHATGPT_TOKEN".into(), "tok_unindexed".into());

        let tokens = resolve_chatgpt_tokens(&env_vars);
        assert_eq!(
            tokens,
            vec!["tok_unindexed", "tok_two", "tok_five", "tok_seven"]
        );
    }

    #[test]
    fn validate_urls_and_timeout() {
        // Base URL validation
        assert_eq!(
            validate_base_url("https://chatgpt.com/").unwrap(),
            "https://chatgpt.com"
        );
        assert_eq!(
            validate_base_url("http://localhost:8080/v1").unwrap(),
            "http://localhost:8080/v1"
        );
        assert!(validate_base_url("").is_err());
        assert!(validate_base_url("not-a-url").is_err());
        assert!(validate_base_url("ftp://example.com").is_err());
        assert!(validate_base_url("http://").is_err());

        // Proxy URL validation
        assert_eq!(
            validate_proxy_url("http://127.0.0.1:7890").unwrap(),
            "http://127.0.0.1:7890"
        );
        assert_eq!(
            validate_proxy_url("socks5://user:pass@127.0.0.1:1080").unwrap(),
            "socks5://user:pass@127.0.0.1:1080"
        );
        assert_eq!(
            validate_proxy_url("socks5h://127.0.0.1:1080").unwrap(),
            "socks5h://127.0.0.1:1080"
        );
        assert!(validate_proxy_url("").is_err());
        assert!(validate_proxy_url("invalid_proxy").is_err());
        assert!(validate_proxy_url("ftp://127.0.0.1").is_err());

        // Timeout validation
        assert_eq!(validate_timeout_secs(60.0).unwrap(), 60.0);
        assert_eq!(validate_timeout_secs(0.5).unwrap(), 0.5);
        assert!(validate_timeout_secs(0.0).is_err());
        assert!(validate_timeout_secs(-10.0).is_err());
        assert!(validate_timeout_secs(f64::NAN).is_err());
        assert!(validate_timeout_secs(f64::INFINITY).is_err());
    }

    #[test]
    fn resolve_chatgpt_config_env_overrides() {
        let section = ChatgptSection {
            enabled: false,
            base_url: Some("https://file.example.com".into()),
            proxy_url: Some("http://file-proxy:8080".into()),
            timeout_secs: Some(30.0),
        };

        let mut env_vars = HashMap::new();
        env_vars.insert("CHATGPT_ENABLED".into(), "true".into());
        env_vars.insert("CHATGPT_BASE_URL".into(), "https://env.example.com".into());
        env_vars.insert("CHATGPT_PROXY".into(), "socks5://127.0.0.1:1080".into());
        env_vars.insert("CHATGPT_TIMEOUT".into(), "120".into());
        env_vars.insert("CHATGPT_TOKEN_0".into(), "tok_env_0".into());

        let res = resolve_chatgpt_config(&section, &env_vars).unwrap();
        assert!(res.enabled);
        assert_eq!(res.base_url, "https://env.example.com");
        assert_eq!(res.proxy_url.as_deref(), Some("socks5://127.0.0.1:1080"));
        assert_eq!(res.timeout_secs, 120.0);
        assert_eq!(res.tokens, vec!["tok_env_0"]);
    }

    #[test]
    fn chatgpt_models_constant_and_function() {
        let expected = [
            "auto",
            "gpt-5",
            "gpt-5-1",
            "gpt-5-2",
            "gpt-5-3",
            "gpt-5-3-mini",
            "gpt-5-mini",
            "gpt-image-2",
            "codex-gpt-image-2",
        ];
        assert_eq!(CHATGPT_MODELS, expected);
        assert_eq!(chatgpt_models(), expected);
    }

    #[test]
    fn should_advertise_models_rules() {
        // Disabled channel: never advertise
        let mut cfg = ChatGptConfig::default();
        cfg.enabled = false;
        cfg.tokens = vec!["tok".into()];
        assert!(!cfg.should_advertise_models());

        // Enabled channel but no tokens: does not advertise (no anonymous upstream)
        cfg.enabled = true;
        cfg.tokens = vec![];
        assert!(!cfg.should_advertise_models());

        // Enabled channel with tokens: advertise!
        cfg.tokens = vec!["tok".into()];
        assert!(cfg.should_advertise_models());
    }
}
