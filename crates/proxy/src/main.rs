//! OpenAI-compatible proxy over the DeepSeek private API.
//!
//! Rust port of `deepseek_proxy`. Serves:
//!
//! - `GET  /healthz`
//! - `GET  /v1/models`
//! - `POST /v1/chat/completions` (stream + non-stream)
//!
//! ## Session model
//!
//! DeepSeek keeps conversation state server-side, so a thread is just
//! `(chat_session_id, parent_message_id)`. Threads are keyed by an explicit
//! `thread_id`, then the `x-thread-id` header, then the client session id
//! (`x-session-affinity` / `x-session-id`), and only then by a message hash —
//! the same precedence as the Python proxy. Continuation turns send only the
//! newest user message, matching the Python proxy's delta behaviour; a fresh
//! thread sends the flattened transcript.
//!
//! ## Not yet ported
//!
//! - SQLite persistence of threads (in-memory only; lost on restart)
//! - `tools/format.py` prompt + tool-call dialect injection
//! - multi-account failover

mod cli;
mod config;
mod dialect;
mod images;
mod log;
mod pool;
mod store;

use axum::{
    extract::State,
    http::HeaderMap,
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use client::{CompletionBody, COMPLETION_PATH};
use futures_util::StreamExt;
use protocol::{route_event, Channel, FragmentRouter};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio_stream::wrappers::ReceiverStream;

#[derive(Clone)]
struct Settings {
    thinking_default: bool,
    search_enabled: bool,
    model_name: String,
    /// Throttle retries before the first byte (`RATE_LIMIT_MAX_RETRIES`).
    rate_limit_max_retries: u32,
    /// Base backoff (`RATE_LIMIT_BACKOFF_S`), floored at 0.5s like Python.
    rate_limit_backoff_secs: f64,
    /// Correction turns for malformed tool calls (`TOOL_REPAIR_MAX_RETRIES`).
    tool_repair_max_retries: u32,
    /// Whole-turn deadline in seconds (`REQUEST_TIMEOUT`).
    request_timeout_secs: f64,
    /// Largest accepted image in bytes (`IMAGE_MAX_BYTES`).
    image_max_bytes: usize,
    /// SSE keep-alive interval in seconds (`HEARTBEAT_INTERVAL`).
    heartbeat_interval_secs: f64,
}

#[derive(Debug, Clone)]
struct ThreadState {
    session_id: String,
    parent_message_id: Option<i64>,
    /// User/tool messages exchanged so far on this thread. Drives the periodic
    /// tool-format reminder; kept for resume state and diagnostics.
    #[allow(dead_code)]
    exchanges: usize,
}

#[derive(Clone)]
struct AppState {
    pool: Arc<pool::Pool>,
    settings: Settings,
    threads: Arc<Mutex<HashMap<String, ThreadState>>>,
    /// One mutex per thread id, ported from `turn_locks` in
    /// `deepseek_proxy/backend.py`. A turn and all of its correction turns must
    /// not interleave with another turn on the same thread: both would read the
    /// same parent id, and whichever finished last would orphan the other's
    /// message, which DeepSeek renders as an edit.
    turns: Arc<tokio::sync::Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>>,
    store: store::Store,
    stats: Arc<Stats>,
}

impl AppState {
    /// Take the turn lock for `key`, creating it on first use.
    ///
    /// The guard is held from before the parent id is read until the whole turn
    /// (including every correction turn) is finished, which is what keeps two
    /// overlapping requests on one thread from forking the conversation.
    async fn turn_guard(
        &self,
        key: &str,
    ) -> tokio::sync::OwnedMutexGuard<()> {
        let lock = {
            let mut map = self.turns.lock().await;
            map.entry(key.to_string())
                .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
                .clone()
        };
        lock.lock_owned().await
    }
}

/// Process counters reported by `GET /stats`. Cheap atomics only; anything
/// expensive (account gates, db counts) is read live by the handler.
#[derive(Default)]
struct Stats {
    started: Option<Instant>,
    total: AtomicU64,
    ok: AtomicU64,
    errors: AtomicU64,
    auth_failed: AtomicU64,
    retries: AtomicU64,
    tool_calls: AtomicU64,
    /// Images dropped because a turn carried more than DeepSeek accepts.
    images_dropped: AtomicU64,
}

impl Stats {
    fn new() -> Self {
        Self {
            started: Some(Instant::now()),
            ..Default::default()
        }
    }

    fn uptime_secs(&self) -> u64 {
        self.started.map(|t| t.elapsed().as_secs()).unwrap_or(0)
    }
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    #[serde(default)]
    model: Option<String>,
    /// Explicit conversation key, mirroring the Python proxy's `thread_id`.
    #[serde(default)]
    thread_id: Option<String>,
    #[serde(default)]
    messages: Vec<Msg>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    reasoning_effort: Option<String>,
    #[serde(default)]
    thinking: Option<Value>,
    #[serde(default)]
    reasoning: Option<Value>,
    #[serde(default)]
    tools: Vec<Value>,
    /// OpenAI `tool_choice`: "none" drops every tool, "required" forces a call,
    /// an object narrows to one named function.
    #[serde(default)]
    tool_choice: Option<Value>,
    /// OpenAI `parallel_tool_calls`; defaults to true like the OpenAI API.
    #[serde(default = "default_true")]
    parallel_tool_calls: bool,
    /// OpenAI `stream_options`; only `include_usage` is honoured.
    #[serde(default)]
    stream_options: Option<Value>,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
struct Msg {
    role: String,
    #[serde(default)]
    content: Option<Value>,
    // Deserialized for completeness; the prompt builders only need role/content.
    #[serde(default)]
    #[allow(dead_code)]
    tool_calls: Vec<Value>,
    /// Tool-call id this result answers, rendered as `Tool result (id=...)`.
    #[serde(default)]
    tool_call_id: Option<String>,
    #[serde(default)]
    #[allow(dead_code)] // part of the OpenAI message contract; not used in prompts
    name: Option<String>,
}

/// Prompts for one turn: the delta the session needs now, and the full
/// flattened transcript used to rebuild the session when it overflows.
///
/// Mirrors the Python proxy's `prompt` / `replay_prompt` pair.
fn build_prompts(
    req: &ChatRequest,
    existing: Option<&ThreadState>,
    system_extra: &str,
    image_note: Option<&str>,
) -> (String, String) {
    let transcript = flatten_transcript(&req.messages);
    let mut replay = if system_extra.is_empty() {
        transcript
    } else {
        format!("{transcript}\n\n{system_extra}")
    };
    let mut delta = match existing {
        Some(_) => continuation_prompt(delta_messages(&req.messages)).unwrap_or_default(),
        None => replay.clone(),
    };
    // A turn that lost images says so, in both prompts: `replay` is what
    // rebuilds a remote session after a context overflow, so the notice has to
    // survive that too. Placed before the tool reminder, which stays last.
    if let Some(note) = image_note {
        delta = format!("{note}\n\n{delta}");
        replay = format!("{note}\n\n{replay}");
    }
    // Re-anchor the wire format on EVERY turn. Two placements:
    //   - at the very start of a fresh thread, so the model is already reminded
    //     before it reads the first message (nothing has been taught yet in the
    //     delta case, and compaction can drop the original system block);
    //   - at the very end of every turn, closest to the generation, so a long
    //     thread that drifts off the format gets corrected immediately.
    if !req.tools.is_empty() {
        let reminder = dialect::periodic_reminder();
        if existing.is_none() {
            for prompt in [&mut delta, &mut replay] {
                *prompt = format!("{reminder}\n\n{prompt}");
            }
        }
        delta.push_str("\n\n");
        delta.push_str(&reminder);
        replay.push_str("\n\n");
        replay.push_str(&reminder);
    }
    // `tool_choice` / `parallel_tool_calls` are part of the request contract
    // and must reach the model as instructions, exactly as Python appends them
    // after `build_prompt()`.
    if let Some(directive) = tool_choice_directive(req) {
        // `directive` already starts with the leading newline Python uses.
        delta.push_str(&directive);
        replay.push_str(&directive);
    }
    (delta, replay)
}

/// The extra prompt lines Python appends for `tool_choice` / parallel calls.
///
/// Port of the block at the top of `ChatService.events()`:
///
/// ```text
/// if tools:
///     if recovery.required:      prompt += "\nYou must return a tool call."
///     if not parallel:           prompt += "\nReturn at most one tool call."
/// elif request.tool_choice == "none":
///     prompt += "\nDo not call tools in this turn; reply with normal text."
/// ```
///
/// `tools` there is the *filtered* list, so `tool_choice: "none"` takes the
/// `elif` branch rather than the `if`.
fn tool_choice_directive(req: &ChatRequest) -> Option<String> {
    let (specs, required) = resolve_tools(req);
    let mut out = String::new();
    if !specs.is_empty() {
        if required {
            out.push_str("\nYou must return a tool call.");
        }
        if !req.parallel_tool_calls {
            out.push_str("\nReturn at most one tool call.");
        }
    } else if matches!(req.tool_choice.as_ref(), Some(Value::String(s)) if s == "none") {
        out.push_str("\nDo not call tools in this turn; reply with normal text.");
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

fn completion_body(
    thread: &ThreadState,
    prompt: &str,
    thinking: bool,
    search: bool,
    ref_file_ids: &[String],
) -> CompletionBody {
    let mut body = CompletionBody::new(&thread.session_id, prompt);
    body.thinking_enabled = thinking;
    body.search_enabled = search;
    body.parent_message_id = thread.parent_message_id;
    body.ref_file_ids = ref_file_ids.to_vec();
    body
}

/// First fatal `hint` error in a fully-buffered SSE body, if any.
#[allow(dead_code)] // used by tests; the live path classifies incrementally
fn first_hint(raw: &str) -> Option<String> {
    let mut current = String::new();
    for line in raw.lines() {
        if let protocol::LineKind::HintError { content } =
            protocol::classify_line(line, &mut current)
        {
            return Some(content);
        }
    }
    None
}

/// Replace the remote session after a context-limit rejection.
///
/// DeepSeek keeps the conversation server-side, so an overflowing session
/// cannot be trimmed from here: create a fresh one and let the caller replay
/// the whole transcript into it. Persisted resume state is reset to match.
async fn rebuild_session(
    state: &AppState,
    key: &str,
    reason: &str,
) -> Result<ThreadState, Response> {
    let conn = state.pool.route(key).await;
    let session_id = {
        let _slot = conn
            .acquire()
            .await
            .map_err(|e| queue_error(&format!("rate slot: {e}")))?;
        conn.client()
            .await
            .create_chat_session()
            .await
            .map_err(|e| backend_error(&format!("create_session: {e}")))?
    };
    tracing::warn!("thread {key}: {reason}; rebuilding in a new remote session");

    let fresh = ThreadState {
        session_id,
        parent_message_id: None,
        // A rebuilt session replays the whole transcript, so the counters
        // restart with it.
        exchanges: 0,
    };
    state
        .threads
        .lock()
        .unwrap()
        .insert(key.to_string(), fresh.clone());
    let _ = state.store.save_thread(
        key,
        &store::ThreadResume {
            session_id: fresh.session_id.clone(),
            parent_message_id: None,
        },
    );
    Ok(fresh)
}

fn content_text(content: &Option<Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .map(|p| match p.get("type").and_then(|t| t.as_str()) {
                Some("image_url") => "[image]".to_string(),
                Some("text") => p
                    .get("text")
                    .and_then(|t| t.as_str())
                    .unwrap_or_default()
                    .to_string(),
                // Python falls back to `str(part)` for unknown part types.
                _ => p.to_string(),
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Resolve the per-request thinking intent. Precedence mirrors the Python
/// proxy: `reasoning_effort` > `reasoning` object > `thinking` object > config.
fn want_thinking(req: &ChatRequest, default: bool) -> bool {
    const OFF: [&str; 4] = ["none", "off", "false", "disabled"];
    const ON: [&str; 6] = ["low", "medium", "high", "on", "true", "enabled"];

    if let Some(effort) = &req.reasoning_effort {
        let e = effort.trim().to_ascii_lowercase();
        if OFF.contains(&e.as_str()) {
            return false;
        }
        if ON.contains(&e.as_str()) {
            return true;
        }
    }
    for obj in [&req.reasoning, &req.thinking].into_iter().flatten() {
        if let Some(inner) = obj
            .get("effort")
            .or_else(|| obj.get("enabled"))
            .or_else(|| obj.get("type"))
        {
            if let Some(s) = inner.as_str() {
                let s = s.trim().to_ascii_lowercase();
                if OFF.contains(&s.as_str()) {
                    return false;
                }
                if ON.contains(&s.as_str()) || s.starts_with("enabled") {
                    return true;
                }
            } else if let Some(b) = inner.as_bool() {
                return b;
            }
        }
    }
    default
}

/// Client-advertised session identity headers, mirroring the Python proxy's
/// `_SESSION_HEADERS`. pi sends `X-Session-Id`/`session_id`, opencode sends
/// `x-session-affinity`. Their session UUID survives compaction, which the
/// first-message hash does not.
const SESSION_HEADERS: [&str; 4] = [
    "x-session-affinity",
    "x-opencode-session",
    "x-session-id",
    "session_id",
];

/// Same sentinel the Python proxy uses to spot Hermes' session-title calls.
const TITLE_PROMPT_MARKER: &str = "You name chat sessions";

fn sha256_hex(prefix: &str, seed: &str, user: Option<&str>) -> String {
    let mut payload = String::new();
    if let Some(user) = user.filter(|u| !u.is_empty()) {
        payload.push_str(user);
        payload.push('\0');
    }
    payload.push_str(seed);
    let digest = Sha256::digest(payload.as_bytes());
    let hexed = hex::encode(digest);
    format!("{prefix}{}", &hexed[..24])
}

/// Title-generation calls inherit the parent session's headers, so they must
/// get their own namespace or their responses cross with the real chat.
fn is_title_request(req: &ChatRequest) -> bool {
    req.messages
        .iter()
        .filter(|m| m.role == "system" || m.role == "developer")
        .any(|m| content_text(&m.content).starts_with(TITLE_PROMPT_MARKER))
}

fn header_value(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// Resolve the conversation key for a request.
///
/// Precedence mirrors the Python proxy: explicit `thread_id` body field, then
/// the `x-thread-id` header, then the client session id (hashed, so a stable
/// UUID keeps one DeepSeek conversation across compaction), and only then a
/// hash of the messages. The old message-only hash stranded a conversation on
/// a fresh remote session every time the first message changed.
fn thread_key(headers: &HeaderMap, req: &ChatRequest) -> String {
    if let Some(v) = req
        .thread_id
        .as_deref()
        .map(str::trim)
        .filter(|v| !v.is_empty())
    {
        return v.to_string();
    }
    if let Some(v) = header_value(headers, "x-thread-id") {
        return v;
    }
    // Title calls must be detected before the session id is consulted.
    let prefix = if is_title_request(req) {
        "title_"
    } else {
        "thread_"
    };
    if let Some(session) = SESSION_HEADERS
        .iter()
        .find_map(|name| header_value(headers, name))
    {
        return sha256_hex(prefix, &session, None);
    }
    if let Some(first) = req
        .messages
        .iter()
        .find(|m| m.role == "user" && !content_text(&m.content).is_empty())
    {
        return sha256_hex(prefix, &content_text(&first.content), None);
    }
    let all: String = req
        .messages
        .iter()
        .map(|m| content_text(&m.content))
        .collect();
    sha256_hex(prefix, &all, None)
}

/// Label for one message: `user`, `assistant`, `system`, or `tool <name>`.
/// Render one message the way `prompts.py::build_prompt` does.
///
/// The role prefixes are byte-identical to Python's `[prompts]` table: the
/// model was anchored on this framing, so `[user]:` instead of `User: ` is a
/// real behavioural difference, not cosmetics.
fn render_message(m: &Msg) -> Option<String> {
    let body = content_text(&m.content);
    match m.role.as_str() {
        "system" | "developer" => {
            if body.is_empty() {
                return None;
            }
            Some(format!("System: {body}"))
        }
        "user" => {
            if body.is_empty() {
                return None;
            }
            Some(format!("User: {body}"))
        }
        "assistant" => {
            if m.tool_calls.is_empty() {
                if body.is_empty() {
                    return None;
                }
                return Some(format!("Assistant: {body}"));
            }
            let calls = dialect::format_tool_calls_for_history(&m.tool_calls);
            if calls.is_empty() {
                return if body.is_empty() {
                    None
                } else {
                    Some(format!("Assistant: {body}"))
                };
            }
            // Python emits the prose and the calls as two parts.
            Some(if body.is_empty() {
                format!("Assistant:\n{calls}")
            } else {
                format!("Assistant: {body}\n\nAssistant:\n{calls}")
            })
        }
        "tool" => {
            let id = m.tool_call_id.as_deref().unwrap_or("");
            Some(format!("Tool result (id={id}):\n{body}"))
        }
        _ => {
            if body.is_empty() {
                None
            } else {
                Some(format!("{}: {body}", m.role))
            }
        }
    }
}

/// Python's `CONTINUE_AFTER_TOOL`, appended when a turn ends on a tool result.
const CONTINUE_AFTER_TOOL: &str = "Now continue with the task based on the tool result above.";

/// Fresh thread: flatten the whole conversation, then the continuation line
/// when the last message is a tool result (Python's `last_role == "tool"`).
fn flatten_transcript(messages: &[Msg]) -> String {
    let blocks: Vec<String> = messages.iter().filter_map(render_message).collect();
    let mut out = blocks.join("\n\n");
    if last_non_system_role(messages) == Some("tool") {
        out.push_str("\n\n");
        out.push_str(CONTINUE_AFTER_TOOL);
    }
    out
}

/// Messages since the last assistant reply, i.e. the turn's new input.
///
/// Port of `prompts.get_new_messages()`: everything after the last assistant
/// message. Both the prompt and the image intake read from this one slice, so
/// an attached image can never belong to a different turn than the text.
fn delta_messages(messages: &[Msg]) -> &[Msg] {
    let start = messages
        .iter()
        .rposition(|m| m.role == "assistant")
        .map(|i| i + 1)
        .unwrap_or(0);
    &messages[start..]
}

fn last_non_system_role(messages: &[Msg]) -> Option<&str> {
    messages
        .iter()
        .rev()
        .find(|m| m.role != "system" && m.role != "developer")
        .map(|m| m.role.as_str())
}

/// Build the continuation prompt for a thread that already exists.
///
/// Port of Python's `build_prompt(new_messages, ...)` where `new_messages` is
/// `get_new_messages()`: every delta message is rendered with its role prefix,
/// in order, and `CONTINUE_AFTER_TOOL` is appended when the delta ends on a
/// tool result. Rendering the whole delta (rather than just the newest user
/// message) is what makes the tool round trip work: the `role: "tool"`
/// messages carry the results the model's call returned.
fn continuation_prompt(delta: &[Msg]) -> Option<String> {
    let blocks: Vec<String> = delta.iter().filter_map(render_message).collect();
    if blocks.is_empty() {
        return None;
    }
    let mut out = blocks.join("\n\n");
    if last_non_system_role(delta) == Some("tool") {
        out.push_str("\n\n");
        out.push_str(CONTINUE_AFTER_TOOL);
    }
    Some(out)
}

async fn healthz() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

/// Python's `/readyz`: login is lazy, so being up is being ready.
async fn readyz() -> Json<Value> {
    Json(json!({"status": "ready", "backend": "lazy_login"}))
}

/// `GET /stats` — everything `deepseek-rs stats` and `status` need.
async fn stats(State(state): State<AppState>) -> Json<Value> {
    let s = &state.stats;
    let counts = state.store.counts().unwrap_or_default();
    let threads_in_memory = state.threads.lock().unwrap().len();

    let accounts: Vec<Value> = state
        .pool
        .snapshot()
        .await
        .into_iter()
        .map(|a| {
            json!({
                "email": a.email,
                "healthy": a.healthy,
                "bound_threads": a.bound_threads,
                "waiting": a.waiting,
                "cooldown_ms": a.cooldown_ms,
            })
        })
        .collect();

    Json(json!({
        "pid": std::process::id(),
        "uptime_secs": s.uptime_secs(),
        "model": state.settings.model_name,
        "thinking": state.settings.thinking_default,
        "search": state.settings.search_enabled,
        "requests": {
            "total": s.total.load(Ordering::Relaxed),
            "ok": s.ok.load(Ordering::Relaxed),
            "errors": s.errors.load(Ordering::Relaxed),
            "auth_failed": s.auth_failed.load(Ordering::Relaxed),
            "retries": s.retries.load(Ordering::Relaxed),
            "tool_calls": s.tool_calls.load(Ordering::Relaxed),
            "images_dropped": s.images_dropped.load(Ordering::Relaxed),
        },
        "threads": { "in_memory": threads_in_memory },
        "db": {
            "threads": counts.threads,
            "bindings": counts.bindings,
            "api_keys": counts.api_keys,
            "accounts": counts.accounts,
            "cached_tokens": counts.cached_tokens,
        },
        "accounts": accounts,
    }))
}

/// `GET /v1/models` — the same three ids Python advertises, in the same shape.
async fn models(_state: State<AppState>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [
            {"id": "DEFAULT", "object": "model", "created": 0, "owned_by": "deepseek"},
            {"id": "EXPERT", "object": "model", "created": 0, "owned_by": "deepseek"},
            {"id": "VISION", "object": "model", "created": 0, "owned_by": "deepseek"},
        ]
    }))
}

/// Reject the request unless the bearer key is known (or none are registered).
fn authorize(state: &AppState, headers: &HeaderMap) -> Result<(), Response> {
    let presented = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .map(|v| v.trim_start_matches("Bearer ").trim().to_string())
        .unwrap_or_default();

    match state.store.verify_api_key(&presented) {
        Ok(true) => Ok(()),
        // `require_api_key` raises FastAPI's HTTPException, whose body is
        // `{"detail": ...}` — not the OpenAI error envelope.
        Ok(false) => {
            let detail = if presented.is_empty() {
                "missing api key"
            } else {
                "invalid api key"
            };
            Err((
                axum::http::StatusCode::UNAUTHORIZED,
                Json(json!({ "detail": detail })),
            )
                .into_response())
        }
        Err(e) => Err(backend_error(&format!("store: {e}"))),
    }
}

/// Why a completion attempt ultimately failed.
#[derive(Debug, Clone)]
enum CompletionError {
    /// Still throttled after every retry; maps to 429 `rate_limit_exceeded`.
    RateLimited {
        retry_after_s: u64,
    },
    /// Login is dead; maps to 502 `backend_auth_failed`.
    LoginDead(String),
    /// Queue full or queue wait timed out; maps to 503 `queue_unavailable`.
    QueueUnavailable(String),
    Other(String),
}

impl std::fmt::Display for CompletionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompletionError::RateLimited { retry_after_s } => write!(
                f,
                "DeepSeek backend rate-limited this turn (retry after {retry_after_s}s)"
            ),
            CompletionError::LoginDead(msg) | CompletionError::Other(msg) => f.write_str(msg),
            CompletionError::QueueUnavailable(msg) => f.write_str(msg),
        }
    }
}

/// Public API error. Port of `ProxyError` in `deepseek_proxy/errors.py`.
///
/// Backend detail stays in the logs; the client gets a stable code and a
/// generic message, exactly as Python does.
#[derive(Debug, Clone)]
struct ProxyError {
    message: String,
    status: axum::http::StatusCode,
    code: &'static str,
    param: Option<String>,
    retry_after: Option<u64>,
}

impl ProxyError {
    fn new(message: impl Into<String>, status: axum::http::StatusCode, code: &'static str) -> Self {
        Self {
            message: message.into(),
            status,
            code,
            param: None,
            retry_after: None,
        }
    }

    fn with_param(mut self, param: impl Into<String>) -> Self {
        self.param = Some(param.into());
        self
    }

    /// `ProxyError.payload()`: `type` follows the status, not the code.
    fn payload(&self) -> Value {
        json!({
            "error": {
                "message": self.message,
                "type": if self.status == axum::http::StatusCode::BAD_REQUEST {
                    "invalid_request_error"
                } else {
                    "server_error"
                },
                "param": self.param,
                "code": self.code,
            }
        })
    }

    fn into_response(self) -> Response {
        let retry_after = self.retry_after;
        let mut resp = (self.status, Json(self.payload())).into_response();
        // `app.py` sets Retry-After whenever the exception carries one.
        if let Some(retry_after) = retry_after {
            if let Ok(v) = axum::http::HeaderValue::from_str(&retry_after.max(1).to_string()) {
                resp.headers_mut().insert("retry-after", v);
            }
        }
        resp
    }
}

/// Port of `error_for()` in `deepseek_proxy/transport.py`: the single place
/// that turns a backend failure into what the client sees and how it is logged.
async fn error_for(state: &AppState, key: &str, err: CompletionError) -> ProxyError {
    match err {
        CompletionError::RateLimited { retry_after_s } => {
            let mut e = ProxyError::new(
                "Backend capacity temporarily unavailable",
                axum::http::StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_exceeded",
            );
            e.retry_after = Some(retry_after_s.max(1));
            e
        }
        CompletionError::LoginDead(msg) => {
            tracing::error!(thread = %key, error = %msg, "backend login failed");
            let _ = state.pool.failover(key).await;
            ProxyError::new(
                "Backend login failed; disabled account, retry with a fresh thread",
                axum::http::StatusCode::BAD_GATEWAY,
                "backend_auth_failed",
            )
        }
        CompletionError::QueueUnavailable(msg) => {
            tracing::warn!(thread = %key, error = %msg, "account queue unavailable");
            ProxyError::new(
                "Request queue unavailable; try again later",
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                "queue_unavailable",
            )
        }
        CompletionError::Other(msg) => {
            tracing::error!(thread = %key, error = %msg, "backend request failed");
            ProxyError::new(
                "Backend request failed",
                axum::http::StatusCode::BAD_GATEWAY,
                "backend_error",
            )
        }
    }
}

/// Send the completion, retrying throttles with linear backoff.
///
/// Port of `Backend.generate()` in `deepseek_proxy/backend.py`. Three details
/// are deliberately identical:
///
/// - only throttle errors retry; anything else is returned immediately;
/// - the wait is `backoff * min(attempt + 1, max(1, max_retries))`, not an
///   exponential doubling;
/// - the wait is applied by cooling the account down, so `acquire()` sleeps.
///   A second timer would race the gate and fire the retry early.
///
/// Only pre-first-byte failures are retried, which is safe for both stream and
/// non-stream callers; a mid-stream failure is surfaced as-is.
async fn completion_with_retry(
    state: &AppState,
    thread_id: &str,
    body: &CompletionBody,
) -> Result<(reqwest::Response, pool::RateTicket), CompletionError> {
    let max_retries = state.settings.rate_limit_max_retries;
    let backoff = state.settings.rate_limit_backoff_secs.max(0.5);
    let mut attempt = 0u32;
    let mut conn = state.pool.route(thread_id).await;

    loop {
        // The slot is held for the WHOLE stream, not just until headers land:
        // the caller keeps this ticket alive until the body is drained, so the
        // next request on this account starts `request_delay` after the stream
        // really finished rather than after it merely started.
        let slot = conn.acquire().await.map_err(|e| {
            // `AccountQueueFull` / `AccountQueueTimeout` map to 503 in Python.
            if e.contains("queue") {
                CompletionError::QueueUnavailable(e)
            } else {
                CompletionError::Other(e)
            }
        })?;
        let client = conn.client().await;

        match client.stream_completion(body).await {
            Ok(resp) => return Ok((resp, slot)),
            Err(e) => {
                let last = e.to_string();

                if pool::is_login_dead(&last) {
                    tracing::warn!("account {} login-dead: {last}", conn.email);
                    drop(slot);
                    match state.pool.failover(thread_id).await {
                        Some(next) => {
                            tracing::warn!("thread failed over to {}", next.email);
                            conn = next;
                            continue;
                        }
                        // No healthy account left: this is a login failure, not a
                        // generic backend error, and `error_for()` says so.
                        None => return Err(CompletionError::LoginDead(last)),
                    }
                }

                if pool::is_invalid_token(&last) && state.pool.recover(&conn).await {
                    tracing::info!("recovered account {}", conn.email);
                    drop(slot);
                    continue;
                }

                if !pool::is_rate_limit(&last) {
                    drop(slot);
                    return Err(CompletionError::Other(last));
                }

                // Python: backoff * min(attempt + 1, max(1, max_retries)).
                let step = (attempt + 1).min(max_retries.max(1)) as f64;
                let wait = backoff * step;
                drop(slot);
                conn.cool_down(std::time::Duration::from_secs_f64(wait))
                    .await;

                if attempt < max_retries {
                    attempt += 1;
                    state.stats.retries.fetch_add(1, Ordering::Relaxed);
                    tracing::info!(
                        thread = %thread_id,
                        attempt,
                        retry_delay_s = wait,
                        "retrying after transient backend error"
                    );
                    continue;
                }

                state.stats.errors.fetch_add(1, Ordering::Relaxed);
                return Err(CompletionError::RateLimited {
                    retry_after_s: wait.ceil() as u64,
                });
            }
        }
    }
}

/// Run correction turns for malformed tool calls.
///
/// Port of the repair loop in `deepseek_proxy/service.py`, including the two
/// subtleties that took a bug to find there:
///
/// - a repaired call that repeats an already-accepted call is dropped, because
///   the model tends to echo the block it was just told about;
/// - an attempt that resolves nothing records ONE `repair_omitted_call` marker
///   instead of re-listing the same failures. Re-listing grows `expected` every
///   attempt until the retry budget is gone.
///
/// Repair turns are non-streaming and never carry thinking/search, matching
/// `turn.generate(repair_prompt, stream=False)`.
/// Port of `deepseek_proxy/tools/recovery.py::Recovery`.
///
/// The validation and recovery policy is not "does the block parse?" — every
/// call is checked against the offered tool's JSON Schema before it is
/// accepted, on the first turn as well as during repair. A call that names an
/// unknown tool, whose arguments are not a JSON object, or whose arguments do
/// not match the schema is converted into a repair failure instead of being
/// forwarded to the client.
struct Recovery {
    specs: Vec<dialect::ToolSpec>,
    calls: Vec<dialect::ToolCall>,
    pending: Vec<dialect::MalformedCall>,
    required: bool,
    parallel: bool,
    previous: HashMap<(String, String), usize>,
    repair_seen: HashMap<(String, String), usize>,
    expected: usize,
    resolved: usize,
    repairing: bool,
}

impl Recovery {
    fn new(specs: Vec<dialect::ToolSpec>, required: bool, parallel: bool) -> Self {
        Self {
            specs,
            calls: Vec::new(),
            pending: Vec::new(),
            required,
            parallel,
            previous: HashMap::new(),
            repair_seen: HashMap::new(),
            expected: 0,
            resolved: 0,
            repairing: false,
        }
    }

    /// Feed one event; returns the calls it produced (zero or one).
    ///
    /// Mirrors `Recovery.accept()`: malformed events join `pending`, text is
    /// ignored, and a parsed call is schema-validated before acceptance.
    fn accept(
        &mut self,
        event: dialect::ScanEvent,
    ) -> Result<Option<dialect::ToolCall>, ProxyError> {
        let call = match event {
            dialect::ScanEvent::Malformed(m) => {
                self.pending.push(m);
                return Ok(None);
            }
            dialect::ScanEvent::Text(_) => return Ok(None),
            dialect::ScanEvent::Tool(c) => c,
        };

        if let Some(reason) = dialect::validation_reason(&self.specs, &call) {
            self.pending.push(dialect::MalformedCall::new(
                Some(call.name.clone()),
                call.arguments.clone(),
                reason,
            ));
            return Ok(None);
        }

        let key = dialect::fingerprint(&call);
        if self.repairing {
            let seen = self.repair_seen.entry(key.clone()).or_insert(0);
            *seen += 1;
            let before = self.previous.get(&key).copied().unwrap_or(0);
            if *seen <= before {
                // The repair echoed a previously accepted call.
                return Ok(None);
            }
        }
        if !self.parallel && !self.calls.is_empty() {
            return Err(ProxyError::new(
                "Backend produced multiple calls with parallel_tool_calls=false",
                axum::http::StatusCode::BAD_GATEWAY,
                "tool_repair_failed",
            ));
        }
        self.calls.push(call.clone());
        self.resolved += 1;
        Ok(Some(call))
    }

    /// `tool_choice: "required"` must yield at least one call.
    fn require_call(&mut self) {
        if self.required && self.calls.is_empty() && self.pending.is_empty() {
            self.pending.push(dialect::MalformedCall::new(
                None,
                String::new(),
                "required_tool_call_missing",
            ));
        }
    }

    /// Port of `begin_repair()`: build the prompt and reset the per-attempt
    /// counters. `previous` snapshots every call accepted so far, so the next
    /// repair cannot re-accept an echo of one of them.
    fn begin_repair(&mut self) -> String {
        let mut prompt = dialect::build_repair_prompt(&self.pending, &self.specs);
        // Only the schema of a tool the model actually named and got wrong on
        // its arguments. Sending the whole catalogue here makes a repair turn
        // larger than the turn it repairs — in exactly the situation (context
        // pressure) where repair is most likely needed.
        let schemas = dialect::repair_schemas(&self.pending, &self.specs);
        if !schemas.is_empty() {
            prompt.push_str("\nSchema(s) for the tool call(s) that failed:\n");
            prompt.push_str(&dialect::py_json_dumps(&Value::Array(schemas)));
        }
        self.expected = self.pending.len();
        self.pending.clear();
        self.resolved = 0;
        self.previous.clear();
        for call in &self.calls {
            *self.previous.entry(dialect::fingerprint(call)).or_insert(0) += 1;
        }
        self.repair_seen.clear();
        self.repairing = true;
        prompt
    }

    /// Port of `end_repair()`: an attempt that resolved nothing records exactly
    /// one omission marker, never the whole failure list again.
    fn end_repair(&mut self) {
        if self.resolved < self.expected {
            self.pending.push(dialect::MalformedCall::new(
                None,
                String::new(),
                "repair_omitted_call",
            ));
        }
    }

    fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }
}

/// How `tool_choice` narrows the offered tools and sets the `required` flag.
///
/// Port of the filtering at the top of `ChatService.events()`:
/// `"none"` drops every tool, a named choice keeps only that function and
/// implies required, and `"required"` keeps all tools and implies required.
fn resolve_tools(req: &ChatRequest) -> (Vec<dialect::ToolSpec>, bool) {
    let mut tools = req.tools.clone();
    let mut required = false;
    match req.tool_choice.as_ref() {
        Some(Value::String(s)) if s == "none" => tools.clear(),
        Some(Value::String(s)) if s == "required" => required = true,
        Some(Value::Object(obj)) => {
            if let Some(name) = obj
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
            {
                tools.retain(|t| {
                    t.get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                        == Some(name)
                });
                required = true;
            }
        }
        _ => {}
    }
    (dialect::tool_specs(&tools), required)
}

/// Wait before retry number `attempt` (0-based).
///
/// Python: `backoff * min(attempt + 1, max(1, max_retries))`, with the base
/// floored at 0.5s by `Backend.generate()`.
fn backoff_delay(state: &AppState, attempt: u32) -> f64 {
    let max = state.settings.rate_limit_max_retries.max(1);
    state.settings.rate_limit_backoff_secs.max(0.5) * f64::from((attempt + 1).min(max))
}

/// HTTP 429 with a `retry-after` header, mirroring `BackendRateLimited`.
#[allow(dead_code)] // kept for callers that build the response directly
fn rate_limit_response(retry_after_s: u64, message: &str) -> Response {
    let mut e = ProxyError::new(
        message,
        axum::http::StatusCode::TOO_MANY_REQUESTS,
        "rate_limit_exceeded",
    );
    e.retry_after = Some(retry_after_s);
    e.into_response()
}

/// HTTP response for a failed completion attempt, via `error_for()`.
async fn completion_error_response(state: &AppState, key: &str, err: CompletionError) -> Response {
    error_for(state, key, err).await.into_response()
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Response {
    if let Err(resp) = authorize(&state, &headers) {
        state.stats.auth_failed.fetch_add(1, Ordering::Relaxed);
        tracing::warn!("rejected request: invalid api key");
        return resp;
    }

    let started = std::time::Instant::now();
    // `ChatService.validate()` runs before anything is queued: a malformed
    // request must not take the thread lock or create a remote session.
    if let Err(e) = validate_request(&req) {
        state.stats.errors.fetch_add(1, Ordering::Relaxed);
        return e.into_response();
    }
    let key = thread_key(&headers, &req);
    // Serialize turns on this thread. Taken BEFORE the parent id is read and
    // held until the turn (plus its correction turns) is finished, so a second
    // request on the same thread waits instead of forking the conversation.
    let turn = state.turn_guard(&key).await;
    let model = req
        .model
        .clone()
        .unwrap_or_else(|| state.settings.model_name.clone());

    // `ChatService.validate()`: hermes sends the public OpenAI id, the backend
    // uses internal names. Keep explicit expert/vision selection and fold the
    // common chat ids onto the default model; reject anything else.
    if !is_supported_model(&model) {
        state.stats.errors.fetch_add(1, Ordering::Relaxed);
        return ProxyError::new(
            "Unknown model; see /v1/models",
            axum::http::StatusCode::BAD_REQUEST,
            "model_not_found",
        )
        .with_param("model")
        .into_response();
    }
    tracing::info!(
        thread = %key,
        model = %model,
        stream = req.stream,
        messages = req.messages.len(),
        tools = req.tools.len(),
        "request in"
    );
    state.stats.total.fetch_add(1, Ordering::Relaxed);

    // In-memory cache first, then SQLite (survives restarts).
    let existing = state
        .threads
        .lock()
        .unwrap()
        .get(&key)
        .cloned()
        .or_else(|| {
            state
                .store
                .get_thread(&key)
                .ok()
                .flatten()
                .map(|r| ThreadState {
                    session_id: r.session_id,
                    parent_message_id: r.parent_message_id.and_then(|p| p.parse().ok()),
                    exchanges: 0,
                })
        });

    // Tool definitions are taught once, on the first turn of a thread. Later
    // turns ride on the server-side conversation, like the Python proxy.
    let system_extra = dialect::inject_tool_descriptions("", &req.tools);

    // `tool_choice` / `parallel_tool_calls` are part of the request contract:
    // "none" drops every tool, a named choice narrows to one function and
    // implies required, "required" keeps all tools and implies required.
    let (specs, required) = resolve_tools(&req);

    // Attach at most one image for this turn, exactly like Python's
    // `load_image(messages, IMAGE_MAX_BYTES)` + `upload_image()` pair: the
    // bytes are uploaded once and the completion references the resulting
    // file id through `ref_file_ids`. Only the delta is inspected, so an image
    // from an earlier turn is not uploaded again.
    //
    // This runs before the prompt is built because a turn carrying several
    // images now degrades (first sent, rest dropped) and the model has to be
    // told what it lost. Failing the turn with a 400 instead would wedge the
    // thread: the client retries the very same delta, so the rejection repeats
    // forever and the session never advances.
    let loaded = match images::load(delta_messages(&req.messages), state.settings.image_max_bytes).await {
        Ok(l) => l,
        Err(e) => {
            state.stats.errors.fetch_add(1, Ordering::Relaxed);
            tracing::warn!(thread = %key, error = %e, "rejected image");
            return ProxyError::new(e.0, axum::http::StatusCode::BAD_REQUEST, "invalid_image")
                .with_param("messages")
                .into_response();
        }
    };
    let image_note = loaded.note();
    if image_note.is_some() {
        state
            .stats
            .images_dropped
            .fetch_add(loaded.dropped as u64, Ordering::Relaxed);
        tracing::warn!(
            thread = %key,
            total = loaded.total,
            dropped = loaded.dropped,
            "dropped extra images; sending the first only"
        );
    }

    let (prompt, replay) = build_prompts(&req, existing.as_ref(), &system_extra, image_note.as_deref());
    if prompt.is_empty() {
        tracing::warn!(thread = %key, "rejecting turn: no user message or tool result");
        return bad_request("no user message or tool result in continuation turn");
    }
    tracing::debug!(
        thread = %key,
        fresh = existing.is_none(),
        prompt_chars = prompt.len(),
        replay_chars = replay.len(),
        "turn built"
    );
    // A brand-new thread has nothing to replay into, so a limit hit is final.
    let can_recover = existing.is_some() && !replay.is_empty();

    // Create the remote session lazily, on the first turn for this thread.
    let thread = match existing {
        Some(t) => t,
        None => {
            let conn = state.pool.route(&key).await;
            let session_id = {
                let _slot = match conn.acquire().await {
                    Ok(s) => s,
                    Err(e) => return queue_error(&format!("rate slot: {e}")),
                };
                let client = conn.client().await;
                match client.create_chat_session().await {
                    Ok(s) => s,
                    Err(e) => return backend_error(&format!("create_session: {e}")),
                }
            };
            let t = ThreadState {
                session_id,
                parent_message_id: None,
                exchanges: 0,
            };
            tracing::info!(thread = %key, session = %t.session_id, "new remote session");
            state.threads.lock().unwrap().insert(key.clone(), t.clone());
            t
        }
    };

    let ref_file_ids = match loaded.data {
        None => Vec::new(),
        Some(bytes) => {
            let conn = state.pool.route(&key).await;
            let _slot = match conn.acquire().await {
                Ok(s) => s,
                Err(e) => return queue_error(&format!("rate slot: {e}")),
            };
            match conn.client().await.upload_image(&bytes, &model).await {
                Ok(uploaded) => {
                    tracing::info!(
                        thread = %key,
                        file_id = %uploaded.file_id,
                        bytes = uploaded.size_bytes,
                        width = uploaded.width,
                        height = uploaded.height,
                        "image uploaded"
                    );
                    vec![uploaded.file_id]
                }
                // Python degrades to text-only rather than failing the turn.
                Err(e) => {
                    tracing::warn!(thread = %key, error = %e, "image upload failed; continuing text-only");
                    Vec::new()
                }
            }
        }
    };

    let thinking = want_thinking(&req, state.settings.thinking_default);
    let body = completion_body(
        &thread,
        &prompt,
        thinking,
        state.settings.search_enabled,
        &ref_file_ids,
    );

    let id = format!("chatcmpl-{}", uuid_like());
    let created = unix_secs();

    let (resp, slot) = match completion_with_retry(&state, &key, &body).await {
        Ok(r) => r,
        Err(e) => return completion_error_response(&state, &key, e).await,
    };

    tracing::debug!(thread = %key, thinking, parent = ?thread.parent_message_id, "upstream completion accepted");

    stream_response(
        turn,
        state,
        key,
        resp,
        slot,
        body,
        id,
        model,
        created,
        replay,
        can_recover,
        req.stream_options
            .as_ref()
            .and_then(|o| o.get("include_usage"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        started,
        specs,
        required,
        req.parallel_tool_calls,
    )
    .await
}

/// Result of draining one upstream SSE response into OpenAI chunks.
struct Pumped {
    parent: Option<i64>,
    filter: dialect::StreamToolFilter,
    /// True once any chunk reached the client; a hint after that is fatal.
    emitted: bool,
    /// Fatal `event: hint` payload, if the upstream aborted the stream.
    hint: Option<String>,
    /// Every text fragment seen, for the malformed-call detector. The filter
    /// keeps only parseable calls, so the raw text must be kept separately.
    text: String,
    /// Prose actually forwarded to the client (tool blocks excluded). Python's
    /// `Completion.text`; used for the `usage` completion-token count.
    prose: String,
    /// Upstream `accumulated_token_usage` (prompt side), when the stream
    /// reported one. Python reads the same field off its `done` event.
    prompt_tokens: u64,
}

/// Latest `accumulated_token_usage` anywhere in an event, if present.
///
/// The backend sends it twice: on the initial `response` object and as a
/// `BATCH` child patch (`{"p":"accumulated_token_usage","v":51}`).
fn find_token_usage(value: &Value) -> Option<u64> {
    if let Some(n) = value
        .get("accumulated_token_usage")
        .and_then(|v| v.as_u64())
    {
        return Some(n);
    }
    if value.get("p").and_then(|p| p.as_str()) == Some("accumulated_token_usage") {
        return value.get("v").and_then(|v| v.as_u64());
    }
    match value {
        Value::Object(map) => map.values().find_map(find_token_usage),
        Value::Array(items) => items.iter().find_map(find_token_usage),
        _ => None,
    }
}

/// Token estimate for a piece of text, matching Python's `estimate_tokens`
/// (tiktoken `cl100k_base`). Falls back to a 4-chars-per-token heuristic when
/// the encoding cannot be loaded.
///
/// The BPE table is loaded once and reused. `tiktoken_rs::cl100k_base()`
/// rebuilds it on every call (tens of milliseconds); this runs once per turn
/// on every completion, so the rebuild was pure waste.
fn estimate_tokens(text: &str) -> u64 {
    if text.is_empty() {
        return 0;
    }
    static BPE: std::sync::OnceLock<Option<tiktoken_rs::CoreBPE>> = std::sync::OnceLock::new();
    match BPE.get_or_init(|| tiktoken_rs::cl100k_base().ok()) {
        Some(bpe) => bpe.encode_with_special_tokens(text).len() as u64,
        None => (text.chars().count() as u64).div_ceil(4),
    }
}

/// OpenAI `usage` block.
///
/// Field provenance, verified against a live two-turn probe on one thread:
///
/// - `prompt_tokens` is forwarded verbatim from the backend's
///   `accumulated_token_usage`, the only token number DeepSeek sends. Measured
///   on the wire it is **prompt/input-side, not total**: a turn whose answer was
///   273 tokens was followed by a prompt that grew by only 26. So it maps to
///   `prompt_tokens` and must NOT be treated as a total (subtracting the
///   completion from it would under-count by a whole turn).
/// - `completion_tokens` has no upstream source at all — DeepSeek never reports
///   output tokens — so it is estimated locally with `cl100k_base`. This is the
///   one number that is self-counted, and it is approximate.
/// - `total_tokens` is derived from the two.
///
/// Note this deliberately counts *prose only*: reasoning is excluded, matching
/// Python's `Completion.usage()`. OpenAI would include it in `completion_tokens`.
fn usage_block(prompt_tokens: u64, text: &str, calls: &[dialect::ToolCall]) -> Value {
    let mut completion = estimate_tokens(text);
    for call in calls {
        completion += estimate_tokens(&format!("{}{}", call.name, call.arguments));
    }
    json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion,
        "total_tokens": prompt_tokens + completion,
    })
}

/// Read one upstream stream to completion, forwarding chunks to `tx`.
///
/// Two deadlines guard this, both from the Python proxy:
///
/// - `idle` (the heartbeat interval) bounds how long we wait for the *next*
///   byte, and emits `: keep-alive` while we wait. Python does this with
///   `asyncio.wait_for(queue.get(), heartbeat_interval)`.
/// - The caller bounds the whole turn with `request_timeout`.
async fn pump(
    resp: reqwest::Response,
    tx: &tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
    id: &str,
    model: &str,
    created: u64,
    idle: std::time::Duration,
    include_usage: bool,
) -> Pumped {
    let mut router = FragmentRouter::new();
    let mut events = Vec::new();
    let mut buf = String::new();
    let mut current_event = String::new();
    let mut parent: Option<i64> = None;
    let mut filter = dialect::StreamToolFilter::new();
    let mut emitted = false;
    let mut text_all = String::new();
    let mut prose = String::new();
    let mut prompt_tokens: u64 = 0;
    // Diagnostics for the "clean stream, no text" failure: the upstream can
    // answer with something we cannot route (an HTML/JSON error page on a 200,
    // or fragments in an unexpected shape). Without the raw bytes the only
    // visible symptom is "closed the stream without content", which says
    // nothing about which of those it was.
    let mut raw_bytes: usize = 0;
    let mut raw_events: usize = 0;
    let mut head = String::new();

    let mut byte_stream = resp.bytes_stream();
    loop {
        // Python's `stream()` waits `heartbeat_interval` for the next event and
        // emits `: keep-alive` on timeout — it does NOT kill the stream. The
        // turn-wide `request_timeout` is what bounds a real hang.
        let next = tokio::select! {
            biased;
            // The client hung up. Stop reading immediately: otherwise this task
            // stays parked on the socket forever, never releases the account's
            // rate slot, and every later request on this account blocks in
            // `acquire()` — silently, with no log line. Python gets the same
            // effect from `producer.cancel()` in the stream's `finally`.
            _ = tx.closed() => {
                tracing::info!(thread = %id, "client disconnected, stopping upstream read");
                return Pumped {
                    parent,
                    filter,
                    emitted,
                    hint: None,
                    text: text_all,
                    prose,
                    prompt_tokens,
                };
            }
            r = tokio::time::timeout(idle, byte_stream.next()) => r,
        };
        let chunk = match next {
            Ok(Some(chunk)) => chunk,
            Ok(None) => break,
            Err(_) => {
                if tx
                    .send(Ok(Event::default().comment("keep-alive")))
                    .await
                    .is_err()
                {
                    break;
                }
                continue;
            }
        };
        // A read error is a real failure. The old `let Ok(..) else { break }`
        // turned a dead socket into a clean, empty, *successful* stream — which
        // is why stalled turns produced no log line and no error at all.
        let bytes = match chunk {
            Ok(b) => b,
            Err(e) => {
                tracing::error!(thread = %id, error = %e, "upstream stream read failed");
                return Pumped {
                    parent,
                    filter,
                    emitted,
                    hint: Some(format!("stream read failed: {e}")),
                    text: text_all,
                    prose,
                    prompt_tokens,
                };
            }
        };
        raw_bytes += bytes.len();
        if head.len() < 1024 {
            head.push_str(&String::from_utf8_lossy(&bytes));
            if head.len() > 1024 {
                head.truncate(1024);
            }
        }
        buf.push_str(&String::from_utf8_lossy(&bytes));
        while let Some(pos) = buf.find('\n') {
            let line = buf[..pos].to_string();
            buf.drain(..=pos);
            let value = match protocol::classify_line(&line, &mut current_event) {
                protocol::LineKind::Ignored => continue,
                protocol::LineKind::HintError { content } => {
                    return Pumped {
                        parent,
                        filter,
                        emitted,
                        hint: Some(content),
                        text: text_all,
                        prose,
                        prompt_tokens,
                    };
                }
                protocol::LineKind::Event(value) => value,
            };
            // Last id wins. aiodeepseek's `ask_stream` keeps the final
            // non-None `message_id` it sees and parents the next turn to that,
            // so an early id must not win here either.
            if let Some(mid) = value.get("response_message_id").and_then(|v| v.as_i64()) {
                parent = Some(mid);
            }
            // `accumulated_token_usage` is patched mid-stream, once as a plain
            // field and again as a BATCH child. Keep the latest value; Python
            // reads it off the same field when building its `done` event.
            if let Some(n) = find_token_usage(&value) {
                prompt_tokens = n;
            }
            raw_events += 1;
            events.clear();
            route_event(&value, &mut router, &mut events);
            for ev in &events {
                let delta = match ev.channel {
                    Channel::Text => {
                        text_all.push_str(&ev.text);
                        // Tool blocks are held back and surfaced as tool_calls.
                        let safe = filter.feed(&ev.text);
                        if safe.is_empty() {
                            continue;
                        }
                        prose.push_str(&safe);
                        json!({ "content": safe })
                    }
                    Channel::Reasoning => json!({ "reasoning_content": ev.text }),
                };
                let mut chunk = json!({
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": null}]
                });
                // OpenAI: with `include_usage`, every content chunk carries
                // `usage: null`; the real numbers arrive in a final chunk.
                if include_usage {
                    chunk["usage"] = Value::Null;
                }
                emitted = true;
                if tx
                    .send(Ok(Event::default().data(chunk.to_string())))
                    .await
                    .is_err()
                {
                    return Pumped {
                        parent,
                        filter,
                        emitted,
                        hint: None,
                        text: text_all,
                        prose,
                        prompt_tokens,
                    };
                }
            }
        }
    }

    if text_all.is_empty() {
        tracing::error!(
            thread = %id,
            raw_bytes,
            raw_events,
            head = %head.replace('\n', "\\n"),
            "upstream stream ended with no routable text; raw body above"
        );
    }

    // A 200 response that is not an SSE stream at all (a JSON `{code,biz_msg}`
    // error body) used to fall through as "empty response", hiding the reason.
    // Surface the body as a hint so the caller can classify it: `code:26`
    // means the parent id is stale and the turn is retryable.
    let hint = if raw_events == 0 && !head.trim().is_empty() {
        Some(format!("non-SSE upstream body: {}", head.trim()))
    } else {
        None
    };

    Pumped {
        parent,
        filter,
        emitted,
        hint,
        text: text_all,
        prose,
        prompt_tokens,
    }
}

/// Send an OpenAI-shaped SSE error event and stop the stream.
///
/// The old behaviour was to log a fatal hint and then emit `finish_reason:
/// stop`, so clients saw an empty but *successful* completion. That is why the
/// errors never showed up anywhere.
/// Send a `ProxyError` payload as a final SSE event, then `[DONE]`.
///
/// Python's `stream()` does exactly this: `yield sse(event.payload())` followed
/// by `data: [DONE]`, so a mid-stream failure is still an error the client can
/// see rather than a clean `finish_reason: stop`.
async fn send_stream_error(
    tx: &tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
    err: ProxyError,
) {
    let _ = tx
        .send(Ok(Event::default().data(err.payload().to_string())))
        .await;
    let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
}

/// Save the conversation cursor so the next turn appends below the newest
/// message instead of branching off an older one.
fn persist_cursor(
    threads: &Mutex<HashMap<String, ThreadState>>,
    store: &store::Store,
    key: &str,
    session_id: &str,
    cursor: Option<i64>,
) {
    let Some(mid) = cursor else { return };
    if let Some(t) = threads.lock().unwrap().get_mut(key) {
        t.parent_message_id = Some(mid);
        t.session_id = session_id.to_string();
    }
    let _ = store.save_thread(
        key,
        &store::ThreadResume {
            session_id: session_id.to_string(),
            parent_message_id: Some(mid.to_string()),
        },
    );
}

/// Fail a turn without losing the cursor.
///
/// Every upstream turn leaves messages in the session, even when it errors. If
/// the cursor were not saved here, the next request would resume from the
/// previous message and DeepSeek would render it as an edit branching off
/// mid-thread — the symptom this replaces.
async fn fail_turn(
    state: &AppState,
    key: &str,
    session_id: &str,
    cursor: Option<i64>,
    tx: &tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
    err: ProxyError,
) {
    persist_cursor(&state.threads, &state.store, key, session_id, cursor);
    send_stream_error(tx, err).await;
}

#[allow(clippy::too_many_arguments)]
async fn stream_response(
    turn: tokio::sync::OwnedMutexGuard<()>,
    state: AppState,
    key: String,
    resp: reqwest::Response,
    slot: pool::RateTicket,
    body: CompletionBody,
    id: String,
    model: String,
    created: u64,
    replay: String,
    can_recover: bool,
    include_usage: bool,
    started: std::time::Instant,
    specs: Vec<dialect::ToolSpec>,
    required: bool,
    parallel: bool,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(32);

    tokio::spawn(async move {
        // Held for the whole turn, including every correction turn, so another
        // request on this thread cannot start until this one has committed its
        // parent id. Dropped automatically when the task ends.
        let _turn = turn;
        let mut attempt: u32 = 0;
        let mut context_recovered = false;
        let mut cursor_recovered = false;
        let mut body = body;
        // The conversation cursor, ported from aiodeepseek's
        // `Conversation._parent_message_id`: the id of the NEWEST upstream
        // message. DeepSeek renders a message sent under an older parent as an
        // edited/alternate sibling, so every reply we generate — the main turn
        // and every correction turn — must advance this, and the value saved at
        // the end of the turn must be the last one, never the first.
        let mut session_id = body.chat_session_id.clone();
        let mut cursor = body.parent_message_id;
        let mut next = Some((resp, slot));

        // Opening chunk: OpenAI always sends the assistant role first.
        let opening = json!({
            "id": id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
        });
        if tx
            .send(Ok(Event::default().data(opening.to_string())))
            .await
            .is_err()
        {
            return;
        }

        // Drain one upstream response, retrying while it is still safe.
        //
        // A fatal hint arrives BEFORE any content (aiodeepseek raises it out of
        // the stream iterator), so retrying the whole turn cannot duplicate
        // output the client already has.
        let pumped = loop {
            let (resp, slot) = next.take().expect("attempt in flight");
            // Hold the account's rate slot until the upstream body is drained.
            let idle = std::time::Duration::from_secs_f64(state.settings.heartbeat_interval_secs);
            // The whole turn is bounded, like `asyncio.timeout(REQUEST_TIMEOUT)`.
            // A stalled upstream becomes a visible error instead of a hang.
            let budget = std::time::Duration::from_secs_f64(state.settings.request_timeout_secs)
                .saturating_sub(started.elapsed());
            let pumped =
                match tokio::time::timeout(
                    budget,
                    pump(resp, &tx, &id, &model, created, idle, include_usage),
                )
                    .await
                {
                    Ok(p) => p,
                    Err(_) => {
                        state.stats.errors.fetch_add(1, Ordering::Relaxed);
                        tracing::error!(
                            thread = %key,
                            timeout_s = state.settings.request_timeout_secs,
                            "turn timed out waiting for upstream"
                        );
                        fail_turn(
                            &state,
                            &key,
                            &session_id,
                            cursor,
                            &tx,
                            ProxyError::new(
                                "Request processing timed out",
                                axum::http::StatusCode::GATEWAY_TIMEOUT,
                                "request_timeout",
                            ),
                        )
                        .await;
                        return;
                    }
                };
            drop(slot);

            // Advance the cursor only on a turn that actually succeeded.
            //
            // Port of aiodeepseek's `Conversation.ask_stream`, which assigns
            // `_parent_message_id = last_message_id` AFTER the stream loop
            // completes: when the stream raises (throttle, hint error) the
            // assignment never happens and the previous cursor is kept.
            //
            // Advancing on failure is what poisoned the cursor: a throttled
            // turn still carries a `response_message_id` for a message the
            // backend discarded, so the next turn parented to a message that
            // never existed and DeepSeek answered `{"code":26,"biz_msg":
            // "invalid message id"}`.
            let Some(hint) = pumped.hint.clone() else {
                if let Some(mid) = pumped.parent {
                    cursor = Some(mid);
                }
                break pumped;
            };

            // Failed turn: keep the cursor we came in with.
            cursor = body.parent_message_id;

            if protocol::is_invalid_message_id(&hint) && !pumped.emitted && !cursor_recovered {
                // `{"code":26,"biz_msg":"invalid message id"}`: the saved
                // cursor does not exist upstream. Message ids are scoped to a
                // remote session, so this means the parent was deleted or the
                // session itself is gone. Clearing the cursor and resending the
                // DELTA would keep the old session id and open a fresh branch at
                // its root: every earlier turn is silently lost and the model
                // answers as if the conversation just started. Rebuild the
                // session and replay the whole transcript instead.
                cursor_recovered = true;
                attempt += 1;
                state.stats.retries.fetch_add(1, Ordering::Relaxed);
                if can_recover {
                    tracing::warn!(
                        thread = %key,
                        hint = %hint,
                        "stale parent message id; rebuilding the session and replaying the transcript"
                    );
                    match rebuild_session(&state, &key, &hint).await {
                        Ok(fresh) => {
                            let thinking = body.thinking_enabled;
                            let search = body.search_enabled;
                            body = completion_body(
                                &fresh,
                                &replay,
                                thinking,
                                search,
                                &body.ref_file_ids,
                            );
                            session_id = fresh.session_id.clone();
                            cursor = None;
                        }
                        Err(_) => {
                            fail_turn(
                                &state,
                                &key,
                                &session_id,
                                cursor,
                                &tx,
                                ProxyError::new(
                                    "Backend request failed",
                                    axum::http::StatusCode::BAD_GATEWAY,
                                    "backend_error",
                                ),
                            )
                            .await;
                            return;
                        }
                    }
                } else {
                    // Nothing to replay (first turn): dropping the cursor is
                    // enough to recover.
                    tracing::warn!(
                        thread = %key,
                        hint = %hint,
                        "stale parent message id; retrying with a cleared cursor"
                    );
                    cursor = None;
                    body.parent_message_id = None;
                }
            } else if protocol::is_context_limit(&hint) {
                // Context overflow: rebuild the remote session and replay once.
                if context_recovered || !can_recover || pumped.emitted {
                    state.stats.errors.fetch_add(1, Ordering::Relaxed);
                    tracing::error!(thread = %key, "upstream hint error: {hint}");
                    fail_turn(
                        &state,
                        &key,
                        &session_id,
                        cursor,
                        &tx,
                        ProxyError::new(
                            hint.clone(),
                            axum::http::StatusCode::BAD_GATEWAY,
                            "context_length_exceeded",
                        ),
                    )
                    .await;
                    return;
                }
                context_recovered = true;
                match rebuild_session(&state, &key, &hint).await {
                    Ok(fresh) => {
                        let thinking = body.thinking_enabled;
                        let search = body.search_enabled;
                        body = completion_body(&fresh, &replay, thinking, search, &body.ref_file_ids);
                        // The rebuilt session is a different conversation: point
                        // the cursor at it, or the next turn parents into the
                        // session we just abandoned.
                        session_id = fresh.session_id.clone();
                        cursor = None;
                    }
                    Err(_) => {
                        fail_turn(
                            &state,
                            &key,
                            &session_id,
                            cursor,
                            &tx,
                            ProxyError::new(
                                "Backend request failed",
                                axum::http::StatusCode::BAD_GATEWAY,
                                "backend_error",
                            ),
                        )
                        .await;
                        return;
                    }
                }
            } else if pool::is_rate_limit(&hint) && !pumped.emitted {
                let wait = backoff_delay(&state, attempt);
                if attempt >= state.settings.rate_limit_max_retries {
                    state.stats.errors.fetch_add(1, Ordering::Relaxed);
                    tracing::error!(
                        thread = %key,
                        retries = attempt,
                        "upstream still throttled: {hint}"
                    );
                    fail_turn(
                        &state,
                        &key,
                        &session_id,
                        cursor,
                        &tx,
                        error_for(
                            &state,
                            &key,
                            CompletionError::RateLimited {
                                retry_after_s: wait.ceil() as u64,
                            },
                        )
                        .await,
                    )
                    .await;
                    return;
                }
                attempt += 1;
                state.stats.retries.fetch_add(1, Ordering::Relaxed);
                tracing::info!(
                    thread = %key,
                    attempt,
                    retry_delay_s = wait,
                    hint = %hint,
                    "retrying after backend throttle"
                );
                state
                    .pool
                    .route(&key)
                    .await
                    .cool_down(std::time::Duration::from_secs_f64(wait))
                    .await;
            } else {
                state.stats.errors.fetch_add(1, Ordering::Relaxed);
                tracing::error!(thread = %key, "upstream hint error: {hint}");
                let login_dead = pool::is_login_dead(&hint);
                let err = if login_dead {
                    CompletionError::LoginDead(hint.clone())
                } else {
                    CompletionError::Other(hint.clone())
                };
                fail_turn(&state, &key, &session_id, cursor, &tx, error_for(&state, &key, err).await)
                    .await;
                return;
            }

            match completion_with_retry(&state, &key, &body).await {
                Ok(v) => next = Some(v),
                Err(e) => {
                    fail_turn(&state, &key, &session_id, cursor, &tx, error_for(&state, &key, e).await)
                        .await;
                    return;
                }
            }
        };

        let Pumped {
            parent,
            mut filter,
            text,
            prose,
            prompt_tokens,
            ..
        } = pumped;

        // Truly empty = the upstream sent no text at all, no call, no hint.
        //
        // Do NOT test `emitted`: the stream filter holds text back while it
        // waits for a closing tag, so a perfectly normal reply that ends inside
        // a partial `<invoke` has `emitted == false`. Testing `emitted` here
        // (an earlier version did) failed those turns as "empty response" —
        // behaviour the Python proxy never had, because it flushes the buffer
        // and finishes normally.
        if text.trim().is_empty() && filter.calls.is_empty() {
            state.stats.errors.fetch_add(1, Ordering::Relaxed);
            tracing::error!(
                thread = %key,
                elapsed_ms = started.elapsed().as_millis() as u64,
                "upstream closed the stream without content or a hint"
            );
            fail_turn(
                &state,
                &key,
                &session_id,
                cursor,
                &tx,
                ProxyError::new(
                    "Backend returned an empty response",
                    axum::http::StatusCode::BAD_GATEWAY,
                    "empty_response",
                ),
            )
            .await;
            return;
        }

        // Validate what the model produced and run correction turns, exactly
        // like `Recovery` does after the stream is drained. The streamed calls
        // were parsed but NOT yet validated, so re-run them through the policy
        // here: an unknown tool or bad arguments must trigger a repair turn
        // rather than reach the client.
        // `scan(&text)` replays the whole reply in order, yielding text, valid
        // calls and malformed blocks. It supersedes `filter.calls` (parsed from
        // the same text) and the separate `extract_malformed` pass, so the
        // policy sees each event exactly once.
        filter.calls.clear();
        let mut recovery = Recovery::new(specs, required, parallel);
        for event in dialect::scan(&text) {
            match recovery.accept(event) {
                Ok(_) => {}
                Err(e) => {
                    state.stats.errors.fetch_add(1, Ordering::Relaxed);
                    fail_turn(&state, &key, &session_id, cursor, &tx, e).await;
                    return;
                }
            }
        }
        recovery.require_call();

        // Chain correction turns after the reply we just received: the cursor
        // already points at it, so the repair continues the conversation instead
        // of branching off the pre-turn message (DeepSeek renders siblings as
        // edited/alternate messages).

        let max = state.settings.tool_repair_max_retries;
        for attempt in 1..=max {
            if !recovery.has_pending() {
                break;
            }
            let prompt = recovery.begin_repair();
            let expected = recovery.expected;
            tracing::warn!(thread = %key, attempt, expected, "tool repair");

            // The repair goes to whichever session is live right now, parented
            // to the newest message — including a rebuilt one.
            let turn_state = ThreadState {
                session_id: session_id.clone(),
                parent_message_id: cursor,
                exchanges: 0,
            };
            let body = completion_body(&turn_state, &prompt, false, false, &[]);
            let raw = match completion_with_retry(&state, &key, &body).await {
                Ok((resp, _slot)) => match resp.text().await {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::error!("thread {key}: repair body failed: {e}");
                        break;
                    }
                },
                Err(e) => {
                    tracing::error!("thread {key}: repair completion failed: {e}");
                    break;
                }
            };
            let (_, repair_text, repair_mid) = route_all(&raw);
            // The correction answer is now the newest upstream message, so it —
            // not the malformed reply it fixed — becomes the parent of the
            // client's next turn. Saving `parent` here is what used to make
            // DeepSeek draw the next message as an edit of the broken answer.
            if let Some(mid) = repair_mid {
                cursor = Some(mid);
            }
            for event in dialect::scan(&repair_text) {
                match recovery.accept(event) {
                    Ok(_) => {}
                    Err(e) => {
                        state.stats.errors.fetch_add(1, Ordering::Relaxed);
                        fail_turn(&state, &key, &session_id, cursor, &tx, e).await;
                        return;
                    }
                }
            }
            recovery.end_repair();
            tracing::debug!(
                thread = %key,
                attempt,
                remaining = recovery.pending.len(),
                "tool repair settled"
            );
        }

        // Python raises here rather than returning a turn with no usable calls.
        if recovery.has_pending() {
            state.stats.errors.fetch_add(1, Ordering::Relaxed);
            fail_turn(
                &state,
                &key,
                &session_id,
                cursor,
                &tx,
                ProxyError::new(
                    "Could not recover a valid tool response",
                    axum::http::StatusCode::BAD_GATEWAY,
                    "tool_repair_failed",
                ),
            )
            .await;
            return;
        }
        filter.calls = recovery.calls;

        // Flush any held-back prose, then emit tool calls (if any).
        let tail = filter.finish();
        if !tail.is_empty() {
            let chunk = json!({
                "id": id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {"content": tail}, "finish_reason": null}]
            });
            let _ = tx.send(Ok(Event::default().data(chunk.to_string()))).await;
        }

        let finish_reason = if filter.calls.is_empty() {
            "stop"
        } else {
            let calls: Vec<Value> = filter
                .calls
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    json!({
                        "index": i,
                        "id": format!("call_{i}"),
                        "type": "function",
                        "function": { "name": c.name, "arguments": c.arguments }
                    })
                })
                .collect();
            let chunk = json!({
                "id": id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {"tool_calls": calls}, "finish_reason": null}]
            });
            let _ = tx.send(Ok(Event::default().data(chunk.to_string()))).await;
            "tool_calls"
        };

        state
            .stats
            .tool_calls
            .fetch_add(filter.calls.len() as u64, Ordering::Relaxed);
        state.stats.ok.fetch_add(1, Ordering::Relaxed);
        tracing::info!(
            thread = %key,
            finish_reason,
            tool_calls = filter.calls.len(),
            parent = ?parent,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "stream done"
        );

        let done = json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]
        });
        let _ = tx.send(Ok(Event::default().data(done.to_string()))).await;
        // With `include_usage`, OpenAI sends one last chunk with no choices and
        // the token counts, immediately before `[DONE]`.
        if include_usage {
            let usage = usage_block(prompt_tokens, &prose, &filter.calls);
            let chunk = json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": usage
            });
            let _ = tx.send(Ok(Event::default().data(chunk.to_string()))).await;
        }
        let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;

        persist_cursor(&state.threads, &state.store, &key, &session_id, cursor);
    });

    Sse::new(ReceiverStream::new(rx)).into_response()
}

fn route_all(raw: &str) -> (String, String, Option<i64>) {
    let events = protocol::route_stream(raw);
    let mut reasoning = String::new();
    let mut answer = String::new();
    for ev in events {
        match ev.channel {
            Channel::Text => answer.push_str(&ev.text),
            Channel::Reasoning => reasoning.push_str(&ev.text),
        }
    }
    let mid = raw
        .lines()
        .filter_map(|l| l.trim().strip_prefix("data:"))
        .filter_map(|p| serde_json::from_str::<Value>(p.trim()).ok())
        .filter_map(|v| v.get("response_message_id").and_then(|x| x.as_i64()))
        .last();
    (reasoning, answer, mid)
}

fn uuid_like() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let n = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{n:x}")
}

fn unix_secs() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Port of `ChatService.validate()`'s request-shape checks (the pydantic model
/// plus the tool-schema pass). Everything here is a client error: the handler
/// returns it before touching the pool or the thread lock.
fn validate_request(req: &ChatRequest) -> Result<(), ProxyError> {
    // `model` is optional here: the handler defaults it from config, which the
    // Python proxy does not allow but this port documents deliberately.
    if req.messages.is_empty() {
        return Err(invalid("messages", "List should have at least 1 item"));
    }
    if let Some(id) = &req.thread_id {
        if id.is_empty() || id.chars().count() > 256 {
            return Err(invalid("thread_id", "Invalid thread id"));
        }
    }
    // `tool_choice`: only auto/none/required or a named-function object.
    if let Some(choice) = &req.tool_choice {
        let ok = match choice {
            Value::String(s) => matches!(s.as_str(), "auto" | "none" | "required"),
            Value::Object(obj) => obj
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .is_some_and(|n| !n.is_empty()),
            _ => false,
        };
        if !ok {
            return Err(invalid("tool_choice", "Input should be a valid tool choice"));
        }
    }
    for (i, m) in req.messages.iter().enumerate() {
        let at = |field: &str| format!("messages.{i}.{field}");
        if m.role == "tool" && m.tool_call_id.as_deref().unwrap_or_default().is_empty() {
            return Err(invalid(
                &at("tool_call_id"),
                "Tool messages require tool_call_id",
            ));
        }
        // Assistant messages may carry only tool_calls; every other role needs content.
        let content_missing = match &m.content {
            None => true,
            Some(Value::String(s)) => s.is_empty(),
            Some(Value::Array(parts)) => parts.is_empty(),
            Some(_) => false,
        };
        if content_missing && !(m.role == "assistant" && !m.tool_calls.is_empty()) {
            return Err(invalid(
                &at("content"),
                "Message requires content or assistant tool_calls",
            ));
        }
        if let Some(Value::Array(parts)) = &m.content {
            for (j, part) in parts.iter().enumerate() {
                let ok = match part.get("type").and_then(|t| t.as_str()) {
                    Some("text") => part.get("text").and_then(|t| t.as_str()).is_some(),
                    Some("image_url") => {
                        m.role == "user"
                            && part
                                .get("image_url")
                                .and_then(|i| i.get("url"))
                                .and_then(|u| u.as_str())
                                .is_some()
                    }
                    _ => false,
                };
                if !ok {
                    return Err(invalid(
                        &at(&format!("content.{j}")),
                        "Unsupported or invalid message content part",
                    ));
                }
            }
        }
    }
    for tool in &req.tools {
        let Some(func) = tool.get("function") else {
            return Err(invalid("tools", "Invalid or unsupported tool schema"));
        };
        let name = func.get("name").and_then(|n| n.as_str()).unwrap_or_default();
        if name.is_empty() || name.chars().count() > 128 {
            return Err(invalid("tools", "Invalid or unsupported tool schema"));
        }
        let parameters = func
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| json!({}));
        if !parameters.is_object() {
            return Err(invalid("tools", "Invalid or unsupported tool schema"));
        }
        // Python rejects the schema itself, and refuses remote `$ref`s so no
        // network fetch can happen during validation.
        if has_external_ref(&parameters) || jsonschema::validator_for(&parameters).is_err() {
            return Err(invalid("tools", "Invalid or unsupported tool schema"));
        }
    }
    Ok(())
}

fn invalid(param: &str, message: &str) -> ProxyError {
    ProxyError::new(message, axum::http::StatusCode::BAD_REQUEST, "invalid_request")
        .with_param(param)
}

/// True when the schema contains a `$ref`/`$dynamicRef` that points outside the
/// document. Port of the recursive check in `ChatService.validate()`.
fn has_external_ref(node: &Value) -> bool {
    match node {
        Value::Object(map) => map.iter().any(|(k, v)| {
            (k == "$ref" || k == "$dynamicRef")
                .then(|| !v.as_str().unwrap_or_default().starts_with('#'))
                .unwrap_or(false)
                || has_external_ref(v)
        }),
        Value::Array(items) => items.iter().any(has_external_ref),
        _ => false,
    }
}

/// `ChatService.validate()`'s supported-model set, case-insensitive.
fn is_supported_model(model: &str) -> bool {
    matches!(
        model.to_ascii_uppercase().as_str(),
        "DEFAULT" | "EXPERT" | "VISION" | "DEEPSEEK-CHAT" | "DEEPSEEK-V3"
    )
}

fn bad_request(msg: &str) -> Response {
    ProxyError::new(msg, axum::http::StatusCode::BAD_REQUEST, "invalid_request").into_response()
}

/// Backend failure. Port of `error_for()`'s fallback: the detail is logged and
/// the client gets the generic OpenAI envelope, never the raw message.
fn backend_error(msg: &str) -> Response {
    tracing::error!(error = %msg, "backend request failed");
    ProxyError::new(
        "Backend request failed",
        axum::http::StatusCode::BAD_GATEWAY,
        "backend_error",
    )
    .into_response()
}

/// Account queue saturated. Python raises `AccountQueueFull`/
/// `AccountQueueTimeout`, which `error_for()` maps to 503 `queue_unavailable`.
fn queue_error(msg: &str) -> Response {
    tracing::warn!(error = %msg, "account queue unavailable");
    ProxyError::new(
        "Request queue unavailable; try again later",
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        "queue_unavailable",
    )
    .into_response()
}

/// Build a connected account pool from resolved config.
pub(crate) async fn connect_pool(
    cfg: &config::Config,
    store: &store::Store,
    accounts: Vec<pool::AccountSpec>,
) -> Result<Arc<pool::Pool>, Box<dyn std::error::Error>> {
    let pool_config = pool::PoolConfig {
        request_delay: std::time::Duration::from_secs_f64(cfg.request_delay_secs),
        queue_limit: cfg.queue_limit,
        queue_timeout: std::time::Duration::from_secs_f64(cfg.queue_timeout_secs),
        login_cooldown: std::time::Duration::from_secs(cfg.login_cooldown_secs),
    };
    let p = pool::Pool::connect(accounts, pool_config, store.clone()).await?;
    Ok(Arc::new(p))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Rust ignores SIGPIPE by default, so `deepseek-rs stats | head` panics with
    // "failed printing to stdout: Broken pipe". Restore the usual Unix
    // behaviour: die quietly when the reader goes away. (No SIGPIPE on Windows.)
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }

    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("serve");

    // `serve` is the only long-running command; everything else is short-lived
    // and lives in `cli`.
    if cmd != "serve" {
        let code = cli::run(&args).await?;
        std::process::exit(code);
    }

    let profile = log::Profile::from_env();
    log::init(profile);

    // Claim the pidfile before doing any work, so a second `serve` fails fast
    // instead of fighting over the DB lock.
    let _pid_guard = cli::PidGuard::claim()?;

    let cfg = config::Config::load()?;
    let store = store::Store::open_path(&cfg.db_path)?;

    let accounts = cli::load_accounts(&store);
    if accounts.is_empty() {
        return Err(
            "no DeepSeek accounts configured — add one with `deepseek-rs accounts add <email> <password>`"
                .into(),
        );
    }

    tracing::info!(
        profile = ?profile,
        accounts = accounts.len(),
        request_delay_secs = cfg.request_delay_secs,
        "starting deepseek-rs"
    );
    tracing::info!(
        "connecting {} account(s), {}s between requests",
        accounts.len(),
        cfg.request_delay_secs
    );
    let pool = connect_pool(&cfg, &store, accounts).await?;
    tracing::info!("login ok for {} account(s)", pool.len());

    // Fail fast on a broken PoW path rather than on the first user request.
    {
        let conn = pool.any();
        let client = conn.client().await;
        client.pow_challenge(COMPLETION_PATH).await?;
        tracing::info!("pow challenge ok (account {})", conn.email);
    }

    let settings = Settings {
        thinking_default: cfg.thinking,
        search_enabled: cfg.search,
        model_name: cfg.model.clone(),
        rate_limit_max_retries: cfg.rate_limit_max_retries,
        rate_limit_backoff_secs: cfg.rate_limit_backoff_secs,
        tool_repair_max_retries: cfg.tool_repair_max_retries,
        request_timeout_secs: cfg.request_timeout_secs,
        image_max_bytes: cfg.image_max_bytes,
        heartbeat_interval_secs: cfg.heartbeat_interval_secs,
    };
    tracing::info!(
        db_path = %cfg.db_path,
        model = %settings.model_name,
        thinking = settings.thinking_default,
        search = settings.search_enabled,
        "settings loaded"
    );

    let state = AppState {
        pool,
        settings,
        threads: Arc::new(Mutex::new(HashMap::new())),
        turns: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        store,
        stats: Arc::new(Stats::new()),
    };

    // axum buffers the whole body before `Json<ChatRequest>` runs and refuses
    // anything over 2 MiB by default. A single 10 MiB image is ~13.3 MiB of
    // base64, so the default rejects perfectly valid turns with
    // "Failed to buffer the request body: length limit exceeded". The cap is
    // derived from IMAGE_MAX_BYTES, not a magic number.
    let body_limit = state.settings.image_max_bytes.saturating_mul(2).max(32 << 20);

    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/stats", get(stats))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
        // Must come AFTER the routes: `Router::layer` only wraps routes that
        // already exist, so placing it earlier silently does nothing.
        .layer(axum::extract::DefaultBodyLimit::max(body_limit));

    let addr = format!("{}:{}", cfg.host, cfg.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("listening on http://{addr}");

    // Graceful shutdown: finish in-flight requests, drop the pidfile.
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    tracing::info!("stopped");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    // Windows has no SIGTERM; Ctrl-C / Ctrl-Break are the shutdown triggers.
    #[cfg(unix)]
    {
        let term = async {
            use tokio::signal::unix::{signal, SignalKind};
            if let Ok(mut s) = signal(SignalKind::terminate()) {
                s.recv().await;
            }
        };
        tokio::select! {
            _ = ctrl_c => tracing::info!("SIGINT received, shutting down"),
            _ = term => tracing::info!("SIGTERM received, shutting down"),
        }
    }
    #[cfg(not(unix))]
    {
        ctrl_c.await;
        tracing::info!("Ctrl-C received, shutting down");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg_with(role: &str, content: &str) -> Msg {
        Msg {
            role: role.into(),
            content: Some(serde_json::json!(content)),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }

    fn thread(exchanges: usize) -> ThreadState {
        ThreadState {
            session_id: "s".into(),
            parent_message_id: None,
            exchanges,
        }
    }

    fn request(messages: Vec<Msg>, tools: Vec<serde_json::Value>) -> ChatRequest {
        ChatRequest {
            model: None,
            thread_id: None,
            messages,
            stream: false,
            reasoning_effort: None,
            thinking: None,
            reasoning: None,
            tools,
            tool_choice: None,
            parallel_tool_calls: true,
            stream_options: None,
        }
    }

    fn bash_tool() -> serde_json::Value {
        serde_json::json!({
            "type": "function",
            "function": {"name": "bash", "description": "run a command"}
        })
    }

    fn tool_with_schema() -> serde_json::Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "bash",
                "description": "run a command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"]
                }
            }
        })
    }

    #[test]
    /// One delta feeds both the prompt and the image intake, so they can never
    /// disagree about which turn they are describing (Python: `get_new_messages`).
    #[test]
    fn delta_is_everything_after_the_last_assistant() {
        let messages = vec![
            msg_with("user", "old"),
            msg_with("assistant", "reply"),
            msg_with("user", "new"),
        ];
        let delta = delta_messages(&messages);
        assert_eq!(delta.len(), 1);
        assert_eq!(content_text(&delta[0].content), "new");

        let (prompt, _) = build_prompts(&request(messages, vec![]), Some(&thread(1)), "", None);
        assert_eq!(prompt, "User: new");
    }

    /// Tool round trip: the delta carries the assistant's call and the tool
    /// results, and the continuation line is appended once.
    #[test]
    fn continuation_renders_whole_delta_and_continues_after_tool() {
        let mut call = msg_with("assistant", "");
        call.tool_calls = vec![serde_json::json!({
            "id": "call_0",
            "type": "function",
            "function": {"name": "bash", "arguments": "{}"}
        })];
        let mut result = msg_with("tool", "output");
        result.tool_call_id = Some("call_0".into());
        let messages = vec![msg_with("user", "run it"), call, result];

        let (prompt, _) = build_prompts(&request(messages, vec![]), Some(&thread(1)), "", None);
        assert!(prompt.contains("run it") || prompt.starts_with("Tool result"));
        assert!(prompt.contains("Tool result (id=call_0):\noutput"));
        assert_eq!(prompt.matches(CONTINUE_AFTER_TOOL).count(), 1);
    }

    #[test]
    fn validation_rejects_tool_message_without_id() {
        let mut m = msg_with("tool", "output");
        m.tool_call_id = None;
        let req = request(vec![m], vec![]);
        let e = validate_request(&req).unwrap_err();
        assert_eq!(e.status, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(e.code, "invalid_request");
        assert_eq!(e.param.as_deref(), Some("messages.0.tool_call_id"));
    }

    #[test]
    fn validation_rejects_empty_messages_and_unknown_tool_choice() {
        let mut req = request(vec![], vec![]);
        assert_eq!(validate_request(&req).unwrap_err().code, "invalid_request");

        let mut req = request(vec![msg_with("user", "hi")], vec![]);
        req.tool_choice = Some(serde_json::json!("banana"));
        assert_eq!(
            validate_request(&req).unwrap_err().param.as_deref(),
            Some("tool_choice")
        );
        for good in ["auto", "none", "required"] {
            req.tool_choice = Some(serde_json::json!(good));
            assert!(validate_request(&req).is_ok(), "{good} must be accepted");
        }
    }

    #[test]
    fn validation_rejects_invalid_tool_schema_and_external_ref() {
        let mut bad = tool_with_schema();
        bad["function"]["parameters"] = serde_json::json!({"type": 123});
        let req = request(vec![msg_with("user", "hi")], vec![bad]);
        assert_eq!(
            validate_request(&req).unwrap_err().param.as_deref(),
            Some("tools")
        );

        let mut remote = tool_with_schema();
        remote["function"]["parameters"] = serde_json::json!({
            "type": "object",
            "properties": {"a": {"$ref": "https://example.com/schema.json"}}
        });
        let req = request(vec![msg_with("user", "hi")], vec![remote]);
        assert_eq!(
            validate_request(&req).unwrap_err().param.as_deref(),
            Some("tools")
        );

        let req = request(vec![msg_with("user", "hi")], vec![tool_with_schema()]);
        assert!(validate_request(&req).is_ok());
    }

    #[test]
    fn validation_accepts_assistant_tool_calls_without_content() {
        let mut m = msg_with("assistant", "");
        m.tool_calls = vec![serde_json::json!({
            "id": "call_0", "type": "function",
            "function": {"name": "bash", "arguments": "{}"}
        })];
        let req = request(vec![m], vec![]);
        assert!(validate_request(&req).is_ok());
    }

    #[test]
    fn tool_choice_none_drops_tools_and_adds_directive() {
        let mut req = request(vec![msg_with("user", "hi")], vec![bash_tool()]);
        req.tool_choice = Some(serde_json::json!("none"));
        let (specs, required) = resolve_tools(&req);
        assert!(specs.is_empty());
        assert!(!required);
        let (delta, _) = build_prompts(&req, None, "", None);
        assert!(delta.contains("Do not call tools in this turn; reply with normal text."));
    }

    #[test]
    fn tool_choice_required_adds_must_call_directive() {
        let mut req = request(vec![msg_with("user", "hi")], vec![bash_tool()]);
        req.tool_choice = Some(serde_json::json!("required"));
        let (_, required) = resolve_tools(&req);
        assert!(required);
        let (delta, _) = build_prompts(&req, None, "", None);
        assert!(delta.contains("You must return a tool call."));
    }

    #[test]
    fn named_tool_choice_narrows_tools() {
        let mut req = request(vec![msg_with("user", "hi")], vec![bash_tool()]);
        req.tool_choice = Some(serde_json::json!({
            "type": "function",
            "function": {"name": "bash"}
        }));
        let (specs, required) = resolve_tools(&req);
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, "bash");
        assert!(required);
    }

    #[test]
    fn parallel_tool_calls_false_adds_directive() {
        let mut req = request(vec![msg_with("user", "hi")], vec![bash_tool()]);
        req.parallel_tool_calls = false;
        let (delta, _) = build_prompts(&req, None, "", None);
        assert!(delta.contains("Return at most one tool call."));
    }

    #[test]
    fn recovery_rejects_unknown_tool_and_dumps_schemas() {
        let specs = dialect::tool_specs(&[tool_with_schema()]);
        let mut recovery = Recovery::new(specs, false, true);
        let produced = recovery
            .accept(dialect::ScanEvent::Tool(dialect::ToolCall {
                name: "nope".into(),
                arguments: "{}".into(),
            }))
            .unwrap();
        assert!(produced.is_none());
        assert!(recovery.has_pending());
        let prompt = recovery.begin_repair();
        assert!(prompt.contains("unknown_or_disallowed_tool"));
        assert!(prompt.contains("Valid tool names: bash."));
    }

    #[test]
    fn recovery_required_call_missing_becomes_a_failure() {
        let specs = dialect::tool_specs(&[tool_with_schema()]);
        let mut recovery = Recovery::new(specs, true, true);
        recovery.require_call();
        assert!(recovery.has_pending());
        assert_eq!(recovery.pending[0].reason, "required_tool_call_missing");
    }

    #[test]
    fn recovery_rejects_parallel_calls_when_disabled() {
        let specs = dialect::tool_specs(&[tool_with_schema()]);
        let mut recovery = Recovery::new(specs, false, false);
        let ok = dialect::ScanEvent::Tool(dialect::ToolCall {
            name: "bash".into(),
            arguments: "{\"command\": \"ls\"}".into(),
        });
        assert!(recovery.accept(ok.clone()).unwrap().is_some());
        let second = recovery.accept(ok).unwrap_err();
        assert_eq!(second.code, "tool_repair_failed");
    }

    #[test]
    fn history_is_role_labelled() {
        let text = flatten_transcript(&[
            msg_with("system", "be nice"),
            msg_with("user", "hi"),
            msg_with("assistant", "hello"),
        ]);
        assert_eq!(text, "System: be nice\n\nUser: hi\n\nAssistant: hello");
    }

    #[test]
    fn tool_message_is_labelled_with_its_name() {
        let mut m = msg_with("tool", "exit 0");
        m.name = Some("bash".into());
        assert_eq!(render_message(&m).unwrap(), "Tool result (id=):\nexit 0");
    }

    #[test]
    fn assistant_tool_calls_are_reserialized() {
        let mut m = msg_with("assistant", "");
        m.tool_calls = vec![serde_json::json!({
            "type": "function",
            "function": {"name": "bash", "arguments": "{\"command\":\"ls\"}"}
        })];
        assert_eq!(
            render_message(&m).unwrap(),
            "Assistant:\n<invoke name=\"bash\">{\"command\":\"ls\"}</invoke>"
        );
    }

    #[test]
    fn reminder_is_appended_to_every_turn() {
        let tools = vec![bash_tool()];
        // Fresh thread: reminder goes after the transcript.
        let req = request(vec![msg_with("user", "hi")], tools.clone());
        let (delta, replay) = build_prompts(&req, None, "", None);
        assert!(delta.starts_with(&dialect::periodic_reminder()));
        assert!(delta.contains("User: hi"));
        assert!(delta.ends_with(&dialect::periodic_reminder()));
        assert!(replay.starts_with(&dialect::periodic_reminder()));
        assert!(replay.ends_with(&dialect::periodic_reminder()));

        // Continuation: still appended, still last.
        let req = request(vec![msg_with("user", "again")], tools);
        let (delta, _) = build_prompts(&req, Some(&thread(3)), "", None);
        assert!(delta.starts_with("User: again"));
        assert!(delta.ends_with(&dialect::periodic_reminder()));
    }

    #[test]
    fn no_tools_means_no_reminder() {
        let req = request(vec![msg_with("user", "hi")], Vec::new());
        let (delta, replay) = build_prompts(&req, None, "", None);
        assert_eq!(delta, "User: hi");
        assert_eq!(replay, "User: hi");
    }

    #[test]
    fn repair_prompt_carries_only_the_failed_tools_schema() {
        // A schema dump helps a correction turn learn the argument names for the
        // tool it already named. It must NOT include tools that did not fail:
        // that grows the repair prompt past the turn it repairs, in exactly the
        // context-pressure situation where repair is most likely needed.
        let other = serde_json::json!({
            "type": "function",
            "function": {
                "name": "search",
                "description": "look things up",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
            }
        });
        let specs = dialect::tool_specs(&[tool_with_schema(), other]);
        let mut recovery = Recovery::new(specs, false, true);
        recovery.pending.push(dialect::MalformedCall::new(
            Some("bash".into()),
            "{}".into(),
            "arguments_do_not_match_schema",
        ));
        let prompt = recovery.begin_repair();
        assert!(prompt.contains("Schema(s) for the tool call(s) that failed:"));
        assert!(prompt.contains("\"command\""), "schema must survive the dump");
        assert!(prompt.contains("required"), "required must survive the dump");
        assert!(!prompt.contains("\"search\""), "unfailed tools must not be dumped");
    }

    #[test]
    fn repair_prompt_sends_no_schema_for_unknown_or_missing_calls() {
        let specs = dialect::tool_specs(&[tool_with_schema()]);
        for (name, reason) in [
            (Some("nope".to_string()), "unknown_or_disallowed_tool"),
            (None, "required_tool_call_missing"),
            (None, "repair_omitted_call"),
            (Some("bash".to_string()), "unterminated_block"),
        ] {
            let mut recovery = Recovery::new(specs.clone(), false, true);
            recovery
                .pending
                .push(dialect::MalformedCall::new(name.clone(), String::new(), reason));
            let prompt = recovery.begin_repair();
            assert!(
                !prompt.contains("Schema(s) for the tool call(s) that failed:"),
                "{reason} must not trigger a schema dump"
            );
            // The failure is still named, and the valid names are still listed.
            assert!(prompt.contains(reason), "{reason} must be reported");
            assert!(prompt.contains("Valid tool names: bash."));
        }
    }

    /// The live path: real model text -> scan -> accept -> begin_repair.
    ///
    /// The hand-built `MalformedCall` tests passed while this was broken,
    /// because they used reason strings that never occur at runtime. A body
    /// that is not JSON, or is not an object, still names a real tool, so the
    /// repair turn must carry that tool's schema.
    #[test]
    fn wire_shape_failures_send_the_named_tools_schema() {
        let bad_json = concat!("<in", "voke name=\"bash\">oops</in", "voke>");
        let not_object = concat!("<in", "voke name=\"bash\">[1,2]</in", "voke>");
        for text in [bad_json, not_object] {
            let specs = dialect::tool_specs(&[tool_with_schema()]);
            let mut recovery = Recovery::new(specs, false, true);
            for event in dialect::scan(text) {
                recovery.accept(event).expect("malformed is not an error");
            }
            assert!(!recovery.pending.is_empty(), "{text} must reach repair");
            let prompt = recovery.begin_repair();
            assert!(
                prompt.contains("Schema(s) for the tool call(s) that failed:"),
                "{text} must carry a schema"
            );
            assert!(prompt.contains("\"command\""), "{text}: schema body missing");
        }
    }

    #[test]
    fn repair_schemas_dedupes_repeated_failures() {
        let specs = dialect::tool_specs(&[tool_with_schema()]);
        let bad = |name: &str| {
            dialect::MalformedCall::new(
                Some(name.into()),
                "{}".into(),
                "arguments_do_not_match_schema",
            )
        };
        let schemas = dialect::repair_schemas(&[bad("bash"), bad("bash")], &specs);
        assert_eq!(schemas.len(), 1);
    }

    #[test]
    fn cursor_saves_the_newest_message_not_the_first() {
        // The regression this guards: after a correction turn, DeepSeek must
        // see the client's next message as a child of the CORRECTED answer.
        // Saving the malformed reply's id made it a sibling, which the UI
        // draws as an edited message.
        let store = store::Store::open_memory().unwrap();
        let threads = Mutex::new(HashMap::new());
        threads.lock().unwrap().insert(
            "t1".to_string(),
            ThreadState {
                session_id: "s1".into(),
                parent_message_id: Some(100),
                exchanges: 0,
            },
        );

        persist_cursor(&threads, &store, "t1", "s1", Some(101));
        assert_eq!(
            threads.lock().unwrap()["t1"].parent_message_id,
            Some(101)
        );

        // A later correction advances it again.
        persist_cursor(&threads, &store, "t1", "s1", Some(102));
        assert_eq!(
            threads.lock().unwrap()["t1"].parent_message_id,
            Some(102)
        );
        assert_eq!(
            store
                .get_thread("t1")
                .unwrap()
                .unwrap()
                .parent_message_id
                .as_deref(),
            Some("102")
        );
    }

    #[test]
    fn cursor_follows_a_rebuilt_session() {
        // After a context-limit rebuild the cursor must point into the NEW
        // session, or the next turn parents into the abandoned one.
        let store = store::Store::open_memory().unwrap();
        let threads = Mutex::new(HashMap::new());
        threads.lock().unwrap().insert(
            "t1".to_string(),
            ThreadState {
                session_id: "old".into(),
                parent_message_id: Some(50),
                exchanges: 0,
            },
        );
        persist_cursor(&threads, &store, "t1", "fresh", Some(51));
        let t = threads.lock().unwrap();
        assert_eq!(t["t1"].session_id, "fresh");
        assert_eq!(t["t1"].parent_message_id, Some(51));
    }

    #[test]
    fn no_cursor_leaves_resume_state_untouched() {
        let store = store::Store::open_memory().unwrap();
        let threads = Mutex::new(HashMap::new());
        threads.lock().unwrap().insert(
            "t1".to_string(),
            ThreadState {
                session_id: "s1".into(),
                parent_message_id: Some(7),
                exchanges: 0,
            },
        );
        persist_cursor(&threads, &store, "t1", "s1", None);
        assert_eq!(threads.lock().unwrap()["t1"].parent_message_id, Some(7));
    }

    #[test]
    fn route_all_keeps_the_last_message_id() {
        // aiodeepseek's `ask_stream` keeps the final non-None message id; an
        // early one must not win, or the next turn branches.
        let raw = concat!(
            "data: {\"response_message_id\": 10}\n\n",
            "data: {\"response_message_id\": 11}\n\n",
        );
        assert_eq!(route_all(raw).2, Some(11));
    }

    #[test]
    fn reminder_text_names_the_failure_modes() {
        let text = dialect::periodic_reminder();
        println!("-----BEGIN-----\n{text}\n-----END-----");
        assert!(text.contains("DROPPED"));
        assert!(text.contains("Never emit the </invoke> ending tag twice."));
        assert!(text.contains(dialect::TEMPLATE));
    }
}
