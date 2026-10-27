//! OpenAI-compatible proxy over the DeepSeek private API.
//!
//! Rust port of `deepseek_proxy`. Serves:
//!
//! - `GET  /healthz`
//! - `GET  /v1/models`
//! - `POST /v1/chat/completions` (SSE streaming only; `stream: true` required)
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

mod cli;
mod config;
mod dialect;
mod history;
mod images;
mod log;
mod pool;
mod responses;
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
use chatgpt_client::{ChatGptClient, ClientConfig};
use client::{CompletionBody, COMPLETION_PATH};
use futures_util::StreamExt;
use protocol::{route_event, Channel, FragmentRouter};
use serde::{Deserialize, Serialize};
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
    /// Output budget per Responses channel id (`[channels.<id>]`).
    channel_max_output_tokens: std::collections::HashMap<String, u32>,
    /// Every request gets a brand-new remote session carrying only its own
    /// input. Off by default; when off nothing in this file changes behaviour.
    fresh_session_per_request: bool,
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
    responses: responses::Runtime,
    store: store::Store,
    stats: Arc<Stats>,
}

impl AppState {
    /// Take the turn lock for `key`, creating it on first use.
    ///
    /// The guard is held from before the parent id is read until the whole turn
    /// (including every correction turn) is finished, which is what keeps two
    /// overlapping requests on one thread from forking the conversation.
    async fn turn_guard(&self, key: &str) -> tokio::sync::OwnedMutexGuard<()> {
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
    /// Pi Desktop's OpenAI Completions session fallback. The proxy hashes it
    /// into the local thread key and never persists the raw value.
    #[serde(default)]
    prompt_cache_key: Option<String>,
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

#[derive(Debug, Clone, Deserialize, Serialize)]
struct Msg {
    role: String,
    #[serde(default)]
    content: Option<Value>,
    // Deserialized for completeness; the prompt builders only need role/content.
    #[serde(default)]
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

/// A request narrowed to its own trailing input.
///
/// Fresh-session mode sends exactly one turn, so the messages it may replay are
/// the ones after the last assistant reply. Everything earlier belongs to other
/// requests and must not leak into a decision call.
fn fresh_request(req: &ChatRequest) -> ChatRequest {
    ChatRequest {
        model: req.model.clone(),
        thread_id: req.thread_id.clone(),
        prompt_cache_key: req.prompt_cache_key.clone(),
        messages: delta_messages(&req.messages).to_vec(),
        stream: req.stream,
        reasoning_effort: req.reasoning_effort.clone(),
        thinking: req.thinking.clone(),
        reasoning: req.reasoning.clone(),
        tools: req.tools.clone(),
        tool_choice: req.tool_choice.clone(),
        parallel_tool_calls: req.parallel_tool_calls,
        stream_options: req.stream_options.clone(),
    }
}

fn completion_body(
    thread: &ThreadState,
    prompt: &str,
    thinking: bool,
    search: bool,
    ref_file_ids: &[String],
    model: &str,
) -> CompletionBody {
    let mut body = CompletionBody::new_with_model_type(
        &thread.session_id,
        prompt,
        resolve_upstream_model_type(model),
    );
    body.thinking_enabled = thinking;
    body.search_enabled = search;
    body.parent_message_id = thread.parent_message_id;
    body.ref_file_ids = ref_file_ids.to_vec();
    body
}

/// Normalize public model ids to the private completion body's model type.
///
/// The upstream spelling is kept in one place so completion and image upload
/// paths cannot silently diverge. `default` preserves the existing wire value;
/// `expert` and `vision` are the explicit private model types exposed by this
/// proxy. Their upstream behavior still needs live validation.
fn resolve_upstream_model_type(model: &str) -> &'static str {
    match model.trim().to_ascii_uppercase().as_str() {
        "EXPERT" => "expert",
        "VISION" => "vision",
        _ => "default",
    }
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
    conn: &Arc<pool::Connection>,
    key: &str,
    reason: &str,
    persist: bool,
) -> Result<ThreadState, Response> {
    let session_id = {
        let _slot = conn.acquire().await.map_err(|e| queue_error(&e))?;
        conn.client()
            .await
            .create_chat_session()
            .await
            .map_err(|e| backend_error(&format!("create_session: {e}")))?
    };
    tracing::info!(thread = %key, reason, "creating a new remote session");
    let fresh = ThreadState {
        session_id,
        parent_message_id: None,
        exchanges: 0,
    };
    // Fresh-session mode keeps no resume state, so a later restart cannot make
    // a decision call inherit an earlier session.
    if persist {
        state
            .store
            .save_thread(
                key,
                &store::ThreadResume {
                    session_id: fresh.session_id.clone(),
                    parent_message_id: None,
                },
            )
            .map_err(|e| backend_error(&format!("save new thread: {e}")))?;
    }
    state
        .threads
        .lock()
        .unwrap()
        .insert(key.to_string(), fresh.clone());
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

struct ResolvedThreadKey {
    key: String,
    source: &'static str,
}

/// Resolve the conversation key for a request.
///
/// Precedence mirrors the Python proxy: explicit `thread_id` body field, then
/// the `x-thread-id` header, then the client session id, then Pi's
/// `prompt_cache_key`, and only then a hash of the messages. The raw identity
/// values are never logged or persisted by this layer.
fn thread_key(headers: &HeaderMap, req: &ChatRequest) -> ResolvedThreadKey {
    if let Some(v) = req
        .thread_id
        .as_deref()
        .map(str::trim)
        .filter(|v| !v.is_empty())
    {
        return ResolvedThreadKey {
            key: v.to_string(),
            source: "thread_id",
        };
    }
    if let Some(v) = header_value(headers, "x-thread-id") {
        return ResolvedThreadKey {
            key: v,
            source: "x-thread-id",
        };
    }
    // Title calls must be detected before the session id is consulted.
    let prefix = if is_title_request(req) {
        "title_"
    } else {
        "thread_"
    };
    for name in SESSION_HEADERS {
        if let Some(session) = header_value(headers, name) {
            return ResolvedThreadKey {
                key: sha256_hex(prefix, &session, None),
                source: name,
            };
        }
    }
    if let Some(cache_key) = req
        .prompt_cache_key
        .as_deref()
        .map(str::trim)
        .filter(|v| !v.is_empty())
    {
        return ResolvedThreadKey {
            key: sha256_hex(prefix, cache_key, None),
            source: "prompt_cache_key",
        };
    }
    if let Some(first) = req
        .messages
        .iter()
        .find(|m| m.role == "user" && !content_text(&m.content).is_empty())
    {
        return ResolvedThreadKey {
            key: sha256_hex(prefix, &content_text(&first.content), None),
            source: "message_hash",
        };
    }
    let all: String = req
        .messages
        .iter()
        .map(|m| content_text(&m.content))
        .collect();
    ResolvedThreadKey {
        key: sha256_hex(prefix, &all, None),
        source: "message_hash",
    }
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

static CHATGPT_CONFIG: std::sync::RwLock<Option<config::ChatGptConfig>> =
    std::sync::RwLock::new(None);

pub fn set_chatgpt_config(cfg: config::ChatGptConfig) {
    if let Ok(mut lock) = CHATGPT_CONFIG.write() {
        *lock = Some(cfg);
    }
}

pub fn get_chatgpt_config() -> config::ChatGptConfig {
    if let Ok(lock) = CHATGPT_CONFIG.read() {
        if let Some(cfg) = &*lock {
            return cfg.clone();
        }
    }
    config::Config::load()
        .map(|c| c.chatgpt)
        .unwrap_or_default()
}

/// Build the list of advertised models based on current ChatGPT configuration.
///
/// Returns existing base entries (`DEFAULT`, `EXPERT`, `VISION`) plus ChatGPT entries
/// when the ChatGPT Web channel is enabled and has valid tokens configured.
/// The list is deduplicated, returned in stable order, and matches the Google/OpenAI shape
/// (`object=list`, `id/object/created/owned_by`).
pub(crate) fn build_models_list(chatgpt: &config::ChatGptConfig) -> Value {
    let mut entries = Vec::new();
    let mut seen = std::collections::HashSet::new();

    let base_models = [
        ("DEFAULT", "deepseek"),
        ("EXPERT", "deepseek"),
        ("VISION", "deepseek"),
    ];

    for (id, owner) in base_models {
        if seen.insert(id.to_string()) {
            entries.push(json!({
                "id": id,
                "object": "model",
                "created": 0,
                "owned_by": owner,
            }));
        }
    }

    if chatgpt.should_advertise_models() {
        for id in config::chatgpt_models() {
            if seen.insert(id.to_string()) {
                entries.push(json!({
                    "id": id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "chatgpt",
                }));
            }
        }
    }

    json!({
        "object": "list",
        "data": entries,
    })
}

/// `GET /v1/models` — current entries plus ChatGPT entries when enabled.
async fn models(_state: State<AppState>) -> Json<Value> {
    Json(build_models_list(&get_chatgpt_config()))
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
        Ok(false) => {
            let (message, code) = if presented.is_empty() {
                ("missing api key", "missing_api_key")
            } else {
                ("invalid api key", "invalid_api_key")
            };
            Err(
                ProxyError::new(message, axum::http::StatusCode::UNAUTHORIZED, code)
                    .into_response(),
            )
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
                "type": if self.status.is_client_error() {
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
    conn: &Arc<pool::Connection>,
    thread_id: &str,
    body: &CompletionBody,
) -> Result<(reqwest::Response, pool::RateTicket), CompletionError> {
    let max_retries = state.settings.rate_limit_max_retries;
    let backoff = state.settings.rate_limit_backoff_secs.max(0.5);
    let mut attempt = 0u32;
    let mut token_recovered = false;

    loop {
        // The slot is held for the WHOLE stream, not just until headers land:
        // the caller keeps this ticket alive until the body is drained, so the
        // next request on this account starts `request_delay` after the stream
        // really finished rather than after it merely started.
        let slot_wait_started = Instant::now();
        let slot = conn.acquire().await.map_err(|e| {
            // `AccountQueueFull` / `AccountQueueTimeout` map to 503 in Python.
            if e.contains("queue") {
                CompletionError::QueueUnavailable(e)
            } else {
                CompletionError::Other(e)
            }
        })?;
        tracing::debug!(
            thread = %thread_id,
            attempt,
            account_slot_wait_ms = slot_wait_started.elapsed().as_millis() as u64,
            "account slot acquired"
        );
        let client = conn.client().await;

        match client.stream_completion(body).await {
            Ok(resp) => return Ok((resp, slot)),
            Err(e) => {
                let last = e.to_string();

                if pool::is_login_dead(&last) {
                    drop(slot);
                    // error_for may route the NEXT request to a replacement.
                    // Never reuse this account's remote session on that account.
                    return Err(CompletionError::LoginDead(last));
                }

                if pool::is_invalid_token(&last) && !token_recovered {
                    token_recovered = true;
                    if state.pool.recover(conn).await {
                        drop(slot);
                        continue;
                    }
                    drop(slot);
                    return Err(CompletionError::LoginDead(last));
                }

                if !pool::is_rate_limit(&last) {
                    drop(slot);
                    return Err(CompletionError::Other(last));
                }

                // Publish cooldown before releasing the in-flight permit, so
                // a waiting session cannot start during the retry backoff.
                let step = (attempt + 1).min(max_retries.max(1)) as f64;
                let wait = backoff * step;
                conn.cool_down(std::time::Duration::from_secs_f64(wait))
                    .await;
                drop(slot);
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

/// Request history retained only for this turn; SQLite stores its digest.
struct TurnHistory {
    messages: Vec<Value>,
    scope: String,
}

impl TurnHistory {
    fn new(req: &ChatRequest, model: &str) -> Self {
        let scope = json!([resolve_upstream_model_type(model), req.tools]);
        Self {
            messages: req
                .messages
                .iter()
                .map(|m| serde_json::to_value(m).expect("message fields are JSON values"))
                .collect(),
            scope: scope.to_string(),
        }
    }

    fn base_len(&self) -> usize {
        self.messages
            .iter()
            .rposition(|m| m["role"] == "assistant")
            .map(|i| i + 1)
            .unwrap_or(0)
    }

    fn fingerprint(&self, messages: &[Value]) -> String {
        let mut digest = Sha256::new();
        digest.update(self.scope.as_bytes());
        digest.update(b"\0");
        digest.update(history::fingerprint(messages).as_bytes());
        format!("branch-v1:{}", hex::encode(digest.finalize()))
    }

    fn can_continue(
        &self,
        previous: Option<&ThreadState>,
        saved: Option<&store::ThreadHistory>,
        account: &str,
    ) -> bool {
        let base_len = self.base_len();
        match (previous, saved) {
            (Some(previous), Some(saved)) => {
                previous.parent_message_id.is_some()
                    && base_len > 0
                    && saved.account_email == account
                    && saved.message_count == base_len
                    && saved.fingerprint == self.fingerprint(&self.messages[..base_len])
            }
            _ => false,
        }
    }

    fn completed(
        &mut self,
        prose: &str,
        calls: &[dialect::ToolCall],
        account: &str,
    ) -> store::ThreadHistory {
        let tool_calls: Vec<Value> = calls
            .iter()
            .enumerate()
            .map(|(i, call)| {
                json!({
                    "id": format!("call_{i}"), "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments}
                })
            })
            .collect();
        self.messages
            .push(json!({"role": "assistant", "content": prose, "tool_calls": tool_calls}));
        store::ThreadHistory {
            fingerprint: self.fingerprint(&self.messages),
            account_email: account.to_string(),
            message_count: self.messages.len(),
        }
    }
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Response {
    if let Err(resp) = authorize(&state, &headers) {
        state.stats.auth_failed.fetch_add(1, Ordering::Relaxed);
        return resp;
    }
    let started = Instant::now();
    if let Err(e) = validate_request(&req) {
        state.stats.errors.fetch_add(1, Ordering::Relaxed);
        return e.into_response();
    }
    let model = req
        .model
        .clone()
        .unwrap_or_else(|| state.settings.model_name.clone());
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
    let resolved_key = thread_key(&headers, &req);
    let key = resolved_key.key;
    let turn_lock_started = Instant::now();
    let turn = state.turn_guard(&key).await;
    let turn_lock_wait_ms = turn_lock_started.elapsed().as_millis() as u64;
    // A session and its cursor belong to one account. Keep that connection for
    // uploads, retries and correction turns, including after a cooldown.
    let conn = match state.pool.route_checked(&key).await {
        Ok(conn) => conn,
        Err(e) => return route_unavailable(e).into_response(),
    };
    // Fresh-session mode: a decision call must never see another request's
    // context. Read no stored thread and reuse no history, so every request
    // starts from an empty remote session. When the switch is off this block
    // is skipped and the previous behaviour is unchanged.
    let fresh = state.settings.fresh_session_per_request;
    let previous = if fresh {
        None
    } else {
        match state.store.get_thread(&key) {
            Ok(value) => value.map(|r| ThreadState {
                session_id: r.session_id,
                parent_message_id: r.parent_message_id.and_then(|p| p.parse().ok()),
                exchanges: 0,
            }),
            Err(e) => return backend_error(&format!("read thread: {e}")),
        }
    };
    let saved_history = if fresh {
        None
    } else {
        match state.store.get_thread_history(&key) {
            Ok(saved) => saved,
            Err(e) => return backend_error(&format!("read thread history: {e}")),
        }
    };
    let turn_history = TurnHistory::new(&req, &model);
    let continuation =
        !fresh && turn_history.can_continue(previous.as_ref(), saved_history.as_ref(), &conn.email);
    let existing = if continuation { previous } else { None };
    tracing::info!(
        thread = %key,
        thread_source = resolved_key.source,
        turn_lock_wait_ms,
        model = %model,
        messages = req.messages.len(),
        continuation,
        "request in"
    );

    let system_extra = dialect::inject_tool_descriptions("", &req.tools);
    let (specs, required) = resolve_tools(&req);
    // Fresh-session mode sends only this request's own input: the remote
    // session is new, so replaying earlier turns would reintroduce exactly the
    // history this mode exists to exclude. Attachments are loaded from the same
    // slice so image evidence and text still describe one turn.
    let inputs: &[Msg] = if fresh || continuation {
        delta_messages(&req.messages)
    } else {
        &req.messages
    };
    let loaded = match images::load(inputs, state.settings.image_max_bytes).await {
        Ok(loaded) => loaded,
        Err(e) => {
            state.stats.errors.fetch_add(1, Ordering::Relaxed);
            return ProxyError::new(e.0, axum::http::StatusCode::BAD_REQUEST, "invalid_image")
                .with_param("messages")
                .into_response();
        }
    };
    let image_note = loaded.note();
    state
        .stats
        .images_dropped
        .fetch_add(loaded.dropped as u64, Ordering::Relaxed);
    let (prompt, replay) = if fresh {
        // A fresh session has no delta to continue: the first (and only) prompt
        // is built from this request's own input.
        let fresh_req = fresh_request(&req);
        build_prompts(&fresh_req, None, &system_extra, image_note.as_deref())
    } else {
        build_prompts(
            &req,
            existing.as_ref(),
            &system_extra,
            image_note.as_deref(),
        )
    };
    if prompt.is_empty() {
        return bad_request("no user message or tool result in continuation turn");
    }
    // Invalidate BEFORE upstream work. An interrupted, failed or unseen answer
    // must not later be mistaken for a fully acknowledged successful head.
    // Fresh-session mode writes no thread state at all, so there is nothing to
    // invalidate and the remote session is created unconditionally below.
    if !fresh {
        if let Err(e) = state.store.clear_thread_history(&key) {
            return backend_error(&format!("invalidate thread history: {e}"));
        }
    }
    let can_recover = !fresh && continuation && !replay.is_empty();
    let thread = match existing {
        Some(thread) => thread,
        None => {
            let reason = if fresh {
                "fresh session per request"
            } else {
                "new or divergent request history"
            };
            match rebuild_session(&state, &conn, &key, reason, !fresh).await {
                Ok(thread) => thread,
                Err(response) => return response,
            }
        }
    };
    state
        .threads
        .lock()
        .unwrap()
        .insert(key.clone(), thread.clone());
    let ref_file_ids = match loaded.data {
        None => Vec::new(),
        Some(bytes) => {
            let _slot = match conn.acquire().await {
                Ok(slot) => slot,
                Err(e) => return queue_error(&e),
            };
            match conn
                .client()
                .await
                .upload_image(&bytes, resolve_upstream_model_type(&model))
                .await
            {
                Ok(uploaded) => vec![uploaded.file_id],
                Err(e) => {
                    tracing::warn!(thread = %key, error = %e, "image upload failed; continuing text-only");
                    Vec::new()
                }
            }
        }
    };
    let body = completion_body(
        &thread,
        &prompt,
        want_thinking(&req, state.settings.thinking_default),
        state.settings.search_enabled,
        &ref_file_ids,
        &model,
    );
    let (resp, slot) = match completion_with_retry(&state, &conn, &key, &body).await {
        Ok(result) => result,
        Err(e) => return completion_error_response(&state, &key, e).await,
    };
    stream_response(
        turn,
        state,
        key,
        conn,
        turn_history,
        resp,
        slot,
        body,
        format!("chatcmpl-{}", uuid_like()),
        model,
        unix_secs(),
        replay,
        can_recover,
        req.stream_options
            .as_ref()
            .and_then(|o| o.get("include_usage"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
        started,
        specs,
        required,
        req.parallel_tool_calls,
        fresh,
    )
    .await
}

/// Result of draining one upstream SSE response into OpenAI chunks.
struct Pumped {
    parent: Option<i64>,
    filter: dialect::StreamToolFilter,
    /// True once any chunk reached the client; a hint after that is fatal.
    emitted: bool,
    /// A final upstream completion marker was received, not merely EOF.
    complete: bool,
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
fn is_upstream_finished(value: &Value) -> bool {
    (value.get("p").and_then(Value::as_str) == Some("response/status")
        && value.get("v").and_then(Value::as_str) == Some("FINISHED"))
        || value.pointer("/v/response/status").and_then(Value::as_str) == Some("FINISHED")
}

async fn pump(
    resp: reqwest::Response,
    tx: &tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
    id: &str,
    model: &str,
    created: u64,
    idle: std::time::Duration,
    include_usage: bool,
) -> Pumped {
    let mut result = Pumped {
        parent: None,
        filter: dialect::StreamToolFilter::new(),
        emitted: false,
        complete: false,
        hint: None,
        text: String::new(),
        prose: String::new(),
        prompt_tokens: 0,
    };
    let mut router = FragmentRouter::new();
    let mut events = Vec::new();
    // Buffer bytes until a complete line: a UTF-8 scalar may cross HTTP chunks.
    let mut pending = Vec::new();
    let mut event_name = String::new();
    let mut head = Vec::new();
    let mut event_count = 0usize;
    let mut stream = resp.bytes_stream();
    loop {
        let next = tokio::select! {
            biased;
            _ = tx.closed() => return result,
            next = tokio::time::timeout(idle, stream.next()) => next,
        };
        let eof = match next {
            Ok(Some(Ok(bytes))) => {
                let take = (1024usize.saturating_sub(head.len())).min(bytes.len());
                head.extend_from_slice(&bytes[..take]);
                pending.extend_from_slice(&bytes);
                false
            }
            Ok(Some(Err(e))) => {
                result.complete = false;
                result.hint = Some(format!("stream read failed: {e}"));
                return result;
            }
            Ok(None) => {
                if !pending.is_empty() {
                    pending.push(b'\n');
                }
                true
            }
            Err(_) => {
                if tx
                    .send(Ok(Event::default().comment("keep-alive")))
                    .await
                    .is_err()
                {
                    return result;
                }
                continue;
            }
        };
        while let Some(pos) = pending.iter().position(|b| *b == b'\n') {
            let line = String::from_utf8_lossy(&pending[..pos]).into_owned();
            pending.drain(..=pos);
            if line.trim() == "data: [DONE]" {
                result.complete = true;
                continue;
            }
            let value = match protocol::classify_line(&line, &mut event_name) {
                protocol::LineKind::Ignored => continue,
                protocol::LineKind::HintError { content } => {
                    result.complete = false;
                    result.hint = Some(content);
                    return result;
                }
                protocol::LineKind::Event(value) => value,
            };
            event_count += 1;
            result.complete |= is_upstream_finished(&value);
            if let Some(mid) = value.get("response_message_id").and_then(Value::as_i64) {
                result.parent = Some(mid);
            }
            if let Some(tokens) = find_token_usage(&value) {
                result.prompt_tokens = tokens;
            }
            events.clear();
            route_event(&value, &mut router, &mut events);
            for ev in &events {
                let delta = match ev.channel {
                    Channel::Text => {
                        result.text.push_str(&ev.text);
                        let safe = result.filter.feed(&ev.text);
                        if safe.is_empty() {
                            continue;
                        }
                        result.prose.push_str(&safe);
                        json!({"content": safe})
                    }
                    Channel::Reasoning => json!({"reasoning_content": ev.text}),
                };
                let mut chunk = json!({
                    "id": id, "object": "chat.completion.chunk", "created": created,
                    "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": null}]
                });
                if include_usage {
                    chunk["usage"] = Value::Null;
                }
                result.emitted = true;
                if tx
                    .send(Ok(Event::default().data(chunk.to_string())))
                    .await
                    .is_err()
                {
                    return result;
                }
            }
        }
        if eof {
            break;
        }
    }
    if event_count == 0 && !head.is_empty() {
        // Preserve recovery classification without exposing an arbitrary body
        // (which could contain model output) through hint/error logging.
        let raw = String::from_utf8_lossy(&head);
        let hint = if protocol::is_invalid_message_id(&raw) {
            "invalid message id"
        } else if protocol::is_context_limit(&raw) {
            "context_length_exceeded"
        } else if pool::is_rate_limit(&raw) {
            "rate limit exceeded"
        } else if pool::is_login_dead(&raw) {
            "login failed"
        } else if pool::is_invalid_token(&raw) {
            "invalid token"
        } else {
            "non-SSE upstream response"
        };
        result.hint = Some(hint.to_string());
    }
    result
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
    conn: Arc<pool::Connection>,
    mut turn_history: TurnHistory,
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
    // Fresh-session mode: write no resume state for this turn.
    fresh: bool,
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
        let mut opening = json!({
            "id": id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
        });
        if include_usage {
            opening["usage"] = Value::Null;
        }
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
            let pumped = match tokio::time::timeout(
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
            if tx.is_closed() {
                drop(slot);
                return;
            }
            if pumped.hint.is_none() && !pumped.complete {
                drop(slot);
                state.stats.errors.fetch_add(1, Ordering::Relaxed);
                fail_turn(
                    &state,
                    &key,
                    &session_id,
                    cursor,
                    &tx,
                    ProxyError::new(
                        "Backend stream ended before completion",
                        axum::http::StatusCode::BAD_GATEWAY,
                        "incomplete_response",
                    ),
                )
                .await;
                return;
            }

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
                // A completed answer without its own cursor must not certify
                // the previous parent as the head of this new history.
                drop(slot);
                cursor = pumped.parent;
                break pumped;
            };

            // Failed turn: keep the cursor we came in with.
            cursor = body.parent_message_id;

            if protocol::is_invalid_message_id(&hint) && !pumped.emitted && !cursor_recovered {
                drop(slot);
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
                    match rebuild_session(&state, &conn, &key, &hint, true).await {
                        Ok(fresh) => {
                            let thinking = body.thinking_enabled;
                            let search = body.search_enabled;
                            let model_type = body.model_type.clone();
                            body = completion_body(
                                &fresh,
                                &replay,
                                thinking,
                                search,
                                &body.ref_file_ids,
                                &model_type,
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
                drop(slot);
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
                match rebuild_session(&state, &conn, &key, &hint, true).await {
                    Ok(fresh) => {
                        let thinking = body.thinking_enabled;
                        let search = body.search_enabled;
                        let model_type = body.model_type.clone();
                        body = completion_body(
                            &fresh,
                            &replay,
                            thinking,
                            search,
                            &body.ref_file_ids,
                            &model_type,
                        );
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
                    drop(slot);
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
                conn.cool_down(std::time::Duration::from_secs_f64(wait))
                    .await;
                drop(slot);
            } else {
                drop(slot);
                state.stats.errors.fetch_add(1, Ordering::Relaxed);
                tracing::error!(thread = %key, "upstream hint error: {hint}");
                let login_dead = pool::is_login_dead(&hint);
                let err = if login_dead {
                    CompletionError::LoginDead(hint.clone())
                } else {
                    CompletionError::Other(hint.clone())
                };
                fail_turn(
                    &state,
                    &key,
                    &session_id,
                    cursor,
                    &tx,
                    error_for(&state, &key, err).await,
                )
                .await;
                return;
            }

            match completion_with_retry(&state, &conn, &key, &body).await {
                Ok(v) => next = Some(v),
                Err(e) => {
                    fail_turn(
                        &state,
                        &key,
                        &session_id,
                        cursor,
                        &tx,
                        error_for(&state, &key, e).await,
                    )
                    .await;
                    return;
                }
            }
        };

        let Pumped {
            parent,
            mut filter,
            text,
            mut prose,
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
            if cursor.is_none() {
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
            let body = completion_body(&turn_state, &prompt, false, false, &[], &body.model_type);
            let budget = std::time::Duration::from_secs_f64(state.settings.request_timeout_secs)
                .saturating_sub(started.elapsed());
            let repair = async {
                let (resp, _slot) = completion_with_retry(&state, &conn, &key, &body).await?;
                // Keep the account slot until the correction body is drained.
                resp.text()
                    .await
                    .map_err(|e| CompletionError::Other(e.to_string()))
            };
            let raw = tokio::select! {
                biased;
                _ = tx.closed() => return,
                result = tokio::time::timeout(budget, repair) => match result {
                    Ok(Ok(raw)) => raw,
                    _ => {
                        recovery.end_repair();
                        break;
                    }
                },
            };
            if first_hint(&raw).is_some()
                || !raw.lines().any(|line| {
                    let mut event = String::new();
                    match protocol::classify_line(line, &mut event) {
                        protocol::LineKind::Event(value) => is_upstream_finished(&value),
                        _ => line.trim() == "data: [DONE]",
                    }
                })
            {
                recovery.end_repair();
                break;
            }
            let (_, repair_text, repair_mid) = route_all(&raw);
            // The correction answer is now the newest upstream message, so it —
            // not the malformed reply it fixed — becomes the parent of the
            // client's next turn. Saving `parent` here is what used to make
            // DeepSeek draw the next message as an edit of the broken answer.
            // Missing ids revoke cursor trust; never certify the prior answer
            // as the head of a successful correction turn.
            cursor = repair_mid;
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
        prose.push_str(&tail);
        if !tail.is_empty() {
            let mut chunk = json!({
                "id": id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {"content": tail}, "finish_reason": null}]
            });
            if include_usage {
                chunk["usage"] = Value::Null;
            }
            if tx
                .send(Ok(Event::default().data(chunk.to_string())))
                .await
                .is_err()
            {
                return;
            }
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
                        "id": tool_call_id(),
                        "type": "function",
                        "function": { "name": c.name, "arguments": c.arguments }
                    })
                })
                .collect();
            let mut chunk = json!({
                "id": id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {"tool_calls": calls}, "finish_reason": null}]
            });
            if include_usage {
                chunk["usage"] = Value::Null;
            }
            if tx
                .send(Ok(Event::default().data(chunk.to_string())))
                .await
                .is_err()
            {
                return;
            }
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

        let mut done = json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]
        });
        if include_usage {
            done["usage"] = Value::Null;
        }
        if tx
            .send(Ok(Event::default().data(done.to_string())))
            .await
            .is_err()
        {
            return;
        }
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
            if tx
                .send(Ok(Event::default().data(chunk.to_string())))
                .await
                .is_err()
            {
                return;
            }
        }
        // Commit before DONE: well-behaved clients close immediately after it.
        // If the terminal send fails, revoke trust before releasing the lock.
        if tx.is_closed() {
            return;
        }
        if let Some(mid) = cursor {
            // Fresh-session mode keeps no resume state: persisting it would let
            // a later request inherit this one's context, which is exactly what
            // the mode forbids.
            if !fresh {
                let history = turn_history.completed(&prose, &filter.calls, &conn.email);
                let resume = store::ThreadResume {
                    session_id: session_id.clone(),
                    parent_message_id: Some(mid.to_string()),
                };
                if let Err(e) = state.store.save_thread_checkpoint(&key, &resume, &history) {
                    tracing::error!(thread = %key, error = %e, "could not commit successful history checkpoint");
                    send_stream_error(
                        &tx,
                        ProxyError::new(
                            "Could not save conversation state",
                            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                            "state_commit_failed",
                        ),
                    )
                    .await;
                    return;
                }
            }
            state.threads.lock().unwrap().insert(
                key.clone(),
                ThreadState {
                    session_id,
                    parent_message_id: Some(mid),
                    exchanges: 0,
                },
            );
        }
        if tx.send(Ok(Event::default().data("[DONE]"))).await.is_err() {
            if let Err(e) = state.store.clear_thread_history(&key) {
                tracing::error!(thread = %key, error = %e, "could not invalidate interrupted history");
            }
        }
    });

    let mut response = Sse::new(ReceiverStream::new(rx)).into_response();
    response.headers_mut().insert(
        "x-accel-buffering",
        axum::http::HeaderValue::from_static("no"),
    );
    response
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

/// Unique OpenAI tool-call id for an emitted call.
///
/// Ids must be unique across the whole conversation, not just within a turn.
/// Clients such as Pi Desktop deduplicate transcripts by tool-call id: the old
/// per-turn `call_{index}` scheme reused `call_0` every turn, so they dropped
/// each later round, the history diverged from the saved head, and the proxy
/// opened a fresh remote session on every tool call.
fn tool_call_id() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let seed = format!(
        "{}:{}:{}",
        std::process::id(),
        uuid_like(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    );
    let digest = hex::encode(Sha256::digest(seed.as_bytes()));
    format!("call_{}", &digest[..24])
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
    if !req.stream {
        return Err(invalid(
            "stream",
            "Only stream=true is supported by this endpoint",
        ));
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
            return Err(invalid(
                "tool_choice",
                "Input should be a valid tool choice",
            ));
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
        let name = func
            .get("name")
            .and_then(|n| n.as_str())
            .unwrap_or_default();
        if name.is_empty() || name.chars().count() > 128 {
            return Err(invalid("tools", "Invalid or unsupported tool schema"));
        }
        let parameters = func.get("parameters").cloned().unwrap_or_else(|| json!({}));
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
    ProxyError::new(
        message,
        axum::http::StatusCode::BAD_REQUEST,
        "invalid_request",
    )
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

fn route_unavailable(err: pool::RouteError) -> ProxyError {
    let retry_after = match err {
        pool::RouteError::NoHealthyAccount { retry_after_ms } => {
            retry_after_ms.map(|ms| ms.div_ceil(1000).max(1))
        }
    };
    tracing::warn!(error = %err, "no healthy account available");
    let mut response = ProxyError::new(
        "No healthy DeepSeek account is currently available",
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        "queue_unavailable",
    );
    response.retry_after = retry_after;
    response
}
pub(crate) async fn connect_pool(
    cfg: &config::Config,
    store: &store::Store,
    accounts: Vec<pool::AccountSpec>,
) -> Result<Arc<pool::Pool>, Box<dyn std::error::Error>> {
    let pool_config = pool::PoolConfig {
        request_delay: std::time::Duration::from_secs_f64(cfg.request_delay_secs),
        max_inflight_per_account: cfg.max_inflight_per_account.max(1),
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
        channel_max_output_tokens: cfg.channel_max_output_tokens.clone(),
        fresh_session_per_request: cfg.fresh_session_per_request,
    };
    set_chatgpt_config(cfg.chatgpt.clone());
    tracing::info!(
        db_path = %cfg.db_path,
        model = %settings.model_name,
        thinking = settings.thinking_default,
        search = settings.search_enabled,
        "settings loaded"
    );

    // Build the active Responses channels. ChatGPT Web is model-routed and
    // only registered when authenticated tokens are configured.
    let mut response_channels: Vec<Arc<dyn responses::channel::ResponsesChannel>> =
        vec![Arc::new(responses::deepseek::DeepSeekWeb)];
    if cfg.chatgpt.should_advertise_models() {
        let clients = cfg
            .chatgpt
            .tokens
            .iter()
            .map(|token| {
                ChatGptClient::new(ClientConfig {
                    base_url: cfg.chatgpt.base_url.clone(),
                    access_token: token.clone(),
                    proxy_url: cfg.chatgpt.proxy_url.clone(),
                    timeout: std::time::Duration::from_secs_f64(cfg.chatgpt.timeout_secs),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let channel =
            responses::chatgpt::ChatGptWeb::new(clients).map_err(|error| error.to_string())?;
        response_channels.push(Arc::new(channel));
        tracing::info!(models = ?config::chatgpt_models(), "ChatGPT Web Responses channel enabled");
    }
    // A missing, malformed or mismatched capability profile is a startup
    // error: silently falling back could enable features the operator meant
    // to disable.
    let responses = responses::Runtime::load_channels(
        response_channels,
        std::env::var_os("RESPONSES_CAPABILITY_PROFILE").map(std::path::PathBuf::from),
    )?;
    let state = AppState {
        pool,
        settings,
        threads: Arc::new(Mutex::new(HashMap::new())),
        turns: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        store,
        responses,
        stats: Arc::new(Stats::new()),
    };

    // axum buffers the whole body before `Json<ChatRequest>` runs and refuses
    // anything over 2 MiB by default. A single 10 MiB image is ~13.3 MiB of
    // base64, so the default rejects perfectly valid turns with
    // "Failed to buffer the request body: length limit exceeded". The cap is
    // derived from IMAGE_MAX_BYTES, not a magic number.
    let body_limit = state
        .settings
        .image_max_bytes
        .saturating_mul(2)
        .max(32 << 20);

    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/stats", get(stats))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/responses", post(responses::create))
        .route(
            "/v1/responses/{response_id}",
            get(responses::retrieve).delete(responses::delete),
        )
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
            prompt_cache_key: None,
            messages,
            stream: true,
            reasoning_effort: None,
            thinking: None,
            reasoning: None,
            tools,
            tool_choice: None,
            parallel_tool_calls: true,
            stream_options: None,
        }
    }

    #[test]
    fn prompt_cache_key_is_a_fallback_thread_identity() {
        let mut req = request(vec![msg_with("user", "same prompt")], vec![]);
        req.prompt_cache_key = Some("pi-session-b".into());
        let resolved = thread_key(&HeaderMap::new(), &req);
        assert_eq!(resolved.source, "prompt_cache_key");
        assert_eq!(resolved.key, sha256_hex("thread_", "pi-session-b", None));
    }

    #[test]
    fn explicit_session_header_wins_over_prompt_cache_fallback() {
        let mut req = request(vec![msg_with("user", "same prompt")], vec![]);
        req.prompt_cache_key = Some("pi-session-b".into());
        let mut headers = HeaderMap::new();
        headers.insert("x-session-affinity", "pi-session-a".parse().unwrap());
        let resolved = thread_key(&headers, &req);
        assert_eq!(resolved.source, "x-session-affinity");
        assert_eq!(resolved.key, sha256_hex("thread_", "pi-session-a", None));
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

    /// Fresh-session mode narrows a request to its own trailing input, so a
    /// decision call cannot see any earlier turn.
    #[test]
    fn fresh_request_carries_only_the_current_turn() {
        let messages = vec![
            msg_with("user", "old question"),
            msg_with("assistant", "old answer"),
            msg_with("user", "decide now"),
        ];
        let narrowed = fresh_request(&request(messages.clone(), vec![]));
        assert_eq!(narrowed.messages.len(), 1);
        assert_eq!(content_text(&narrowed.messages[0].content), "decide now");
        // The prompt for a fresh session is built from that single turn only.
        let (prompt, replay) = build_prompts(&narrowed, None, "", None);
        assert_eq!(prompt, "User: decide now");
        assert!(!prompt.contains("old question"));
        assert!(!replay.contains("old answer"));
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
        let req = request(vec![], vec![]);
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
        assert!(
            prompt.contains("\"command\""),
            "schema must survive the dump"
        );
        assert!(
            prompt.contains("required"),
            "required must survive the dump"
        );
        assert!(
            !prompt.contains("\"search\""),
            "unfailed tools must not be dumped"
        );
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
            recovery.pending.push(dialect::MalformedCall::new(
                name.clone(),
                String::new(),
                reason,
            ));
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
            assert!(
                prompt.contains("\"command\""),
                "{text}: schema body missing"
            );
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
        assert_eq!(threads.lock().unwrap()["t1"].parent_message_id, Some(101));

        // A later correction advances it again.
        persist_cursor(&threads, &store, "t1", "s1", Some(102));
        assert_eq!(threads.lock().unwrap()["t1"].parent_message_id, Some(102));
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
    #[test]
    fn stream_false_is_rejected_and_model_types_are_normalized() {
        let mut req = request(vec![msg_with("user", "hi")], vec![]);
        req.stream = false;
        let err = validate_request(&req).unwrap_err();
        assert_eq!(err.param.as_deref(), Some("stream"));
        assert_eq!(err.code, "invalid_request");

        assert_eq!(resolve_upstream_model_type("DEFAULT"), "default");
        assert_eq!(resolve_upstream_model_type("deepseek-chat"), "default");
        assert_eq!(resolve_upstream_model_type("EXPERT"), "expert");
        assert_eq!(resolve_upstream_model_type("vision"), "vision");
    }

    #[test]
    fn branch_history_distinguishes_continuation_regeneration_and_rewind() {
        let first = request(vec![msg_with("user", "root")], vec![]);
        let saved = TurnHistory::new(&first, "DEFAULT").completed("answer A", &[], "account-a");
        let remote = ThreadState {
            session_id: "remote-a".into(),
            parent_message_id: Some(2),
            exchanges: 0,
        };
        let second = request(
            vec![
                msg_with("user", "root"),
                msg_with("assistant", "answer A"),
                msg_with("user", "secret"),
            ],
            vec![],
        );
        assert!(TurnHistory::new(&second, "deepseek-chat").can_continue(
            Some(&remote),
            Some(&saved),
            "account-a"
        ));
        // Same prompt again is a regeneration, not a continuation below its answer.
        assert!(!TurnHistory::new(&first, "DEFAULT").can_continue(
            Some(&remote),
            Some(&saved),
            "account-a"
        ));
        assert!(!TurnHistory::new(&second, "DEFAULT").can_continue(
            Some(&remote),
            Some(&saved),
            "account-b"
        ));
        assert!(!TurnHistory::new(&second, "DEFAULT").can_continue(
            Some(&remote),
            None,
            "account-a"
        ));
        assert!(!TurnHistory::new(&second, "EXPERT").can_continue(
            Some(&remote),
            Some(&saved),
            "account-a"
        ));
        let newer = TurnHistory::new(&second, "DEFAULT").completed("answer B", &[], "account-a");
        let rewound = request(
            vec![
                msg_with("user", "root"),
                msg_with("assistant", "answer A"),
                msg_with("user", "different future"),
            ],
            vec![],
        );
        assert!(!TurnHistory::new(&rewound, "DEFAULT").can_continue(
            Some(&remote),
            Some(&newer),
            "account-a"
        ));
        let edited = request(
            vec![
                msg_with("user", "changed root"),
                msg_with("assistant", "answer A"),
                msg_with("user", "secret"),
            ],
            vec![],
        );
        assert!(!TurnHistory::new(&edited, "DEFAULT").can_continue(
            Some(&remote),
            Some(&saved),
            "account-a"
        ));
    }

    #[test]
    fn branch_checkpoint_restores_without_an_in_memory_head() {
        let store = store::Store::open_memory().unwrap();
        let first = request(vec![msg_with("user", "root")], vec![]);
        let saved = TurnHistory::new(&first, "DEFAULT").completed("answer", &[], "account-a");
        store
            .save_thread_checkpoint(
                "thread",
                &store::ThreadResume {
                    session_id: "remote".into(),
                    parent_message_id: Some("2".into()),
                },
                &saved,
            )
            .unwrap();
        let resume = store.get_thread("thread").unwrap().unwrap();
        let remote = ThreadState {
            session_id: resume.session_id,
            parent_message_id: resume.parent_message_id.and_then(|v| v.parse().ok()),
            exchanges: 0,
        };
        let next = request(
            vec![
                msg_with("user", "root"),
                msg_with("assistant", "answer"),
                msg_with("user", "next"),
            ],
            vec![],
        );
        let history = TurnHistory::new(&next, "DEFAULT");
        assert!(history.can_continue(
            Some(&remote),
            store.get_thread_history("thread").unwrap().as_ref(),
            "account-a"
        ));
        store.clear_thread_history("thread").unwrap();
        assert!(!history.can_continue(
            Some(&remote),
            store.get_thread_history("thread").unwrap().as_ref(),
            "account-a"
        ));
    }

    async fn branch_test_state() -> AppState {
        let store = store::Store::open_memory().unwrap();
        let pool = pool::Pool::connect(
            vec![pool::AccountSpec {
                email: "test@example.invalid".into(),
                password: String::new(),
                token: Some("local-test-token".into()),
            }],
            pool::PoolConfig::default(),
            store.clone(),
        )
        .await
        .unwrap();
        AppState {
            pool: Arc::new(pool),
            store,
            settings: Settings {
                thinking_default: false,
                search_enabled: false,
                model_name: "DEFAULT".into(),
                rate_limit_max_retries: 0,
                rate_limit_backoff_secs: 0.5,
                tool_repair_max_retries: 0,
                request_timeout_secs: 2.0,
                image_max_bytes: 1024,
                heartbeat_interval_secs: 0.1,
                channel_max_output_tokens: std::collections::HashMap::new(),
                fresh_session_per_request: false,
            },
            threads: Arc::new(Mutex::new(HashMap::new())),
            turns: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            responses: responses::Runtime::new(Arc::new(responses::deepseek::DeepSeekWeb), None)
                .unwrap()
                .0,
            stats: Arc::new(Stats::new()),
        }
    }

    async fn branch_test_stream(req: &ChatRequest, raw: &str) -> (AppState, Response) {
        branch_test_stream_with_parent(req, raw, None).await
    }

    async fn branch_test_stream_with_parent(
        req: &ChatRequest,
        raw: &str,
        parent_message_id: Option<i64>,
    ) -> (AppState, Response) {
        let state = branch_test_state().await;
        let remote = ThreadState {
            session_id: "test-remote".into(),
            parent_message_id,
            exchanges: 0,
        };
        state
            .store
            .save_thread(
                "test-thread",
                &store::ThreadResume {
                    session_id: remote.session_id.clone(),
                    parent_message_id: None,
                },
            )
            .unwrap();
        state
            .threads
            .lock()
            .unwrap()
            .insert("test-thread".into(), remote.clone());
        let conn = state.pool.any();
        let slot = conn.acquire().await.unwrap();
        let resp = reqwest::Response::from(
            axum::http::Response::builder()
                .status(200)
                .header("content-type", "text/event-stream")
                .body(raw.to_string())
                .unwrap(),
        );
        let turn = state.turn_guard("test-thread").await;
        let (specs, required) = resolve_tools(req);
        let response = stream_response(
            turn,
            state.clone(),
            "test-thread".into(),
            conn,
            TurnHistory::new(req, "DEFAULT"),
            resp,
            slot,
            completion_body(&remote, "test", false, false, &[], "DEFAULT"),
            "test-completion".into(),
            "DEFAULT".into(),
            1,
            "test".into(),
            false,
            true,
            Instant::now(),
            specs,
            required,
            true,
            false,
        )
        .await;
        (state, response)
    }

    #[tokio::test]
    async fn branch_stream_commits_actual_client_history() {
        let req = request(vec![msg_with("user", "hello")], vec![]);
        let raw = "data: {\"response_message_id\":2}\n\ndata: {\"v\":\"中文 answer\"}\n\ndata: {\"p\":\"response/status\",\"v\":\"FINISHED\"}\n\n";
        let (state, response) = branch_test_stream(&req, raw).await;
        assert_eq!(response.headers()["x-accel-buffering"], "no");
        let bytes = axum::body::to_bytes(response.into_body(), 65536)
            .await
            .unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(body.contains("中文 answer"));
        assert_eq!(body.matches("data: [DONE]").count(), 1);
        let saved = state
            .store
            .get_thread_history("test-thread")
            .unwrap()
            .unwrap();
        let next = request(
            vec![
                msg_with("user", "hello"),
                msg_with("assistant", "中文 answer"),
                msg_with("user", "continue"),
            ],
            vec![],
        );
        let remote = state.threads.lock().unwrap()["test-thread"].clone();
        assert!(TurnHistory::new(&next, "DEFAULT").can_continue(
            Some(&remote),
            Some(&saved),
            "test@example.invalid"
        ));
    }

    #[tokio::test]
    async fn branch_tool_checkpoint_accepts_client_id_rewrite() {
        let req = request(vec![msg_with("user", "run")], vec![tool_with_schema()]);
        let raw = "data: {\"response_message_id\":2}\n\ndata: {\"v\":\"<invoke name=\\\"bash\\\">{\\\"command\\\":\\\"whoami\\\"}</invoke>\"}\n\ndata: {\"p\":\"response/status\",\"v\":\"FINISHED\"}\n\n";
        let (state, response) = branch_test_stream(&req, raw).await;
        let bytes = axum::body::to_bytes(response.into_body(), 65536)
            .await
            .unwrap();
        let sse = String::from_utf8(bytes.to_vec()).unwrap();
        assert_eq!(sse.matches("\"tool_calls\":[").count(), 1);
        let mut assistant = msg_with("assistant", "");
        assistant.tool_calls = vec![
            json!({"id":"call_0|fc_call_0","type":"function","function":{"name":"bash","arguments":"{ \"command\": \"whoami\" }"}}),
        ];
        let mut result = msg_with("tool", "test-user");
        result.tool_call_id = Some("call_0|fc_call_0".into());
        let next = request(
            vec![msg_with("user", "run"), assistant, result],
            vec![tool_with_schema()],
        );
        let saved = state
            .store
            .get_thread_history("test-thread")
            .unwrap()
            .unwrap();
        let remote = state.threads.lock().unwrap()["test-thread"].clone();
        assert!(TurnHistory::new(&next, "DEFAULT").can_continue(
            Some(&remote),
            Some(&saved),
            "test@example.invalid"
        ));
    }

    /// Tool-call ids the stream emitted, in order.
    fn emitted_tool_call_ids(sse: &str) -> Vec<String> {
        sse.lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter_map(|data| serde_json::from_str::<Value>(data).ok())
            .filter_map(|chunk| {
                chunk["choices"][0]["delta"]["tool_calls"]
                    .as_array()
                    .cloned()
            })
            .flatten()
            .filter_map(|call| call["id"].as_str().map(str::to_string))
            .collect()
    }

    /// Upstream reply that calls `bash` once with `command`.
    fn bash_call_reply(command: &str) -> String {
        let block = format!(
            "<invoke name=\"bash\">{}</invoke>",
            json!({ "command": command })
        );
        let events = [
            json!({ "response_message_id": 2 }),
            json!({ "v": block }),
            json!({ "p": "response/status", "v": "FINISHED" }),
        ];
        events
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .collect()
    }

    fn assistant_call(id: &str, command: &str) -> Msg {
        let mut assistant = msg_with("assistant", "");
        assistant.tool_calls = vec![json!({
            "id": id,
            "type": "function",
            "function": {"name": "bash", "arguments": json!({ "command": command }).to_string()}
        })];
        assistant
    }

    fn tool_result(id: &str, output: &str) -> Msg {
        let mut result = msg_with("tool", output);
        result.tool_call_id = Some(id.into());
        result
    }

    async fn sse_body(response: Response) -> String {
        let bytes = axum::body::to_bytes(response.into_body(), 65536)
            .await
            .unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    /// Pi Desktop drops every later tool call and result whose id repeats an
    /// earlier one in the transcript. Per-turn ids (`call_0` every turn) made
    /// it delete each round after the first, so the next request diverged from
    /// the saved history, a new remote session was opened on every tool call,
    /// and the model re-issued calls it could no longer see.
    #[tokio::test]
    async fn tool_call_ids_stay_unique_across_turns() {
        let tools = vec![tool_with_schema()];
        let first = request(vec![msg_with("user", "run")], tools.clone());
        let (_, response) = branch_test_stream(&first, &bash_call_reply("whoami")).await;
        let first_sse = sse_body(response).await;
        let first_ids = emitted_tool_call_ids(&first_sse);
        assert_eq!(first_ids.len(), 1);

        let second = request(
            vec![
                msg_with("user", "run"),
                assistant_call(&first_ids[0], "whoami"),
                tool_result(&first_ids[0], "test-user"),
            ],
            tools.clone(),
        );
        let (state, response) = branch_test_stream(&second, &bash_call_reply("hostname")).await;
        let second_ids = emitted_tool_call_ids(&sse_body(response).await);
        assert_eq!(second_ids.len(), 1);
        assert_ne!(first_ids[0], second_ids[0], "every turn needs a fresh id");
        for id in first_ids.iter().chain(&second_ids) {
            assert!(id.starts_with("call_") && id.len() <= 40, "{id}");
        }

        let saved = state
            .store
            .get_thread_history("test-thread")
            .unwrap()
            .unwrap();
        let remote = state.threads.lock().unwrap()["test-thread"].clone();
        // With both rounds kept, the next request extends the saved head.
        let both_rounds = vec![
            msg_with("user", "run"),
            assistant_call(&first_ids[0], "whoami"),
            tool_result(&first_ids[0], "test-user"),
            assistant_call(&second_ids[0], "hostname"),
            tool_result(&second_ids[0], "test-host"),
        ];
        let next = request(both_rounds, tools.clone());
        assert!(TurnHistory::new(&next, "DEFAULT").can_continue(
            Some(&remote),
            Some(&saved),
            "test@example.invalid"
        ));
        // The transcript a colliding id produced: the second round deleted.
        assert!(!TurnHistory::new(&second, "DEFAULT").can_continue(
            Some(&remote),
            Some(&saved),
            "test@example.invalid"
        ));
    }

    #[test]
    fn tool_call_ids_never_repeat() {
        let ids: std::collections::HashSet<String> = (0..10_000).map(|_| tool_call_id()).collect();
        assert_eq!(ids.len(), 10_000);
        assert!(ids
            .iter()
            .all(|id| id.starts_with("call_") && id.len() <= 40));
    }

    #[tokio::test]
    async fn branch_partial_and_failed_streams_leave_no_trusted_head() {
        let req = request(vec![msg_with("user", "hello")], vec![]);
        for raw in [
            "data: {\"response_message_id\":2}\n\ndata: {\"v\":\"partial\"}\n\n",
            "event: hint\ndata: {\"type\":\"error\",\"content\":\"synthetic failure\"}\n\n",
        ] {
            let (state, response) = branch_test_stream(&req, raw).await;
            let bytes = axum::body::to_bytes(response.into_body(), 65536)
                .await
                .unwrap();
            let sse = String::from_utf8(bytes.to_vec()).unwrap();
            assert!(sse.contains("\"error\""));
            assert_eq!(sse.matches("data: [DONE]").count(), 1);
            assert!(state
                .store
                .get_thread_history("test-thread")
                .unwrap()
                .is_none());
        }
    }

    #[tokio::test]
    async fn branch_unexpected_upstream_body_is_redacted_before_error_logging() {
        let resp = reqwest::Response::from(
            axum::http::Response::builder()
                .status(200)
                .body("synthetic private model output")
                .unwrap(),
        );
        let (tx, _rx) = tokio::sync::mpsc::channel(32);
        let outcome = pump(
            resp,
            &tx,
            "test-completion",
            "DEFAULT",
            1,
            std::time::Duration::from_millis(100),
            false,
        )
        .await;
        assert!(!outcome.complete);
        assert_eq!(outcome.hint.as_deref(), Some("non-SSE upstream response"));
    }

    #[tokio::test]
    async fn branch_answer_without_new_cursor_cannot_certify_old_parent() {
        let req = request(
            vec![
                msg_with("user", "root"),
                msg_with("assistant", "previous answer"),
                msg_with("user", "next"),
            ],
            vec![],
        );
        let raw = "data: {\"v\":\"new answer\"}\n\ndata: {\"p\":\"response/status\",\"v\":\"FINISHED\"}\n\n";
        let (state, response) = branch_test_stream_with_parent(&req, raw, Some(7)).await;
        let bytes = axum::body::to_bytes(response.into_body(), 65536)
            .await
            .unwrap();
        let sse = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(sse.contains("new answer"));
        assert!(!sse.contains("\"error\""));
        assert_eq!(sse.matches("data: [DONE]").count(), 1);
        assert!(state
            .store
            .get_thread_history("test-thread")
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn branch_disconnected_client_leaves_no_trusted_head() {
        let req = request(vec![msg_with("user", "hello")], vec![]);
        let (state, response) =
            branch_test_stream(&req, "data: {\"v\":\"answer\"}\n\ndata: [DONE]\n\n").await;
        drop(response);
        let guard = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            state.turn_guard("test-thread"),
        )
        .await
        .unwrap();
        assert!(state
            .store
            .get_thread_history("test-thread")
            .unwrap()
            .is_none());
        let conn = state.pool.any();
        let _slot = conn.acquire().await.unwrap();
        drop(guard);
    }

    #[test]
    fn test_models_advertisement_disabled_or_no_tokens() {
        // 1. Channel disabled completely
        let mut chatgpt = config::ChatGptConfig::default();
        chatgpt.enabled = false;
        chatgpt.tokens = vec!["tok-1".into()];
        let list = build_models_list(&chatgpt);
        assert_eq!(list["object"], "list");
        let data = list["data"].as_array().unwrap();
        let ids: Vec<&str> = data.iter().map(|m| m["id"].as_str().unwrap()).collect();
        assert_eq!(ids, vec!["DEFAULT", "EXPERT", "VISION"]);
        for m in data {
            assert_eq!(m["object"], "model");
            assert_eq!(m["created"], 0);
            assert_eq!(m["owned_by"], "deepseek");
        }

        // 2. Channel enabled but no tokens (cannot run anonymously)
        chatgpt.enabled = true;
        chatgpt.tokens = vec![];
        let list = build_models_list(&chatgpt);
        let data = list["data"].as_array().unwrap();
        let ids: Vec<&str> = data.iter().map(|m| m["id"].as_str().unwrap()).collect();
        assert_eq!(ids, vec!["DEFAULT", "EXPERT", "VISION"]);
    }

    #[test]
    fn test_models_advertisement_enabled_with_tokens() {
        let mut chatgpt = config::ChatGptConfig::default();
        chatgpt.enabled = true;
        chatgpt.tokens = vec!["tok-1".into()];

        let list = build_models_list(&chatgpt);
        assert_eq!(list["object"], "list");
        let data = list["data"].as_array().unwrap();
        let ids: Vec<&str> = data.iter().map(|m| m["id"].as_str().unwrap()).collect();

        // 3 base models + 9 ChatGPT models in stable order
        let expected_ids = vec![
            "DEFAULT",
            "EXPERT",
            "VISION",
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
        assert_eq!(ids, expected_ids);

        // Verify shape and ownership
        for m in &data[..3] {
            assert_eq!(m["object"], "model");
            assert_eq!(m["created"], 0);
            assert_eq!(m["owned_by"], "deepseek");
        }
        for m in &data[3..] {
            assert_eq!(m["object"], "model");
            assert_eq!(m["created"], 0);
            assert_eq!(m["owned_by"], "chatgpt");
        }
    }

    #[tokio::test]
    async fn test_models_handler_execution() {
        let state = branch_test_state().await;
        // Test disabled
        let mut cfg = config::ChatGptConfig::default();
        cfg.enabled = false;
        set_chatgpt_config(cfg);
        let Json(val) = models(State(state.clone())).await;
        assert_eq!(val["object"], "list");
        let data = val["data"].as_array().unwrap();
        assert_eq!(data.len(), 3);

        // Test enabled with token
        let mut cfg_enabled = config::ChatGptConfig::default();
        cfg_enabled.enabled = true;
        cfg_enabled.tokens = vec!["active_token".into()];
        set_chatgpt_config(cfg_enabled);
        let Json(val_enabled) = models(State(state)).await;
        assert_eq!(val_enabled["object"], "list");
        let data_enabled = val_enabled["data"].as_array().unwrap();
        assert_eq!(data_enabled.len(), 12);
    }
}
