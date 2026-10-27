//! `/v1/responses`: the channel-independent HTTP and state layer.
//!
//! This module owns what every channel shares: bearer-key ownership,
//! capability gating, stored response chains (`previous_response_id`), the
//! per-thread turn lock, Responses stream events and persistence. Generation
//! itself is delegated to a [`channel::ResponsesChannel`]; the DeepSeek web
//! adapter lives in [`deepseek`]. Protocol rules live in `responses-core`.
//!
//! The Chat Completions endpoint is untouched by this module.

pub mod channel;
pub mod chatgpt;
pub mod deepseek;
#[cfg(test)]
mod tests;

use crate::store::ResponseRecord;
use crate::{backend_error, AppState, ProxyError, ThreadState};
use axum::{
    body::Bytes,
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json,
};
use channel::{ResponsesChannel, TurnError, TurnEvent, TurnOutcome, TurnRequest};
use responses_core::capability::{Capability, CapabilitySet, ProfileDocument};
use responses_core::events::{EventEncoder, StreamEvent};
use responses_core::ids::new_id;
use responses_core::plan::{CreateRequest, ReasoningEffort, RequestError, Requirement, TurnPlan};
use responses_core::transcript::{text_of, Item, Part, Role};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt as _;

/// One channel plus its effective capability set.
#[derive(Clone)]
struct ChannelRuntime {
    channel: Arc<dyn ResponsesChannel>,
    capabilities: CapabilitySet,
}

/// The active channels and their effective capabilities.
#[derive(Clone)]
pub struct Runtime {
    /// Backward-compatible default channel used by existing unit-test helpers.
    pub channel: Arc<dyn ResponsesChannel>,
    pub capabilities: CapabilitySet,
    channels: Vec<ChannelRuntime>,
}

impl Runtime {
    /// Build a runtime for one channel.
    pub fn new(
        channel: Arc<dyn ResponsesChannel>,
        profile: Option<&ProfileDocument>,
    ) -> Result<(Self, Vec<String>), String> {
        Self::from_channels(vec![channel], profile)
    }

    /// Build a runtime for several model-routed channels.
    pub fn from_channels(
        channels: Vec<Arc<dyn ResponsesChannel>>,
        profile: Option<&ProfileDocument>,
    ) -> Result<(Self, Vec<String>), String> {
        if channels.is_empty() {
            return Err("at least one Responses channel is required".into());
        }
        let mut warnings = Vec::new();
        let mut configured = Vec::with_capacity(channels.len());
        let mut profile_applied = profile.is_none();
        for channel in channels {
            let mut capabilities = channel.capabilities();
            if capabilities.channel() != channel.id() {
                return Err(format!(
                    "channel '{}' declared capabilities for '{}'",
                    channel.id(),
                    capabilities.channel()
                ));
            }
            warnings.extend(capabilities.restrict_to_engine().into_iter().map(|id| {
                format!("channel declares '{id}', which this engine cannot represent; disabled")
            }));
            if let Some(profile) = profile.filter(|profile| profile.channel == channel.id()) {
                warnings.extend(
                    capabilities
                        .narrow(profile)
                        .map_err(|error| error.to_string())?,
                );
                profile_applied = true;
            }
            configured.push(ChannelRuntime {
                channel,
                capabilities,
            });
        }
        if !profile_applied {
            let profile_channel = profile
                .map(|profile| profile.channel.as_str())
                .unwrap_or("");
            return Err(format!(
                "capability profile is for channel '{}', but no active channel matches it",
                profile_channel
            ));
        }
        let default = configured.first().cloned().expect("non-empty channel list");
        Ok((
            Self {
                channel: default.channel,
                capabilities: default.capabilities,
                channels: configured,
            },
            warnings,
        ))
    }

    /// Select the channel and capability set that can serve one model.
    pub fn for_model(&self, model: &str) -> Option<(Arc<dyn ResponsesChannel>, CapabilitySet)> {
        self.channels
            .iter()
            .find(|runtime| runtime.channel.supports_model(model))
            .map(|runtime| (runtime.channel.clone(), runtime.capabilities.clone()))
    }

    /// Build a multi-channel runtime at startup, reading the optional profile file.
    pub fn load_channels(
        channels: Vec<Arc<dyn ResponsesChannel>>,
        profile_path: Option<std::path::PathBuf>,
    ) -> Result<Self, String> {
        let document = match &profile_path {
            None => None,
            Some(path) => {
                let bytes = std::fs::read(path).map_err(|error| {
                    format!("read capability profile {}: {error}", path.display())
                })?;
                Some(ProfileDocument::parse(&bytes).map_err(|error| error.to_string())?)
            }
        };
        let (runtime, warnings) = Self::from_channels(channels, document.as_ref())?;
        for warning in warnings {
            tracing::warn!("{warning}");
        }
        tracing::info!(
            channels = %runtime
                .channels
                .iter()
                .map(|channel| channel.channel.id())
                .collect::<Vec<_>>()
                .join(","),
            profile = profile_path.is_some(),
            "Responses capabilities"
        );
        Ok(runtime)
    }
    /// Build the runtime at startup, reading the optional profile file.
    pub fn load(
        channel: Arc<dyn ResponsesChannel>,
        profile_path: Option<std::path::PathBuf>,
    ) -> Result<Self, String> {
        let document = match &profile_path {
            None => None,
            Some(path) => {
                let bytes = std::fs::read(path).map_err(|error| {
                    format!("read capability profile {}: {error}", path.display())
                })?;
                Some(ProfileDocument::parse(&bytes).map_err(|error| error.to_string())?)
            }
        };
        let (runtime, warnings) = Self::new(channel, document.as_ref())?;
        for warning in warnings {
            tracing::warn!("{warning}");
        }
        tracing::info!(
            channel = runtime.channel.id(),
            profile = profile_path.is_some(),
            enabled = %runtime.capabilities.enabled_ids().join(","),
            "Responses capabilities"
        );
        Ok(runtime)
    }
}
/// Messages from a turn task to the HTTP handler.
enum Outbound {
    /// The upstream accepted the turn; the response is committed to HTTP 200.
    Started,
    /// The turn failed before starting; send this response instead.
    Rejected(Response),
    /// One stream event (streaming requests only).
    Event(StreamEvent),
    /// Terminal result for collected requests.
    Finished(Result<Value, Response>),
}

/// Per-turn data owned by the task.
struct Job {
    thread_id: String,
    owner_id: String,
    response_id: String,
    model: String,
    store: bool,
    stream: bool,
    /// Reasoning is enabled; otherwise channel reasoning is never forwarded.
    reasoning: bool,
    parallel_tool_calls: bool,
    previous_response_id: Option<String>,
    /// Context replayed into this turn: stored chain plus this input.
    transcript: Vec<Item>,
}

/// Authorize the request and resolve its owner. Open installations (no
/// registered keys) share the `anonymous` owner.
fn owner_of(state: &AppState, headers: &HeaderMap) -> Result<String, Response> {
    if let Err(response) = crate::authorize(state, headers) {
        state.stats.auth_failed.fetch_add(1, Ordering::Relaxed);
        return Err(response);
    }
    let presented = headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .map(|value| value.trim_start_matches("Bearer ").trim())
        .unwrap_or_default();
    match state.store.api_key_owner(presented) {
        Ok(Some(owner)) => Ok(owner),
        Ok(None) => Err(ProxyError::new(
            "invalid api key",
            StatusCode::UNAUTHORIZED,
            "invalid_api_key",
        )
        .into_response()),
        Err(error) => Err(backend_error(&format!("resolve response owner: {error}"))),
    }
}

fn request_error(error: RequestError) -> Response {
    let mut proxy = ProxyError::new(error.message, StatusCode::BAD_REQUEST, error.code);
    proxy.param = error.param;
    proxy.into_response()
}

fn not_found(param: &str, message: &str) -> Response {
    ProxyError::new(message, StatusCode::NOT_FOUND, "not_found")
        .with_param(param)
        .into_response()
}

/// Reject a requirement the active channel cannot serve.
fn gate(capabilities: &CapabilitySet, requirement: &Requirement) -> RequestError {
    let capability = requirement.capability;
    let missing = std::iter::once(capability)
        .chain(capability.dependencies().iter().copied())
        .find(|candidate| !capabilities.enabled(*candidate))
        .unwrap_or(capability);
    let message = if missing == capability {
        format!(
            "'{}' requires capability '{}', which channel '{}' does not support",
            requirement.param,
            capability.id(),
            capabilities.channel()
        )
    } else {
        format!(
            "'{}' requires capability '{}', which needs '{}'; channel '{}' does not support it",
            requirement.param,
            capability.id(),
            missing.id(),
            capabilities.channel()
        )
    };
    RequestError::unsupported(&requirement.param, &message)
}

/// Function parameter schemas must compile and stay local, so validation can
/// never fetch a remote `$ref`.
fn validate_tool_schemas(plan: &TurnPlan) -> Result<(), RequestError> {
    for tool in &plan.tools {
        if crate::has_external_ref(&tool.parameters)
            || jsonschema::validator_for(&tool.parameters).is_err()
        {
            return Err(RequestError::invalid(
                "tools",
                &format!(
                    "function '{}' has an invalid or unsupported parameters schema",
                    tool.name
                ),
            ));
        }
    }
    Ok(())
}

/// First capability a replayed transcript needs that is not enabled now.
fn replay_requirement(transcript: &[Item], capabilities: &CapabilitySet) -> Option<Requirement> {
    transcript
        .iter()
        .filter_map(|item| match item {
            Item::Message { content, .. } => Some(content),
            _ => None,
        })
        .flatten()
        .filter_map(|part| match part {
            Part::Image { .. } => Some(Capability::ImageInput),
            Part::File { .. } => Some(Capability::FileInput),
            Part::Text { .. } => None,
        })
        .find(|capability| !capabilities.enabled(*capability))
        .map(|capability| Requirement {
            capability,
            param: "previous_response_id".into(),
        })
}

fn digest(owner: &str, seed: &str) -> String {
    let hash = Sha256::digest(format!("{owner}\0{seed}").as_bytes());
    format!("responses_{}", &hex::encode(hash)[..24])
}

/// Conversation key for a request that does not continue a stored response.
///
/// Prefers a client session identity, then `prompt_cache_key`, then the
/// first user message, always scoped to the caller. A matching key only lets
/// a turn reuse upstream state when the replayed history also matches; any
/// mismatch rebuilds the upstream session.
fn derive_thread_id(headers: &HeaderMap, owner: &str, plan: &TurnPlan) -> String {
    let session = [
        "x-thread-id",
        "x-session-affinity",
        "x-opencode-session",
        "x-session-id",
        "session_id",
    ]
    .iter()
    .find_map(|name| crate::header_value(headers, name).map(|value| format!("{name}:{value}")));
    let seed = session
        .or_else(|| {
            plan.prompt_cache_key
                .as_ref()
                .map(|key| format!("prompt_cache_key:{key}"))
        })
        .or_else(|| {
            plan.input
                .iter()
                .find_map(|item| match item {
                    Item::Message {
                        role: Role::User,
                        content,
                    } => Some(text_of(content)).filter(|text| !text.is_empty()),
                    _ => None,
                })
                .map(|text| format!("first_user:{text}"))
        })
        .unwrap_or_else(|| format!("fresh:{}", new_id("thread")));
    digest(owner, &seed)
}

/// `POST /v1/responses`.
pub async fn create(State(state): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    let owner = match owner_of(&state, &headers) {
        Ok(owner) => owner,
        Err(response) => return response,
    };
    let request: CreateRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(error) => {
            // serde messages can quote request values; report only the position.
            let message = if error.is_data() {
                format!(
                    "request body has an invalid field type or value at line {} column {}",
                    error.line(),
                    error.column()
                )
            } else {
                format!(
                    "request body is not valid JSON (line {} column {})",
                    error.line(),
                    error.column()
                )
            };
            state.stats.errors.fetch_add(1, Ordering::Relaxed);
            return ProxyError::new(message, StatusCode::BAD_REQUEST, "invalid_request")
                .into_response();
        }
    };
    let plan = match TurnPlan::from_request(request) {
        Ok(plan) => plan,
        Err(error) => {
            state.stats.errors.fetch_add(1, Ordering::Relaxed);
            return request_error(error);
        }
    };
    let runtime = state.responses.clone();
    let model = plan
        .model
        .clone()
        .unwrap_or_else(|| state.settings.model_name.clone());
    let Some((channel, capabilities)) = runtime.for_model(&model) else {
        state.stats.errors.fetch_add(1, Ordering::Relaxed);
        return ProxyError::new(
            "Unknown model; see /v1/models",
            StatusCode::BAD_REQUEST,
            "model_not_found",
        )
        .with_param("model")
        .into_response();
    };
    // `max_output_tokens` is enforced by the proxy where a channel has a local
    // budget. A channel without a budget remains fail-closed.
    let channel_limit = state
        .settings
        .channel_max_output_tokens
        .get(channel.id())
        .copied();
    if let Some(requested) = plan.echo.max_output_tokens {
        match channel_limit {
            Some(limit) if requested > i64::from(limit) => {
                state.stats.errors.fetch_add(1, Ordering::Relaxed);
                return ProxyError::new(
                    format!(
                        "max_output_tokens {requested} exceeds this channel's limit of {limit}"
                    ),
                    StatusCode::BAD_REQUEST,
                    "max_output_tokens_exceeded",
                )
                .with_param("max_output_tokens")
                .into_response();
            }
            Some(_) => {}
            None if !capabilities.enabled(Capability::MaxOutputTokens) => {
                state.stats.errors.fetch_add(1, Ordering::Relaxed);
                return ProxyError::new(
                    format!(
                        "'max_output_tokens' requires capability 'max_output_tokens', which channel '{}' does not support",
                        channel.id()
                    ),
                    StatusCode::BAD_REQUEST,
                    "unsupported_parameter",
                )
                .with_param("max_output_tokens")
                .into_response();
            }
            None => {}
        }
    }
    let ignored_sampling = state.settings.fresh_session_per_request;
    let requirement = plan.unmet(&capabilities).filter(|requirement| {
        if requirement.capability == Capability::MaxOutputTokens && channel_limit.is_some() {
            return false;
        }
        !(ignored_sampling && requirement.capability == Capability::SamplingControls)
    });
    if let Some(requirement) = requirement {
        state.stats.errors.fetch_add(1, Ordering::Relaxed);
        return request_error(gate(&capabilities, requirement));
    }
    if let Err(error) = validate_tool_schemas(&plan) {
        state.stats.errors.fetch_add(1, Ordering::Relaxed);
        return request_error(error);
    }
    // `store` is honoured only when the channel enables stored responses.
    let store = plan.store_requested && capabilities.enabled(Capability::StoredResponses);
    let reasoning = capabilities.enabled(Capability::Reasoning);

    // An unknown parent, or one owned by another key, is a request error.
    let parent = match plan.previous_response_id.as_deref() {
        None => None,
        Some(id) => match state.store.get_response(id, &owner) {
            Ok(Some(record)) => Some(record),
            Ok(None) => {
                state.stats.errors.fetch_add(1, Ordering::Relaxed);
                return ProxyError::new(
                    "Previous response not found",
                    StatusCode::BAD_REQUEST,
                    "previous_response_not_found",
                )
                .with_param("previous_response_id")
                .into_response();
            }
            Err(error) => {
                state.stats.errors.fetch_add(1, Ordering::Relaxed);
                return backend_error(&format!("read previous response: {error}"));
            }
        },
    };
    let mut transcript: Vec<Item> = match &parent {
        None => Vec::new(),
        Some(record) => match serde_json::from_str(&record.transcript_json) {
            Ok(items) => items,
            Err(error) => {
                state.stats.errors.fetch_add(1, Ordering::Relaxed);
                return backend_error(&format!("decode stored transcript: {error}"));
            }
        },
    };
    // Replayed context is gated like fresh input: a narrowed profile must not
    // be bypassed by content an earlier configuration accepted.
    if let Some(requirement) = replay_requirement(&transcript, &capabilities) {
        state.stats.errors.fetch_add(1, Ordering::Relaxed);
        return request_error(gate(&capabilities, &requirement));
    }
    transcript.extend(plan.input.iter().cloned());
    let thread_id = parent
        .as_ref()
        .map(|record| record.thread_id.clone())
        .unwrap_or_else(|| derive_thread_id(&headers, &owner, &plan));

    let response_id = new_id("resp");
    let encoder = EventEncoder::new(&response_id, &model, &plan, store);
    let job = Job {
        thread_id: thread_id.clone(),
        owner_id: owner,
        response_id,
        model: model.clone(),
        store,
        stream: plan.stream,
        reasoning,
        parallel_tool_calls: plan.parallel_tool_calls,
        previous_response_id: plan.previous_response_id.clone(),
        transcript: transcript.clone(),
    };
    let turn = TurnRequest {
        thread_id,
        model,
        instructions: plan.instructions.clone(),
        transcript,
        tools: plan.tools.clone(),
        hosted_tools: plan
            .echo
            .tools
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|tool| tool.get("type").and_then(Value::as_str).map(str::to_string))
            .collect(),
        tool_choice: plan.tool_choice.clone(),
        parallel_tool_calls: plan.parallel_tool_calls,
        // Without the capability the channel is asked not to reason at all,
        // even when its own default would.
        reasoning_effort: if reasoning {
            plan.reasoning_effort
        } else {
            Some(ReasoningEffort::None)
        },
        verbosity: plan.verbosity,
        temperature: plan.echo.temperature,
        top_p: plan.echo.top_p,
        max_output_tokens: plan.echo.max_output_tokens,
    };

    let (tx, mut rx) = mpsc::channel::<Outbound>(64);
    state.stats.total.fetch_add(1, Ordering::Relaxed);
    tokio::spawn(run(
        state.clone(),
        runtime.channel.clone(),
        job,
        turn,
        encoder,
        tx,
    ));

    match rx.recv().await {
        Some(Outbound::Started) => {}
        Some(Outbound::Rejected(response)) => return response,
        _ => return backend_error("Responses turn ended before it started"),
    }
    if plan.stream {
        sse(&state, rx)
    } else {
        collect(rx).await
    }
}

fn sse(state: &AppState, rx: mpsc::Receiver<Outbound>) -> Response {
    let stream = ReceiverStream::new(rx).filter_map(|message| match message {
        Outbound::Event(event) => Some(Ok::<Event, Infallible>(
            Event::default()
                .event(event.kind)
                .data(event.data.to_string()),
        )),
        _ => None,
    });
    let interval = state.settings.heartbeat_interval_secs;
    let interval = if interval.is_finite() && interval > 0.0 {
        std::time::Duration::from_secs_f64(interval)
    } else {
        std::time::Duration::from_secs(10)
    };
    let mut response = Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(interval).text("keep-alive"))
        .into_response();
    response.headers_mut().insert(
        "x-accel-buffering",
        axum::http::HeaderValue::from_static("no"),
    );
    response
}

async fn collect(mut rx: mpsc::Receiver<Outbound>) -> Response {
    while let Some(message) = rx.recv().await {
        if let Outbound::Finished(result) = message {
            return match result {
                Ok(body) => (StatusCode::OK, Json(body)).into_response(),
                Err(response) => response,
            };
        }
    }
    backend_error("Responses turn ended without a result")
}

/// Send encoder events when streaming. Returns false when the client is gone.
async fn send_events(out: &mpsc::Sender<Outbound>, stream: bool, events: Vec<StreamEvent>) -> bool {
    if !stream {
        return true;
    }
    for event in events {
        if out.send(Outbound::Event(event)).await.is_err() {
            return false;
        }
    }
    true
}

/// Mark the response started: HTTP 200 for the handler, then
/// `response.created` / `response.in_progress` for streaming clients.
async fn ensure_started(
    started: &mut bool,
    out: &mpsc::Sender<Outbound>,
    encoder: &mut EventEncoder,
    stream: bool,
) -> bool {
    if *started {
        return true;
    }
    *started = true;
    if out.send(Outbound::Started).await.is_err() {
        return false;
    }
    let events = encoder.start();
    send_events(out, stream, events).await
}

/// Encode one channel event. Reasoning is forwarded only when the active
/// capabilities enable it, whatever the channel emits.
async fn forward(
    event: TurnEvent,
    started: &mut bool,
    out: &mpsc::Sender<Outbound>,
    encoder: &mut EventEncoder,
    job: &Job,
) -> bool {
    if !ensure_started(started, out, encoder, job.stream).await {
        return false;
    }
    let events = match event {
        TurnEvent::Started => return true,
        TurnEvent::Text(text) => encoder.text_delta(&text),
        TurnEvent::Reasoning(text) if job.reasoning => encoder.reasoning_delta(&text),
        TurnEvent::Reasoning(_) => return true,
        TurnEvent::Hosted(result) => encoder.hosted_tool_completed(result),
    };
    send_events(out, job.stream, events).await
}

/// Responses error code for a failure after the response started.
fn failure_code(error: &ProxyError) -> &'static str {
    if error.status == StatusCode::TOO_MANY_REQUESTS {
        "rate_limit_exceeded"
    } else if error.code == "invalid_image" {
        "invalid_image"
    } else {
        "server_error"
    }
}

async fn fail(
    out: &mpsc::Sender<Outbound>,
    encoder: &mut EventEncoder,
    stream: bool,
    error: ProxyError,
) {
    if stream {
        let event = encoder.fail(failure_code(&error), &error.message);
        let _ = out.send(Outbound::Event(event)).await;
    } else {
        let _ = out
            .send(Outbound::Finished(Err(error.into_response())))
            .await;
    }
}

/// One turn: lock, generate, encode, commit. Runs detached from the handler
/// so a slow upstream never blocks the HTTP layer, and stops when the client
/// disconnects.
async fn run(
    state: AppState,
    channel: Arc<dyn ResponsesChannel>,
    job: Job,
    turn: TurnRequest,
    mut encoder: EventEncoder,
    out: mpsc::Sender<Outbound>,
) {
    // Held through commit: a second turn on the same thread waits instead of
    // reading a cursor this turn is about to replace.
    let _turn_lock = state.turn_guard(&job.thread_id).await;
    let (events_tx, mut events_rx) = mpsc::channel::<TurnEvent>(64);
    let mut adapter = channel.run_turn(state.clone(), turn, events_tx);
    let mut started = false;
    let mut events_open = true;
    let result = loop {
        tokio::select! {
            biased;
            _ = out.closed() => return,
            event = events_rx.recv(), if events_open => match event {
                Some(event) => {
                    if !forward(event, &mut started, &out, &mut encoder, &job).await {
                        return;
                    }
                }
                None => events_open = false,
            },
            result = &mut adapter => break result,
        }
    };
    while let Ok(event) = events_rx.try_recv() {
        if !forward(event, &mut started, &out, &mut encoder, &job).await {
            return;
        }
    }

    let outcome = match result {
        Err(TurnError::Cancelled) => return,
        Err(error) => {
            state.stats.errors.fetch_add(1, Ordering::Relaxed);
            if !started {
                let response = match error {
                    TurnError::Proxy(error) => error.into_response(),
                    TurnError::Http(response) => response,
                    TurnError::Cancelled => return,
                };
                let _ = out.send(Outbound::Rejected(response)).await;
                return;
            }
            let error = match error {
                TurnError::Proxy(error) => error,
                TurnError::Http(response) => ProxyError::new(
                    "Backend request failed",
                    if response.status().is_success() {
                        StatusCode::BAD_GATEWAY
                    } else {
                        response.status()
                    },
                    "backend_error",
                ),
                TurnError::Cancelled => return,
            };
            fail(&out, &mut encoder, job.stream, error).await;
            return;
        }
        Ok(outcome) => outcome,
    };
    if !ensure_started(&mut started, &out, &mut encoder, job.stream).await {
        return;
    }
    if !job.parallel_tool_calls && outcome.calls.len() > 1 {
        state.stats.errors.fetch_add(1, Ordering::Relaxed);
        fail(
            &out,
            &mut encoder,
            job.stream,
            ProxyError::new(
                "Backend produced multiple calls with parallel_tool_calls=false",
                StatusCode::BAD_GATEWAY,
                "tool_repair_failed",
            ),
        )
        .await;
        return;
    }
    let finish = encoder.finish(&outcome.calls, outcome.usage);
    if !send_events(&out, job.stream, finish).await || out.is_closed() {
        return;
    }
    // A truncated turn is still persisted: the checkpoint keeps the remote
    // session usable, so the next request continues instead of replaying.
    let truncated = outcome.truncated.clone();
    let response = match truncated.as_deref() {
        Some(reason) => encoder.incomplete_response(reason),
        None => encoder.completed_response(),
    };
    if let Err(error) = commit(&state, &job, &response, &encoder, &outcome) {
        tracing::error!(thread = %job.thread_id, error = %error, "could not commit Responses state");
        state.stats.errors.fetch_add(1, Ordering::Relaxed);
        fail(
            &out,
            &mut encoder,
            job.stream,
            ProxyError::new(
                "Could not save conversation state",
                StatusCode::INTERNAL_SERVER_ERROR,
                "state_commit_failed",
            ),
        )
        .await;
        return;
    }
    match (truncated.as_deref(), job.stream) {
        (Some(reason), true) => {
            let incomplete = encoder.incomplete(reason);
            let _ = out.send(Outbound::Event(incomplete)).await;
        }
        (Some(_), false) => {
            let _ = out.send(Outbound::Finished(Ok(response))).await;
        }
        (None, true) => {
            let completed = encoder.complete();
            let _ = out.send(Outbound::Event(completed)).await;
        }
        (None, false) => {
            let _ = out.send(Outbound::Finished(Ok(response))).await;
        }
    }
    state.stats.ok.fetch_add(1, Ordering::Relaxed);
}

/// Persist a successful turn: the response resource (when stored) and the
/// channel checkpoint, atomically when both exist.
fn commit(
    state: &AppState,
    job: &Job,
    response: &Value,
    encoder: &EventEncoder,
    outcome: &TurnOutcome,
) -> Result<(), String> {
    if job.store {
        let mut transcript = job.transcript.clone();
        transcript.extend(encoder.transcript_items());
        let record = ResponseRecord {
            response_id: job.response_id.clone(),
            owner_id: job.owner_id.clone(),
            previous_response_id: job.previous_response_id.clone(),
            thread_id: job.thread_id.clone(),
            model: job.model.clone(),
            status: match outcome.truncated {
                Some(_) => "incomplete".into(),
                None => "completed".into(),
            },
            response_json: response.to_string(),
            transcript_json: serde_json::to_string(&transcript)
                .map_err(|error| format!("encode transcript: {error}"))?,
        };
        let saved = match &outcome.checkpoint {
            Some(checkpoint) => state.store.save_response_checkpoint(
                &job.thread_id,
                &checkpoint.resume,
                &checkpoint.history,
                &record,
            ),
            None => state.store.save_response(&record),
        };
        saved.map_err(|error| error.to_string())?;
    } else if let Some(checkpoint) = &outcome.checkpoint {
        // No conversation content is stored: only the resume cursor and the
        // history fingerprint, exactly like Chat Completions.
        state
            .store
            .save_thread_checkpoint(&job.thread_id, &checkpoint.resume, &checkpoint.history)
            .map_err(|error| error.to_string())?;
    }
    if let Some(checkpoint) = &outcome.checkpoint {
        state.threads.lock().unwrap().insert(
            job.thread_id.clone(),
            ThreadState {
                session_id: checkpoint.resume.session_id.clone(),
                parent_message_id: checkpoint
                    .resume
                    .parent_message_id
                    .as_deref()
                    .and_then(|id| id.parse().ok()),
                exchanges: 0,
            },
        );
    }
    Ok(())
}

/// `GET /v1/responses/{response_id}`.
pub async fn retrieve(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(response_id): Path<String>,
    Query(query): Query<HashMap<String, String>>,
) -> Response {
    let owner = match owner_of(&state, &headers) {
        Ok(owner) => owner,
        Err(response) => return response,
    };
    if query.get("stream").is_some_and(|value| value == "true") {
        return request_error(RequestError::unsupported(
            "stream",
            "resuming a response stream is not supported",
        ));
    }
    match state.store.get_response(&response_id, &owner) {
        Ok(Some(record)) => match serde_json::from_str::<Value>(&record.response_json) {
            Ok(value) => (StatusCode::OK, Json(value)).into_response(),
            Err(error) => backend_error(&format!("decode stored response: {error}")),
        },
        Ok(None) => not_found("response_id", "Response not found"),
        Err(error) => backend_error(&format!("read response: {error}")),
    }
}

/// `DELETE /v1/responses/{response_id}`.
pub async fn delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(response_id): Path<String>,
) -> Response {
    let owner = match owner_of(&state, &headers) {
        Ok(owner) => owner,
        Err(response) => return response,
    };
    match state.store.delete_response(&response_id, &owner) {
        Ok(true) => (
            StatusCode::OK,
            Json(json!({"id": response_id, "object": "response.deleted", "deleted": true})),
        )
            .into_response(),
        Ok(false) => not_found("response_id", "Response not found"),
        Err(error) => backend_error(&format!("delete response: {error}")),
    }
}
