//! DeepSeek web channel for `/v1/responses`.
//!
//! The adapter maps one [`TurnRequest`] onto the same DeepSeek turn that Chat
//! Completions runs, so both endpoints share the proven machinery at the
//! crate root: history fingerprints and fixed-account turns, prompt
//! flattening, conservative replay into a fresh remote session, throttle
//! retries, cursor recovery and schema-validated tool repair. Chat
//! Completions code is called, never changed.
//!
//! Only the stream decoder is local: [`drain`] turns DeepSeek SSE into
//! [`TurnEvent`]s instead of OpenAI chunks, so Chat's `pump` stays untouched.
//!
//! The capability ceiling in [`ceiling`] is this adapter's contract. Every
//! capability is declared explicitly, so an unsupported one is a visible
//! decision rather than an omission.

use super::channel::{
    Checkpoint, ResponsesChannel, TurnError, TurnEvent, TurnOutcome, TurnRequest,
};
use crate::{
    backend_error, backoff_delay, build_prompts, completion_body, completion_with_retry,
    delta_messages, dialect, error_for, estimate_tokens, find_token_usage, first_hint, images,
    invalid, is_supported_model, is_upstream_finished, persist_cursor, pool, queue_error,
    rebuild_session, resolve_tools, resolve_upstream_model_type, route_all, route_unavailable,
    store, AppState, ChatRequest, CompletionError, Msg, ProxyError, Recovery, ThreadState,
    TurnHistory,
};
use axum::http::StatusCode;
use futures_util::{future::BoxFuture, FutureExt, StreamExt};
use protocol::{classify_line, route_event, Channel, FragmentRouter, LineKind};
use responses_core::capability::{Capability, CapabilitySet, CapabilityStatus, Provider};
use responses_core::events::{FunctionCall, Usage};
use responses_core::plan::{FunctionTool, ReasoningEffort, ToolChoice, Verbosity};
use responses_core::transcript::{Item, Part, Role};
use serde_json::{json, Value};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Channel id, matched against a capability profile's `channel`.
pub const CHANNEL_ID: &str = "deepseek_web";

/// The DeepSeek web (private API) upstream.
pub struct DeepSeekWeb;

/// Everything this adapter implements, and an explicit reason for
/// everything it does not.
pub fn ceiling() -> CapabilitySet {
    use Capability as C;
    let supported: [(Capability, Provider, &str); 11] = [
        (C::Text, Provider::Upstream, "flattened_prompt"),
        (C::Streaming, Provider::Proxy, "upstream_sse_reencoded"),
        (C::NonStreaming, Provider::Proxy, "collected_stream"),
        (C::ImageInput, Provider::Upstream, "single_image_per_turn"),
        (C::FunctionTools, Provider::Proxy, "prompted_tool_protocol"),
        (C::ToolChoice, Provider::Proxy, "prompted_directive"),
        (C::Reasoning, Provider::Upstream, "thinking_toggle"),
        (C::Verbosity, Provider::Proxy, "prompted_directive"),
        // The upstream has no such parameter; the proxy stops forwarding at
        // the budget and reports `incomplete_details.reason=max_output_tokens`.
        (C::MaxOutputTokens, Provider::Proxy, "local_token_budget"),
        (C::StoredResponses, Provider::Proxy, "local_transcript"),
        (
            C::PreviousResponse,
            Provider::Proxy,
            "local_transcript_replay",
        ),
    ];
    let unsupported: [(Capability, &str); 19] = [
        (C::FileInput, "no_upstream_file_input"),
        (C::CustomTools, "no_custom_tool_grammar"),
        (C::WebSearch, "hosted_tool_not_implemented"),
        (C::FileSearch, "hosted_tool_not_implemented"),
        (C::CodeInterpreter, "hosted_tool_not_implemented"),
        (C::Mcp, "hosted_tool_not_implemented"),
        (C::ImageGeneration, "hosted_tool_not_implemented"),
        (C::ComputerUse, "hosted_tool_not_implemented"),
        (C::Shell, "hosted_tool_not_implemented"),
        (C::StructuredOutput, "no_upstream_schema_enforcement"),
        (C::SamplingControls, "no_upstream_sampling_parameters"),
        (C::Logprobs, "no_upstream_logprobs"),
        (C::Conversations, "not_implemented"),
        (C::Background, "not_implemented"),
        (C::StreamResume, "not_implemented"),
        (C::AutoTruncation, "replays_into_fresh_session_instead"),
        (C::Compaction, "not_implemented"),
        (C::PromptTemplates, "not_implemented"),
        (C::ExactUsage, "output_tokens_estimated_locally"),
    ];
    let mut set = CapabilitySet::new(CHANNEL_ID);
    for (capability, provider, mode) in supported {
        set = set.with(capability, CapabilityStatus::supported(provider, mode));
    }
    for (capability, reason) in unsupported {
        set = set.with(capability, CapabilityStatus::unsupported(reason));
    }
    set
}

impl ResponsesChannel for DeepSeekWeb {
    fn id(&self) -> &'static str {
        CHANNEL_ID
    }

    fn capabilities(&self) -> CapabilitySet {
        ceiling()
    }

    fn supports_model(&self, model: &str) -> bool {
        is_supported_model(model)
    }

    fn run_turn(
        &self,
        state: AppState,
        turn: TurnRequest,
        events: mpsc::Sender<TurnEvent>,
    ) -> BoxFuture<'static, Result<TurnOutcome, TurnError>> {
        run(state, turn, events).boxed()
    }
}

/// The chat-shaped request the shared helpers consume.
///
/// An assistant message and the function calls that follow it become one
/// assistant message with `tool_calls`, which is exactly how
/// `TurnHistory::completed` records an answer. A stored answer therefore
/// replays into the fingerprint it was checkpointed with, and the next turn
/// can continue the remote session instead of replaying it.
fn chat_request(turn: &TurnRequest) -> Result<ChatRequest, ProxyError> {
    let mut messages: Vec<Msg> = Vec::with_capacity(turn.transcript.len() + 1);
    if let Some(instructions) = &turn.instructions {
        messages.push(message("system", Some(json!(instructions))));
    }
    for item in &turn.transcript {
        match item {
            Item::Message { role, content } => {
                let role = match role {
                    Role::System => "system",
                    Role::Developer => "developer",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                };
                messages.push(message(role, chat_content(content)?));
            }
            Item::FunctionCall {
                call_id,
                name,
                arguments,
            } => {
                let call = json!({
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                });
                let joins_previous = messages
                    .last()
                    .is_some_and(|previous| previous.role == "assistant");
                if joins_previous {
                    if let Some(previous) = messages.last_mut() {
                        previous.tool_calls.push(call);
                    }
                } else {
                    let mut assistant = message("assistant", None);
                    assistant.tool_calls.push(call);
                    messages.push(assistant);
                }
            }
            Item::FunctionCallOutput { call_id, output } => {
                let mut result = message("tool", Some(json!(output)));
                result.tool_call_id = Some(call_id.clone());
                messages.push(result);
            }
        }
    }
    Ok(ChatRequest {
        model: Some(turn.model.clone()),
        thread_id: None,
        prompt_cache_key: None,
        messages,
        stream: true,
        reasoning_effort: None,
        thinking: None,
        reasoning: None,
        tools: turn.tools.iter().map(chat_tool).collect(),
        tool_choice: chat_tool_choice(&turn.tool_choice),
        parallel_tool_calls: turn.parallel_tool_calls,
        stream_options: None,
    })
}

fn message(role: &str, content: Option<Value>) -> Msg {
    Msg {
        role: role.to_string(),
        content,
        tool_calls: Vec::new(),
        tool_call_id: None,
        name: None,
    }
}

/// Text-only content becomes a string; content with an image becomes the
/// Chat Completions part list that image intake and fingerprints read.
fn chat_content(parts: &[Part]) -> Result<Option<Value>, ProxyError> {
    if parts.iter().all(|part| matches!(part, Part::Text { .. })) {
        let text: Vec<&str> = parts
            .iter()
            .filter_map(|part| match part {
                Part::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        return Ok(Some(json!(text.join("\n"))));
    }
    let mut out = Vec::with_capacity(parts.len());
    for part in parts {
        match part {
            Part::Text { text } => out.push(json!({"type": "text", "text": text})),
            Part::Image { url, .. } => {
                out.push(json!({"type": "image_url", "image_url": {"url": url}}))
            }
            // Gated by `file_input`, which this channel does not declare; a
            // replayed transcript is gated the same way before it gets here.
            Part::File { .. } => {
                return Err(ProxyError::new(
                    "input_file is not supported by this channel",
                    StatusCode::BAD_REQUEST,
                    "unsupported_parameter",
                )
                .with_param("input"))
            }
        }
    }
    Ok(Some(Value::Array(out)))
}

fn chat_tool(tool: &FunctionTool) -> Value {
    let mut function = json!({"name": tool.name, "parameters": tool.parameters});
    if let Some(description) = &tool.description {
        function["description"] = json!(description);
    }
    json!({"type": "function", "function": function})
}

fn chat_tool_choice(choice: &ToolChoice) -> Option<Value> {
    match choice {
        ToolChoice::Auto => None,
        ToolChoice::None => Some(json!("none")),
        ToolChoice::Required => Some(json!("required")),
        ToolChoice::Function(name) => Some(json!({"type": "function", "function": {"name": name}})),
    }
}

/// DeepSeek's only reasoning control is the thinking toggle.
fn thinking_for(effort: Option<ReasoningEffort>, default: bool) -> bool {
    match effort {
        Some(ReasoningEffort::None | ReasoningEffort::Minimal) => false,
        Some(_) => true,
        None => default,
    }
}

/// DeepSeek has no verbosity parameter; the preference becomes a per-turn
/// instruction. `medium` is the model's natural register and adds nothing.
fn verbosity_directive(verbosity: Verbosity) -> Option<&'static str> {
    match verbosity {
        Verbosity::Low => Some("\nAnswer concisely: keep the reply as short as the task allows."),
        Verbosity::Medium => None,
        Verbosity::High => {
            Some("\nAnswer thoroughly, with complete explanations and relevant detail.")
        }
    }
}

fn store_failure(what: &str, error: store::StoreError) -> TurnError {
    TurnError::Http(backend_error(&format!("{what}: {error}")))
}

/// Fail a turn without losing the newest upstream cursor, like Chat's
/// `fail_turn`. The trusted history marker was cleared before upstream work,
/// so the saved cursor can never certify the failed answer.
fn fail_with_cursor(
    state: &AppState,
    key: &str,
    session_id: &str,
    cursor: Option<i64>,
    error: impl Into<TurnError>,
) -> TurnError {
    persist_cursor(&state.threads, &state.store, key, session_id, cursor);
    error.into()
}

/// Run one turn on the DeepSeek web upstream.
///
/// The steps and recovery rules mirror `stream_response` in `main.rs`; the
/// comments there explain why each rule exists. The differences are
/// Responses-specific: errors are returned instead of written into a chat
/// stream, a failed image upload fails the turn instead of silently dropping
/// the image, web search is never enabled (no hosted tool is declared), and
/// the verbosity directive is appended to the prompt.
async fn run(
    state: AppState,
    turn: TurnRequest,
    events: mpsc::Sender<TurnEvent>,
) -> Result<TurnOutcome, TurnError> {
    let started = Instant::now();
    let limit = Duration::from_secs_f64(state.settings.request_timeout_secs);
    let key = turn.thread_id.clone();
    let req = chat_request(&turn)?;

    // One account serves the whole turn: uploads, retries, rebuilds, repair.
    let conn = state
        .pool
        .route_checked(&key)
        .await
        .map_err(route_unavailable)?;
    let previous = state
        .store
        .get_thread(&key)
        .map_err(|error| store_failure("read thread", error))?
        .map(|resume| ThreadState {
            session_id: resume.session_id,
            parent_message_id: resume.parent_message_id.and_then(|p| p.parse().ok()),
            exchanges: 0,
        });
    let saved_history = state
        .store
        .get_thread_history(&key)
        .map_err(|error| store_failure("read thread history", error))?;
    let mut history = TurnHistory::new(&req, &turn.model);
    let continuation = history.can_continue(previous.as_ref(), saved_history.as_ref(), &conn.email);
    let existing = if continuation { previous } else { None };
    tracing::info!(
        thread = %key,
        model = %turn.model,
        items = turn.transcript.len(),
        continuation,
        "responses turn in"
    );

    let system_extra = dialect::inject_tool_descriptions("", &req.tools);
    let (specs, required) = resolve_tools(&req);
    // A rebuilt branch replays all of its inputs, so it reloads their image.
    let inputs: &[Msg] = if continuation {
        delta_messages(&req.messages)
    } else {
        &req.messages
    };
    let loaded = images::load(inputs, state.settings.image_max_bytes)
        .await
        .map_err(|error| {
            ProxyError::new(error.0, StatusCode::BAD_REQUEST, "invalid_image").with_param("input")
        })?;
    let image_note = loaded.note();
    state
        .stats
        .images_dropped
        .fetch_add(loaded.dropped as u64, Ordering::Relaxed);
    let (mut prompt, mut replay) = build_prompts(
        &req,
        existing.as_ref(),
        &system_extra,
        image_note.as_deref(),
    );
    if prompt.is_empty() {
        return Err(invalid(
            "input",
            "no user message or function call output to send in this turn",
        )
        .into());
    }
    let can_recover = continuation && !replay.is_empty();
    if let Some(directive) = turn.verbosity.and_then(verbosity_directive) {
        prompt.push_str(directive);
        replay.push_str(directive);
    }

    // Invalidate BEFORE upstream work: an interrupted or failed answer must
    // never be mistaken later for a trusted head.
    state
        .store
        .clear_thread_history(&key)
        .map_err(|error| store_failure("invalidate thread history", error))?;
    let thread = match existing {
        Some(thread) => thread,
        None => rebuild_session(&state, &conn, &key, "new or divergent request history")
            .await
            .map_err(TurnError::Http)?,
    };
    state
        .threads
        .lock()
        .unwrap()
        .insert(key.clone(), thread.clone());

    let ref_file_ids = match loaded.data {
        None => Vec::new(),
        Some(bytes) => {
            let _slot = conn
                .acquire()
                .await
                .map_err(|error| TurnError::Http(queue_error(&error)))?;
            match conn
                .client()
                .await
                .upload_image(&bytes, resolve_upstream_model_type(&turn.model))
                .await
            {
                Ok(uploaded) => vec![uploaded.file_id],
                Err(error) => {
                    tracing::warn!(thread = %key, error = %error, "image upload failed");
                    return Err(ProxyError::new(
                        "Image upload to the backend failed",
                        StatusCode::BAD_GATEWAY,
                        "backend_error",
                    )
                    .into());
                }
            }
        }
    };

    let mut body = completion_body(
        &thread,
        &prompt,
        thinking_for(turn.reasoning_effort, state.settings.thinking_default),
        false,
        &ref_file_ids,
        &turn.model,
    );
    let mut next = match completion_with_retry(&state, &conn, &key, &body).await {
        Ok(sent) => Some(sent),
        Err(error) => return Err(error_for(&state, &key, error).await.into()),
    };

    let mut session_id = body.chat_session_id.clone();
    let mut cursor = body.parent_message_id;
    let mut attempt: u32 = 0;
    let mut context_recovered = false;
    let mut cursor_recovered = false;
    let mut announced = false;
    // The upstream cannot enforce an output limit, so `drain` stops forwarding
    // once this many tokens (reasoning included) have been produced.
    let output_budget = turn
        .max_output_tokens
        .and_then(|limit| u64::try_from(limit).ok());

    let drained = loop {
        let (resp, slot) = next.take().expect("attempt in flight");
        let budget = limit.saturating_sub(started.elapsed());
        let drained =
            match tokio::time::timeout(budget, drain(resp, &events, &mut announced, output_budget))
                .await
            {
                Ok(drained) => drained,
                Err(_) => {
                    tracing::error!(
                        thread = %key,
                        timeout_s = state.settings.request_timeout_secs,
                        "turn timed out waiting for upstream"
                    );
                    return Err(fail_with_cursor(
                        &state,
                        &key,
                        &session_id,
                        cursor,
                        ProxyError::new(
                            "Request processing timed out",
                            StatusCode::GATEWAY_TIMEOUT,
                            "request_timeout",
                        ),
                    ));
                }
            };
        if drained.cancelled {
            return Err(TurnError::Cancelled);
        }
        if drained.hint.is_none() && !drained.complete {
            drop(slot);
            return Err(fail_with_cursor(
                &state,
                &key,
                &session_id,
                cursor,
                ProxyError::new(
                    "Backend stream ended before completion",
                    StatusCode::BAD_GATEWAY,
                    "incomplete_response",
                ),
            ));
        }
        let Some(hint) = drained.hint.clone() else {
            // A completed answer without its own cursor must not certify the
            // previous parent as the head of this new history.
            drop(slot);
            cursor = drained.parent;
            break drained;
        };

        // Failed attempt: keep the cursor it started from.
        cursor = body.parent_message_id;
        if protocol::is_invalid_message_id(&hint) && !drained.emitted && !cursor_recovered {
            drop(slot);
            cursor_recovered = true;
            attempt += 1;
            state.stats.retries.fetch_add(1, Ordering::Relaxed);
            if can_recover {
                tracing::warn!(
                    thread = %key,
                    hint = %hint,
                    "stale parent message id; rebuilding the session and replaying the transcript"
                );
                let fresh = match rebuild_session(&state, &conn, &key, &hint).await {
                    Ok(fresh) => fresh,
                    Err(response) => {
                        return Err(fail_with_cursor(
                            &state,
                            &key,
                            &session_id,
                            cursor,
                            TurnError::Http(response),
                        ))
                    }
                };
                body = completion_body(
                    &fresh,
                    &replay,
                    body.thinking_enabled,
                    body.search_enabled,
                    &body.ref_file_ids,
                    &body.model_type,
                );
                session_id = fresh.session_id.clone();
                cursor = None;
            } else {
                tracing::warn!(
                    thread = %key,
                    hint = %hint,
                    "stale parent message id; retrying with a cleared cursor"
                );
                cursor = None;
                body.parent_message_id = None;
            }
        } else if protocol::is_context_limit(&hint) {
            drop(slot);
            if context_recovered || !can_recover || drained.emitted {
                tracing::error!(thread = %key, "upstream hint error: {hint}");
                return Err(fail_with_cursor(
                    &state,
                    &key,
                    &session_id,
                    cursor,
                    ProxyError::new(
                        "Conversation exceeds the backend context window",
                        StatusCode::BAD_GATEWAY,
                        "context_length_exceeded",
                    ),
                ));
            }
            context_recovered = true;
            let fresh = match rebuild_session(&state, &conn, &key, &hint).await {
                Ok(fresh) => fresh,
                Err(response) => {
                    return Err(fail_with_cursor(
                        &state,
                        &key,
                        &session_id,
                        cursor,
                        TurnError::Http(response),
                    ))
                }
            };
            body = completion_body(
                &fresh,
                &replay,
                body.thinking_enabled,
                body.search_enabled,
                &body.ref_file_ids,
                &body.model_type,
            );
            session_id = fresh.session_id.clone();
            cursor = None;
        } else if pool::is_rate_limit(&hint) && !drained.emitted {
            let wait = backoff_delay(&state, attempt);
            if attempt >= state.settings.rate_limit_max_retries {
                drop(slot);
                tracing::error!(thread = %key, retries = attempt, "upstream still throttled: {hint}");
                let error = error_for(
                    &state,
                    &key,
                    CompletionError::RateLimited {
                        retry_after_s: wait.ceil() as u64,
                    },
                )
                .await;
                return Err(fail_with_cursor(&state, &key, &session_id, cursor, error));
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
            // Publish the cooldown before releasing the in-flight permit.
            conn.cool_down(Duration::from_secs_f64(wait)).await;
            drop(slot);
        } else {
            drop(slot);
            tracing::error!(thread = %key, "upstream hint error: {hint}");
            let error = if pool::is_login_dead(&hint) {
                CompletionError::LoginDead(hint.clone())
            } else {
                CompletionError::Other(hint.clone())
            };
            let error = error_for(&state, &key, error).await;
            return Err(fail_with_cursor(&state, &key, &session_id, cursor, error));
        }

        match completion_with_retry(&state, &conn, &key, &body).await {
            Ok(sent) => next = Some(sent),
            Err(error) => {
                let error = error_for(&state, &key, error).await;
                return Err(fail_with_cursor(&state, &key, &session_id, cursor, error));
            }
        }
    };

    let Drained {
        mut filter,
        text,
        mut prose,
        reasoning,
        prompt_tokens,
        truncated,
        ..
    } = drained;

    // Do not test `emitted`: the tool filter can hold every byte back.
    if text.trim().is_empty() && filter.calls.is_empty() {
        tracing::error!(
            thread = %key,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "upstream closed the stream without content or a hint"
        );
        return Err(fail_with_cursor(
            &state,
            &key,
            &session_id,
            cursor,
            ProxyError::new(
                "Backend returned an empty response",
                StatusCode::BAD_GATEWAY,
                "empty_response",
            ),
        ));
    }

    // Validate every call against its schema and run correction turns,
    // chained below the newest upstream message.
    // A truncated turn never executes tool calls: the budget may have cut the
    // block mid-arguments, and a half-parsed call would be worse than none.
    filter.calls.clear();
    let mut recovery = Recovery::new(specs, required, turn.parallel_tool_calls);
    if !truncated {
        for event in dialect::scan(&text) {
            if let Err(error) = recovery.accept(event) {
                return Err(fail_with_cursor(&state, &key, &session_id, cursor, error));
            }
        }
        recovery.require_call();
    }
    for repair_attempt in 1..=state.settings.tool_repair_max_retries {
        if !recovery.has_pending() || cursor.is_none() {
            break;
        }
        let repair_prompt = recovery.begin_repair();
        tracing::warn!(
            thread = %key,
            attempt = repair_attempt,
            expected = recovery.expected,
            "tool repair"
        );
        let head = ThreadState {
            session_id: session_id.clone(),
            parent_message_id: cursor,
            exchanges: 0,
        };
        let repair_body =
            completion_body(&head, &repair_prompt, false, false, &[], &body.model_type);
        let budget = limit.saturating_sub(started.elapsed());
        let repair = async {
            let (resp, _slot) = completion_with_retry(&state, &conn, &key, &repair_body).await?;
            // Keep the account slot until the correction body is drained.
            resp.text()
                .await
                .map_err(|error| CompletionError::Other(error.to_string()))
        };
        let raw = tokio::select! {
            biased;
            _ = events.closed() => return Err(TurnError::Cancelled),
            result = tokio::time::timeout(budget, repair) => match result {
                Ok(Ok(raw)) => raw,
                _ => {
                    recovery.end_repair();
                    break;
                }
            },
        };
        if first_hint(&raw).is_some() || !repair_finished(&raw) {
            recovery.end_repair();
            break;
        }
        let (_, repair_text, repair_mid) = route_all(&raw);
        // The correction is now the newest upstream message. A missing id
        // revokes cursor trust instead of certifying the reply it fixed.
        cursor = repair_mid;
        for event in dialect::scan(&repair_text) {
            if let Err(error) = recovery.accept(event) {
                return Err(fail_with_cursor(&state, &key, &session_id, cursor, error));
            }
        }
        recovery.end_repair();
    }
    if recovery.has_pending() {
        return Err(fail_with_cursor(
            &state,
            &key,
            &session_id,
            cursor,
            ProxyError::new(
                "Could not recover a valid tool response",
                StatusCode::BAD_GATEWAY,
                "tool_repair_failed",
            ),
        ));
    }
    let accepted = recovery.calls;

    // Flush prose the tool filter held back at the very end of the stream.
    let tail = filter.finish();
    if !tail.is_empty() {
        prose.push_str(&tail);
        if events.send(TurnEvent::Text(tail)).await.is_err() {
            return Err(TurnError::Cancelled);
        }
    }

    state
        .stats
        .tool_calls
        .fetch_add(accepted.len() as u64, Ordering::Relaxed);
    // `prose` is exactly the answer text the client received, so the stored
    // transcript replays into this fingerprint.
    let checkpoint = cursor.map(|mid| Checkpoint {
        resume: store::ThreadResume {
            session_id: session_id.clone(),
            parent_message_id: Some(mid.to_string()),
        },
        history: history.completed(&prose, &accepted, &conn.email),
    });
    let reasoning_tokens = estimate_tokens(&reasoning);
    let mut output_tokens = estimate_tokens(&prose) + reasoning_tokens;
    for call in &accepted {
        output_tokens += estimate_tokens(&format!("{}{}", call.name, call.arguments));
    }
    let usage = Usage {
        input_tokens: prompt_tokens,
        cached_tokens: 0,
        output_tokens,
        reasoning_tokens,
    };
    let calls: Vec<FunctionCall> = accepted
        .into_iter()
        .map(|call| FunctionCall {
            name: call.name,
            arguments: call.arguments,
        })
        .collect();
    let truncated = truncated.then(|| "max_output_tokens".to_string());
    tracing::info!(
        thread = %key,
        tool_calls = calls.len(),
        trusted_head = checkpoint.is_some(),
        truncated = truncated.is_some(),
        elapsed_ms = started.elapsed().as_millis() as u64,
        "responses turn done"
    );
    Ok(TurnOutcome {
        calls,
        usage,
        checkpoint,
        truncated,
    })
}

/// True when a buffered correction body reached a completion marker.
fn repair_finished(raw: &str) -> bool {
    raw.lines().any(|line| {
        let mut event = String::new();
        match classify_line(line, &mut event) {
            LineKind::Event(value) => is_upstream_finished(&value),
            _ => line.trim() == "data: [DONE]",
        }
    })
}

/// Result of draining one upstream response.
struct Drained {
    /// Newest upstream message id seen in this response.
    parent: Option<i64>,
    /// A final completion marker arrived, not merely EOF.
    complete: bool,
    /// Fatal hint, or a fixed classification of a non-SSE body.
    hint: Option<String>,
    /// Text reached the consumer; a later hint can no longer be retried.
    emitted: bool,
    /// The consumer went away.
    cancelled: bool,
    filter: dialect::StreamToolFilter,
    /// Every answer fragment, tool blocks included, for the tool scanner.
    text: String,
    /// Answer text forwarded to the consumer.
    prose: String,
    reasoning: String,
    prompt_tokens: u64,
    /// Set once the output budget is exhausted; later text is not forwarded.
    truncated: bool,
}

/// Output budget in tokens, counting reasoning the way OpenAI does.
///
/// `None` means the turn is unbounded.
fn over_budget(prose: &str, reasoning: &str, budget: Option<u64>) -> bool {
    match budget {
        None => false,
        Some(limit) => estimate_tokens(prose) + estimate_tokens(reasoning) >= limit,
    }
}

/// Longest prefix of `text` whose estimate fits in `tokens`.
///
/// Cutting by characters would overshoot for dense scripts, so the prefix is
/// found with the same estimator the budget check uses.
fn trim_to_tokens(text: &str, tokens: u64) -> String {
    if tokens == 0 || text.is_empty() {
        return String::new();
    }
    if estimate_tokens(text) <= tokens {
        return text.to_string();
    }
    let chars: Vec<char> = text.chars().collect();
    let (mut low, mut high) = (0usize, chars.len());
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        let candidate: String = chars[..mid].iter().collect();
        if estimate_tokens(&candidate) <= tokens {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    chars[..low].iter().collect()
}

/// Read one upstream response, forwarding answer and reasoning text.
///
/// `announced` flips on the first routed fragment, when
/// [`TurnEvent::Started`] commits the HTTP response. A hint that arrives
/// before any fragment therefore still reaches the client as a plain HTTP
/// error once retries are exhausted.
async fn drain(
    resp: reqwest::Response,
    events: &mpsc::Sender<TurnEvent>,
    announced: &mut bool,
    budget: Option<u64>,
) -> Drained {
    let mut out = Drained {
        parent: None,
        complete: false,
        hint: None,
        emitted: false,
        cancelled: false,
        filter: dialect::StreamToolFilter::new(),
        text: String::new(),
        prose: String::new(),
        reasoning: String::new(),
        prompt_tokens: 0,
        truncated: false,
    };
    let mut router = FragmentRouter::new();
    let mut fragments = Vec::new();
    // Buffer bytes until a complete line: a UTF-8 scalar may cross chunks.
    let mut pending: Vec<u8> = Vec::new();
    let mut event_name = String::new();
    let mut head: Vec<u8> = Vec::new();
    let mut event_count = 0usize;
    let mut stream = resp.bytes_stream();
    loop {
        let next = tokio::select! {
            biased;
            _ = events.closed() => {
                out.cancelled = true;
                return out;
            }
            next = stream.next() => next,
        };
        let eof = match next {
            Some(Ok(bytes)) => {
                let take = 1024usize.saturating_sub(head.len()).min(bytes.len());
                head.extend_from_slice(&bytes[..take]);
                pending.extend_from_slice(&bytes);
                false
            }
            Some(Err(_)) => {
                out.hint = Some("stream read failed".into());
                return out;
            }
            None => {
                if !pending.is_empty() {
                    pending.push(b'\n');
                }
                true
            }
        };
        while let Some(position) = pending.iter().position(|byte| *byte == b'\n') {
            let line = String::from_utf8_lossy(&pending[..position]).into_owned();
            pending.drain(..=position);
            if line.trim() == "data: [DONE]" {
                out.complete = true;
                continue;
            }
            let value = match classify_line(&line, &mut event_name) {
                LineKind::Ignored => continue,
                LineKind::HintError { content } => {
                    out.complete = false;
                    out.hint = Some(content);
                    return out;
                }
                LineKind::Event(value) => value,
            };
            event_count += 1;
            out.complete |= is_upstream_finished(&value);
            if let Some(mid) = value.get("response_message_id").and_then(Value::as_i64) {
                out.parent = Some(mid);
            }
            if let Some(tokens) = find_token_usage(&value) {
                out.prompt_tokens = tokens;
            }
            fragments.clear();
            route_event(&value, &mut router, &mut fragments);
            for fragment in fragments.drain(..) {
                if !*announced {
                    *announced = true;
                    if events.send(TurnEvent::Started).await.is_err() {
                        out.cancelled = true;
                        return out;
                    }
                }
                // Past the budget the turn is already decided: keep draining so
                // the cursor and account slot stay correct, but forward nothing.
                if out.truncated {
                    continue;
                }
                // Trim the fragment that crosses the budget instead of dropping
                // it whole, so the client still receives what fits.
                let fragment = match budget {
                    Some(limit) => {
                        let room = limit
                            .saturating_sub(estimate_tokens(&out.prose))
                            .saturating_sub(estimate_tokens(&out.reasoning));
                        if room == 0 {
                            out.truncated = true;
                            continue;
                        }
                        let trimmed = trim_to_tokens(&fragment.text, room);
                        if trimmed.len() < fragment.text.len() {
                            out.truncated = true;
                        }
                        if trimmed.is_empty() {
                            continue;
                        }
                        let mut fragment = fragment;
                        fragment.text = trimmed;
                        fragment
                    }
                    None => fragment,
                };
                let event = match fragment.channel {
                    Channel::Text => {
                        out.text.push_str(&fragment.text);
                        let mut safe = out.filter.feed(&fragment.text);
                        // A truncated tail can end inside a tool block the filter
                        // is holding back. Release it: the block cannot execute
                        // anyway, and the prose around it is still useful.
                        if safe.is_empty() && out.truncated {
                            safe = out.filter.finish();
                        }
                        if safe.is_empty() {
                            continue;
                        }
                        out.prose.push_str(&safe);
                        TurnEvent::Text(safe)
                    }
                    Channel::Reasoning => {
                        out.reasoning.push_str(&fragment.text);
                        TurnEvent::Reasoning(fragment.text)
                    }
                };
                out.emitted = true;
                if events.send(event).await.is_err() {
                    out.cancelled = true;
                    return out;
                }
            }
        }
        if eof {
            break;
        }
    }
    if event_count == 0 && !head.is_empty() {
        out.hint = Some(classify_non_sse(&String::from_utf8_lossy(&head)).to_string());
    }
    out
}

/// Fixed hint for a body that was not SSE. Recovery still sees the right
/// class, but no upstream text (which could be model output) reaches the
/// hint and therefore the logs.
fn classify_non_sse(raw: &str) -> &'static str {
    if protocol::is_invalid_message_id(raw) {
        "invalid message id"
    } else if protocol::is_context_limit(raw) {
        "context_length_exceeded"
    } else if pool::is_rate_limit(raw) {
        "rate limit exceeded"
    } else if pool::is_login_dead(raw) {
        "login failed"
    } else if pool::is_invalid_token(raw) {
        "invalid token"
    } else {
        "non-SSE upstream response"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::ToolCall;
    use crate::responses::tests::test_state;
    use crate::responses::Runtime;
    use crate::store::ThreadResume;
    use responses_core::capability::Support;
    use responses_core::events::EventEncoder;
    use responses_core::plan::TurnPlan;
    use std::sync::Arc;

    fn text(value: &str) -> Part {
        Part::Text { text: value.into() }
    }

    fn user(value: &str) -> Item {
        Item::Message {
            role: Role::User,
            content: vec![text(value)],
        }
    }

    fn assistant(value: &str) -> Item {
        Item::Message {
            role: Role::Assistant,
            content: vec![text(value)],
        }
    }

    fn lookup_tool() -> FunctionTool {
        FunctionTool {
            name: "lookup".into(),
            description: Some("Look something up".into()),
            parameters: json!({
                "type": "object",
                "properties": {"q": {"type": "integer"}},
                "required": ["q"],
            }),
            strict: None,
        }
    }

    fn turn(transcript: Vec<Item>) -> TurnRequest {
        TurnRequest {
            thread_id: "responses_test".into(),
            model: "DEFAULT".into(),
            instructions: None,
            transcript,
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: true,
            reasoning_effort: None,
            verbosity: None,
            temperature: None,
            top_p: None,
            max_output_tokens: None,
        }
    }

    fn deepseek_runtime() -> Runtime {
        Runtime::new(Arc::new(DeepSeekWeb), None).unwrap().0
    }

    fn response(raw: &str) -> reqwest::Response {
        reqwest::Response::from(
            axum::http::Response::builder()
                .status(200)
                .header("content-type", "text/event-stream")
                .body(raw.to_string())
                .unwrap(),
        )
    }

    #[test]
    fn ceiling_declares_every_capability_and_only_engine_supported_ones() {
        let ceiling = ceiling();
        for capability in Capability::ALL {
            let status = ceiling
                .entry(capability)
                .unwrap_or_else(|| panic!("'{}' is undeclared", capability.id()));
            if status.support == Support::Supported {
                assert!(capability.engine_supported(), "'{}'", capability.id());
            }
        }
        let (runtime, warnings) = Runtime::new(Arc::new(DeepSeekWeb), None).unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(
            runtime.capabilities.enabled_ids(),
            [
                "text",
                "streaming",
                "non_streaming",
                "image_input",
                "function_tools",
                "tool_choice",
                "reasoning",
                "verbosity",
                "max_output_tokens",
                "stored_responses",
                "previous_response",
            ]
        );
    }

    #[test]
    fn transcript_maps_onto_the_chat_turn_shape() {
        let mut request = turn(vec![
            Item::Message {
                role: Role::Developer,
                content: vec![text("be careful")],
            },
            Item::Message {
                role: Role::User,
                content: vec![
                    text("look"),
                    Part::Image {
                        url: "data:image/png;base64,AAAA".into(),
                        detail: "auto".into(),
                    },
                ],
            },
            assistant("Checking."),
            Item::FunctionCall {
                call_id: "call_a".into(),
                name: "lookup".into(),
                arguments: "{\"q\":1}".into(),
            },
            Item::FunctionCall {
                call_id: "call_b".into(),
                name: "lookup".into(),
                arguments: "{\"q\":2}".into(),
            },
            Item::FunctionCallOutput {
                call_id: "call_a".into(),
                output: "one".into(),
            },
            Item::FunctionCallOutput {
                call_id: "call_b".into(),
                output: "two".into(),
            },
        ]);
        request.instructions = Some("system rules".into());
        request.tools = vec![lookup_tool()];
        request.tool_choice = ToolChoice::Function("lookup".into());

        let chat = chat_request(&request).unwrap();
        let roles: Vec<&str> = chat.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(
            roles,
            ["system", "developer", "user", "assistant", "tool", "tool"]
        );
        assert_eq!(chat.messages[0].content, Some(json!("system rules")));
        assert_eq!(
            chat.messages[2].content,
            Some(json!([
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ]))
        );
        let answer = &chat.messages[3];
        assert_eq!(answer.content, Some(json!("Checking.")));
        assert_eq!(answer.tool_calls.len(), 2);
        assert_eq!(answer.tool_calls[1]["id"], "call_b");
        assert_eq!(answer.tool_calls[1]["function"]["arguments"], "{\"q\":2}");
        assert_eq!(chat.messages[4].tool_call_id.as_deref(), Some("call_a"));
        assert_eq!(chat.messages[5].content, Some(json!("two")));
        assert!(chat.stream);
        assert_eq!(chat.tools[0]["function"]["name"], "lookup");
        assert_eq!(
            chat.tool_choice,
            Some(json!({"type": "function", "function": {"name": "lookup"}}))
        );
        let (specs, required) = resolve_tools(&chat);
        assert_eq!(specs.len(), 1);
        assert!(required);
    }

    #[test]
    fn a_function_call_without_preceding_text_opens_its_own_assistant_message() {
        let chat = chat_request(&turn(vec![
            user("run"),
            Item::FunctionCall {
                call_id: "call_a".into(),
                name: "lookup".into(),
                arguments: "{}".into(),
            },
        ]))
        .unwrap();
        assert_eq!(chat.messages.len(), 2);
        assert_eq!(chat.messages[1].role, "assistant");
        assert_eq!(chat.messages[1].content, None);
        assert_eq!(chat.messages[1].tool_calls.len(), 1);
    }

    #[test]
    fn file_parts_are_rejected_instead_of_dropped() {
        let request = turn(vec![Item::Message {
            role: Role::User,
            content: vec![
                text("read this"),
                Part::File {
                    filename: Some("a.pdf".into()),
                    file_data: Some("data:application/pdf;base64,AAAA".into()),
                    file_url: None,
                },
            ],
        }]);
        let error = chat_request(&request).unwrap_err();
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(error.code, "unsupported_parameter");
    }

    #[test]
    fn a_stored_answer_replays_into_the_checkpoint_it_produced() {
        let email = "account@example.invalid";
        let mut first = turn(vec![user("run it")]);
        first.tools = vec![lookup_tool()];
        let accepted = vec![ToolCall {
            name: "lookup".into(),
            arguments: "{\"q\":1}".into(),
        }];
        let saved = TurnHistory::new(&chat_request(&first).unwrap(), &first.model).completed(
            "Checking.",
            &accepted,
            email,
        );

        // What the HTTP layer stores for that answer, with server call ids.
        let plan =
            TurnPlan::from_request(serde_json::from_value(json!({"input": "run it"})).unwrap())
                .unwrap();
        let mut encoder = EventEncoder::new("resp_1", "DEFAULT", &plan, true);
        let _ = encoder.text_delta("Checking.");
        let _ = encoder.finish(
            &[FunctionCall {
                name: "lookup".into(),
                arguments: "{\"q\":1}".into(),
            }],
            Usage::default(),
        );
        let mut transcript = first.transcript.clone();
        transcript.extend(encoder.transcript_items());
        let call_id = match transcript.last() {
            Some(Item::FunctionCall { call_id, .. }) => call_id.clone(),
            other => panic!("expected a function call, got {other:?}"),
        };
        transcript.push(Item::FunctionCallOutput {
            call_id,
            output: "result".into(),
        });
        let mut next = first.clone();
        next.transcript = transcript;

        let remote = ThreadState {
            session_id: "remote".into(),
            parent_message_id: Some(9),
            exchanges: 0,
        };
        let resumed = TurnHistory::new(&chat_request(&next).unwrap(), &next.model);
        assert!(resumed.can_continue(Some(&remote), Some(&saved), email));
        let regenerated = TurnHistory::new(&chat_request(&first).unwrap(), &first.model);
        assert!(!regenerated.can_continue(Some(&remote), Some(&saved), email));
    }

    #[test]
    fn reasoning_effort_toggles_thinking_and_verbosity_becomes_a_directive() {
        assert!(!thinking_for(Some(ReasoningEffort::None), true));
        assert!(!thinking_for(Some(ReasoningEffort::Minimal), true));
        assert!(thinking_for(Some(ReasoningEffort::Low), false));
        assert!(thinking_for(Some(ReasoningEffort::XHigh), false));
        assert!(thinking_for(None, true));
        assert!(!thinking_for(None, false));
        assert!(verbosity_directive(Verbosity::Low).is_some());
        assert!(verbosity_directive(Verbosity::Medium).is_none());
        assert!(verbosity_directive(Verbosity::High).is_some());
    }

    #[tokio::test]
    async fn drain_streams_reasoning_and_prose_and_holds_back_tool_blocks() {
        let raw = concat!(
            "data: {\"response_message_id\":7}\n\n",
            "data: {\"v\":{\"response\":{\"fragments\":[{\"type\":\"THINK\",\"content\":\"plan\"}]}}}\n\n",
            "data: {\"p\":\"response/fragments\",\"o\":\"APPEND\",\"v\":[{\"type\":\"RESPONSE\",\"content\":\"Hi \"}]}\n\n",
            "data: {\"v\":\"there<invoke name=\\\"bash\\\">{\\\"command\\\":\\\"ls\\\"}</invoke>\"}\n\n",
            "data: {\"p\":\"response/status\",\"v\":\"FINISHED\"}\n\n",
        );
        let (tx, mut rx) = mpsc::channel(16);
        let mut announced = false;
        let drained = drain(response(raw), &tx, &mut announced, None).await;
        drop(tx);
        let mut seen = Vec::new();
        while let Some(event) = rx.recv().await {
            seen.push(event);
        }
        assert_eq!(
            seen,
            [
                TurnEvent::Started,
                TurnEvent::Reasoning("plan".into()),
                TurnEvent::Text("Hi ".into()),
                TurnEvent::Text("there".into()),
            ]
        );
        assert!(announced && drained.complete && drained.emitted && !drained.cancelled);
        assert_eq!(drained.hint, None);
        assert_eq!(drained.parent, Some(7));
        assert_eq!(drained.prose, "Hi there");
        assert_eq!(drained.reasoning, "plan");
        assert!(drained.text.ends_with("</invoke>"));
        assert_eq!(drained.filter.calls.len(), 1);
    }

    #[tokio::test]
    async fn a_hint_before_any_fragment_does_not_start_the_response() {
        let raw = "event: hint\ndata: {\"type\":\"error\",\"content\":\"rate limit exceeded\"}\n\n";
        let (tx, mut rx) = mpsc::channel(4);
        let mut announced = false;
        let drained = drain(response(raw), &tx, &mut announced, None).await;
        assert_eq!(drained.hint.as_deref(), Some("rate limit exceeded"));
        assert!(!announced && !drained.emitted && !drained.complete);
        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn non_sse_bodies_become_fixed_hints() {
        let (tx, _rx) = mpsc::channel(4);
        let mut announced = false;
        let stale = drain(
            response("{\"code\":26,\"biz_msg\":\"invalid message id\"}"),
            &tx,
            &mut announced,
            None,
        )
        .await;
        assert_eq!(stale.hint.as_deref(), Some("invalid message id"));
        let private = drain(
            response("synthetic private model output"),
            &tx,
            &mut announced,
            None,
        )
        .await;
        assert_eq!(private.hint.as_deref(), Some("non-SSE upstream response"));
        assert!(!announced);
    }

    #[test]
    fn non_sse_hints_classify_as_themselves() {
        for hint in [
            "invalid message id",
            "context_length_exceeded",
            "rate limit exceeded",
            "login failed",
            "invalid token",
            "non-SSE upstream response",
        ] {
            assert_eq!(classify_non_sse(hint), hint);
        }
    }

    /// Once the budget is reached, later text is not forwarded, but the stream
    /// is still drained so the cursor and account slot stay correct.
    #[tokio::test]
    async fn drain_stops_forwarding_at_the_output_budget() {
        let (tx, mut rx) = mpsc::channel(16);
        let mut announced = false;
        let raw = concat!(
            "data: {\"v\":\"alpha beta gamma delta epsilon zeta eta theta\"}\n\n",
            "data: {\"v\":\" iota kappa lambda mu\"}\n\n",
            "data: {\"response_message_id\":7}\n\n",
            "data: {\"p\":\"response/status\",\"v\":\"FINISHED\"}\n\n"
        );
        let drained = drain(response(raw), &tx, &mut announced, Some(3)).await;
        assert!(drained.truncated, "the budget must be reported");
        assert!(drained.complete, "the upstream is still drained");
        assert_eq!(drained.parent, Some(7), "the cursor survives truncation");
        let mut forwarded = String::new();
        while let Ok(TurnEvent::Text(text)) = rx.try_recv() {
            forwarded.push_str(&text);
        }
        // The stream filter may hold a trailing partial tag back, so assert on
        // the drain's decisions instead of on exact chunk boundaries.
        assert!(
            estimate_tokens(&forwarded) <= 3,
            "forwarded text must stay within the budget: {forwarded:?}"
        );
        assert!(
            !forwarded.contains("iota"),
            "text past the budget must not reach the client: {forwarded:?}"
        );
    }

    #[test]
    fn budget_arithmetic_counts_reasoning_and_trims_by_tokens() {
        assert!(over_budget("some words here", "", Some(1)));
        assert!(over_budget("", "reasoning also counts", Some(1)));
        assert!(!over_budget("short", "", Some(64)));
        assert!(!over_budget("anything at all", "and more", None));
        assert_eq!(trim_to_tokens("abcdef", 0), "");
        assert_eq!(trim_to_tokens("short", 64), "short");
        let trimmed = trim_to_tokens("alpha beta gamma delta", 2);
        assert!(!trimmed.is_empty() && trimmed.len() < "alpha beta gamma delta".len());
        assert!(estimate_tokens(&trimmed) <= 2);
    }

    #[tokio::test]
    async fn drain_stops_when_the_consumer_is_gone() {
        let (tx, rx) = mpsc::channel(1);
        drop(rx);
        let mut announced = false;
        let drained = drain(
            response("data: {\"v\":\"text\"}\n\n"),
            &tx,
            &mut announced,
            None,
        )
        .await;
        assert!(drained.cancelled);
    }

    #[tokio::test]
    async fn an_invalid_image_fails_before_any_upstream_work() {
        let state = test_state(deepseek_runtime()).await;
        let request = turn(vec![Item::Message {
            role: Role::User,
            content: vec![
                text("look"),
                Part::Image {
                    url: "data:image/png;base64,***".into(),
                    detail: "auto".into(),
                },
            ],
        }]);
        let (tx, mut rx) = mpsc::channel(8);
        let Err(TurnError::Proxy(error)) = run(state, request, tx).await else {
            panic!("expected a client error");
        };
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(error.code, "invalid_image");
        assert!(rx.try_recv().is_err(), "nothing was started");
    }

    #[tokio::test]
    async fn a_continuation_without_new_input_is_rejected_before_upstream_work() {
        let state = test_state(deepseek_runtime()).await;
        let first = turn(vec![user("root")]);
        let saved = TurnHistory::new(&chat_request(&first).unwrap(), &first.model).completed(
            "answer",
            &[],
            "test@example.invalid",
        );
        state
            .store
            .save_thread_checkpoint(
                &first.thread_id,
                &ThreadResume {
                    session_id: "remote".into(),
                    parent_message_id: Some("2".into()),
                },
                &saved,
            )
            .unwrap();
        let (tx, _rx) = mpsc::channel(8);
        let ended_on_answer = turn(vec![user("root"), assistant("answer")]);
        let Err(TurnError::Proxy(error)) = run(state.clone(), ended_on_answer, tx).await else {
            panic!("expected a client error");
        };
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(error.param.as_deref(), Some("input"));
        // Nothing was attempted, so the trusted head survives.
        assert!(state
            .store
            .get_thread_history(&first.thread_id)
            .unwrap()
            .is_some());
    }
}
