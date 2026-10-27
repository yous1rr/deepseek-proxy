//! ChatGPT Web Responses channel adapter.
//!
//! ChatGPT's private web protocol exposes no stable resumable response cursor,
//! so continuation is handled by replaying the locally stored transcript.

use super::channel::{ResponsesChannel, TurnError, TurnEvent, TurnOutcome, TurnRequest};
use crate::{dialect, tool_failure, AppState, ProxyError};
use axum::http::StatusCode;
use chatgpt_client::{ChatGptClient, ClientConfig, ConversationRequest};
use chatgpt_protocol::{consume_sse_line, ConversationEvent, PatchState};
use futures_util::{future::BoxFuture, FutureExt, StreamExt};
use responses_core::capability::{Capability, CapabilitySet, CapabilityStatus, Provider};
use responses_core::events::{FunctionCall, HostedSource, HostedToolKind, HostedToolResult, Usage};
use responses_core::plan::{FunctionTool, ReasoningEffort, ToolChoice, Verbosity};
use responses_core::transcript::{Item, Part, Role};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tool_engine::{Recovery, ToolPolicy};

pub const CHANNEL_ID: &str = "chatgpt_web";
const MAX_MESSAGES: usize = 256;
const MAX_IMAGES: usize = 16;
const MAX_TEXT_BYTES: usize = 1024 * 1024;
const MAX_SSE_LINE_BYTES: usize = chatgpt_protocol::MAX_PAYLOAD_BYTES;
const MAX_EVENTS: usize = 100_000;

/// Authenticated ChatGPT Web clients, selected round-robin per turn.
pub struct ChatGptWeb {
    clients: Vec<ChatGptClient>,
    next_client: AtomicUsize,
}

impl ChatGptWeb {
    /// Create the adapter from configured clients.
    pub fn new(clients: Vec<ChatGptClient>) -> Result<Self, &'static str> {
        if clients.is_empty() {
            return Err("at least one ChatGPT Web client is required");
        }
        Ok(Self {
            clients,
            next_client: AtomicUsize::new(0),
        })
    }

    /// Build clients from `CHATGPT_ACCESS_TOKENS` or `CHATGPT_ACCESS_TOKEN`.
    /// Optional `CHATGPT_BASE_URL` and `CHATGPT_PROXY_URL` configure transport.
    pub fn from_env() -> Result<Self, &'static str> {
        let tokens = std::env::var("CHATGPT_ACCESS_TOKENS")
            .or_else(|_| std::env::var("CHATGPT_ACCESS_TOKEN"))
            .map_err(|_| "CHATGPT_ACCESS_TOKEN(S) is not configured")?;
        let tokens: Vec<_> = tokens
            .split(',')
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .collect();
        if tokens.is_empty() {
            return Err("CHATGPT_ACCESS_TOKEN(S) is not configured");
        }
        let base_url =
            std::env::var("CHATGPT_BASE_URL").unwrap_or_else(|_| "https://chatgpt.com".into());
        let proxy_url = std::env::var("CHATGPT_PROXY_URL")
            .ok()
            .filter(|v| !v.trim().is_empty());
        let clients = tokens
            .into_iter()
            .map(|token| {
                ChatGptClient::new(ClientConfig {
                    base_url: base_url.clone(),
                    access_token: token.to_owned(),
                    proxy_url: proxy_url.clone(),
                    timeout: Duration::from_secs(300),
                })
                .map_err(|_| "invalid ChatGPT Web client configuration")
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(clients)
    }

    fn client(&self) -> ChatGptClient {
        let index = self.next_client.fetch_add(1, Ordering::Relaxed) % self.clients.len();
        self.clients[index].clone()
    }
}

/// Explicit adapter contract for the ChatGPT Web capabilities implemented here.
pub fn ceiling() -> CapabilitySet {
    use Capability as C;
    let supported: [(Capability, Provider, &str); 13] = [
        // The web upstream has no function-call API: tools are taught in the
        // prompt and parsed back out, exactly like the DeepSeek channel, so the
        // provider is this proxy rather than the upstream.
        (C::FunctionTools, Provider::Proxy, "prompted_tool_protocol"),
        (C::ToolChoice, Provider::Proxy, "prompted_directive"),
        (C::MaxOutputTokens, Provider::Proxy, "local_token_budget"),
        (C::Text, Provider::Upstream, "chatgpt_web_conversation"),
        (C::Streaming, Provider::Proxy, "chatgpt_web_sse"),
        (
            C::NonStreaming,
            Provider::Proxy,
            "collected_chatgpt_web_sse",
        ),
        (
            C::ImageInput,
            Provider::Upstream,
            "uploaded_multimodal_assets",
        ),
        // Only the request-side effort reaches the upstream; the web stream
        // exposes no reasoning text, so the mode says exactly that.
        (C::Reasoning, Provider::Upstream, "thinking_effort_only"),
        (C::Verbosity, Provider::Proxy, "prompt_instruction"),
        (C::StoredResponses, Provider::Proxy, "local_transcript"),
        (
            C::PreviousResponse,
            Provider::Proxy,
            "local_transcript_replay",
        ),
        (
            C::WebSearch,
            Provider::Upstream,
            "search_system_hint_and_source_items",
        ),
        (
            C::ImageGeneration,
            Provider::Upstream,
            "image_gen_asset_items",
        ),
    ];
    let unsupported: [(Capability, &str); 17] = [
        (C::FileInput, "no_chatgpt_web_file_input"),
        // Freeform grammars are not teached by the prompted dialect; only
        // JSON-Schema function tools are.
        (C::CustomTools, "no_custom_tool_grammar"),
        (C::FileSearch, "no_file_search_implementation"),
        (C::CodeInterpreter, "no_code_interpreter_event_mapping"),
        (C::Mcp, "no_mcp_client"),
        (C::ComputerUse, "no_computer_use_event_mapping"),
        (C::Shell, "no_shell_tool_execution"),
        (C::StructuredOutput, "no_upstream_schema_enforcement"),
        (C::SamplingControls, "chatgpt_web_has_no_sampling_controls"),
        (C::Logprobs, "no_logprobs_mapping"),
        (C::Conversations, "no_remote_conversation_resource_api"),
        (C::Background, "no_background_response_api"),
        (C::StreamResume, "no_stream_resume_cursor"),
        (C::AutoTruncation, "no_context_truncation"),
        (C::Compaction, "no_context_compaction"),
        (C::PromptTemplates, "no_prompt_template_api"),
        (C::ExactUsage, "upstream_usage_unavailable_tokens_estimated"),
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
/// Slugs this channel routes. Kept in step with [`crate::config::CHATGPT_MODELS`]
/// by a test below, so the advertised catalogue and the router cannot drift.
pub(crate) fn supports_model(model: &str) -> bool {
    let model = model.trim().to_ascii_lowercase();
    crate::config::CHATGPT_MODELS.contains(&model.as_str())
}

impl ResponsesChannel for ChatGptWeb {
    fn id(&self) -> &'static str {
        CHANNEL_ID
    }
    fn capabilities(&self) -> CapabilitySet {
        ceiling()
    }
    fn supports_model(&self, model: &str) -> bool {
        supports_model(model)
    }
    fn run_turn(
        &self,
        state: AppState,
        turn: TurnRequest,
        events: mpsc::Sender<TurnEvent>,
    ) -> BoxFuture<'static, Result<TurnOutcome, TurnError>> {
        let client = self.client();
        async move { run_turn(client, state, turn, events).await }.boxed()
    }
}

/// Convert the full neutral transcript to ChatGPT's author/content/parts
/// structure. Each input image is uploaded in place, preserving part order.
/// Convert the neutral transcript into ChatGPT's author/content/parts shape.
///
/// When the turn offers tools, the prompted protocol is merged into the system
/// message and prior calls replay as `<invoke>` blocks, so the model sees the
/// same wire format it is being asked to emit. Each input image is uploaded in
/// place, preserving part order.
async fn web_messages(
    client: &ChatGptClient,
    turn: &TurnRequest,
    policy: &ToolPolicy,
    max_image_bytes: usize,
    events: &mpsc::Sender<TurnEvent>,
    started: Instant,
    timeout: Duration,
) -> Result<Vec<Value>, TurnError> {
    if turn.transcript.len() > MAX_MESSAGES {
        return Err(bad_request(
            "input",
            "Conversation exceeds the supported message limit",
        ));
    }
    let mut messages = Vec::with_capacity(turn.transcript.len() + 2);
    let mut text_bytes = 0usize;
    let mut image_count = 0usize;
    // The protocol and the caller's instructions both belong in the system
    // slot; the protocol is appended so it stays closest to the transcript.
    let instructions = turn.instructions.as_deref().filter(|v| !v.is_empty());
    let system = if policy.has_tools() {
        let raw: Vec<Value> = turn.tools.iter().map(function_tool_json).collect();
        dialect::inject_tool_descriptions(instructions.unwrap_or(""), &raw)
    } else {
        instructions.unwrap_or("").to_string()
    };
    if !system.is_empty() {
        text_bytes = system.len();
        if text_bytes > MAX_TEXT_BYTES {
            return Err(bad_request(
                "instructions",
                "Instructions exceed the supported size limit",
            ));
        }
        messages.push(web_message("system", vec![json!(system)]));
    }
    for item in &turn.transcript {
        let value = match item {
            Item::Message { role, content } => {
                let mut parts = Vec::with_capacity(content.len());
                for part in content {
                    match part {
                        Part::Text { text } => {
                            text_bytes = text_bytes.saturating_add(text.len());
                            if text_bytes > MAX_TEXT_BYTES {
                                return Err(bad_request(
                                    "input",
                                    "Conversation text exceeds the supported size limit",
                                ));
                            }
                            parts.push(json!(text));
                        }
                        Part::Image { url, .. } => {
                            image_count += 1;
                            if image_count > MAX_IMAGES {
                                return Err(bad_request(
                                    "input",
                                    "Conversation exceeds the supported image limit",
                                ));
                            }
                            let image = tokio::select! {
                                biased;
                                _ = events.closed() => return Err(TurnError::Cancelled),
                                result = tokio::time::timeout(
                                    timeout.saturating_sub(started.elapsed()),
                                    load_upload(client, url, image_count, max_image_bytes),
                                ) => match result {
                                    Ok(result) => result?,
                                    Err(_) => return Err(timeout_error()),
                                },
                            };
                            parts.push(image);
                        }
                        Part::File { .. } => {
                            return Err(ProxyError::new(
                                "File input is not supported by the ChatGPT Web channel",
                                StatusCode::BAD_REQUEST,
                                "unsupported_parameter",
                            )
                            .with_param("input")
                            .into())
                        }
                    }
                }
                web_message(role_name(*role), parts)
            }
            Item::FunctionCall {
                call_id,
                name,
                arguments,
            } => {
                // Replay in the taught wire format so the transcript looks like
                // something the model itself produced. Consecutive calls join
                // one assistant message, matching how the dialect batches them.
                let block = format!("<invoke name=\"{name}\">{arguments}</invoke>");
                let _ = call_id;
                match messages.last_mut() {
                    Some(last) if last["author"]["role"] == "assistant" => {
                        if let Some(parts) = last["content"]["parts"].as_array_mut() {
                            parts.push(json!(format!("\n\n{block}")));
                        }
                        continue;
                    }
                    _ => web_message("assistant", vec![json!(block)]),
                }
            }
            Item::FunctionCallOutput { call_id, output } => web_message(
                "user",
                vec![json!(format!("Tool result (id={call_id}):\n{output}"))],
            ),
        };
        messages.push(value);
        if messages.len() > MAX_MESSAGES {
            return Err(bad_request(
                "input",
                "Conversation exceeds the supported message limit",
            ));
        }
        if started.elapsed() >= timeout {
            return Err(timeout_error());
        }
    }
    if policy.has_tools() {
        // Re-anchor the wire format immediately before the model answers, and
        // carry the `tool_choice` / `parallel_tool_calls` contract.
        let mut suffix = format!("\n\n{}", dialect::periodic_reminder());
        suffix.push_str(&policy.directive);
        match messages.last_mut() {
            Some(last) if last["author"]["role"] == "user" => {
                if let Some(parts) = last["content"]["parts"].as_array_mut() {
                    parts.push(json!(suffix));
                }
            }
            _ => messages.push(web_message("user", vec![json!(suffix.trim_start())])),
        }
    }
    if let Some(verbosity) = turn.verbosity {
        let instruction = verbosity_instruction(verbosity);
        if !instruction.is_empty() {
            messages.insert(0, web_message("system", vec![json!(instruction)]));
        }
    }
    Ok(messages)
}

fn web_message(role: &str, parts: Vec<Value>) -> Value {
    json!({"author":{"role":role}, "content":{"content_type":"multimodal_text", "parts":parts}})
}
fn role_name(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::Developer => "developer",
        Role::User => "user",
        Role::Assistant => "assistant",
    }
}
fn verbosity_instruction(value: Verbosity) -> &'static str {
    match value {
        Verbosity::Low => "Answer concisely, keeping the response as short as the task allows.",
        Verbosity::Medium => "",
        Verbosity::High => "Answer thoroughly, including relevant explanations and details.",
    }
}

async fn load_upload(
    client: &ChatGptClient,
    url: &str,
    index: usize,
    max_bytes: usize,
) -> Result<Value, TurnError> {
    let msg: crate::Msg = serde_json::from_value(
        json!({"role":"user", "content":[{"type":"image_url", "image_url":{"url":url}}]}),
    )
    .map_err(|_| bad_request("input", "Invalid image input"))?;
    let loaded = crate::images::load(&[msg], max_bytes)
        .await
        .map_err(|_| bad_request("input", "Invalid or unavailable image"))?;
    let bytes = loaded
        .data
        .ok_or_else(|| bad_request("input", "Invalid or unavailable image"))?;
    let mime =
        image_mime(url, &bytes).ok_or_else(|| bad_request("input", "Unsupported image format"))?;
    let ext = match mime {
        "image/jpeg" => "jpg",
        "image/gif" => "gif",
        _ => "png",
    };
    client
        .upload_image(bytes, mime, &format!("image_{index}.{ext}"))
        .await
        .map_err(client_error)
}
fn image_mime(url: &str, bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        Some("image/jpeg")
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some("image/gif")
    } else {
        match url.strip_prefix("data:")?.split(';').next()? {
            "image/png" => Some("image/png"),
            "image/jpeg" => Some("image/jpeg"),
            "image/gif" => Some("image/gif"),
            _ => None,
        }
    }
}
fn thinking_effort(value: Option<ReasoningEffort>) -> Option<String> {
    match value {
        Some(ReasoningEffort::None | ReasoningEffort::Minimal) | None => None,
        Some(ReasoningEffort::Low) => Some("low".into()),
        Some(ReasoningEffort::Medium) => Some("medium".into()),
        Some(ReasoningEffort::High) => Some("high".into()),
        Some(ReasoningEffort::XHigh) => Some("extended".into()),
    }
}
fn system_hints(model: &str, hosted_tools: &[String]) -> Vec<String> {
    let mut hints = Vec::new();
    if hosted_tools.iter().any(|tool| {
        matches!(
            tool.as_str(),
            "web_search"
                | "web_search_preview"
                | "web_search_2025_08_26"
                | "web_search_preview_2025_03_11"
        )
    }) {
        hints.push("search".to_string());
    }
    if hosted_tools.iter().any(|tool| tool == "image_generation")
        || matches!(
            model.to_ascii_lowercase().as_str(),
            "gpt-image-2" | "codex-gpt-image-2"
        )
    {
        hints.push("picture_v2".to_string());
    }
    hints
}
/// Collect the answer of one upstream conversation call.
///
/// `sink` receives prose the tool filter has released; a correction turn passes
/// `None` because its text is only scanned, never shown to the client.
#[allow(clippy::too_many_arguments)]
async fn read_answer(
    client: &ChatGptClient,
    request: &ConversationRequest,
    events: &mpsc::Sender<TurnEvent>,
    started: Instant,
    timeout: Duration,
    budget: Option<u64>,
    sink: Option<&mut dialect::StreamToolFilter>,
) -> Result<Answer, TurnError> {
    let response = tokio::select! {
        biased;
        _ = events.closed() => return Err(TurnError::Cancelled),
        result = tokio::time::timeout(timeout.saturating_sub(started.elapsed()), client.conversation(request)) => match result {
            Ok(Ok(response)) => response,
            Ok(Err(error)) => return Err(client_error(error)),
            Err(_) => return Err(timeout_error()),
        },
    };
    if sink.is_some() && events.send(TurnEvent::Started).await.is_err() {
        return Err(TurnError::Cancelled);
    }

    let mut protocol = PatchState::new();
    let mut buffered = Vec::new();
    let mut stream = response.bytes_stream();
    let mut answer = Answer::default();
    let mut reasoning = String::new();
    let mut event_count = 0usize;
    let mut done = false;
    let mut sink = sink;
    loop {
        let next = tokio::select! {
            biased;
            _ = events.closed() => return Err(TurnError::Cancelled),
            value = tokio::time::timeout(timeout.saturating_sub(started.elapsed()), stream.next()) => match value {
                Ok(value) => value, Err(_) => return Err(timeout_error()),
            },
        };
        let Some(chunk) = next else { break };
        buffered.extend_from_slice(
            &chunk.map_err(|_| upstream_error("ChatGPT Web stream could not be read"))?,
        );
        let mut offset = 0usize;
        while offset < buffered.len() {
            let Some(relative_end) = buffered[offset..].iter().position(|byte| *byte == b'\n')
            else {
                if buffered.len().saturating_sub(offset) > MAX_SSE_LINE_BYTES {
                    return Err(upstream_error("ChatGPT Web sent an oversized event"));
                }
                break;
            };
            let end = offset + relative_end;
            let line_end = if end > offset && buffered[end - 1] == b'\r' {
                end - 1
            } else {
                end
            };
            if line_end - offset > MAX_SSE_LINE_BYTES {
                return Err(upstream_error("ChatGPT Web sent an oversized event"));
            }
            let line = std::str::from_utf8(&buffered[offset..line_end])
                .map_err(|_| upstream_error("ChatGPT Web sent an invalid event"))?;
            if let Some(event) = consume_sse_line(&mut protocol, line)
                .map_err(|_| upstream_error("ChatGPT Web sent an invalid event"))?
            {
                event_count += 1;
                if event_count > MAX_EVENTS {
                    return Err(upstream_error(
                        "ChatGPT Web response exceeded the event limit",
                    ));
                }
                match event {
                    ConversationEvent::TextDelta(text) => {
                        answer.text.push_str(&text);
                        forward_text(&text, sink.as_deref_mut(), events, &mut answer, budget)
                            .await?;
                    }
                    ConversationEvent::ReasoningDelta(text) => {
                        reasoning.push_str(&text);
                        if sink.is_some() {
                            send(events, TurnEvent::Reasoning(text)).await?;
                        }
                    }
                    ConversationEvent::ImageTool {
                        conversation_id,
                        file_ids,
                        sediment_ids,
                    } => {
                        if sink.is_none() || conversation_id.is_empty() {
                            continue;
                        }
                        let images = tokio::select! {
                            biased;
                            _ = events.closed() => return Err(TurnError::Cancelled),
                            result = tokio::time::timeout(timeout.saturating_sub(started.elapsed()), client.image_results(&conversation_id, &file_ids, &sediment_ids)) => match result {
                                Ok(Ok(value)) => value,
                                Ok(Err(_)) => return Err(upstream_error("Generated image could not be retrieved")),
                                Err(_) => return Err(timeout_error()),
                            }
                        };
                        for (index, image) in images.into_iter().enumerate() {
                            send(
                                events,
                                TurnEvent::Hosted(HostedToolResult {
                                    kind: HostedToolKind::ImageGeneration,
                                    id: format!("ig_{event_count}_{index}"),
                                    query: None,
                                    sources: Vec::new(),
                                    result: Some(image),
                                    output: None,
                                    files: Vec::new(),
                                }),
                            )
                            .await?;
                        }
                    }
                    ConversationEvent::ModerationBlocked => {
                        return Err(upstream_error("ChatGPT Web blocked this response"))
                    }
                    ConversationEvent::Done => {
                        done = true;
                        offset = end + 1;
                        break;
                    }
                    ConversationEvent::SearchTool {
                        conversation_id,
                        query,
                        sources,
                    } => {
                        if sink.is_none() {
                            continue;
                        }
                        let sources = sources
                            .into_iter()
                            .map(|source| HostedSource {
                                title: source.title,
                                url: source.url,
                                snippet: source.snippet,
                            })
                            .collect();
                        send(
                            events,
                            TurnEvent::Hosted(HostedToolResult {
                                kind: HostedToolKind::WebSearch,
                                id: format!("ws_{event_count}"),
                                query,
                                sources,
                                result: None,
                                output: None,
                                files: Vec::new(),
                            }),
                        )
                        .await?;
                        let _ = conversation_id;
                    }
                    ConversationEvent::Started { .. } | ConversationEvent::RawIgnored => {}
                }
            }
            offset = end + 1;
        }
        if offset > 0 {
            buffered.drain(..offset);
        }
        if done {
            break;
        }
    }
    if !buffered.is_empty() && !done {
        let line = std::str::from_utf8(&buffered)
            .map_err(|_| upstream_error("ChatGPT Web sent an invalid event"))?;
        if let Some(event) = consume_sse_line(&mut protocol, line)
            .map_err(|_| upstream_error("ChatGPT Web sent an invalid event"))?
        {
            match event {
                ConversationEvent::TextDelta(text) => {
                    answer.text.push_str(&text);
                    forward_text(&text, sink.as_deref_mut(), events, &mut answer, budget).await?;
                }
                ConversationEvent::ReasoningDelta(text) => {
                    reasoning.push_str(&text);
                    if sink.is_some() {
                        send(events, TurnEvent::Reasoning(text)).await?;
                    }
                }
                ConversationEvent::Done => done = true,
                _ => {}
            }
        }
    }
    if !done && answer.text.is_empty() && reasoning.is_empty() {
        return Err(upstream_error("ChatGPT Web ended without a response"));
    }
    answer.reasoning = reasoning;
    Ok(answer)
}

/// Push one text delta through the tool filter (or straight out when no tools
/// are offered), honouring the local output budget.
async fn forward_text(
    text: &str,
    sink: Option<&mut dialect::StreamToolFilter>,
    events: &mpsc::Sender<TurnEvent>,
    answer: &mut Answer,
    budget: Option<u64>,
) -> Result<(), TurnError> {
    let Some(filter) = sink else { return Ok(()) };
    if answer.truncated {
        return Ok(());
    }
    let mut safe = filter.feed(text);
    if let Some(limit) = budget {
        let room = limit
            .saturating_sub(estimate_tokens(&answer.prose))
            .saturating_sub(estimate_tokens(&answer.reasoning));
        if room == 0 {
            answer.truncated = true;
            return Ok(());
        }
        let trimmed = trim_to_tokens(&safe, room);
        if trimmed.len() < safe.len() {
            answer.truncated = true;
            safe = trimmed;
        }
    }
    if safe.is_empty() {
        return Ok(());
    }
    answer.prose.push_str(&safe);
    send(events, TurnEvent::Text(safe)).await
}

/// Chat's answer, split into the scan input, the forwarded prose and the budget
/// state.
#[derive(Debug, Default)]
struct Answer {
    /// Visible text including tool syntax; this is what gets scanned.
    text: String,
    /// Prose already forwarded to the client.
    prose: String,
    reasoning: String,
    /// The local budget cut this answer short, so no call may execute.
    truncated: bool,
}

/// Resolve the turn's tool contract in the OpenAI shape the dialect expects.
fn tool_policy(turn: &TurnRequest) -> ToolPolicy {
    let tools: Vec<Value> = turn.tools.iter().map(function_tool_json).collect();
    let choice = match &turn.tool_choice {
        ToolChoice::Auto => None,
        ToolChoice::None => Some(json!("none")),
        ToolChoice::Required => Some(json!("required")),
        ToolChoice::Function(name) => Some(json!({"type": "function", "function": {"name": name}})),
    };
    ToolPolicy::resolve(&tools, choice.as_ref(), turn.parallel_tool_calls)
}

fn function_tool_json(tool: &FunctionTool) -> Value {
    let mut function = json!({"name": tool.name, "parameters": tool.parameters});
    if let Some(description) = &tool.description {
        function["description"] = json!(description);
    }
    json!({"type": "function", "function": function})
}

/// Cheap token estimate for the local output budget.
fn estimate_tokens(text: &str) -> u64 {
    (text.chars().count() as u64).div_ceil(4)
}

/// Longest prefix of `text` that fits in `tokens`.
///
/// The estimate counts characters, so the trim does too; that keeps the two in
/// agreement for non-ASCII text instead of cutting a multi-byte character.
fn trim_to_tokens(text: &str, tokens: u64) -> String {
    let keep = (tokens.saturating_mul(4)) as usize;
    if text.chars().count() <= keep {
        return text.to_string();
    }
    text.chars().take(keep).collect()
}

/// Run one turn on the ChatGPT Web upstream.
///
/// The web protocol has no function-call API and no resumable cursor, so tools
/// are prompted (shared `tool-engine` dialect) and a correction turn is just the
/// same transcript plus the failed answer and the correction prompt.
async fn run_turn(
    client: ChatGptClient,
    state: AppState,
    turn: TurnRequest,
    events: mpsc::Sender<TurnEvent>,
) -> Result<TurnOutcome, TurnError> {
    let started = Instant::now();
    let timeout = Duration::from_secs_f64(state.settings.request_timeout_secs.max(0.001));
    let policy = tool_policy(&turn);
    let messages = tokio::select! {
        biased;
        _ = events.closed() => return Err(TurnError::Cancelled),
        result = web_messages(&client, &turn, &policy, state.settings.image_max_bytes, &events, started, timeout) => result?,
    };
    let hints = system_hints(&turn.model, &turn.hosted_tools);
    let mut request = ConversationRequest {
        messages: messages.clone(),
        model: turn.model.clone(),
        thinking_effort: thinking_effort(turn.reasoning_effort),
        system_hints: hints,
    };
    // `max_output_tokens` arrives only when the channel declares a budget.
    let budget = turn.max_output_tokens.map(|limit| limit.max(0) as u64);
    let mut filter = dialect::StreamToolFilter::default();
    let answer = read_answer(
        &client,
        &request,
        &events,
        started,
        timeout,
        budget,
        Some(&mut filter),
    )
    .await?;
    let mut prose = answer.prose.clone();
    let mut truncated = answer.truncated;

    // Flush the prose the filter held back at the very end of the stream.
    let tail = filter.finish();
    if !tail.is_empty() {
        prose.push_str(&tail);
        send(&events, TurnEvent::Text(tail)).await?;
    }

    let mut recovery = Recovery::new(policy.specs.clone(), policy.required, policy.parallel);
    // A truncated answer may have lost the tail of a call, so none executes.
    if !truncated {
        recovery.accept_text(&answer.text).map_err(tool_failure)?;
    }
    let mut transcript = messages.clone();
    for attempt in 1..=state.settings.tool_repair_max_retries {
        if !recovery.needs_repair() {
            break;
        }
        let prompt = recovery.begin_repair();
        tracing::warn!(
            thread = %turn.thread_id,
            attempt,
            expected = recovery.expected_count(),
            "tool repair"
        );
        // The web backend accepts a full message array, so the correction is a
        // new pair on the same transcript: the failed answer, then the prompt.
        transcript.push(web_message("assistant", vec![json!(answer.text)]));
        transcript.push(web_message("user", vec![json!(prompt)]));
        request.messages = transcript.clone();
        request.system_hints = Vec::new();
        request.thinking_effort = None;
        let correction = tokio::select! {
            biased;
            _ = events.closed() => return Err(TurnError::Cancelled),
            result = read_answer(&client, &request, &events, started, timeout, budget, None) => result,
        };
        match correction {
            Ok(answer) => {
                if let Err(error) = recovery.accept_text(&answer.text) {
                    return Err(tool_failure(error).into());
                }
                recovery.end_repair();
            }
            // A failed correction ends the loop; the outstanding failure is
            // reported as a repair failure rather than masking the answer.
            Err(_) => {
                recovery.end_repair();
                break;
            }
        }
    }
    let accepted = match recovery.finish() {
        Ok(calls) => calls,
        Err(error) => return Err(tool_failure(error).into()),
    };
    state
        .stats
        .tool_calls
        .fetch_add(accepted.len() as u64, Ordering::Relaxed);

    let reasoning_tokens = estimate_tokens(&answer.reasoning);
    let mut output_tokens = estimate_tokens(&prose) + reasoning_tokens;
    for call in &accepted {
        output_tokens += estimate_tokens(&format!("{}{}", call.name, call.arguments));
    }
    let calls: Vec<FunctionCall> = accepted
        .into_iter()
        .map(|call| FunctionCall {
            name: call.name,
            arguments: call.arguments,
        })
        .collect();
    Ok(TurnOutcome {
        calls,
        usage: Usage {
            input_tokens: estimate_tokens(&serde_json::to_string(&messages).unwrap_or_default()),
            cached_tokens: 0,
            output_tokens,
            reasoning_tokens,
        },
        checkpoint: None,
        truncated: truncated.then(|| "max_output_tokens".to_string()),
    })
}
async fn send(tx: &mpsc::Sender<TurnEvent>, event: TurnEvent) -> Result<(), TurnError> {
    tx.send(event).await.map_err(|_| TurnError::Cancelled)
}
fn client_error(error: chatgpt_client::Error) -> TurnError {
    use chatgpt_client::ErrorKind;
    let (status, code, message) = match error.kind() {
        ErrorKind::Authentication => (
            StatusCode::BAD_GATEWAY,
            "upstream_auth_error",
            "ChatGPT Web authentication failed",
        ),
        ErrorKind::RateLimit => (
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_exceeded",
            "ChatGPT Web is rate limited",
        ),
        ErrorKind::Timeout | ErrorKind::Network => (
            StatusCode::GATEWAY_TIMEOUT,
            "request_timeout",
            "ChatGPT Web request timed out",
        ),
        _ => (
            StatusCode::BAD_GATEWAY,
            "backend_error",
            "ChatGPT Web request failed",
        ),
    };
    ProxyError::new(message, status, code).into()
}
fn bad_request(param: &str, message: &str) -> TurnError {
    ProxyError::new(message, StatusCode::BAD_REQUEST, "invalid_request")
        .with_param(param)
        .into()
}
fn upstream_error(message: &str) -> TurnError {
    ProxyError::new(message, StatusCode::BAD_GATEWAY, "backend_error").into()
}
fn timeout_error() -> TurnError {
    ProxyError::new(
        "Request processing timed out",
        StatusCode::GATEWAY_TIMEOUT,
        "request_timeout",
    )
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use responses_core::capability::Support;

    #[test]
    fn model_allowlist_is_case_insensitive_and_excludes_deepseek() {
        for model in crate::config::CHATGPT_MODELS {
            assert!(supports_model(model), "{model}");
            assert!(supports_model(&model.to_ascii_uppercase()), "{model}");
            assert!(supports_model(&format!("  {model}  ")), "{model}");
        }
        for model in [
            "deepseek-chat",
            "deepseek-v3",
            "gpt-4o",
            // "Luna" is the display title of the gpt-5-6 family, not a slug.
            "gpt-6-luna",
            "gpt-5-6-luna",
        ] {
            assert!(!supports_model(model), "{model}");
        }
    }

    #[test]
    fn ceiling_declares_all_capabilities_and_only_engine_supported_features() {
        let mut caps = ceiling();
        for capability in Capability::ALL {
            let status = caps
                .entry(capability)
                .unwrap_or_else(|| panic!("undeclared {}", capability.id()));
            if status.support == Support::Supported {
                assert!(capability.engine_supported());
            }
        }
        assert_eq!(caps.status(Capability::WebSearch), Support::Supported);
        assert_eq!(caps.status(Capability::ImageGeneration), Support::Supported);
        // Tools are prompted, not native: the proxy provides them, so the
        // upstream cannot silently drop the contract.
        assert_eq!(caps.status(Capability::FunctionTools), Support::Supported);
        assert_eq!(caps.status(Capability::ToolChoice), Support::Supported);
        assert_eq!(caps.status(Capability::MaxOutputTokens), Support::Supported);
        assert_eq!(
            caps.status(Capability::CodeInterpreter),
            Support::Unsupported
        );
        assert_eq!(caps.status(Capability::CustomTools), Support::Unsupported);
        assert!(caps.restrict_to_engine().is_empty());
        assert_eq!(
            caps.enabled_ids(),
            [
                "text",
                "streaming",
                "non_streaming",
                "image_input",
                "function_tools",
                "tool_choice",
                "web_search",
                "image_generation",
                "reasoning",
                "verbosity",
                "max_output_tokens",
                "stored_responses",
                "previous_response"
            ]
        );
    }
    fn turn(items: Vec<Item>) -> TurnRequest {
        TurnRequest {
            thread_id: "test".into(),
            model: "gpt-5".into(),
            instructions: Some("follow rules".into()),
            transcript: items,
            tools: Vec::new(),
            hosted_tools: Vec::new(),
            tool_choice: responses_core::plan::ToolChoice::Auto,
            parallel_tool_calls: true,
            reasoning_effort: Some(ReasoningEffort::High),
            verbosity: None,
            temperature: None,
            top_p: None,
            max_output_tokens: None,
        }
    }

    #[tokio::test]
    async fn transcript_conversion_preserves_roles_and_part_order() {
        let client = ChatGptClient::new(ClientConfig {
            base_url: "https://chatgpt.invalid".into(),
            access_token: "offline".into(),
            proxy_url: None,
            timeout: Duration::from_secs(1),
        })
        .unwrap();
        let request = turn(vec![
            Item::Message {
                role: Role::User,
                content: vec![
                    Part::Text { text: "one".into() },
                    Part::Text { text: "two".into() },
                ],
            },
            Item::Message {
                role: Role::Assistant,
                content: vec![Part::Text {
                    text: "answer".into(),
                }],
            },
        ]);
        let (tx, _rx) = mpsc::channel(2);
        let policy = tool_policy(&request);
        let messages = web_messages(
            &client,
            &request,
            &policy,
            1024,
            &tx,
            Instant::now(),
            Duration::from_secs(1),
        )
        .await
        .unwrap();
        assert_eq!(messages[0]["author"]["role"], "system");
        assert_eq!(messages[0]["content"]["parts"][0], "follow rules");
        assert_eq!(messages[1]["author"]["role"], "user");
        assert_eq!(messages[1]["content"]["parts"], json!(["one", "two"]));
        assert_eq!(messages[2]["author"]["role"], "assistant");
    }

    #[tokio::test]
    async fn tools_are_taught_and_prior_calls_replay_as_invoke_blocks() {
        let client = ChatGptClient::new(ClientConfig {
            base_url: "https://chatgpt.invalid".into(),
            access_token: "offline".into(),
            proxy_url: None,
            timeout: Duration::from_secs(1),
        })
        .unwrap();
        let mut request = turn(vec![
            Item::Message {
                role: Role::User,
                content: vec![Part::Text {
                    text: "list files".into(),
                }],
            },
            Item::FunctionCall {
                call_id: "call_1".into(),
                name: "bash".into(),
                arguments: "{\"command\": \"ls\"}".into(),
            },
            Item::FunctionCallOutput {
                call_id: "call_1".into(),
                output: "a.txt".into(),
            },
        ]);
        request.tools = vec![FunctionTool {
            name: "bash".into(),
            description: Some("run a command".into()),
            parameters: json!({"type": "object", "properties": {"command": {"type": "string"}}}),
            strict: None,
        }];
        let (tx, _rx) = mpsc::channel(2);
        let policy = tool_policy(&request);
        assert!(policy.has_tools());
        let messages = web_messages(
            &client,
            &request,
            &policy,
            1024,
            &tx,
            Instant::now(),
            Duration::from_secs(1),
        )
        .await
        .unwrap();
        // The system message teaches the wire format and names the tool.
        let system = messages[0]["content"]["parts"][0].as_str().unwrap();
        assert!(system.contains("TOOL CALL PROTOCOL"), "{system}");
        assert!(system.contains("### bash"), "{system}");
        assert!(system.contains("follow rules"), "{system}");
        // A prior call replays in the same syntax the model must emit.
        let call = messages[2]["content"]["parts"][0].as_str().unwrap();
        assert_eq!(call, "<invoke name=\"bash\">{\"command\": \"ls\"}</invoke>");
        // The result and the re-anchoring reminder follow it.
        let result = messages[3]["content"]["parts"][0].as_str().unwrap();
        assert!(result.starts_with("Tool result (id=call_1):"), "{result}");
        let result_parts = messages[3]["content"]["parts"].as_array().unwrap();
        let reminder = result_parts.last().unwrap().as_str().unwrap();
        assert!(reminder.contains("HARD FORMAT REQUIREMENT"), "{reminder}");
    }

    #[test]
    fn a_turn_without_tools_keeps_the_system_message_plain() {
        let request = turn(vec![]);
        let policy = tool_policy(&request);
        assert!(!policy.has_tools());
        assert!(policy.directive.is_empty());
    }

    #[test]
    fn tool_choice_maps_onto_the_shared_policy() {
        let mut request = turn(vec![]);
        request.tools = vec![
            FunctionTool {
                name: "a".into(),
                description: None,
                parameters: json!({"type": "object"}),
                strict: None,
            },
            FunctionTool {
                name: "b".into(),
                description: None,
                parameters: json!({"type": "object"}),
                strict: None,
            },
        ];
        request.tool_choice = ToolChoice::Function("b".into());
        let policy = tool_policy(&request);
        assert_eq!(policy.specs.len(), 1);
        assert_eq!(policy.specs[0].name, "b");
        assert!(policy.required);

        request.tool_choice = ToolChoice::None;
        let policy = tool_policy(&request);
        assert!(!policy.has_tools());
        assert!(policy.directive.contains("Do not call tools"));

        request.tool_choice = ToolChoice::Auto;
        request.parallel_tool_calls = false;
        let policy = tool_policy(&request);
        assert_eq!(policy.specs.len(), 2);
        assert!(policy.directive.contains("Return at most one tool call."));
    }

    #[test]
    fn the_output_budget_trims_at_a_character_boundary() {
        assert_eq!(trim_to_tokens("abcdefgh", 1), "abcd");
        assert_eq!(trim_to_tokens("ab", 10), "ab");
        assert_eq!(trim_to_tokens("答案答案答", 1), "答案答案");
        assert_eq!(trim_to_tokens("答案答案", 0), "");
    }

    #[test]
    fn reasoning_verbosity_and_model_hints_map() {
        assert_eq!(system_hints("gpt-image-2", &[]), ["picture_v2"]);
        assert_eq!(
            thinking_effort(Some(ReasoningEffort::XHigh)).as_deref(),
            Some("extended")
        );
        assert_eq!(
            verbosity_instruction(Verbosity::Low),
            "Answer concisely, keeping the response as short as the task allows."
        );
        assert_eq!(
            ConversationEvent::TextDelta("hello".into()),
            ConversationEvent::TextDelta("hello".into())
        );
        assert_eq!(
            ConversationEvent::ReasoningDelta("plan".into()),
            ConversationEvent::ReasoningDelta("plan".into())
        );
    }
}
