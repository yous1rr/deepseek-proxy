//! ChatGPT Web Responses channel adapter.
//!
//! ChatGPT's private web protocol exposes no stable resumable response cursor,
//! so continuation is handled by replaying the locally stored transcript.

use super::channel::{ResponsesChannel, TurnError, TurnEvent, TurnOutcome, TurnRequest};
use crate::{AppState, ProxyError};
use axum::http::StatusCode;
use chatgpt_client::{ChatGptClient, ClientConfig, ConversationRequest};
use chatgpt_protocol::{consume_sse_line, ConversationEvent, PatchState};
use futures_util::{future::BoxFuture, FutureExt, StreamExt};
use responses_core::capability::{Capability, CapabilitySet, CapabilityStatus, Provider};
use responses_core::events::{HostedSource, HostedToolKind, HostedToolResult, Usage};
use responses_core::plan::{ReasoningEffort, Verbosity};
use responses_core::transcript::{Item, Part, Role};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

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
    let supported: [(Capability, Provider, &str); 10] = [
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
        (C::Reasoning, Provider::Upstream, "thinking_effort"),
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
    let unsupported: [(Capability, &str); 20] = [
        (C::FileInput, "no_chatgpt_web_file_input"),
        (C::FunctionTools, "no_function_call_event_mapping"),
        (C::ToolChoice, "no_function_tool_execution"),
        (C::CustomTools, "no_custom_tool_execution"),
        (C::FileSearch, "no_file_search_implementation"),
        (C::CodeInterpreter, "no_code_interpreter_event_mapping"),
        (C::Mcp, "no_mcp_client"),
        (C::ComputerUse, "no_computer_use_event_mapping"),
        (C::Shell, "no_shell_tool_execution"),
        (C::StructuredOutput, "no_upstream_schema_enforcement"),
        (C::SamplingControls, "chatgpt_web_has_no_sampling_controls"),
        (C::MaxOutputTokens, "chatgpt_web_has_no_output_token_limit"),
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
fn supports_model(model: &str) -> bool {
    matches!(
        model.trim().to_ascii_lowercase().as_str(),
        "auto"
            | "gpt-5"
            | "gpt-5-1"
            | "gpt-5-2"
            | "gpt-5-3"
            | "gpt-5-3-mini"
            | "gpt-5-mini"
            | "gpt-image-2"
            | "codex-gpt-image-2"
    )
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
async fn web_messages(
    client: &ChatGptClient,
    turn: &TurnRequest,
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
    if let Some(instructions) = turn.instructions.as_deref().filter(|v| !v.is_empty()) {
        text_bytes = instructions.len();
        if text_bytes > MAX_TEXT_BYTES {
            return Err(bad_request(
                "instructions",
                "Instructions exceed the supported size limit",
            ));
        }
        messages.push(web_message("system", vec![json!(instructions)]));
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
            } => web_message(
                "assistant",
                vec![json!(format!(
                    "Function call {name} (id {call_id}): {arguments}"
                ))],
            ),
            Item::FunctionCallOutput { call_id, output } => web_message(
                "user",
                vec![json!(format!("Function result (id {call_id}): {output}"))],
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
async fn run_turn(
    client: ChatGptClient,
    state: AppState,
    turn: TurnRequest,
    events: mpsc::Sender<TurnEvent>,
) -> Result<TurnOutcome, TurnError> {
    let started = Instant::now();
    let timeout = Duration::from_secs_f64(state.settings.request_timeout_secs.max(0.001));
    let messages = tokio::select! {
        biased;
        _ = events.closed() => return Err(TurnError::Cancelled),
        result = web_messages(&client, &turn, state.settings.image_max_bytes, &events, started, timeout) => result?,
    };
    let request = ConversationRequest {
        messages: messages.clone(),
        model: turn.model.clone(),
        thinking_effort: thinking_effort(turn.reasoning_effort),
        system_hints: system_hints(&turn.model, &turn.hosted_tools),
    };
    let response = tokio::select! {
        biased;
        _ = events.closed() => return Err(TurnError::Cancelled),
        result = tokio::time::timeout(timeout.saturating_sub(started.elapsed()), client.conversation(&request)) => match result {
            Ok(Ok(response)) => response, Ok(Err(error)) => return Err(client_error(error)), Err(_) => return Err(timeout_error()),
        },
    };
    if events.send(TurnEvent::Started).await.is_err() {
        return Err(TurnError::Cancelled);
    }

    let mut protocol = PatchState::new();
    let mut pending = Vec::new();
    let mut stream = response.bytes_stream();
    let mut output = String::new();
    let mut reasoning = String::new();
    let mut event_count = 0usize;
    let mut done = false;
    loop {
        let next = tokio::select! {
            biased;
            _ = events.closed() => return Err(TurnError::Cancelled),
            value = tokio::time::timeout(timeout.saturating_sub(started.elapsed()), stream.next()) => match value {
                Ok(value) => value, Err(_) => return Err(timeout_error()),
            },
        };
        let Some(chunk) = next else { break };
        pending.extend_from_slice(
            &chunk.map_err(|_| upstream_error("ChatGPT Web stream could not be read"))?,
        );
        let mut offset = 0usize;
        while offset < pending.len() {
            let Some(relative_end) = pending[offset..].iter().position(|byte| *byte == b'\n')
            else {
                if pending.len().saturating_sub(offset) > MAX_SSE_LINE_BYTES {
                    return Err(upstream_error("ChatGPT Web sent an oversized event"));
                }
                break;
            };
            let end = offset + relative_end;
            let line_end = if end > offset && pending[end - 1] == b'\r' {
                end - 1
            } else {
                end
            };
            if line_end - offset > MAX_SSE_LINE_BYTES {
                return Err(upstream_error("ChatGPT Web sent an oversized event"));
            }
            let line = std::str::from_utf8(&pending[offset..line_end])
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
                        output.push_str(&text);
                        send(&events, TurnEvent::Text(text)).await?;
                    }
                    ConversationEvent::ReasoningDelta(text) => {
                        reasoning.push_str(&text);
                        send(&events, TurnEvent::Reasoning(text)).await?;
                    }
                    ConversationEvent::ImageTool {
                        conversation_id,
                        file_ids,
                        sediment_ids,
                    } => {
                        if !conversation_id.is_empty() {
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
                                    &events,
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
                        let sources = sources
                            .into_iter()
                            .map(|source| HostedSource {
                                title: source.title,
                                url: source.url,
                                snippet: source.snippet,
                            })
                            .collect();
                        send(
                            &events,
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
            pending.drain(..offset);
        }
        if done {
            break;
        }
    }
    if !pending.is_empty() && !done {
        let line = std::str::from_utf8(&pending)
            .map_err(|_| upstream_error("ChatGPT Web sent an invalid event"))?;
        if let Some(event) = consume_sse_line(&mut protocol, line)
            .map_err(|_| upstream_error("ChatGPT Web sent an invalid event"))?
        {
            match event {
                ConversationEvent::TextDelta(text) => {
                    output.push_str(&text);
                    send(&events, TurnEvent::Text(text)).await?;
                }
                ConversationEvent::ReasoningDelta(text) => {
                    reasoning.push_str(&text);
                    send(&events, TurnEvent::Reasoning(text)).await?;
                }
                ConversationEvent::Done => done = true,
                _ => {}
            }
        }
    }
    if !done && output.is_empty() && reasoning.is_empty() {
        return Err(upstream_error("ChatGPT Web ended without a response"));
    }
    let reasoning_tokens = estimate_tokens(&reasoning);
    Ok(TurnOutcome {
        calls: Vec::new(),
        usage: Usage {
            input_tokens: estimate_tokens(&serde_json::to_string(&messages).unwrap_or_default()),
            cached_tokens: 0,
            output_tokens: estimate_tokens(&output).saturating_add(reasoning_tokens),
            reasoning_tokens,
        },
        checkpoint: None,
        truncated: None,
    })
}
async fn send(tx: &mpsc::Sender<TurnEvent>, event: TurnEvent) -> Result<(), TurnError> {
    tx.send(event).await.map_err(|_| TurnError::Cancelled)
}
fn estimate_tokens(text: &str) -> u64 {
    (text.chars().count() as u64).div_ceil(4)
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
        for model in [
            "auto",
            "gpt-5",
            "gpt-5-1",
            "gpt-5-2",
            "gpt-5-3",
            "gpt-5-3-mini",
            "gpt-5-mini",
            "gpt-image-2",
            "codex-gpt-image-2",
        ] {
            assert!(supports_model(model));
            assert!(supports_model(&model.to_ascii_uppercase()));
        }
        for model in ["deepseek-chat", "deepseek-v3", "gpt-4o"] {
            assert!(!supports_model(model));
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
        assert_eq!(
            caps.status(Capability::CodeInterpreter),
            Support::Unsupported
        );
        assert!(caps.restrict_to_engine().is_empty());
        assert_eq!(
            caps.enabled_ids(),
            [
                "text",
                "streaming",
                "non_streaming",
                "image_input",
                "web_search",
                "image_generation",
                "reasoning",
                "verbosity",
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
        let messages = web_messages(
            &client,
            &request,
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
