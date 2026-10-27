//! Contract between the `/v1/responses` HTTP layer and an upstream channel.
//!
//! The HTTP layer owns authentication, ownership, capability gating, stored
//! response chains, the per-thread turn lock, event encoding and persistence.
//! A channel only has to turn one [`TurnRequest`] into model output: it emits
//! [`TurnEvent`]s while generating and returns a [`TurnOutcome`].
//!
//! Adding a channel means implementing [`ResponsesChannel`] and declaring an
//! honest capability ceiling. See `RESPONSES_ARCHITECTURE.md`.

use crate::store::{ThreadHistory, ThreadResume};
use crate::{AppState, ProxyError};
use futures_util::future::BoxFuture;
use responses_core::capability::CapabilitySet;
use responses_core::events::{FunctionCall, HostedToolResult, Usage};
use responses_core::plan::{FunctionTool, ReasoningEffort, ToolChoice, Verbosity};
use responses_core::transcript::Item;
use tokio::sync::mpsc;

/// Everything a channel needs for one model turn. Only requested features
/// that passed capability gating reach this struct.
#[derive(Debug, Clone)]
pub struct TurnRequest {
    /// Proxy-side conversation key. Channels that keep server-side sessions
    /// use it to find resumable state; the turn lock is already held for it.
    pub thread_id: String,
    pub model: String,
    /// This request's instructions. They are not inherited from earlier
    /// responses in a `previous_response_id` chain.
    pub instructions: Option<String>,
    /// The full conversation: replayed stored context plus this input.
    pub transcript: Vec<Item>,
    pub tools: Vec<FunctionTool>,
    /// Hosted Responses tool types requested by this turn, such as web_search.
    pub hosted_tools: Vec<String>,
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: bool,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub verbosity: Option<Verbosity>,
    /// Sampling and length controls reach a channel only when it enables
    /// `sampling_controls` / `max_output_tokens`; default values (1.0) may
    /// arrive regardless. DeepSeek web enables neither, so it reads none.
    #[allow(dead_code)]
    pub temperature: Option<f64>,
    #[allow(dead_code)]
    pub top_p: Option<f64>,
    #[allow(dead_code)]
    pub max_output_tokens: Option<i64>,
}

/// Progress reported while a turn runs.
#[derive(Debug, Clone, PartialEq)]
pub enum TurnEvent {
    /// The upstream accepted the turn. Errors returned before this event are
    /// sent to the client as plain HTTP errors (so SDK retries keep working);
    /// later errors end the stream with `response.failed`.
    Started,
    /// Answer text, already free of any channel-specific tool syntax.
    Text(String),
    /// Raw reasoning text, when the channel exposes it.
    Reasoning(String),
    /// A completed hosted-tool output item from the upstream channel.
    Hosted(HostedToolResult),
}

/// Resumable upstream state to store with a successful turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Checkpoint {
    pub resume: ThreadResume,
    pub history: ThreadHistory,
}

/// Result of a successful turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnOutcome {
    pub calls: Vec<FunctionCall>,
    pub usage: Usage,
    /// `None` when the upstream exposed no resumable cursor; the next turn
    /// then replays the transcript.
    pub checkpoint: Option<Checkpoint>,
    /// `Some(reason)` when the channel stopped at its output budget instead of
    /// letting the model finish. The reason is reported verbatim as
    /// `incomplete_details.reason`.
    pub truncated: Option<String>,
}

/// Why a turn did not produce an outcome.
pub enum TurnError {
    /// A client-safe error.
    Proxy(ProxyError),
    /// A prepared HTTP error from a shared helper.
    Http(axum::response::Response),
    /// The consumer went away; nothing is reported.
    Cancelled,
}

impl std::fmt::Debug for TurnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Proxy(error) => write!(f, "Proxy({error:?})"),
            Self::Http(response) => write!(f, "Http({})", response.status()),
            Self::Cancelled => f.write_str("Cancelled"),
        }
    }
}

impl From<ProxyError> for TurnError {
    fn from(error: ProxyError) -> Self {
        Self::Proxy(error)
    }
}

/// One upstream channel behind `/v1/responses`.
pub trait ResponsesChannel: Send + Sync + 'static {
    /// Stable id, matched against a capability profile's `channel`.
    fn id(&self) -> &'static str;

    /// The adapter's capability ceiling: everything it implements.
    fn capabilities(&self) -> CapabilitySet;

    /// Whether `model` names a model this channel can serve.
    fn supports_model(&self, model: &str) -> bool;

    /// Run one turn. The caller holds the turn lock for `turn.thread_id`.
    ///
    /// Send events in order on `events`. When `events` is closed the
    /// consumer is gone: stop upstream work and return
    /// [`TurnError::Cancelled`].
    fn run_turn(
        &self,
        state: AppState,
        turn: TurnRequest,
        events: mpsc::Sender<TurnEvent>,
    ) -> BoxFuture<'static, Result<TurnOutcome, TurnError>>;
}
