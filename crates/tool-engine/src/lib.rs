//! Channel-independent prompted tool calling.
//!
//! Some upstreams expose no native function-call API: the model is a chat
//! window, and the only way to make it call a tool is to teach a wire format in
//! the prompt and parse it back out of the answer text. That is the same
//! problem for every such channel, so this crate owns the parts that do not
//! depend on which upstream produced the text:
//!
//! * [`dialect`] — the `<invoke name="tool">{json}</invoke>` wire format:
//!   teaching prose, extraction, schema validation, the incremental streaming
//!   filter, and the correction prompt.
//! * [`recovery`] — the accept/repair state machine, plus [`ToolPolicy`] which
//!   turns `tools` / `tool_choice` / `parallel_tool_calls` into the offered
//!   tool set and its prompt directives.
//!
//! Nothing here does IO, holds a session, or knows about a cursor, so a channel
//! keeps full control of how a correction turn is delivered. A channel with a
//! resumable server-side session sends the correction into that session; a
//! channel that replays the whole transcript appends it and re-sends.
//!
//! [`ToolPolicy`]: recovery::ToolPolicy

pub mod dialect;
pub mod recovery;

pub use recovery::{Recovery, ToolError, ToolPolicy};
