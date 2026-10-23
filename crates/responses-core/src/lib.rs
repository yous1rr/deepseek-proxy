//! Channel-independent engine for the OpenAI Responses protocol.
//!
//! This crate knows nothing about HTTP serving, storage, accounts, or any
//! particular upstream. It turns a `POST /v1/responses` body into a validated
//! [`plan::TurnPlan`], decides from a [`capability::CapabilitySet`] whether the
//! active channel can serve it, keeps conversation state as a neutral
//! [`transcript::Item`] list, and encodes generation progress as Responses
//! stream events with [`events::EventEncoder`].
//!
//! A channel adapter (for example the DeepSeek web adapter in the proxy crate)
//! only has to declare its capability ceiling, consume a transcript, and
//! report text, reasoning, and function calls. Everything that makes the wire
//! format look like the Responses API lives here, so a second channel reuses
//! it unchanged.
//!
//! Design rules:
//!
//! - **Fail closed.** A capability that was not declared supported is off, and
//!   an operator profile can only narrow a channel's ceiling, never widen it.
//! - **No silent emulation.** A parameter that would change model behaviour is
//!   rejected unless the channel declares the matching capability.
//! - **No secrets or raw upstream data** pass through this crate.

pub mod capability;
pub mod events;
pub mod ids;
pub mod plan;
pub mod transcript;
