//! Correction policy for prompted tool calls.
//!
//! A channel that has no native function-call API teaches a wire format in the
//! prompt and parses it back out of the model's text. That round trip can fail:
//! the body is not JSON, the arguments miss the schema, a required call never
//! arrives. This module owns the decision of what to do about it — accept, or
//! record a failure and ask for a correction — with no IO and no knowledge of
//! any particular upstream.
//!
//! The caller drives it: feed every [`crate::dialect::ScanEvent`] from the
//! answer, then loop over [`Recovery::needs_repair`] /
//! [`Recovery::begin_repair`] / [`Recovery::end_repair`], sending the returned
//! prompt through whatever continuation the channel supports.

use crate::dialect::{self, MalformedCall, ScanEvent, ToolCall, ToolSpec};
use std::collections::HashMap;

/// Why a turn could not produce a usable set of tool calls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolError {
    /// The model emitted more than one call while `parallel_tool_calls` is off.
    ParallelCallsDisabled,
    /// Correction turns ran out with failures still outstanding.
    RepairFailed,
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::ParallelCallsDisabled => {
                "Backend produced multiple calls with parallel_tool_calls=false"
            }
            Self::RepairFailed => "Could not recover a valid tool response",
        })
    }
}

impl std::error::Error for ToolError {}

/// Accepted calls, outstanding failures and the correction bookkeeping.
#[derive(Debug, Clone, Default)]
pub struct Recovery {
    specs: Vec<ToolSpec>,
    calls: Vec<ToolCall>,
    pending: Vec<MalformedCall>,
    required: bool,
    parallel: bool,
    previous: HashMap<(String, String), usize>,
    repair_seen: HashMap<(String, String), usize>,
    expected: usize,
    resolved: usize,
    repairing: bool,
}

impl Recovery {
    /// Build a recovery state for one turn.
    ///
    /// `required` comes from `tool_choice` (`required` or a named function) and
    /// makes an answer without any call a failure. `parallel` mirrors
    /// `parallel_tool_calls`.
    pub fn new(specs: Vec<ToolSpec>, required: bool, parallel: bool) -> Self {
        Self {
            specs,
            parallel,
            required,
            ..Self::default()
        }
    }

    /// Feed one event from the answer text.
    ///
    /// Malformed blocks and schema violations join the pending list; a text
    /// event is ignored. Returns the call when one was accepted.
    pub fn accept(&mut self, event: ScanEvent) -> Result<Option<ToolCall>, ToolError> {
        let call = match event {
            ScanEvent::Malformed(m) => {
                self.pending.push(m);
                return Ok(None);
            }
            ScanEvent::Text(_) => return Ok(None),
            ScanEvent::Tool(c) => c,
        };

        if let Some(reason) = dialect::validation_reason(&self.specs, &call) {
            self.pending.push(MalformedCall::new(
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
                // The correction echoed a call that was already accepted.
                return Ok(None);
            }
        }
        if !self.parallel && !self.calls.is_empty() {
            return Err(ToolError::ParallelCallsDisabled);
        }
        self.calls.push(call.clone());
        self.resolved += 1;
        Ok(Some(call))
    }

    /// Feed a whole answer body, then apply the `required` rule.
    ///
    /// A truncated turn must not call this: the output budget may have cut a
    /// block mid-arguments, and a half-parsed call is worse than none.
    pub fn accept_text(&mut self, text: &str) -> Result<(), ToolError> {
        for event in dialect::scan(text) {
            self.accept(event)?;
        }
        self.require_call();
        Ok(())
    }

    /// `tool_choice: "required"` must yield at least one call.
    pub fn require_call(&mut self) {
        if self.required && self.calls.is_empty() && self.pending.is_empty() {
            self.pending.push(MalformedCall::new(
                None,
                String::new(),
                "required_tool_call_missing",
            ));
        }
    }

    /// Whether a correction turn is still worth sending.
    pub fn needs_repair(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Outstanding failures, for logging.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Calls accepted so far.
    pub fn calls(&self) -> &[ToolCall] {
        &self.calls
    }

    /// Build the correction prompt and reset the per-attempt counters.
    ///
    /// `previous` snapshots every call accepted so far, so the next correction
    /// cannot re-accept an echo of one of them.
    pub fn begin_repair(&mut self) -> String {
        let mut prompt = dialect::build_repair_prompt(&self.pending, &self.specs);
        // Only the schema of a tool the model actually named and got wrong on
        // its arguments. Sending the whole catalogue here makes a correction
        // turn larger than the turn it repairs — in exactly the situation
        // (context pressure) where a correction is most likely needed.
        let schemas = dialect::repair_schemas(&self.pending, &self.specs);
        if !schemas.is_empty() {
            prompt.push_str("\nSchema(s) for the tool call(s) that failed:\n");
            prompt.push_str(&dialect::py_json_dumps(&serde_json::Value::Array(schemas)));
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

    /// Close a correction attempt. An attempt that resolved nothing records
    /// exactly one omission marker, never the whole failure list again.
    pub fn end_repair(&mut self) {
        if self.resolved < self.expected {
            self.pending.push(MalformedCall::new(
                None,
                String::new(),
                "repair_omitted_call",
            ));
        }
    }

    /// How many calls the current correction attempt still has to resolve.
    pub fn expected_count(&self) -> usize {
        self.expected
    }

    /// Failure records still outstanding, for diagnostics and tests.
    pub fn pending(&self) -> &[MalformedCall] {
        &self.pending
    }

    /// Finish the turn with the accepted calls.
    pub fn finish(self) -> Result<Vec<ToolCall>, ToolError> {
        if self.pending.is_empty() {
            Ok(self.calls)
        } else {
            Err(ToolError::RepairFailed)
        }
    }
}

/// How `tool_choice` narrows the offered tools and sets the `required` flag.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolPolicy {
    pub specs: Vec<ToolSpec>,
    pub required: bool,
    pub parallel: bool,
    /// Extra prompt lines the request contract requires, already newline
    /// prefixed, matching the OpenAI proxy's `ChatService` behaviour.
    pub directive: String,
}

impl ToolPolicy {
    /// Resolve a request's tools, `tool_choice` and `parallel_tool_calls`.
    ///
    /// `tool_choice` may be `"none"`, `"auto"`, `"required"` or a named
    /// function object; anything else leaves the tools untouched.
    pub fn resolve(
        tools: &[serde_json::Value],
        tool_choice: Option<&serde_json::Value>,
        parallel: bool,
    ) -> Self {
        let mut tools = tools.to_vec();
        let mut required = false;
        match tool_choice {
            Some(serde_json::Value::String(s)) if s == "none" => tools.clear(),
            Some(serde_json::Value::String(s)) if s == "required" => required = true,
            Some(serde_json::Value::Object(obj)) => {
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
        let specs = dialect::tool_specs(&tools);
        let directive = Self::directive(&specs, required, parallel, tool_choice);
        Self {
            specs,
            required,
            parallel,
            directive,
        }
    }

    fn directive(
        specs: &[ToolSpec],
        required: bool,
        parallel: bool,
        tool_choice: Option<&serde_json::Value>,
    ) -> String {
        let mut out = String::new();
        if !specs.is_empty() {
            if required {
                out.push_str("\nYou must return a tool call.");
            }
            if !parallel {
                out.push_str("\nReturn at most one tool call.");
            }
        } else if matches!(tool_choice, Some(serde_json::Value::String(s)) if s == "none") {
            out.push_str("\nDo not call tools in this turn; reply with normal text.");
        }
        out
    }

    /// Whether this turn offers any tool at all.
    pub fn has_tools(&self) -> bool {
        !self.specs.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn specs() -> Vec<ToolSpec> {
        dialect::tool_specs(&[json!({
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "look something up",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"]
                }
            }
        })])
    }

    #[test]
    fn accepts_a_conforming_call_and_rejects_an_unknown_tool() {
        let mut recovery = Recovery::new(specs(), false, true);
        recovery
            .accept_text("<invoke name=\"lookup\">{\"q\":\"x\"}</invoke>")
            .unwrap();
        assert_eq!(recovery.calls().len(), 1);
        assert!(!recovery.needs_repair());
        assert_eq!(recovery.finish().unwrap().len(), 1);

        let mut recovery = Recovery::new(specs(), false, true);
        recovery
            .accept_text("<invoke name=\"missing\">{\"q\":\"x\"}</invoke>")
            .unwrap();
        assert!(recovery.needs_repair());
        assert_eq!(recovery.finish(), Err(ToolError::RepairFailed));
    }

    #[test]
    fn required_without_a_call_becomes_a_failure() {
        let mut recovery = Recovery::new(specs(), true, true);
        recovery.accept_text("no call here").unwrap();
        assert!(recovery.needs_repair());
    }

    #[test]
    fn two_calls_with_parallel_disabled_are_rejected() {
        let mut recovery = Recovery::new(specs(), false, false);
        let error = recovery
            .accept_text(concat!(
                "<invoke name=\"lookup\">{\"q\":\"a\"}</invoke>",
                "<invoke name=\"lookup\">{\"q\":\"b\"}</invoke>"
            ))
            .unwrap_err();
        assert_eq!(error, ToolError::ParallelCallsDisabled);
    }

    #[test]
    fn a_repair_cannot_re_accept_an_echo_of_an_accepted_call() {
        let mut recovery = Recovery::new(specs(), false, true);
        recovery
            .accept_text("<invoke name=\"lookup\">{\"q\":\"a\"}</invoke><invoke name=\"lookup\">{\"q\":\"b\"}</invoke>")
            .unwrap();
        // Force a repair by adding a failure.
        recovery.pending.push(MalformedCall::new(
            Some("lookup".into()),
            "not json".into(),
            "invalid_json_body",
        ));
        let prompt = recovery.begin_repair();
        assert!(prompt.contains("invalid_json_body"));
        // The model repeats the accepted calls and adds nothing.
        recovery
            .accept_text("<invoke name=\"lookup\">{\"q\":\"a\"}</invoke><invoke name=\"lookup\">{\"q\":\"b\"}</invoke>")
            .unwrap();
        recovery.end_repair();
        assert_eq!(recovery.finish(), Err(ToolError::RepairFailed));
    }

    #[test]
    fn policy_narrows_tools_and_builds_the_directive() {
        let tools = vec![
            json!({"type":"function","function":{"name":"a","parameters":{"type":"object"}}}),
            json!({"type":"function","function":{"name":"b","parameters":{"type":"object"}}}),
        ];
        let none = ToolPolicy::resolve(&tools, Some(&json!("none")), true);
        assert!(!none.has_tools());
        assert_eq!(
            none.directive,
            "\nDo not call tools in this turn; reply with normal text."
        );

        let required = ToolPolicy::resolve(&tools, Some(&json!("required")), false);
        assert_eq!(required.specs.len(), 2);
        assert!(required.required);
        assert!(required.directive.contains("You must return a tool call."));
        assert!(required.directive.contains("Return at most one tool call."));

        let named = ToolPolicy::resolve(
            &tools,
            Some(&json!({"type":"function","function":{"name":"b"}})),
            true,
        );
        assert_eq!(named.specs.len(), 1);
        assert_eq!(named.specs[0].name, "b");
        assert!(named.required);

        let auto = ToolPolicy::resolve(&tools, Some(&json!("auto")), true);
        assert_eq!(auto.specs.len(), 2);
        assert!(!auto.required);
        assert!(auto.directive.is_empty());
    }
}
