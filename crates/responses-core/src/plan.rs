//! Request validation and turn planning.
//!
//! [`TurnPlan::from_request`] validates a `POST /v1/responses` body, records
//! which capabilities it needs, and normalizes it into a channel-neutral plan.
//! The caller gates the plan against the active channel with
//! [`TurnPlan::unmet`] before any upstream work starts.

use crate::capability::{Capability, CapabilitySet};
use crate::transcript::{normalize_input, Item};
use serde::Deserialize;
use serde_json::Value;

/// A client error with an OpenAI-shaped `param` and stable `code`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestError {
    pub message: String,
    pub param: Option<String>,
    pub code: &'static str,
}

impl RequestError {
    pub fn invalid(param: &str, message: &str) -> Self {
        Self {
            message: message.to_string(),
            param: Some(param.to_string()),
            code: "invalid_request",
        }
    }

    pub fn unsupported(param: &str, message: &str) -> Self {
        Self {
            message: message.to_string(),
            param: Some(param.to_string()),
            code: "unsupported_parameter",
        }
    }
}

impl std::fmt::Display for RequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RequestError {}

/// Raw request body. Unknown fields are collected in `extra` and rejected
/// unless they are known to be harmless hints.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct CreateRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub input: Option<Value>,
    #[serde(default)]
    pub instructions: Option<Value>,
    #[serde(default)]
    pub previous_response_id: Option<String>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub store: Option<bool>,
    #[serde(default)]
    pub background: Option<bool>,
    #[serde(default)]
    pub conversation: Option<Value>,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub reasoning: Option<Value>,
    #[serde(default)]
    pub text: Option<Value>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub max_output_tokens: Option<i64>,
    #[serde(default)]
    pub max_tool_calls: Option<i64>,
    #[serde(default)]
    pub top_logprobs: Option<i64>,
    #[serde(default)]
    pub truncation: Option<String>,
    #[serde(default)]
    pub include: Option<Vec<String>>,
    #[serde(default)]
    pub metadata: Option<Value>,
    #[serde(default)]
    pub prompt_cache_key: Option<String>,
    #[serde(default)]
    pub prompt: Option<Value>,
    #[serde(default)]
    pub context_management: Option<Value>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

/// Fields accepted and ignored: hints that do not change what is generated.
const IGNORED_FIELDS: &[&str] = &[
    "prompt_cache_retention",
    "prompt_cache_options",
    "safety_identifier",
    "user",
    "service_tier",
    "stream_options",
];

/// A capability the request needs, and the parameter that needs it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Requirement {
    pub capability: Capability,
    pub param: String,
}

/// Collected requirements, one entry per capability.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Needs(pub(crate) Vec<Requirement>);

impl Needs {
    pub(crate) fn add(&mut self, capability: Capability, param: &str) {
        if !self.0.iter().any(|need| need.capability == capability) {
            self.0.push(Requirement {
                capability,
                param: param.to_string(),
            });
        }
    }
}

/// A function tool the model may call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionTool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
    pub strict: Option<bool>,
}

/// Normalized tool selection over function tools.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Function(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
}

impl ReasoningEffort {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "none" => Some(Self::None),
            "minimal" => Some(Self::Minimal),
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            "xhigh" => Some(Self::XHigh),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verbosity {
    Low,
    Medium,
    High,
}

impl Verbosity {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

/// Raw request values echoed back on the response object.
#[derive(Debug, Clone, PartialEq)]
pub struct Echo {
    pub tools: Value,
    pub tool_choice: Value,
    pub metadata: Value,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_output_tokens: Option<i64>,
    pub truncation: String,
    pub reasoning_summary: Option<String>,
}

/// A validated, channel-neutral request.
#[derive(Debug, Clone)]
pub struct TurnPlan {
    pub model: Option<String>,
    pub stream: bool,
    /// What the client asked for. The server may still refuse to store when
    /// the channel does not enable `stored_responses`.
    pub store_requested: bool,
    pub previous_response_id: Option<String>,
    pub instructions: Option<String>,
    /// This request's input items, already normalized.
    pub input: Vec<Item>,
    pub tools: Vec<FunctionTool>,
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: bool,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub verbosity: Option<Verbosity>,
    pub prompt_cache_key: Option<String>,
    pub echo: Echo,
    pub requirements: Vec<Requirement>,
}

impl TurnPlan {
    /// Validate a request body and plan the turn.
    pub fn from_request(request: CreateRequest) -> Result<Self, RequestError> {
        let mut needs = Needs::default();
        needs.add(Capability::Text, "input");

        for (key, value) in &request.extra {
            if IGNORED_FIELDS.contains(&key.as_str()) || value.is_null() {
                continue;
            }
            return Err(RequestError::unsupported(
                key,
                &format!("parameter '{key}' is not supported"),
            ));
        }

        let stream = request.stream.unwrap_or(false);
        if stream {
            needs.add(Capability::Streaming, "stream");
        } else {
            needs.add(Capability::NonStreaming, "stream");
        }
        if request.background == Some(true) {
            needs.add(Capability::Background, "background");
        }
        if request
            .conversation
            .as_ref()
            .is_some_and(|value| !value.is_null())
        {
            if request.previous_response_id.is_some() {
                return Err(RequestError::invalid(
                    "previous_response_id",
                    "previous_response_id cannot be used together with conversation",
                ));
            }
            needs.add(Capability::Conversations, "conversation");
        }
        if request
            .prompt
            .as_ref()
            .is_some_and(|value| !value.is_null())
        {
            needs.add(Capability::PromptTemplates, "prompt");
        }
        if request
            .context_management
            .as_ref()
            .is_some_and(|value| !value.is_null())
        {
            needs.add(Capability::Compaction, "context_management");
        }
        let truncation = match request.truncation.as_deref() {
            None | Some("disabled") => "disabled",
            Some("auto") => {
                needs.add(Capability::AutoTruncation, "truncation");
                "auto"
            }
            Some(_) => {
                return Err(RequestError::invalid(
                    "truncation",
                    "truncation must be 'auto' or 'disabled'",
                ))
            }
        };

        if let Some(temperature) = request.temperature {
            if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
                return Err(RequestError::invalid(
                    "temperature",
                    "temperature must be between 0 and 2",
                ));
            }
            if temperature != 1.0 {
                needs.add(Capability::SamplingControls, "temperature");
            }
        }
        if let Some(top_p) = request.top_p {
            if !top_p.is_finite() || !(0.0..=1.0).contains(&top_p) {
                return Err(RequestError::invalid(
                    "top_p",
                    "top_p must be between 0 and 1",
                ));
            }
            if top_p != 1.0 {
                needs.add(Capability::SamplingControls, "top_p");
            }
        }
        if let Some(limit) = request.max_output_tokens {
            if limit < 1 {
                return Err(RequestError::invalid(
                    "max_output_tokens",
                    "max_output_tokens must be a positive integer",
                ));
            }
            needs.add(Capability::MaxOutputTokens, "max_output_tokens");
        }
        if request.max_tool_calls.is_some_and(|limit| limit < 1) {
            return Err(RequestError::invalid(
                "max_tool_calls",
                "max_tool_calls must be a positive integer",
            ));
        }
        if let Some(count) = request.top_logprobs {
            if !(0..=20).contains(&count) {
                return Err(RequestError::invalid(
                    "top_logprobs",
                    "top_logprobs must be between 0 and 20",
                ));
            }
            if count > 0 {
                needs.add(Capability::Logprobs, "top_logprobs");
            }
        }
        if request
            .include
            .iter()
            .flatten()
            .any(|entry| entry == "message.output_text.logprobs")
        {
            needs.add(Capability::Logprobs, "include");
        }

        let instructions = match request.instructions {
            None | Some(Value::Null) => None,
            Some(Value::String(text)) => Some(text).filter(|text| !text.is_empty()),
            Some(_) => {
                return Err(RequestError::unsupported(
                    "instructions",
                    "instructions must be a string",
                ))
            }
        };

        let verbosity = parse_text(request.text.as_ref(), &mut needs)?;
        let (reasoning_effort, reasoning_summary) =
            parse_reasoning(request.reasoning.as_ref(), &mut needs)?;

        let raw_tools = request.tools.clone().unwrap_or_default();
        let mut tools: Vec<FunctionTool> = Vec::new();
        let mut hosted_tools = false;
        for (index, tool) in raw_tools.iter().enumerate() {
            let at = format!("tools[{index}]");
            let kind = tool.get("type").and_then(Value::as_str).ok_or_else(|| {
                RequestError::invalid(&format!("{at}.type"), "tools require a type")
            })?;
            let hosted = match kind {
                "function" => {
                    let parsed = parse_function_tool(tool, &at)?;
                    if tools.iter().any(|existing| existing.name == parsed.name) {
                        return Err(RequestError::invalid(
                            &format!("{at}.name"),
                            "function tool names must be unique",
                        ));
                    }
                    tools.push(parsed);
                    None
                }
                "custom" => Some(Capability::CustomTools),
                "web_search"
                | "web_search_preview"
                | "web_search_2025_08_26"
                | "web_search_preview_2025_03_11" => Some(Capability::WebSearch),
                "file_search" => Some(Capability::FileSearch),
                "code_interpreter" => Some(Capability::CodeInterpreter),
                "mcp" => Some(Capability::Mcp),
                "image_generation" => Some(Capability::ImageGeneration),
                "computer" | "computer_use_preview" => Some(Capability::ComputerUse),
                "shell" | "local_shell" | "apply_patch" => Some(Capability::Shell),
                other => {
                    return Err(RequestError::unsupported(
                        &format!("{at}.type"),
                        &format!("tool type '{other}' is not supported"),
                    ))
                }
            };
            if let Some(capability) = hosted {
                hosted_tools = true;
                needs.add(capability, &at);
            }
        }
        if !tools.is_empty() {
            needs.add(Capability::FunctionTools, "tools");
        }

        let (tool_choice, tools) =
            parse_tool_choice(request.tool_choice.as_ref(), tools, hosted_tools)?;
        if matches!(tool_choice, ToolChoice::Required | ToolChoice::Function(_)) && tools.is_empty()
        {
            if !hosted_tools {
                return Err(RequestError::invalid(
                    "tool_choice",
                    "tool_choice requires at least one function tool",
                ));
            }
        } else if !tools.is_empty() && !matches!(tool_choice, ToolChoice::Auto) {
            needs.add(Capability::ToolChoice, "tool_choice");
        }

        let input = match request.input.as_ref() {
            None | Some(Value::Null) => {
                return Err(RequestError::invalid("input", "input is required"))
            }
            Some(input) => normalize_input(input, &mut needs)?,
        };

        let previous_response_id = match request.previous_response_id {
            None => None,
            Some(id) if id.is_empty() || id.len() > 256 => {
                return Err(RequestError::invalid(
                    "previous_response_id",
                    "previous_response_id must be 1 to 256 characters",
                ))
            }
            Some(id) => {
                needs.add(Capability::PreviousResponse, "previous_response_id");
                Some(id)
            }
        };

        let metadata = request.metadata.unwrap_or(Value::Null);
        validate_metadata(&metadata)?;

        Ok(Self {
            model: request.model.filter(|model| !model.is_empty()),
            stream,
            store_requested: request.store.unwrap_or(true),
            previous_response_id,
            instructions,
            input,
            tools,
            tool_choice,
            parallel_tool_calls: request.parallel_tool_calls.unwrap_or(true),
            reasoning_effort,
            verbosity,
            prompt_cache_key: request.prompt_cache_key.filter(|key| !key.is_empty()),
            echo: Echo {
                tools: Value::Array(raw_tools),
                tool_choice: request
                    .tool_choice
                    .unwrap_or_else(|| Value::String("auto".into())),
                metadata,
                temperature: request.temperature,
                top_p: request.top_p,
                max_output_tokens: request.max_output_tokens,
                truncation: truncation.to_string(),
                reasoning_summary,
            },
            requirements: needs.0,
        })
    }

    /// The first requirement the channel cannot serve, if any.
    pub fn unmet<'a>(&'a self, capabilities: &CapabilitySet) -> Option<&'a Requirement> {
        self.requirements
            .iter()
            .find(|need| !capabilities.enabled(need.capability))
    }
}

fn parse_text(text: Option<&Value>, needs: &mut Needs) -> Result<Option<Verbosity>, RequestError> {
    let object = match text {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Object(object)) => object,
        Some(_) => return Err(RequestError::invalid("text", "text must be an object")),
    };
    let mut verbosity = None;
    for (key, value) in object {
        match key.as_str() {
            "format" => {
                if value.is_null() {
                    continue;
                }
                match value.get("type").and_then(Value::as_str) {
                    Some("text") => {}
                    Some("json_object") | Some("json_schema") => {
                        needs.add(Capability::StructuredOutput, "text.format")
                    }
                    _ => {
                        return Err(RequestError::invalid(
                            "text.format",
                            "text.format.type must be text, json_object or json_schema",
                        ))
                    }
                }
            }
            "verbosity" => {
                if value.is_null() {
                    continue;
                }
                let parsed = value.as_str().and_then(Verbosity::parse).ok_or_else(|| {
                    RequestError::invalid("text.verbosity", "verbosity must be low, medium or high")
                })?;
                if parsed != Verbosity::Medium {
                    needs.add(Capability::Verbosity, "text.verbosity");
                    verbosity = Some(parsed);
                }
            }
            other => {
                return Err(RequestError::unsupported(
                    &format!("text.{other}"),
                    &format!("text.{other} is not supported"),
                ))
            }
        }
    }
    Ok(verbosity)
}

fn parse_reasoning(
    reasoning: Option<&Value>,
    needs: &mut Needs,
) -> Result<(Option<ReasoningEffort>, Option<String>), RequestError> {
    let object = match reasoning {
        None | Some(Value::Null) => return Ok((None, None)),
        Some(Value::Object(object)) => object,
        Some(_) => {
            return Err(RequestError::invalid(
                "reasoning",
                "reasoning must be an object",
            ))
        }
    };
    let mut effort = None;
    let mut summary = None;
    for (key, value) in object {
        match key.as_str() {
            "effort" => {
                if value.is_null() {
                    continue;
                }
                let parsed = value
                    .as_str()
                    .and_then(ReasoningEffort::parse)
                    .ok_or_else(|| {
                        RequestError::invalid(
                            "reasoning.effort",
                            "effort must be none, minimal, low, medium, high or xhigh",
                        )
                    })?;
                if !matches!(parsed, ReasoningEffort::None | ReasoningEffort::Minimal) {
                    needs.add(Capability::Reasoning, "reasoning.effort");
                }
                effort = Some(parsed);
            }
            // Summaries are best effort in the Responses API. A channel that
            // cannot summarize returns an empty `summary` list.
            "summary" | "generate_summary" => {
                if value.is_null() {
                    continue;
                }
                match value.as_str() {
                    Some(mode @ ("auto" | "concise" | "detailed")) => {
                        summary = Some(mode.to_string())
                    }
                    _ => {
                        return Err(RequestError::invalid(
                            &format!("reasoning.{key}"),
                            "summary must be auto, concise or detailed",
                        ))
                    }
                }
            }
            other => {
                return Err(RequestError::unsupported(
                    &format!("reasoning.{other}"),
                    &format!("reasoning.{other} is not supported"),
                ))
            }
        }
    }
    Ok((effort, summary))
}

fn parse_function_tool(tool: &Value, at: &str) -> Result<FunctionTool, RequestError> {
    let name = tool
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| {
            RequestError::invalid(&format!("{at}.name"), "function tools require a name")
        })?;
    if name.chars().count() > 64
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err(RequestError::invalid(
            &format!("{at}.name"),
            "function names must be 1-64 characters of a-z, A-Z, 0-9, '_' or '-'",
        ));
    }
    let parameters = match tool.get("parameters") {
        None | Some(Value::Null) => serde_json::json!({"type": "object", "properties": {}}),
        Some(Value::Object(object)) => Value::Object(object.clone()),
        Some(_) => {
            return Err(RequestError::invalid(
                &format!("{at}.parameters"),
                "parameters must be a JSON Schema object",
            ))
        }
    };
    let strict = match tool.get("strict") {
        None | Some(Value::Null) => None,
        Some(Value::Bool(strict)) => Some(*strict),
        Some(_) => {
            return Err(RequestError::invalid(
                &format!("{at}.strict"),
                "strict must be a boolean",
            ))
        }
    };
    Ok(FunctionTool {
        name: name.to_string(),
        description: tool
            .get("description")
            .and_then(Value::as_str)
            .filter(|description| !description.is_empty())
            .map(str::to_string),
        parameters,
        strict,
    })
}

fn parse_tool_choice(
    choice: Option<&Value>,
    tools: Vec<FunctionTool>,
    hosted_tools: bool,
) -> Result<(ToolChoice, Vec<FunctionTool>), RequestError> {
    let object = match choice {
        None | Some(Value::Null) => return Ok((ToolChoice::Auto, tools)),
        Some(Value::String(mode)) => {
            let choice = match mode.as_str() {
                "auto" => ToolChoice::Auto,
                "none" => ToolChoice::None,
                "required" => ToolChoice::Required,
                _ => {
                    return Err(RequestError::invalid(
                        "tool_choice",
                        "tool_choice must be auto, none, required or a tool choice object",
                    ))
                }
            };
            return Ok((choice, tools));
        }
        Some(Value::Object(object)) => object,
        Some(_) => {
            return Err(RequestError::invalid(
                "tool_choice",
                "tool_choice must be a string or an object",
            ))
        }
    };
    match object.get("type").and_then(Value::as_str) {
        Some("function") => {
            let name = object
                .get("name")
                .and_then(Value::as_str)
                .or_else(|| {
                    object
                        .get("function")
                        .and_then(|function| function.get("name"))
                        .and_then(Value::as_str)
                })
                .filter(|name| !name.is_empty())
                .ok_or_else(|| {
                    RequestError::invalid("tool_choice.name", "a function tool_choice needs a name")
                })?;
            if !tools.iter().any(|tool| tool.name == name) {
                return Err(RequestError::invalid(
                    "tool_choice.name",
                    "tool_choice names a function that is not in tools",
                ));
            }
            Ok((ToolChoice::Function(name.to_string()), tools))
        }
        Some("allowed_tools") => {
            let required = match object.get("mode").and_then(Value::as_str) {
                None | Some("auto") => false,
                Some("required") => true,
                Some(_) => {
                    return Err(RequestError::invalid(
                        "tool_choice.mode",
                        "allowed_tools mode must be auto or required",
                    ))
                }
            };
            let allowed = object
                .get("tools")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    RequestError::invalid("tool_choice.tools", "allowed_tools needs a tools list")
                })?;
            let mut names = Vec::new();
            for entry in allowed {
                match entry.get("type").and_then(Value::as_str) {
                    Some("function") => {
                        if let Some(name) = entry.get("name").and_then(Value::as_str) {
                            names.push(name.to_string());
                        }
                    }
                    _ => {
                        return Err(RequestError::unsupported(
                            "tool_choice.tools",
                            "allowed_tools may only list function tools",
                        ))
                    }
                }
            }
            if hosted_tools {
                return Err(RequestError::unsupported(
                    "tool_choice",
                    "allowed_tools cannot be combined with hosted tools",
                ));
            }
            let narrowed: Vec<FunctionTool> = tools
                .into_iter()
                .filter(|tool| names.contains(&tool.name))
                .collect();
            if narrowed.is_empty() {
                return Err(RequestError::invalid(
                    "tool_choice.tools",
                    "allowed_tools does not match any function in tools",
                ));
            }
            let choice = if required {
                ToolChoice::Required
            } else {
                ToolChoice::Auto
            };
            Ok((choice, narrowed))
        }
        Some(other) => Err(RequestError::unsupported(
            "tool_choice",
            &format!("tool_choice type '{other}' is not supported"),
        )),
        None => Err(RequestError::invalid(
            "tool_choice",
            "tool_choice objects need a type",
        )),
    }
}

fn validate_metadata(metadata: &Value) -> Result<(), RequestError> {
    let object = match metadata {
        Value::Null => return Ok(()),
        Value::Object(object) => object,
        _ => {
            return Err(RequestError::invalid(
                "metadata",
                "metadata must be an object",
            ))
        }
    };
    if object.len() > 16 {
        return Err(RequestError::invalid(
            "metadata",
            "metadata may contain at most 16 keys",
        ));
    }
    for (key, value) in object {
        let valid_value = value
            .as_str()
            .is_some_and(|text| text.chars().count() <= 512);
        if key.chars().count() > 64 || !valid_value {
            return Err(RequestError::invalid(
                "metadata",
                "metadata keys must be at most 64 characters and values strings of at most 512",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::{CapabilityStatus, Provider};
    use serde_json::json;

    fn plan(body: Value) -> Result<TurnPlan, RequestError> {
        TurnPlan::from_request(serde_json::from_value(body).unwrap())
    }

    fn needed(plan: &TurnPlan) -> Vec<Capability> {
        plan.requirements
            .iter()
            .map(|need| need.capability)
            .collect()
    }

    #[test]
    fn minimal_request_needs_text_and_its_transport() {
        let streaming = plan(json!({"input": "hi", "stream": true})).unwrap();
        assert_eq!(
            needed(&streaming),
            vec![Capability::Text, Capability::Streaming]
        );
        assert!(streaming.store_requested);
        let collected = plan(json!({"input": "hi"})).unwrap();
        assert_eq!(
            needed(&collected),
            vec![Capability::Text, Capability::NonStreaming]
        );
    }

    #[test]
    fn behaviour_changing_parameters_become_requirements() {
        let p = plan(json!({
            "input": "hi",
            "temperature": 0.2,
            "max_output_tokens": 10,
            "text": {"format": {"type": "json_schema"}, "verbosity": "low"},
            "reasoning": {"effort": "high", "summary": "auto"},
            "truncation": "auto",
            "background": true,
            "previous_response_id": "resp_1"
        }))
        .unwrap();
        let caps = needed(&p);
        for capability in [
            Capability::SamplingControls,
            Capability::MaxOutputTokens,
            Capability::StructuredOutput,
            Capability::Verbosity,
            Capability::Reasoning,
            Capability::AutoTruncation,
            Capability::Background,
            Capability::PreviousResponse,
        ] {
            assert!(
                caps.contains(&capability),
                "{capability:?} must be required"
            );
        }
        assert_eq!(p.verbosity, Some(Verbosity::Low));
        assert_eq!(p.reasoning_effort, Some(ReasoningEffort::High));
        assert_eq!(p.echo.reasoning_summary.as_deref(), Some("auto"));
    }

    #[test]
    fn defaults_and_hints_do_not_add_requirements() {
        let p = plan(json!({
            "input": "hi",
            "temperature": 1.0,
            "top_p": 1,
            "text": {"verbosity": "medium", "format": {"type": "text"}},
            "reasoning": {"effort": "none"},
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": "k",
            "safety_identifier": "u",
            "service_tier": "auto",
            "stream_options": {"include_obfuscation": false}
        }))
        .unwrap();
        assert_eq!(needed(&p), vec![Capability::Text, Capability::NonStreaming]);
        assert_eq!(p.reasoning_effort, Some(ReasoningEffort::None));
    }

    #[test]
    fn unknown_parameters_are_rejected_not_ignored() {
        let error = plan(json!({"input": "hi", "moderation": {"policy": "x"}})).unwrap_err();
        assert_eq!(error.code, "unsupported_parameter");
        assert_eq!(error.param.as_deref(), Some("moderation"));
    }

    #[test]
    fn hosted_tools_need_their_capability() {
        let p = plan(json!({"input": "hi", "tools": [
            {"type": "web_search"}, {"type": "mcp", "server_label": "x"}
        ]}))
        .unwrap();
        let caps = needed(&p);
        assert!(caps.contains(&Capability::WebSearch));
        assert!(caps.contains(&Capability::Mcp));
        assert!(plan(json!({"input": "hi", "tools": [{"type": "teleport"}]})).is_err());
    }

    #[test]
    fn function_tools_and_tool_choice_are_validated() {
        let tools = json!([
            {"type": "function", "name": "lookup", "parameters": {"type": "object"}, "strict": true},
            {"type": "function", "name": "other"}
        ]);
        let named = plan(json!({"input": "hi", "tools": tools,
            "tool_choice": {"type": "function", "name": "lookup"}}))
        .unwrap();
        assert_eq!(named.tool_choice, ToolChoice::Function("lookup".into()));
        assert!(needed(&named).contains(&Capability::ToolChoice));
        assert_eq!(named.tools[0].strict, Some(true));

        let allowed = plan(json!({"input": "hi", "tools": tools, "tool_choice": {
            "type": "allowed_tools", "mode": "required",
            "tools": [{"type": "function", "name": "other"}]
        }}))
        .unwrap();
        assert_eq!(allowed.tool_choice, ToolChoice::Required);
        assert_eq!(allowed.tools.len(), 1);
        assert_eq!(allowed.tools[0].name, "other");

        for bad in [
            json!({"input": "hi", "tool_choice": "required"}),
            json!({"input": "hi", "tools": tools, "tool_choice": {"type": "function", "name": "missing"}}),
            json!({"input": "hi", "tools": [{"type": "function", "name": "bad name"}]}),
            json!({"input": "hi", "tools": [
                {"type": "function", "name": "dup"}, {"type": "function", "name": "dup"}
            ]}),
            json!({"input": "hi", "tools": [{"type": "function", "name": "f", "parameters": []}]}),
        ] {
            assert!(plan(bad.clone()).is_err(), "{bad} must be rejected");
        }
    }

    #[test]
    fn invalid_values_are_rejected() {
        for bad in [
            json!({}),
            json!({"input": "hi", "temperature": 3}),
            json!({"input": "hi", "top_logprobs": 21}),
            json!({"input": "hi", "truncation": "middle"}),
            json!({"input": "hi", "instructions": [{"role": "system"}]}),
            json!({"input": "hi", "reasoning": {"effort": "huge"}}),
            json!({"input": "hi", "previous_response_id": ""}),
            json!({"input": "hi", "metadata": {"k": 1}}),
            json!({"input": "hi", "conversation": "c", "previous_response_id": "r"}),
        ] {
            assert!(plan(bad.clone()).is_err(), "{bad} must be rejected");
        }
    }

    #[test]
    fn unmet_reports_the_first_missing_capability() {
        let channel = CapabilitySet::new("c")
            .with(
                Capability::Text,
                CapabilityStatus::supported(Provider::Upstream, "native"),
            )
            .with(
                Capability::NonStreaming,
                CapabilityStatus::supported(Provider::Proxy, "collected"),
            );
        assert!(plan(json!({"input": "hi"}))
            .unwrap()
            .unmet(&channel)
            .is_none());
        let streaming = plan(json!({"input": "hi", "stream": true})).unwrap();
        let unmet = streaming.unmet(&channel).unwrap();
        assert_eq!(unmet.capability, Capability::Streaming);
        assert_eq!(unmet.param, "stream");
    }
}
