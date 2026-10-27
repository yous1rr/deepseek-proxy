//! Tool-call wire format: teaching text and parsing.
//!
//! Port of the parts of `deepseek_proxy/tools/format.py` that matter for a
//! working proxy: the `json_invoke` dialect (what pi speaks), prompt
//! injection, and extraction of `<invoke name="t">{...}</invoke>` blocks from
//! model output.
//!
//! Not ported: the repair/healing machinery for malformed blocks, and the
//! `xml_params` / legacy `dsml` dialects. Only the json_invoke spelling is
//! parsed; anything else falls through as plain text.

/// A parsed tool call.
///
/// `arguments` keeps the **raw** JSON text the model emitted, exactly like
/// Python's `function.arguments` string. The recovery policy needs the raw
/// text for two reasons: a repair prompt must report `invalid_arguments_json`
/// for a body that is not valid JSON, and the echo-detection fingerprint
/// canonicalises the parsed value rather than the source bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub name: String,
    /// Raw JSON text between the `<invoke>` tags, untrimmed except for the
    /// surrounding whitespace the parser strips.
    pub arguments: String,
}

impl ToolCall {
    /// Parsed arguments, or `None` when the body is not valid JSON.
    pub fn json(&self) -> Option<serde_json::Value> {
        serde_json::from_str(&self.arguments).ok()
    }
}

/// `(name, canonical json)` — the port of Python's `fingerprint()`.
///
/// `serde_json::Map` is a `BTreeMap` (the workspace does not enable
/// `preserve_order`), so re-serialising a parsed value already sorts keys,
/// matching `json.dumps(..., sort_keys=True)`.
pub fn fingerprint(call: &ToolCall) -> (String, String) {
    let canonical = match call.json() {
        Some(v) => serde_json::to_string(&v).unwrap_or_else(|_| call.arguments.clone()),
        None => call.arguments.clone(),
    };
    (call.name.clone(), canonical)
}

pub const INVOKE_OPEN: &str = "<invoke";
pub const INVOKE_CLOSE: &str = "</invoke>";

/// Concrete example shown to the model.
pub const TEMPLATE: &str =
    "<invoke name=\"tool_name\">{\"param1\": \"value1\", \"count\": 5}</invoke>";

const PROTOCOL_HEADER: &str = "## TOOL CALL PROTOCOL — MANDATORY, STRICT FORMAT";
const TOOLS_HEADER: &str = "## Available Tools";

fn tool_section(name: &str, description: &str, params: &serde_json::Value) -> String {
    let params = serde_json::to_string_pretty(params).unwrap_or_else(|_| "{}".into());
    format!(
        "### {name}\nDescription: {description}\nParameters (JSON Schema):\n```json\n{params}\n```"
    )
}

fn instruction() -> String {
    let mut s = String::new();
    s.push_str("To call a tool, emit a block in EXACTLY this format:\n\n");
    s.push_str(TEMPLATE);
    s.push_str("\n\nThe tool name goes in the name attribute of the opening tag. Between the\n");
    s.push_str("tags put ONE JSON object holding all arguments. Each property in a tool's\n");
    s.push_str("Parameters (JSON Schema) below corresponds to one key of that object.\n\n");
    s.push_str("### Required tool-call rules\n");
    s.push_str("1. ATTRIBUTES: the opening tag MUST carry a name attribute:\n");
    s.push_str("   <invoke name=\"tool_name\">. Extra attributes are ignored, but name\n");
    s.push_str("   is required.\n");
    s.push_str("2. BODY: exactly one JSON object between the tags. Strings, numbers,\n");
    s.push_str("   booleans, null, arrays and nested objects are all fine as JSON values.\n");
    s.push_str("   Do NOT wrap the block in code fences, never emit bare JSON with no tags.\n");
    s.push_str("3. LITERAL MARKERS: if a value must contain the literal text <invoke\n");
    s.push_str("   or </invoke>, write a backslash directly before it.\n");
    s.push_str("4. MULTIPLE CALLS: you may emit several complete blocks back to back IN ONE\n");
    s.push_str("   TURN ONLY for short, independent, read-only calls (e.g. reading files,\n");
    s.push_str("   listing directories, grepping, small quick commands). NEVER batch writes,\n");
    s.push_str("   edits, long-running commands, or complex/stateful operations — emit exactly\n");
    s.push_str("   ONE such call per turn and wait for its result before the next.\n");
    s.push_str("5. You may write normal text before or after tool call blocks.\n");
    s.push_str("6. CALL ONLY WHEN NEEDED: if no tool fits, answer directly with no block.\n");
    s.push_str("   After a tool result arrives, continue the task — do not re-emit the\n");
    s.push_str("   same call.\n");
    s.push_str("7. NO GUESSING: use only parameters from the schema below; never invent values.");
    s
}

fn final_reminder() -> String {
    let mut s = String::new();
    s.push_str("FINAL REMINDER: tool calls use exactly this form:\n\n");
    s.push_str(TEMPLATE);
    s.push_str("\n\nname in the <invoke attribute; one JSON object between the tags.\n");
    s.push_str("Emit each tool call exactly once. Never repeat an opening tag, closing tag,\n");
    s.push_str("or the complete tool-call block after it has been emitted.");
    s
}

/// Per-turn re-anchoring reminder, appended to the END of every turn's prompt.
///
/// The system prompt that teaches the format is only sent on the first turn of
/// a thread, so long threads drift and start emitting malformed blocks. This
/// restates the wire shape and the concrete failure modes (legacy marker tags,
/// a doubled closing tag) right before the model answers.
pub fn periodic_reminder() -> String {
    let mut s = String::new();
    s.push_str("⚠ HARD FORMAT REQUIREMENT — READ BEFORE EMITTING ANY TOOL CALL.\n");
    s.push_str("Violations are NOT repaired for you: a malformed or repeated block is DROPPED\n");
    s.push_str("and never executed, and you will burn this turn on a correction loop instead\n");
    s.push_str("of progressing the task.\n");
    s.push_str("Reminder — tool calls use exactly this form:\n");
    s.push_str(TEMPLATE);
    s.push_str("\n!!IMPORTANT!!\n");
    s.push_str(
        "1. Never emit </｜DSML｜｜ invoke> or related tags those are useless and will be shown to the user.\n",
    );
    s.push_str("2. Never emit the </invoke> ending tag twice.\n");
    s.push_str("Get the format wrong and your tool call silently does nothing.");
    s
}

/// Append the tool protocol + tool definitions to a system prompt.
///
/// Returns `system_prompt` unchanged when `tools` is empty, matching Python.
pub fn inject_tool_descriptions(system_prompt: &str, tools: &[serde_json::Value]) -> String {
    if tools.is_empty() {
        return system_prompt.to_string();
    }

    let mut sections = Vec::new();
    for tool in tools {
        if tool.get("type").and_then(|t| t.as_str()) != Some("function") {
            continue;
        }
        let Some(func) = tool.get("function") else {
            continue;
        };
        let Some(name) = func.get("name").and_then(|n| n.as_str()) else {
            continue;
        };
        let params = func
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        let description = func
            .get("description")
            .and_then(|d| d.as_str())
            .unwrap_or("No description provided.");
        sections.push(tool_section(name, description, &params));
    }

    let tools_block = sections.join("\n\n");
    format!(
        "{system_prompt}\n\n---\n{PROTOCOL_HEADER}\n\n{}\n\n{TOOLS_HEADER}\n\n{tools_block}\n---\n{}",
        instruction(),
        final_reminder()
    )
}
/// Extract `<invoke name="t">{json}</invoke>` blocks from model text.
///
/// Returns calls in order. A block whose body is not a JSON object is skipped
/// rather than mis-parsed — better to lose one call than send garbage args.
pub fn extract_tool_calls(text: &str) -> Vec<ToolCall> {
    let normalized = normalize_dsml(text);
    let mut out = Vec::new();
    let mut rest = normalized.as_str();

    while let Some(start) = rest.find(INVOKE_OPEN) {
        let after_open = &rest[start + INVOKE_OPEN.len()..];

        // Require the next char to be whitespace, so "<invokex" cannot match.
        let Some(first) = after_open.chars().next() else {
            break;
        };
        if !first.is_whitespace() {
            rest = after_open;
            continue;
        }

        let Some(tag_end) = after_open.find('>') else {
            break;
        };
        let attrs = &after_open[..tag_end];

        let body_and_rest = &after_open[tag_end + 1..];
        let Some(close_rel) = body_and_rest.find(INVOKE_CLOSE) else {
            break;
        };
        let body = body_and_rest[..close_rel].trim();

        if let Some(name) = attr_value(attrs, "name") {
            // A block is a call only when its body parses as a JSON object;
            // anything else is a malformed event, exactly like Python's
            // parser (which json.loads-es the body before emitting `tool`).
            if matches!(
                serde_json::from_str::<serde_json::Value>(body),
                Ok(serde_json::Value::Object(_))
            ) {
                out.push(ToolCall {
                    name,
                    arguments: body.to_string(),
                });
            }
        }

        rest = &body_and_rest[close_rel + INVOKE_CLOSE.len()..];
    }

    out
}

/// Rewrite legacy DSML spellings of the invoke tag to the plain `<invoke>`
/// form, and drop DSML wrapper tags. Attributes are preserved on the invoke
/// tag; wrapper tags carry no data.
///
/// Models occasionally fall back to the old marker spelling (single or double
/// fullwidth pipes, optional spaces). Parsing stays universal on purpose, so an
/// old transcript or a drifting model still works no matter what we teach.
pub fn normalize_dsml(text: &str) -> String {
    if !text.contains("DSML") && !text.contains("dsml") {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while let Some(rel) = text[i..].find('<') {
        let start = i + rel;
        let Some(gt_rel) = text[start..].find('>') else {
            break;
        };
        let gt = start + gt_rel;
        let inner = &text[start + 1..gt];

        // Compact form: drop fullwidth pipes and whitespace so every marker
        // spelling collapses to the same shape (`dsmlinvokename=...`).
        let compact: String = inner
            .chars()
            .filter(|c| *c != '｜' && !c.is_whitespace())
            .collect::<String>()
            .to_ascii_lowercase();
        let rest = compact.strip_prefix("dsml");
        let rest = rest.or_else(|| compact.strip_prefix("/dsml"));
        let Some(rest) = rest else {
            out.push_str(&text[i..gt + 1]);
            i = gt + 1;
            continue;
        };

        out.push_str(&text[i..start]);
        let closing = compact.starts_with('/');
        if rest.starts_with("invoke") {
            if closing {
                out.push_str("</invoke>");
            } else {
                out.push_str("<invoke");
                if let Some(name) = attr_value(inner, "name") {
                    out.push_str(&format!(" name=\"{name}\""));
                }
                out.push('>');
            }
        }
        // Wrapper tags (`tool_calls`, `function_calls`, `calls`) are dropped.
        i = gt + 1;
    }
    out.push_str(&text[i..]);
    out
}

fn attr_value(attrs: &str, key: &str) -> Option<String> {
    let needle = format!("{key}=");
    let idx = attrs.find(&needle)?;
    let after = &attrs[idx + needle.len()..];
    let quote = after.chars().next()?;
    if quote != '"' && quote != '\'' {
        return None;
    }
    let inner = &after[1..];
    let end = inner.find(quote)?;
    Some(inner[..end].to_string())
}

/// Remove tool-call blocks, leaving the prose around them.
#[allow(dead_code)] // kept for callers that render a single answer string
pub fn strip_tool_calls(text: &str) -> String {
    let normalized = normalize_dsml(text);
    let mut out = String::new();
    let mut rest = normalized.as_str();
    while let Some(start) = rest.find(INVOKE_OPEN) {
        out.push_str(&rest[..start]);
        let after = &rest[start..];
        match after.find(INVOKE_CLOSE) {
            Some(close) => rest = &after[close + INVOKE_CLOSE.len()..],
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out
}

/// Re-serialize OpenAI `tool_calls` into the wire format the model is taught.
///
/// History is flattened as text, so without this a replayed assistant turn
/// loses the arguments it actually sent.
pub fn format_tool_calls_for_history(calls: &[serde_json::Value]) -> String {
    let mut blocks = Vec::new();
    for call in calls {
        let Some(func) = call.get("function") else {
            continue;
        };
        let Some(name) = func.get("name").and_then(|n| n.as_str()) else {
            continue;
        };
        if name.is_empty() {
            continue;
        }
        let body = match func.get("arguments") {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(v) => v.to_string(),
            None => "{}".to_string(),
        };
        // History replay must round-trip the exact bytes the model sent, so
        // the body is inserted verbatim rather than re-serialised.
        blocks.push(format!("<invoke name=\"{name}\">{body}</invoke>"));
    }
    blocks.join("\n")
}

/// One `<invoke>` block that could not be turned into a call.
///
/// Port of the records `extract_malformed()` yields in
/// `deepseek_proxy/tools/format.py`: `name` is `None` when the block never named
/// a tool (an orphan parameter block), and `reason` is a stable category, never
/// a dump of the body — a repair prompt must not replay the bad call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MalformedCall {
    pub name: Option<String>,
    pub body: String,
    pub reason: String,
}

impl MalformedCall {
    pub fn new(name: Option<String>, body: String, reason: &str) -> Self {
        Self {
            name,
            body,
            reason: reason.to_string(),
        }
    }
}

/// Reasonable ceiling on how much of a bad body is kept for diagnostics.
const MAX_MALFORMED_BODY: usize = 1500;

fn truncate_body(s: &str) -> String {
    let t = s.trim();
    if t.len() <= MAX_MALFORMED_BODY {
        t.to_string()
    } else {
        let mut end = MAX_MALFORMED_BODY;
        while !t.is_char_boundary(end) {
            end -= 1;
        }
        t[..end].to_string()
    }
}

/// Why a block failed, in the same categories the Python detector reports.
fn classify_body(name: Option<&str>, body: &str) -> Option<&'static str> {
    let body = body.trim();
    if body.is_empty() {
        return Some(if name.is_none() {
            "missing_name_and_empty_body"
        } else {
            "invalid_json_body"
        });
    }
    match serde_json::from_str::<serde_json::Value>(body) {
        Ok(serde_json::Value::Object(_)) => None,
        Ok(other) => {
            let _ = other;
            Some("body_not_object")
        }
        Err(_) => Some("invalid_json_body"),
    }
}

/// Records for `<invoke>` blocks that fail to parse, plus orphan parameter
/// blocks and an unterminated trailing block.
///
/// Kept as the tested reference detector; the live path uses [`scan`], which
/// reports the same failures in model-output order.
#[allow(dead_code)]
pub fn extract_malformed(text: &str) -> Vec<MalformedCall> {
    let normalized = normalize_dsml(text);
    let mut out: Vec<MalformedCall> = Vec::new();
    let mut seen: Vec<(Option<String>, String)> = Vec::new();
    let mut covered: Vec<(usize, usize)> = Vec::new();

    let mut push = |name: Option<String>, body: String, reason: &str, out: &mut Vec<_>| {
        let key = (name.clone(), body.clone());
        if seen.contains(&key) {
            return;
        }
        seen.push(key);
        out.push(MalformedCall::new(name, truncate_body(&body), reason));
    };

    let mut rest = normalized.as_str();
    let mut base = 0usize;
    while let Some(start) = rest.find(INVOKE_OPEN) {
        let after_open = &rest[start + INVOKE_OPEN.len()..];
        let Some(first) = after_open.chars().next() else {
            break;
        };
        if !first.is_whitespace() {
            base += start + INVOKE_OPEN.len();
            rest = after_open;
            continue;
        }
        let Some(tag_end) = after_open.find('>') else {
            break;
        };
        let attrs = &after_open[..tag_end];
        let name = attr_value(attrs, "name").filter(|n| !n.is_empty());
        let body_and_rest = &after_open[tag_end + 1..];

        let span_start = base + start;
        match body_and_rest.find(INVOKE_CLOSE) {
            Some(close_rel) => {
                let body = &body_and_rest[..close_rel];
                if let Some(reason) = classify_body(name.as_deref(), body) {
                    push(name, body.to_string(), reason, &mut out);
                }
                let consumed =
                    start + INVOKE_OPEN.len() + tag_end + 1 + close_rel + INVOKE_CLOSE.len();
                covered.push((span_start, base + consumed));
                base += consumed;
                rest = &rest[consumed..];
            }
            None => {
                // Output cut by the token limit before the closing tag.
                let body = body_and_rest;
                let reason = classify_body(name.as_deref(), body).unwrap_or("unterminated_block");
                push(name, body.to_string(), reason, &mut out);
                covered.push((span_start, base + rest.len()));
                break;
            }
        }
    }

    // A bare `<parameter name="x">value</parameter>` with no enclosing invoke:
    // the model confused the two levels. It must reach repair, not be treated
    // as ordinary prose.
    for (at, _) in normalized.match_indices("<parameter") {
        if covered.iter().any(|(s, e)| *s <= at && at < *e) {
            continue;
        }
        let Some(open_end) = normalized[at..].find('>') else {
            continue;
        };
        let attrs = &normalized[at..at + open_end];
        let body_start = at + open_end + 1;
        let body_end = normalized[body_start..]
            .find("</parameter>")
            .map(|r| body_start + r)
            .unwrap_or(normalized.len());
        let mut body = format!("{attrs}>");
        body.push_str(&normalized[body_start..body_end]);
        push(None, body, "orphan_parameter_block", &mut out);
    }

    out
}

/// Correction-turn prompt for malformed blocks, without replaying the bad turn.
///
/// Deliberately minimal: name the failures, restate the wire format, and tell
/// the model to follow the protocol it was already given. Per-call schemas for
/// argument failures are appended by [`crate::Recovery::begin_repair`] via
/// [`repair_schemas`]; the full catalogue is never sent.
pub fn build_repair_prompt(malformed: &[MalformedCall], specs: &[ToolSpec]) -> String {
    let mut out = String::from(
        "Your previous response contained malformed tool call(s) that could not be executed. \
         The failed response is already in the conversation history. Do not quote, copy, \
         explain, or repeat it. Follow the tool-call instructions you were given and \
         re-emit EACH corrected call now, and nothing else.",
    );
    for (i, m) in malformed.iter().enumerate() {
        let name = match &m.name {
            Some(n) => format!("\"{n}\""),
            None => "None".to_string(),
        };
        // Strip any detail after the category separator, mirroring Python.
        let reason = m.reason.split(':').next().unwrap_or("malformed_tool_call");
        out.push_str(&format!(
            "\nFailure {}: name={} reason={}",
            i + 1,
            name,
            reason
        ));
    }
    if !specs.is_empty() {
        let names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
        out.push_str(&format!("\nValid tool names: {}.", names.join(", ")));
    }
    out.push_str("\nRe-emit each fixed call in EXACTLY this shape:\n");
    out.push_str(TEMPLATE);
    out.push_str(
        "\nRules: opening tag MUST carry name=\"tool_name\"; body is ONE JSON object with \
         all arguments. Emit corrected blocks back to back, no chatter, no code fences.",
    );
    out.push_str("\nOnly fix the listed failures. Do not repeat calls already accepted.");
    out
}

/// Schemas worth re-sending on a repair turn, one per tool.
///
/// Deliberately narrow. A schema only helps when the model already named a real
/// tool and got its *arguments* wrong — then the property names and required
/// list are exactly what it is missing. For an unknown tool name, a missing
/// required call, or an omitted call there is nothing to key on, and dumping
/// the catalogue would only burn the context the repair needs.
///
/// The name comes from the model, so an unknown name matches no spec and adds
/// nothing. Same tool failing twice contributes one schema.
pub fn repair_schemas(malformed: &[MalformedCall], specs: &[ToolSpec]) -> Vec<serde_json::Value> {
    let mut seen: Vec<&str> = Vec::new();
    let mut out = Vec::new();
    for m in malformed {
        let Some(name) = m.name.as_deref() else {
            continue;
        };
        let category = m.reason.split(':').next().unwrap_or_default();
        // These are the reasons that actually reach a `MalformedCall`: the
        // wire-shape names come from `classify_body`, and the schema name comes
        // from `validation_reason`. (`invalid_arguments_json` and
        // `arguments_must_be_object` are internal to `validation_reason` and are
        // never recorded — matching on them here was a dead branch.)
        if !matches!(
            category,
            "arguments_do_not_match_schema" | "invalid_json_body" | "body_not_object"
        ) {
            continue;
        }
        let Some(spec) = specs.iter().find(|s| s.name == name) else {
            continue;
        };
        if !seen.contains(&name) {
            seen.push(name);
            out.push(spec.raw.clone());
        }
    }
    out
}

/// One tool offered to the model: the name used for lookup plus the raw
/// OpenAI tool object, which is what the repair prompt dumps verbatim.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolSpec {
    pub name: String,
    pub raw: serde_json::Value,
}

/// Tool specs in request order, skipping entries without a usable name.
pub fn tool_specs(tools: &[serde_json::Value]) -> Vec<ToolSpec> {
    let mut out = Vec::new();
    for tool in tools {
        let Some(func) = tool.get("function") else {
            continue;
        };
        let Some(name) = func.get("name").and_then(|n| n.as_str()) else {
            continue;
        };
        out.push(ToolSpec {
            name: name.to_string(),
            raw: tool.clone(),
        });
    }
    out
}

/// Validate one parsed call exactly like `Recovery.accept()` does.
///
/// Returns the stable failure category, or `None` when the call is acceptable.
/// The categories and their order of precedence match Python: an unknown tool
/// is reported before the arguments are even inspected.
pub fn validation_reason(specs: &[ToolSpec], call: &ToolCall) -> Option<&'static str> {
    let Some(spec) = specs.iter().find(|s| s.name == call.name) else {
        return Some("unknown_or_disallowed_tool");
    };
    let Ok(args) = serde_json::from_str::<serde_json::Value>(&call.arguments) else {
        return Some("invalid_arguments_json");
    };
    if !args.is_object() {
        return Some("arguments_must_be_object");
    }
    let parameters = spec
        .raw
        .get("function")
        .and_then(|f| f.get("parameters"))
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    match jsonschema::validator_for(&parameters) {
        Ok(validator) => {
            if validator.is_valid(&args) {
                None
            } else {
                Some("arguments_do_not_match_schema")
            }
        }
        // Python rejects an invalid schema at request-validation time; if one
        // slips through here, treat it as permissive rather than failing the turn.
        Err(_) => None,
    }
}

/// One parsed piece of model output, in the order the model produced it.
///
/// This is the flattened equivalent of the events `ToolParser` yields: text,
/// a valid call, or a block that failed to parse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScanEvent {
    Text(String),
    Tool(ToolCall),
    Malformed(MalformedCall),
}

/// Split model text into ordered text / tool / malformed events.
///
/// Port of feeding the whole reply through `ToolParser`: unlike
/// `extract_tool_calls` + `extract_malformed` run separately, this preserves the
/// interleaving so the recovery loop sees failures in the order they occurred.
pub fn scan(text: &str) -> Vec<ScanEvent> {
    let normalized = normalize_dsml(text);
    let mut out: Vec<ScanEvent> = Vec::new();
    let mut buf = String::new();
    let mut rest = normalized.as_str();

    while let Some(start) = rest.find(INVOKE_OPEN) {
        let after_open = &rest[start + INVOKE_OPEN.len()..];
        let Some(first) = after_open.chars().next() else {
            break;
        };
        if !first.is_whitespace() {
            buf.push_str(&rest[..start + INVOKE_OPEN.len()]);
            rest = after_open;
            continue;
        }
        let Some(tag_end) = after_open.find('>') else {
            break;
        };
        let attrs = &after_open[..tag_end];
        let body_and_rest = &after_open[tag_end + 1..];
        let name = attr_value(attrs, "name").filter(|n| !n.is_empty());

        match body_and_rest.find(INVOKE_CLOSE) {
            Some(close_rel) => {
                let body = &body_and_rest[..close_rel];
                buf.push_str(&rest[..start]);
                if !buf.is_empty() {
                    out.push(ScanEvent::Text(std::mem::take(&mut buf)));
                }
                match classify_body(name.as_deref(), body) {
                    None => {
                        if let Some(name) = name {
                            out.push(ScanEvent::Tool(ToolCall {
                                name,
                                arguments: body.trim().to_string(),
                            }));
                        }
                    }
                    Some(reason) => out.push(ScanEvent::Malformed(MalformedCall::new(
                        name,
                        body.to_string(),
                        reason,
                    ))),
                }
                rest = &body_and_rest[close_rel + INVOKE_CLOSE.len()..];
            }
            None => {
                // Output cut before the closing tag: an unterminated block.
                buf.push_str(&rest[..start]);
                if !buf.is_empty() {
                    out.push(ScanEvent::Text(std::mem::take(&mut buf)));
                }
                let reason =
                    classify_body(name.as_deref(), body_and_rest).unwrap_or("unterminated_block");
                out.push(ScanEvent::Malformed(MalformedCall::new(
                    name,
                    body_and_rest.to_string(),
                    reason,
                )));
                rest = "";
            }
        }
    }
    buf.push_str(rest);
    if !buf.is_empty() {
        out.push(ScanEvent::Text(buf));
    }

    // A bare `<parameter name="x">value</parameter>` with no enclosing invoke:
    // the model confused the two levels, so it must reach repair, not prose.
    let mut with_orphans = Vec::with_capacity(out.len());
    for event in out {
        match event {
            ScanEvent::Text(text) if text.contains("<parameter") => {
                let mut remainder = text.as_str();
                while let Some(at) = remainder.find("<parameter") {
                    if at > 0 {
                        with_orphans.push(ScanEvent::Text(remainder[..at].to_string()));
                    }
                    let tail = &remainder[at..];
                    let open_end = tail.find('>').map(|i| i + 1).unwrap_or(tail.len());
                    let body_end = tail
                        .find("</parameter>")
                        .map(|i| i + "</parameter>".len())
                        .unwrap_or(tail.len());
                    let block = &tail[..body_end.max(open_end)];
                    with_orphans.push(ScanEvent::Malformed(MalformedCall::new(
                        None,
                        block.to_string(),
                        "orphan_parameter_block",
                    )));
                    remainder = &tail[body_end.max(open_end)..];
                }
                if !remainder.is_empty() {
                    with_orphans.push(ScanEvent::Text(remainder.to_string()));
                }
            }
            other => with_orphans.push(other),
        }
    }
    with_orphans
}

/// Tool names offered to the model, in request order, deduplicated.
///
/// Superseded by [`tool_specs`] on the live path; kept for the ordering test.
#[allow(dead_code)]
pub fn tool_names(tools: &[serde_json::Value]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for tool in tools {
        let Some(name) = tool
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
        else {
            continue;
        };
        if !out.iter().any(|n| n == name) {
            out.push(name.to_string());
        }
    }
    out
}

/// Python's `json.dumps(value, ensure_ascii=False)` with default separators.
///
/// Used for the `Available schemas:` dump so the repair prompt matches Python's
/// byte for byte: `{"a": 1}` rather than serde's compact `{"a":1}`. Non-ASCII
/// is kept as-is, which is what `ensure_ascii=False` does.
pub fn py_json_dumps(value: &serde_json::Value) -> String {
    let mut out = String::new();
    write_py_json(value, &mut out);
    out
}

fn write_py_json(value: &serde_json::Value, out: &mut String) {
    match value {
        serde_json::Value::Null => out.push_str("null"),
        serde_json::Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        serde_json::Value::Number(n) => out.push_str(&n.to_string()),
        serde_json::Value::String(s) => {
            out.push('"');
            for ch in s.chars() {
                match ch {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    '\u{08}' => out.push_str("\\b"),
                    '\u{0c}' => out.push_str("\\f"),
                    c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                    c => out.push(c),
                }
            }
            out.push('"');
        }
        serde_json::Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py_json(item, out);
            }
            out.push(']');
        }
        serde_json::Value::Object(map) => {
            out.push('{');
            for (i, (key, val)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py_json(&serde_json::Value::String(key.clone()), out);
                out.push_str(": ");
                write_py_json(val, out);
            }
            out.push('}');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extracts_single_call() {
        let text = "Sure.\n<invoke name=\"bash\">{\"command\": \"ls\"}</invoke>";
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bash");
        assert_eq!(calls[0].json().unwrap(), json!({"command": "ls"}));
    }

    #[test]
    fn extracts_multiple_calls() {
        let text = concat!(
            "<invoke name=\"a\">{\"x\": 1}</invoke>\n",
            "<invoke name=\"b\">{\"y\": [1,2]}</invoke>"
        );
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].name, "b");
        assert_eq!(calls[1].json().unwrap(), json!({"y": [1, 2]}));
    }

    #[test]
    fn skips_malformed_body() {
        assert!(extract_tool_calls("<invoke name=\"b\">not json</invoke>").is_empty());
    }

    #[test]
    fn skips_missing_name() {
        assert!(extract_tool_calls("<invoke>{\"x\": 1}</invoke>").is_empty());
    }

    #[test]
    fn does_not_match_invokex() {
        assert!(extract_tool_calls("<invokex name=\"a\">{\"x\": 1}</invokex>").is_empty());
    }

    #[test]
    fn strip_leaves_prose() {
        let text = "before <invoke name=\"t\">{\"a\":1}</invoke> after";
        let stripped = strip_tool_calls(text);
        assert!(stripped.contains("before"));
        assert!(stripped.contains("after"));
        assert!(!stripped.contains("invoke"));
    }

    #[test]
    fn normalizes_dsml_invoke_tags() {
        // Single-pipe DSML spelling, with attributes preserved.
        let text = "<｜DSML｜invoke name=\"bash\">{\"command\":\"ls\"}</｜DSML｜invoke>";
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bash");
        assert_eq!(calls[0].json().unwrap(), json!({"command": "ls"}));
    }

    #[test]
    fn normalizes_dsml_with_wrapper_and_double_pipes() {
        let text = "<｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name=\"a\">{\"x\":1}</｜｜DSML｜｜invoke></｜｜DSML｜｜tool_calls>";
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[0].json().unwrap(), json!({"x": 1}));
    }

    #[test]
    fn strip_removes_dsml_blocks_too() {
        let text = "before <｜DSML｜invoke name=\"t\">{\"a\":1}</｜DSML｜invoke> after";
        let stripped = strip_tool_calls(text);
        assert!(stripped.contains("before"));
        assert!(stripped.contains("after"));
        assert!(!stripped.contains("DSML"));
    }

    #[test]
    fn detector_reports_invalid_json() {
        let found = extract_malformed("<invoke name=\"bash\">not json</invoke>");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name.as_deref(), Some("bash"));
        assert_eq!(found[0].reason, "invalid_json_body");
    }

    #[test]
    fn detector_reports_non_object_body() {
        let found = extract_malformed("<invoke name=\"a\">[1,2]</invoke>");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].reason, "body_not_object");
    }

    #[test]
    fn detector_ignores_good_blocks() {
        assert!(extract_malformed("<invoke name=\"a\">{\"x\":1}</invoke>").is_empty());
    }

    #[test]
    fn detector_finds_unterminated_block() {
        let found = extract_malformed("text <invoke name=\"a\">{\"x\":1}");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name.as_deref(), Some("a"));
        assert_eq!(found[0].reason, "unterminated_block");
    }

    #[test]
    fn detector_finds_orphan_parameter_block() {
        let found = extract_malformed("<parameter name=\"command\">ls</parameter>");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name, None);
        assert_eq!(found[0].reason, "orphan_parameter_block");
    }

    #[test]
    fn detector_dedupes_repeated_blocks() {
        let block = "<invoke name=\"a\">bad</invoke>";
        let text = format!("{block}{block}");
        assert_eq!(extract_malformed(&text).len(), 1);
    }

    #[test]
    fn detector_handles_dsml_spelling() {
        let found = extract_malformed("<｜DSML｜invoke name=\"a\">nope</｜DSML｜invoke>");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].reason, "invalid_json_body");
    }

    #[test]
    fn repair_prompt_names_failures_without_bodies() {
        let malformed = vec![MalformedCall {
            name: Some("bash".into()),
            body: "not json".into(),
            reason: "invalid_json_body".into(),
        }];
        let specs = vec![
            ToolSpec {
                name: "bash".into(),
                raw: json!({"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}),
            },
            ToolSpec {
                name: "edit".into(),
                raw: json!({"type": "function", "function": {"name": "edit", "parameters": {"type": "object"}}}),
            },
        ];
        let prompt = build_repair_prompt(&malformed, &specs);
        assert!(prompt.contains("Failure 1: name=\"bash\" reason=invalid_json_body"));
        assert!(prompt.contains("Valid tool names: bash, edit."));
        assert!(prompt.contains(TEMPLATE));
        // `Recovery.begin_repair()` appends this line, so a repair turn must too.
        assert!(
            prompt.contains("Only fix the listed failures. Do not repeat calls already accepted.")
        );
        // The schemas are appended by `begin_repair`, not here.
        assert!(!prompt.contains("Available schemas"));
        // The bad body must not be replayed into the same conversation.
        assert!(!prompt.contains("not json"));
    }

    #[test]
    fn tool_names_are_deduped_in_order() {
        let tools = vec![
            json!({"type": "function", "function": {"name": "b"}}),
            json!({"type": "function", "function": {"name": "a"}}),
            json!({"type": "function", "function": {"name": "b"}}),
        ];
        assert_eq!(tool_names(&tools), vec!["b", "a"]);
    }

    fn bash_spec() -> ToolSpec {
        ToolSpec {
            name: "bash".into(),
            raw: json!({
                "type": "function",
                "function": {
                    "name": "bash",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"]
                    }
                }
            }),
        }
    }

    fn call(name: &str, args: &str) -> ToolCall {
        ToolCall {
            name: name.into(),
            arguments: args.into(),
        }
    }

    #[test]
    fn validation_accepts_a_schema_conforming_call() {
        let specs = vec![bash_spec()];
        assert_eq!(
            validation_reason(&specs, &call("bash", "{\"command\": \"ls\"}")),
            None
        );
    }

    #[test]
    fn validation_rejects_unknown_tool() {
        let specs = vec![bash_spec()];
        assert_eq!(
            validation_reason(&specs, &call("nope", "{}")),
            Some("unknown_or_disallowed_tool")
        );
    }

    #[test]
    fn validation_rejects_non_object_arguments() {
        let specs = vec![bash_spec()];
        assert_eq!(
            validation_reason(&specs, &call("bash", "[1,2]")),
            Some("arguments_must_be_object")
        );
    }

    #[test]
    fn validation_rejects_invalid_json() {
        let specs = vec![bash_spec()];
        assert_eq!(
            validation_reason(&specs, &call("bash", "not json")),
            Some("invalid_arguments_json")
        );
    }

    #[test]
    fn validation_rejects_schema_mismatch() {
        let specs = vec![bash_spec()];
        // `command` is required by the schema and missing here.
        assert_eq!(
            validation_reason(&specs, &call("bash", "{}")),
            Some("arguments_do_not_match_schema")
        );
    }

    #[test]
    fn fingerprint_sorts_keys_like_python() {
        // Python canonicalises with sort_keys=True, so key order must not
        // distinguish two otherwise identical calls.
        let a = call("bash", "{\"b\": 1, \"a\": 2}");
        let b = call("bash", "{\"a\": 2, \"b\": 1}");
        assert_eq!(fingerprint(&a), fingerprint(&b));
    }

    #[test]
    fn scan_preserves_order_of_text_calls_and_failures() {
        let text = concat!(
            "before ",
            "<invoke name=\"bash\">{\"command\": \"ls\"}</invoke>",
            " middle ",
            "<invoke name=\"bash\">oops</invoke>",
            " after"
        );
        let events = scan(text);
        assert!(matches!(events[0], ScanEvent::Text(ref t) if t.contains("before")));
        assert!(matches!(events[1], ScanEvent::Tool(_)));
        assert!(matches!(events[2], ScanEvent::Text(ref t) if t.contains("middle")));
        assert!(matches!(events[3], ScanEvent::Malformed(_)));
        assert!(matches!(events[4], ScanEvent::Text(ref t) if t.contains("after")));
    }

    #[test]
    fn scan_reports_orphan_parameter_blocks() {
        let events = scan("text <parameter name=\"command\">ls</parameter>");
        assert!(events
            .iter()
            .any(|e| matches!(e, ScanEvent::Malformed(m) if m.reason == "orphan_parameter_block")));
    }

    #[test]
    fn py_json_dumps_matches_python_separators() {
        // Python: json.dumps({"a": 1, "b": [1, 2]}) == '{"a": 1, "b": [1, 2]}'
        let v = json!({"a": 1, "b": [1, 2]});
        assert_eq!(py_json_dumps(&v), "{\"a\": 1, \"b\": [1, 2]}");
    }

    #[test]
    fn py_json_dumps_keeps_non_ascii() {
        // ensure_ascii=False
        assert_eq!(py_json_dumps(&json!("héllo")), "\"héllo\"");
    }

    #[test]
    fn inject_noop_without_tools() {
        assert_eq!(inject_tool_descriptions("sys", &[]), "sys");
    }

    #[test]
    fn inject_adds_protocol_and_tools() {
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "bash",
                "description": "run a command",
                "parameters": {"type": "object"}
            }
        })];
        let out = inject_tool_descriptions("BASE", &tools);
        assert!(out.starts_with("BASE"));
        assert!(out.contains("TOOL CALL PROTOCOL"));
        assert!(out.contains("### bash"));
        assert!(out.contains("FINAL REMINDER"));
    }
}
/// Incremental filter that separates tool-call blocks from prose in a
/// streamed text channel.
///
/// `<invoke ...>{json}</invoke>` blocks can arrive split across chunks, so
/// text is buffered until a block is complete (or provably absent). Text that
/// precedes an opening tag is released immediately; a trailing partial tag
/// (e.g. `...<inv`) is held back so it is never emitted half-formed.
pub struct StreamToolFilter {
    pending: String,
    /// Completed calls, in discovery order.
    pub calls: Vec<ToolCall>,
}

impl Default for StreamToolFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamToolFilter {
    pub fn new() -> Self {
        Self {
            pending: String::new(),
            calls: Vec::new(),
        }
    }

    /// Feed a text chunk; returns prose that is safe to emit as content now.
    pub fn feed(&mut self, text: &str) -> String {
        self.pending.push_str(text);
        let normalized = normalize_dsml(&self.pending);
        if normalized != self.pending {
            self.pending = normalized;
        }
        let mut out = String::new();

        loop {
            match self.pending.find(INVOKE_OPEN) {
                None => {
                    // Hold back a trailing incomplete tag like "<inv" or a
                    // half-arrived marker spelling, so it is never emitted
                    // half-formed.
                    let keep = partial_tag_suffix(&self.pending);
                    let split = self.pending.len() - keep;
                    out.push_str(&self.pending[..split]);
                    self.pending.drain(..split);
                    break;
                }
                Some(start) => {
                    let after = &self.pending[start..];
                    match after.find(INVOKE_CLOSE) {
                        Some(close) => {
                            // Complete block: parse and consume it.
                            out.push_str(&self.pending[..start]);
                            let block = after[..close + INVOKE_CLOSE.len()].to_string();
                            self.calls.extend(extract_tool_calls(&block));
                            self.pending.drain(..start + close + INVOKE_CLOSE.len());
                        }
                        None => {
                            // Incomplete block: emit everything before it.
                            out.push_str(&self.pending[..start]);
                            self.pending.drain(..start);
                            break;
                        }
                    }
                }
            }
        }

        out
    }

    /// Flush at end of stream. An unterminated block degrades to text.
    pub fn finish(&mut self) -> String {
        let rest = std::mem::take(&mut self.pending);
        // Only a *complete* block would have been parsed; anything left is prose.
        rest
    }
}

/// Length of the trailing incomplete tag in `s`: everything from the last
/// `<` that has no closing `>` yet. Capped so ordinary prose with a stray `<`
/// is not held back forever.
fn partial_tag_suffix(s: &str) -> usize {
    match s.rfind('<') {
        Some(i) if !s[i..].contains('>') => {
            let len = s.len() - i;
            if len <= 256 {
                len
            } else {
                0
            }
        }
        _ => 0,
    }
}

#[cfg(test)]
mod stream_tests {
    use super::*;

    #[test]
    fn passes_plain_text_through() {
        let mut f = StreamToolFilter::new();
        assert_eq!(f.feed("hello "), "hello ");
        assert_eq!(f.feed("world"), "world");
        assert_eq!(f.finish(), "");
        assert!(f.calls.is_empty());
    }

    #[test]
    fn extracts_block_split_across_chunks() {
        let mut f = StreamToolFilter::new();
        assert_eq!(f.feed("Sure. <inv"), "Sure. ");
        assert_eq!(f.feed("oke name=\"bash\">{\"command\":"), "");
        assert_eq!(f.feed("\"ls\"}</invoke>"), "");
        assert_eq!(f.calls.len(), 1);
        assert_eq!(f.calls[0].name, "bash");
        assert_eq!(
            f.calls[0].json().unwrap(),
            serde_json::json!({"command": "ls"})
        );
    }

    #[test]
    fn holds_back_partial_tag_until_finish() {
        let mut f = StreamToolFilter::new();
        assert_eq!(f.feed("text <inv"), "text ");
        // "<invoke" is still an incomplete opening tag: hold it, do not emit.
        assert_eq!(f.feed("oke"), "");
        // Only at end of stream does it degrade to text.
        assert_eq!(f.finish(), "<invoke");
    }

    #[test]
    fn unterminated_block_becomes_text_at_finish() {
        let mut f = StreamToolFilter::new();
        assert_eq!(f.feed("a <invoke name=\"x\">{"), "a ");
        assert_eq!(f.finish(), "<invoke name=\"x\">{");
        assert!(f.calls.is_empty());
    }

    #[test]
    fn prose_after_block_is_emitted() {
        let mut f = StreamToolFilter::new();
        let out = f.feed("<invoke name=\"t\">{\"a\":1}</invoke> done");
        assert_eq!(out, " done");
        assert_eq!(f.calls.len(), 1);
    }
}
