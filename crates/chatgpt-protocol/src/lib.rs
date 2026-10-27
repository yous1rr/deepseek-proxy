//! ChatGPT Web conversation SSE and JSON-patch protocol handling.
//!
//! This crate intentionally exposes channel-neutral events to an adapter. It
//! does not make network requests, execute tool payloads, or include upstream
//! JSON in errors. Unknown valid JSON is ignored; malformed and oversized
//! payloads return small, safe error values.

use serde_json::{Map, Value};
use std::fmt;

/// Maximum size of one SSE data payload, in bytes.
pub const MAX_PAYLOAD_BYTES: usize = 4 * 1024 * 1024;
/// Maximum size of the accumulated raw or sanitized assistant text, in bytes.
pub const MAX_CUMULATIVE_TEXT_BYTES: usize = 8 * 1024 * 1024;
/// Maximum number of distinct output file and sediment IDs retained together.
pub const MAX_IDS: usize = 64;
/// Maximum number of distinct search source URLs emitted for one payload.
pub const MAX_SOURCE_URLS: usize = 64;

const ANNOTATION_START: char = '\u{e200}';
const ANNOTATION_END: char = '\u{e201}';
const ANNOTATION_SEPARATOR: char = '\u{e202}';

/// One SSE line. Multi-line `data:` aggregation is left to the transport layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Line<'a> {
    Data(&'a str),
    Event(&'a str),
    Comment,
    Blank,
}

/// Classify one SSE line without retaining or combining stream state.
pub fn parse_sse_line(line: &str) -> Line<'_> {
    let line = line.strip_suffix('\r').unwrap_or(line);
    if line.is_empty() {
        return Line::Blank;
    }
    if line.starts_with(':') {
        return Line::Comment;
    }

    if let Some(value) = line.strip_prefix("data:") {
        return Line::Data(value.strip_prefix(' ').unwrap_or(value));
    }
    if line == "data" {
        return Line::Data("");
    }
    if let Some(value) = line.strip_prefix("event:") {
        return Line::Event(value.strip_prefix(' ').unwrap_or(value));
    }
    if line == "event" {
        return Line::Event("");
    }
    Line::Blank
}

/// Mutable state needed while consuming one upstream conversation stream.
///
/// The listed protocol fields are public for adapters. The private visibility
/// cursor prevents patches following an explicitly hidden/user/tool message
/// from being mistaken for user-visible assistant text. Use [`PatchState::new`]
/// or [`Default::default`] to construct a state.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PatchState {
    pub text: String,
    pub raw_text: String,
    pub conversation_id: String,
    pub file_ids: Vec<String>,
    pub sediment_ids: Vec<String>,
    pub tool_invoked: Option<bool>,
    pub turn_use_case: String,
    pub blocked: bool,
    assistant_visible: Option<bool>,
}

impl PatchState {
    /// Create an empty conversation state.
    pub fn new() -> Self {
        Self::default()
    }
}

/// A channel-neutral semantic event produced from ChatGPT Web payloads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConversationEvent {
    Started {
        conversation_id: String,
    },
    TextDelta(String),
    /// Reserved: the visible-message filter drops every non-`final` channel, so
    /// no reasoning text is produced today. The variant stays so the adapter
    /// and its tests do not have to change when a thought channel is mapped.
    #[allow(dead_code)]
    ReasoningDelta(String),
    ImageTool {
        conversation_id: String,
        file_ids: Vec<String>,
        sediment_ids: Vec<String>,
    },
    SearchTool {
        conversation_id: String,
        query: Option<String>,
        sources: Vec<Source>,
    },
    ModerationBlocked,
    Done,
    RawIgnored,
}

/// A validated web-search source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Source {
    pub title: String,
    pub url: String,
    pub snippet: Option<String>,
}

/// Safe, payload-free protocol errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProtocolError {
    InvalidJson,
    InvalidUtf8,
    OversizedPayload,
}

impl fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidJson => "invalid JSON protocol payload",
            Self::InvalidUtf8 => "invalid UTF-8 protocol payload",
            Self::OversizedPayload => "protocol payload exceeds configured limits",
        };
        f.write_str(message)
    }
}

impl std::error::Error for ProtocolError {}

/// Consume one SSE line. Only `data:` lines are dispatched; comments, event
/// names, and blank/unknown lines produce `None`.
pub fn consume_sse_line(
    state: &mut PatchState,
    line: &str,
) -> Result<Option<ConversationEvent>, ProtocolError> {
    match parse_sse_line(line) {
        Line::Data(payload) => consume_payload(state, payload).map(Some),
        Line::Event(_) | Line::Comment | Line::Blank => Ok(None),
    }
}

/// Consume one decoded SSE data payload and update conversation state.
pub fn consume_payload(
    state: &mut PatchState,
    payload: &str,
) -> Result<ConversationEvent, ProtocolError> {
    if payload.len() > MAX_PAYLOAD_BYTES {
        return Err(ProtocolError::OversizedPayload);
    }
    if payload.trim() == "[DONE]" {
        return Ok(ConversationEvent::Done);
    }
    if payload.trim().is_empty() {
        return Ok(ConversationEvent::RawIgnored);
    }

    let value: Value = serde_json::from_str(payload).map_err(|_| ProtocolError::InvalidJson)?;
    if !value.is_object() && !value.is_array() {
        return consume_scalar(state, &value);
    }

    // Work against a clone so a cumulative-text limit failure cannot leave
    // partially applied nested patches in the caller's state.
    let mut next = state.clone();
    let old_text = next.text.clone();
    let was_unstarted = next.conversation_id.is_empty();

    update_conversation_id(&mut next, &value);
    update_metadata(&mut next, &value);
    if contains_blocked_moderation(&value) {
        next.blocked = true;
    }

    let mut image_messages = Vec::new();
    collect_image_tool_messages(&value, &mut image_messages);
    let image_tool = !image_messages.is_empty();
    let mut image_ids = Ids::default();
    for message in &image_messages {
        let (files, sediments) = extract_image_ids(message);
        image_ids.extend(&files, &sediments);
    }

    let patch_event = contains_patch_operation(&value);
    let user_message = contains_user_message(&value);
    if image_tool
        || (next.tool_invoked == Some(true) && !user_message)
        || (patch_event && !user_message)
    {
        if !image_tool {
            let (files, sediments) = extract_image_ids(&value);
            image_ids.extend(&files, &sediments);
        }
        normalize_state_ids(&mut next);
        add_state_ids(&mut next, &image_ids.file_ids, &image_ids.sediment_ids);
    }

    let (is_search, query, sources) = search_details(&value);
    apply_text_tree(&mut next, &value)?;

    let started = was_unstarted && !next.conversation_id.is_empty();
    let event = if contains_blocked_moderation(&value) {
        ConversationEvent::ModerationBlocked
    } else if image_tool {
        ConversationEvent::ImageTool {
            conversation_id: next.conversation_id.clone(),
            file_ids: next.file_ids.clone(),
            sediment_ids: next.sediment_ids.clone(),
        }
    } else if next.text != old_text {
        ConversationEvent::TextDelta(text_delta(&old_text, &next.text))
    } else if is_search {
        ConversationEvent::SearchTool {
            conversation_id: next.conversation_id.clone(),
            query,
            sources,
        }
    } else if started {
        ConversationEvent::Started {
            conversation_id: next.conversation_id.clone(),
        }
    } else {
        ConversationEvent::RawIgnored
    };

    *state = next;
    Ok(event)
}

fn consume_scalar(
    state: &mut PatchState,
    value: &Value,
) -> Result<ConversationEvent, ProtocolError> {
    if let Some(text) = value.as_str() {
        if is_protocol_version(text) || state.assistant_visible == Some(false) {
            return Ok(ConversationEvent::RawIgnored);
        }
        let mut next = state.clone();
        let old_text = next.text.clone();
        append_raw_text(&mut next, text)?;
        let event = if next.text != old_text {
            ConversationEvent::TextDelta(text_delta(&old_text, &next.text))
        } else {
            ConversationEvent::RawIgnored
        };
        *state = next;
        return Ok(event);
    }
    Ok(ConversationEvent::RawIgnored)
}

fn is_protocol_version(text: &str) -> bool {
    let Some(digits) = text.strip_prefix('v') else {
        return false;
    };
    !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
}

fn update_conversation_id(state: &mut PatchState, value: &Value) {
    if !state.conversation_id.is_empty() {
        return;
    }
    visit_values(value, &mut |node| {
        if !state.conversation_id.is_empty() {
            return;
        }
        if let Some(conversation_id) = node
            .as_object()
            .and_then(|object| object.get("conversation_id"))
            .and_then(Value::as_str)
            .filter(|conversation_id| !conversation_id.is_empty())
        {
            state.conversation_id = conversation_id.to_owned();
        }
    });
}

fn update_metadata(state: &mut PatchState, value: &Value) {
    visit_values(value, &mut |node| {
        let Some(metadata) = node
            .as_object()
            .and_then(|object| object.get("metadata"))
            .and_then(Value::as_object)
        else {
            return;
        };
        if let Some(tool_invoked) = metadata.get("tool_invoked").and_then(Value::as_bool) {
            state.tool_invoked = Some(tool_invoked);
        }
        if let Some(turn_use_case) = metadata
            .get("turn_use_case")
            .and_then(Value::as_str)
            .filter(|turn_use_case| !turn_use_case.is_empty())
        {
            state.turn_use_case = turn_use_case.to_owned();
        }
    });
}

fn contains_blocked_moderation(value: &Value) -> bool {
    let mut blocked = false;
    visit_values(value, &mut |node| {
        let Some(object) = node.as_object() else {
            return;
        };
        if object.get("type").and_then(Value::as_str) == Some("moderation")
            && object
                .get("moderation_response")
                .and_then(Value::as_object)
                .and_then(|response| response.get("blocked"))
                .and_then(Value::as_bool)
                == Some(true)
        {
            blocked = true;
        }
    });
    blocked
}

fn contains_patch_operation(value: &Value) -> bool {
    let mut found = false;
    visit_values(value, &mut |node| {
        if node
            .as_object()
            .and_then(|object| object.get("o"))
            .and_then(Value::as_str)
            .is_some_and(|operation| operation.eq_ignore_ascii_case("patch"))
        {
            found = true;
        }
    });
    found
}

fn contains_user_message(value: &Value) -> bool {
    let mut found = false;
    visit_values(value, &mut |node| {
        if node.as_object().is_some_and(is_user_message_object) {
            found = true;
        }
    });
    found
}

fn is_user_message_object(object: &Map<String, Value>) -> bool {
    if object
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|kind| kind.eq_ignore_ascii_case("input_message"))
    {
        return true;
    }
    let author_role = object
        .get("author")
        .and_then(Value::as_object)
        .and_then(|author| author.get("role"))
        .and_then(Value::as_str);
    let direct_role = object.get("role").and_then(Value::as_str);
    author_role.is_some_and(|role| role.eq_ignore_ascii_case("user"))
        || (direct_role.is_some_and(|role| role.eq_ignore_ascii_case("user"))
            && (object.contains_key("content") || object.contains_key("parts")))
}

fn collect_image_tool_messages<'a>(value: &'a Value, output: &mut Vec<&'a Value>) {
    visit_values(value, &mut |node| {
        if is_image_tool_message(node) {
            output.push(node);
        }
    });
}

fn is_image_tool_message(value: &Value) -> bool {
    let Some(message) = value.as_object() else {
        return false;
    };
    let role = message
        .get("author")
        .and_then(Value::as_object)
        .and_then(|author| author.get("role"))
        .and_then(Value::as_str);
    let task_type = message
        .get("metadata")
        .and_then(Value::as_object)
        .and_then(|metadata| metadata.get("async_task_type"))
        .and_then(Value::as_str);
    role.is_some_and(|role| role.eq_ignore_ascii_case("tool")) && task_type == Some("image_gen")
}

#[derive(Default)]
struct Ids {
    file_ids: Vec<String>,
    sediment_ids: Vec<String>,
}

impl Ids {
    fn extend(&mut self, file_ids: &[String], sediment_ids: &[String]) {
        for file_id in file_ids {
            push_unique_id(&mut self.file_ids, &self.sediment_ids, file_id);
        }
        for sediment_id in sediment_ids {
            push_unique_id(&mut self.sediment_ids, &self.file_ids, sediment_id);
        }
    }
}

fn normalize_state_ids(state: &mut PatchState) {
    let mut ids = Ids::default();
    ids.extend(&state.file_ids, &state.sediment_ids);
    state.file_ids = ids.file_ids;
    state.sediment_ids = ids.sediment_ids;
}

fn add_state_ids(state: &mut PatchState, file_ids: &[String], sediment_ids: &[String]) {
    let mut ids = Ids::default();
    ids.extend(&state.file_ids, &state.sediment_ids);
    ids.extend(file_ids, sediment_ids);
    state.file_ids = ids.file_ids;
    state.sediment_ids = ids.sediment_ids;
}

fn push_unique_id(target: &mut Vec<String>, other: &[String], candidate: &str) {
    if candidate.is_empty()
        || target.iter().any(|value| value == candidate)
        || target.len().saturating_add(other.len()) >= MAX_IDS
    {
        return;
    }
    target.push(candidate.to_owned());
}

/// Extract output-style file and sediment IDs from JSON values.
///
/// User-message and `input_message` subtrees are deliberately skipped so input
/// image references cannot be mistaken for generated outputs. Plain `file_…`
/// strings are accepted only when they match ChatGPT's fixed 37-character
/// image ID form; URI pointers are accepted only for the two known schemes.
pub fn extract_image_ids(value: &Value) -> (Vec<String>, Vec<String>) {
    let mut ids = Ids::default();
    collect_ids(value, false, false, &mut ids);
    (ids.file_ids, ids.sediment_ids)
}

fn collect_ids(value: &Value, inherited_user: bool, input_message: bool, ids: &mut Ids) {
    if inherited_user || input_message {
        return;
    }
    match value {
        Value::Object(object) => {
            if is_user_message_object(object) {
                return;
            }
            for (key, child) in object {
                collect_ids(child, false, key == "input_message", ids);
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_ids(item, false, false, ids);
            }
        }
        Value::String(text) => collect_ids_from_string(text, ids),
        _ => {}
    }
}

fn collect_ids_from_string(text: &str, ids: &mut Ids) {
    const REAL_ID_PREFIX: &str = "file_00000000";
    let bytes = text.as_bytes();
    let mut offset = 0;
    while offset + REAL_ID_PREFIX.len() <= bytes.len() {
        if bytes[offset..].starts_with(REAL_ID_PREFIX.as_bytes()) {
            let end = offset + REAL_ID_PREFIX.len() + 24;
            let left_boundary = offset == 0 || !is_ascii_word(bytes[offset - 1]);
            let right_boundary = end >= bytes.len() || !is_ascii_word(bytes[end]);
            if end <= bytes.len()
                && left_boundary
                && right_boundary
                && bytes[offset + REAL_ID_PREFIX.len()..end]
                    .iter()
                    .all(u8::is_ascii_hexdigit)
            {
                if let Some(id) = text.get(offset..end) {
                    push_unique_id(&mut ids.file_ids, &ids.sediment_ids, id);
                }
                offset = end;
            } else {
                offset += 1;
            }
        } else {
            offset += 1;
        }
    }

    collect_pointer_ids(text, "file-service://", true, ids);
    collect_pointer_ids(text, "sediment://", false, ids);
}

fn is_ascii_word(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn collect_pointer_ids(text: &str, prefix: &str, is_file: bool, ids: &mut Ids) {
    for (start, _) in text.match_indices(prefix) {
        let id_start = start + prefix.len();
        let bytes = text.as_bytes();
        let mut end = id_start;
        while end < bytes.len()
            && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_' || bytes[end] == b'-')
        {
            end += 1;
        }
        if end == id_start {
            continue;
        }
        let Some(candidate) = text.get(id_start..end) else {
            continue;
        };
        let lower = candidate.to_ascii_lowercase();
        if lower == "file_upload" || lower.starts_with("file_upload_") {
            continue;
        }
        if is_file {
            push_unique_id(&mut ids.file_ids, &ids.sediment_ids, candidate);
        } else {
            push_unique_id(&mut ids.sediment_ids, &ids.file_ids, candidate);
        }
    }
}

fn apply_text_tree(state: &mut PatchState, value: &Value) -> Result<(), ProtocolError> {
    match value {
        Value::Array(items) => {
            for item in items {
                apply_text_tree(state, item)?;
            }
        }
        Value::Object(object) => {
            if is_message_object(object) {
                let visible = is_visible_assistant_message(object);
                state.assistant_visible = Some(visible);
                if visible {
                    let text = assistant_message_text(object);
                    replace_raw_text(state, &text)?;
                }
                return Ok(());
            }

            let path = object.get("p").and_then(Value::as_str).unwrap_or("");
            let operation = object.get("o").and_then(Value::as_str).unwrap_or("");
            let patch_value = object.get("v");
            if operation.eq_ignore_ascii_case("patch") {
                if let Some(patch_value) = patch_value {
                    apply_text_tree(state, patch_value)?;
                }
                return Ok(());
            }

            if let Some(text) = patch_value.and_then(Value::as_str) {
                let has_text_path = is_text_path(path);
                let unscoped = path.trim_matches('/').is_empty() && operation.is_empty();
                let pathless_append =
                    path.trim_matches('/').is_empty() && operation.eq_ignore_ascii_case("append");
                if (has_text_path || unscoped || pathless_append)
                    && state.assistant_visible != Some(false)
                {
                    if has_text_path || pathless_append {
                        if operation.eq_ignore_ascii_case("append") {
                            append_raw_text(state, text)?;
                        } else if operation.eq_ignore_ascii_case("replace")
                            || operation.eq_ignore_ascii_case("add")
                        {
                            replace_raw_text(state, text)?;
                        }
                    } else if unscoped {
                        append_raw_text(state, text)?;
                    }
                    return Ok(());
                }
                if has_text_path || unscoped || pathless_append {
                    return Ok(());
                }
            }

            for child in object.values() {
                apply_text_tree(state, child)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn is_message_object(object: &Map<String, Value>) -> bool {
    object.get("author").and_then(Value::as_object).is_some()
}

fn is_visible_assistant_message(message: &Map<String, Value>) -> bool {
    let Some(role) = message
        .get("author")
        .and_then(Value::as_object)
        .and_then(|author| author.get("role"))
        .and_then(Value::as_str)
    else {
        return false;
    };
    if !role.eq_ignore_ascii_case("assistant") {
        return false;
    }

    if message
        .get("metadata")
        .and_then(Value::as_object)
        .and_then(|metadata| metadata.get("is_visually_hidden_from_conversation"))
        .and_then(Value::as_bool)
        == Some(true)
    {
        return false;
    }

    let recipient = message
        .get("recipient")
        .and_then(Value::as_str)
        .unwrap_or("");
    if !recipient.trim().is_empty() && !recipient.trim().eq_ignore_ascii_case("all") {
        return false;
    }
    let channel = message.get("channel").and_then(Value::as_str).unwrap_or("");
    channel.trim().is_empty() || channel.trim().eq_ignore_ascii_case("final")
}

fn assistant_message_text(message: &Map<String, Value>) -> String {
    let Some(content) = message.get("content") else {
        return String::new();
    };
    if let Some(content) = content.as_str() {
        return content.to_owned();
    }
    let Some(content_object) = content.as_object() else {
        return String::new();
    };
    if let Some(parts) = content_object.get("parts").and_then(Value::as_array) {
        let mut text = String::new();
        for part in parts {
            if let Some(part) = part.as_str() {
                text.push_str(part);
            } else if let Some(part_object) = part.as_object() {
                if part_object
                    .get("type")
                    .and_then(Value::as_str)
                    .is_some_and(|kind| matches!(kind, "text" | "output_text"))
                {
                    if let Some(part_text) = part_object.get("text").and_then(Value::as_str) {
                        text.push_str(part_text);
                    }
                }
            }
        }
        if !text.is_empty() {
            return text;
        }
    }
    content_object
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned()
}

fn is_text_path(path: &str) -> bool {
    matches!(
        path.trim_matches('/'),
        "message/content/parts/0" | "message/content/text"
    )
}

fn append_raw_text(state: &mut PatchState, text: &str) -> Result<(), ProtocolError> {
    let new_len = state
        .raw_text
        .len()
        .checked_add(text.len())
        .ok_or(ProtocolError::OversizedPayload)?;
    if new_len > MAX_CUMULATIVE_TEXT_BYTES {
        return Err(ProtocolError::OversizedPayload);
    }
    let mut next = String::with_capacity(new_len);
    next.push_str(&state.raw_text);
    next.push_str(text);
    replace_raw_text(state, &next)
}

fn replace_raw_text(state: &mut PatchState, text: &str) -> Result<(), ProtocolError> {
    if text.len() > MAX_CUMULATIVE_TEXT_BYTES {
        return Err(ProtocolError::OversizedPayload);
    }
    let sanitized = sanitize_text(text);
    if sanitized.len() > MAX_CUMULATIVE_TEXT_BYTES {
        return Err(ProtocolError::OversizedPayload);
    }
    state.raw_text.clear();
    state.raw_text.push_str(text);
    state.text = sanitized;
    Ok(())
}

/// Text to emit for the step from `previous` to `current`.
///
/// The web stream is a snapshot per patch, not an append-only log, so `current`
/// is not guaranteed to start with `previous`: a closed citation annotation can
/// rewrite what was already sent (whitespace before punctuation is dropped) and
/// a re-sent snapshot repeats text verbatim. Emitting only the part past the
/// longest common prefix keeps the forwarded text equal to `current` instead of
/// duplicating it, and emits nothing when the snapshot is unchanged.
fn text_delta(previous: &str, current: &str) -> String {
    if let Some(suffix) = current.strip_prefix(previous) {
        return suffix.to_owned();
    }
    let common = previous
        .char_indices()
        .zip(current.chars())
        .find(|((_, left), right)| left != right)
        .map(|((index, _), _)| index)
        .unwrap_or_else(|| previous.len().min(current.len()));
    let boundary = (0..=common)
        .rev()
        .find(|index| current.is_char_boundary(*index))
        .unwrap_or(0);
    current[boundary..].to_owned()
}

/// Remove ChatGPT's private-use annotation markers while retaining readable
/// citation labels and safe HTTP(S) links from URL annotations.
pub fn sanitize_text(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut cursor = 0;
    while let Some(relative_start) = text[cursor..].find(ANNOTATION_START) {
        let start = cursor + relative_start;
        output.push_str(&text[cursor..start]);
        let payload_start = start + ANNOTATION_START.len_utf8();
        let Some(relative_end) = text[payload_start..].find(ANNOTATION_END) else {
            cursor = text.len();
            break;
        };
        let end_start = payload_start + relative_end;
        let end = end_start + ANNOTATION_END.len_utf8();
        let replacement = annotation_text(&text[payload_start..end_start]);
        let mut next_cursor = end;
        if replacement.is_empty() {
            if next_is_punctuation(&text[end..]) {
                while output.chars().last().is_some_and(char::is_whitespace) {
                    output.pop();
                }
            } else if output.chars().last().is_some_and(char::is_whitespace) {
                while let Some(character) = text[next_cursor..].chars().next() {
                    if !character.is_whitespace() {
                        break;
                    }
                    next_cursor += character.len_utf8();
                }
            }
        }
        output.push_str(&replacement);
        cursor = next_cursor;
    }
    if cursor < text.len() {
        output.push_str(&text[cursor..]);
    }
    output
        .chars()
        .filter(|character| {
            !matches!(
                *character,
                ANNOTATION_START | ANNOTATION_END | ANNOTATION_SEPARATOR
            )
        })
        .collect()
}

fn annotation_text(payload: &str) -> String {
    let parts: Vec<&str> = payload.split(ANNOTATION_SEPARATOR).collect();
    let kind = parts
        .first()
        .copied()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    let data = &parts[1..];
    if kind == "url" {
        let label = data.first().copied().unwrap_or("").trim();
        let url = data.get(1).copied().unwrap_or("").trim();
        let label = strip_annotation_markers(label);
        let safe_url = sanitize_http_url(url);
        return match (label.is_empty(), safe_url) {
            (false, Some(url)) => format!("{label} ({url})"),
            (false, None) => label,
            (true, Some(url)) => url,
            (true, None) => String::new(),
        };
    }

    for part in data {
        let candidate = strip_annotation_markers(part.trim());
        let lower = candidate.to_ascii_lowercase();
        if !candidate.is_empty() && !lower.starts_with("turn") && !lower.starts_with("source") {
            return candidate;
        }
    }
    String::new()
}

fn strip_annotation_markers(value: &str) -> String {
    value
        .chars()
        .filter(|character| {
            !matches!(
                *character,
                ANNOTATION_START | ANNOTATION_END | ANNOTATION_SEPARATOR
            )
        })
        .collect()
}

fn next_is_punctuation(value: &str) -> bool {
    value
        .chars()
        .next()
        .is_some_and(|character| matches!(character, '.' | ',' | ';' | ':' | '!' | '?'))
}

fn sanitize_http_url(value: &str) -> Option<String> {
    let value = value.trim();
    if value.is_empty()
        || value.chars().any(|character| {
            character.is_whitespace() || character.is_control() || character == '\\'
        })
    {
        return None;
    }

    let delimiter = value.find("://")?;
    let scheme = &value[..delimiter];
    let scheme = if scheme.eq_ignore_ascii_case("http") {
        "http"
    } else if scheme.eq_ignore_ascii_case("https") {
        "https"
    } else {
        return None;
    };
    let remainder = &value[delimiter + 3..];
    let safe_end = remainder
        .find(|character| character == '?' || character == '#')
        .unwrap_or(remainder.len());
    let without_query = &remainder[..safe_end];
    let path_start = without_query.find('/').unwrap_or(without_query.len());
    let authority = &without_query[..path_start];
    let path = &without_query[path_start..];
    if authority.is_empty() || authority.contains('@') {
        return None;
    }

    if authority.starts_with('[') {
        let close = authority.find(']')?;
        let host = &authority[1..close];
        if host.is_empty()
            || !host
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() || matches!(byte, b':' | b'.' | b'%'))
        {
            return None;
        }
        let suffix = &authority[close + 1..];
        if !suffix.is_empty() {
            let port = suffix.strip_prefix(':')?;
            if port.is_empty() || port.parse::<u16>().is_err() {
                return None;
            }
        }
    } else {
        let (host, port) = match authority.rsplit_once(':') {
            Some((host, port)) if !host.contains(':') => (host, Some(port)),
            Some(_) => return None,
            None => (authority, None),
        };
        if host.is_empty()
            || host.starts_with('.')
            || host.ends_with('.')
            || host
                .bytes()
                .any(|byte| !(byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-')))
        {
            return None;
        }
        if host
            .split('.')
            .any(|label| label.is_empty() || label.starts_with('-') || label.ends_with('-'))
        {
            return None;
        }
        if let Some(port) = port {
            if port.is_empty() || port.parse::<u16>().is_err() {
                return None;
            }
        }
    }

    Some(format!("{scheme}://{authority}{path}"))
}

fn search_details(value: &Value) -> (bool, Option<String>, Vec<Source>) {
    let mut is_search = false;
    visit_values(value, &mut |node| {
        let Some(object) = node.as_object() else {
            return;
        };
        if object.keys().any(|key| {
            matches!(
                key.as_str(),
                "sources" | "search_results" | "search_result_groups" | "url_citations"
            )
        }) {
            is_search = true;
        }
        for key in ["type", "name", "tool_name"] {
            let Some(text) = object.get(key).and_then(Value::as_str) else {
                continue;
            };
            let text = text.to_ascii_lowercase();
            if text.contains("search") || text == "browser" || text.starts_with("web.") {
                is_search = true;
            }
        }
        if object
            .get("recipient")
            .and_then(Value::as_str)
            .is_some_and(|recipient| {
                let recipient = recipient.to_ascii_lowercase();
                recipient.contains("search") || recipient == "browser"
            })
        {
            is_search = true;
        }
        if object
            .get("author")
            .and_then(Value::as_object)
            .and_then(|author| author.get("name"))
            .and_then(Value::as_str)
            .is_some_and(|name| {
                let name = name.to_ascii_lowercase();
                name.contains("search") || name == "browser" || name.starts_with("web.")
            })
        {
            is_search = true;
        }
        if object
            .get("metadata")
            .and_then(Value::as_object)
            .and_then(|metadata| metadata.get("async_task_type"))
            .and_then(Value::as_str)
            .is_some_and(|task| task.to_ascii_lowercase().contains("search"))
        {
            is_search = true;
        }
    });

    let query = first_query(value)
        .map(|query| sanitize_text(query.trim()))
        .filter(|query| !query.is_empty());
    let mut sources = Vec::new();
    collect_sources(value, false, &mut sources);
    if !sources.is_empty() {
        is_search = true;
    }
    (is_search, query, sources)
}

fn first_query(value: &Value) -> Option<String> {
    let mut found = None;
    visit_values(value, &mut |node| {
        if found.is_some() {
            return;
        }
        let Some(object) = node.as_object() else {
            return;
        };
        for key in ["query", "search_query", "searchQuery"] {
            if let Some(query) = object
                .get(key)
                .and_then(Value::as_str)
                .filter(|query| !query.trim().is_empty())
            {
                found = Some(query.to_owned());
                return;
            }
        }
    });
    found
}

fn collect_sources(value: &Value, source_context: bool, sources: &mut Vec<Source>) {
    match value {
        Value::Object(object) => {
            let url = ["url", "link", "source_url", "uri"]
                .iter()
                .find_map(|key| object.get(*key).and_then(Value::as_str));
            let is_url_citation = object
                .get("type")
                .and_then(Value::as_str)
                .is_some_and(|kind| kind.eq_ignore_ascii_case("url_citation"));
            let looks_like_source = source_context
                || is_url_citation
                || object.keys().any(|key| {
                    matches!(
                        key.as_str(),
                        "title" | "snippet" | "description" | "site_name"
                    )
                });
            if looks_like_source {
                if let Some(url) = url.and_then(sanitize_http_url) {
                    let title = ["title", "source_title", "site_name", "name"]
                        .iter()
                        .find_map(|key| object.get(*key).and_then(Value::as_str))
                        .map(|title| sanitize_text(title.trim()))
                        .unwrap_or_default();
                    let snippet = ["snippet", "description", "extract", "text"]
                        .iter()
                        .find_map(|key| object.get(*key).and_then(Value::as_str))
                        .map(|snippet| sanitize_text(snippet.trim()))
                        .filter(|snippet| !snippet.is_empty());
                    push_source(
                        sources,
                        Source {
                            title,
                            url,
                            snippet,
                        },
                    );
                }
            }

            for (key, child) in object {
                let child_context = source_context
                    || matches!(
                        key.as_str(),
                        "sources"
                            | "search_results"
                            | "search_result_groups"
                            | "results"
                            | "citations"
                            | "url_citations"
                            | "web_search_results"
                    );
                collect_sources(child, child_context, sources);
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_sources(item, source_context, sources);
            }
        }
        Value::String(text) if source_context && text.trim().starts_with("http") => {
            if let Some(url) = sanitize_http_url(text) {
                push_source(
                    sources,
                    Source {
                        title: String::new(),
                        url,
                        snippet: None,
                    },
                );
            }
        }
        _ => {}
    }
}

fn push_source(sources: &mut Vec<Source>, source: Source) {
    if sources.len() >= MAX_SOURCE_URLS || sources.iter().any(|existing| existing.url == source.url)
    {
        return;
    }
    sources.push(source);
}

fn visit_values<'a>(value: &'a Value, visitor: &mut impl FnMut(&'a Value)) {
    visitor(value);
    match value {
        Value::Array(items) => {
            for item in items {
                visit_values(item, visitor);
            }
        }
        Value::Object(object) => {
            for child in object.values() {
                visit_values(child, visitor);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_sse_line_kinds_and_optional_space() {
        assert_eq!(parse_sse_line("data: {\"a\":1}"), Line::Data("{\"a\":1}"));
        assert_eq!(
            parse_sse_line("data:  two spaces"),
            Line::Data(" two spaces")
        );
        assert_eq!(parse_sse_line("event: message"), Line::Event("message"));
        assert_eq!(parse_sse_line(": keepalive"), Line::Comment);
        assert_eq!(parse_sse_line("\r"), Line::Blank);
        assert_eq!(parse_sse_line("unknown: value"), Line::Blank);
    }

    #[test]
    fn applies_append_replace_and_nested_patch_operations() {
        let mut state = PatchState::new();
        assert_eq!(
            consume_payload(
                &mut state,
                r#"{"p":"/message/content/parts/0","o":"append","v":"Hello"}"#
            )
            .unwrap(),
            ConversationEvent::TextDelta("Hello".into())
        );
        assert_eq!(
            consume_payload(
                &mut state,
                r#"{"p":"/message/content/parts/0","o":"append","v":" world"}"#
            )
            .unwrap(),
            ConversationEvent::TextDelta(" world".into())
        );
        // A `replace` that rewrites the text cannot be expressed over an
        // append-only event stream. Only the part past the longest common
        // prefix is sent, so an already-forwarded `<invoke>` block can never be
        // repeated; the client keeps the stale remainder instead of seeing the
        // snapshot twice.
        assert_eq!(
            consume_payload(
                &mut state,
                r#"{"p":"/message/content/parts/0","o":"replace","v":"Hi"}"#
            )
            .unwrap(),
            ConversationEvent::TextDelta("i".into())
        );
        assert_eq!(
            consume_payload(
                &mut state,
                r#"{"p":"","o":"patch","v":[{"p":"/message/content/parts/0","o":"append","v":" there"},{"p":"","o":"patch","v":[{"p":"/message/content/parts/0","o":"replace","v":"Nested"},{"p":"/message/content/parts/0","o":"append","v":" patch"}]}]}"#
            )
            .unwrap(),
            ConversationEvent::TextDelta("Nested patch".into())
        );
        assert_eq!(state.text, "Nested patch");
        assert_eq!(state.raw_text, "Nested patch");
    }

    #[test]
    fn only_visible_assistant_messages_and_their_patches_become_text() {
        let mut state = PatchState::new();
        let visible = json!({
            "conversation_id": "conv-visible",
            "v": {"message": {
                "author": {"role": "assistant"},
                "recipient": "all",
                "channel": "final",
                "content": {"parts": ["answer"]}
            }}
        });
        assert_eq!(
            consume_payload(&mut state, &visible.to_string()).unwrap(),
            ConversationEvent::TextDelta("answer".into())
        );
        assert_eq!(state.text, "answer");

        let hidden = json!({"message": {
            "author": {"role": "assistant"},
            "recipient": "web.run",
            "channel": "analysis",
            "content": {"parts": ["private"]}
        }});
        assert_eq!(
            consume_payload(&mut state, &hidden.to_string()).unwrap(),
            ConversationEvent::RawIgnored
        );
        assert_eq!(state.text, "answer");
        assert_eq!(
            consume_payload(
                &mut state,
                r#"{"p":"/message/content/parts/0","o":"append","v":" hidden patch"}"#
            )
            .unwrap(),
            ConversationEvent::RawIgnored
        );
        assert_eq!(state.text, "answer");

        for hidden_message in [
            json!({"message": {"author":{"role":"assistant"},"recipient":"all","channel":"analysis","content":{"parts":["analysis"]}}}),
            json!({"message": {"author":{"role":"assistant"},"recipient":"all","channel":"final","metadata":{"is_visually_hidden_from_conversation":true},"content":{"parts":["hidden"]}}}),
            json!({"message": {"author":{"role":"user"},"content":{"parts":["input"]}}}),
        ] {
            let _ = consume_payload(&mut state, &hidden_message.to_string()).unwrap();
            assert_eq!(state.text, "answer");
        }
    }

    #[test]
    fn moderation_and_done_are_semantic_events() {
        let mut state = PatchState::new();
        assert_eq!(
            consume_payload(
                &mut state,
                r#"{"type":"moderation","conversation_id":"conv-1","moderation_response":{"blocked":true}}"#
            )
            .unwrap(),
            ConversationEvent::ModerationBlocked
        );
        assert!(state.blocked);
        assert_eq!(
            consume_sse_line(&mut state, "event: message").unwrap(),
            None
        );
        assert_eq!(
            consume_sse_line(&mut state, "data: [DONE]").unwrap(),
            Some(ConversationEvent::Done)
        );
    }

    #[test]
    fn sanitizes_private_annotations_and_redacts_annotation_url_parameters() {
        let input = "Useful text\u{e200}cite\u{e202}turn1\u{e202}source2\u{e201}. See \u{e200}url\u{e202}Guide\u{e202}https://example.test/page?secret=private\u{e201}.";
        assert_eq!(
            sanitize_text(&input),
            "Useful text. See Guide (https://example.test/page)."
        );
        assert_eq!(sanitize_text("visible\u{e200}unfinished"), "visible");
    }

    /// A closed citation annotation rewrites text that was already forwarded.
    /// The incremental delta must describe the new snapshot, not repeat it.
    #[test]
    fn text_delta_reports_a_rewritten_snapshot_instead_of_repeating_it() {
        // The annotation is unterminated at first, so sanitizing truncates the
        // tail; once it closes, the whitespace before the period disappears.
        assert_eq!(text_delta("See ", "See."), ".");
        assert_eq!(text_delta("answer", "answer"), "");
        assert_eq!(text_delta("answer", "answer more"), " more");
        // A fully rewritten snapshot cannot be expressed incrementally.
        assert_eq!(text_delta("abc", "xyz"), "xyz");
        assert_eq!(text_delta("", "new"), "new");
        // Multi-byte text must not be split mid-character.
        assert_eq!(text_delta("答案", "答案：好了"), "：好了");
    }

    #[test]
    fn excludes_user_assets_and_extracts_generated_image_tool_pointers() {
        let mut state = PatchState::new();
        let input = json!({
            "type": "input_message",
            "asset_pointer": "sediment://root-user-pointer",
            "input_message": {
                "author": {"role": "user"},
                "content": {"parts": [
                    {"asset_pointer": "sediment://input-image"},
                    {"asset_pointer": "file-service://file_input"},
                    "file_00000000000000000000000000000000"
                ]}
            }
        });
        assert_eq!(extract_image_ids(&input), (Vec::new(), Vec::new()));
        assert_eq!(
            consume_payload(&mut state, &input.to_string()).unwrap(),
            ConversationEvent::RawIgnored
        );
        assert!(state.file_ids.is_empty());
        assert!(state.sediment_ids.is_empty());

        let generated_id = "file_00000000000000000000000000000000";
        let generated = json!({
            "conversation_id": "conv-image",
            "v": {"message": {
                "author": {"role": "tool"},
                "metadata": {"async_task_type": "image_gen"},
                "content": {"content_type": "multimodal_text", "parts": [
                    {"asset_pointer": format!("file-service://{generated_id}")},
                    {"asset_pointer": format!("sediment://{generated_id}")},
                    {"asset_pointer": "file-service://file_upload"}
                ]}
            }}
        });
        let event = consume_payload(&mut state, &generated.to_string()).unwrap();
        assert_eq!(
            event,
            ConversationEvent::ImageTool {
                conversation_id: "conv-image".into(),
                file_ids: vec![generated_id.into()],
                sediment_ids: vec![generated_id.into()],
            }
        );
        assert_eq!(state.file_ids, vec![generated_id]);
        assert_eq!(state.sediment_ids, vec![generated_id]);

        let direct = json!({"items": [generated_id, "file_00000000not-a-real-id"]});
        assert_eq!(extract_image_ids(&direct).0, vec![generated_id]);
    }

    #[test]
    fn extracts_and_sanitizes_search_sources() {
        let mut state = PatchState::new();
        let payload = json!({
            "type": "web_search_call",
            "query": "Rust protocol",
            "sources": [
                {"title": "Rust Guide", "url": "https://example.test/rust?token=private#section", "snippet": "A \u{e200}cite\u{e202}turn0\u{e202}source0\u{e201} guide."},
                {"title": "unsafe scheme", "url": "javascript:alert(1)"}
            ]
        });
        assert_eq!(
            consume_payload(&mut state, &payload.to_string()).unwrap(),
            ConversationEvent::SearchTool {
                conversation_id: String::new(),
                query: Some("Rust protocol".into()),
                sources: vec![Source {
                    title: "Rust Guide".into(),
                    url: "https://example.test/rust".into(),
                    snippet: Some("A guide.".into()),
                }],
            }
        );
    }

    #[test]
    fn malformed_and_oversized_payloads_return_safe_errors_without_partial_state() {
        let mut state = PatchState::new();
        assert_eq!(
            consume_payload(&mut state, "{private malformed payload"),
            Err(ProtocolError::InvalidJson)
        );
        assert!(!ProtocolError::InvalidJson.to_string().contains("private"));

        let oversized = " ".repeat(MAX_PAYLOAD_BYTES + 1);
        assert_eq!(
            consume_payload(&mut state, &oversized),
            Err(ProtocolError::OversizedPayload)
        );

        state.raw_text = "x".repeat(MAX_CUMULATIVE_TEXT_BYTES);
        let before = state.clone();
        assert_eq!(
            consume_payload(
                &mut state,
                r#"{"p":"/message/content/parts/0","o":"append","v":"y"}"#
            ),
            Err(ProtocolError::OversizedPayload)
        );
        assert_eq!(state, before);
    }

    #[test]
    fn emits_started_and_accepts_json_string_text() {
        let mut state = PatchState::new();
        assert_eq!(
            consume_payload(
                &mut state,
                r#"{"type":"resume_conversation_token","conversation_id":"conv-start"}"#
            )
            .unwrap(),
            ConversationEvent::Started {
                conversation_id: "conv-start".into(),
            }
        );
        assert_eq!(
            consume_payload(&mut state, r#""JSON string text""#).unwrap(),
            ConversationEvent::TextDelta("JSON string text".into())
        );
        assert_eq!(state.text, "JSON string text");
    }

    #[test]
    fn unknown_valid_payloads_and_protocol_version_are_ignored_safely() {
        let mut state = PatchState::new();
        assert_eq!(
            consume_payload(&mut state, r#""v1""#).unwrap(),
            ConversationEvent::RawIgnored
        );
        assert_eq!(
            consume_payload(
                &mut state,
                r#"{"type":"future_event","secret":"do not echo"}"#
            )
            .unwrap(),
            ConversationEvent::RawIgnored
        );
    }
}
