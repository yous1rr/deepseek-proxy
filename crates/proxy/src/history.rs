//! Conversation history normalization and versioned SHA-256 fingerprinting.
//!
//! Replaces arbitrary client or provider metadata with a canonical representation
//! of the meaningful conversation prefix so branch detection and session reuse
//! are deterministic across client implementations (e.g. Pi Desktop).

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Returns a versioned SHA-256 fingerprint of the normalized conversation messages.
pub fn fingerprint(messages: &[Value]) -> String {
    let normalized = normalize_messages(messages);
    let mut hasher = Sha256::new();
    let json_bytes = serde_json::to_vec(&normalized).expect("serialization is infallible");
    hasher.update(&json_bytes);
    format!("v1:{}", hex::encode(hasher.finalize()))
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct NormalizedMessage {
    role: String,
    name: Option<String>,
    content: Vec<NormalizedContentPart>,
    tool_calls: Vec<NormalizedToolCall>,
    tool_call_id: Option<NormalizedToolCallRef>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "type")]
enum NormalizedContentPart {
    Text {
        text: String,
    },
    Image {
        url: Option<String>,
        detail: Option<String>,
        media_type: Option<String>,
        data: Option<String>,
        extra: Option<String>,
    },
    Unknown {
        raw: String,
    },
}

#[derive(Debug, PartialEq, Eq, Serialize)]
struct NormalizedToolCall {
    ordinal: usize,
    call_type: String,
    name: String,
    arguments: NormalizedArgs,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", content = "value")]
enum NormalizedArgs {
    Json(String),
    Raw(String),
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", content = "value")]
enum NormalizedToolCallRef {
    Matched(usize),
    Unknown(String),
}

/// Recursively formats a JSON value into a canonical string:
/// keys sorted lexicographically, no unnecessary whitespace.
fn write_canonical_json(val: &Value, out: &mut String) {
    match val {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => {
            if let Ok(escaped) = serde_json::to_string(s) {
                out.push_str(&escaped);
            } else {
                out.push('"');
                out.push_str(s);
                out.push('"');
            }
        }
        Value::Array(arr) => {
            out.push('[');
            for (i, elem) in arr.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical_json(elem, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut entries: Vec<(&str, &Value)> =
                map.iter().map(|(k, v)| (k.as_str(), v)).collect();
            entries.sort_by_key(|&(k, _)| k);
            out.push('{');
            for (i, (k, v)) in entries.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                if let Ok(escaped_key) = serde_json::to_string(k) {
                    out.push_str(&escaped_key);
                } else {
                    out.push('"');
                    out.push_str(k);
                    out.push('"');
                }
                out.push(':');
                write_canonical_json(v, out);
            }
            out.push('}');
        }
    }
}

/// Normalizes tool call arguments: parsed to canonical JSON if valid, else raw string.
fn normalize_args(args: Option<&Value>) -> NormalizedArgs {
    match args {
        None | Some(Value::Null) => NormalizedArgs::Json("null".to_string()),
        Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
            Ok(parsed) => {
                let mut out = String::new();
                write_canonical_json(&parsed, &mut out);
                NormalizedArgs::Json(out)
            }
            Err(_) => NormalizedArgs::Raw(s.clone()),
        },
        Some(val) => {
            let mut out = String::new();
            write_canonical_json(val, &mut out);
            NormalizedArgs::Json(out)
        }
    }
}

/// Attempts to parse an image content part from a JSON object.
fn try_extract_image(map: &serde_json::Map<String, Value>) -> Option<NormalizedContentPart> {
    let part_type = map.get("type").and_then(|v| v.as_str());
    let has_image_type = matches!(part_type, Some("image_url") | Some("image"));
    let is_implicit_image_url = part_type.is_none() && map.contains_key("image_url");

    if !has_image_type && !is_implicit_image_url {
        return None;
    }

    let mut url = None;
    let mut detail = None;
    let mut media_type = None;
    let mut data = None;

    if let Some(img_url_val) = map.get("image_url") {
        match img_url_val {
            Value::String(s) => {
                url = Some(s.clone());
            }
            Value::Object(sub_map) => {
                if let Some(u) = sub_map.get("url").and_then(|v| v.as_str()) {
                    url = Some(u.to_string());
                }
                if let Some(d) = sub_map.get("detail").and_then(|v| v.as_str()) {
                    detail = Some(d.to_string());
                }
            }
            _ => {}
        }
    }

    if url.is_none() {
        if let Some(u) = map.get("url").and_then(|v| v.as_str()) {
            url = Some(u.to_string());
        }
    }

    if detail.is_none() {
        if let Some(d) = map.get("detail").and_then(|v| v.as_str()) {
            detail = Some(d.to_string());
        }
    }

    if let Some(source_val) = map.get("source").and_then(|v| v.as_object()) {
        if let Some(d) = source_val.get("data").and_then(|v| v.as_str()) {
            data = Some(d.to_string());
        }
        if let Some(mt) = source_val.get("media_type").and_then(|v| v.as_str()) {
            media_type = Some(mt.to_string());
        }
    }

    if data.is_none() {
        if let Some(d) = map.get("data").and_then(|v| v.as_str()) {
            data = Some(d.to_string());
        }
    }

    if media_type.is_none() {
        if let Some(mt) = map.get("media_type").and_then(|v| v.as_str()) {
            media_type = Some(mt.to_string());
        }
    }

    // Canonicalize standalone data URIs. If separate source fields also exist,
    // retain the URL so conflicting payloads cannot collapse to one digest.
    if data.is_none() && media_type.is_none() {
        if let Some(ref u) = url {
            if u.starts_with("data:") || u.starts_with("DATA:") {
                if let Some((header, payload)) = u[5..].split_once(',') {
                    if let Some((mime, encoding)) = header.rsplit_once(';') {
                        if encoding.eq_ignore_ascii_case("base64") {
                            media_type = Some(mime.to_ascii_lowercase());
                            data = Some(payload.to_string());
                            url = None;
                        }
                    }
                }
            }
        }
    }

    // If it did not have an image type and no image content was actually extracted,
    // this is not an image (e.g. an object with an unrelated "source" or "url" field).
    if url.is_none() && data.is_none() && !has_image_type {
        return None;
    }

    let mut extra_map = serde_json::Map::new();
    for (k, v) in map {
        if !matches!(
            k.as_str(),
            "type"
                | "image_url"
                | "source"
                | "url"
                | "detail"
                | "data"
                | "media_type"
                | "id"
                | "cache_control"
                | "provider_metadata"
        ) {
            extra_map.insert(k.clone(), v.clone());
        }
    }
    let extra = if extra_map.is_empty() {
        None
    } else {
        let mut out = String::new();
        write_canonical_json(&Value::Object(extra_map), &mut out);
        Some(out)
    };
    Some(NormalizedContentPart::Image {
        url,
        detail,
        media_type,
        data,
        extra,
    })
}

/// Normalizes message content: concatenates adjacent text parts, preserves whitespace,
/// normalizes null, empty string, and empty array to empty parts list, and extracts images.
fn normalize_content(content_val: Option<&Value>) -> Vec<NormalizedContentPart> {
    let Some(val) = content_val else {
        return Vec::new();
    };

    let mut parts = Vec::new();
    let mut text_buf = String::new();

    let flush_text = |buf: &mut String, out: &mut Vec<NormalizedContentPart>| {
        if !buf.is_empty() {
            out.push(NormalizedContentPart::Text {
                text: std::mem::take(buf),
            });
        }
    };

    match val {
        Value::Null => {}
        Value::String(s) => {
            if !s.is_empty() {
                parts.push(NormalizedContentPart::Text { text: s.clone() });
            }
        }
        Value::Array(items) => {
            for item in items {
                match item {
                    Value::String(s) => {
                        text_buf.push_str(s);
                    }
                    Value::Object(map) => {
                        let part_type = map.get("type").and_then(|v| v.as_str());
                        let is_text_type = part_type == Some("text");
                        let is_implicit_text = part_type.is_none() && map.contains_key("text");

                        if is_text_type || is_implicit_text {
                            if let Some(t) = map.get("text").and_then(|v| v.as_str()) {
                                text_buf.push_str(t);
                            }
                        } else if let Some(img_part) = try_extract_image(map) {
                            flush_text(&mut text_buf, &mut parts);
                            parts.push(img_part);
                        } else if matches!(
                            part_type,
                            Some("thinking")
                                | Some("thought")
                                | Some("reasoning")
                                | Some("redacted_thinking")
                        ) {
                            // Reasoning metadata content blocks are ignored per spec.
                        } else {
                            flush_text(&mut text_buf, &mut parts);
                            let mut raw = String::new();
                            write_canonical_json(item, &mut raw);
                            parts.push(NormalizedContentPart::Unknown { raw });
                        }
                    }
                    _ => {
                        flush_text(&mut text_buf, &mut parts);
                        let mut raw = String::new();
                        write_canonical_json(item, &mut raw);
                        parts.push(NormalizedContentPart::Unknown { raw });
                    }
                }
            }
            flush_text(&mut text_buf, &mut parts);
        }
        Value::Object(map) => {
            let part_type = map.get("type").and_then(|v| v.as_str());
            let is_text_type = part_type == Some("text");
            let is_implicit_text = part_type.is_none() && map.contains_key("text");

            if is_text_type || is_implicit_text {
                if let Some(t) = map.get("text").and_then(|v| v.as_str()) {
                    if !t.is_empty() {
                        parts.push(NormalizedContentPart::Text {
                            text: t.to_string(),
                        });
                    }
                }
            } else if let Some(img_part) = try_extract_image(map) {
                parts.push(img_part);
            } else if matches!(
                part_type,
                Some("thinking") | Some("thought") | Some("reasoning") | Some("redacted_thinking")
            ) {
                // Reasoning metadata content blocks are ignored per spec.
            } else {
                let mut raw = String::new();
                write_canonical_json(val, &mut raw);
                parts.push(NormalizedContentPart::Unknown { raw });
            }
        }
        _ => {
            let mut raw = String::new();
            write_canonical_json(val, &mut raw);
            parts.push(NormalizedContentPart::Unknown { raw });
        }
    }

    parts
}

/// Normalizes a sequence of message values: canonicalizes roles, names, content,
/// assistant tool calls (by occurrence ordinal), and tool call results (linked to
/// the most recent corresponding assistant call occurrence).
fn normalize_messages(messages: &[Value]) -> Vec<NormalizedMessage> {
    let mut normalized = Vec::with_capacity(messages.len());
    let mut next_tool_call_ordinal: usize = 0;
    let mut latest_call_by_raw_id: HashMap<String, usize> = HashMap::new();

    for msg in messages {
        let role = msg
            .get("role")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_lowercase();

        let name = msg
            .get("name")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string());

        let content = normalize_content(msg.get("content"));

        let mut tool_calls = Vec::new();
        if role == "assistant" {
            if let Some(calls) = msg.get("tool_calls").and_then(|v| v.as_array()) {
                for call in calls {
                    let ordinal = next_tool_call_ordinal;
                    next_tool_call_ordinal += 1;

                    if let Some(raw_id) = call.get("id").and_then(|v| v.as_str()) {
                        if !raw_id.is_empty() {
                            latest_call_by_raw_id.insert(raw_id.to_string(), ordinal);
                        }
                    }

                    let call_type = call
                        .get("type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("function")
                        .to_string();

                    let (fn_name, args_val) = if let Some(func) = call.get("function") {
                        let n = func.get("name").and_then(|v| v.as_str()).unwrap_or("");
                        let a = func.get("arguments");
                        (n, a)
                    } else {
                        let n = call.get("name").and_then(|v| v.as_str()).unwrap_or("");
                        let a = call.get("arguments");
                        (n, a)
                    };

                    let arguments = normalize_args(args_val);

                    tool_calls.push(NormalizedToolCall {
                        ordinal,
                        call_type,
                        name: fn_name.to_string(),
                        arguments,
                    });
                }
            } else if let Some(func_call) = msg.get("function_call").and_then(|v| v.as_object()) {
                let ordinal = next_tool_call_ordinal;
                next_tool_call_ordinal += 1;

                if let Some(raw_id) = func_call.get("id").and_then(|v| v.as_str()) {
                    if !raw_id.is_empty() {
                        latest_call_by_raw_id.insert(raw_id.to_string(), ordinal);
                    }
                }

                let fn_name = func_call.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let args_val = func_call.get("arguments");
                let arguments = normalize_args(args_val);

                tool_calls.push(NormalizedToolCall {
                    ordinal,
                    call_type: "function".to_string(),
                    name: fn_name.to_string(),
                    arguments,
                });
            }
        }

        let tool_call_id = msg
            .get("tool_call_id")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|raw_id| match latest_call_by_raw_id.get(raw_id) {
                Some(&call_ordinal) => NormalizedToolCallRef::Matched(call_ordinal),
                None => NormalizedToolCallRef::Unknown(raw_id.to_string()),
            });

        normalized.push(NormalizedMessage {
            role,
            name,
            content,
            tool_calls,
            tool_call_id,
        });
    }

    normalized
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_ordinary_next_request_prefix_equals_successful_history() {
        // Turn 1 completion pushes the assistant response to messages.
        let mut t1_history = vec![json!({"role": "user", "content": "Hello, world!"})];
        // main.rs TurnHistory::completed pushes assistant message with prose and calls.
        t1_history.push(json!({
            "role": "assistant",
            "content": "Hi there! How can I help you today?",
            "tool_calls": []
        }));
        let completed_fp = fingerprint(&t1_history);

        // Turn 2 incoming request includes the assistant message serialized from Msg,
        // plus an additional user turn.
        let t2_request = vec![
            json!({
                "role": "user",
                "content": "Hello, world!",
                "tool_calls": [],
                "tool_call_id": null,
                "name": null
            }),
            json!({
                "role": "assistant",
                "content": "Hi there! How can I help you today?",
                "tool_calls": [],
                "tool_call_id": null,
                "name": null
            }),
            json!({
                "role": "user",
                "content": "What is the capital of France?",
                "tool_calls": [],
                "tool_call_id": null,
                "name": null
            }),
        ];
        // Base prefix is through the last assistant (2 messages).
        let base_fp = fingerprint(&t2_request[..2]);
        assert_eq!(
            completed_fp, base_fp,
            "next request prefix must match completed history"
        );

        // Now test with tool calls in Turn 1 completion:
        let mut t1_tool_history = vec![json!({"role": "user", "content": "Compute 2+2"})];
        t1_tool_history.push(json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_0",
                "type": "function",
                "function": {"name": "calculator", "arguments": "{\"expr\": \"2+2\"}"}
            }]
        }));
        let completed_tool_fp = fingerprint(&t1_tool_history);

        // In Turn 2, client may return tool call with rewritten ID e.g. call_0|fc_0:
        let t2_tool_request = vec![
            json!({
                "role": "user",
                "content": "Compute 2+2",
                "tool_calls": [],
                "tool_call_id": null,
                "name": null
            }),
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_0|fc_0",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": "{\"expr\":\"2+2\"}"}
                }],
                "tool_call_id": null,
                "name": null
            }),
            json!({
                "role": "tool",
                "tool_call_id": "call_0|fc_0",
                "content": "4",
                "tool_calls": [],
                "name": null
            }),
        ];
        let base_tool_fp = fingerprint(&t2_tool_request[..2]);
        assert_eq!(
            completed_tool_fp, base_tool_fp,
            "next request prefix with tool calls must match completed history even with ID rewrite"
        );
    }

    #[test]
    fn test_truncate_edit_changes_history() {
        let base = vec![
            json!({"role": "user", "content": "First message"}),
            json!({"role": "assistant", "content": "First reply"}),
            json!({"role": "user", "content": "Second message"}),
            json!({"role": "assistant", "content": "Second reply"}),
        ];
        let base_fp = fingerprint(&base);

        // Truncate to first turn:
        let truncated = &base[..2];
        let trunc_fp = fingerprint(truncated);
        assert_ne!(
            base_fp, trunc_fp,
            "truncating history must change fingerprint"
        );

        // Edit user prompt:
        let mut edited_user = base.clone();
        edited_user[0] = json!({"role": "user", "content": "First message edited"});
        let edited_user_fp = fingerprint(&edited_user);
        assert_ne!(
            base_fp, edited_user_fp,
            "editing user message must change fingerprint"
        );

        // Edit assistant reply:
        let mut edited_assistant = base.clone();
        edited_assistant[1] = json!({"role": "assistant", "content": "First reply edited"});
        let edited_assistant_fp = fingerprint(&edited_assistant);
        assert_ne!(
            base_fp, edited_assistant_fp,
            "editing assistant message must change fingerprint"
        );
    }

    #[test]
    fn test_different_provider_ui_reasoning_fields_dont_alter() {
        let clean = vec![
            json!({"role": "user", "content": "Hello"}),
            json!({"role": "assistant", "content": "World"}),
        ];
        let with_metadata = vec![
            json!({
                "id": "msg_001",
                "role": "user",
                "content": "Hello",
                "reasoning_content": "User greeted politely",
                "thinking": "No tool needed",
                "thought": "Thinking process",
                "provider_metadata": {"tokens": 5},
                "metadata": {"ui_fold": false},
                "created": 1700000000
            }),
            json!({
                "id": "msg_002",
                "role": "assistant",
                "content": "World",
                "reasoning_content": "Replying with standard greeting continuation",
                "thinking": "Completed response",
                "client_id": "client-abc-123",
                "ui": {"highlight": true}
            }),
        ];

        assert_eq!(
            fingerprint(&clean),
            fingerprint(&with_metadata),
            "top-level message id, reasoning_content, thinking, and provider/UI metadata must not alter digest"
        );
    }

    #[test]
    fn test_text_shapes_and_empty_equivalent() {
        // Shapes: single string vs joined text part array
        let string_shape = vec![json!({"role": "user", "content": "hello world"})];
        let parts_shape_1 = vec![json!({"role": "user", "content": [
            {"type": "text", "text": "hello "},
            {"type": "text", "text": "world"}
        ]})];
        let parts_shape_2 = vec![json!({"role": "user", "content": [
            {"type": "text", "text": "hel"},
            {"type": "text", "text": "lo"},
            {"type": "text", "text": " world"}
        ]})];
        assert_eq!(
            fingerprint(&string_shape),
            fingerprint(&parts_shape_1),
            "string content and adjacent text parts must compare equally"
        );
        assert_eq!(
            fingerprint(&string_shape),
            fingerprint(&parts_shape_2),
            "arbitrary adjacent text slices must compare equally"
        );

        // Empty equivalents: null, "", [], and [{"type": "text", "text": ""}]
        let empty_null = vec![json!({"role": "assistant", "content": null})];
        let empty_string = vec![json!({"role": "assistant", "content": ""})];
        let empty_array = vec![json!({"role": "assistant", "content": []})];
        let empty_parts =
            vec![json!({"role": "assistant", "content": [{"type": "text", "text": ""}]})];
        let missing_content = vec![json!({"role": "assistant"})];

        assert_eq!(fingerprint(&empty_null), fingerprint(&empty_string));
        assert_eq!(fingerprint(&empty_null), fingerprint(&empty_array));
        assert_eq!(fingerprint(&empty_null), fingerprint(&empty_parts));
        assert_eq!(fingerprint(&empty_null), fingerprint(&missing_content));
    }

    #[test]
    fn test_tool_id_rewrite_json_key_whitespace_equivalence() {
        let history_a = vec![
            json!({"role": "user", "content": "search for rust"}),
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": "{\"page\": 1, \"query\": \"rust\"}"
                    }
                }]
            }),
            json!({
                "role": "tool",
                "tool_call_id": "call_0",
                "content": "rust result"
            }),
        ];

        // Rewritten ID: call_0 -> call_0|fc_call_0, different key order and spacing in arguments:
        let history_b = vec![
            json!({"role": "user", "content": "search for rust"}),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_0|fc_call_0",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": "{\n  \"query\": \"rust\",\n  \"page\": 1\n}"
                    }
                }]
            }),
            json!({
                "role": "tool",
                "tool_call_id": "call_0|fc_call_0",
                "content": "rust result"
            }),
        ];

        assert_eq!(
            fingerprint(&history_a),
            fingerprint(&history_b),
            "tool ID rewrite and JSON key order/whitespace differences must not change fingerprint"
        );
    }

    #[test]
    fn test_changed_arguments_results_alter() {
        let base = vec![
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": "{\"query\": \"rust\"}"
                    }
                }]
            }),
            json!({
                "role": "tool",
                "tool_call_id": "call_0",
                "content": "rust result"
            }),
        ];
        let base_fp = fingerprint(&base);

        // Changed argument:
        let changed_arg = vec![
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": "{\"query\": \"python\"}"
                    }
                }]
            }),
            json!({
                "role": "tool",
                "tool_call_id": "call_0",
                "content": "rust result"
            }),
        ];
        assert_ne!(
            base_fp,
            fingerprint(&changed_arg),
            "changed arguments must alter digest"
        );

        // Changed tool result:
        let changed_res = vec![
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": "{\"query\": \"rust\"}"
                    }
                }]
            }),
            json!({
                "role": "tool",
                "tool_call_id": "call_0",
                "content": "python result"
            }),
        ];
        assert_ne!(
            base_fp,
            fingerprint(&changed_res),
            "changed results must alter digest"
        );

        // Changed tool function name:
        let changed_fn = vec![
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": "{\"query\": \"rust\"}"
                    }
                }]
            }),
            json!({
                "role": "tool",
                "tool_call_id": "call_0",
                "content": "rust result"
            }),
        ];
        assert_ne!(
            base_fp,
            fingerprint(&changed_fn),
            "changed tool name must alter digest"
        );

        // Invalid JSON arguments retained raw:
        let raw_arg_1 = vec![json!({
            "role": "assistant",
            "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "invalid 1"}}]
        })];
        let raw_arg_2 = vec![json!({
            "role": "assistant",
            "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "invalid 2"}}]
        })];
        assert_ne!(
            fingerprint(&raw_arg_1),
            fingerprint(&raw_arg_2),
            "different invalid raw arguments must alter digest"
        );
    }

    #[test]
    fn test_repeated_ids_most_recent_result_association() {
        // Conversation where assistant outputs "call_0" on each turn:
        let history_repeated = vec![
            // Turn 1
            json!({
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "f1", "arguments": "{}"}
                }]
            }),
            json!({"role": "tool", "tool_call_id": "call_0", "content": "res1"}),
            // Turn 2
            json!({
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "f2", "arguments": "{}"}
                }]
            }),
            json!({"role": "tool", "tool_call_id": "call_0", "content": "res2"}),
        ];

        // Same conversation where client gave distinct globally unique IDs:
        let history_unique_ids = vec![
            // Turn 1
            json!({
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_first",
                    "type": "function",
                    "function": {"name": "f1", "arguments": "{}"}
                }]
            }),
            json!({"role": "tool", "tool_call_id": "call_first", "content": "res1"}),
            // Turn 2
            json!({
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_second",
                    "type": "function",
                    "function": {"name": "f2", "arguments": "{}"}
                }]
            }),
            json!({"role": "tool", "tool_call_id": "call_second", "content": "res2"}),
        ];

        assert_eq!(
            fingerprint(&history_repeated),
            fingerprint(&history_unique_ids),
            "repeated IDs associated with most recent call must match unique ordinal assignment"
        );

        // If Turn 2's result instead answers the earlier call ("call_first"):
        let history_cross_call = vec![
            json!({
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_first",
                    "type": "function",
                    "function": {"name": "f1", "arguments": "{}"}
                }]
            }),
            json!({"role": "tool", "tool_call_id": "call_first", "content": "res1"}),
            json!({
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_second",
                    "type": "function",
                    "function": {"name": "f2", "arguments": "{}"}
                }]
            }),
            json!({"role": "tool", "tool_call_id": "call_first", "content": "res2"}),
        ];
        assert_ne!(
            fingerprint(&history_unique_ids),
            fingerprint(&history_cross_call),
            "tool result referencing an earlier call must produce a different digest"
        );

        // Unknown result IDs must remain distinct from each other:
        let unknown_1 = vec![json!({"role": "tool", "tool_call_id": "orphan_a", "content": "ok"})];
        let unknown_2 = vec![json!({"role": "tool", "tool_call_id": "orphan_b", "content": "ok"})];
        assert_ne!(
            fingerprint(&unknown_1),
            fingerprint(&unknown_2),
            "distinct unknown result IDs must produce different digests"
        );
    }

    #[test]
    fn test_image_url_details_changes() {
        let base_image = vec![json!({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/pic1.jpg", "detail": "high"}}
        ]})];
        let base_fp = fingerprint(&base_image);

        // Changed URL:
        let changed_url = vec![json!({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/pic2.jpg", "detail": "high"}}
        ]})];
        assert_ne!(
            base_fp,
            fingerprint(&changed_url),
            "image URL change must alter digest"
        );

        // Changed detail:
        let changed_detail = vec![json!({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/pic1.jpg", "detail": "low"}}
        ]})];
        assert_ne!(
            base_fp,
            fingerprint(&changed_detail),
            "image detail change must alter digest"
        );

        // Omitted detail vs specified detail:
        let no_detail = vec![json!({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/pic1.jpg"}}
        ]})];
        assert_ne!(
            base_fp,
            fingerprint(&no_detail),
            "omitting image detail must alter digest"
        );

        // Object image_url vs string image_url equivalence (both have no detail):
        let string_image_url = vec![json!({"role": "user", "content": [
            {"type": "image_url", "image_url": "https://example.com/pic1.jpg"}
        ]})];
        assert_eq!(
            fingerprint(&no_detail),
            fingerprint(&string_image_url),
            "object image_url and string image_url with same url and no detail must compare equally"
        );

        // Base64 source image:
        let b64_image_1 = vec![json!({"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
        ]})];
        let b64_image_2 = vec![json!({"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "BBBB"}}
        ]})];
        assert_ne!(fingerprint(&b64_image_1), fingerprint(&b64_image_2));
    }

    #[test]
    fn test_role_and_order_changes() {
        // Role change:
        let user_msg = vec![json!({"role": "user", "content": "ping"})];
        let system_msg = vec![json!({"role": "system", "content": "ping"})];
        assert_ne!(
            fingerprint(&user_msg),
            fingerprint(&system_msg),
            "role change must alter digest"
        );

        // Name change:
        let name_alice = vec![json!({"role": "user", "name": "alice", "content": "ping"})];
        let name_bob = vec![json!({"role": "user", "name": "bob", "content": "ping"})];
        assert_ne!(
            fingerprint(&name_alice),
            fingerprint(&name_bob),
            "name change must alter digest"
        );

        // Message order change:
        let order_1 = vec![
            json!({"role": "user", "content": "one"}),
            json!({"role": "user", "content": "two"}),
        ];
        let order_2 = vec![
            json!({"role": "user", "content": "two"}),
            json!({"role": "user", "content": "one"}),
        ];
        assert_ne!(
            fingerprint(&order_1),
            fingerprint(&order_2),
            "order change must alter digest"
        );
    }

    #[test]
    fn test_empty_messages_produces_deterministic_fingerprint() {
        let fp1 = fingerprint(&[]);
        let fp2 = fingerprint(&[]);
        assert_eq!(fp1, fp2);
        assert!(fp1.starts_with("v1:"));
        assert_eq!(fp1.len(), 3 + 64);
    }

    #[test]
    fn test_image_data_uri_and_anthropic_base64_equivalence() {
        let openai_data_uri = vec![json!({
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="}
            }]
        })];
        let anthropic_b64 = vec![json!({
            "role": "user",
            "content": [{
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "iVBORw0KGgoAAAANSUhEUg=="
                }
            }]
        })];
        assert_eq!(
            fingerprint(&openai_data_uri),
            fingerprint(&anthropic_b64),
            "OpenAI data URI and Anthropic base64 source must produce identical fingerprint"
        );

        let changed_b64 = vec![json!({
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,different_payload=="}
            }]
        })];
        assert_ne!(
            fingerprint(&openai_data_uri),
            fingerprint(&changed_b64),
            "Different base64 payload must alter fingerprint"
        );

        let mut conflicting = openai_data_uri.clone();
        conflicting[0]["content"][0]["data"] = json!("separate-payload");
        let mut changed_url = conflicting.clone();
        changed_url[0]["content"][0]["image_url"]["url"] =
            json!("data:image/png;base64,different_payload==");
        assert_ne!(
            fingerprint(&conflicting),
            fingerprint(&changed_url),
            "separate data must not hide a change to the actual image URL"
        );
    }

    #[test]
    fn test_content_thinking_blocks_ignored() {
        let clean = vec![
            json!({"role": "user", "content": "hello"}),
            json!({"role": "assistant", "content": "world"}),
        ];
        let with_thinking_parts = vec![
            json!({"role": "user", "content": "hello"}),
            json!({
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me formulate the answer..."},
                    {"type": "text", "text": "world"}
                ]
            }),
        ];
        assert_eq!(
            fingerprint(&clean),
            fingerprint(&with_thinking_parts),
            "Reasoning/thinking parts in content must be ignored"
        );
    }

    #[test]
    fn test_non_image_object_with_source_field_not_misidentified_as_image() {
        let empty = vec![json!({"role": "user", "content": ""})];
        let obj_source = vec![json!({"role": "user", "content": [{"source": "wikipedia"}]})];
        assert_ne!(
            fingerprint(&empty),
            fingerprint(&obj_source),
            "Non-image object with source field must not be collapsed to empty image"
        );
        let other_source = vec![json!({"role": "user", "content": [{"source": "manual"}]})];
        assert_ne!(fingerprint(&obj_source), fingerprint(&other_source));

        let document = vec![json!({"role": "user", "content": [{
            "type": "document", "source": {"data": "same payload", "title": "first"}
        }]})];
        let mut changed_document = document.clone();
        changed_document[0]["content"][0]["source"]["title"] = json!("second");
        assert_ne!(
            fingerprint(&document),
            fingerprint(&changed_document),
            "unknown content must preserve its complete source semantics"
        );
    }

    #[test]
    fn test_multiple_tool_calls_in_one_turn_rewritten_independently() {
        let original = vec![
            json!({"role": "user", "content": "compute and search"}),
            json!({
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_calc_0",
                        "type": "function",
                        "function": {"name": "calculator", "arguments": "{\"x\": 1}"}
                    },
                    {
                        "id": "call_search_1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{\"q\": \"rust\"}"}
                    }
                ]
            }),
            json!({"role": "tool", "tool_call_id": "call_calc_0", "content": "2"}),
            json!({"role": "tool", "tool_call_id": "call_search_1", "content": "docs"}),
        ];

        // Client rewrites IDs for both calls and results
        let rewritten = vec![
            json!({"role": "user", "content": "compute and search"}),
            json!({
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_calc_0|fc_0",
                        "type": "function",
                        "function": {"name": "calculator", "arguments": "{\"x\": 1}"}
                    },
                    {
                        "id": "call_search_1|fc_1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{\"q\": \"rust\"}"}
                    }
                ]
            }),
            json!({"role": "tool", "tool_call_id": "call_calc_0|fc_0", "content": "2"}),
            json!({"role": "tool", "tool_call_id": "call_search_1|fc_1", "content": "docs"}),
        ];

        assert_eq!(
            fingerprint(&original),
            fingerprint(&rewritten),
            "Multiple tool calls and results rewritten together must produce identical fingerprint"
        );

        // Swapping tool result order must change fingerprint
        let swapped_results = vec![
            json!({"role": "user", "content": "compute and search"}),
            json!({
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_calc_0",
                        "type": "function",
                        "function": {"name": "calculator", "arguments": "{\"x\": 1}"}
                    },
                    {
                        "id": "call_search_1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{\"q\": \"rust\"}"}
                    }
                ]
            }),
            json!({"role": "tool", "tool_call_id": "call_search_1", "content": "docs"}),
            json!({"role": "tool", "tool_call_id": "call_calc_0", "content": "2"}),
        ];

        assert_ne!(
            fingerprint(&original),
            fingerprint(&swapped_results),
            "Swapped tool result order must alter fingerprint"
        );
    }
}
