//! Neutral conversation transcript.
//!
//! Client input is normalized once into [`Item`]s. Stored responses keep this
//! form, `previous_response_id` replays it, and every channel adapter reads it.
//! Echoed `reasoning` items are dropped on input: no channel can replay
//! another model's private reasoning, and keeping them would only make a
//! transcript differ from the one the channel produced.

use crate::capability::Capability;
use crate::plan::{Needs, RequestError};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One conversation item.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Item {
    Message {
        role: Role,
        content: Vec<Part>,
    },
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    FunctionCallOutput {
        call_id: String,
        output: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    Developer,
    User,
    Assistant,
}

impl Role {
    fn parse(role: &str) -> Option<Self> {
        match role {
            "system" => Some(Self::System),
            "developer" => Some(Self::Developer),
            "user" => Some(Self::User),
            "assistant" => Some(Self::Assistant),
            _ => None,
        }
    }
}

/// One message content part.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Part {
    Text {
        text: String,
    },
    Image {
        url: String,
        detail: String,
    },
    File {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file_data: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file_url: Option<String>,
    },
}

/// Text parts joined by newlines; non-text parts are skipped.
pub fn text_of(parts: &[Part]) -> String {
    parts
        .iter()
        .filter_map(|part| match part {
            Part::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Normalize the request `input` field.
pub(crate) fn normalize_input(input: &Value, needs: &mut Needs) -> Result<Vec<Item>, RequestError> {
    match input {
        Value::String(text) => {
            if text.is_empty() {
                return Err(RequestError::invalid("input", "input must not be empty"));
            }
            Ok(vec![Item::Message {
                role: Role::User,
                content: vec![Part::Text { text: text.clone() }],
            }])
        }
        Value::Array(raw) => {
            let mut items = Vec::with_capacity(raw.len());
            for (index, value) in raw.iter().enumerate() {
                if let Some(item) = normalize_item(value, &format!("input[{index}]"), needs)? {
                    items.push(item);
                }
            }
            if items.is_empty() {
                return Err(RequestError::invalid(
                    "input",
                    "input must contain at least one message or function_call_output",
                ));
            }
            Ok(items)
        }
        _ => Err(RequestError::invalid(
            "input",
            "input must be a string or an array of input items",
        )),
    }
}

fn required_str<'a>(
    object: &'a serde_json::Map<String, Value>,
    field: &str,
    at: &str,
) -> Result<&'a str, RequestError> {
    match object.get(field) {
        Some(Value::String(value)) if !value.is_empty() => Ok(value),
        _ => Err(RequestError::invalid(
            &format!("{at}.{field}"),
            &format!("{field} must be a non-empty string"),
        )),
    }
}

fn normalize_item(
    value: &Value,
    at: &str,
    needs: &mut Needs,
) -> Result<Option<Item>, RequestError> {
    let Some(object) = value.as_object() else {
        return Err(RequestError::invalid(at, "input items must be objects"));
    };
    let kind = match object.get("type") {
        None | Some(Value::Null) => "message",
        Some(Value::String(kind)) => kind.as_str(),
        Some(_) => {
            return Err(RequestError::invalid(
                &format!("{at}.type"),
                "type must be a string",
            ))
        }
    };
    match kind {
        "message" => {
            let role_text = required_str(object, "role", at)?;
            let Some(role) = Role::parse(role_text) else {
                return Err(RequestError::invalid(
                    &format!("{at}.role"),
                    "role must be user, assistant, system or developer",
                ));
            };
            let content = object.get("content").ok_or_else(|| {
                RequestError::invalid(&format!("{at}.content"), "message content is required")
            })?;
            let parts = normalize_content(role, content, &format!("{at}.content"), needs)?;
            if parts.is_empty() {
                if role == Role::User {
                    return Err(RequestError::invalid(
                        &format!("{at}.content"),
                        "user message content must not be empty",
                    ));
                }
                return Ok(None);
            }
            Ok(Some(Item::Message {
                role,
                content: parts,
            }))
        }
        "function_call" => {
            let call_id = required_str(object, "call_id", at)?.to_string();
            let name = required_str(object, "name", at)?.to_string();
            let arguments = match object.get("arguments") {
                None | Some(Value::Null) => "{}".to_string(),
                Some(Value::String(arguments)) => arguments.clone(),
                Some(_) => {
                    return Err(RequestError::invalid(
                        &format!("{at}.arguments"),
                        "arguments must be a JSON-encoded string",
                    ))
                }
            };
            Ok(Some(Item::FunctionCall {
                call_id,
                name,
                arguments,
            }))
        }
        "function_call_output" => {
            let call_id = required_str(object, "call_id", at)?.to_string();
            let output = match object.get("output") {
                Some(Value::String(output)) => output.clone(),
                Some(Value::Array(parts)) => {
                    let mut texts = Vec::new();
                    for part in parts {
                        match part.get("type").and_then(Value::as_str) {
                            Some("input_text") | Some("output_text") | Some("text") => texts.push(
                                part.get("text")
                                    .and_then(Value::as_str)
                                    .unwrap_or_default()
                                    .to_string(),
                            ),
                            _ => {
                                return Err(RequestError::unsupported(
                                    &format!("{at}.output"),
                                    "function_call_output supports text parts only",
                                ))
                            }
                        }
                    }
                    texts.join("\n")
                }
                _ => {
                    return Err(RequestError::invalid(
                        &format!("{at}.output"),
                        "output must be a string or an array of text parts",
                    ))
                }
            };
            Ok(Some(Item::FunctionCallOutput { call_id, output }))
        }
        // A client returning a previous output array echoes reasoning items.
        // They cannot be replayed into any channel, so they are dropped.
        "reasoning" => Ok(None),
        other => Err(RequestError::unsupported(
            &format!("{at}.type"),
            &format!("input item type '{other}' is not supported"),
        )),
    }
}

fn normalize_content(
    role: Role,
    content: &Value,
    at: &str,
    needs: &mut Needs,
) -> Result<Vec<Part>, RequestError> {
    match content {
        Value::String(text) => Ok(if text.is_empty() {
            Vec::new()
        } else {
            vec![Part::Text { text: text.clone() }]
        }),
        Value::Array(raw) => {
            let mut parts = Vec::with_capacity(raw.len());
            for (index, part) in raw.iter().enumerate() {
                let at = format!("{at}[{index}]");
                let kind = part.get("type").and_then(Value::as_str).ok_or_else(|| {
                    RequestError::invalid(&format!("{at}.type"), "content parts require a type")
                })?;
                match kind {
                    "input_text" | "output_text" | "text" => {
                        let text = part.get("text").and_then(Value::as_str).ok_or_else(|| {
                            RequestError::invalid(&format!("{at}.text"), "text must be a string")
                        })?;
                        if !text.is_empty() {
                            parts.push(Part::Text {
                                text: text.to_string(),
                            });
                        }
                    }
                    "refusal" => {
                        let text = part
                            .get("refusal")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        if !text.is_empty() {
                            parts.push(Part::Text {
                                text: text.to_string(),
                            });
                        }
                    }
                    "input_image" => {
                        if role != Role::User {
                            return Err(RequestError::invalid(
                                &at,
                                "images are only accepted in user messages",
                            ));
                        }
                        if part.get("file_id").is_some_and(|id| !id.is_null()) {
                            return Err(RequestError::unsupported(
                                &format!("{at}.file_id"),
                                "image file_id references are not supported; send image_url",
                            ));
                        }
                        let url = match part.get("image_url") {
                            Some(Value::String(url)) if !url.is_empty() => url.clone(),
                            Some(Value::Object(object)) => object
                                .get("url")
                                .and_then(Value::as_str)
                                .filter(|url| !url.is_empty())
                                .map(str::to_string)
                                .ok_or_else(|| {
                                    RequestError::invalid(
                                        &format!("{at}.image_url"),
                                        "image_url must be a non-empty string",
                                    )
                                })?,
                            _ => {
                                return Err(RequestError::invalid(
                                    &format!("{at}.image_url"),
                                    "image_url must be a non-empty string",
                                ))
                            }
                        };
                        let detail = match part.get("detail") {
                            None | Some(Value::Null) => "auto".to_string(),
                            Some(Value::String(detail))
                                if matches!(
                                    detail.as_str(),
                                    "auto" | "low" | "high" | "original"
                                ) =>
                            {
                                detail.clone()
                            }
                            Some(_) => {
                                return Err(RequestError::invalid(
                                    &format!("{at}.detail"),
                                    "detail must be auto, low, high or original",
                                ))
                            }
                        };
                        needs.add(Capability::ImageInput, &at);
                        parts.push(Part::Image { url, detail });
                    }
                    "input_file" => {
                        if part.get("file_id").is_some_and(|id| !id.is_null()) {
                            return Err(RequestError::unsupported(
                                &format!("{at}.file_id"),
                                "file_id references are not supported",
                            ));
                        }
                        let text_field = |name: &str| {
                            part.get(name)
                                .and_then(Value::as_str)
                                .filter(|value| !value.is_empty())
                                .map(str::to_string)
                        };
                        let file_data = text_field("file_data");
                        let file_url = text_field("file_url");
                        if file_data.is_none() && file_url.is_none() {
                            return Err(RequestError::invalid(
                                &at,
                                "input_file requires file_data or file_url",
                            ));
                        }
                        needs.add(Capability::FileInput, &at);
                        parts.push(Part::File {
                            filename: text_field("filename"),
                            file_data,
                            file_url,
                        });
                    }
                    other => {
                        return Err(RequestError::unsupported(
                            &format!("{at}.type"),
                            &format!("content part type '{other}' is not supported"),
                        ))
                    }
                }
            }
            Ok(parts)
        }
        _ => Err(RequestError::invalid(
            at,
            "content must be a string or an array of content parts",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn normalize(value: Value) -> (Result<Vec<Item>, RequestError>, Needs) {
        let mut needs = Needs::default();
        let result = normalize_input(&value, &mut needs);
        (result, needs)
    }

    #[test]
    fn string_input_is_one_user_message() {
        let (items, _) = normalize(json!("hello"));
        assert_eq!(
            items.unwrap(),
            vec![Item::Message {
                role: Role::User,
                content: vec![Part::Text {
                    text: "hello".into()
                }]
            }]
        );
    }

    #[test]
    fn echoed_output_items_normalize_to_the_same_shape() {
        let (items, _) = normalize(json!([
            {"role": "user", "content": "q"},
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {"type": "message", "role": "assistant", "id": "msg_1", "status": "completed",
             "content": [{"type": "output_text", "text": "a", "annotations": []}]},
            {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "f", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": [{"type": "input_text", "text": "r"}]}
        ]));
        let items = items.unwrap();
        assert_eq!(items.len(), 4, "reasoning is dropped");
        assert_eq!(
            items[1],
            Item::Message {
                role: Role::Assistant,
                content: vec![Part::Text { text: "a".into() }]
            }
        );
        assert_eq!(
            items[3],
            Item::FunctionCallOutput {
                call_id: "call_1".into(),
                output: "r".into()
            }
        );
    }

    #[test]
    fn images_require_the_image_capability_and_a_user_role() {
        let (items, needs) = normalize(json!([{"role": "user", "content": [
            {"type": "input_text", "text": "see"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}
        ]}]));
        assert!(items.is_ok());
        assert!(needs
            .0
            .iter()
            .any(|need| need.capability == Capability::ImageInput));
        let (assistant_image, _) = normalize(json!([{"role": "assistant", "content": [
            {"type": "input_image", "image_url": "https://example.com/a.png"}
        ]}]));
        assert!(assistant_image.is_err());
    }

    #[test]
    fn unsupported_items_and_parts_fail_explicitly() {
        for value in [
            json!([{"type": "item_reference", "id": "x"}]),
            json!([{"type": "web_search_call", "id": "x"}]),
            json!([{"role": "user", "content": [{"type": "input_audio"}]}]),
            json!([{"role": "user", "content": [{"type": "input_image", "file_id": "file_1"}]}]),
            json!([{"type": "function_call_output", "call_id": "c", "output": [{"type": "input_image"}]}]),
        ] {
            let (result, _) = normalize(value);
            assert_eq!(result.unwrap_err().code, "unsupported_parameter");
        }
    }

    #[test]
    fn invalid_shapes_are_rejected() {
        for value in [
            json!(""),
            json!([]),
            json!(42),
            json!([{"role": "tool", "content": "x"}]),
            json!([{"role": "user", "content": ""}]),
            json!([{"type": "function_call", "name": "f"}]),
            json!([{"type": "function_call", "call_id": "c", "name": "f", "arguments": {}}]),
        ] {
            let (result, _) = normalize(value.clone());
            assert!(result.is_err(), "{value} must be rejected");
        }
    }

    #[test]
    fn text_parts_join_with_newlines() {
        assert_eq!(
            text_of(&[
                Part::Text { text: "a".into() },
                Part::Image {
                    url: "u".into(),
                    detail: "auto".into()
                },
                Part::Text { text: "b".into() },
            ]),
            "a\nb"
        );
    }
}
