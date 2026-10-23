//! Responses object and stream event encoding.
//!
//! [`EventEncoder`] is a small state machine. It turns channel progress
//! (reasoning text, answer text, function calls, usage) into the ordered,
//! sequence-numbered events of the Responses streaming protocol, and keeps the
//! response object those events describe. The same encoder serves streaming
//! and collected (non-streaming) requests, so both always agree.

use crate::ids::{new_id, unix_seconds};
use crate::plan::TurnPlan;
use crate::transcript::{Item, Part, Role};
use serde_json::{json, Map, Value};

/// One encoded stream event: its `type` and full JSON payload.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamEvent {
    pub kind: &'static str,
    pub data: Value,
}

/// A function call the channel decided to make.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// Token accounting. Channels report what they know; estimated values are
/// documented by the channel's `exact_usage` capability being disabled.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: u64,
    pub cached_tokens: u64,
    pub output_tokens: u64,
    pub reasoning_tokens: u64,
}

impl Usage {
    pub fn to_json(self) -> Value {
        json!({
            "input_tokens": self.input_tokens,
            "input_tokens_details": {"cached_tokens": self.cached_tokens},
            "output_tokens": self.output_tokens,
            "output_tokens_details": {"reasoning_tokens": self.reasoning_tokens},
            "total_tokens": self.input_tokens + self.output_tokens,
        })
    }
}

#[derive(Debug, Clone)]
enum Slot {
    Reasoning {
        id: String,
        text: String,
        done: bool,
    },
    Message {
        id: String,
        text: String,
        done: bool,
    },
    Call {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
    },
}

impl Slot {
    fn item(&self, completed: bool) -> Value {
        let status = if completed {
            "completed"
        } else {
            "in_progress"
        };
        match self {
            Self::Reasoning { id, text, .. } => json!({
                "id": id,
                "type": "reasoning",
                "status": status,
                "summary": [],
                "content": [{"type": "reasoning_text", "text": text}],
            }),
            Self::Message { id, text, .. } => json!({
                "id": id,
                "type": "message",
                "role": "assistant",
                "status": status,
                "content": [{"type": "output_text", "text": text, "annotations": [], "logprobs": []}],
            }),
            Self::Call {
                id,
                call_id,
                name,
                arguments,
            } => json!({
                "id": id,
                "type": "function_call",
                "status": status,
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
            }),
        }
    }

    fn is_done(&self) -> bool {
        match self {
            Self::Reasoning { done, .. } | Self::Message { done, .. } => *done,
            Self::Call { .. } => true,
        }
    }
}

/// Builds the response object and its stream events for one turn.
#[derive(Debug, Clone)]
pub struct EventEncoder {
    base: Map<String, Value>,
    sequence: u64,
    output: Vec<Slot>,
    reasoning_index: Option<usize>,
    reasoning_open: bool,
    message_index: Option<usize>,
    usage: Option<Usage>,
    completed_at: Option<u64>,
}

impl EventEncoder {
    /// Start a response for a planned turn.
    pub fn new(response_id: &str, model: &str, plan: &TurnPlan, store: bool) -> Self {
        let reasoning = if plan.reasoning_effort.is_some() || plan.echo.reasoning_summary.is_some()
        {
            json!({
                "effort": plan.reasoning_effort.map(|effort| effort.as_str()),
                "summary": plan.echo.reasoning_summary,
            })
        } else {
            Value::Null
        };
        let mut text = json!({"format": {"type": "text"}});
        if let Some(verbosity) = plan.verbosity {
            text["verbosity"] = json!(verbosity.as_str());
        }
        let base = json!({
            "id": response_id,
            "object": "response",
            "created_at": unix_seconds(),
            "model": model,
            "instructions": plan.instructions,
            "previous_response_id": plan.previous_response_id,
            "tools": plan.echo.tools,
            "tool_choice": plan.echo.tool_choice,
            "parallel_tool_calls": plan.parallel_tool_calls,
            "temperature": plan.echo.temperature,
            "top_p": plan.echo.top_p,
            "max_output_tokens": plan.echo.max_output_tokens,
            "truncation": plan.echo.truncation,
            "text": text,
            "reasoning": reasoning,
            "metadata": plan.echo.metadata,
            "store": store,
            "background": false,
            "incomplete_details": null,
        });
        let Value::Object(base) = base else {
            unreachable!("response base is a JSON object")
        };
        Self {
            base,
            sequence: 0,
            output: Vec::new(),
            reasoning_index: None,
            reasoning_open: true,
            message_index: None,
            usage: None,
            completed_at: None,
        }
    }

    pub fn response_id(&self) -> &str {
        self.base
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
    }

    fn emit(&mut self, kind: &'static str, mut data: Value) -> StreamEvent {
        data["type"] = json!(kind);
        data["sequence_number"] = json!(self.sequence);
        self.sequence += 1;
        StreamEvent { kind, data }
    }

    fn snapshot_as(&self, status: &str, error: Value, completed_at: Option<u64>) -> Value {
        let mut object = self.base.clone();
        let completed = status == "completed";
        let output: Vec<Value> = self
            .output
            .iter()
            .map(|slot| slot.item(completed || slot.is_done()))
            .collect();
        object.insert("status".into(), json!(status));
        object.insert("output".into(), Value::Array(output));
        object.insert("error".into(), error);
        object.insert(
            "usage".into(),
            self.usage.map(Usage::to_json).unwrap_or(Value::Null),
        );
        object.insert("completed_at".into(), json!(completed_at));
        Value::Object(object)
    }

    /// The in-progress response object.
    pub fn snapshot(&self) -> Value {
        self.snapshot_as("in_progress", Value::Null, None)
    }

    /// `response.created` and `response.in_progress`.
    pub fn start(&mut self) -> Vec<StreamEvent> {
        let snapshot = self.snapshot();
        vec![
            self.emit("response.created", json!({"response": snapshot})),
            self.emit("response.in_progress", json!({"response": snapshot})),
        ]
    }

    /// Raw reasoning text. Reasoning that arrives after the answer started is
    /// dropped: already-sent items cannot be reordered.
    pub fn reasoning_delta(&mut self, delta: &str) -> Vec<StreamEvent> {
        if delta.is_empty() || !self.reasoning_open {
            return Vec::new();
        }
        let mut events = Vec::new();
        let index = match self.reasoning_index {
            Some(index) => index,
            None => {
                let index = self.output.len();
                let slot = Slot::Reasoning {
                    id: new_id("rs"),
                    text: String::new(),
                    done: false,
                };
                let item = slot.item(false);
                let item_id = item["id"].clone();
                self.output.push(slot);
                self.reasoning_index = Some(index);
                events.push(self.emit(
                    "response.output_item.added",
                    json!({"output_index": index, "item": item}),
                ));
                events.push(self.emit(
                    "response.content_part.added",
                    json!({"item_id": item_id, "output_index": index, "content_index": 0,
                           "part": {"type": "reasoning_text", "text": ""}}),
                ));
                index
            }
        };
        let item_id = match &mut self.output[index] {
            Slot::Reasoning { id, text, .. } => {
                text.push_str(delta);
                id.clone()
            }
            _ => unreachable!("reasoning index points at a reasoning slot"),
        };
        events.push(self.emit(
            "response.reasoning_text.delta",
            json!({"item_id": item_id, "output_index": index, "content_index": 0, "delta": delta}),
        ));
        events
    }

    fn close_reasoning(&mut self) -> Vec<StreamEvent> {
        self.reasoning_open = false;
        let Some(index) = self.reasoning_index else {
            return Vec::new();
        };
        let (item_id, text) = match &mut self.output[index] {
            Slot::Reasoning { id, text, done } if !*done => {
                *done = true;
                (id.clone(), text.clone())
            }
            _ => return Vec::new(),
        };
        let item = self.output[index].item(true);
        vec![
            self.emit(
                "response.reasoning_text.done",
                json!({"item_id": item_id, "output_index": index, "content_index": 0, "text": text}),
            ),
            self.emit(
                "response.content_part.done",
                json!({"item_id": item_id, "output_index": index, "content_index": 0,
                       "part": {"type": "reasoning_text", "text": text}}),
            ),
            self.emit(
                "response.output_item.done",
                json!({"output_index": index, "item": item}),
            ),
        ]
    }

    /// Answer text.
    pub fn text_delta(&mut self, delta: &str) -> Vec<StreamEvent> {
        if delta.is_empty() {
            return Vec::new();
        }
        let mut events = self.close_reasoning();
        let index = match self.message_index {
            Some(index) => index,
            None => {
                let index = self.output.len();
                let slot = Slot::Message {
                    id: new_id("msg"),
                    text: String::new(),
                    done: false,
                };
                let item = slot.item(false);
                let item_id = item["id"].clone();
                self.output.push(slot);
                self.message_index = Some(index);
                events.push(self.emit(
                    "response.output_item.added",
                    json!({"output_index": index, "item": item}),
                ));
                events.push(self.emit(
                    "response.content_part.added",
                    json!({"item_id": item_id, "output_index": index, "content_index": 0,
                           "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []}}),
                ));
                index
            }
        };
        let item_id = match &mut self.output[index] {
            Slot::Message { id, text, .. } => {
                text.push_str(delta);
                id.clone()
            }
            _ => unreachable!("message index points at a message slot"),
        };
        events.push(self.emit(
            "response.output_text.delta",
            json!({"item_id": item_id, "output_index": index, "content_index": 0,
                   "delta": delta, "logprobs": []}),
        ));
        events
    }

    fn close_message(&mut self) -> Vec<StreamEvent> {
        let Some(index) = self.message_index else {
            return Vec::new();
        };
        let (item_id, text) = match &mut self.output[index] {
            Slot::Message { id, text, done } if !*done => {
                *done = true;
                (id.clone(), text.clone())
            }
            _ => return Vec::new(),
        };
        let item = self.output[index].item(true);
        vec![
            self.emit(
                "response.output_text.done",
                json!({"item_id": item_id, "output_index": index, "content_index": 0,
                       "text": text, "logprobs": []}),
            ),
            self.emit(
                "response.content_part.done",
                json!({"item_id": item_id, "output_index": index, "content_index": 0,
                       "part": {"type": "output_text", "text": text, "annotations": [], "logprobs": []}}),
            ),
            self.emit(
                "response.output_item.done",
                json!({"output_index": index, "item": item}),
            ),
        ]
    }

    /// Close open items, append function calls and record usage.
    ///
    /// Does not emit the terminal event: call [`Self::completed_response`],
    /// persist it, then [`Self::complete`]. A failed commit can still end the
    /// stream with [`Self::fail`].
    pub fn finish(&mut self, calls: &[FunctionCall], usage: Usage) -> Vec<StreamEvent> {
        let mut events = self.close_reasoning();
        events.extend(self.close_message());
        for call in calls {
            let index = self.output.len();
            let slot = Slot::Call {
                id: new_id("fc"),
                call_id: new_id("call"),
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            };
            let added = slot.item(false);
            let done = slot.item(true);
            let item_id = done["id"].clone();
            self.output.push(slot);
            events.push(self.emit(
                "response.output_item.added",
                json!({"output_index": index, "item": {
                    "id": added["id"], "type": "function_call", "status": "in_progress",
                    "call_id": added["call_id"], "name": added["name"], "arguments": ""
                }}),
            ));
            events.push(self.emit(
                "response.function_call_arguments.delta",
                json!({"item_id": item_id, "output_index": index, "delta": call.arguments}),
            ));
            events.push(self.emit(
                "response.function_call_arguments.done",
                json!({"item_id": item_id, "output_index": index, "name": call.name,
                       "arguments": call.arguments}),
            ));
            events.push(self.emit(
                "response.output_item.done",
                json!({"output_index": index, "item": done}),
            ));
        }
        self.usage = Some(usage);
        events
    }

    /// The completed response object, to persist before announcing it.
    pub fn completed_response(&mut self) -> Value {
        let completed_at = *self.completed_at.get_or_insert_with(unix_seconds);
        self.snapshot_as("completed", Value::Null, Some(completed_at))
    }

    /// `response.completed`, carrying exactly what [`Self::completed_response`]
    /// returned.
    pub fn complete(&mut self) -> StreamEvent {
        let response = self.completed_response();
        self.emit("response.completed", json!({ "response": response }))
    }

    /// The response object for a turn that stopped at the output budget.
    ///
    /// `status` is `incomplete` with `incomplete_details.reason` set, which is
    /// how OpenAI reports a truncated answer and what clients check before
    /// trusting an output limit.
    pub fn incomplete_response(&mut self, reason: &str) -> Value {
        let completed_at = *self.completed_at.get_or_insert_with(unix_seconds);
        let mut object = self.snapshot_as("incomplete", Value::Null, Some(completed_at));
        if let Value::Object(map) = &mut object {
            map.insert("incomplete_details".into(), json!({ "reason": reason }));
        }
        object
    }

    /// `response.incomplete`, carrying exactly what
    /// [`Self::incomplete_response`] returned.
    pub fn incomplete(&mut self, reason: &str) -> StreamEvent {
        let response = self.incomplete_response(reason);
        self.emit("response.incomplete", json!({ "response": response }))
    }

    /// `response.failed`. `code` is a Responses error code such as
    /// `server_error` or `rate_limit_exceeded`; `message` must be client safe.
    pub fn fail(&mut self, code: &str, message: &str) -> StreamEvent {
        let response = self.snapshot_as("failed", json!({"code": code, "message": message}), None);
        self.emit("response.failed", json!({"response": response}))
    }

    /// Output items as transcript items for `previous_response_id` replay.
    /// Reasoning is omitted, exactly as it is dropped from client input.
    pub fn transcript_items(&self) -> Vec<Item> {
        self.output
            .iter()
            .filter_map(|slot| match slot {
                Slot::Reasoning { .. } => None,
                Slot::Message { text, .. } if text.is_empty() => None,
                Slot::Message { text, .. } => Some(Item::Message {
                    role: Role::Assistant,
                    content: vec![Part::Text { text: text.clone() }],
                }),
                Slot::Call {
                    call_id,
                    name,
                    arguments,
                    ..
                } => Some(Item::FunctionCall {
                    call_id: call_id.clone(),
                    name: name.clone(),
                    arguments: arguments.clone(),
                }),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::CreateRequest;

    fn encoder(body: Value) -> EventEncoder {
        let request: CreateRequest = serde_json::from_value(body).unwrap();
        let plan = TurnPlan::from_request(request).unwrap();
        EventEncoder::new("resp_test", "DEFAULT", &plan, true)
    }

    fn kinds(events: &[StreamEvent]) -> Vec<&'static str> {
        events.iter().map(|event| event.kind).collect()
    }

    #[test]
    fn full_turn_emits_ordered_sequence_numbered_events() {
        let mut encoder = encoder(json!({"input": "hi", "stream": true}));
        let mut events = encoder.start();
        events.extend(encoder.reasoning_delta("think"));
        events.extend(encoder.text_delta("Hel"));
        events.extend(encoder.reasoning_delta("late reasoning is dropped"));
        events.extend(encoder.text_delta("lo"));
        events.extend(encoder.finish(
            &[FunctionCall {
                name: "lookup".into(),
                arguments: "{\"q\":1}".into(),
            }],
            Usage {
                input_tokens: 5,
                cached_tokens: 0,
                output_tokens: 3,
                reasoning_tokens: 1,
            },
        ));
        let completed = encoder.completed_response();
        events.push(encoder.complete());

        assert_eq!(
            kinds(&events),
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.reasoning_text.delta",
                "response.reasoning_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        for (index, event) in events.iter().enumerate() {
            assert_eq!(event.data["sequence_number"], json!(index));
            assert_eq!(event.data["type"], json!(event.kind));
        }
        let output = completed["output"].as_array().unwrap();
        assert_eq!(output.len(), 3);
        assert_eq!(output[0]["type"], "reasoning");
        assert_eq!(output[0]["content"][0]["text"], "think");
        assert_eq!(output[1]["content"][0]["text"], "Hello");
        assert_eq!(output[2]["type"], "function_call");
        assert!(output[2]["call_id"].as_str().unwrap().starts_with("call_"));
        assert_eq!(completed["status"], "completed");
        assert_eq!(completed["usage"]["total_tokens"], 8);
        assert_eq!(
            events.last().unwrap().data["response"],
            completed,
            "the announced response is the persisted one"
        );

        let transcript = encoder.transcript_items();
        assert_eq!(transcript.len(), 2, "reasoning is not replayed");
        assert!(matches!(
            &transcript[0],
            Item::Message {
                role: Role::Assistant,
                ..
            }
        ));
        assert!(matches!(&transcript[1], Item::FunctionCall { name, .. } if name == "lookup"));
    }

    #[test]
    fn failure_keeps_partial_output_marked_in_progress() {
        let mut encoder = encoder(json!({"input": "hi"}));
        encoder.start();
        encoder.text_delta("partial");
        let failed = encoder.fail("server_error", "Backend request failed");
        assert_eq!(failed.kind, "response.failed");
        let response = &failed.data["response"];
        assert_eq!(response["status"], "failed");
        assert_eq!(response["error"]["code"], "server_error");
        assert_eq!(response["output"][0]["status"], "in_progress");
        assert!(response["completed_at"].is_null());
    }

    #[test]
    fn calls_only_turn_has_no_empty_message() {
        let mut encoder = encoder(json!({"input": "hi"}));
        encoder.start();
        encoder.finish(
            &[FunctionCall {
                name: "f".into(),
                arguments: "{}".into(),
            }],
            Usage::default(),
        );
        let response = encoder.completed_response();
        assert_eq!(response["output"].as_array().unwrap().len(), 1);
        assert_eq!(response["output"][0]["type"], "function_call");
    }

    #[test]
    fn response_echoes_request_settings() {
        let encoder = encoder(json!({
            "input": "hi",
            "instructions": "be brief",
            "metadata": {"k": "v"},
            "reasoning": {"effort": "low"},
            "text": {"verbosity": "low"},
            "parallel_tool_calls": false
        }));
        let snapshot = encoder.snapshot();
        assert_eq!(snapshot["object"], "response");
        assert_eq!(snapshot["status"], "in_progress");
        assert_eq!(snapshot["instructions"], "be brief");
        assert_eq!(snapshot["metadata"]["k"], "v");
        assert_eq!(snapshot["reasoning"]["effort"], "low");
        assert_eq!(snapshot["text"]["verbosity"], "low");
        assert_eq!(snapshot["parallel_tool_calls"], false);
        assert_eq!(snapshot["tool_choice"], "auto");
        assert_eq!(snapshot["store"], true);
    }
}
