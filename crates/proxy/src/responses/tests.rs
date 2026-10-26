//! `/v1/responses` handler tests, driven by a scripted fake channel.
//!
//! The fake implements [`ResponsesChannel`] exactly like a real adapter, so
//! these tests pin the channel-independent contract: event order, error
//! timing, ownership, stored chains, capability gating and cancellation.

use super::channel::{
    Checkpoint, ResponsesChannel, TurnError, TurnEvent, TurnOutcome, TurnRequest,
};
use super::{create, delete, derive_thread_id, retrieve, Runtime};
use crate::store::{ThreadHistory, ThreadResume};
use crate::{pool, store, AppState, ProxyError, Settings, Stats};
use axum::body::Bytes;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use futures_util::future::BoxFuture;
use futures_util::{FutureExt, StreamExt};
use responses_core::capability::{
    Capability, CapabilitySet, CapabilityStatus, ProfileDocument, Provider,
};
use responses_core::events::{FunctionCall, Usage};
use responses_core::ids::new_id;
use responses_core::plan::{CreateRequest, ReasoningEffort, TurnPlan};
use responses_core::transcript::{Item, Part, Role};
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

/// An `AppState` for tests: in-memory store, one offline account and the
/// given Responses runtime.
pub(super) async fn test_state(responses: Runtime) -> AppState {
    let store = store::Store::open_memory().unwrap();
    let pool = pool::Pool::connect(
        vec![pool::AccountSpec {
            email: "test@example.invalid".into(),
            password: String::new(),
            token: Some("local-test-token".into()),
        }],
        pool::PoolConfig::default(),
        store.clone(),
    )
    .await
    .unwrap();
    AppState {
        pool: Arc::new(pool),
        settings: Settings {
            thinking_default: true,
            search_enabled: false,
            model_name: "fake-model".into(),
            rate_limit_max_retries: 0,
            rate_limit_backoff_secs: 0.5,
            tool_repair_max_retries: 0,
            request_timeout_secs: 5.0,
            image_max_bytes: 1024,
            heartbeat_interval_secs: 30.0,
            channel_max_output_tokens: HashMap::new(),
            fresh_session_per_request: false,
        },
        threads: Arc::new(Mutex::new(HashMap::new())),
        turns: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        responses,
        store,
        stats: Arc::new(Stats::new()),
    }
}

enum Step {
    Text(&'static str),
    Reasoning(&'static str),
}

enum Ending {
    Complete {
        calls: Vec<(&'static str, &'static str)>,
        checkpoint: bool,
        /// Reason reported when the channel stopped at its output budget.
        truncated: Option<&'static str>,
    },
    Fail {
        status: StatusCode,
        code: &'static str,
        retry_after: Option<u64>,
    },
    /// Never finishes; only a cancelled turn ends.
    Hang,
}

struct Script {
    steps: Vec<Step>,
    ending: Ending,
}

fn answer(text: &'static str) -> Script {
    Script {
        steps: vec![Step::Text(text)],
        ending: Ending::Complete {
            calls: Vec::new(),
            checkpoint: true,
            truncated: None,
        },
    }
}

/// Sets its flag when the turn future is dropped or finishes.
struct DropFlag(Arc<AtomicBool>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

struct FakeChannel {
    capabilities: CapabilitySet,
    scripts: Mutex<VecDeque<Script>>,
    seen: Mutex<Vec<TurnRequest>>,
    dropped: Arc<AtomicBool>,
}

impl FakeChannel {
    fn new(enabled: &[Capability], scripts: Vec<Script>) -> Self {
        let capabilities =
            Capability::ALL
                .into_iter()
                .fold(CapabilitySet::new("fake"), |set, capability| {
                    let status = if enabled.contains(&capability) {
                        CapabilityStatus::supported(Provider::Proxy, "fake")
                    } else {
                        CapabilityStatus::unsupported("fake")
                    };
                    set.with(capability, status)
                });
        Self {
            capabilities,
            scripts: Mutex::new(scripts.into()),
            seen: Mutex::new(Vec::new()),
            dropped: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl ResponsesChannel for FakeChannel {
    fn id(&self) -> &'static str {
        "fake"
    }

    fn capabilities(&self) -> CapabilitySet {
        self.capabilities.clone()
    }

    fn supports_model(&self, model: &str) -> bool {
        model == "fake-model"
    }

    fn run_turn(
        &self,
        _state: AppState,
        turn: TurnRequest,
        events: mpsc::Sender<TurnEvent>,
    ) -> BoxFuture<'static, Result<TurnOutcome, TurnError>> {
        let thread_id = turn.thread_id.clone();
        self.seen.lock().unwrap().push(turn);
        let script = self
            .scripts
            .lock()
            .unwrap()
            .pop_front()
            .expect("a scripted turn");
        let guard = DropFlag(self.dropped.clone());
        async move {
            let _guard = guard;
            for step in script.steps {
                let event = match step {
                    Step::Text(text) => TurnEvent::Text(text.into()),
                    Step::Reasoning(text) => TurnEvent::Reasoning(text.into()),
                };
                if events.send(event).await.is_err() {
                    return Err(TurnError::Cancelled);
                }
            }
            match script.ending {
                Ending::Complete {
                    calls,
                    checkpoint,
                    truncated,
                } => Ok(TurnOutcome {
                    calls: calls
                        .into_iter()
                        .map(|(name, arguments)| FunctionCall {
                            name: name.into(),
                            arguments: arguments.into(),
                        })
                        .collect(),
                    usage: Usage {
                        input_tokens: 3,
                        output_tokens: 2,
                        ..Usage::default()
                    },
                    checkpoint: checkpoint.then(|| Checkpoint {
                        resume: ThreadResume {
                            session_id: format!("remote-{thread_id}"),
                            parent_message_id: Some("1".into()),
                        },
                        history: ThreadHistory {
                            fingerprint: "fake".into(),
                            account_email: "fake@example.invalid".into(),
                            message_count: 1,
                        },
                    }),
                    truncated: truncated.map(str::to_string),
                }),
                Ending::Fail {
                    status,
                    code,
                    retry_after,
                } => {
                    let mut error = ProxyError::new("synthetic failure", status, code);
                    error.retry_after = retry_after;
                    Err(TurnError::Proxy(error))
                }
                Ending::Hang => std::future::pending().await,
            }
        }
        .boxed()
    }
}

const DEFAULTS: [Capability; 10] = [
    Capability::Text,
    Capability::Streaming,
    Capability::NonStreaming,
    Capability::ImageInput,
    Capability::FunctionTools,
    Capability::ToolChoice,
    Capability::Reasoning,
    Capability::Verbosity,
    Capability::StoredResponses,
    Capability::PreviousResponse,
];

struct Harness {
    state: AppState,
    fake: Arc<FakeChannel>,
}

async fn harness(enabled: &[Capability], scripts: Vec<Script>) -> Harness {
    let fake = Arc::new(FakeChannel::new(enabled, scripts));
    let (runtime, warnings) = Runtime::new(fake.clone(), None).unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    Harness {
        state: test_state(runtime).await,
        fake,
    }
}

fn bearer(key: &str) -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert("authorization", format!("Bearer {key}").parse().unwrap());
    headers
}

async fn post(state: &AppState, headers: HeaderMap, body: Value) -> Response {
    create(State(state.clone()), headers, Bytes::from(body.to_string())).await
}

async fn get(state: &AppState, headers: HeaderMap, id: &str, query: &[(&str, &str)]) -> Response {
    let query = query
        .iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect();
    retrieve(
        State(state.clone()),
        headers,
        Path(id.to_string()),
        Query(query),
    )
    .await
}

async fn remove(state: &AppState, headers: HeaderMap, id: &str) -> Response {
    delete(State(state.clone()), headers, Path(id.to_string())).await
}

async fn body_text(response: Response) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .unwrap();
    String::from_utf8(bytes.to_vec()).unwrap()
}

async fn body_json(response: Response) -> Value {
    serde_json::from_str(&body_text(response).await).unwrap()
}

/// `(event, data)` pairs of a Responses SSE body; partial blocks are skipped.
fn sse_events(raw: &str) -> Vec<(String, Value)> {
    raw.split("\n\n")
        .filter_map(|block| {
            let mut kind = None;
            let mut data = None;
            for line in block.lines() {
                if let Some(value) = line.strip_prefix("event:") {
                    kind = Some(value.trim().to_string());
                } else if let Some(value) = line.strip_prefix("data:") {
                    data = serde_json::from_str(value.trim()).ok();
                }
            }
            Some((kind?, data?))
        })
        .collect()
}

fn kinds(events: &[(String, Value)]) -> Vec<&str> {
    events.iter().map(|(kind, _)| kind.as_str()).collect()
}

fn plan_from(body: Value) -> TurnPlan {
    TurnPlan::from_request(serde_json::from_value::<CreateRequest>(body).unwrap()).unwrap()
}

fn text_message(role: Role, text: &str) -> Item {
    Item::Message {
        role,
        content: vec![Part::Text { text: text.into() }],
    }
}

#[tokio::test]
async fn streaming_turn_emits_ordered_events_and_stores_the_completed_response() {
    let h = harness(
        &DEFAULTS,
        vec![Script {
            steps: vec![
                Step::Reasoning("think"),
                Step::Text("Hel"),
                Step::Text("lo"),
            ],
            ending: Ending::Complete {
                calls: Vec::new(),
                checkpoint: true,
                truncated: None,
            },
        }],
    )
    .await;
    let response = post(
        &h.state,
        HeaderMap::new(),
        json!({"model": "fake-model", "input": "hi", "stream": true}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["x-accel-buffering"], "no");
    let events = sse_events(&body_text(response).await);
    assert_eq!(
        kinds(&events),
        [
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
            "response.completed",
        ]
    );
    for (index, (kind, data)) in events.iter().enumerate() {
        assert_eq!(data["type"], kind.as_str());
        assert_eq!(data["sequence_number"], index as u64);
    }
    let completed = &events.last().unwrap().1["response"];
    assert_eq!(completed["status"], "completed");
    assert_eq!(completed["output"][0]["type"], "reasoning");
    assert_eq!(completed["output"][1]["content"][0]["text"], "Hello");
    let id = completed["id"].as_str().unwrap();
    let stored = get(&h.state, HeaderMap::new(), id, &[]).await;
    assert_eq!(stored.status(), StatusCode::OK);
    assert_eq!(&body_json(stored).await, completed);
    assert_eq!(h.state.stats.total.load(Ordering::Relaxed), 1);
    assert_eq!(h.state.stats.ok.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn collected_turn_returns_the_completed_response_with_the_default_model() {
    let h = harness(&DEFAULTS, vec![answer("pong")]).await;
    let response = post(&h.state, HeaderMap::new(), json!({"input": "ping"})).await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["object"], "response");
    assert_eq!(body["model"], "fake-model");
    assert_eq!(body["status"], "completed");
    assert_eq!(body["store"], true);
    assert_eq!(body["output"][0]["type"], "message");
    assert_eq!(body["output"][0]["content"][0]["text"], "pong");
    assert_eq!(body["usage"]["input_tokens"], 3);
    assert_eq!(body["usage"]["total_tokens"], 5);
}

#[tokio::test]
async fn previous_response_replays_the_stored_chain_without_inheriting_instructions() {
    let h = harness(
        &DEFAULTS,
        vec![answer("first answer"), answer("second answer")],
    )
    .await;
    let first = body_json(
        post(
            &h.state,
            HeaderMap::new(),
            json!({"model": "fake-model", "instructions": "be terse", "input": "question one"}),
        )
        .await,
    )
    .await;
    let second = body_json(
        post(
            &h.state,
            HeaderMap::new(),
            json!({"model": "fake-model", "input": "question two", "previous_response_id": first["id"]}),
        )
        .await,
    )
    .await;
    assert_eq!(second["status"], "completed");
    assert_eq!(second["previous_response_id"], first["id"]);
    assert_eq!(second["instructions"], Value::Null);
    let seen = h.fake.seen.lock().unwrap();
    assert_eq!(seen[0].instructions.as_deref(), Some("be terse"));
    assert_eq!(seen[1].instructions, None);
    assert_eq!(seen[1].thread_id, seen[0].thread_id);
    assert_eq!(
        seen[1].transcript,
        [
            text_message(Role::User, "question one"),
            text_message(Role::Assistant, "first answer"),
            text_message(Role::User, "question two"),
        ]
    );
}

#[tokio::test]
async fn function_calls_round_trip_through_previous_response_id() {
    let h = harness(
        &DEFAULTS,
        vec![
            Script {
                steps: Vec::new(),
                ending: Ending::Complete {
                    calls: vec![("lookup", "{\"q\":1}")],
                    checkpoint: true,
                    truncated: None,
                },
            },
            answer("done"),
        ],
    )
    .await;
    let tools = json!([{
        "type": "function",
        "name": "lookup",
        "parameters": {"type": "object", "properties": {"q": {"type": "integer"}}},
    }]);
    let first = body_json(
        post(
            &h.state,
            HeaderMap::new(),
            json!({"model": "fake-model", "input": "look it up", "tools": tools}),
        )
        .await,
    )
    .await;
    let call = &first["output"][0];
    assert_eq!(call["type"], "function_call");
    assert_eq!(call["name"], "lookup");
    assert_eq!(call["arguments"], "{\"q\":1}");
    let call_id = call["call_id"].as_str().unwrap().to_string();
    let second = body_json(
        post(
            &h.state,
            HeaderMap::new(),
            json!({
                "model": "fake-model",
                "previous_response_id": first["id"],
                "input": [{"type": "function_call_output", "call_id": call_id, "output": "42"}],
                "tools": tools,
            }),
        )
        .await,
    )
    .await;
    assert_eq!(second["status"], "completed");
    let seen = h.fake.seen.lock().unwrap();
    assert_eq!(
        seen[1].transcript[1..],
        [
            Item::FunctionCall {
                call_id: call_id.clone(),
                name: "lookup".into(),
                arguments: "{\"q\":1}".into(),
            },
            Item::FunctionCallOutput {
                call_id,
                output: "42".into(),
            },
        ]
    );
}

#[tokio::test]
async fn stored_responses_are_isolated_by_api_key() {
    let h = harness(&DEFAULTS, vec![answer("secret")]).await;
    h.state
        .store
        .insert_api_key("alice", "sk-alice-test")
        .unwrap();
    h.state.store.insert_api_key("bob", "sk-bob-test").unwrap();
    let created = body_json(
        post(
            &h.state,
            bearer("sk-alice-test"),
            json!({"model": "fake-model", "input": "hi"}),
        )
        .await,
    )
    .await;
    let id = created["id"].as_str().unwrap();
    let bob = || bearer("sk-bob-test");
    assert_eq!(
        get(&h.state, bob(), id, &[]).await.status(),
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        remove(&h.state, bob(), id).await.status(),
        StatusCode::NOT_FOUND
    );
    let chained = post(
        &h.state,
        bob(),
        json!({"model": "fake-model", "input": "more", "previous_response_id": id}),
    )
    .await;
    assert_eq!(chained.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        body_json(chained).await["error"]["code"],
        "previous_response_not_found"
    );
    assert_eq!(
        get(&h.state, HeaderMap::new(), id, &[]).await.status(),
        StatusCode::UNAUTHORIZED
    );
    let alice = || bearer("sk-alice-test");
    assert_eq!(
        get(&h.state, alice(), id, &[]).await.status(),
        StatusCode::OK
    );
    assert_eq!(
        body_json(remove(&h.state, alice(), id).await).await,
        json!({"id": id, "object": "response.deleted", "deleted": true})
    );
    assert_eq!(
        get(&h.state, alice(), id, &[]).await.status(),
        StatusCode::NOT_FOUND
    );
}

#[tokio::test]
async fn a_missing_parent_is_a_request_error() {
    let h = harness(&DEFAULTS, Vec::new()).await;
    let response = post(
        &h.state,
        HeaderMap::new(),
        json!({"model": "fake-model", "input": "hi", "previous_response_id": "resp_missing"}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let error = body_json(response).await;
    assert_eq!(error["error"]["code"], "previous_response_not_found");
    assert_eq!(error["error"]["param"], "previous_response_id");
}

#[tokio::test]
async fn store_false_keeps_only_the_channel_checkpoint() {
    let h = harness(&DEFAULTS, vec![answer("ephemeral")]).await;
    let body = body_json(
        post(
            &h.state,
            HeaderMap::new(),
            json!({"model": "fake-model", "input": "hi", "store": false}),
        )
        .await,
    )
    .await;
    assert_eq!(body["store"], false);
    let id = body["id"].as_str().unwrap();
    assert_eq!(
        get(&h.state, HeaderMap::new(), id, &[]).await.status(),
        StatusCode::NOT_FOUND
    );
    let thread_id = h.fake.seen.lock().unwrap()[0].thread_id.clone();
    assert!(h
        .state
        .store
        .get_thread_history(&thread_id)
        .unwrap()
        .is_some());
    assert_eq!(
        h.state
            .store
            .get_thread(&thread_id)
            .unwrap()
            .unwrap()
            .session_id,
        format!("remote-{thread_id}")
    );
}

#[tokio::test]
async fn a_channel_without_stored_responses_never_stores() {
    let h = harness(
        &[Capability::Text, Capability::NonStreaming],
        vec![answer("ok")],
    )
    .await;
    let body = body_json(
        post(
            &h.state,
            HeaderMap::new(),
            json!({"model": "fake-model", "input": "hi"}),
        )
        .await,
    )
    .await;
    assert_eq!(body["store"], false);
    let id = body["id"].as_str().unwrap();
    assert_eq!(
        get(&h.state, HeaderMap::new(), id, &[]).await.status(),
        StatusCode::NOT_FOUND
    );
    let chained = post(
        &h.state,
        HeaderMap::new(),
        json!({"model": "fake-model", "input": "more", "previous_response_id": id}),
    )
    .await;
    assert_eq!(chained.status(), StatusCode::BAD_REQUEST);
    let error = body_json(chained).await;
    assert_eq!(error["error"]["code"], "unsupported_parameter");
    assert_eq!(error["error"]["param"], "previous_response_id");
}

#[tokio::test]
async fn unsupported_capabilities_and_unknown_fields_are_rejected_before_any_turn() {
    let h = harness(&DEFAULTS, Vec::new()).await;
    for (body, param) in [
        (
            json!({"model": "fake-model", "input": "hi", "temperature": 0.2}),
            "temperature",
        ),
        (
            json!({"model": "fake-model", "input": "hi", "background": true}),
            "background",
        ),
        (
            json!({"model": "fake-model", "input": "hi", "tools": [{"type": "web_search"}]}),
            "tools[0]",
        ),
        (
            json!({"model": "fake-model", "input": "hi", "made_up": 1}),
            "made_up",
        ),
    ] {
        let response = post(&h.state, HeaderMap::new(), body).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{param}");
        let error = body_json(response).await;
        assert_eq!(error["error"]["code"], "unsupported_parameter", "{param}");
        assert_eq!(error["error"]["param"], param);
    }
    let response = post(
        &h.state,
        HeaderMap::new(),
        json!({"model": "gpt-unknown", "input": "hi"}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        body_json(response).await["error"]["code"],
        "model_not_found"
    );
    assert!(h.fake.seen.lock().unwrap().is_empty());
    assert_eq!(h.state.stats.errors.load(Ordering::Relaxed), 5);
}

#[tokio::test]
async fn errors_before_the_turn_starts_are_plain_http_errors() {
    let h = harness(
        &DEFAULTS,
        vec![Script {
            steps: Vec::new(),
            ending: Ending::Fail {
                status: StatusCode::TOO_MANY_REQUESTS,
                code: "rate_limit_exceeded",
                retry_after: Some(7),
            },
        }],
    )
    .await;
    let response = post(
        &h.state,
        HeaderMap::new(),
        json!({"model": "fake-model", "input": "hi", "stream": true}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(response.headers()["retry-after"], "7");
    assert_eq!(
        body_json(response).await["error"]["code"],
        "rate_limit_exceeded"
    );
    assert_eq!(h.state.stats.errors.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn errors_after_the_turn_started_end_the_stream_with_response_failed() {
    let h = harness(
        &DEFAULTS,
        vec![Script {
            steps: vec![Step::Text("partial")],
            ending: Ending::Fail {
                status: StatusCode::BAD_GATEWAY,
                code: "backend_error",
                retry_after: None,
            },
        }],
    )
    .await;
    let response = post(
        &h.state,
        HeaderMap::new(),
        json!({"model": "fake-model", "input": "hi", "stream": true}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let events = sse_events(&body_text(response).await);
    let (kind, data) = events.last().unwrap();
    assert_eq!(kind, "response.failed");
    assert_eq!(data["response"]["status"], "failed");
    assert_eq!(data["response"]["error"]["code"], "server_error");
    assert!(!kinds(&events).contains(&"response.completed"));
    let id = events[0].1["response"]["id"].as_str().unwrap();
    assert_eq!(
        get(&h.state, HeaderMap::new(), id, &[]).await.status(),
        StatusCode::NOT_FOUND
    );
}

#[tokio::test]
async fn a_collected_turn_that_fails_after_starting_returns_an_http_error() {
    let h = harness(
        &DEFAULTS,
        vec![Script {
            steps: vec![Step::Text("partial")],
            ending: Ending::Fail {
                status: StatusCode::BAD_GATEWAY,
                code: "backend_error",
                retry_after: None,
            },
        }],
    )
    .await;
    let response = post(
        &h.state,
        HeaderMap::new(),
        json!({"model": "fake-model", "input": "hi"}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(body_json(response).await["error"]["code"], "backend_error");
}

#[tokio::test]
async fn multiple_calls_with_parallel_tool_calls_false_fail_the_turn() {
    let h = harness(
        &DEFAULTS,
        vec![Script {
            steps: Vec::new(),
            ending: Ending::Complete {
                calls: vec![("lookup", "{}"), ("lookup", "{}")],
                checkpoint: true,
                truncated: None,
            },
        }],
    )
    .await;
    let response = post(
        &h.state,
        HeaderMap::new(),
        json!({
            "model": "fake-model",
            "input": "go",
            "parallel_tool_calls": false,
            "tools": [{"type": "function", "name": "lookup"}],
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(
        body_json(response).await["error"]["code"],
        "tool_repair_failed"
    );
}

#[tokio::test]
async fn a_client_disconnect_cancels_the_turn_and_commits_nothing() {
    let h = harness(
        &DEFAULTS,
        vec![Script {
            steps: vec![Step::Text("partial")],
            ending: Ending::Hang,
        }],
    )
    .await;
    let response = post(
        &h.state,
        HeaderMap::new(),
        json!({"model": "fake-model", "input": "hi", "stream": true}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    let mut raw = String::new();
    while !raw.contains("response.output_text.delta") {
        let chunk = body.next().await.expect("stream open").expect("chunk");
        raw.push_str(&String::from_utf8_lossy(&chunk));
    }
    drop(body);
    let thread_id = h.fake.seen.lock().unwrap()[0].thread_id.clone();
    // The turn lock is released only after the adapter future is dropped.
    let _lock = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        h.state.turn_guard(&thread_id),
    )
    .await
    .expect("turn released");
    assert!(h.fake.dropped.load(Ordering::SeqCst));
    let id = sse_events(&raw)[0].1["response"]["id"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(
        get(&h.state, HeaderMap::new(), &id, &[]).await.status(),
        StatusCode::NOT_FOUND
    );
    assert!(h
        .state
        .store
        .get_thread_history(&thread_id)
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn reasoning_is_neither_requested_nor_shown_without_the_capability() {
    let h = harness(
        &[
            Capability::Text,
            Capability::Streaming,
            Capability::StoredResponses,
        ],
        vec![Script {
            steps: vec![Step::Reasoning("hidden thought"), Step::Text("visible")],
            ending: Ending::Complete {
                calls: Vec::new(),
                checkpoint: false,
                truncated: None,
            },
        }],
    )
    .await;
    let response = post(
        &h.state,
        HeaderMap::new(),
        json!({"model": "fake-model", "input": "hi", "stream": true}),
    )
    .await;
    let raw = body_text(response).await;
    let events = sse_events(&raw);
    assert!(kinds(&events)
        .iter()
        .all(|kind| !kind.starts_with("response.reasoning")));
    let completed = &events.last().unwrap().1["response"];
    assert_eq!(completed["output"].as_array().unwrap().len(), 1);
    assert_eq!(completed["output"][0]["content"][0]["text"], "visible");
    assert!(!raw.contains("hidden thought"));
    assert_eq!(
        h.fake.seen.lock().unwrap()[0].reasoning_effort,
        Some(ReasoningEffort::None)
    );
}

#[tokio::test]
async fn a_replayed_image_is_gated_by_the_current_capabilities() {
    let mut h = harness(&DEFAULTS, vec![answer("I see it")]).await;
    let first = body_json(
        post(
            &h.state,
            HeaderMap::new(),
            json!({"model": "fake-model", "input": [{"role": "user", "content": [
                {"type": "input_text", "text": "look"},
                {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
            ]}]}),
        )
        .await,
    )
    .await;
    assert_eq!(first["status"], "completed");
    let narrowed: Vec<Capability> = DEFAULTS
        .into_iter()
        .filter(|capability| *capability != Capability::ImageInput)
        .collect();
    let (runtime, _) =
        Runtime::new(Arc::new(FakeChannel::new(&narrowed, Vec::new())), None).unwrap();
    h.state.responses = runtime;
    let chained = post(
        &h.state,
        HeaderMap::new(),
        json!({"model": "fake-model", "input": "and now?", "previous_response_id": first["id"]}),
    )
    .await;
    assert_eq!(chained.status(), StatusCode::BAD_REQUEST);
    let error = body_json(chained).await;
    assert_eq!(error["error"]["code"], "unsupported_parameter");
    assert_eq!(error["error"]["param"], "previous_response_id");
}

#[tokio::test]
async fn retrieving_with_stream_true_is_unsupported() {
    let h = harness(&DEFAULTS, Vec::new()).await;
    let response = get(
        &h.state,
        HeaderMap::new(),
        "resp_any",
        &[("stream", "true")],
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        body_json(response).await["error"]["code"],
        "unsupported_parameter"
    );
}

#[test]
fn profiles_only_narrow_the_channel_ceiling() {
    let fake = Arc::new(FakeChannel::new(&DEFAULTS, Vec::new()));
    let profile = ProfileDocument::parse(
        json!({
            "format": "responses-capability-profile",
            "schema_version": 1,
            "channel": "fake",
            "generated_by": "test",
            "capabilities": {
                "text": {"support": "supported", "provider": "upstream"},
                "streaming": {"support": "supported", "provider": "proxy"},
                "non_streaming": {"support": "supported", "provider": "proxy"},
                "stored_responses": {"support": "supported", "provider": "proxy"},
                "previous_response": {"support": "unsupported"},
                "web_search": {"support": "supported", "provider": "upstream"},
                "from_the_future": {"support": "supported"},
            },
        })
        .to_string()
        .as_bytes(),
    )
    .unwrap();
    let (runtime, warnings) = Runtime::new(fake.clone(), Some(&profile)).unwrap();
    assert_eq!(
        runtime.capabilities.enabled_ids(),
        ["text", "streaming", "non_streaming", "stored_responses"]
    );
    assert!(warnings
        .iter()
        .any(|warning| warning.contains("from_the_future")));
    assert!(warnings
        .iter()
        .any(|warning| warning.contains("web_search")));

    let mut other_channel = profile.clone();
    other_channel.channel = "deepseek_web".into();
    assert!(Runtime::new(fake, Some(&other_channel)).is_err());
}

#[test]
fn runtime_load_fails_closed_on_a_missing_profile() {
    let fake: Arc<dyn ResponsesChannel> = Arc::new(FakeChannel::new(&DEFAULTS, Vec::new()));
    let missing = std::env::temp_dir().join(format!("{}-missing-profile.json", new_id("test")));
    assert!(Runtime::load(fake.clone(), Some(missing)).is_err());
    assert!(Runtime::load(fake, None).is_ok());
}

#[test]
fn thread_ids_prefer_session_identity_and_stay_owner_scoped() {
    let plan = plan_from(json!({"input": "same first message", "prompt_cache_key": "cache-a"}));
    let mut session = HeaderMap::new();
    session.insert("x-session-affinity", "session-a".parse().unwrap());
    let by_session = derive_thread_id(&session, "owner-a", &plan);
    assert!(by_session.starts_with("responses_"));
    assert_eq!(by_session, derive_thread_id(&session, "owner-a", &plan));
    assert_ne!(by_session, derive_thread_id(&session, "owner-b", &plan));
    let by_cache_key = derive_thread_id(&HeaderMap::new(), "owner-a", &plan);
    assert_ne!(by_session, by_cache_key);
    let mut explicit = session.clone();
    explicit.insert("x-thread-id", "thread-1".parse().unwrap());
    assert_ne!(derive_thread_id(&explicit, "owner-a", &plan), by_session);

    let first_user = plan_from(json!({"input": "same first message"}));
    assert_eq!(
        derive_thread_id(&HeaderMap::new(), "owner-a", &first_user),
        derive_thread_id(&HeaderMap::new(), "owner-a", &first_user)
    );
    let no_user = plan_from(json!({"input": [{"role": "developer", "content": "rules only"}]}));
    assert_ne!(
        derive_thread_id(&HeaderMap::new(), "owner-a", &no_user),
        derive_thread_id(&HeaderMap::new(), "owner-a", &no_user)
    );
}

/// A channel budget makes `max_output_tokens` valid even when the channel does
/// not declare the capability, because the proxy enforces it locally.
#[tokio::test]
async fn channel_budget_accepts_a_request_within_the_limit() {
    let mut h = harness(&DEFAULTS, vec![answer("ok")]).await;
    h.state
        .settings
        .channel_max_output_tokens
        .insert("fake".into(), 8192);
    let response = post(
        &h.state,
        bearer(""),
        json!({"input": "hi", "stream": false, "max_output_tokens": 4096}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["status"], "completed");
}

/// Above the configured budget the request is rejected, naming the limit.
#[tokio::test]
async fn request_above_the_channel_budget_is_rejected() {
    let mut h = harness(&DEFAULTS, vec![answer("unused")]).await;
    h.state
        .settings
        .channel_max_output_tokens
        .insert("fake".into(), 1024);
    let response = post(
        &h.state,
        bearer(""),
        json!({"input": "hi", "stream": false, "max_output_tokens": 2048}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_json(response).await;
    assert_eq!(body["error"]["code"], "max_output_tokens_exceeded");
    assert_eq!(body["error"]["param"], "max_output_tokens");
    assert!(body["error"]["message"].as_str().unwrap().contains("1024"));
}

/// Without a channel budget the field stays fail-closed.
#[tokio::test]
async fn request_without_a_channel_budget_stays_fail_closed() {
    let h = harness(&DEFAULTS, vec![answer("unused")]).await;
    let response = post(
        &h.state,
        bearer(""),
        json!({"input": "hi", "stream": false, "max_output_tokens": 4096}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_json(response).await;
    assert_eq!(body["error"]["code"], "unsupported_parameter");
}

/// A truncated turn ends as `incomplete` and still keeps its checkpoint, so
/// the next request continues the remote session instead of replaying.
#[tokio::test]
async fn truncated_turn_reports_incomplete_and_keeps_the_checkpoint() {
    let script = Script {
        steps: vec![Step::Text("partial answer")],
        ending: Ending::Complete {
            calls: Vec::new(),
            checkpoint: true,
            truncated: Some("max_output_tokens"),
        },
    };
    let mut h = harness(&DEFAULTS, vec![script]).await;
    h.state
        .settings
        .channel_max_output_tokens
        .insert("fake".into(), 8192);
    let response = post(
        &h.state,
        bearer(""),
        json!({"input": "hi", "stream": false, "max_output_tokens": 16}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["status"], "incomplete");
    assert_eq!(body["incomplete_details"]["reason"], "max_output_tokens");
    let stored = h.fake.seen.lock().unwrap()[0].max_output_tokens;
    assert_eq!(stored, Some(16), "the budget reaches the channel");
    assert!(
        h.state
            .store
            .get_thread_history(&h.fake.seen.lock().unwrap()[0].thread_id)
            .unwrap()
            .is_some(),
        "a truncated turn still checkpoints"
    );
}

/// The budget is forwarded unchanged when the channel has no limit configured
/// but the capability is enabled.
#[tokio::test]
async fn channel_receives_the_requested_budget() {
    let mut enabled = DEFAULTS.to_vec();
    enabled.push(Capability::MaxOutputTokens);
    let h = harness(&enabled, vec![answer("ok")]).await;
    let response = post(
        &h.state,
        bearer(""),
        json!({"input": "hi", "stream": false, "max_output_tokens": 64}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(h.fake.seen.lock().unwrap()[0].max_output_tokens, Some(64));
}

/// Magpie's classifier asks at temperature 0. In fresh-session decision mode
/// that is accepted and dropped; the channel still declares no sampling
/// support, so every other mode keeps failing closed.
#[tokio::test]
async fn fresh_session_mode_accepts_sampling_it_cannot_forward() {
    let mut h = harness(&DEFAULTS, vec![answer("ok")]).await;
    h.state.settings.fresh_session_per_request = true;
    let accepted = post(
        &h.state,
        bearer(""),
        json!({"input": "decide", "stream": false, "temperature": 0}),
    )
    .await;
    assert_eq!(accepted.status(), StatusCode::OK);
    let body = body_json(accepted).await;
    assert_eq!(body["status"], "completed");

    let mut h = harness(&DEFAULTS, vec![answer("ok")]).await;
    h.state.settings.fresh_session_per_request = false;
    let refused = post(
        &h.state,
        bearer(""),
        json!({"input": "decide", "stream": false, "temperature": 0}),
    )
    .await;
    assert_eq!(refused.status(), StatusCode::BAD_REQUEST);
    let body = body_json(refused).await;
    assert_eq!(body["error"]["code"], "unsupported_parameter");
    assert_eq!(body["error"]["param"], "temperature");
}
