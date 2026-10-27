use super::*;
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    thread,
    time::{Duration, Instant},
};

struct Reply {
    status: u16,
    content_type: &'static str,
    body: Vec<u8>,
}
impl Reply {
    fn json(value: Value) -> Self {
        Self {
            status: 200,
            content_type: "application/json",
            body: serde_json::to_vec(&value).unwrap(),
        }
    }
    fn html(body: &str) -> Self {
        Self {
            status: 200,
            content_type: "text/html; charset=utf-8",
            body: body.as_bytes().to_vec(),
        }
    }
    fn sse(body: &str) -> Self {
        Self {
            status: 200,
            content_type: "text/event-stream",
            body: body.as_bytes().to_vec(),
        }
    }
    fn bytes(content_type: &'static str, body: Vec<u8>) -> Self {
        Self {
            status: 200,
            content_type,
            body,
        }
    }
}
#[derive(Debug)]
struct Captured {
    line: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}
fn mock_server(replies: Vec<Reply>) -> (String, thread::JoinHandle<Vec<Captured>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    let worker = thread::spawn(move || {
        let mut captured = Vec::new();
        for reply in replies {
            let deadline = Instant::now() + Duration::from_secs(10);
            let (mut stream, _) = loop {
                match listener.accept() {
                    Ok(pair) => break pair,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        if Instant::now() >= deadline {
                            panic!("mock server timed out waiting for request");
                        }
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(e) => panic!("mock listener failed: {e}"),
                }
            };
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut bytes = Vec::new();
            let mut chunk = [0u8; 4096];
            let mut header_end = None;
            let mut content_length = 0usize;
            loop {
                let n = stream.read(&mut chunk).unwrap();
                if n == 0 {
                    break;
                }
                bytes.extend_from_slice(&chunk[..n]);
                if header_end.is_none() {
                    if let Some(pos) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                        header_end = Some(pos + 4);
                        let head = String::from_utf8_lossy(&bytes[..pos]);
                        content_length = head
                            .lines()
                            .skip(1)
                            .find_map(|l| {
                                l.split_once(':')
                                    .filter(|(k, _)| k.eq_ignore_ascii_case("content-length"))
                                    .and_then(|(_, v)| v.trim().parse().ok())
                            })
                            .unwrap_or(0);
                    }
                }
                if header_end.is_some_and(|end| bytes.len() >= end + content_length) {
                    break;
                }
            }
            let end = header_end.expect("request headers");
            let head = String::from_utf8_lossy(&bytes[..end - 4]);
            let mut lines = head.lines();
            let line = lines.next().unwrap_or_default().to_owned();
            let headers = lines
                .filter_map(|l| l.split_once(':'))
                .map(|(k, v)| (k.to_ascii_lowercase(), v.trim().to_owned()))
                .collect();
            let request_body = bytes[end..end + content_length].to_vec();
            captured.push(Captured {
                line,
                headers,
                body: request_body,
            });
            let reason = match reply.status {
                200 => "OK",
                201 => "Created",
                204 => "No Content",
                302 => "Found",
                401 => "Unauthorized",
                429 => "Too Many Requests",
                _ => "Mock",
            };
            let response = format!("HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",reply.status,reason,reply.content_type,reply.body.len());
            stream.write_all(response.as_bytes()).unwrap();
            stream.write_all(&reply.body).unwrap();
            stream.flush().unwrap();
        }
        captured
    });
    (base, worker)
}
fn client(base: String, token: &str) -> ChatGptClient {
    ChatGptClient::new(ClientConfig {
        base_url: base,
        access_token: token.into(),
        proxy_url: None,
        timeout: Duration::from_secs(10),
    })
    .unwrap()
}
fn message() -> Value {
    json!({"id":"message-1","author":{"role":"user"},"content":{"content_type":"text","parts":["hello upstream"]},"metadata":{}})
}
fn common_replies(conversation: Reply) -> Vec<Reply> {
    vec![
        Reply::html(
            "<html data-build=\"c/build/_\"><script src=\"/c/build/_app.js\"></script></html>",
        ),
        Reply::json(json!({"prepare_token":"prepare-token"})),
        Reply::json(json!({"token":"requirements-token","so_token":"so-token"})),
        conversation,
    ]
}

#[tokio::test]
async fn authenticated_bootstrap_sentinel_and_normal_conversation_flow() {
    let (base, worker) = mock_server(common_replies(Reply::sse("data: [DONE]\n\n")));
    let api = client(base, "bearer-value");
    let response = api
        .conversation(&ConversationRequest {
            messages: vec![message()],
            model: "gpt-5".into(),
            thinking_effort: Some("high".into()),
            system_hints: vec![],
        })
        .await
        .unwrap();
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
    assert_eq!(
        response.bytes().await.unwrap().as_ref(),
        b"data: [DONE]\n\n"
    );
    let captured = worker.join().unwrap();
    assert_eq!(
        captured
            .iter()
            .map(|r| r
                .line
                .split_whitespace()
                .nth(1)
                .unwrap()
                .split('?')
                .next()
                .unwrap())
            .collect::<Vec<_>>(),
        vec![
            "/",
            "/backend-api/sentinel/chat-requirements/prepare",
            "/backend-api/sentinel/chat-requirements/finalize",
            "/backend-api/conversation"
        ]
    );
    assert_eq!(
        captured[1].headers.get("authorization").map(String::as_str),
        Some("Bearer bearer-value")
    );
    assert_eq!(
        captured[1]
            .headers
            .get("x-openai-target-path")
            .map(String::as_str),
        Some("/backend-api/sentinel/chat-requirements/prepare")
    );
    let p: Value = serde_json::from_slice(&captured[1].body).unwrap();
    assert!(p["p"].as_str().unwrap().starts_with("gAAAAAC"));
    let final_body: Value = serde_json::from_slice(&captured[2].body).unwrap();
    assert_eq!(final_body["prepare_token"], "prepare-token");
    assert_eq!(
        captured[3]
            .headers
            .get("openai-sentinel-chat-requirements-token")
            .map(String::as_str),
        Some("requirements-token")
    );
    assert_eq!(
        captured[3]
            .headers
            .get("openai-sentinel-so-token")
            .map(String::as_str),
        Some("so-token")
    );
    let body: Value = serde_json::from_slice(&captured[3].body).unwrap();
    assert_eq!(body["messages"][0]["content"]["parts"][0], "hello upstream");
    assert_eq!(body["model"], "gpt-5");
    assert_eq!(body["thinking_effort"], "high");
    assert_eq!(body["history_and_training_disabled"], true);
    assert!(body["parent_message_id"].as_str().unwrap().contains('-'));
}

#[tokio::test]
async fn hosted_workflow_uses_prepare_and_conversation_routes() {
    let mut replies = common_replies(Reply::sse("data: [DONE]\n\n"));
    replies.insert(3, Reply::json(json!({"conduit_token":"conduit-value"})));
    let (base, worker) = mock_server(replies);
    let api = client(base, "secret");
    let response = api
        .conversation(&ConversationRequest {
            messages: vec![message()],
            model: "gpt-5-5".into(),
            thinking_effort: None,
            system_hints: vec!["search".into()],
        })
        .await
        .unwrap();
    drop(response);
    let captured = worker.join().unwrap();
    let paths: Vec<_> = captured
        .iter()
        .map(|r| r.line.split_whitespace().nth(1).unwrap().to_owned())
        .collect();
    assert_eq!(
        paths,
        vec![
            "/",
            "/backend-api/sentinel/chat-requirements/prepare",
            "/backend-api/sentinel/chat-requirements/finalize",
            "/backend-api/f/conversation/prepare",
            "/backend-api/f/conversation"
        ]
    );
    let prep: Value = serde_json::from_slice(&captured[3].body).unwrap();
    assert_eq!(
        prep["partial_query"]["content"]["parts"][0],
        "hello upstream"
    );
    assert_eq!(
        captured[3]
            .headers
            .get("x-conduit-token")
            .map(String::as_str),
        Some("no-token")
    );
    let run: Value = serde_json::from_slice(&captured[4].body).unwrap();
    assert_eq!(run["history_and_training_disabled"], true);
    assert_eq!(run["force_use_search"], true);
    assert_eq!(
        captured[4]
            .headers
            .get("x-conduit-token")
            .map(String::as_str),
        Some("conduit-value")
    );
}

#[tokio::test]
async fn upload_and_image_result_flow_excludes_input_asset_and_never_auths_blob_url() {
    let mut image = b"\x89PNG\r\n\x1a\n\0\0\0\x0dIHDR".to_vec();
    image.extend_from_slice(&1u32.to_be_bytes());
    image.extend_from_slice(&2u32.to_be_bytes());
    let (listener, addr) = {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let a = l.local_addr().unwrap();
        (l, a)
    };
    let base = format!("http://{addr}");
    let document = json!({"mapping":{
        "user-node":{"message":{"author":{"role":"user"},"content":{"parts":[{"content_type":"image_asset_pointer","asset_pointer":"file-service://input_file"}]}}},
        "assistant-node":{"message":{"id":"assistant-message","author":{"role":"assistant"},"metadata":{"async_task_type":"image_gen"},"content":{"parts":[{"content_type":"image_asset_pointer","asset_pointer":"file-service://output_file"}]}}}
    }});
    let image_for_reply = image.clone();
    let base_for_worker = base.clone();
    let worker = thread::spawn(move || {
        let replies = vec![
            Reply::json(
                json!({"file_id":"input_file","upload_url":format!("{base_for_worker}/blob?sig=signature-value")}),
            ),
            Reply::bytes("text/plain", Vec::new()),
            Reply::json(json!({})),
            Reply::json(document.clone()),
            Reply::json(document),
            Reply::json(
                json!({"download_url":format!("{base_for_worker}/image-data?sig=download-signature")}),
            ),
            Reply::bytes("image/png", image_for_reply),
        ];
        let mut captured = Vec::new();
        for reply in replies {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut data = Vec::new();
            let mut buf = [0u8; 4096];
            let mut end = None;
            let mut len = 0usize;
            loop {
                let n = stream.read(&mut buf).unwrap();
                if n == 0 {
                    break;
                }
                data.extend_from_slice(&buf[..n]);
                if end.is_none() {
                    if let Some(i) = data.windows(4).position(|w| w == b"\r\n\r\n") {
                        end = Some(i + 4);
                        let h = String::from_utf8_lossy(&data[..i]);
                        len = h
                            .lines()
                            .skip(1)
                            .find_map(|l| {
                                l.split_once(':')
                                    .filter(|(k, _)| k.eq_ignore_ascii_case("content-length"))
                                    .and_then(|(_, v)| v.trim().parse().ok())
                            })
                            .unwrap_or(0)
                    }
                }
                if end.is_some_and(|e| data.len() >= e + len) {
                    break;
                }
            }
            let e = end.unwrap();
            let h = String::from_utf8_lossy(&data[..e - 4]);
            let mut lines = h.lines();
            let line = lines.next().unwrap().to_owned();
            let headers = lines
                .filter_map(|l| l.split_once(':'))
                .map(|(k, v)| (k.to_ascii_lowercase(), v.trim().to_owned()))
                .collect();
            captured.push(Captured {
                line,
                headers,
                body: data[e..e + len].to_vec(),
            });
            let response=format!("HTTP/1.1 {} OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",reply.status,reply.content_type,reply.body.len());
            stream.write_all(response.as_bytes()).unwrap();
            stream.write_all(&reply.body).unwrap();
        }
        captured
    });
    let api = client(base, "image-bearer");
    let uploaded = api
        .upload_image(image.clone(), "image/png", "input.png")
        .await
        .unwrap();
    assert_eq!(uploaded["asset_pointer"], "file-service://input_file");
    let result = api
        .image_results("conv123", &["input_file".into(), "output_file".into()], &[])
        .await
        .unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(B64.decode(&result[0]).unwrap(), image);
    let captured = worker.join().unwrap();
    assert_eq!(
        captured[0].headers.get("authorization").map(String::as_str),
        Some("Bearer image-bearer")
    );
    assert_eq!(
        captured[1].line.split_whitespace().nth(1),
        Some("/blob?sig=signature-value")
    );
    assert!(!captured[1].headers.contains_key("authorization"));
    assert_eq!(
        captured[5].line.split_whitespace().nth(1),
        Some("/backend-api/files/output_file/download")
    );
    assert_eq!(
        captured[6].line.split_whitespace().nth(1),
        Some("/image-data?sig=download-signature")
    );
    assert!(!captured[6].headers.contains_key("authorization"));
}

#[tokio::test]
async fn sandbox_artifacts_use_interpreter_download_route() {
    let doc = json!({"mapping":{"assistant-node":{"message":{"id":"msg_1","author":{"role":"assistant"},"content":{"parts":["sandbox:/mnt/data/report.pptx"]}}}}});
    let (base, worker) = mock_server(vec![
        Reply::json(doc),
        Reply::json(
            json!({"download_url":"https://files.example.test/report.pptx?signature=opaque"}),
        ),
    ]);
    let api = client(base, "token");
    let url = api
        .artifact_url("conv_1", "file_1", Some("/mnt/data/report.pptx"))
        .await
        .unwrap();
    assert_eq!(
        url,
        "https://files.example.test/report.pptx?signature=opaque"
    );
    let requests = worker.join().unwrap();
    assert_eq!(
        requests[1]
            .line
            .split_whitespace()
            .nth(1)
            .unwrap()
            .split('?')
            .next(),
        Some("/backend-api/conversation/conv_1/interpreter/download")
    );
    assert!(requests[1].line.contains("message_id=msg_1"));
    assert_eq!(
        requests[1]
            .headers
            .get("x-openai-target-route")
            .map(String::as_str),
        Some("/backend-api/conversation/{conversation_id}/interpreter/download")
    );
}

#[tokio::test]
async fn search_poll_returns_completed_answer_and_sources() {
    let doc = json!({"mapping":{"answer":{"message":{"author":{"role":"assistant"},"create_time":2,"metadata":{"finish_details":{"type":"finished_successfully"}},"content":{"parts":["answer text",{"title":"source","url":"https://example.test"}]}}}}});
    let (base, worker) = mock_server(vec![Reply::json(doc)]);
    let api = client(base, "token");
    let result = api.poll_search_result("conv_1").await.unwrap();
    assert_eq!(result["answer"], "answer text");
    assert_eq!(result["sources"][0]["url"], "https://example.test");
    assert_eq!(
        worker.join().unwrap()[0].line.split_whitespace().nth(1),
        Some("/backend-api/conversation/conv_1")
    );
}

#[test]
fn configuration_and_error_debug_are_redacted_and_classified() {
    let config = ClientConfig {
        access_token: "access-secret".into(),
        proxy_url: Some(
            "http://alice:proxy-secret@proxy.example:8080?credential=query-secret".into(),
        ),
        ..Default::default()
    };
    let debug = format!("{config:?}");
    assert!(!debug.contains("access-secret"));
    assert!(!debug.contains("proxy-secret"));
    assert!(!debug.contains("query-secret"));
    assert!(ChatGptClient::new(ClientConfig {
        base_url: "https://u:p@example.test".into(),
        ..Default::default()
    })
    .is_err());
    assert!(ChatGptClient::new(ClientConfig {
        base_url: "https://example.test?secret=x".into(),
        ..Default::default()
    })
    .is_err());
    assert!(ChatGptClient::new(ClientConfig {
        base_url: "https://example.test/#fragment".into(),
        ..Default::default()
    })
    .is_err());
    let auth = Error::http(401, None);
    assert!(auth.is_auth_error());
    assert_eq!(auth.status_code(), Some(401));
    let limited = Error::http(429, Some(12));
    assert!(limited.is_rate_limited());
    assert_eq!(limited.retry_after(), Some(12));
    let text = format!("{} {:?}", auth, auth);
    assert!(!text.contains("credential"));
    assert!(!text.contains("secret"));
}
