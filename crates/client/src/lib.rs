//! DeepSeek private-API client: auth, PoW header, streaming completions.
//!
//! Port of `aiodeepseek`'s `_AuthClient` + `_BaseClient` + `_ChatClient`.
//! The proxy is the only consumer; there is deliberately no `Conversation`
//! abstraction (that was the main source of monkey-patching in Python).

pub mod device_ids;
pub mod images;

pub use device_ids::get_device_id;

use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use thiserror::Error;

pub const BASE_URL: &str = "https://chat.deepseek.com";
pub const COMPLETION_PATH: &str = "/api/v0/chat/completion";
pub const UPLOAD_PATH: &str = "/api/v0/file/upload_file";
const FETCH_FILES_PATH: &str = "/api/v0/file/fetch_files";

/// Upload statuses the server reports while it is still ingesting the file.
const UPLOAD_PENDING_STATUSES: [&str; 4] = ["PENDING", "PROCESSING", "UPLOADING", "PARSING"];
/// Seconds between `/file/fetch_files` polls (Python: `_UPLOAD_POLL_INTERVAL`).
const UPLOAD_POLL_INTERVAL_SECS: f64 = 2.0;
/// Give up polling after this long (Python: `_UPLOAD_POLL_TIMEOUT`).
const UPLOAD_POLL_TIMEOUT_SECS: f64 = 30.0;

/// An image the backend has accepted and that a completion can reference via
/// `ref_file_ids`.
#[derive(Debug, Clone)]
pub struct UploadedImage {
    pub file_id: String,
    pub size_bytes: usize,
    pub width: u32,
    pub height: u32,
}
const LOGIN_PATH: &str = "/api/v0/users/login";
const SESSION_CREATE_PATH: &str = "/api/v0/chat_session/create";
const POW_CHALLENGE_PATH: &str = "/api/v0/chat/create_pow_challenge";

/// Headers the shipped iOS client sends. Kept byte-identical to Python so the
/// server sees the same client identity.
pub fn base_headers() -> reqwest::header::HeaderMap {
    use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
    let mut h = HeaderMap::new();
    let pairs = [
        ("Content-Type", "application/json"),
        ("Accept", "application/json"),
        (
            "User-Agent",
            "DeepSeek/2 CFNetwork/1568.100.1 Darwin/24.0.0",
        ),
        ("x-client-platform", "ios"),
        ("x-client-version", "2.0.4"),
        ("x-client-bundle-id", "com.deepseek.chat"),
        ("x-client-locale", "en_US"),
        ("x-client-timezone-offset", "3600"),
    ];
    for (k, v) in pairs {
        h.insert(
            HeaderName::from_static(k.to_ascii_lowercase().leak()),
            HeaderValue::from_static(v),
        );
    }
    h
}

#[derive(Debug, Error)]
pub enum ClientError {
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),
    #[error("api error {code}: {message}")]
    Api { code: i64, message: String },
    #[error("unexpected response shape: {0}")]
    Shape(String),
    #[error("pow: no nonce found within difficulty {difficulty}")]
    PowUnsolved { difficulty: u64 },
    #[error("invalid challenge: {0}")]
    BadChallenge(String),
}

/// A proof-of-work challenge from `/api/v0/chat/create_pow_challenge`.
#[derive(Debug, Clone, Deserialize)]
pub struct Challenge {
    pub salt: String,
    #[serde(deserialize_with = "de_string")]
    pub expire_at: String,
    #[serde(deserialize_with = "de_u64")]
    pub difficulty: u64,
    pub challenge: String,
    #[serde(default)]
    pub algorithm: String,
    #[serde(default)]
    pub signature: String,
}

fn de_string<'de, D: serde::Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    // The server sends this as a JSON string today but it is a number in some
    // responses; accept both.
    let v = Value::deserialize(d)?;
    match v {
        Value::String(s) => Ok(s),
        Value::Number(n) => Ok(n.to_string()),
        other => Err(serde::de::Error::custom(format!(
            "expected string or number, got {other}"
        ))),
    }
}

fn de_u64<'de, D: serde::Deserializer<'de>>(d: D) -> Result<u64, D::Error> {
    let v = Value::deserialize(d)?;
    match v {
        Value::Number(n) => n
            .as_u64()
            .ok_or_else(|| serde::de::Error::custom("difficulty not a u64")),
        Value::String(s) => s
            .parse()
            .map_err(|_| serde::de::Error::custom("difficulty not numeric")),
        other => Err(serde::de::Error::custom(format!(
            "expected number, got {other}"
        ))),
    }
}

/// Body sent to `/api/v0/chat/completion`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CompletionBody {
    pub chat_session_id: String,
    pub parent_message_id: Option<i64>,
    pub prompt: String,
    pub ref_file_ids: Vec<String>,
    pub thinking_enabled: bool,
    pub search_enabled: bool,
    pub audio_id: Option<String>,
    pub preempt: bool,
    pub model_type: String,
}

impl CompletionBody {
    /// Construct a completion body with default model type (`"default"`).
    pub fn new(session_id: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self::new_with_model_type(session_id, prompt, "default")
    }

    /// Construct a completion body with an explicit upstream model type.
    pub fn new_with_model_type(
        session_id: impl Into<String>,
        prompt: impl Into<String>,
        model_type: impl Into<String>,
    ) -> Self {
        Self {
            chat_session_id: session_id.into(),
            parent_message_id: None,
            prompt: prompt.into(),
            ref_file_ids: Vec::new(),
            thinking_enabled: false,
            search_enabled: false,
            audio_id: None,
            preempt: false,
            model_type: model_type.into(),
        }
    }

    /// Alias for [`CompletionBody::new_with_model_type`].
    pub fn new_with_model(
        session_id: impl Into<String>,
        prompt: impl Into<String>,
        model_type: impl Into<String>,
    ) -> Self {
        Self::new_with_model_type(session_id, prompt, model_type)
    }

    /// Builder-style setter to configure or override the upstream model type.
    pub fn with_model_type(mut self, model_type: impl Into<String>) -> Self {
        self.model_type = model_type.into();
        self
    }

    /// Alias for [`CompletionBody::with_model_type`].
    pub fn with_model(self, model_type: impl Into<String>) -> Self {
        self.with_model_type(model_type)
    }

    /// In-place setter to configure or override the upstream model type.
    pub fn set_model_type(&mut self, model_type: impl Into<String>) -> &mut Self {
        self.model_type = model_type.into();
        self
    }
}

/// Authenticated DeepSeek client.
#[derive(Clone)]
pub struct Client {
    http: reqwest::Client,
    token: String,
}

impl Client {
    /// Build a client around an existing bearer token (skips the login round trip).
    pub fn from_token(token: String) -> Self {
        Self::new(token)
    }

    fn new(token: String) -> Self {
        Self {
            http: reqwest::Client::builder()
                .danger_accept_invalid_certs(false)
                .build()
                .expect("reqwest client"),
            token,
        }
    }

    pub fn token(&self) -> &str {
        &self.token
    }

    fn authed_headers(&self) -> reqwest::header::HeaderMap {
        let mut h = base_headers();
        h.insert(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_str(&format!("Bearer {}", self.token))
                .expect("token header"),
        );
        h
    }

    /// `POST /api/v0/users/login` -> bearer token.
    ///
    /// `device_id` must come from [`get_device_id`]; the server rejects
    /// fabricated values with `RISK_DEVICE_DETECTED`.
    pub async fn login(email: &str, password: &str, device_id: &str) -> Result<Self, ClientError> {
        let http = reqwest::Client::new();
        let body = json!({
            "email": email,
            "password": password,
            "device_id": device_id,
            "os": "ios",
        });
        let resp = http
            .post(format!("{BASE_URL}{LOGIN_PATH}"))
            .headers(base_headers())
            .json(&body)
            .send()
            .await?;
        let data: Value = resp.json().await?;
        let token = data
            .pointer("/data/biz_data/user/token")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ClientError::Shape(format!("login: {data}")))?
            .to_string();
        Ok(Self::new(token))
    }

    /// `POST /api/v0/chat_session/create` -> session id.
    pub async fn create_chat_session(&self) -> Result<String, ClientError> {
        let resp = self
            .http
            .post(format!("{BASE_URL}{SESSION_CREATE_PATH}"))
            .headers(self.authed_headers())
            .json(&json!({ "character_id": 1 }))
            .send()
            .await?;
        let data: Value = resp.json().await?;
        data.pointer("/data/biz_data/chat_session/id")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .ok_or_else(|| ClientError::Shape(format!("create_session: {data}")))
    }

    /// Fetch a fresh PoW challenge for `target_path`.
    pub async fn pow_challenge(&self, target_path: &str) -> Result<Challenge, ClientError> {
        let resp = self
            .http
            .post(format!("{BASE_URL}{POW_CHALLENGE_PATH}"))
            .headers(self.authed_headers())
            .json(&json!({ "target_path": target_path }))
            .send()
            .await?;
        let data: Value = resp.json().await?;
        let ch = data
            .pointer("/data/biz_data/challenge")
            .or_else(|| data.pointer("/data/biz_data"))
            .ok_or_else(|| ClientError::Shape(format!("pow_challenge: {data}")))?;
        serde_json::from_value(ch.clone())
            .map_err(|e| ClientError::BadChallenge(format!("{e}: {ch}")))
    }

    /// Solve a challenge and return the base64 `X-DS-PoW-Response` value.
    ///
    /// The JSON key order matches Python (`algorithm, challenge, salt,
    /// signature, answer, target_path`) and uses compact separators, so the
    /// header is byte-identical.
    pub async fn pow_header(&self, target_path: &str) -> Result<String, ClientError> {
        let ch = self.pow_challenge(target_path).await?;
        let base = pow::pow_base(&ch.salt, &ch.expire_at);
        let nonce = pow::solve_hex(&base, &ch.challenge, ch.difficulty).ok_or(
            ClientError::PowUnsolved {
                difficulty: ch.difficulty,
            },
        )?;

        let algorithm = if ch.algorithm.is_empty() {
            "DeepSeekHashV1"
        } else {
            ch.algorithm.as_str()
        };
        let payload = format!(
            concat!(
                "{{\"algorithm\":{},\"challenge\":{},\"salt\":{},",
                "\"signature\":{},\"answer\":{},\"target_path\":{}}}"
            ),
            serde_json::to_string(algorithm)?,
            serde_json::to_string(&ch.challenge)?,
            serde_json::to_string(&ch.salt)?,
            serde_json::to_string(&ch.signature)?,
            nonce,
            serde_json::to_string(target_path)?,
        );
        Ok(base64::engine::general_purpose::STANDARD.encode(payload))
    }

    /// Poll `/file/fetch_files` until the upload leaves a pending status.
    ///
    /// Port of `_FilesClient._wait_for_file`: `CONTENT_EMPTY` and any
    /// `*ERROR*`/`*FAIL*` status are hard failures, and the deadline is wall
    /// clock, not per-poll.
    async fn wait_for_file(&self, file_id: &str) -> Result<String, ClientError> {
        let deadline = std::time::Instant::now()
            + std::time::Duration::from_secs_f64(UPLOAD_POLL_TIMEOUT_SECS);
        loop {
            let resp = self
                .http
                .get(format!("{BASE_URL}{FETCH_FILES_PATH}"))
                .headers(self.authed_headers())
                .query(&[("file_ids", file_id)])
                .send()
                .await?;
            let body: Value = resp.json().await?;
            let file = body
                .pointer("/data/biz_data/files")
                .and_then(|f| f.as_array())
                .and_then(|f| f.first())
                .ok_or_else(|| {
                    ClientError::Shape(format!("fetch_files: empty list for {file_id}"))
                })?;
            let status = file
                .get("status")
                .and_then(|s| s.as_str())
                .unwrap_or_default()
                .to_string();

            if !UPLOAD_PENDING_STATUSES.contains(&status.as_str()) {
                if status == "CONTENT_EMPTY" {
                    return Err(ClientError::Api {
                        code: 0,
                        message: format!(
                            "uploaded file {file_id} rejected by DeepSeek: CONTENT_EMPTY"
                        ),
                    });
                }
                if status.to_ascii_uppercase().contains("ERROR")
                    || status.to_ascii_uppercase().contains("FAIL")
                {
                    return Err(ClientError::Shape(format!(
                        "file processing failed: status={status} meta={file}"
                    )));
                }
                return Ok(status);
            }

            if std::time::Instant::now() >= deadline {
                return Err(ClientError::Shape(format!(
                    "timed out waiting for file {file_id} (last status={status})"
                )));
            }
            tokio::time::sleep(std::time::Duration::from_secs_f64(
                UPLOAD_POLL_INTERVAL_SECS,
            ))
            .await;
        }
    }

    /// Upload PNG/JPEG bytes and return the id a completion can reference.
    ///
    /// Port of `_FilesClient._upload_image`: a PoW challenge scoped to the
    /// upload path, a multipart body, then polling until the file is ready.
    pub async fn upload_image(
        &self,
        data: &[u8],
        model: &str,
    ) -> Result<UploadedImage, ClientError> {
        let meta = images::meta(data).map_err(ClientError::Shape)?;
        let filename = if meta.mime == "image/png" {
            "image.png"
        } else {
            "image.jpg"
        };
        let pow = self.pow_header(UPLOAD_PATH).await?;

        let mut headers = self.authed_headers();
        // reqwest sets the multipart boundary itself; the base header would
        // override it and the server would reject the body.
        headers.remove(reqwest::header::CONTENT_TYPE);
        for (name, value) in [
            ("x-ds-pow-response", pow),
            ("x-thinking-enabled", "0".to_string()),
            ("x-file-size", data.len().to_string()),
            ("x-model-type", model.to_string()),
        ] {
            headers.insert(
                reqwest::header::HeaderName::from_bytes(name.as_bytes())
                    .expect("static header name"),
                reqwest::header::HeaderValue::from_str(&value).map_err(|e| {
                    ClientError::Shape(format!("header {name}: {e}"))
                })?,
            );
        }

        let part = reqwest::multipart::Part::bytes(data.to_vec())
            .file_name(filename.to_string())
            .mime_str(meta.mime)
            .map_err(|e| ClientError::Shape(format!("multipart: {e}")))?;
        let form = reqwest::multipart::Form::new().part("file", part);

        let resp = self
            .http
            .post(format!("{BASE_URL}{UPLOAD_PATH}"))
            .headers(headers)
            .multipart(form)
            .send()
            .await?;
        let body: Value = resp.json().await?;
        let file_id = body
            .pointer("/data/biz_data/id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ClientError::Shape(format!("upload: {body}")))?
            .to_string();
        let status = body
            .pointer("/data/biz_data/status")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();

        if UPLOAD_PENDING_STATUSES.contains(&status.as_str()) {
            self.wait_for_file(&file_id).await?;
        }

        Ok(UploadedImage {
            file_id,
            size_bytes: data.len(),
            width: meta.width,
            height: meta.height,
        })
    }

    /// Start a streaming completion. Caller drives the byte stream.
    pub async fn stream_completion(
        &self,
        body: &CompletionBody,
    ) -> Result<reqwest::Response, ClientError> {
        let pow = self.pow_header(COMPLETION_PATH).await?;
        let mut headers = self.authed_headers();
        headers.insert(
            reqwest::header::ACCEPT,
            reqwest::header::HeaderValue::from_static("text/event-stream"),
        );
        headers.insert(
            reqwest::header::HeaderName::from_static("x-ds-pow-response"),
            reqwest::header::HeaderValue::from_str(&pow).expect("pow header"),
        );
        let resp = self
            .http
            .post(format!("{BASE_URL}{COMPLETION_PATH}"))
            .headers(headers)
            .json(body)
            .send()
            .await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(ClientError::Api {
                code: status.as_u16() as i64,
                message: text.chars().take(400).collect(),
            });
        }
        Ok(resp)
    }
}

impl From<serde_json::Error> for ClientError {
    fn from(e: serde_json::Error) -> Self {
        ClientError::Shape(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn challenge_accepts_string_and_number_expire_at() {
        let a: Challenge = serde_json::from_value(json!({
            "salt": "s", "expire_at": "1792333031301", "difficulty": 144000,
            "challenge": "ab", "algorithm": "DeepSeekHashV1", "signature": "x"
        }))
        .unwrap();
        assert_eq!(a.expire_at, "1792333031301");
        assert_eq!(a.difficulty, 144000);

        let b: Challenge = serde_json::from_value(json!({
            "salt": "s", "expire_at": 1792333031301u64, "difficulty": "144000",
            "challenge": "ab"
        }))
        .unwrap();
        assert_eq!(b.expire_at, "1792333031301");
        assert_eq!(b.difficulty, 144000);
    }

    #[test]
    fn completion_body_default_values_and_serialization() {
        let body = CompletionBody::new("sess_test_1", "test prompt");
        assert_eq!(body.model_type, "default");
        assert_eq!(body.chat_session_id, "sess_test_1");
        assert_eq!(body.prompt, "test prompt");
        assert_eq!(body.parent_message_id, None);
        assert!(body.ref_file_ids.is_empty());
        assert!(!body.thinking_enabled);
        assert!(!body.search_enabled);
        assert_eq!(body.audio_id, None);
        assert!(!body.preempt);

        let val = serde_json::to_value(&body).unwrap();
        assert_eq!(
            val,
            json!({
                "chat_session_id": "sess_test_1",
                "parent_message_id": null,
                "prompt": "test prompt",
                "ref_file_ids": [],
                "thinking_enabled": false,
                "search_enabled": false,
                "audio_id": null,
                "preempt": false,
                "model_type": "default"
            })
        );
    }

    #[test]
    fn completion_body_model_type_override_preserves_other_defaults() {
        let default_body = CompletionBody::new("sess_test_2", "hello world");
        let default_val = serde_json::to_value(&default_body).unwrap();

        let builder_body =
            CompletionBody::new("sess_test_2", "hello world").with_model_type("deepseek_code");
        let ctor_body =
            CompletionBody::new_with_model_type("sess_test_2", "hello world", "deepseek_code");
        let mut setter_body = CompletionBody::new("sess_test_2", "hello world");
        setter_body.set_model_type("deepseek_code");

        let alias_builder =
            CompletionBody::new("sess_test_2", "hello world").with_model("deepseek_code");
        let alias_ctor =
            CompletionBody::new_with_model("sess_test_2", "hello world", "deepseek_code");

        assert_eq!(builder_body, ctor_body);
        assert_eq!(builder_body, setter_body);
        assert_eq!(builder_body, alias_builder);
        assert_eq!(builder_body, alias_ctor);

        let overridden_val = serde_json::to_value(&builder_body).unwrap();
        assert_eq!(overridden_val["model_type"], "deepseek_code");

        // Verify all fields other than model_type remain identical to default
        let default_obj = default_val.as_object().unwrap();
        let overridden_obj = overridden_val.as_object().unwrap();

        assert_eq!(default_obj.len(), overridden_obj.len());
        for (k, v) in default_obj {
            if k == "model_type" {
                assert_eq!(v, "default");
                assert_eq!(overridden_obj[k], "deepseek_code");
            } else {
                assert_eq!(
                    &overridden_obj[k], v,
                    "field '{k}' unexpectedly changed when overriding model_type"
                );
            }
        }
    }

    #[test]
    fn completion_body_supports_proxy_model_mappings() {
        for model in ["default", "deepseek_chat", "deepseek_reasoner", "deepseek_code", "vision"] {
            let body = CompletionBody::new_with_model_type("session_xyz", "prompt_abc", model);
            let val = serde_json::to_value(&body).unwrap();
            assert_eq!(val["model_type"], model);
            assert_eq!(val["chat_session_id"], "session_xyz");
            assert_eq!(val["prompt"], "prompt_abc");
            assert_eq!(val["parent_message_id"], Value::Null);
            assert_eq!(val["ref_file_ids"], json!([]));
            assert_eq!(val["thinking_enabled"], false);
            assert_eq!(val["search_enabled"], false);
            assert_eq!(val["audio_id"], Value::Null);
            assert_eq!(val["preempt"], false);
        }
    }
}
