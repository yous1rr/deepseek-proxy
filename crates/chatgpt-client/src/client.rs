use super::{new_id, pow, turnstile, ClientConfig, ConversationRequest, Error, ErrorKind};
use reqwest::{
    header::{HeaderMap, HeaderName, HeaderValue, ACCEPT, AUTHORIZATION, CONTENT_TYPE},
    Method, RequestBuilder, Response,
};
use serde_json::{json, Value};
use std::{
    fmt,
    time::{Duration, Instant},
};
use url::Url;

const MAX_JSON_BODY: usize = 2 * 1024 * 1024;
const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;
const MAX_IMAGE_DIMENSION: u32 = 100_000;
const DEFAULT_CLIENT_VERSION: &str = "prod-a194cd50d4416d3c0b47c740f206b12ce60f5887";
const DEFAULT_CLIENT_BUILD_NUMBER: &str = "6708908";
const USER_AGENT: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 Edg/143.0.0.0";
const SEC_CH_UA: &str =
    "\"Microsoft Edge\";v=\"143\", \"Chromium\";v=\"143\", \"Not A(Brand\";v=\"24\"";
type ExtraHeader = (&'static str, String);

#[derive(Clone)]
pub struct ChatGptClient {
    pub(super) config: ClientConfig,
    pub(super) base: Url,
    pub(super) http: reqwest::Client,
    pub(super) user_agent: String,
    pub(super) device_id: String,
    pub(super) session_id: String,
}
impl fmt::Debug for ChatGptClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChatGptClient")
            .field("base_url", &self.base.origin().ascii_serialization())
            .field("access_token", &"[REDACTED]")
            .field(
                "proxy_url",
                &self.config.proxy_url.as_ref().map(|_| "[REDACTED]"),
            )
            .finish()
    }
}

#[derive(Debug)]
pub(super) struct Requirements {
    pub token: String,
    pub proof: String,
    pub turnstile: String,
    pub so_token: String,
}

impl ChatGptClient {
    /// Construct a client, rejecting credentials, queries, and fragments in the base URL.
    pub fn new(config: ClientConfig) -> Result<Self, Error> {
        if config.timeout.is_zero()
            || config.base_url.len() > 2048
            || config.base_url.trim() != config.base_url
            || config.base_url.chars().any(char::is_control)
            || config.access_token.len() > 8192
            || config.access_token.chars().any(char::is_control)
            || config
                .proxy_url
                .as_ref()
                .is_some_and(|p| p.len() > 4096 || p.chars().any(char::is_control))
        {
            return Err(Error::new(ErrorKind::Configuration));
        }
        let base =
            Url::parse(&config.base_url).map_err(|_| Error::new(ErrorKind::Configuration))?;
        let authority = config
            .base_url
            .split_once("://")
            .map(|(_, tail)| tail.split(['/', '?', '#']).next().unwrap_or_default())
            .unwrap_or_default();
        if !matches!(base.scheme(), "http" | "https")
            || base.host_str().is_none()
            || authority.contains('@')
            || !base.username().is_empty()
            || base.password().is_some()
            || base.query().is_some()
            || base.fragment().is_some()
        {
            return Err(Error::new(ErrorKind::Configuration));
        }
        let mut builder = reqwest::Client::builder()
            .timeout(config.timeout)
            .redirect(reqwest::redirect::Policy::none());
        if let Some(proxy) = config.proxy_url.as_deref() {
            builder = builder.proxy(
                reqwest::Proxy::all(proxy).map_err(|_| Error::new(ErrorKind::Configuration))?,
            );
        }
        let http = builder
            .build()
            .map_err(|_| Error::new(ErrorKind::Configuration))?;
        Ok(Self {
            config,
            base,
            http,
            user_agent: USER_AGENT.into(),
            device_id: new_id(),
            session_id: new_id(),
        })
    }

    pub(super) fn url(&self, path: &str) -> Result<Url, Error> {
        if !path.starts_with('/')
            || path.len() > 2048
            || path.chars().any(char::is_control)
            || path.contains(['?', '#'])
            || path.split('/').any(|piece| piece == "." || piece == "..")
        {
            return Err(Error::new(ErrorKind::InvalidRequest));
        }
        self.base
            .join(path)
            .map_err(|_| Error::new(ErrorKind::InvalidRequest))
    }

    pub(super) fn headers(&self, path: &str, extra: &[ExtraHeader]) -> Result<HeaderMap, Error> {
        if path.len() > 2048 || path.chars().any(char::is_control) {
            return Err(Error::new(ErrorKind::InvalidRequest));
        }
        let origin = self.base.origin().ascii_serialization();
        let pairs = [
            ("user-agent", self.user_agent.clone()), ("origin", origin.clone()), ("referer", format!("{origin}/")),
            ("accept-language", "zh-CN,zh;q=0.9,en;q=0.8,en-US;q=0.7".into()), ("cache-control", "no-cache".into()),
            ("pragma", "no-cache".into()), ("priority", "u=1, i".into()), ("sec-ch-ua", SEC_CH_UA.into()),
            ("sec-ch-ua-arch", "\"x86\"".into()), ("sec-ch-ua-bitness", "\"64\"".into()),
            ("sec-ch-ua-full-version", "\"143.0.3650.96\"".into()),
            ("sec-ch-ua-full-version-list", "\"Microsoft Edge\";v=\"143.0.3650.96\", \"Chromium\";v=\"143.0.7499.147\", \"Not A(Brand\";v=\"24.0.0.0\"".into()),
            ("sec-ch-ua-mobile", "?0".into()), ("sec-ch-ua-model", "\"\"".into()), ("sec-ch-ua-platform", "\"Windows\"".into()),
            ("sec-ch-ua-platform-version", "\"19.0.0\"".into()), ("sec-fetch-dest", "empty".into()),
            ("sec-fetch-mode", "cors".into()), ("sec-fetch-site", "same-origin".into()),
            ("oai-device-id", self.device_id.clone()), ("oai-session-id", self.session_id.clone()),
            ("oai-language", "zh-CN".into()), ("oai-client-version", DEFAULT_CLIENT_VERSION.into()),
            ("oai-client-build-number", DEFAULT_CLIENT_BUILD_NUMBER.into()),
            ("x-openai-target-path", path.into()), ("x-openai-target-route", path.into()),
        ];
        let mut headers = HeaderMap::new();
        for (name, value) in pairs {
            headers.insert(
                HeaderName::from_bytes(name.as_bytes())
                    .map_err(|_| Error::new(ErrorKind::InvalidRequest))?,
                HeaderValue::from_str(&value).map_err(|_| Error::new(ErrorKind::InvalidRequest))?,
            );
        }
        if !self.config.access_token.is_empty() {
            let mut value = HeaderValue::from_str(&format!("Bearer {}", self.config.access_token))
                .map_err(|_| Error::new(ErrorKind::InvalidRequest))?;
            value.set_sensitive(true);
            headers.insert(AUTHORIZATION, value);
        }
        for (name, value) in extra {
            let mut value =
                HeaderValue::from_str(value).map_err(|_| Error::new(ErrorKind::InvalidRequest))?;
            if name.eq_ignore_ascii_case("authorization")
                || name.to_ascii_lowercase().contains("token")
            {
                value.set_sensitive(true);
            }
            headers.insert(
                HeaderName::from_bytes(name.as_bytes())
                    .map_err(|_| Error::new(ErrorKind::InvalidRequest))?,
                value,
            );
        }
        Ok(headers)
    }

    fn cap_timeout(&self, cap: Duration) -> Duration {
        self.config.timeout.min(cap)
    }
    pub(super) async fn execute(
        &self,
        builder: RequestBuilder,
        cap: Duration,
    ) -> Result<Response, Error> {
        let timeout = self.cap_timeout(cap);
        let future = builder.timeout(timeout).send();
        let response = tokio::time::timeout(timeout, future)
            .await
            .map_err(|_| Error::new(ErrorKind::Timeout))?
            .map_err(|e| Error::from_reqwest(&e))?;
        if response.status().is_success() {
            Ok(response)
        } else {
            Err(http_error(&response))
        }
    }
    pub(super) async fn read_bounded(
        &self,
        response: Response,
        limit: usize,
        cap: Duration,
    ) -> Result<Vec<u8>, Error> {
        if response.content_length().is_some_and(|n| n > limit as u64) {
            return Err(Error::new(ErrorKind::InvalidResponse));
        }
        let timeout = self.cap_timeout(cap);
        let read = async move {
            let mut response = response;
            let mut result = Vec::new();
            while let Some(chunk) = response
                .chunk()
                .await
                .map_err(|e| Error::from_reqwest(&e))?
            {
                if result.len().saturating_add(chunk.len()) > limit {
                    return Err(Error::new(ErrorKind::InvalidResponse));
                }
                result.extend_from_slice(&chunk);
            }
            Ok::<_, Error>(result)
        };
        tokio::time::timeout(timeout, read)
            .await
            .map_err(|_| Error::new(ErrorKind::Timeout))?
    }
    pub(super) async fn read_json(
        &self,
        response: Response,
        cap: Duration,
    ) -> Result<Value, Error> {
        let bytes = self.read_bounded(response, MAX_JSON_BODY, cap).await?;
        serde_json::from_slice(&bytes).map_err(|_| Error::new(ErrorKind::InvalidResponse))
    }
    pub(super) async fn send(
        &self,
        method: Method,
        path: &str,
        body: Option<&Value>,
        extra: &[ExtraHeader],
        cap: Duration,
    ) -> Result<Response, Error> {
        let url = self.url(path)?;
        let headers = self.headers(path, extra)?;
        let mut request = self.http.request(method, url).headers(headers);
        if let Some(value) = body {
            if serde_json::to_vec(value)
                .map_err(|_| Error::new(ErrorKind::InvalidRequest))?
                .len()
                > MAX_JSON_BODY
            {
                return Err(Error::new(ErrorKind::InvalidRequest));
            }
            request = request.json(value);
        }
        self.execute(request, cap).await
    }
    pub(super) async fn json(
        &self,
        method: Method,
        path: &str,
        body: Option<&Value>,
        extra: &[ExtraHeader],
        cap: Duration,
    ) -> Result<Value, Error> {
        let response = self.send(method, path, body, extra, cap).await?;
        self.read_json(response, cap).await
    }
    pub(super) async fn read_and_discard(
        &self,
        response: Response,
        cap: Duration,
    ) -> Result<(), Error> {
        self.read_bounded(response, MAX_JSON_BODY, cap)
            .await
            .map(|_| ())
    }

    async fn bootstrap(&self) -> Result<pow::PowResources, Error> {
        let mut headers = self.headers("/", &[("accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8".into()), ("sec-fetch-dest", "document".into()), ("sec-fetch-mode", "navigate".into()), ("sec-fetch-site", "none".into()), ("sec-fetch-user", "?1".into()), ("upgrade-insecure-requests", "1".into())])?;
        headers.insert(ACCEPT, HeaderValue::from_static("text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"));
        let response = self
            .execute(
                self.http.get(self.url("/")?).headers(headers),
                Duration::from_secs(30),
            )
            .await?;
        let bytes = self
            .read_bounded(response, MAX_JSON_BODY, Duration::from_secs(30))
            .await?;
        let html = String::from_utf8_lossy(&bytes);
        Ok(pow::parse_pow_resources(&html))
    }

    async fn requirements(&self, resources: &pow::PowResources) -> Result<Requirements, Error> {
        let base = if self.config.access_token.is_empty() {
            "/backend-anon/sentinel/chat-requirements"
        } else {
            "/backend-api/sentinel/chat-requirements"
        };
        let legacy_config = pow::legacy_config(&self.user_agent, resources);
        let p = pow::legacy_token(&legacy_config)
            .map_err(|_| Error::new(ErrorKind::InvalidResponse))?;
        // The reference builds a second independent browser-like PoW configuration.
        let proof_config = pow::legacy_config(&self.user_agent, resources);
        let prepare_path = format!("{base}/prepare");
        let prepare_body = json!({"p": p});
        let prepare = self
            .json(
                Method::POST,
                &prepare_path,
                Some(&prepare_body),
                &[("content-type", "application/json".into())],
                Duration::from_secs(30),
            )
            .await?;
        if prepare
            .pointer("/arkose/required")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return Err(Error::new(ErrorKind::Unsupported));
        }
        let proof = if prepare
            .pointer("/proofofwork/required")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            let seed = prepare
                .pointer("/proofofwork/seed")
                .and_then(Value::as_str)
                .filter(|s| s.len() <= 4096)
                .ok_or_else(|| Error::new(ErrorKind::InvalidResponse))?;
            let difficulty = prepare
                .pointer("/proofofwork/difficulty")
                .and_then(Value::as_str)
                .filter(|s| s.len() <= 128)
                .ok_or_else(|| Error::new(ErrorKind::InvalidResponse))?;
            pow::proof_token(seed, difficulty, &proof_config).map_err(|e| {
                Error::new(if e == pow::PowError::Unsolved {
                    ErrorKind::ProofOfWork
                } else {
                    ErrorKind::InvalidResponse
                })
            })?
        } else {
            String::new()
        };
        let turnstile_token = if prepare
            .pointer("/turnstile/required")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            let dx = prepare
                .pointer("/turnstile/dx")
                .and_then(Value::as_str)
                .ok_or_else(|| Error::new(ErrorKind::Unsupported))?;
            turnstile::solve(dx, &p).ok_or_else(|| Error::new(ErrorKind::Unsupported))?
        } else {
            String::new()
        };
        let prepare_token = prepare
            .get("prepare_token")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty() && s.len() <= 8192)
            .ok_or_else(|| Error::new(ErrorKind::InvalidResponse))?;
        let finalize_path = format!("{base}/finalize");
        let finalize_body = json!({"prepare_token":prepare_token,"proof_token":proof,"turnstile_token":turnstile_token});
        let finalized = self
            .json(
                Method::POST,
                &finalize_path,
                Some(&finalize_body),
                &[("content-type", "application/json".into())],
                Duration::from_secs(30),
            )
            .await?;
        let token = finalized
            .get("token")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty() && s.len() <= 8192)
            .ok_or_else(|| Error::new(ErrorKind::InvalidResponse))?
            .to_owned();
        let so_token = finalized
            .get("so_token")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        Ok(Requirements {
            token,
            proof,
            turnstile: turnstile_token,
            so_token,
        })
    }

    fn requirement_headers(&self, req: &Requirements, accept: &str) -> Vec<ExtraHeader> {
        let mut headers = vec![
            ("accept", accept.to_owned()),
            ("content-type", "application/json".into()),
            ("openai-sentinel-chat-requirements-token", req.token.clone()),
        ];
        if !req.proof.is_empty() {
            headers.push(("openai-sentinel-proof-token", req.proof.clone()));
        }
        if !req.turnstile.is_empty() {
            headers.push(("openai-sentinel-turnstile-token", req.turnstile.clone()));
        }
        if !req.so_token.is_empty() {
            headers.push(("openai-sentinel-so-token", req.so_token.clone()));
        }
        headers
    }

    /// Submit a normal or hosted-workflow conversation and return its raw SSE response.
    pub async fn conversation(&self, request: &ConversationRequest) -> Result<Response, Error> {
        let messages = validate_messages(request)?;
        let model = validate_model(&request.model)?;
        validate_hints(&request.system_hints)?;
        let resources = self.bootstrap().await?;
        let requirements = self.requirements(&resources).await?;
        let hosted = !request.system_hints.is_empty();
        let effort = request
            .thinking_effort
            .as_deref()
            .and_then(normalize_effort);
        let (path, body, mut headers) = if hosted {
            if self.config.access_token.is_empty() {
                return Err(Error::new(ErrorKind::Authentication));
            }
            let conduit = self
                .prepare_hosted(request, &messages, &model, &requirements)
                .await?;
            let path = "/backend-api/f/conversation";
            let mut body = json!({
                "action":"next", "messages":messages, "model":model, "parent_message_id":new_id(),
                "client_prepare_state":"sent", "timezone_offset_min":-480, "timezone":"Asia/Shanghai",
                "conversation_mode":{"kind":"primary_assistant"}, "enable_message_followups":true,
                "system_hints":request.system_hints, "supports_buffering":true, "supported_encodings":["v1"],
                "history_and_training_disabled":true, "client_contextual_info":{"is_dark_mode":false,"time_since_loaded":1200,
                "page_height":1072,"page_width":1724,"pixel_ratio":1.2,"screen_height":1440,"screen_width":2560,"app_name":"chatgpt.com"},
                "paragen_cot_summary_display_override":"allow", "force_parallel_switch":"auto"
            });
            if request.system_hints.iter().any(|h| h == "search") {
                body["force_use_search"] = json!(true);
                body["client_reported_search_source"] = json!("conversation_composer_web_icon");
            }
            if let Some(value) = effort {
                body["thinking_effort"] = json!(value);
            }
            let mut headers = self.requirement_headers(&requirements, "text/event-stream");
            headers.push(("x-conduit-token", conduit));
            headers.push(("x-oai-turn-trace-id", new_id()));
            (path, body, headers)
        } else {
            let path = if self.config.access_token.is_empty() {
                "/backend-anon/conversation"
            } else {
                "/backend-api/conversation"
            };
            let timezone = if self.config.access_token.is_empty() {
                "America/Los_Angeles"
            } else {
                "Asia/Shanghai"
            };
            let mut body = json!({
                "action":"next", "messages":messages, "model":model, "parent_message_id":new_id(),
                "conversation_mode":{"kind":"primary_assistant"}, "conversation_origin":null, "force_paragen":false,
                "force_paragen_model_slug":"", "force_rate_limit":false, "force_use_sse":true,
                "history_and_training_disabled":true, "reset_rate_limits":false, "suggestions":[],
                "supported_encodings":[], "system_hints":request.system_hints, "timezone":timezone,
                "timezone_offset_min":-480, "variant_purpose":"comparison_implicit", "websocket_request_id":new_id(),
                "client_contextual_info":{"is_dark_mode":false,"time_since_loaded":120,"page_height":900,"page_width":1400,
                "pixel_ratio":2,"screen_height":1440,"screen_width":2560}
            });
            if let Some(value) = effort {
                body["thinking_effort"] = json!(value);
            }
            let headers = self.requirement_headers(&requirements, "text/event-stream");
            (path, body, headers)
        };
        let response = self
            .send(
                Method::POST,
                path,
                Some(&body),
                &headers,
                self.config.timeout,
            )
            .await?;
        let content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if !content_type
            .to_ascii_lowercase()
            .starts_with("text/event-stream")
        {
            return Err(Error::new(ErrorKind::InvalidResponse));
        }
        Ok(response)
    }

    async fn prepare_hosted(
        &self,
        request: &ConversationRequest,
        messages: &[Value],
        model: &str,
        requirements: &Requirements,
    ) -> Result<String, Error> {
        let path = "/backend-api/f/conversation/prepare";
        let prompt = latest_user_text(messages);
        let body = json!({
            "action":"next", "fork_from_shared_post":false, "parent_message_id":new_id(), "model":model,
            "client_prepare_state":"success", "timezone_offset_min":-480, "timezone":"Asia/Shanghai",
            "conversation_mode":{"kind":"primary_assistant"}, "system_hints":request.system_hints,
            "partial_query":{"id":new_id(),"author":{"role":"user"},"content":{"content_type":"text","parts":[prompt]}},
            "supports_buffering":true,"supported_encodings":["v1"],"client_contextual_info":{"app_name":"chatgpt.com"}
        });
        let mut headers = self.requirement_headers(requirements, "*/*");
        headers.push(("x-conduit-token", "no-token".into()));
        let response = self
            .send(
                Method::POST,
                path,
                Some(&body),
                &headers,
                Duration::from_secs(60),
            )
            .await?;
        let payload = self.read_json(response, Duration::from_secs(60)).await?;
        payload
            .get("conduit_token")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty() && s.len() <= 8192)
            .map(str::to_owned)
            .ok_or_else(|| Error::new(ErrorKind::InvalidResponse))
    }

    /// Fetch the authenticated conversation document.
    pub async fn conversation_detail(&self, id: &str) -> Result<Value, Error> {
        validate_identifier(id)?;
        let path = format!("/backend-api/conversation/{id}");
        self.json(
            Method::GET,
            &path,
            None,
            &[("accept", "application/json".into())],
            Duration::from_secs(60),
        )
        .await
    }

    /// Upload an image and return a `multimodal_text` asset part.
    pub async fn upload_image(
        &self,
        data: Vec<u8>,
        mime: &str,
        filename: &str,
    ) -> Result<Value, Error> {
        if self.config.access_token.is_empty() {
            return Err(Error::new(ErrorKind::Authentication));
        }
        if data.is_empty()
            || data.len() > MAX_IMAGE_BYTES
            || !matches!(mime, "image/png" | "image/jpeg" | "image/gif")
            || filename.is_empty()
            || filename.len() > 180
            || filename
                .chars()
                .any(|c| c.is_control() || c == '/' || c == '\\')
        {
            return Err(Error::new(ErrorKind::InvalidRequest));
        }
        let (width, height) =
            image_dimensions(&data, mime).ok_or_else(|| Error::new(ErrorKind::InvalidRequest))?;
        let size = data.len();
        let path = "/backend-api/files";
        let upload_body = json!({"file_name":filename,"file_size":size,"use_case":"multimodal","width":width,"height":height});
        let meta = self
            .json(
                Method::POST,
                path,
                Some(&upload_body),
                &[("accept", "application/json".into())],
                Duration::from_secs(60),
            )
            .await?;
        let file_id = meta
            .get("file_id")
            .and_then(Value::as_str)
            .filter(|id| safe_identifier(id))
            .ok_or_else(|| Error::new(ErrorKind::InvalidResponse))?;
        let upload = meta
            .get("upload_url")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::new(ErrorKind::InvalidResponse))?;
        let upload_url = self
            .validate_transfer_url(upload)
            .ok_or_else(|| Error::new(ErrorKind::InvalidResponse))?;
        let origin = self.base.origin().ascii_serialization();
        let referer = format!("{origin}/");
        let upload_headers = [
            ("content-type", mime),
            ("x-ms-blob-type", "BlockBlob"),
            ("x-ms-version", "2020-04-08"),
            ("origin", origin.as_str()),
            ("referer", referer.as_str()),
            ("user-agent", self.user_agent.as_str()),
            ("accept", "application/json, text/plain, */*"),
            ("accept-language", "en-US,en;q=0.8"),
        ];
        let mut headers = HeaderMap::new();
        for (name, value) in upload_headers {
            headers.insert(
                HeaderName::from_bytes(name.as_bytes())
                    .map_err(|_| Error::new(ErrorKind::InvalidRequest))?,
                HeaderValue::from_str(value).map_err(|_| Error::new(ErrorKind::InvalidRequest))?,
            );
        }
        let upload_response = self
            .execute(
                self.http.put(upload_url).headers(headers).body(data),
                Duration::from_secs(120),
            )
            .await?;
        self.read_and_discard(upload_response, Duration::from_secs(120))
            .await?;
        let uploaded_path = format!("/backend-api/files/{file_id}/uploaded");
        let uploaded_body = json!({});
        let uploaded = self
            .send(
                Method::POST,
                &uploaded_path,
                Some(&uploaded_body),
                &[("content-type", "application/json".into())],
                Duration::from_secs(60),
            )
            .await?;
        self.read_and_discard(uploaded, Duration::from_secs(60))
            .await?;
        Ok(
            json!({"content_type":"image_asset_pointer","asset_pointer":format!("file-service://{file_id}"),"width":width,"height":height,"size_bytes":size}),
        )
    }
    pub(super) fn validate_transfer_url(&self, raw: &str) -> Option<Url> {
        let authority = raw.split_once("://")?.1.split(['/', '?', '#']).next()?;
        if authority.contains('@') {
            return None;
        }
        let url = Url::parse(raw).ok()?;
        if !url.username().is_empty() || url.password().is_some() || url.fragment().is_some() {
            return None;
        }
        let same_origin = url.origin() == self.base.origin();
        if url.scheme() != "https"
            && !(same_origin && self.base.scheme() == "http" && url.scheme() == "http")
        {
            return None;
        }
        if !same_origin && is_private_host(url.host_str()?) {
            return None;
        }
        Some(url)
    }

    /// Polls conversation details and returns the resolved answer and sources for hosted search.
    pub async fn poll_search_result(&self, conversation_id: &str) -> Result<Value, Error> {
        validate_identifier(conversation_id)?;
        const MAX_POLLS: usize = 100;
        let deadline = Instant::now() + self.config.timeout.min(Duration::from_secs(300));
        let mut last_answer = String::new();
        let mut stable = 0usize;
        let mut last = None;
        for attempt in 0..MAX_POLLS {
            if Instant::now() >= deadline {
                break;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            let fetched =
                tokio::time::timeout(remaining, self.conversation_detail(conversation_id))
                    .await
                    .map_err(|_| Error::new(ErrorKind::Timeout))?;
            match fetched {
                Ok(document) => {
                    let result = search_extract(conversation_id, &document);
                    if !result.0.is_empty() {
                        if result.0 == last_answer {
                            stable += 1;
                        } else {
                            last_answer.clone_from(&result.0);
                            stable = 0;
                        }
                        let done = matches!(
                            result.1.as_str(),
                            "finished_successfully" | "finished_partial_completion"
                        );
                        last = Some(
                            json!({"conversation_id":conversation_id,"status":result.1,"answer":result.0,"sources":result.2}),
                        );
                        if done || stable >= 1 {
                            return last.ok_or_else(|| Error::new(ErrorKind::InvalidResponse));
                        }
                    }
                }
                Err(error)
                    if matches!(
                        error.status_code(),
                        Some(404 | 409 | 423 | 429 | 500 | 502 | 503 | 504)
                    ) => {}
                Err(error) => return Err(error),
            }
            if attempt + 1 < MAX_POLLS {
                tokio::time::sleep(
                    Duration::from_secs(3).min(deadline.saturating_duration_since(Instant::now())),
                )
                .await;
            }
        }
        last.ok_or_else(|| Error::new(ErrorKind::Timeout))
    }
}

fn http_error(response: &Response) -> Error {
    let retry_after = response
        .headers()
        .get("retry-after")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<u64>().ok());
    Error::http(response.status().as_u16(), retry_after)
}
fn validate_identifier(id: &str) -> Result<(), Error> {
    if safe_identifier(id) {
        Ok(())
    } else {
        Err(Error::new(ErrorKind::InvalidRequest))
    }
}
pub(super) fn safe_identifier(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}
fn validate_model(model: &str) -> Result<String, Error> {
    if model.len() > 128 || model.chars().any(char::is_control) {
        return Err(Error::new(ErrorKind::InvalidRequest));
    }
    Ok(if model.is_empty() {
        "auto".into()
    } else {
        model.into()
    })
}
fn validate_hints(hints: &[String]) -> Result<(), Error> {
    if hints.len() > 16
        || hints
            .iter()
            .any(|h| h.is_empty() || h.len() > 128 || h.chars().any(char::is_control))
    {
        Err(Error::new(ErrorKind::InvalidRequest))
    } else {
        Ok(())
    }
}
fn validate_messages(request: &ConversationRequest) -> Result<Vec<Value>, Error> {
    if request.messages.is_empty() || request.messages.len() > 256 {
        return Err(Error::new(ErrorKind::InvalidRequest));
    }
    let mut messages = Vec::with_capacity(request.messages.len());
    for message in &request.messages {
        let mut message = message.clone();
        if !message.is_object()
            || message.get("id").and_then(Value::as_str).is_none()
            || message
                .pointer("/author/role")
                .and_then(Value::as_str)
                .is_none()
            || !matches!(
                message
                    .pointer("/content/content_type")
                    .and_then(Value::as_str),
                Some("text" | "multimodal_text")
            )
            || message
                .pointer("/content/parts")
                .and_then(Value::as_array)
                .is_none()
        {
            return Err(Error::new(ErrorKind::InvalidRequest));
        }
        if !message.get("metadata").is_some_and(Value::is_object) {
            message["metadata"] = json!({});
        }
        messages.push(message);
    }
    Ok(messages)
}
fn normalize_effort(value: &str) -> Option<&'static str> {
    match value.trim().to_ascii_lowercase().as_str() {
        "low" => Some("low"),
        "medium" => Some("medium"),
        "high" => Some("high"),
        "standard" => Some("standard"),
        "max" => Some("max"),
        "xhigh" | "extended" => Some("extended"),
        _ => None,
    }
}
fn latest_user_text(messages: &[Value]) -> String {
    let Some(message) = messages
        .iter()
        .rev()
        .find(|m| m.pointer("/author/role").and_then(Value::as_str) == Some("user"))
    else {
        return String::new();
    };
    let Some(parts) = message.pointer("/content/parts").and_then(Value::as_array) else {
        return String::new();
    };
    let mut text = String::new();
    for part in parts {
        if let Some(value) = part.as_str() {
            text.push_str(value);
        } else if let Some(value) = part.get("text").and_then(Value::as_str) {
            text.push_str(value);
        }
    }
    text
}
fn is_private_host(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost")
        || host.to_ascii_lowercase().ends_with(".localhost")
        || host.to_ascii_lowercase().ends_with(".local")
    {
        return true;
    }
    match host.parse::<std::net::IpAddr>() {
        Ok(std::net::IpAddr::V4(ip)) => {
            ip.is_private() || ip.is_loopback() || ip.is_link_local() || ip.is_unspecified()
        }
        Ok(std::net::IpAddr::V6(ip)) => {
            ip.is_loopback()
                || ip.is_unspecified()
                || ip.is_unique_local()
                || ip.is_unicast_link_local()
        }
        Err(_) => false,
    }
}

fn image_dimensions(data: &[u8], mime: &str) -> Option<(u32, u32)> {
    let (width, height) = match mime {
        "image/png" if data.len() >= 24 && data.starts_with(b"\x89PNG\r\n\x1a\n") => (
            u32::from_be_bytes(data[16..20].try_into().ok()?),
            u32::from_be_bytes(data[20..24].try_into().ok()?),
        ),
        "image/gif" if data.len() >= 10 && data.starts_with(b"GIF8") => (
            u16::from_le_bytes(data[6..8].try_into().ok()?) as u32,
            u16::from_le_bytes(data[8..10].try_into().ok()?) as u32,
        ),
        "image/jpeg" if data.starts_with(&[0xff, 0xd8]) => jpeg_dimensions(data)?,
        _ => return None,
    };
    if width == 0
        || height == 0
        || width > MAX_IMAGE_DIMENSION
        || height > MAX_IMAGE_DIMENSION
        || u64::from(width) * u64::from(height) > 500_000_000
    {
        None
    } else {
        Some((width, height))
    }
}
fn jpeg_dimensions(data: &[u8]) -> Option<(u32, u32)> {
    let mut i = 2usize;
    while i + 4 <= data.len() {
        if data[i] != 0xff {
            i += 1;
            continue;
        }
        while i < data.len() && data[i] == 0xff {
            i += 1;
        }
        if i >= data.len() {
            break;
        }
        let marker = data[i];
        i += 1;
        if marker == 0xd8 || marker == 0xd9 || (0xd0..=0xd7).contains(&marker) {
            continue;
        }
        if i + 2 > data.len() {
            break;
        }
        let length = u16::from_be_bytes([data[i], data[i + 1]]) as usize;
        if length < 2 || i + length > data.len() {
            break;
        }
        if matches!(marker,0xc0..=0xc3|0xc5..=0xc7|0xc9..=0xcb|0xcd..=0xcf) && length >= 7 {
            return Some((
                u16::from_be_bytes([data[i + 5], data[i + 6]]) as u32,
                u16::from_be_bytes([data[i + 3], data[i + 4]]) as u32,
            ));
        }
        i += length;
    }
    None
}

fn search_extract(id: &str, document: &Value) -> (String, String, Vec<Value>) {
    let mut messages = document
        .pointer("/mapping")
        .and_then(Value::as_object)
        .map(|m| {
            m.values()
                .filter_map(|node| node.get("message"))
                .filter(|m| m.pointer("/author/role").and_then(Value::as_str) == Some("assistant"))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    messages.sort_by(|a, b| {
        a.get("create_time")
            .and_then(Value::as_f64)
            .partial_cmp(&b.get("create_time").and_then(Value::as_f64))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let Some(message) = messages.last() else {
        return (String::new(), String::new(), Vec::new());
    };
    let mut parts = Vec::new();
    if let Some(content) = message.get("content") {
        if let Some(text) = content.get("text").and_then(Value::as_str) {
            parts.push(text.trim().to_owned());
        }
        if let Some(values) = content.get("parts").and_then(Value::as_array) {
            for part in values {
                if let Some(s) = part.as_str() {
                    parts.push(s.trim().to_owned());
                } else if part.is_object() {
                    for key in ["text", "summary", "content"] {
                        if let Some(s) = part.get(key).and_then(Value::as_str) {
                            parts.push(s.trim().to_owned());
                        }
                    }
                }
            }
        }
    }
    let answer = parts
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    let status = message
        .pointer("/metadata/finish_details/type")
        .or_else(|| message.pointer("/metadata/status"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    let mut sources = Vec::new();
    let mut seen = std::collections::HashSet::new();
    collect_search_sources(message, &mut sources, &mut seen);
    let _ = id;
    (answer, status, sources)
}
fn collect_search_sources(
    value: &Value,
    out: &mut Vec<Value>,
    seen: &mut std::collections::HashSet<String>,
) {
    match value {
        Value::Object(map) => {
            let metadata = map.get("metadata").and_then(Value::as_object);
            let url = map
                .get("url")
                .or_else(|| map.get("link"))
                .or_else(|| map.get("source_url"))
                .or_else(|| metadata.and_then(|m| m.get("url")))
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            if (url.starts_with("https://") || url.starts_with("http://"))
                && !url.chars().any(char::is_control)
                && seen.insert(url.to_owned())
            {
                out.push(json!({"title":map.get("title").or_else(||map.get("name")).or_else(||map.get("source")).and_then(Value::as_str).unwrap_or(""),"url":url,"snippet":map.get("snippet").or_else(||map.get("text")).or_else(||map.get("description")).and_then(Value::as_str).unwrap_or(""),"source_type":map.get("type").or_else(||map.get("source_type")).and_then(Value::as_str).unwrap_or("")}));
            }
            for item in map.values() {
                collect_search_sources(item, out, seen);
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_search_sources(item, out, seen);
            }
        }
        _ => {}
    }
}

mod artifacts;
