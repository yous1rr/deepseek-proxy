use super::ChatGptClient;
use crate::{Error, ErrorKind};
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use serde_json::Value;
use std::{
    collections::HashSet,
    time::{Duration, Instant},
};

const MAX_POLLS: usize = 60;
const POLL_INTERVAL: Duration = Duration::from_secs(2);
const IMAGE_RESULT_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;

impl ChatGptClient {
    /// Poll a conversation for assistant/tool image outputs and return each validated image as Base64.
    /// User-message assets are excluded even when listed in `file_ids` or `sediment_ids`.
    pub async fn image_results(
        &self,
        conversation_id: &str,
        file_ids: &[String],
        sediment_ids: &[String],
    ) -> Result<Vec<String>, Error> {
        super::validate_identifier(conversation_id)?;
        if file_ids.len().saturating_add(sediment_ids.len()) > 64 {
            return Err(Error::new(ErrorKind::InvalidRequest));
        }
        let mut file_hints = Vec::new();
        let mut sediment_hints = Vec::new();
        for id in file_ids {
            if !super::safe_identifier(id) {
                return Err(Error::new(ErrorKind::InvalidRequest));
            }
            if !file_hints.contains(id) {
                file_hints.push(id.clone());
            }
        }
        for id in sediment_ids {
            if !super::safe_identifier(id) {
                return Err(Error::new(ErrorKind::InvalidRequest));
            }
            if !sediment_hints.contains(id) {
                sediment_hints.push(id.clone());
            }
        }
        let deadline = Instant::now() + self.config.timeout.min(IMAGE_RESULT_TIMEOUT);
        let mut previous: Option<(Vec<String>, Vec<String>)> = None;
        let mut outputs = (Vec::new(), Vec::new());
        for attempt in 0..MAX_POLLS {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            let detail = tokio::time::timeout(remaining, self.conversation_detail(conversation_id))
                .await
                .map_err(|_| Error::new(ErrorKind::Timeout))?;
            let document = match detail {
                Ok(value) => value,
                Err(error) if retryable(&error) => {
                    if attempt + 1 < MAX_POLLS {
                        tokio::time::sleep(
                            POLL_INTERVAL.min(deadline.saturating_duration_since(Instant::now())),
                        )
                        .await;
                    }
                    continue;
                }
                Err(error) => return Err(error),
            };
            let inputs = input_asset_ids(&document);
            let mut found = tool_image_ids(&document);
            found.0.retain(|id| !inputs.contains(id));
            found.1.retain(|id| !inputs.contains(id));
            prioritize(&mut found.0, &file_hints);
            prioritize(&mut found.1, &sediment_hints);
            if !found.0.is_empty() || !found.1.is_empty() {
                if previous.as_ref() == Some(&found) {
                    outputs = found;
                    break;
                }
                previous = Some(found.clone());
                outputs = found;
            }
            if attempt + 1 < MAX_POLLS {
                tokio::time::sleep(
                    POLL_INTERVAL.min(deadline.saturating_duration_since(Instant::now())),
                )
                .await;
            }
        }
        if outputs.0.is_empty() && outputs.1.is_empty() {
            return Err(Error::new(ErrorKind::Timeout));
        }
        let mut results = Vec::with_capacity(outputs.0.len() + outputs.1.len());
        for file_id in outputs.0 {
            let url = self.artifact_url(conversation_id, &file_id, None).await?;
            let bytes = self.download_image(&url).await?;
            results.push(B64.encode(bytes));
        }
        for attachment_id in outputs.1 {
            let url = self
                .artifact_url(conversation_id, &attachment_id, None)
                .await?;
            let bytes = self.download_image(&url).await?;
            results.push(B64.encode(bytes));
        }
        if results.is_empty() {
            return Err(Error::new(ErrorKind::InvalidResponse));
        }
        Ok(results)
    }

    /// Resolve a file or sandbox artifact to a safe URL. This method only returns
    /// the validated URL; it never follows it with the bearer-authenticated client.
    pub async fn artifact_url(
        &self,
        conversation_id: &str,
        file_id: &str,
        sandbox_path: Option<&str>,
    ) -> Result<String, Error> {
        super::validate_identifier(conversation_id)?;
        super::validate_identifier(file_id)?;
        if self.config.access_token.is_empty() {
            return Err(Error::new(ErrorKind::Authentication));
        }
        if let Some(path) = sandbox_path {
            validate_sandbox_path(path)?;
            let detail = self.conversation_detail(conversation_id).await?;
            let message_id = find_artifact_message(&detail, file_id, path)
                .ok_or_else(|| Error::new(ErrorKind::InvalidResponse))?;
            let route = "/backend-api/conversation/{conversation_id}/interpreter/download";
            let api_path =
                format!("/backend-api/conversation/{conversation_id}/interpreter/download");
            let params = [
                ("message_id", message_id),
                ("sandbox_path", path.to_owned()),
            ];
            if let Some(url) = self.try_download_route(&api_path, route, &params).await? {
                return Ok(url);
            }
        }
        let attachment_path =
            format!("/backend-api/conversation/{conversation_id}/attachment/{file_id}/download");
        let attachment_route =
            "/backend-api/conversation/{conversation_id}/attachment/{attachment_id}/download";
        let file_path = format!("/backend-api/files/{file_id}/download");
        let legacy_file_path = format!("/backend-api/files/download/{file_id}");
        let mut routes = Vec::with_capacity(3);
        if file_id.starts_with("sediment") {
            routes.push((attachment_path.as_str(), attachment_route));
        }
        routes.push((file_path.as_str(), "/backend-api/files/download/{file_id}"));
        routes.push((
            legacy_file_path.as_str(),
            "/backend-api/files/download/{file_id}",
        ));
        if !file_id.starts_with("sediment") {
            routes.push((attachment_path.as_str(), attachment_route));
        }
        for (path, route) in routes {
            if let Some(url) = self.try_download_route(path, route, &[]).await? {
                return Ok(url);
            }
        }
        Err(Error::new(ErrorKind::InvalidResponse))
    }

    async fn try_download_route(
        &self,
        path: &str,
        target_route: &str,
        query: &[(&str, String)],
    ) -> Result<Option<String>, Error> {
        let mut url = self.url(path)?;
        if !query.is_empty() {
            url.query_pairs_mut()
                .extend_pairs(query.iter().map(|(k, v)| (*k, v.as_str())));
        }
        let headers = self.headers(
            path,
            &[
                ("accept", "application/json".into()),
                ("x-openai-target-route", target_route.into()),
            ],
        )?;
        let response = match self
            .execute(self.http.get(url).headers(headers), Duration::from_secs(60))
            .await
        {
            Ok(response) => response,
            Err(error) if matches!(error.status_code(), Some(400 | 404 | 405 | 410 | 422)) => {
                return Ok(None)
            }
            Err(error) => return Err(error),
        };
        let payload = self.read_json(response, Duration::from_secs(60)).await?;
        for key in ["download_url", "url"] {
            if let Some(raw) = payload.get(key).and_then(Value::as_str) {
                if let Some(valid) = self.validate_transfer_url(raw) {
                    return Ok(Some(valid.to_string()));
                }
            }
        }
        Ok(None)
    }

    async fn download_image(&self, raw_url: &str) -> Result<Vec<u8>, Error> {
        let url = self
            .validate_transfer_url(raw_url)
            .ok_or_else(|| Error::new(ErrorKind::InvalidResponse))?;
        // No bearer, cookies, or ChatGPT headers are attached to this URL.
        let response = self
            .execute(self.http.get(url), Duration::from_secs(120))
            .await?;
        let bytes = self
            .read_bounded(response, MAX_IMAGE_BYTES, Duration::from_secs(120))
            .await?;
        if !is_image_bytes(&bytes) {
            return Err(Error::new(ErrorKind::InvalidResponse));
        }
        Ok(bytes)
    }
}

fn retryable(error: &Error) -> bool {
    matches!(
        error.status_code(),
        Some(404 | 409 | 423 | 429 | 500 | 502 | 503 | 504)
    ) || matches!(error.kind(), ErrorKind::Timeout | ErrorKind::Network)
}
fn prioritize(values: &mut [String], hints: &[String]) {
    values.sort_by_key(|value| !hints.contains(value));
}
fn validate_sandbox_path(path: &str) -> Result<(), Error> {
    let lower = path.to_ascii_lowercase();
    if !path.starts_with("/mnt/data/")
        || path.len() > 512
        || path.contains("..")
        || path.contains('\\')
        || path.chars().any(char::is_control)
        || lower.contains("%2e")
        || lower.contains("%2f")
        || lower.contains("%5c")
    {
        return Err(Error::new(ErrorKind::InvalidRequest));
    }
    Ok(())
}

fn input_asset_ids(document: &Value) -> HashSet<String> {
    let mut files = HashSet::new();
    let mut attachments = HashSet::new();
    let Some(mapping) = document.get("mapping").and_then(Value::as_object) else {
        return files;
    };
    for node in mapping.values() {
        let Some(message) = node.get("message") else {
            continue;
        };
        if message.pointer("/author/role").and_then(Value::as_str) == Some("user") {
            if let Some(content) = message.get("content") {
                collect_asset_refs(content, &mut files, &mut attachments);
            }
        }
    }
    files.extend(attachments);
    files
}
fn tool_image_ids(document: &Value) -> (Vec<String>, Vec<String>) {
    let mut files = HashSet::new();
    let mut attachments = HashSet::new();
    let Some(mapping) = document.get("mapping").and_then(Value::as_object) else {
        return (Vec::new(), Vec::new());
    };
    for node in mapping.values() {
        let Some(message) = node.get("message") else {
            continue;
        };
        let role = message
            .pointer("/author/role")
            .and_then(Value::as_str)
            .unwrap_or("");
        if role != "tool" && role != "assistant" {
            continue;
        }
        let image_gen = message
            .pointer("/metadata/async_task_type")
            .and_then(Value::as_str)
            == Some("image_gen");
        let has_pointer = contains_asset_pointer(message);
        if role == "assistant" && !image_gen && !has_pointer {
            continue;
        }
        collect_asset_refs(message, &mut files, &mut attachments);
    }
    let mut files: Vec<_> = files.into_iter().collect();
    let mut attachments: Vec<_> = attachments.into_iter().collect();
    files.sort();
    attachments.sort();
    (files, attachments)
}
fn contains_asset_pointer(value: &Value) -> bool {
    match value {
        Value::Object(map) => {
            map.get("content_type").and_then(Value::as_str) == Some("image_asset_pointer")
                || map
                    .get("asset_pointer")
                    .and_then(Value::as_str)
                    .is_some_and(|s| {
                        s.starts_with("file-service://") || s.starts_with("sediment://")
                    })
                || map.values().any(contains_asset_pointer)
        }
        Value::Array(values) => values.iter().any(contains_asset_pointer),
        _ => false,
    }
}
fn collect_asset_refs(
    value: &Value,
    files: &mut HashSet<String>,
    attachments: &mut HashSet<String>,
) {
    match value {
        Value::Object(map) => {
            if let Some(pointer) = map.get("asset_pointer").and_then(Value::as_str) {
                collect_pointer(pointer, files, attachments);
            }
            for child in map.values() {
                collect_asset_refs(child, files, attachments);
            }
        }
        Value::Array(values) => {
            for child in values {
                collect_asset_refs(child, files, attachments);
            }
        }
        Value::String(text) => {
            for prefix in ["file-service://", "sediment://"] {
                for tail in text.split(prefix).skip(1) {
                    let id: String = tail
                        .chars()
                        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
                        .collect();
                    if super::safe_identifier(&id) && id != "file_upload" {
                        if prefix.starts_with("file-service") {
                            files.insert(id);
                        } else {
                            attachments.insert(id);
                        }
                    }
                }
            }
            for part in text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
                if is_real_image_file_id(part) {
                    files.insert(part.to_owned());
                }
            }
        }
        _ => {}
    }
}
fn collect_pointer(value: &str, files: &mut HashSet<String>, attachments: &mut HashSet<String>) {
    if let Some(id) = value.strip_prefix("file-service://") {
        if super::safe_identifier(id) && id != "file_upload" {
            files.insert(id.to_owned());
        }
    }
    if let Some(id) = value.strip_prefix("sediment://") {
        if super::safe_identifier(id) {
            attachments.insert(id.to_owned());
        }
    }
}
fn is_real_image_file_id(value: &str) -> bool {
    value.len() == 37
        && value.starts_with("file_00000000")
        && value[13..].bytes().all(|b| b.is_ascii_hexdigit())
}

fn find_artifact_message(document: &Value, file_id: &str, sandbox_path: &str) -> Option<String> {
    let mapping = document.get("mapping")?.as_object()?;
    for (node_id, node) in mapping {
        let Some(message) = node.get("message") else {
            continue;
        };
        let role = message
            .pointer("/author/role")
            .and_then(Value::as_str)
            .unwrap_or("");
        if role != "assistant" && role != "tool" {
            continue;
        }
        let mut found = false;
        artifact_match(message, file_id, sandbox_path, &mut found);
        if found {
            if let Some(id) = message
                .get("id")
                .and_then(Value::as_str)
                .filter(|s| super::safe_identifier(s))
            {
                return Some(id.to_owned());
            }
            if super::safe_identifier(node_id) {
                return Some(node_id.clone());
            }
        }
    }
    None
}
fn artifact_match(value: &Value, file_id: &str, sandbox_path: &str, found: &mut bool) {
    if *found {
        return;
    }
    match value {
        Value::Object(map) => {
            if map.get("sandbox_path").and_then(Value::as_str) == Some(sandbox_path)
                || map.get("file_id").and_then(Value::as_str) == Some(file_id)
                || map.get("id").and_then(Value::as_str) == Some(file_id)
            {
                *found = true;
                return;
            }
            for child in map.values() {
                artifact_match(child, file_id, sandbox_path, found);
            }
        }
        Value::Array(values) => {
            for child in values {
                artifact_match(child, file_id, sandbox_path, found);
            }
        }
        Value::String(text) => {
            if text.contains(sandbox_path) || text.contains(file_id) {
                *found = true;
            }
        }
        _ => {}
    }
}
fn is_image_bytes(data: &[u8]) -> bool {
    data.starts_with(b"\x89PNG\r\n\x1a\n")
        || data.starts_with(&[0xff, 0xd8, 0xff])
        || data.starts_with(b"GIF87a")
        || data.starts_with(b"GIF89a")
        || (data.len() >= 12 && data.starts_with(b"RIFF") && &data[8..12] == b"WEBP")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn excludes_user_asset_and_extracts_only_assistant_tool_outputs() {
        let doc = json!({"mapping":{
            "u":{"message":{"author":{"role":"user"},"content":{"parts":[{"content_type":"image_asset_pointer","asset_pointer":"file-service://input"}]}}},
            "a":{"message":{"author":{"role":"assistant"},"metadata":{"async_task_type":"image_gen"},"content":{"parts":[{"content_type":"image_asset_pointer","asset_pointer":"file-service://output"},{"asset_pointer":"sediment://attach"}]}}}
        }});
        let input = input_asset_ids(&doc);
        let mut out = tool_image_ids(&doc);
        out.0.retain(|v| !input.contains(v));
        out.1.retain(|v| !input.contains(v));
        assert_eq!(input, HashSet::from(["input".to_owned()]));
        assert_eq!(out.0, vec!["output"]);
        assert_eq!(out.1, vec!["attach"]);
    }
    #[test]
    fn sandbox_paths_are_confined() {
        assert!(validate_sandbox_path("/mnt/data/result.pptx").is_ok());
        assert!(validate_sandbox_path("/mnt/data/../secret").is_err());
        assert!(validate_sandbox_path("/mnt/data/%2e%2e/secret").is_err());
    }
}
