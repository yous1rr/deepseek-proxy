//! ChatGPT Web accounts: credential storage, expiry tracking and alerts.
//!
//! The ChatGPT web session (and the `auth.json` that tools like the Codex CLI
//! write) hands out an **access token only**. There is no refresh token to use,
//! so nothing here can extend an account's life: what it can do is read the
//! token's own `exp` claim and warn before the account silently stops working.
//!
//! That warning goes through Bark (an iOS push service). Configure it with
//! `BARK_URL` (a full `https://api.day.app/<key>` base) or `BARK_KEY`; without
//! either, expiry is only logged.

use crate::store::{ChatGptAccountRow, Store};
use std::time::{SystemTime, UNIX_EPOCH};

/// Warning stages, ordered from earliest to latest. An account only alerts once
/// per stage, so a long-running server does not repeat itself every sweep.
pub const STAGES: [(&str, i64); 3] = [("72h", 72 * 3600), ("24h", 24 * 3600), ("2h", 2 * 3600)];

/// Decoded timing information from an access token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TokenTimes {
    /// `exp`, in unix seconds.
    pub expires_at: Option<i64>,
    /// `iat`, in unix seconds.
    pub issued_at: Option<i64>,
}

/// Decode a JWT's `exp`/`iat` **without** verifying the signature.
///
/// This is only used to decide when to warn, never to authorise anything: the
/// token is still verified by the upstream on every request.
pub fn token_times(token: &str) -> TokenTimes {
    let mut parts = token.split('.');
    let (Some(_header), Some(payload)) = (parts.next(), parts.next()) else {
        return TokenTimes::default();
    };
    let Some(bytes) = base64url_decode(payload) else {
        return TokenTimes::default();
    };
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return TokenTimes::default();
    };
    let number = |key: &str| value.get(key).and_then(serde_json::Value::as_i64);
    TokenTimes {
        expires_at: number("exp"),
        issued_at: number("iat"),
    }
}

/// The email claim OpenAI puts on its tokens, when present.
pub fn token_email(token: &str) -> Option<String> {
    let payload = token.split('.').nth(1)?;
    let bytes = base64url_decode(payload)?;
    let value = serde_json::from_slice::<serde_json::Value>(&bytes).ok()?;
    value
        .get("https://api.openai.com/profile")
        .and_then(|profile| profile.get("email"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
}

fn base64url_decode(input: &str) -> Option<Vec<u8>> {
    use base64::Engine;
    let trimmed = input.trim_end_matches('=');
    base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(trimmed)
        .ok()
}

/// A stable, non-secret account identity derived from the token.
///
/// Two pastes of the same token map to the same id, so `add` updates rather than
/// duplicates; a different token yields a different id.
pub fn account_id(token: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(token.as_bytes());
    format!("cg_{}", hex::encode(&digest[..12]))
}

/// Human-readable remaining life, for the CLI.
pub fn describe_expiry(expires_at: Option<i64>, now: i64) -> String {
    let Some(expires_at) = expires_at else {
        return "unknown (token has no exp)".to_string();
    };
    let remaining = expires_at - now;
    if remaining <= 0 {
        return format!("expired {} ago", duration_text(-remaining));
    }
    duration_text(remaining)
}

fn duration_text(seconds: i64) -> String {
    let days = seconds / 86_400;
    let hours = (seconds % 86_400) / 3_600;
    let minutes = (seconds % 3_600) / 60;
    if days > 0 {
        format!("{days}d {hours}h")
    } else if hours > 0 {
        format!("{hours}h {minutes}m")
    } else {
        format!("{minutes}m")
    }
}

/// Which warning stage an account has just entered, if any.
///
/// Returns the tightest applicable stage that has not been announced yet, so a
/// server that was down for a week sends one message rather than three.
pub fn due_stage(row: &ChatGptAccountRow, now: i64) -> Option<&'static str> {
    let expires_at = row.expires_at?;
    let remaining = expires_at - now;
    let announced = row.notified_stage.as_deref();
    let due = STAGES
        .iter()
        .filter(|(_, window)| remaining <= *window)
        .map(|(stage, _)| *stage)
        .last()?;
    let rank = |stage: &str| STAGES.iter().position(|(name, _)| *name == stage);
    match announced {
        Some(previous)
            if rank(previous).is_some_and(|old| rank(due).is_some_and(|new| new <= old)) =>
        {
            None
        }
        _ => Some(due),
    }
}

pub fn has_72h_notice(row: &ChatGptAccountRow) -> bool {
    matches!(row.notified_stage.as_deref(), Some("72h" | "24h" | "2h"))
}

/// Bark push configuration, from the environment only.
#[derive(Debug, Clone)]
pub struct Bark {
    base: String,
}

impl Bark {
    /// Build the notifier, or `None` when Bark is not configured.
    ///
    /// `BARK_URL` is the full base (`https://api.day.app/<key>`); `BARK_KEY`
    /// plus the default host is the shorter form.
    pub fn from_env() -> Option<Self> {
        Self::resolve(
            std::env::var("BARK_URL").ok(),
            std::env::var("BARK_KEY").ok(),
        )
    }

    /// Testable constructor.
    pub fn resolve(url: Option<String>, key: Option<String>) -> Option<Self> {
        let base = url
            .map(|value| value.trim().trim_end_matches('/').to_string())
            .filter(|value| !value.is_empty())
            .or_else(|| {
                key.map(|value| value.trim().to_string())
                    .filter(|value| !value.is_empty())
                    .map(|value| format!("https://api.day.app/{value}"))
            })?;
        if !base.starts_with("https://") && !base.starts_with("http://") {
            return None;
        }
        Some(Self { base })
    }

    /// Send one notification. The body is percent-encoded into the path, which
    /// is Bark's documented form for short messages.
    pub async fn send(&self, title: &str, body: &str) -> Result<(), String> {
        let url = format!("{}/{}/{}", self.base, urlencode(title), urlencode(body));
        let response = reqwest::Client::new()
            .get(&url)
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await
            .map_err(|_| "bark request failed".to_string())?;
        if response.status().is_success() {
            Ok(())
        } else {
            Err(format!("bark returned {}", response.status().as_u16()))
        }
    }
}

/// Percent-encode everything outside the unreserved set.
fn urlencode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(*byte as char)
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

/// Current unix time in seconds.
pub fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_secs() as i64)
        .unwrap_or(0)
}

/// One sweep: warn about every account that has entered a new stage.
///
/// Returns the ids that were warned about so the caller can log a summary.
pub async fn sweep(store: &Store, bark: Option<&Bark>) -> Vec<String> {
    let now = now();
    let Ok(rows) = store.chatgpt_accounts() else {
        return Vec::new();
    };
    let mut warned = Vec::new();
    for row in rows {
        let Some(stage) = due_stage(&row, now) else {
            continue;
        };
        let who = row
            .label
            .clone()
            .or_else(|| row.email.clone())
            .unwrap_or_else(|| row.id.clone());
        let remaining = row
            .expires_at
            .map(|expires| duration_text((expires - now).max(0)))
            .unwrap_or_else(|| "unknown".to_string());
        let title = format!("ChatGPT token expires in {remaining}");
        let body =
            format!("{who} ({stage} warning). Add a fresh token with: deepseek-rs chatgpt add");
        match bark {
            Some(bark) => match bark.send(&title, &body).await {
                Ok(()) => {
                    // Only record the stage once it was actually delivered, so a
                    // failed push is retried on the next sweep.
                    let _ = store.record_chatgpt_notice(&row.id, stage);
                    warned.push(row.id.clone());
                }
                Err(error) => {
                    tracing::warn!(account = %row.id, %error, "bark notification failed");
                }
            },
            None => {
                tracing::warn!(
                    account = %row.id,
                    who = %who,
                    stage,
                    "ChatGPT token is expiring; configure BARK_URL or BARK_KEY for push alerts"
                );
                let _ = store.record_chatgpt_notice(&row.id, stage);
                warned.push(row.id.clone());
            }
        }
    }
    warned
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a JWT-shaped token with the given payload. The signature is a
    /// placeholder: nothing here verifies it.
    fn token(payload: &str) -> String {
        use base64::Engine;
        let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;
        format!(
            "{}.{}.{}",
            engine.encode(b"{\"alg\":\"none\"}"),
            engine.encode(payload.as_bytes()),
            engine.encode(b"signature")
        )
    }

    #[test]
    fn decodes_exp_and_iat_without_verifying_the_signature() {
        let jwt = token(r#"{"exp":1900000000,"iat":1800000000}"#);
        let times = token_times(&jwt);
        assert_eq!(times.expires_at, Some(1_900_000_000));
        assert_eq!(times.issued_at, Some(1_800_000_000));
    }

    #[test]
    fn an_unparseable_token_yields_no_times_instead_of_panicking() {
        for value in ["", "not-a-token", "a.b", "a.!!!.c", "a.bm90IGpzb24.c"] {
            let times = token_times(value);
            assert_eq!(times.expires_at, None, "{value}");
            assert_eq!(times.issued_at, None, "{value}");
        }
    }

    #[test]
    fn reads_the_openai_email_claim() {
        let jwt = token(r#"{"https://api.openai.com/profile":{"email":"a@example.test"}}"#);
        assert_eq!(token_email(&jwt).as_deref(), Some("a@example.test"));
        assert_eq!(token_email("nope"), None);
    }

    #[test]
    fn ids_are_stable_per_token_and_distinct_across_tokens() {
        let first = account_id("token-one");
        assert_eq!(first, account_id("token-one"));
        assert_ne!(first, account_id("token-two"));
        assert!(first.starts_with("cg_"));
        // The id must not leak the token.
        assert!(!first.contains("token-one"));
    }

    #[test]
    fn describes_remaining_life() {
        let now = 1_000_000;
        assert_eq!(describe_expiry(Some(now + 90), now), "1m");
        assert_eq!(describe_expiry(Some(now + 3 * 3600 + 60), now), "3h 1m");
        assert_eq!(describe_expiry(Some(now + 3 * 86_400 + 3600), now), "3d 1h");
        assert_eq!(describe_expiry(Some(now - 60), now), "expired 1m ago");
        assert!(describe_expiry(None, now).contains("unknown"));
    }

    fn row(expires_in: i64, notified: Option<&str>) -> ChatGptAccountRow {
        ChatGptAccountRow {
            id: "cg_test".into(),
            label: None,
            email: Some("a@example.test".into()),
            plan: None,
            access_token: "secret".into(),
            expires_at: Some(1_000_000 + expires_in),
            proxy_url: None,
            notified_stage: notified.map(str::to_string),
            notified_at: None,
            created_at: 0,
            updated_at: 0,
        }
    }

    #[test]
    fn only_the_tightest_new_stage_is_announced() {
        let now = 1_000_000;
        // Far from expiry: nothing to say.
        assert_eq!(due_stage(&row(100 * 3600, None), now), None);
        // Inside 72h.
        assert_eq!(due_stage(&row(70 * 3600, None), now), Some("72h"));
        // A server that was down past two stages sends the tightest one.
        assert_eq!(due_stage(&row(3600, None), now), Some("2h"));
        // Already announced at that stage: stay quiet.
        assert_eq!(due_stage(&row(3600, Some("2h")), now), None);
        // Announced at a looser stage, now tighter.
        assert_eq!(due_stage(&row(3600, Some("72h")), now), Some("2h"));
        // No exp claim at all.
        let mut unknown = row(3600, None);
        unknown.expires_at = None;
        assert_eq!(due_stage(&unknown, now), None);
    }

    #[test]
    fn bark_accepts_a_full_url_or_a_key_and_rejects_bad_input() {
        assert!(Bark::resolve(Some("https://api.day.app/key".into()), None).is_some());
        assert!(Bark::resolve(None, Some("key".into())).is_some());
        assert!(Bark::resolve(Some("  ".into()), Some("  ".into())).is_none());
        assert!(Bark::resolve(Some("ftp://example.test".into()), None).is_none());
        // The key form defaults to the Bark host.
        let bark = Bark::resolve(None, Some("abc".into())).unwrap();
        assert_eq!(bark.base, "https://api.day.app/abc");
    }

    #[test]
    fn urls_are_percent_encoded_so_the_body_cannot_break_the_path() {
        assert_eq!(urlencode("ok-._~"), "ok-._~");
        assert_eq!(urlencode("a b/c"), "a%20b%2Fc");
        assert_eq!(urlencode("答案"), "%E7%AD%94%E6%A1%88");
    }

    #[test]
    fn a_row_never_prints_its_token() {
        let text = format!("{:?}", row(3600, None));
        assert!(!text.contains("secret"), "{text}");
        assert!(text.contains("REDACTED"), "{text}");
    }
}
