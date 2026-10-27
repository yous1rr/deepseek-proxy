//! ChatGPT Sentinel's SHA3-512 proof-of-work and legacy requirements token.
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use serde_json::{json, Value};
use std::time::{SystemTime, UNIX_EPOCH};

pub const DEFAULT_SCRIPT: &str = "https://chatgpt.com/backend-api/sentinel/sdk.js";
pub const MAX_POW_ITERATIONS: u64 = 500_000;

#[derive(Debug, Clone)]
pub struct PowResources {
    pub scripts: Vec<String>,
    pub data_build: String,
}

/// Lightweight HTML attribute scanner: only reads script `src` and `data-build`;
/// it never downloads or evaluates the referenced JavaScript.
pub fn parse_pow_resources(html: &str) -> PowResources {
    let mut scripts = Vec::new();
    let lower = html.to_ascii_lowercase();
    let mut from = 0;
    while let Some(rel) = lower[from..].find("<script") {
        let start = from + rel + 7;
        let Some(end_rel) = html[start..].find('>') else {
            break;
        };
        let end = start + end_rel;
        let attrs = parse_attrs(&html[start..end]);
        if let Some(src) = attrs
            .get("src")
            .filter(|s| !s.is_empty() && s.len() <= 2048)
        {
            if !src.chars().any(char::is_control) {
                scripts.push(src.clone());
            }
        }
        from = end + 1;
    }
    if scripts.is_empty() {
        scripts.push(DEFAULT_SCRIPT.to_owned());
    }
    let mut data_build = String::new();
    for src in &scripts {
        if let Some(i) = src.find("c/") {
            if let Some(end) = src[i..].find("/_") {
                data_build = src[i..i + end + 2].to_owned();
                break;
            }
        }
    }
    if data_build.is_empty() {
        if let Some(i) = lower.find("<html") {
            if let Some(end) = html[i..].find('>') {
                data_build = parse_attrs(&html[i..i + end])
                    .remove("data-build")
                    .unwrap_or_default();
            }
        }
    }
    PowResources {
        scripts,
        data_build,
    }
}

fn parse_attrs(mut input: &str) -> std::collections::HashMap<String, String> {
    let mut attrs = std::collections::HashMap::new();
    while !input.is_empty() {
        input = input.trim_start();
        if input.is_empty() || input.starts_with('/') {
            break;
        }
        let n = input
            .find(|c: char| c.is_whitespace() || c == '=' || c == '/')
            .unwrap_or(input.len());
        if n == 0 {
            input = &input[1..];
            continue;
        }
        let name = input[..n].to_ascii_lowercase();
        input = &input[n..];
        input = input.trim_start();
        if !input.starts_with('=') {
            continue;
        }
        input = input[1..].trim_start();
        if input.is_empty() {
            break;
        }
        let (value, rest) = if input.starts_with('"') || input.starts_with('\'') {
            let q = input.as_bytes()[0] as char;
            if let Some(i) = input[1..].find(q) {
                (&input[1..i + 1], &input[i + 2..])
            } else {
                (&input[1..], "")
            }
        } else {
            let i = input.find(char::is_whitespace).unwrap_or(input.len());
            (&input[..i], &input[i..])
        };
        attrs.insert(name, value.to_owned());
        input = rest;
    }
    attrs
}

pub fn legacy_config(user_agent: &str, resources: &PowResources) -> Value {
    const NAV: &[&str] = &[
        "registerProtocolHandler−function registerProtocolHandler() { [native code] }",
        "storage−[object StorageManager]",
        "locks−[object LockManager]",
        "appCodeName−Mozilla",
        "permissions−[object Permissions]",
        "share−function share() { [native code] }",
        "webdriver−false",
        "managed−[object NavigatorManagedData]",
        "canShare−function canShare() { [native code] }",
        "vendor−Google Inc.",
        "mediaDevices−[object MediaDevices]",
        "vibrate−function vibrate() { [native code] }",
        "storageBuckets−[object StorageBucketManager]",
        "mediaCapabilities−[object MediaCapabilities]",
        "cookieEnabled−true",
        "virtualKeyboard−[object VirtualKeyboard]",
        "product−Gecko",
        "presentation−[object Presentation]",
        "onLine−true",
        "mimeTypes−[object MimeTypeArray]",
        "credentials−[object CredentialsContainer]",
        "serviceWorker−[object ServiceWorkerContainer]",
        "keyboard−[object Keyboard]",
        "gpu−[object GPU]",
        "doNotTrack",
        "serial−[object Serial]",
        "pdfViewerEnabled−true",
        "language−zh-CN",
        "geolocation−[object Geolocation]",
        "userAgentData−[object NavigatorUAData]",
        "getUserMedia−function getUserMedia() { [native code] }",
        "sendBeacon−function sendBeacon() { [native code] }",
        "hardwareConcurrency−32",
        "windowControlsOverlay−[object WindowControlsOverlay]",
    ];
    const WINDOWS: &[&str] = &[
        "0",
        "window",
        "self",
        "document",
        "name",
        "location",
        "customElements",
        "history",
        "navigation",
        "innerWidth",
        "innerHeight",
        "scrollX",
        "scrollY",
        "visualViewport",
        "screenX",
        "screenY",
        "outerWidth",
        "outerHeight",
        "devicePixelRatio",
        "screen",
        "chrome",
        "navigator",
        "onresize",
        "performance",
        "crypto",
        "indexedDB",
        "sessionStorage",
        "localStorage",
        "scheduler",
        "alert",
        "atob",
        "btoa",
        "fetch",
        "matchMedia",
        "postMessage",
        "queueMicrotask",
        "requestAnimationFrame",
        "setInterval",
        "setTimeout",
        "caches",
        "__NEXT_DATA__",
        "__BUILD_MANIFEST",
        "__NEXT_PRELOADREADY",
    ];
    const DOC: &[&str] = &[
        "__reactContainer$fzelfjyxej8",
        "_reactListening5dehydibo78",
        "location",
    ];
    const CORES: &[u64] = &[8, 16, 24, 32];
    const SCREENS: &[u64] = &[3000, 2340, 4000, 6000];
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let perf = (crate::random_u64() % 100_000_000) as u64;
    json!([
        SCREENS[(crate::random_u64() as usize) % SCREENS.len()],
        legacy_time(),
        4294705152u64,
        1,
        user_agent,
        resources.scripts[(crate::random_u64() as usize) % resources.scripts.len()],
        resources.data_build,
        "en-US",
        "en-US,es-US,en,es",
        random_fraction(),
        NAV[(crate::random_u64() as usize) % NAV.len()],
        DOC[(crate::random_u64() as usize) % DOC.len()],
        WINDOWS[(crate::random_u64() as usize) % WINDOWS.len()],
        perf as f64,
        crate::new_id(),
        "",
        CORES[(crate::random_u64() as usize) % CORES.len()],
        now_ms.saturating_sub(perf) as f64,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    ])
}

fn random_fraction() -> f64 {
    (crate::random_u64() >> 11) as f64 / ((1u64 << 53) as f64)
}
fn legacy_time() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .saturating_sub(5 * 3600);
    let days = (seconds / 86_400) as i64;
    let rem = seconds % 86_400;
    let (y, m, d) = civil_from_days(days);
    let months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    format!(
        "{} {} {:02} {:04} {:02}:{:02}:{:02} GMT-0500 (Eastern Standard Time)",
        weekdays[((days + 3).rem_euclid(7)) as usize],
        months[(m - 1) as usize],
        d,
        y,
        rem / 3600,
        (rem / 60) % 60,
        rem % 60
    )
}
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let mut y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    if m <= 2 {
        y += 1;
    }
    (y, m, d)
}

pub fn legacy_token(config: &Value) -> Result<String, PowError> {
    let bytes = serde_json::to_vec(config).map_err(|_| PowError::Invalid)?;
    Ok(format!("gAAAAAC{}", B64.encode(bytes)))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowError {
    Invalid,
    Unsolved,
}

/// Return the exact Python `pow.py` SHA3-512 proof token (`gAAAAAB` + base64 answer).
pub fn proof_token(seed: &str, difficulty: &str, config: &Value) -> Result<String, PowError> {
    let target = decode_hex(difficulty).ok_or(PowError::Invalid)?;
    let items = config.as_array().ok_or(PowError::Invalid)?;
    if target.is_empty() || target.len() > 64 || items.len() < 10 {
        return Err(PowError::Invalid);
    }
    for nonce in 0..MAX_POW_ITERATIONS {
        let mut candidate = Vec::with_capacity(items.len());
        candidate.extend_from_slice(&items[..3]);
        candidate.push(json!(nonce));
        candidate.extend_from_slice(&items[4..9]);
        candidate.push(json!(nonce >> 1));
        candidate.extend_from_slice(&items[10..]);
        let encoded = B64
            .encode(serde_json::to_vec(&Value::Array(candidate)).map_err(|_| PowError::Invalid)?);
        let mut input = Vec::with_capacity(seed.len() + encoded.len());
        input.extend_from_slice(seed.as_bytes());
        input.extend_from_slice(encoded.as_bytes());
        let digest = sha3_512(&input);
        if digest[..target.len()] <= target[..] {
            return Ok(format!("gAAAAAB{encoded}"));
        }
    }
    Err(PowError::Unsolved)
}
fn decode_hex(s: &str) -> Option<Vec<u8>> {
    if s.is_empty() || s.len() % 2 != 0 || !s.is_ascii() {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok())
        .collect()
}

pub fn sha3_512(data: &[u8]) -> [u8; 64] {
    const RATE: usize = 72;
    let mut state = [0u64; 25];
    let mut blocks = data.chunks_exact(RATE);
    for block in &mut blocks {
        absorb(&mut state, block);
        keccak_f1600(&mut state);
    }
    let rest = blocks.remainder();
    let mut final_block = [0u8; RATE];
    final_block[..rest.len()].copy_from_slice(rest);
    final_block[rest.len()] = 0x06;
    final_block[RATE - 1] |= 0x80;
    absorb(&mut state, &final_block);
    keccak_f1600(&mut state);
    let mut out = [0u8; 64];
    for (i, word) in out.chunks_exact_mut(8).enumerate() {
        word.copy_from_slice(&state[i].to_le_bytes());
    }
    out
}
fn absorb(state: &mut [u64; 25], block: &[u8]) {
    for (i, bytes) in block.chunks_exact(8).enumerate() {
        state[i] ^= u64::from_le_bytes(bytes.try_into().expect("eight-byte word"));
    }
}
fn keccak_f1600(a: &mut [u64; 25]) {
    const RC: [u64; 24] = [
        0x1,
        0x8082,
        0x800000000000808a,
        0x8000000080008000,
        0x808b,
        0x80000001,
        0x8000000080008081,
        0x8000000000008009,
        0x8a,
        0x88,
        0x80008009,
        0x8000000a,
        0x8000808b,
        0x800000000000008b,
        0x8000000000008089,
        0x8000000000008003,
        0x8000000000008002,
        0x8000000000000080,
        0x800a,
        0x800000008000000a,
        0x8000000080008081,
        0x8000000000008080,
        0x80000001,
        0x8000000080008008,
    ];
    const ROT: [u32; 25] = [
        0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56,
        14,
    ];
    for rc in RC {
        let mut c = [0u64; 5];
        for x in 0..5 {
            c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
        }
        for x in 0..5 {
            let d = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
            for y in 0..5 {
                a[x + 5 * y] ^= d;
            }
        }
        let mut b = [0u64; 25];
        for x in 0..5 {
            for y in 0..5 {
                let i = x + 5 * y;
                b[y + 5 * ((2 * x + 3 * y) % 5)] = a[i].rotate_left(ROT[i]);
            }
        }
        for y in 0..5 {
            for x in 0..5 {
                a[x + 5 * y] = b[x + 5 * y] ^ ((!b[(x + 1) % 5 + 5 * y]) & b[(x + 2) % 5 + 5 * y]);
            }
        }
        a[0] ^= rc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sha3_512_known_python_vector() {
        assert_eq!(hex(&sha3_512(b"abc")), "b751850b1a57168a5693cd924b6b096e08f621827444f70d884f5d0240d2712e10e116e9192af3c91a7ec57647e3934057340b4cf408d5a56592f8274eec53f0");
    }
    #[test]
    fn python_pow_candidate_layout() {
        let c = json!([
            3000,
            "date",
            4294705152u64,
            1,
            "ua",
            "src",
            "build",
            "en-US",
            "langs",
            0.3,
            "nav",
            "doc",
            "window",
            1.0,
            "id",
            "",
            8,
            2.0,
            0,
            0,
            0,
            0,
            0,
            0,
            0
        ]);
        let token = proof_token(
            "seed",
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            &c,
        )
        .unwrap();
        let bytes = B64.decode(&token[7..]).unwrap();
        assert_eq!(String::from_utf8(bytes).unwrap(), "[3000,\"date\",4294705152,0,\"ua\",\"src\",\"build\",\"en-US\",\"langs\",0,\"nav\",\"doc\",\"window\",1.0,\"id\",\"\",8,2.0,0,0,0,0,0,0,0]");
    }
    #[test]
    fn parses_resource_urls() {
        let r = parse_pow_resources(
            "<html data-build=\"c/page/_\"><script src=\"/c/app/_build.js\"></script>",
        );
        assert_eq!(r.scripts, vec!["/c/app/_build.js"]);
        assert_eq!(r.data_build, "c/app/_");
    }
    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }
}
