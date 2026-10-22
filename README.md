# deepseek-rs

An OpenAI-compatible API server for DeepSeek. One binary, no runtime.

DeepSeek's chat has no public API. This server exposes the standard OpenAI
`/v1/chat/completions` surface, translates each request into what DeepSeek's
web client actually sends — login, proof-of-work, server-side sessions,
streamed tokens — and returns ordinary OpenAI-compatible SSE.

Point any OpenAI client at it and it works.

```bash
curl http://127.0.0.1:9174/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"hello"}],"stream":true}'
```

## Features

- **OpenAI-compatible streaming** — `/v1/chat/completions` and `/v1/models`,
  with `tool_calls` deltas and usage accounting. `stream: true` is required;
  non-streaming JSON responses are not supported.
- **Tool calling** — JSON-Schema validated arguments, and a bounded repair turn
  when the model emits a malformed call instead of surfacing the error.
- **Branch-aware sessions** — resume matching histories with a delta; replay the
  selected history after edits, regeneration, or revision changes without client changes.
- **Multiple accounts** — several DeepSeek accounts behind one endpoint, with
  per-account rate gating, health tracking, and failover.
- **Long threads** — if the backend reports a context limit, the session is
  rebuilt and the transcript replayed once, transparently.
- **Vision** — one image per turn, from a `data:` URL or a public HTTP URL.
- **Persistent state** — thread bindings, API-key hashes, and cached login
  tokens live in SQLite and survive restarts.
- **Single binary** — `cargo build --release`, done. No interpreter, no
  container, no virtualenv.

## Quick start

```bash
cargo build --release

export DEEPSEEK_EMAIL_0='you@example.com'
export DEEPSEEK_PASSWORD_0='your-password'

./target/release/deepseek-rs serve
```

Listens on `0.0.0.0:9174`; the base URL for clients is
`http://127.0.0.1:9174/v1`.

Run it detached instead — `start` returns only once the server answers
`/healthz`, so it is safe to chain on:

```bash
deepseek-rs start --bg
deepseek-rs status
deepseek-rs logs -f
```

## Configuration

Environment beats `config.toml` beats built-in defaults. Every key is optional.
`CONFIG_PATH` selects a different file; a missing file is not an error.

```toml
[proxy]
host = "0.0.0.0"
port = 9174

[storage]
# Relative paths resolve against this file's directory, not the process cwd.
db_path = "deepseek-rs.db"

[deepseek]
model    = "deepseek-chat"   # used when the client omits one
thinking = false             # default reasoning flag
search   = false             # web search (heavily throttled upstream)

[limits]
request_delay_secs = 2       # pause between requests on one account
queue_limit        = 64      # requests allowed to queue behind a busy account
queue_timeout_secs = 120     # how long one may wait for a slot
request_timeout_secs = 300   # per-turn wall clock
heartbeat_interval_secs = 10 # SSE keep-alive
rate_limit_max_retries = 5   # retries for a throttled turn
rate_limit_backoff_secs = 10.0
tool_repair_max_retries = 5  # 0 disables repair
image_max_bytes = 10485760   # 10 MiB, on bytes and base64 text
```

Frequently used environment overrides: `PROXY_HOST`, `PROXY_PORT`, `DB_PATH`,
`CONFIG_PATH`, `DEEPSEEK_RUNTIME_DIR`, `MODEL_NAME`, `THINKING_ENABLED`,
`SEARCH_ENABLED`, `REQUEST_DELAY`, `QUEUE_LIMIT`, `QUEUE_TIMEOUT`,
`REQUEST_TIMEOUT`, `HEARTBEAT_INTERVAL`, `IMAGE_MAX_BYTES`,
`RATE_LIMIT_MAX_RETRIES`, `RATE_LIMIT_BACKOFF_S`, `TOOL_REPAIR_MAX_RETRIES`.

Credentials never go in `config.toml`.

## Accounts

Two sources, and the environment wins on an email collision.

```bash
# environment
export DEEPSEEK_EMAIL_0='a@example.com'
export DEEPSEEK_PASSWORD_0='...'
export DEEPSEEK_TOKEN_1='...'     # an existing bearer token also works

# or the database
deepseek-rs accounts add a@example.com 'password'
deepseek-rs accounts add b@example.com --token '...'
deepseek-rs accounts
deepseek-rs accounts rm a@example.com
```

Login tokens are cached and reused across restarts. A token the server rejects
is dropped and fetched again on the next attempt, so a stale cache heals
itself. An explicit `DEEPSEEK_TOKEN_<i>` always wins.

`deepseek-rs check` verifies login for every configured account and probes the
PoW challenge path, exiting non-zero on failure.

## Client authentication

```bash
deepseek-rs keys add laptop sk-my-key
deepseek-rs keys
deepseek-rs keys rm laptop
```

Keys are stored as SHA-256 hashes; plaintext is never written. With **no** keys
registered, authentication is open — fine on a trusted loopback bind, worth
closing before exposing the port.

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/chat/completions` | Streaming completions (`stream: true`) |
| `GET` | `/v1/models` | Advertised model ids |
| `GET` | `/healthz` | Liveness |
| `GET` | `/readyz` | Readiness |
| `GET` | `/stats` | Counters, per-account state, database counts |

`/stats` is what `deepseek-rs stats` and `status` read:

```json
{"pid":0,"uptime_secs":0,"model":"deepseek-chat","thinking":false,"search":false,
 "requests":{"total":0,"ok":0,"errors":0,"auth_failed":0,"retries":0,
             "tool_calls":0,"images_dropped":0},
 "threads":{"in_memory":0},
 "db":{"threads":0,"bindings":0,"api_keys":0,"accounts":0,"cached_tokens":0},
 "accounts":[{"email":"...","healthy":true,"bound_threads":0,"waiting":0,"cooldown_ms":null}]}
```

With the server down, `stats` falls back to database counts and says so on
stderr.

## CLI

```
SERVER
  serve                       foreground (Ctrl-C stops)
  start [-b|--bg]             foreground by default; -b detaches, waits until healthy
  stop                        SIGTERM (SIGKILL after 15s)
  restart [-b]                stop, then start
  status                      pid, url, uptime, account count
  health                      one-shot check (exit 1 when down)
  stats [--json]              counters, per-account state, db counts

LOGS
  logs [-n N] [-f] [--path P] last N lines (default 50); -f follows

ACCOUNTS
  accounts | accounts add <email> <password> | accounts add <email> --token <t> | accounts rm <email>

API KEYS
  keys | keys add <name> <key> | keys rm <name|id>

MISC
  check | url | version | help
```

Background mode keeps a pidfile and a log under `DEEPSEEK_RUNTIME_DIR`
(default `${XDG_STATE_HOME:-~/.local/state}/deepseek-rs`). `serve` claims the
pidfile too, so a second instance fails immediately instead of contending for
the database lock.

## Images

One image per turn, inline (`data:image/png;base64,...`) or as a public HTTP(S)
URL. It is uploaded once and referenced by file id, never re-sent on later
turns.

Attach several and the server sends the first, **drops the rest**, and tells the
model how many it lost so they can be re-attached one at a time. A hard
rejection would be worse: clients rebuild the same request on retry, so the turn
would never complete and the conversation would wedge. The dropped count is in
`/stats` as `requests.images_dropped`.

Remote URLs must resolve entirely to public addresses — loopback, link-local,
private, CGNAT, and reserved ranges are refused, and redirects are not followed.

## Behaviour worth knowing

- **Session keys** prefer body `thread_id`, then `x-thread-id`, then
  `x-session-affinity`, `x-opencode-session`, `x-session-id`, and `session_id`.
  Without an explicit identity the first user message supplies a fallback hash.
  Distinct conversations should send distinct session keys.
- **Context limits are recovered, not reported.** A limit on a continuation
  triggers a fresh remote session and one transcript replay. Only before any
  content is sent, and only for existing threads.
- **Tool calls are repaired.** A malformed or schema-violating call gets a
  bounded correction turn instead of an error to the client.
- **Rate limits retry before the first byte**, so streaming clients can still be
  served; what cannot be retried fails with a typed error.
- **Stale cursors rebuild the session** from the transcript once rather than
  failing the turn.

## Session branches

Send the complete effective `messages` history for the branch you selected,
with `stream: true` and a stable session header. Pi Desktop already sends this
shape for edit/resend, regeneration and revision switching. No client patch or
new branch header is required.

The proxy compares a canonical history fingerprint against the last successful
head before reusing the remote session and cursor. Ordinary continuation sends
only the new input upstream. Rewinding, editing earlier messages, regenerating
the same request, changing the model/tool contract, or receiving a compacted
history starts a fresh DeepSeek session and replays the selected history. A
new Pi Desktop fork uses its own session key and is an independent thread.

An additive SQLite `thread_history` table stores only the fingerprint, account
identity and message count. Successful checkpoints survive restart; old
databases replay once before establishing a trusted checkpoint. Incomplete or
failed streams do not establish a trusted head. Sessions and cursors stay on
one account for the entire turn; an authentication failure may rebind the next
request, which then replays on that account instead of reusing a foreign cursor.

This release reconstructs selected branches; it does not reuse arbitrary old
DeepSeek parent cursors or expose a branch-management API. Delta-only client
histories cannot identify a rewind reliably and may cause a fresh replay.
Replay retains the existing one-image-per-prompt limit and reports dropped
attachments.

Run the synthetic live verification against an isolated proxy and database copy:

```bash
python tools/verify_branches.py --base-url http://127.0.0.1:19176/v1 --db /path/to/test-copy.db
```

Use `--api-key-env NAME` when that test instance requires a key. The verifier
uses read-only SQLite access and never starts the server or edits credentials.
See [the implementation plan](BRANCH_IMPLEMENTATION_PLAN.md) and
[portable-client evidence](PI_DESKTOP_BRANCH_VALIDATION.md).

## Architecture

```
crates/pow       DeepSeekHashV1 + multithreaded nonce search
crates/protocol  SSE parsing + fragment routing + OpenAI chunk shaping
crates/client    login, session, PoW header, streaming, image upload
crates/proxy     axum server, tool dialect, SQLite storage, auth, retry
corpus/          captured SSE streams used as test fixtures
```

Dependencies point strictly downward: `proxy` → `client` → `protocol` → `pow`.
Nothing below `proxy` knows about HTTP serving.

## DeepSeekHashV1

Requests are gated behind a proof-of-work that is *almost* SHA3-256: the
Keccak-f[1600] permutation skips round 0 and runs rounds 1..=23. No off-the-shelf
SHA3 crate produces this digest, so the permutation lives in
`crates/pow/src/lib.rs`; round constants past `RC[0]` are standard Keccak.

## Development

```bash
cargo build --release
cargo test              # includes byte-for-byte corpus equality
cargo test -p proxy     # server-side behaviour only
```

The `corpus/` fixtures are raw byte streams captured from DeepSeek's completion
endpoint, paired with the routing output each should produce. They are what
keeps the stream parser honest.

### Logging

| | prod (default) | debug (`DEEPSEEK_LOG=debug`) |
|---|---|---|
| spans | hidden | shown |
| target / file / line | omitted | shown |
| thread ids | omitted | shown |
| ANSI | only when stdout is a TTY | only when stdout is a TTY |

`RUST_LOG` overrides the filter entirely (blank is ignored rather than meaning
"log nothing"); `NO_COLOR` or `DEEPSEEK_LOG_ANSI=1|0` forces colour off or on.

## License

MIT. See [LICENSE](LICENSE).
