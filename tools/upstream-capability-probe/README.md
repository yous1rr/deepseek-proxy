# upstream-capability-probe

A dependency-free Python tool that safely probes an OpenAI-style upstream
(any server exposing `/v1/chat/completions` and/or `/v1/responses`) and writes
a machine-readable **capability profile**. The profile is meant to drive
assembly decisions in a proxy: which wire protocol to use, whether streaming
is available, whether tool calling needs local emulation, and which optional
fields can be trusted.

The tool is deliberately conservative:

- **Synthetic probes only** — fixed, tiny prompts and a 1×1 PNG; no user data.
- **Redacted profile** — no prompts, completion text, response bodies, headers,
  tokens, response IDs, or authentication values are stored.
- **Bounded reads** — JSON bodies are capped at 2 MiB, SSE bodies at 256 KiB.
- **No redirects** — `Authorization` is never forwarded to another origin.
- **No retries, no destructive calls** — at most one request per probe step.
- **HTTP/HTTPS only** — embedded credentials and other schemes are rejected.

> **Scope:** this tool measures HTTP capability, not semantic equivalence.
> A `supported` verdict means the upstream accepted a synthetic request and
> returned the expected structural shape. It does not prove full OpenAI API
> compatibility, correct tool-call semantics, or identical model behaviour.

## Requirements

Python 3.10+ and the standard library only. No installation is required.

## Quick start

From this directory:

```bash
# Read-only inventory (no model generation): GET /models only
PYTHONPATH=src python -m upstream_capability_probe \
  --base-url https://api.example.com \
  --api-key-env UPSTREAM_API_KEY \
  --model gpt-4o-mini \
  --no-write-probes \
  --output profile.readonly.json

# Full probe set plus a converted Rust proxy capability profile
PYTHONPATH=src python -m upstream_capability_probe \
  --base-url https://api.example.com/v1 \
  --api-key-env UPSTREAM_API_KEY \
  --model gpt-4o-mini \
  --timeout 45 \
  --output profile.json \
  --proxy-profile proxy-profile.json \
  --channel openai_compat

# Offline conversion of an existing probe profile (no request is sent)
PYTHONPATH=src python -m upstream_capability_probe \
  --from-profile profile.json \
  --proxy-profile proxy-profile.json \
  --channel openai_compat
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
$env:UPSTREAM_API_KEY = "..."  # never pass the key as an argument
python -m upstream_capability_probe --base-url https://api.example.com --api-key-env UPSTREAM_API_KEY --no-write-probes
```

`--output -` (the default) writes the JSON profile to stdout and progress
lines to stderr. Progress lines contain only probe id, status, and reason.

### CLI reference

| Flag | Default | Meaning |
| --- | --- | --- |
| `--base-url` | required for probe runs | API root. `https://host`, `https://host/v1`, `https://host/openai/v1`, or a full `/chat/completions`, `/responses`, `/models` URL are accepted. Not used with `--from-profile`. |
| `--api-key-env` | unset | Name of the environment variable holding the bearer token. If unset, requests are unauthenticated. |
| `--model` | `gpt-4o-mini` | Model name placed in probe payloads. |
| `--output` | `-` | Probe profile path; `-` means stdout. |
| `--timeout` | `30` | Per-request timeout in seconds (0–3600). |
| `--no-write-probes` | off | Skip all `stateful` probes; run only `read_only` probes. |
| `--proxy-profile` | unset | Also write a converted Rust proxy profile (`responses-capability-profile`). Requires `--channel`. |
| `--channel` | unset | Channel id for the proxy profile: 1–64 chars of `[A-Za-z0-9_.-]`. Required with `--proxy-profile`, invalid without it. |
| `--from-profile` | unset | Offline mode: read an existing probe profile JSON (path or `-` for stdin), convert it, and exit without sending any request. Requires `--proxy-profile`; rejects `--base-url`/`--output`. |

Exit codes: `0` normal completion (including unsupported/inconclusive
results), `1` output write failure, `2` every executed probe failed at the
transport level or argument parsing failed.

### Base URL normalization

| Input | API root | `chat/completions` URL |
| --- | --- | --- |
| `https://h` | `/v1` | `https://h/v1/chat/completions` |
| `https://h/` | `/v1` | `https://h/v1/chat/completions` |
| `https://h/v1` | `/v1` | `https://h/v1/chat/completions` |
| `https://h/openai/v1` | `/openai/v1` | `https://h/openai/v1/chat/completions` |
| `https://h/v1/chat/completions` | `/v1` | `https://h/v1/chat/completions` |
| `https://h/proxy` | `/proxy` | `https://h/proxy/chat/completions` |

Query strings are preserved on request URLs but never echoed into the profile.

## Probe catalog

`read_only` probes never invoke the model. `stateful` probes trigger
generation and may create provider-side history or cost, so they are skipped
with `--no-write-probes`.

| Probe id | Kind | Capability | What it checks |
| --- | --- | --- | --- |
| `models.list` | read_only | `models_list` | `GET /models` returns a model list |
| `chat.completions.non_stream` | stateful | `chat_completions_non_stream` | Non-streaming Chat Completions shape: `choices[0].message`, finish reason, `usage` |
| `chat.completions.stream` | stateful | `chat_completions_stream` | `stream: true` returns `text/event-stream` with data events; derived: SSE `[DONE]`, named events |
| `chat.completions.function_calling` | stateful | `chat_completions_function_calling` | A `tools` + forced `tool_choice` request returns a valid `tool_calls` entry |
| `chat.completions.tool_result_continuation` | stateful | `chat_completions_tool_result_continuation` | A tool result sent back in a second turn produces a normal assistant message |
| `chat.completions.vision_data_uri` | stateful | `chat_completions_vision_data_uri` | A `data:image/png` part is accepted in `image_url` |
| `responses.non_stream` | stateful | `responses_non_stream` | `/responses` returns a `response` object with non-empty `output` |
| `responses.stream` | stateful | `responses_stream` | `/responses` with `stream: true` returns SSE; derived: terminal event, named events |
| `responses.max_output_tokens` | stateful | `responses_max_output_tokens` | A 16-token limit causes an explicit limit-related incomplete response with consistent usage within the cap |
| `responses.function_calling` | stateful | `responses_function_calling` | A Responses `function` tool returns a `function_call` output item |
| `responses.tool_result_continuation` | stateful | `responses_tool_result_continuation` | `function_call_output` continuation is accepted |
| `responses.vision_data_uri` | stateful | `responses_vision_data_uri` | `input_image` with a data URI is accepted |

Derived capabilities reported alongside their parent probe:

- `chat_completions_sse_done` — Chat SSE contains `data: [DONE]`.
- `chat_completions_sse_named_events` — Chat SSE uses `event:` names.
- `responses_sse_done` — Responses SSE exposes a terminal event
  (`response.completed` / `response.failed` / `response.incomplete`).
- `responses_sse_named_events` — Responses SSE uses `event:` names.
- `chat_completions_usage` / `responses_usage` — a `usage` object is present
  with recognized token-count fields.

### `max_output_tokens`: acceptance and enforcement

The independent `responses.max_output_tokens` probe asks for more output than
its fixed `max_output_tokens: 16` budget. Other Responses probes omit this
optional field so its rejection cannot hide text, streaming, tools, or vision
support. This tests the Responses field only; Chat's `max_tokens` and
`max_completion_tokens` are separate parameters and provide no evidence here.

- `supported`: a `response` object has `status: "incomplete"`,
  `incomplete_details.reason: "max_output_tokens"`, and consistent integer
  usage (`input_tokens >= 0`, `0 < output_tokens <= 16`,
  `total_tokens = input_tokens + output_tokens`). Optional reasoning usage
  must also be consistent; the budget can be consumed entirely by reasoning.
- `unsupported`: HTTP 400/422 explicitly names `max_output_tokens` with an
  allowlisted unsupported/unknown-parameter code, or a valid response reports
  consistent usage above the requested cap. An absent endpoint is also
  unsupported under the usual HTTP classification.
- `inconclusive`: HTTP 200 or a parameter echo without enforcement evidence,
  a completed response within budget, missing/invalid usage, generic validation
  rejection (including a larger model-specific minimum), timeout, 429, or 5xx.
  These results convert to proxy `unknown`.

For field rejection, the transport extracts only a fixed allowlisted parameter
marker from bounded JSON. It discards the error body and never retains or
reports upstream error messages. The test is one observation at one small cap;
it does not discover a model's full limit range or prove tokenizer accuracy.
`--no-write-probes` skips it. Older profiles without the probe remain `unknown`.

## Profile format

```json
{
  "schema_version": 1,
  "tool": {"name": "upstream-capability-probe", "version": "0.1.0"},
  "generated_at": "2025-01-01T00:00:00Z",
  "target": {
    "base_url": "https://api.example.com/v1",
    "auth": "bearer_env",
    "api_key_env": "UPSTREAM_API_KEY"
  },
  "options": {
    "model": "gpt-4o-mini",
    "timeout_seconds": 30.0,
    "write_probes": true,
    "read_only_probes_only": false,
    "probe_kinds": {"read_only": 1, "stateful": 11},
    "planned_probes": [{"id": "...", "title": "...", "capability": "...", "kind": "..."}]
  },
  "summary": {
    "probes": {"total": 12, "supported": 10, "unsupported": 2, "inconclusive": 0, "skipped": 0, "error": 0},
    "capabilities": {"total": 18, "supported": 13, "unsupported": 5, "inconclusive": 0, "skipped": 0, "error": 0}
  },
  "assembly_hints": {
    "native_responses_stream": {"status": "supported", "supported": true, "capability": "responses_stream"},
    "chat_completions_stream": {"status": "supported", "supported": true, "capability": "chat_completions_stream"}
  },
  "capabilities": {
    "chat_completions_stream": {"status": "supported", "supported": true, "reason": "sse_stream_observed", "probe": "chat.completions.stream"}
  },
  "probes": [
    {
      "id": "chat.completions.stream",
      "title": "Chat Completions stream=true",
      "capability": "chat_completions_stream",
      "kind": "stateful",
      "status": "supported",
      "supported": true,
      "reason": "sse_stream_observed",
      "evidence": {"is_event_stream": true, "event_count": 8, "sse_done_seen": true, "event_names": ["message"]},
      "http_status": 200
    }
  ],
  "security": {
    "contains_prompts": false,
    "contains_response_bodies": false,
    "contains_completion_text": false,
    "contains_tokens": false,
    "contains_headers": false,
    "contains_auth_values": false,
    "raw_response_retained": false
  },
  "limitations": ["..."]
}
```

### Status values

| Status | `supported` | Meaning |
| --- | --- | --- |
| `supported` | `true` | Probe observed the expected structural shape. |
| `unsupported` | `false` | The endpoint is absent, rejected the probe shape, or returned an incompatible shape. |
| `inconclusive` | `null` | The probe could not reach a verdict (timeout, rate limit, 5xx, malformed but ambiguous response). |
| `skipped` | `null` | Skipped by `--no-write-probes`. Never treat as unsupported. |
| `error` | `null` | Authentication/transport failure unrelated to capability. |

Each probe result always carries `id`, `title`, `capability`, `kind`,
`status`, `supported`, `reason`, `evidence`, `http_status` (when known), and
`error_class` (when known). `reason` is a fixed snake_case token, never
upstream text. `evidence` is an allow-listed set of structural facts:
booleans, counts, content types, event names, and JSON field-name lists.

### Safety of `evidence`

`evidence` may contain:

- counts (`event_count`, `tool_call_count`, `choices_count`, …)
- booleans (`is_event_stream`, `sse_done_seen`, `usage_present`, …)
- short validated event names (`message`, `response.completed`, …)
- allow-listed JSON field names (`usage`, `choices`, `output`, …)
- MIME types (`application/json`, `text/event-stream`)

It never contains free-form text, response content, IDs, URLs from response
bodies, or header values.

## Interpreting results for proxy assembly

Suggested consumption order (also exposed as `assembly_hints`):

1. Prefer `responses_stream` when `responses_stream == supported`, otherwise
   `chat_completions_stream`.
2. Prefer non-streaming only when the caller needs it and
   `*_non_stream == supported`.
3. Enable native `function_calling` only when the matching protocol's
   function-calling probe is `supported`; otherwise keep local tool emulation.
4. Enable `vision_data_uri` only when the matching protocol's vision probe is
   `supported`.
5. Treat `usage` capabilities as advisory: absence means do not promise exact
   usage accounting.
6. Pass `max_output_tokens` only when `responses_max_output_tokens` is
   `supported` and the active adapter implements the limit.
7. Never treat `inconclusive`, `skipped`, or `error` as `supported`.

## Rust proxy capability profile (`--proxy-profile`)

The Rust proxy's Responses endpoint optionally loads a capability profile from
the path in environment variable `RESPONSES_CAPABILITY_PROFILE` (see
[`../../RESPONSES_ARCHITECTURE.md`](../../RESPONSES_ARCHITECTURE.md)). `--proxy-profile`
converts the probe result into that exact document. `--from-profile` does the
same conversion offline from a saved probe profile.

### Format (exact)

```json
{
  "format": "responses-capability-profile",
  "schema_version": 1,
  "channel": "openai_compat",
  "generated_by": {"tool": "upstream-capability-probe", "version": "0.1.0"},
  "generated_at": "2025-01-02T03:04:05Z",
  "capabilities": {
    "text": {"support": "supported", "provider": "upstream", "evidence": "responses.stream"},
    "streaming": {"support": "supported", "provider": "upstream", "mode": "native_responses_sse", "evidence": "responses.stream"},
    "file_input": {"support": "unknown", "provider": "none"}
  }
}
```

* `format` must be `responses-capability-profile`; `schema_version` must be `1`.
  The proxy rejects a wrong format or version. Unknown top-level fields are
  ignored.
* `channel` must equal the active adapter's channel id; the proxy rejects a
  profile for a different channel.
* `support` is one of `supported` / `unsupported` / `unknown`.
* `provider` says where the capability is realized: `upstream` (pass-through
  to the probed upstream), `proxy` (implemented locally by the proxy), or
  `none` (no provider; typical for `unknown`/`unsupported`).
* `mode` is an optional short tag; `evidence` is optional comma-separated probe
  ids or a short note.
* Exactly the 30 contract ids below are always present. The proxy warns on and
  ignores any other id:
  `text`, `streaming`, `non_streaming`, `image_input`, `file_input`,
  `function_tools`, `tool_choice`, `custom_tools`, `web_search`,
  `file_search`, `code_interpreter`, `mcp`, `image_generation`,
  `computer_use`, `shell`, `reasoning`, `structured_output`, `verbosity`,
  `sampling_controls`, `max_output_tokens`, `logprobs`, `stored_responses`,
  `previous_response`, `conversations`, `background`, `stream_resume`,
  `auto_truncation`, `compaction`, `prompt_templates`, `exact_usage`.
* The document never contains `base_url`, `api_key_env`, headers, prompts,
  response bodies, tokens, or any other free-form target metadata.

### Fail-closed narrowing semantics (proxy behaviour, not changed here)

The profile **cannot enable** anything. Effective capability =
**adapter built-in AND profile**. A capability is enabled only when the
channel adapter implements it **and** the profile says `supported`.
`unsupported`, `unknown`, and absent all disable it. A profile can only narrow
what the adapter already supports.

Dependencies enforced by the proxy: `tool_choice` needs `function_tools`;
`previous_response`, `background`, and `stream_resume` need
`stored_responses`; `stream_resume` also needs `streaming`.

### Conservative probe-to-proxy mapping

Probe statuses other than `supported`/`unsupported` (`inconclusive`,
`skipped`, `error`) are treated as `unknown`.

| Proxy id | Rule | Provider / mode |
| --- | --- | --- |
| `text` | supported if any of `chat_completions_non_stream`, `chat_completions_stream`, `responses_non_stream`, `responses_stream` is supported; unsupported if all four are unsupported; else unknown | `upstream` |
| `streaming` | `responses_stream` supported → upstream; else `chat_completions_stream` supported → proxy; unsupported if both unsupported; else unknown | `native_responses_sse` / `chat_stream_reencoded` |
| `non_streaming` | `responses_non_stream` supported (upstream `native`), else `chat_completions_non_stream` supported (proxy `chat_reencoded`), else streaming supported (proxy `collected_stream`); unsupported only if all paths are unsupported; else unknown | see left |
| `image_input` | supported if either vision probe supported; unsupported if both unsupported; else unknown | `upstream` / `none` |
| `function_tools` | supported only if `responses_function_calling` AND `responses_tool_result_continuation` are supported, or `chat_completions_function_calling` AND `chat_completions_tool_result_continuation` are supported (same protocol pair); unsupported if both function-calling probes unsupported; else unknown | `upstream` / `none` |
| `stored_responses`, `previous_response` | supported if `text` is supported, else unknown | `proxy` / `local_transcript` |
| `exact_usage` | supported if `responses_usage` or `chat_completions_usage` supported; else unknown | `upstream` |
| `max_output_tokens` | independent `responses_max_output_tokens` verdict only; other statuses or absent probe → unknown; Chat token fields do not count | `upstream` when supported, else `none` |
| all other 21 ids | always `unknown` / `none`, no evidence | `none` |

`evidence` lists the probe ids that drove the decision.

### Using the converted profile when onboarding a new adapter

1. Probe the candidate upstream (or reuse a saved probe profile) and convert:
   `--proxy-profile PATH --channel <adapter-id>`.
2. Point the proxy at it: `RESPONSES_CAPABILITY_PROFILE=PATH`. The channel id
   must match the adapter you are configuring, or the proxy rejects the file.
3. Treat every `unknown` as unavailable: the adapter may still implement the
   feature, but the profile cannot prove it, so fail closed. Re-probe rather
   than editing `unknown` to `supported` by hand.
4. Never use the probe profile as an adapter feature matrix: the adapter's
   built-in capabilities remain authoritative for what code exists; the
   profile only narrows.

**DeepSeek web is not an OpenAI-style HTTP upstream.** It has no standard
`/v1/chat/completions` or `/v1/responses` surface to probe, so this tool cannot
produce a meaningful profile for it. The `deepseek_web` channel uses its
built-in capability profile (a mix of proxy implementations and explicit
unsupported entries) unless an operator explicitly narrows it with a
hand-written proxy profile. The HTTP probe can test this proxy's exposed
Responses interface, but cannot enable the `deepseek_web` adapter's missing
output-limit implementation or remove its `max_output_tokens` rejection.

## Development

```bash
# Compile check
python -m compileall -q src tests

# Help smoke test
PYTHONPATH=src python -m upstream_capability_probe --help

# Offline unit/self-tests (loopback HTTP servers only)
PYTHONPATH=src python -m unittest discover -s tests -v
```

Layout:

```
pyproject.toml
README.md
src/upstream_capability_probe/
  __init__.py        package metadata
  __main__.py        python -m entry point
  cli.py             argument parsing, probe run, offline conversion, exit codes
  models.py          ProbeResult, CapabilityProfile, redaction contract
  transport.py       URL normalization, bounded HTTP/SSE transport
  probes.py          probe implementations and registry
  proxy_profile.py   probe profile -> responses-capability-profile converter
tests/
  test_probe.py          offline tests: URL rules, redaction, HTTP classification, SSE parsing
  test_max_output_tokens.py  limit enforcement, safe rejection, local HTTP/CLI integration
  test_proxy_profile.py  offline tests: every mapping rule, channel validation, CLI conversion
```

## Extending the probe set

1. Add a `ProbeSpec` to `PROBE_SPECS` in `probes.py` with a unique
   `id`, `capability`, and explicit `kind`.
2. Return only structural observations in `evidence`; never put response
   text, prompts, IDs, URLs, or header values there.
3. Use `_failure(...)` helpers so 401/403/404/405/timeout/5xx are classified
   consistently and never leak error bodies.
4. Mark any probe that can trigger model generation or server-side state as
   `stateful` so `--no-write-probes` skips it.
5. Add an offline test using the loopback `_ServerCase` harness.

## Non-goals

- It does not validate model quality, safety behaviour, or tokenizer parity.
- It does not test real chat semantics beyond a single synthetic turn.
- It does not call delete/cancel/archive endpoints.
- It does not discover undocumented endpoints.
- It does not prove that the proxy adapter implements a capability; the proxy
  profile can only narrow the adapter's built-in feature set.
- It does not probe the DeepSeek web (non-OpenAI-style) backend; that channel
  keeps its built-in profile unless an operator narrows it deliberately.
