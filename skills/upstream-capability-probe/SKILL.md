---
name: upstream-capability-probe
description: Use when you need to determine which HTTP capabilities an OpenAI-style upstream actually supports before assembling a Responses proxy path, or when you must produce or narrow the Rust proxy's responses-capability-profile. Runs safe, synthetic, redacted probes and produces a machine-readable capability profile plus an optional proxy profile. Use for capability discovery, adapter/module selection, and pre-flight checks against a candidate upstream or gateway.
---

# Upstream Capability Probe

## When to use this skill

Use this skill when:

- You are wiring a new OpenAI-style upstream (or a gateway / aggregator) into a
  proxy and must choose which protocol path and translation modules to enable.
- A proxy currently assumes `/v1/responses` or `/v1/chat/completions` works and
  you need evidence before trusting that assumption.
- You need a repeatable, redacted capability record for comparison across
  upstreams, accounts, or releases.
- You are deciding whether a capability has to be emulated locally (tool
  calling, vision, usage accounting, streaming) or can be passed through.

Do **not** use this skill when:

- You need model-quality evaluation, safety testing, tokenizer parity, or
  semantic correctness. This tool only probes HTTP behaviour.
- You need load, latency, or concurrency measurements.
- The upstream is not HTTP(S) or requires interactive/mTLS authentication the
  tool cannot express.
- You have no authorization to send generation requests. Run
  `--no-write-probes` instead, or stop and ask.

## What this tool is and is not

It sends a small fixed set of **synthetic** requests and records structural
facts: status codes, content types, SSE event names, presence of documented
JSON fields, stream terminator markers, and output-limit enforcement evidence.
It returns a JSON **capability profile**.

It does **not** prove that the upstream is semantically equivalent to the
OpenAI API. `supported` means "the expected HTTP/structural shape appeared",
not "all parameters, edge cases, and behaviours match". Treat a profile as a
routing hint and a risk-reduction input, never as a conformance certificate.

## Inputs and constraints

Required:

- `--base-url`: `http` or `https` only. Accepted forms include
  `https://host`, `https://host/v1`, a custom prefix such as
  `https://host/openai/v1`, or a full endpoint URL ending in
  `/chat/completions`, `/responses`, or `/models`. Query strings are preserved
  on requests but never echoed into the profile. Embedded userinfo is
  rejected; never place credentials in the URL.
- `--model`: a model identifier that the target account can actually use.

Optional:

- `--api-key-env`: **name** of an environment variable containing the bearer
  token. Pass names, never values. If omitted, the tool sends no
  `Authorization` header.
- `--timeout`: per-request seconds, default 30.
- `--output`: path for the JSON profile, or `-` for stdout.
- `--no-write-probes`: skip all `stateful` probes.
- `--proxy-profile`: also write a Rust proxy `responses-capability-profile`
  (see below). Requires `--channel`.
- `--channel`: channel id for the proxy profile, 1–64 chars of
  `[A-Za-z0-9_.-]`.
- `--from-profile`: convert an existing probe profile offline; no request is
  sent and no environment secret is read. Requires `--proxy-profile`.

Hard rules:

- Never print, echo, log, store, or include the API key value in any report,
  command transcript, or file. Prefer `--api-key-env` exclusively.
- Never paste raw upstream responses into chat or files. The tool does not
  retain them; do not add your own capture that does.
- Do not run stateful probes against a production account or a paid endpoint
  without explicit human approval. Stateful probes trigger model generation
  and may create server-side history or incur cost.
- Do not modify the target upstream's data (no delete/cancel/archive calls).
  The bundled probes are non-destructive; keep any new probe non-destructive.
- Do not retry aggressively. One request per probe step is the default.

## Commands

Run from `tools/upstream-capability-probe` (or set `PYTHONPATH` to its `src`).

Read-only inventory (no generation, only `GET /models`):

```bash
PYTHONPATH=src python -m upstream_capability_probe \
  --base-url https://api.example.com \
  --api-key-env UPSTREAM_API_KEY \
  --model gpt-4o-mini \
  --no-write-probes \
  --output profile.readonly.json
```

Full probe set (stateful; requires human approval):

```bash
PYTHONPATH=src python -m upstream_capability_probe \
  --base-url https://api.example.com/v1 \
  --api-key-env UPSTREAM_API_KEY \
  --model gpt-4o-mini \
  --timeout 45 \
  --output profile.json \
  --proxy-profile proxy-profile.json \
  --channel openai_compat
```

Offline conversion of an existing probe profile (no request sent):

```bash
PYTHONPATH=src python -m upstream_capability_probe \
  --from-profile profile.json \
  --proxy-profile proxy-profile.json \
  --channel openai_compat
```

PowerShell equivalent:

```powershell
$env:PYTHONPATH = "src"
python -m upstream_capability_probe --base-url https://api.example.com --api-key-env UPSTREAM_API_KEY --model gpt-4o-mini --no-write-probes --output profile.readonly.json
```

Exit codes: `0` profile written (even with unsupported/inconclusive results),
`1` output write failure, `2` all executed probes failed at transport level or
argument parsing failed. A non-zero exit does not mean "capability missing";
read the per-probe statuses.

## Reading the profile

Top-level keys:

- `schema_version`: profile schema, currently `1`.
- `tool`: tool name/version that produced the file.
- `generated_at`: UTC timestamp.
- `target`: normalized `base_url`, auth mode (`bearer_env`/`none`), and the
  environment variable *name* (never its value).
- `options`: model, timeout, whether write probes ran, probe counts and
  planned probe descriptors.
- `summary`: counts of probe and capability statuses.
- `assembly_hints`: convenience booleans for common proxy decisions.
- `capabilities`: capability name → `{status, supported, reason, probe}`.
- `probes`: per-probe results, each with `id`, `title`, `capability`, `kind`,
  `status`, `supported`, `reason`, `evidence`, `http_status`, `error_class`.
- `security`: explicit guarantees (all false) that the file holds no prompts,
  response bodies, completion text, tokens, headers, or auth values.
- `limitations`: scope statements to carry into any report.

Status handling:

| Status | `supported` | Meaning for assembly |
| --- | --- | --- |
| `supported` | `true` | Expected structural shape observed; candidate for native pass-through. |
| `unsupported` | `false` | Endpoint absent or shape rejected/incompatible; must emulate locally or disable the feature. |
| `inconclusive` | `null` | Timeout, rate limit, 5xx, or ambiguous shape. Do **not** enable native path on this alone; re-test or fail closed. |
| `skipped` | `null` | Skipped by `--no-write-probes`. Unknown, not unsupported. |
| `error` | `null` | Auth or transport problem unrelated to capability; fix access and rerun. |

`reason` values are fixed snake_case tokens such as `sse_stream_observed`,
`endpoint_not_found`, `no_tool_call_observed`, `timeout`, or
`skipped_by_no_write_probes`. Never parse free-form upstream text; there is
none in the profile.

## Selecting modules from capabilities

Use the matching protocol's capability, never a sibling protocol's:

1. **Transport**: if `responses_stream == supported`, assemble the native
   Responses streaming path. Else if `chat_completions_stream == supported`,
   use Chat Completions streaming plus a Responses shim. Else fail closed.
2. **Non-streaming**: only use `responses_non_stream` or
   `chat_completions_non_stream` where the caller needs a complete JSON body.
3. **Function calling**: enable native tool pass-through only when
   `responses_function_calling` / `chat_completions_function_calling` is
   `supported`. Otherwise route through local tool-call emulation.
4. **Tool continuation**: enable native multi-turn tool turns only when
   `*_tool_result_continuation == supported`; otherwise keep the proxy's local
   transcript assembly.
5. **Vision**: enable image pass-through only when the corresponding
   `*_vision_data_uri == supported`.
6. **Usage**: treat `chat_completions_usage` / `responses_usage` as advisory.
   If absent, do not promise exact billing-grade usage accounting.
7. **SSE details**: `*_sse_done` and `*_sse_named_events` tell the stream
   adapter whether it can rely on a sentinel or named event types; always
   still enforce a local terminal condition.
8. **Output limit**: use only `responses_max_output_tokens`, produced by the
   independent `responses.max_output_tokens` stateful probe. It sends a
   16-token budget and a synthetic request for longer output. Support requires
   a `response` object with `status=incomplete`,
   `incomplete_details.reason=max_output_tokens`, consistent nonnegative
   integer usage, positive output within the cap, and a consistent total.
   Reasoning may consume the budget without visible text. HTTP acceptance,
   parameter echo, a completed answer, or missing/invalid usage cannot prove
   enforcement. An explicit unsupported/unknown-parameter code naming this
   field or consistent usage above the cap yields unsupported; ambiguous
   validation, timeout, 429, and 5xx stay inconclusive. This single small-cap
   observation does not discover model-specific minimums or the full range.
   Ordinary Responses probes omit the field. Chat's `max_tokens` and
   `max_completion_tokens` are separate and must never enable this capability.

Any status other than `supported` must not be treated as `supported`. Prefer
`inconclusive` → retry once manually or disable feature; `skipped` → run the
stateful probe with approval; `error` → fix credentials/connectivity.

## Rust proxy capability profile (`--proxy-profile` / `--from-profile`)

The Rust proxy's Responses endpoint optionally loads a profile from
`RESPONSES_CAPABILITY_PROFILE`; see
[`../../RESPONSES_ARCHITECTURE.md`](../../RESPONSES_ARCHITECTURE.md). The
converted document has exactly this shape:

```json
{
  "format": "responses-capability-profile",
  "schema_version": 1,
  "channel": "openai_compat",
  "generated_by": {"tool": "upstream-capability-probe", "version": "0.1.0"},
  "generated_at": "2025-01-02T03:04:05Z",
  "capabilities": {
    "text": {"support": "supported", "provider": "upstream", "evidence": "chat.completions.stream"},
    "file_input": {"support": "unknown", "provider": "none"}
  }
}
```

Rules to repeat when reporting or hand-editing:

- `format` must be `responses-capability-profile`, `schema_version` must be
  `1`, and `channel` must equal the active adapter id. The proxy rejects
  mismatches; unknown top-level fields are ignored.
- `support` ∈ `supported|unsupported|unknown`; `provider` ∈
  `upstream|proxy|none` and says where the capability runs (`upstream`
  pass-through, `proxy` local implementation, `none` no provider);
  `mode`/`evidence` are optional short strings.
- Exactly these 30 ids, all of them, nothing else: `text`, `streaming`,
  `non_streaming`, `image_input`, `file_input`, `function_tools`,
  `tool_choice`, `custom_tools`, `web_search`, `file_search`,
  `code_interpreter`, `mcp`, `image_generation`, `computer_use`, `shell`,
  `reasoning`, `structured_output`, `verbosity`, `sampling_controls`,
  `max_output_tokens`, `logprobs`, `stored_responses`, `previous_response`,
  `conversations`, `background`, `stream_resume`, `auto_truncation`,
  `compaction`, `prompt_templates`, `exact_usage`.
- The document must never contain base URLs, hostnames, env var names,
  headers, prompts, response bodies, or tokens.

### Fail-closed narrowing (proxy semantics; never change these)

Effective capability = **adapter built-in AND profile**. A capability is
enabled only if the adapter implements it **and** the profile says
`supported`; `unsupported`, `unknown`, and absent all disable it. A profile
can never enable something the adapter lacks. Dependencies: `tool_choice`
needs `function_tools`; `previous_response`, `background`, and `stream_resume`
need `stored_responses`; `stream_resume` also needs `streaming`.

### How to use the converted profile when assembling/onboarding an adapter

1. Probe the candidate upstream (or reuse a saved probe profile) and pass
   `--proxy-profile PATH --channel <adapter-id>`. For a saved profile, use
   `--from-profile` so nothing is sent.
2. Set `RESPONSES_CAPABILITY_PROFILE=PATH` on the proxy. The channel id must
   match the adapter being configured; a mismatch is rejected and the proxy
   will not start that channel with the file.
3. When choosing modules: read `capabilities.<id>.support`; enable a native
   path only for `supported`. Treat `unknown` as unavailable and fail closed.
   Do not hand-promote `unknown` to `supported`; re-probe instead.
4. Do not use the proxy profile as the adapter's feature matrix: the adapter's
   built-in set defines what code exists; the profile only narrows it.

### Mapping summary (conservative)

- `text`: supported if any Chat/Responses stream or non-stream probe is
  supported; unsupported only if all four are unsupported; else unknown.
- `streaming`: Responses SSE → `upstream` / `native_responses_sse`; Chat SSE →
  `proxy` / `chat_stream_reencoded`; both unsupported → unsupported.
- `non_streaming`: native Responses → `upstream` / `native`; Chat → `proxy` /
  `chat_reencoded`; existing streaming → `proxy` / `collected_stream`.
- `image_input`: either vision probe supported → supported.
- `function_tools`: only a full same-protocol pair (function calling +
  tool-result continuation) makes it supported; both function-calling probes
  unsupported → unsupported.
- `stored_responses` / `previous_response`: `proxy` / `local_transcript`,
  supported only when `text` is supported, else unknown.
- `exact_usage`: either usage probe supported → supported; else unknown.
- `max_output_tokens`: independent `responses_max_output_tokens` verdict only;
  supported → `upstream`, unsupported → `none`; other statuses or older
  profiles without the probe → `unknown` / `none`.
- All other 21 ids: `unknown` / `none`, no evidence. Probe statuses other than
  `supported`/`unsupported` (`inconclusive`, `skipped`, `error`) become
  `unknown`.

### DeepSeek web channel

The DeepSeek web backend is **not an OpenAI-style HTTP upstream**: it does not
expose `/v1/chat/completions` or `/v1/responses`, so this probe cannot produce
a meaningful profile for it. The `deepseek_web` adapter uses its built-in
capability profile (a mix of proxy implementations and explicit unsupported
entries) unless an operator deliberately narrows it with a hand-written
profile. Never claim a DeepSeek web capability because a probe profile says so.
Probing this proxy's HTTP facade can observe its field rejection, but cannot
provide the `deepseek_web` adapter's missing output-limit implementation.

## Designing new probes

When the built-in set does not cover a capability:

1. Define the minimal synthetic request. It must not contain user data,
   production identifiers, or secrets. Use fixed tiny strings and a 1×1 PNG
   data URI only.
2. Classify it honestly: `read_only` only if no model generation and no
   server-side state is created. Everything else is `stateful`.
3. Observe only structural facts: status code, content type, SSE event names,
   presence of documented field names, counts, and booleans. Never store raw
   response bodies, completion text, prompts, URLs, IDs, or header values.
4. Reuse `_failure` / `_failure_from_stream` helpers so 401/403/404/405/
   timeout/5xx are classified consistently. The transport may parse bounded
   400/422 JSON solely for an allowlisted unsupported-parameter marker; discard
   every error body and never retain its text or arbitrary fields.
5. Add the capability to the registry in `probes.py` (`PROBE_SPECS`) with a
   stable id and capability name.
6. Add an offline test using the loopback `_ServerCase` harness in
   `tests/test_probe.py`. Never point tests at a real upstream.
7. Update the README probe catalog and this skill if the new probe changes
   assembly guidance.

Destructive endpoints (delete/cancel/archive) are out of scope, even behind a
flag. If a capability can only be tested destructively, document it as
unprobed and fail closed in the proxy.

## Mandatory human confirmation for side effects

Before running without `--no-write-probes`:

- Confirm the target is a test account or that the owner approved generation
  charges and provider-side history creation.
- State the model name, base URL host, and that `stateful` probes will run.
- Never run stateful probes against a production database or a live customer
  endpoint on your own initiative.

`--no-write-probes` still performs `GET /models`, which is read-only and safe.
If even that is not authorized, do not run the tool.

## Reporting results

When reporting to a human or another agent:

- Include the probe profile path and (when produced) the proxy profile path,
  tool version, `generated_at`, normalized `target.base_url`, model, channel
  id, and whether write probes ran.
- Summarize `summary.probes` and `summary.capabilities` counts.
- List each capability with status/reason; do not paste `evidence` blobs
  beyond short structural facts.
- Never paste the API key, `Authorization` headers, prompts, completion text,
  response bodies, or response identifiers. The proxy profile in particular
  must contain none of those; if it does, treat the run as failed.
- Carry forward the limitation: HTTP capability only, not semantic
  equivalence, and a proxy profile can only narrow the adapter—never enable it.
