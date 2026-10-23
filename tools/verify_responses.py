#!/usr/bin/env python3
"""
tools/verify_responses.py

Live black-box acceptance for the proxy's `/v1/responses` endpoint, run against
a local proxy that serves a TEMPORARY copy of the proxy database.

Standard Python library only. Never spawns or stops the proxy and never reads
account credentials. Prints only check names, pass/fail, HTTP statuses, error
codes and timings: never prompts, model output, response ids, session ids,
codewords or API keys.

Commands (run them in this order):
  run            capability gates, a Chat Completions regression turn,
                 streaming and collected turns, stored chains, branch forks,
                 function tools, reasoning, stateless replay (store=false),
                 delete, client disconnect and image input. Ends by storing
                 a probe for the restart check in --state. `--phases H,D`
                 reruns selected phases in that order, for example after an
                 upstream account throttled a turn.
  restart-check  after the operator restarts the proxy: the stored probe
                 continues in the same remote session with its context.
  keys           registers two synthetic API keys in the temporary database
                 and checks owner isolation. Writes to --db, so it requires
                 --confirm-temporary-db. Run it last: once keys exist the
                 proxy rejects unauthenticated requests.

--db is read with read-only SQLite connections, except by `keys`. Session
reuse is inferred from `threads.session_id` in that copy: an unchanged session
means the turn continued the remote conversation with a delta; a new one means
the proxy replayed the history into a fresh session.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import re
import secrets
import sqlite3
import struct
import sys
import time
import urllib.parse
import uuid
import zlib
from contextlib import closing
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

LIMITATIONS: List[str] = [
    "Uses synthetic prompts and short instruction-following answers; it does not measure answer quality.",
    "Session reuse is inferred from the temporary database copy, not from the upstream service.",
    "Every upstream turn creates or extends synthetic conversations in the configured DeepSeek accounts.",
    "Image input is checked with one synthetic solid-colour PNG.",
]

ANONYMOUS = "anonymous"
INSTRUCTIONS = "You are a terse assistant in an automated test. Follow the user's instructions exactly."
TOOL_INSTRUCTIONS = (
    "You are an assistant in an automated test. Use the provided tools when asked "
    "and report tool results exactly."
)
TOOLS = [
    {
        "type": "function",
        "name": "get_secret_code",
        "description": "Returns the secret code the user is asking for.",
        "parameters": {
            "type": "object",
            "properties": {"purpose": {"type": "string", "description": "Why the code is needed."}},
            "required": ["purpose"],
            "additionalProperties": False,
        },
    }
]

Events = List[Tuple[str, Any]]
Check = Callable[[str, bool], bool]


class Abort(Exception):
    """A phase cannot continue; the message is safe to print."""


class Proxy:
    def __init__(self, base_url: str, timeout: float) -> None:
        parts = urllib.parse.urlsplit(base_url)
        if parts.scheme != "http" or not parts.hostname:
            raise ValueError("base URL must look like http://127.0.0.1:19177")
        self.host = parts.hostname
        self.port = parts.port or 80
        prefix = parts.path.rstrip("/")
        self.prefix = prefix[: -len("/v1")] if prefix.endswith("/v1") else prefix
        self.timeout = timeout

    def _connection(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    @staticmethod
    def _headers(extra: Optional[Dict[str, str]], bearer: Optional[str]) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if bearer:
            headers["Authorization"] = f"Bearer {bearer}"
        headers.update(extra or {})
        return headers

    def call(
        self,
        method: str,
        path: str,
        body: Any = None,
        headers: Optional[Dict[str, str]] = None,
        bearer: Optional[str] = None,
        raw: Optional[bytes] = None,
    ) -> Tuple[int, Any]:
        """One request; returns (status, parsed JSON body or None)."""
        data = raw if raw is not None else (None if body is None else json.dumps(body).encode("utf-8"))
        with closing(self._connection()) as conn:
            conn.request(method, self.prefix + path, body=data, headers=self._headers(headers, bearer))
            resp = conn.getresponse()
            payload = resp.read()
            try:
                return resp.status, json.loads(payload.decode("utf-8")) if payload else None
            except ValueError:
                return resp.status, None

    def stream(
        self,
        path: str,
        body: Any,
        headers: Optional[Dict[str, str]] = None,
        stop_after: Optional[str] = None,
    ) -> Tuple[int, Dict[str, str], Events, Any]:
        """POST and read SSE. With `stop_after`, the connection is closed right
        after the first event of that type, which disconnects the client."""
        with closing(self._connection()) as conn:
            conn.request("POST", self.prefix + path, body=json.dumps(body).encode("utf-8"), headers=self._headers(headers, None))
            resp = conn.getresponse()
            response_headers = {key.lower(): value for key, value in resp.getheaders()}
            if resp.status != 200:
                payload = resp.read()
                try:
                    error = json.loads(payload.decode("utf-8"))
                except ValueError:
                    error = None
                return resp.status, response_headers, [], error
            events: Events = []
            kind: Optional[str] = None
            data: List[str] = []
            while True:
                line = resp.readline()
                if not line:
                    break
                text = line.decode("utf-8").rstrip("\r\n")
                if not text:
                    if data:
                        joined = "\n".join(data)
                        events.append((kind or "message", joined if joined == "[DONE]" else json.loads(joined)))
                        if stop_after is not None and kind == stop_after:
                            break
                    kind, data = None, []
                elif text.startswith(":"):
                    continue
                elif text.startswith("event:"):
                    kind = text[len("event:"):].strip()
                elif text.startswith("data:"):
                    data.append(text[len("data:"):].lstrip())
            return resp.status, response_headers, events, None


class Database:
    def __init__(self, path: str) -> None:
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise ValueError("--db must name an existing temporary database copy")

    def _open(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=10)

    def response(self, response_id: str) -> Optional[Dict[str, str]]:
        with closing(self._open()) as db:
            row = db.execute(
                "SELECT thread_id, owner_id FROM responses WHERE response_id = ?", (response_id,)
            ).fetchone()
        return {"thread_id": row[0], "owner_id": row[1]} if row else None

    def thread(self, thread_id: Optional[str]) -> Dict[str, Any]:
        if not thread_id:
            return {"session_id": None, "message_count": None}
        with closing(self._open()) as db:
            row = db.execute("SELECT session_id FROM threads WHERE thread_id = ?", (thread_id,)).fetchone()
            history = db.execute(
                "SELECT message_count FROM thread_history WHERE thread_id = ?", (thread_id,)
            ).fetchone()
        return {"session_id": row[0] if row else None, "message_count": history[0] if history else None}

    def thread_of(self, response_id: str) -> Optional[str]:
        row = self.response(response_id)
        return row["thread_id"] if row else None


class Report:
    def __init__(self) -> None:
        self.phases: List[Dict[str, Any]] = []

    def phase(self, name: str, description: str, body: Callable[[Check], None]) -> None:
        print(f"[{name}] {description}")
        checks: List[Dict[str, Any]] = []

        def check(what: str, ok: bool) -> bool:
            passed = bool(ok)
            checks.append({"check": what, "passed": passed})
            print(f"    [{'PASS' if passed else 'FAIL'}] {what}")
            return passed

        started = time.monotonic()
        problem = None
        try:
            body(check)
        except Abort as exc:
            problem = str(exc)
        except (OSError, http.client.HTTPException, ValueError, KeyError, TypeError, IndexError) as exc:
            problem = type(exc).__name__
        if problem:
            checks.append({"check": f"phase aborted: {problem}", "passed": False})
            print(f"    [FAIL] phase aborted: {problem}")
        self.phases.append(
            {
                "phase": name,
                "description": description,
                "passed": bool(checks) and all(c["passed"] for c in checks),
                "elapsed_s": round(time.monotonic() - started, 1),
                "checks": checks,
            }
        )

    def finish(self, command: str, output: Optional[str]) -> int:
        checks = [c for p in self.phases for c in p["checks"]]
        passed = sum(c["passed"] for c in checks)
        status = "PASS" if checks and passed == len(checks) else "FAIL"
        print("-" * 70)
        print(f"VERDICT: {status} ({passed}/{len(checks)} checks, "
              f"{sum(p['passed'] for p in self.phases)}/{len(self.phases)} phases)")
        if output:
            Path(output).write_text(
                json.dumps(
                    {"command": command, "status": status, "phases": self.phases, "limitations": LIMITATIONS},
                    indent=2,
                ),
                encoding="utf-8",
            )
        return 0 if status == "PASS" else 1


def codeword(prefix: str) -> str:
    letters = "".join(secrets.choice("BCDFGHJKLMNPQRSTVWXZ") for _ in range(4))
    return f"{prefix}{letters}{secrets.randbelow(900) + 100}"


def contains(text: str, token: str) -> bool:
    return token.upper() in re.sub(r"[\s*`]", "", text).upper()


def output_text(response: Dict[str, Any]) -> str:
    return "".join(
        part.get("text") or ""
        for item in response.get("output") or []
        if item.get("type") == "message"
        for part in item.get("content") or []
        if part.get("type") == "output_text"
    )


def error_code(parsed: Any) -> Optional[str]:
    if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
        return parsed["error"].get("code")
    return None


def completed_from(events: Events) -> Optional[Dict[str, Any]]:
    if events and events[-1][0] == "response.completed" and isinstance(events[-1][1], dict):
        return events[-1][1].get("response")
    return None


def stream_problem(status: int, events: Events, error: Any) -> str:
    if events and events[-1][0] == "response.failed":
        code = ((events[-1][1].get("response") or {}).get("error") or {}).get("code")
        return f"stream ended with response.failed ({code})"
    return f"stream did not complete (HTTP {status} {error_code(error)})"


def complete(proxy: Proxy, body: Dict[str, Any], headers: Optional[Dict[str, str]] = None,
             bearer: Optional[str] = None) -> Dict[str, Any]:
    status, parsed = proxy.call("POST", "/v1/responses", body, headers, bearer)
    if status == 200 and isinstance(parsed, dict) and parsed.get("status") == "completed":
        return parsed
    raise Abort(f"turn did not complete (HTTP {status} {error_code(parsed)})")


def derived_thread_id(owner: str, header: str, value: str) -> str:
    """The proxy's thread key for a request without previous_response_id."""
    seed = f"{header}:{value.strip()}"
    return "responses_" + hashlib.sha256(f"{owner}\0{seed}".encode("utf-8")).hexdigest()[:24]


def fresh_session() -> Dict[str, str]:
    """A new client session. Fixed prompts would otherwise share one
    first-message thread (and its account binding) across runs."""
    return {"x-session-affinity": str(uuid.uuid4())}


def red_png(size: int = 64) -> str:
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * size for _ in range(size))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def cmd_run(args: argparse.Namespace, proxy: Proxy, db: Database, report: Report) -> None:
    model = args.model
    ctx: Dict[str, Any] = {}

    def gates(check: Check) -> None:
        cases = [
            ("temperature is rejected", {"temperature": 0.2}, "unsupported_parameter", "temperature"),
            ("max_output_tokens is rejected", {"max_output_tokens": 64}, "unsupported_parameter", "max_output_tokens"),
            ("background is rejected", {"background": True}, "unsupported_parameter", "background"),
            ("hosted web_search is rejected", {"tools": [{"type": "web_search"}]}, "unsupported_parameter", "tools[0]"),
            ("structured output is rejected",
             {"text": {"format": {"type": "json_schema", "name": "x", "schema": {"type": "object"}}}},
             "unsupported_parameter", None),
            ("truncation=auto is rejected", {"truncation": "auto"}, "unsupported_parameter", "truncation"),
            ("logprobs are rejected", {"include": ["message.output_text.logprobs"]}, "unsupported_parameter", None),
            ("conversation is rejected", {"conversation": "conv_synthetic"}, "unsupported_parameter", "conversation"),
            ("unknown fields are rejected", {"made_up_field": 1}, "unsupported_parameter", "made_up_field"),
            ("input_file is rejected",
             {"input": [{"role": "user", "content": [
                 {"type": "input_file", "filename": "a.txt", "file_data": "data:text/plain;base64,aGk="}]}]},
             "unsupported_parameter", None),
            ("unknown model is rejected", {"model": "gpt-unknown-model"}, "model_not_found", "model"),
            ("unknown previous_response_id is rejected", {"previous_response_id": "resp_does_not_exist"},
             "previous_response_not_found", "previous_response_id"),
        ]
        for what, extra, code, param in cases:
            body: Dict[str, Any] = {"model": model, "input": "synthetic gate check"}
            body.update(extra)
            status, parsed = proxy.call("POST", "/v1/responses", body)
            error = parsed.get("error", {}) if isinstance(parsed, dict) else {}
            check(what, status == 400 and error.get("code") == code and (param is None or error.get("param") == param))
        status, parsed = proxy.call("POST", "/v1/responses", raw=b"{not json")
        check("malformed JSON is a 400 invalid_request", status == 400 and error_code(parsed) == "invalid_request")
        check("GET of an unknown id is 404", proxy.call("GET", "/v1/responses/resp_does_not_exist")[0] == 404)
        status, parsed = proxy.call("GET", "/v1/responses/resp_does_not_exist?stream=true")
        check("GET with stream=true is rejected", status == 400 and error_code(parsed) == "unsupported_parameter")
        check("DELETE of an unknown id is 404", proxy.call("DELETE", "/v1/responses/resp_does_not_exist")[0] == 404)

    def chat(check: Check) -> None:
        token = codeword("CH")
        body = {"model": model, "stream": True,
                "messages": [{"role": "user", "content": f"Reply with exactly this token and nothing else: {token}"}]}
        status, _, events, _ = proxy.stream("/v1/chat/completions", body, {"x-session-affinity": str(uuid.uuid4())})
        data = [value for _, value in events]
        check("Chat Completions still streams to [DONE]", status == 200 and bool(data) and data[-1] == "[DONE]")
        content = "".join(
            (((value.get("choices") or [{}])[0].get("delta") or {}).get("content") or "")
            for value in data
            if isinstance(value, dict) and value.get("choices")
        )
        check("Chat Completions answer contains the token", contains(content, token))
        check("Chat Completions sent no error chunk", not any(isinstance(v, dict) and "error" in v for v in data))
        status, parsed = proxy.call("POST", "/v1/chat/completions",
                                    {"model": model, "stream": False, "messages": [{"role": "user", "content": "hi"}]})
        check("Chat Completions still rejects stream=false",
              status == 400 and isinstance(parsed, dict) and parsed.get("error", {}).get("param") == "stream")

    def streaming(check: Check) -> None:
        token = codeword("TK")
        body = {"model": model, "stream": True, "input": f"Reply with exactly this token and nothing else: {token}"}
        status, headers, events, error = proxy.stream("/v1/responses", body)
        kinds = [kind for kind, _ in events]
        check("HTTP 200 text/event-stream with x-accel-buffering: no",
              status == 200 and headers.get("content-type", "").startswith("text/event-stream")
              and headers.get("x-accel-buffering") == "no")
        check("opens with response.created then response.in_progress",
              kinds[:2] == ["response.created", "response.in_progress"])
        response = completed_from(events)
        if response is None:
            raise Abort(stream_problem(status, events, error))
        check("ends with response.completed and never response.failed", "response.failed" not in kinds)
        check("sequence numbers count from 0 and match event types",
              all(isinstance(d, dict) and d.get("sequence_number") == i and d.get("type") == k
                  for i, (k, d) in enumerate(events)))
        text = output_text(response)
        deltas = "".join(d.get("delta", "") for k, d in events if k == "response.output_text.delta")
        check("output_text deltas add up to the completed text", bool(deltas) and deltas == text)
        check("answer contains the requested token", contains(text, token))
        usage = response.get("usage") or {}
        check("usage reports input and output tokens",
              usage.get("input_tokens", 0) > 0 and usage.get("output_tokens", 0) > 0)
        check("response object shape",
              response.get("object") == "response" and str(response.get("id", "")).startswith("resp_")
              and response.get("status") == "completed" and response.get("store") is True
              and response.get("model") == model)
        status, stored = proxy.call("GET", f"/v1/responses/{response['id']}")
        check("GET returns the identical stored response", status == 200 and stored == response)
        row = db.response(response["id"])
        thread = db.thread(row["thread_id"] if row else None)
        check("database holds the response, its owner and a 2-message checkpoint",
              row is not None and row["owner_id"] == ANONYMOUS and thread["session_id"] is not None
              and thread["message_count"] == 2)

    def collected(check: Check) -> None:
        token = codeword("TK")
        response = complete(proxy, {"model": model, "input": f"Reply with exactly this token and nothing else: {token}"})
        check("collected answer contains the requested token", contains(output_text(response), token))
        check("output is a single assistant message", [i.get("type") for i in response.get("output", [])] == ["message"])
        status, stored = proxy.call("GET", f"/v1/responses/{response['id']}")
        check("GET returns the identical stored response", status == 200 and stored == response)

    def chain(check: Check) -> None:
        c1 = codeword("CW")
        d1 = complete(proxy, {"model": model, "instructions": INSTRUCTIONS,
                              "input": f"Remember this codeword for later: {c1}. Reply with only OK."})
        thread_id = db.thread_of(d1["id"])
        s1 = db.thread(thread_id)
        check("first turn opens a remote session with a 3-message checkpoint",
              s1["session_id"] is not None and s1["message_count"] == 3)
        d2 = complete(proxy, {"model": model, "instructions": INSTRUCTIONS, "previous_response_id": d1["id"],
                              "input": "What is the codeword I asked you to remember? Reply with the codeword only."})
        check("chained turn recalls the codeword", contains(output_text(d2), c1))
        check("chained response records previous_response_id", d2.get("previous_response_id") == d1["id"])
        s2 = db.thread(thread_id)
        check("continuation stays on the thread and reuses the remote session",
              db.thread_of(d2["id"]) == thread_id and s2["session_id"] == s1["session_id"]
              and s2["message_count"] == 5)
        d3 = complete(proxy, {"model": model, "previous_response_id": d2["id"],
                              "input": "Repeat the codeword once more, codeword only."})
        check("chain without instructions still recalls the codeword", contains(output_text(d3), c1))
        check("instructions are not inherited", d3.get("instructions") is None)
        s3 = db.thread(thread_id)
        check("dropping instructions replays into a fresh session",
              s3["session_id"] not in (None, s2["session_id"]) and s3["message_count"] == 6)
        ctx.update(c1=c1, d1=d1["id"], d2=d2["id"], thread=thread_id, session=s3["session_id"])

    def branches(check: Check) -> None:
        if "d2" not in ctx:
            raise Abort("prerequisite chain phase failed")
        c1, c2, thread_id = ctx["c1"], codeword("CW"), ctx["thread"]
        e2 = complete(proxy, {"model": model, "instructions": INSTRUCTIONS, "previous_response_id": ctx["d1"],
                              "input": f"The codeword has changed. The new codeword is {c2}. Reply with only OK."})
        s2 = db.thread(thread_id)
        check("forking from an earlier response replays into a fresh session",
              db.thread_of(e2["id"]) == thread_id and s2["session_id"] not in (None, ctx["session"])
              and s2["message_count"] == 5)
        e3 = complete(proxy, {"model": model, "instructions": INSTRUCTIONS, "previous_response_id": ctx["d2"],
                              "input": "What is the codeword? Reply with the codeword only."})
        answer = output_text(e3)
        check("sibling branch keeps its own codeword", contains(answer, c1))
        check("sibling branch never sees the other branch's codeword", not contains(answer, c2))
        s3 = db.thread(thread_id)
        check("switching branches replays into a fresh session",
              s3["session_id"] not in (None, s2["session_id"]) and s3["message_count"] == 7)
        e4 = complete(proxy, {"model": model, "instructions": INSTRUCTIONS, "previous_response_id": e2["id"],
                              "input": "What is the current codeword? Reply with the codeword only."})
        check("changed branch answers with the new codeword", contains(output_text(e4), c2))
        s4 = db.thread(thread_id)
        check("returning to a branch that is not the thread head replays again",
              s4["session_id"] not in (None, s3["session_id"]) and s4["message_count"] == 7)
        e5 = complete(proxy, {"model": model, "instructions": INSTRUCTIONS, "previous_response_id": e4["id"],
                              "input": "Say the current codeword again, codeword only."})
        check("continuing the head branch recalls the new codeword", contains(output_text(e5), c2))
        s5 = db.thread(thread_id)
        check("continuing the head branch reuses its session",
              s5["session_id"] == s4["session_id"] and s5["message_count"] == 9)

    def tools(check: Check) -> None:
        secret = codeword("SC")
        body = {"model": model, "instructions": TOOL_INSTRUCTIONS, "tools": TOOLS, "stream": True,
                "tool_choice": {"type": "function", "name": "get_secret_code"},
                "input": "Fetch the secret code with the get_secret_code tool, then tell me the code exactly as the tool returns it."}
        status, _, events, error = proxy.stream("/v1/responses", body, fresh_session())
        response = completed_from(events)
        if response is None:
            raise Abort(stream_problem(status, events, error))
        kinds = [kind for kind, _ in events]
        calls = [item for item in response.get("output", []) if item.get("type") == "function_call"]
        check("exactly one get_secret_code call", len(calls) == 1 and calls[0].get("name") == "get_secret_code")
        call = calls[0] if calls else {}
        try:
            arguments_ok = isinstance(json.loads(call.get("arguments") or "").get("purpose"), str)
        except (ValueError, AttributeError):
            arguments_ok = False
        check("arguments are JSON matching the schema", arguments_ok)
        check("item and call ids are server generated",
              str(call.get("id", "")).startswith("fc_") and str(call.get("call_id", "")).startswith("call_"))
        check("function_call_arguments delta and done are streamed",
              "response.function_call_arguments.delta" in kinds and "response.function_call_arguments.done" in kinds)
        thread_id = db.thread_of(response["id"])
        s1 = db.thread(thread_id)
        check("tool turn leaves a 3-message checkpoint", s1["session_id"] is not None and s1["message_count"] == 3)
        if not calls:
            raise Abort("no function call to answer")
        f2 = complete(proxy, {"model": model, "instructions": TOOL_INSTRUCTIONS, "tools": TOOLS, "tool_choice": "none",
                              "previous_response_id": response["id"],
                              "input": [{"type": "function_call_output", "call_id": call["call_id"],
                                         "output": json.dumps({"secret_code": secret})}]})
        check("answer reports the tool result", contains(output_text(f2), secret))
        s2 = db.thread(thread_id)
        check("tool round trip continues the same remote session",
              s2["session_id"] == s1["session_id"] and s2["message_count"] == 5)

    def reasoning(check: Check) -> None:
        question = "What is 17 multiplied by 23? Answer with the number only."
        body = {"model": model, "stream": True, "input": question, "reasoning": {"effort": "high"}}
        status, _, events, error = proxy.stream("/v1/responses", body, fresh_session())
        response = completed_from(events)
        if response is None:
            raise Abort(stream_problem(status, events, error))
        kinds = [kind for kind, _ in events]
        first_reasoning = kinds.index("response.reasoning_text.delta") if "response.reasoning_text.delta" in kinds else -1
        first_text = kinds.index("response.output_text.delta") if "response.output_text.delta" in kinds else -1
        check("reasoning text streams before the answer", 0 <= first_reasoning < first_text)
        output = response.get("output") or [{}]
        content = output[0].get("content") or [{}]
        check("first item is reasoning with text and an empty summary",
              output[0].get("type") == "reasoning" and bool(content[0].get("text")) and output[0].get("summary") == [])
        check("high-effort answer is correct", "391" in output_text(response))
        usage = response.get("usage") or {}
        check("reasoning tokens are reported", (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0) > 0)
        check("effort is echoed", (response.get("reasoning") or {}).get("effort") == "high")
        quiet = complete(proxy, {"model": model, "input": question, "reasoning": {"effort": "none"},
                                 "text": {"verbosity": "low"}}, fresh_session())
        check("effort none produces no reasoning item", all(i.get("type") != "reasoning" for i in quiet.get("output", [])))
        check("effort none still answers correctly", "391" in output_text(quiet))
        check("verbosity and effort are echoed",
              (quiet.get("text") or {}).get("verbosity") == "low"
              and (quiet.get("reasoning") or {}).get("effort") == "none")

    def stateless(check: Check) -> None:
        word = codeword("SL")
        session = str(uuid.uuid4())
        headers = {"x-session-affinity": session}
        first_user = {"role": "user", "content": f"Remember the codeword {word}. Reply with only OK."}
        i1 = complete(proxy, {"model": model, "store": False, "input": [first_user]}, headers)
        check("store=false is echoed", i1.get("store") is False)
        check("store=false response is neither retrievable nor saved",
              proxy.call("GET", f"/v1/responses/{i1['id']}")[0] == 404 and db.response(i1["id"]) is None)
        thread_id = derived_thread_id(ANONYMOUS, "x-session-affinity", session)
        t1 = db.thread(thread_id)
        check("the session header keys a checkpoint-only thread",
              t1["session_id"] is not None and t1["message_count"] == 2)
        question = {"role": "user", "content": "What is the codeword? Reply with the codeword only."}
        # Echo the output items verbatim, ids and all, as SDK clients do.
        i2 = complete(proxy, {"model": model, "store": False, "input": [first_user, *i1["output"], question]}, headers)
        check("client-replayed history recalls the codeword", contains(output_text(i2), word))
        t2 = db.thread(thread_id)
        check("echoed output items continue the same remote session",
              t2["session_id"] == t1["session_id"] and t2["message_count"] == 4)

    def deletion(check: Check) -> None:
        response = complete(proxy, {"model": model, "input": "Reply with only OK."}, fresh_session())
        path = f"/v1/responses/{response['id']}"
        status, parsed = proxy.call("DELETE", path)
        check("DELETE returns the deletion object",
              status == 200 and parsed == {"id": response["id"], "object": "response.deleted", "deleted": True})
        check("deleted response is gone from the API and the database",
              proxy.call("GET", path)[0] == 404 and db.response(response["id"]) is None)
        check("a second DELETE is 404", proxy.call("DELETE", path)[0] == 404)
        status, parsed = proxy.call("POST", "/v1/responses",
                                    {"model": model, "input": "continue", "previous_response_id": response["id"]})
        check("a deleted response cannot be continued",
              status == 400 and error_code(parsed) == "previous_response_not_found")

    def disconnect(check: Check) -> None:
        headers = {"x-session-affinity": str(uuid.uuid4())}
        body = {"model": model, "stream": True,
                "input": "Write a calm story of about 400 words about a lighthouse keeper. Plain prose, no title."}
        status, _, events, _ = proxy.stream("/v1/responses", body, headers, stop_after="response.output_text.delta")
        created = (events[0][1].get("response") or {}).get("id") if events and events[0][0] == "response.created" else None
        check("stream started and the client hung up after the first text delta",
              status == 200 and created is not None and events[-1][0] == "response.output_text.delta")
        if created is None:
            raise Abort("stream did not start")
        # Same thread: this waits for the cancelled turn to release its lock,
        # so if that turn had not stopped it would be committed by now.
        after = complete(proxy, {"model": model, "input": "Reply with only OK."}, headers)
        check("the next turn on the same thread completes", bool(output_text(after)))
        check("the abandoned response was never stored",
              proxy.call("GET", f"/v1/responses/{created}")[0] == 404 and db.response(created) is None)

    def image(check: Check) -> None:
        body = {"model": model, "input": [{"role": "user", "content": [
            {"type": "input_text", "text": "What is the dominant colour of this image? Answer with one lowercase English word."},
            {"type": "input_image", "image_url": red_png()},
        ]}]}
        response = complete(proxy, body, fresh_session())
        check("image turn completes", response.get("status") == "completed")
        check("answer names the image colour", contains(output_text(response), "red"))

    def restart_probe(check: Check) -> None:
        word = codeword("RS")
        response = complete(proxy, {"model": model, "input": f"Remember the codeword {word}. Reply with only OK."})
        thread_id = db.thread_of(response["id"])
        thread = db.thread(thread_id)
        check("restart probe is stored with a 2-message checkpoint",
              thread["session_id"] is not None and thread["message_count"] == 2)
        Path(args.state).write_text(
            json.dumps({"response_id": response["id"], "codeword": word, "thread_id": thread_id,
                        "session_id": thread["session_id"]}),
            encoding="utf-8",
        )

    phases = {
        "A": ("capability gates and request errors (no upstream turn)", gates),
        "B": ("Chat Completions regression on the same binary", chat),
        "C": ("streaming text turn", streaming),
        "D": ("collected (non-streaming) turn", collected),
        "E": ("stored chain with previous_response_id", chain),
        "F": ("branch forks from earlier responses", branches),
        "G": ("function tool call and function_call_output", tools),
        "H": ("reasoning effort high and none, verbosity", reasoning),
        "I": ("stateless replay with store=false", stateless),
        "J": ("delete", deletion),
        "K": ("client disconnect mid-stream", disconnect),
        "L": ("image input", image),
        "M": ("prepare the restart probe", restart_probe),
    }
    # A rerun lists only some phases, in the order given. F depends on E.
    selected = [p.strip().upper() for p in args.phases.split(",")] if args.phases else list(phases)
    unknown = [p for p in selected if p not in phases]
    if unknown:
        raise SystemExit(f"unknown phases: {','.join(unknown)}")
    for name in selected:
        description, body = phases[name]
        report.phase(name, description, body)


def cmd_restart_check(args: argparse.Namespace, proxy: Proxy, db: Database, report: Report) -> None:
    def restarted(check: Check) -> None:
        state = json.loads(Path(args.state).read_text(encoding="utf-8"))
        response = complete(proxy, {"model": args.model, "previous_response_id": state["response_id"],
                                    "input": "What is the codeword? Reply with the codeword only."})
        check("stored chain survives the restart with its context", contains(output_text(response), state["codeword"]))
        thread = db.thread(state["thread_id"])
        check("the restarted proxy continues the same remote session",
              thread["session_id"] == state["session_id"] and thread["message_count"] == 4)

    report.phase("R", "continue a stored chain after a proxy restart", restarted)


def insert_key(db_path: Path, name: str, key: str) -> str:
    """Register a key exactly like `Store::insert_api_key` does."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    key_id = f"key-{digest[:12]}"
    with closing(sqlite3.connect(db_path, timeout=10)) as db:
        db.execute(
            "INSERT INTO api_keys (id, key_hash, key_prefix, name, created_at) "
            "VALUES (?, ?, ?, ?, CAST(strftime('%s','now') AS INTEGER))",
            (key_id, digest, key[:8], name),
        )
        db.commit()
    return key_id


def cmd_keys(args: argparse.Namespace, proxy: Proxy, db: Database, report: Report) -> None:
    key_a = "sk-check-a-" + secrets.token_urlsafe(18)
    key_b = "sk-check-b-" + secrets.token_urlsafe(18)
    state_path = Path(args.state)
    earlier = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else None

    def isolation(check: Check) -> None:
        owner_a = insert_key(db.path, "live-check-a", key_a)
        insert_key(db.path, "live-check-b", key_b)
        status, parsed = proxy.call("POST", "/v1/responses", {"model": args.model, "input": "Reply with only OK."})
        check("requests without a key are rejected once keys exist",
              status == 401 and error_code(parsed) == "missing_api_key")
        status, parsed = proxy.call("GET", "/v1/responses/resp_does_not_exist", bearer="sk-not-registered")
        check("unregistered keys are rejected", status == 401 and error_code(parsed) == "invalid_api_key")
        response = complete(proxy, {"model": args.model, "input": "Reply with only OK."}, bearer=key_a)
        path = f"/v1/responses/{response['id']}"
        check("key A reads its own response", proxy.call("GET", path, bearer=key_a)[0] == 200)
        check("key B cannot read it", proxy.call("GET", path, bearer=key_b)[0] == 404)
        check("key B cannot delete it", proxy.call("DELETE", path, bearer=key_b)[0] == 404)
        status, parsed = proxy.call("POST", "/v1/responses",
                                    {"model": args.model, "input": "continue", "previous_response_id": response["id"]},
                                    bearer=key_b)
        check("key B cannot continue it", status == 400 and error_code(parsed) == "previous_response_not_found")
        row = db.response(response["id"])
        check("the stored owner is the key id, never the key itself",
              row is not None and row["owner_id"] == owner_a and key_a not in json.dumps(row))
        if earlier:
            check("responses created before keys existed are not visible to a key",
                  proxy.call("GET", f"/v1/responses/{earlier['response_id']}", bearer=key_a)[0] == 404)
        check("key A deletes its own response", proxy.call("DELETE", path, bearer=key_a)[0] == 200)

    report.phase("N", "owner isolation between two API keys", isolation)


def main() -> int:
    parser = argparse.ArgumentParser(description="Live acceptance for /v1/responses against a local proxy.")
    parser.add_argument("command", choices=["run", "restart-check", "keys"])
    parser.add_argument("--base-url", default="http://127.0.0.1:19177", help="proxy root URL")
    parser.add_argument("--db", required=True, help="the proxy's TEMPORARY database copy")
    parser.add_argument("--state", required=True, help="scratch file shared by the commands (holds a codeword)")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--timeout", type=float, default=330.0, help="socket timeout per request, seconds")
    parser.add_argument("--output", help="write a redacted JSON report here")
    parser.add_argument("--allow-remote", action="store_true", help="permit a non-loopback proxy")
    parser.add_argument("--confirm-temporary-db", action="store_true",
                        help="required by `keys`, which inserts two API keys into --db")
    parser.add_argument("--phases", help="`run` only: comma-separated phase letters, run in the given order")
    args = parser.parse_args()

    host = urllib.parse.urlsplit(args.base_url).hostname or ""
    if not args.allow_remote and host not in ("127.0.0.1", "localhost", "::1"):
        parser.error("refusing a non-loopback proxy without --allow-remote")
    if args.command == "keys" and not args.confirm_temporary_db:
        parser.error("`keys` writes to --db; pass --confirm-temporary-db for a temporary copy")

    try:
        proxy = Proxy(args.base_url, args.timeout)
        db = Database(args.db)
        status, _ = proxy.call("GET", "/healthz")
    except (OSError, ValueError, http.client.HTTPException) as exc:
        print(f"[ERROR] proxy or database unavailable: {type(exc).__name__}")
        return 2
    if status != 200:
        print(f"[ERROR] /healthz returned HTTP {status}")
        return 2

    print("=" * 70)
    print(f"/v1/responses live acceptance: {args.command}")
    print("=" * 70)
    report = Report()
    {"run": cmd_run, "restart-check": cmd_restart_check, "keys": cmd_keys}[args.command](args, proxy, db, report)
    return report.finish(args.command, args.output)


if __name__ == "__main__":
    sys.exit(main())
