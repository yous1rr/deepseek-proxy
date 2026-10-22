#!/usr/bin/env python3
"""
tools/verify_branches.py

Black-box verification tool for deepseek-proxy branch compatibility.
Validates branch, rollback, revision switching, and regeneration behavior
as specified in BRANCH_IMPLEMENTATION_PLAN.md and PI_DESKTOP_BRANCH_VALIDATION.md.

Standard Python library only. No third-party dependencies.
Never spawns or stops the proxy process, never modifies real databases,
and never reads account credentials from disk.

Scenarios tested:
  Phase A: Establish a root token in a fresh conversation.
  Phase B: Establish a secret token in continuation of Phase A.
  Phase C: Normal continuation asking for remembered root + secret as small JSON.
  Phase D: Rollback to history through Phase A only (using actual A response),
           asking same question without leaking secret; secret must be null/absent
           and root must be retained.
  Phase E: Switch to full original B/C history and ask for original secret;
           secret must be retained.
  Phase F: Switch back to rollback branch (A + D history with actual D reply)
           and verify secret is absent.
  Phase G: Regenerate identical rollback request safely; verify secret is absent.

If --db is provided (read-only SQLite path of temporary proxy DB copy):
  Polls thread records after each complete SSE response:
  - Normal continuation reuses session_id (Phases B, C).
  - Branch, rollback, and regeneration changes session_id (Phases D, E, F, G).

Security and redaction rules:
  Never prints raw model responses, prompts, account credentials, API keys,
  or session identifiers. Outputs only checks, pass/fail status, phase timings,
  and safe summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

LIMITATIONS: List[str] = [
    "Tests branch context isolation via synthetic tokens and JSON parsing; does not test arbitrary conversational nuance.",
    "DB inspection operates read-only on the specified temporary copy; does not query upstream cloud session state directly.",
    "Emulates Pi Desktop session affinity headers ('x-session-affinity'); does not test desktop UI revision trees.",
    "Relies on the model following instructions to return valid small JSON objects with requested keys.",
    "Assumes the proxy enforces branch-aware session management per BRANCH_IMPLEMENTATION_PLAN.md.",
]


def normalize_endpoint(base_url: str) -> str:
    """Normalize base URL to /v1/chat/completions."""
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def compute_db_thread_id(session_uuid: str) -> str:
    """Compute proxy's hashed thread_id matching main.rs sha256_hex('thread_', session, None)."""
    digest = hashlib.sha256(session_uuid.encode("utf-8")).hexdigest()
    return f"thread_{digest[:24]}"


def open_db_readonly(db_path: str) -> sqlite3.Connection:
    """Open SQLite strictly read-only, without a read/write fallback."""
    path = Path(db_path).resolve(strict=True)
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=5.0)
    conn.execute("PRAGMA query_only=ON")
    conn.isolation_level = None
    return conn


def read_thread_record(conn: sqlite3.Connection, session_uuid: str) -> Optional[Dict[str, Any]]:
    """Read thread record for session_uuid (checks both raw UUID and proxy hashed key)."""
    hashed_id = compute_db_thread_id(session_uuid)
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT thread_id, session_id, parent_message_id, updated_at "
            "FROM threads WHERE thread_id = ? OR thread_id = ?",
            (session_uuid, hashed_id),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "thread_id": row[0],
            "session_id": row[1],
            "parent_message_id": row[2],
            "updated_at": row[3],
        }
    except sqlite3.OperationalError:
        return None


def poll_thread_record(
    conn: sqlite3.Connection,
    session_uuid: str,
    timeout_sec: float = 4.0,
    expect_change_from: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Bounded polling for thread record persistence after SSE [DONE]."""
    deadline = time.time() + timeout_sec
    last_rec = None
    while time.time() < deadline:
        last_rec = read_thread_record(conn, session_uuid)
        if last_rec is not None:
            sid = last_rec.get("session_id")
            if expect_change_from is None:
                return last_rec
            elif sid != expect_change_from:
                return last_rec
        time.sleep(0.05)
    return last_rec


def stream_completion(
    endpoint: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: float = 60.0,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Send completion request and stream SSE response.
    Parses delta.content, tool_calls, and proxy JSON errors until [DONE].
    """
    req_headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    req_headers.update(headers)
    body_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(endpoint, data=body_bytes, headers=req_headers, method="POST")

    content_parts: List[str] = []
    tool_calls_map: Dict[int, Dict[str, Any]] = {}
    stream_error: Optional[str] = None
    received_done = False

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        received_done = True
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        raise RuntimeError("Invalid JSON in SSE event") from None

                    if isinstance(chunk, dict) and "error" in chunk:
                        err = chunk["error"]
                        msg = err.get("message", "Stream error") if isinstance(err, dict) else str(err)
                        stream_error = msg
                        continue

                    if isinstance(chunk, dict):
                        choices = chunk.get("choices")
                        if isinstance(choices, list) and len(choices) > 0:
                            choice = choices[0]
                            delta = choice.get("delta") if isinstance(choice, dict) else {}
                            if isinstance(delta, dict):
                                c = delta.get("content")
                                if c:
                                    content_parts.append(c)
                                tc_list = delta.get("tool_calls")
                                if isinstance(tc_list, list):
                                    for tc in tc_list:
                                        if isinstance(tc, dict):
                                            idx = tc.get("index", 0)
                                            if idx not in tool_calls_map:
                                                tool_calls_map[idx] = {
                                                    "id": tc.get("id") or "",
                                                    "type": tc.get("type") or "function",
                                                    "name": (tc.get("function") or {}).get("name") or "",
                                                    "arguments_parts": [],
                                                }
                                            else:
                                                if tc.get("id"):
                                                    tool_calls_map[idx]["id"] = tc.get("id")
                                                if tc.get("type"):
                                                    tool_calls_map[idx]["type"] = tc.get("type")
                                                fn = tc.get("function") or {}
                                                if fn.get("name"):
                                                    tool_calls_map[idx]["name"] = fn.get("name")
                                            fn = tc.get("function") or {}
                                            arg_piece = fn.get("arguments")
                                            if arg_piece:
                                                tool_calls_map[idx]["arguments_parts"].append(arg_piece)
    except urllib.error.HTTPError as exc:
        # Provider error bodies may contain arbitrary upstream/user data.
        raise RuntimeError(f"HTTP {exc.code}") from None
    except urllib.error.URLError:
        raise RuntimeError("Connection failed") from None

    if stream_error:
        raise RuntimeError("Proxy returned a streaming error")
    if not received_done:
        raise RuntimeError("SSE ended without [DONE]")

    assembled_content = "".join(content_parts)
    assembled_tool_calls: List[Dict[str, Any]] = []
    for idx in sorted(tool_calls_map.keys()):
        item = tool_calls_map[idx]
        assembled_tool_calls.append({
            "id": item["id"],
            "type": item["type"],
            "function": {
                "name": item["name"],
                "arguments": "".join(item["arguments_parts"]),
            },
        })

    return assembled_content, assembled_tool_calls


def parse_json_response(raw_text: str) -> Dict[str, Any]:
    """Parse a small JSON object, optionally inside a Markdown fence."""
    text = raw_text.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if match:
            text = match.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError("Expected a valid JSON object from assistant") from None
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object from assistant")
    return data


def check_has_token(parsed: Dict[str, Any], raw_text: str, key: str, token: str) -> bool:
    """Require the requested JSON field to contain the exact token."""
    return parsed.get(key) == token


def check_secret_absent(parsed: Dict[str, Any], raw_text: str, secret_token: str) -> bool:
    """Require explicit absence/unknown and reject leaked tokens anywhere."""
    if secret_token in raw_text:
        return False
    value = parsed.get("secret")
    return value is None or str(value).strip().lower() in ("unknown", "none", "not set", "null", "")


def execute_check(
    phase_summary: Dict[str, Any],
    name: str,
    passed: bool,
    description: str,
) -> None:
    """Record a check result and print safe sanitized status."""
    phase_summary["checks"].append({
        "name": name,
        "passed": passed,
        "description": description,
    })
    if passed:
        print(f"    [PASS] {description}")
    else:
        print(f"    [FAIL] {description}")
        phase_summary["status"] = "FAIL"


def run_verification(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute all 7 branch verification scenarios."""
    endpoint = normalize_endpoint(args.base_url)
    root_token = f"ROOT_{uuid.uuid4().hex[:8].upper()}"
    secret_token = f"SECRET_{uuid.uuid4().hex[:8].upper()}"
    session_uuid = str(uuid.uuid4())

    headers: Dict[str, str] = {
        "x-session-affinity": session_uuid,
        "x-session-id": session_uuid,
    }
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise ValueError("The requested API key environment variable is empty")
        headers["Authorization"] = f"Bearer {api_key}"

    db_conn = None
    if args.db:
        db_conn = open_db_readonly(args.db)

    print("=" * 70)
    print("DeepSeek Proxy Black-Box Branch Verification")
    print("=" * 70)
    print(f"Endpoint: {endpoint}")
    print(f"Model:    {args.model}")
    print(f"Database: {'configured (read-only)' if db_conn else 'skipped (no --db provided)'}")
    print("-" * 70)

    report: Dict[str, Any] = {
        "status": "PASS",
        "endpoint": endpoint,
        "model": args.model,
        "db_inspected": db_conn is not None,
        "total_duration_sec": 0.0,
        "phases": [],
        "summary": {
            "phases_total": 7,
            "phases_passed": 0,
            "phases_failed": 0,
            "checks_total": 0,
            "checks_passed": 0,
            "checks_failed": 0,
        },
        "limitations": LIMITATIONS,
    }

    start_all = time.time()
    system_msg = {
        "role": "system",
        "content": "You are a precise assistant. When asked for JSON, output only a valid JSON object.",
    }

    # Tracking DB session identifiers across turns (without printing raw IDs)
    sid_A: Optional[str] = None
    sid_B: Optional[str] = None
    sid_C: Optional[str] = None
    sid_D: Optional[str] = None
    sid_E: Optional[str] = None
    sid_F: Optional[str] = None
    sid_G: Optional[str] = None

    def make_payload(messages: List[Dict[str, str]]) -> Dict[str, Any]:
        return {
            "model": args.model,
            "stream": True,
            "messages": messages,
            "prompt_cache_key": session_uuid,
        }

    # -------------------------------------------------------------------------
    # Phase A: Establish Root Token
    # -------------------------------------------------------------------------
    print("[Phase A] Establish root token...")
    t0 = time.time()
    phase_A: Dict[str, Any] = {"phase": "A", "name": "establish_root_token", "status": "PASS", "checks": []}
    prompt_A = f"Remember this root token: {root_token}. Acknowledge with a small JSON object with key 'status' set to 'ok'."
    messages_A = [system_msg, {"role": "user", "content": prompt_A}]
    resp_A, _ = stream_completion(endpoint, headers, make_payload(messages_A), timeout=args.timeout)
    phase_A["duration_sec"] = round(time.time() - t0, 3)

    execute_check(phase_A, "response_non_empty", len(resp_A.strip()) > 0, "Response received and non-empty")
    if db_conn:
        rec_A = poll_thread_record(db_conn, session_uuid, timeout_sec=args.timeout_db)
        sid_A = rec_A.get("session_id") if rec_A else None
        execute_check(phase_A, "db_session_persisted", sid_A is not None, "Thread record created in database")
    report["phases"].append(phase_A)
    if phase_A["status"] == "FAIL":
        raise AssertionError("Phase A failed: unable to establish root token")

    # -------------------------------------------------------------------------
    # Phase B: Establish Secret Token (to be discarded by rollback)
    # -------------------------------------------------------------------------
    print("[Phase B] Establish secret token (to be rolled back)...")
    t0 = time.time()
    phase_B: Dict[str, Any] = {"phase": "B", "name": "establish_secret_token", "status": "PASS", "checks": []}
    prompt_B = f"Remember this secret token: {secret_token}. Acknowledge with a small JSON object with key 'status' set to 'ok'."
    messages_B = [
        system_msg,
        {"role": "user", "content": prompt_A},
        {"role": "assistant", "content": resp_A},
        {"role": "user", "content": prompt_B},
    ]
    resp_B, _ = stream_completion(endpoint, headers, make_payload(messages_B), timeout=args.timeout)
    phase_B["duration_sec"] = round(time.time() - t0, 3)

    execute_check(phase_B, "response_non_empty", len(resp_B.strip()) > 0, "Response received and non-empty")
    if db_conn:
        rec_B = poll_thread_record(db_conn, session_uuid, timeout_sec=args.timeout_db)
        sid_B = rec_B.get("session_id") if rec_B else None
        execute_check(
            phase_B,
            "db_session_reused",
            sid_B is not None and sid_B == sid_A,
            "Session reused on normal continuation",
        )
    report["phases"].append(phase_B)
    if phase_B["status"] == "FAIL":
        raise AssertionError("Phase B failed: unable to establish secret token")

    # -------------------------------------------------------------------------
    # Phase C: Normal Continuation (Ask for remembered root + secret as small JSON)
    # -------------------------------------------------------------------------
    print("[Phase C] Normal continuation query for root and secret tokens...")
    t0 = time.time()
    phase_C: Dict[str, Any] = {"phase": "C", "name": "normal_continuation_query", "status": "PASS", "checks": []}
    prompt_C = (
        "Return the remembered root token and secret token as a JSON object with keys 'root' and 'secret'. "
        "If any token was never established or is unknown, set its value to null. Output only valid JSON."
    )
    messages_C = [
        system_msg,
        {"role": "user", "content": prompt_A},
        {"role": "assistant", "content": resp_A},
        {"role": "user", "content": prompt_B},
        {"role": "assistant", "content": resp_B},
        {"role": "user", "content": prompt_C},
    ]
    resp_C, _ = stream_completion(endpoint, headers, make_payload(messages_C), timeout=args.timeout)
    phase_C["duration_sec"] = round(time.time() - t0, 3)

    try:
        parsed_C = parse_json_response(resp_C)
        json_ok = True
    except Exception:
        parsed_C = {}
        json_ok = False
    execute_check(phase_C, "json_parse", json_ok, "JSON response parsed successfully")
    execute_check(
        phase_C,
        "root_retained",
        check_has_token(parsed_C, resp_C, "root", root_token),
        "Root token retained in normal continuation",
    )
    execute_check(
        phase_C,
        "secret_retained",
        check_has_token(parsed_C, resp_C, "secret", secret_token),
        "Secret token retained in normal continuation",
    )
    if db_conn:
        rec_C = poll_thread_record(db_conn, session_uuid, timeout_sec=args.timeout_db)
        sid_C = rec_C.get("session_id") if rec_C else None
        execute_check(
            phase_C,
            "db_session_reused",
            sid_C is not None and sid_C == sid_B,
            "Session reused on normal continuation",
        )
    report["phases"].append(phase_C)
    if phase_C["status"] == "FAIL":
        raise AssertionError("Phase C failed: normal continuation failed to retain established tokens")

    # -------------------------------------------------------------------------
    # Phase D: Rollback to history only through A (with actual A response)
    # -------------------------------------------------------------------------
    print("[Phase D] Rollback to history through Phase A only...")
    t0 = time.time()
    phase_D: Dict[str, Any] = {"phase": "D", "name": "rollback_branch_query", "status": "PASS", "checks": []}
    # Careful prompt identical to C: does NOT mention or leak secret_token
    prompt_D = prompt_C
    messages_D = [
        system_msg,
        {"role": "user", "content": prompt_A},
        {"role": "assistant", "content": resp_A},
        {"role": "user", "content": prompt_D},
    ]
    resp_D, _ = stream_completion(endpoint, headers, make_payload(messages_D), timeout=args.timeout)
    phase_D["duration_sec"] = round(time.time() - t0, 3)

    try:
        parsed_D = parse_json_response(resp_D)
        json_ok = True
    except Exception:
        parsed_D = {}
        json_ok = False
    execute_check(phase_D, "json_parse", json_ok, "JSON response parsed successfully")
    execute_check(
        phase_D,
        "root_retained",
        check_has_token(parsed_D, resp_D, "root", root_token),
        "Root token retained in rollback branch",
    )
    execute_check(
        phase_D,
        "secret_absent",
        check_secret_absent(parsed_D, resp_D, secret_token),
        "Secret token absent or null in rollback branch",
    )
    if db_conn:
        rec_D = poll_thread_record(db_conn, session_uuid, timeout_sec=args.timeout_db, expect_change_from=sid_C)
        sid_D = rec_D.get("session_id") if rec_D else None
        execute_check(
            phase_D,
            "db_session_changed",
            sid_D is not None and sid_D != sid_C,
            "Session changed on branch rollback",
        )
    report["phases"].append(phase_D)
    if phase_D["status"] == "FAIL":
        raise AssertionError("Phase D failed: rollback branch leaked discarded secret token or lost root")

    # -------------------------------------------------------------------------
    # Phase E: Switch back to full original B/C history and ask original secret
    # -------------------------------------------------------------------------
    print("[Phase E] Switch back to full original B/C history...")
    t0 = time.time()
    phase_E: Dict[str, Any] = {"phase": "E", "name": "switch_original_branch", "status": "PASS", "checks": []}
    prompt_E = "Return the secret token from earlier in this conversation as a JSON object with key 'secret'. Output only valid JSON."
    messages_E = [
        system_msg,
        {"role": "user", "content": prompt_A},
        {"role": "assistant", "content": resp_A},
        {"role": "user", "content": prompt_B},
        {"role": "assistant", "content": resp_B},
        {"role": "user", "content": prompt_C},
        {"role": "assistant", "content": resp_C},
        {"role": "user", "content": prompt_E},
    ]
    resp_E, _ = stream_completion(endpoint, headers, make_payload(messages_E), timeout=args.timeout)
    phase_E["duration_sec"] = round(time.time() - t0, 3)

    try:
        parsed_E = parse_json_response(resp_E)
        json_ok = True
    except Exception:
        parsed_E = {}
        json_ok = False
    execute_check(phase_E, "json_parse", json_ok, "JSON response parsed successfully")
    execute_check(
        phase_E,
        "secret_retained",
        check_has_token(parsed_E, resp_E, "secret", secret_token),
        "Original secret token retained when switching back to original branch",
    )
    if db_conn:
        rec_E = poll_thread_record(db_conn, session_uuid, timeout_sec=args.timeout_db, expect_change_from=sid_D)
        sid_E = rec_E.get("session_id") if rec_E else None
        execute_check(
            phase_E,
            "db_session_changed",
            sid_E is not None and sid_E != sid_D,
            "Session changed when switching branches",
        )
    report["phases"].append(phase_E)
    if phase_E["status"] == "FAIL":
        raise AssertionError("Phase E failed: original secret token lost on branch switch")

    # -------------------------------------------------------------------------
    # Phase F: Switch back to rollback branch with actual reply
    # -------------------------------------------------------------------------
    print("[Phase F] Switch back to rollback branch with actual reply...")
    t0 = time.time()
    phase_F: Dict[str, Any] = {"phase": "F", "name": "switch_rollback_branch", "status": "PASS", "checks": []}
    prompt_F = (
        "State whether any secret token was ever established in this conversation. "
        "Return a JSON object with keys 'has_secret' (boolean) and 'secret' (string or null). Output only valid JSON."
    )
    messages_F = [
        system_msg,
        {"role": "user", "content": prompt_A},
        {"role": "assistant", "content": resp_A},
        {"role": "user", "content": prompt_D},
        {"role": "assistant", "content": resp_D},
        {"role": "user", "content": prompt_F},
    ]
    resp_F, _ = stream_completion(endpoint, headers, make_payload(messages_F), timeout=args.timeout)
    phase_F["duration_sec"] = round(time.time() - t0, 3)

    try:
        parsed_F = parse_json_response(resp_F)
        json_ok = True
    except Exception:
        parsed_F = {}
        json_ok = False
    execute_check(phase_F, "json_parse", json_ok, "JSON response parsed successfully")
    execute_check(
        phase_F,
        "secret_absent",
        check_secret_absent(parsed_F, resp_F, secret_token) and parsed_F.get("has_secret") is not True,
        "Secret token confirmed absent on rollback branch",
    )
    if db_conn:
        rec_F = poll_thread_record(db_conn, session_uuid, timeout_sec=args.timeout_db, expect_change_from=sid_E)
        sid_F = rec_F.get("session_id") if rec_F else None
        execute_check(
            phase_F,
            "db_session_changed",
            sid_F is not None and sid_F != sid_E,
            "Session changed when switching back to rollback branch",
        )
    report["phases"].append(phase_F)
    if phase_F["status"] == "FAIL":
        raise AssertionError("Phase F failed: rollback branch unexpectedly recovered discarded secret")

    # -------------------------------------------------------------------------
    # Phase G: Regenerate identical rollback request safely
    # -------------------------------------------------------------------------
    print("[Phase G] Regenerate identical rollback request...")
    t0 = time.time()
    phase_G: Dict[str, Any] = {"phase": "G", "name": "regenerate_rollback_request", "status": "PASS", "checks": []}
    # In Pi Desktop, regenerate sends identical messages history as the last turn
    messages_G = list(messages_F)
    resp_G, _ = stream_completion(endpoint, headers, make_payload(messages_G), timeout=args.timeout)
    phase_G["duration_sec"] = round(time.time() - t0, 3)

    try:
        parsed_G = parse_json_response(resp_G)
        json_ok = True
    except Exception:
        parsed_G = {}
        json_ok = False
    execute_check(phase_G, "json_parse", json_ok, "JSON response parsed successfully")
    execute_check(
        phase_G,
        "secret_absent",
        check_secret_absent(parsed_G, resp_G, secret_token) and parsed_G.get("has_secret") is not True,
        "Secret token confirmed absent during regeneration",
    )
    if db_conn:
        rec_G = poll_thread_record(db_conn, session_uuid, timeout_sec=args.timeout_db, expect_change_from=sid_F)
        sid_G = rec_G.get("session_id") if rec_G else None
        execute_check(
            phase_G,
            "db_session_changed",
            sid_G is not None and sid_G != sid_F,
            "Session changed on same-prompt regeneration",
        )
    report["phases"].append(phase_G)
    if phase_G["status"] == "FAIL":
        raise AssertionError("Phase G failed: regeneration failed to isolate discarded context")

    # -------------------------------------------------------------------------
    # Aggregate summary
    # -------------------------------------------------------------------------
    total_elapsed = round(time.time() - start_all, 3)
    report["total_duration_sec"] = total_elapsed

    p_passed = sum(1 for p in report["phases"] if p["status"] == "PASS")
    p_failed = sum(1 for p in report["phases"] if p["status"] != "PASS")
    c_total = sum(len(p["checks"]) for p in report["phases"])
    c_passed = sum(sum(1 for c in p["checks"] if c["passed"]) for p in report["phases"])
    c_failed = c_total - c_passed

    report["summary"]["phases_passed"] = p_passed
    report["summary"]["phases_failed"] = p_failed
    report["summary"]["checks_total"] = c_total
    report["summary"]["checks_passed"] = c_passed
    report["summary"]["checks_failed"] = c_failed

    if p_failed > 0 or c_failed > 0:
        report["status"] = "FAIL"

    print("-" * 70)
    print(f"VERDICT: {'ALL 7 PHASES PASSED' if report['status'] == 'PASS' else 'VERIFICATION FAILED'}")
    print(f"Summary: {p_passed}/{len(report['phases'])} phases passed, {c_passed}/{c_total} checks passed ({total_elapsed}s)")
    print("=" * 70)

    if db_conn:
        db_conn.close()

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Black-box branch verification tool for deepseek-proxy.",
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Base URL of proxy, e.g. http://127.0.0.1:19175/v1",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="SQLite read-only path of temporary proxy DB copy",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Name of environment variable containing API key (never printed)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="JSON summary path to write results",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds for completion streams (default: 60.0)",
    )
    parser.add_argument(
        "--timeout-db",
        type=float,
        default=4.0,
        help="Timeout in seconds for bounded DB checkpoint polling (default: 4.0)",
    )
    parser.add_argument(
        "--model",
        default="deepseek-chat",
        help="Model name to test (default: deepseek-chat)",
    )

    args = parser.parse_args()

    report: Optional[Dict[str, Any]] = None
    exit_code = 0

    try:
        report = run_verification(args)
        if report.get("status") != "PASS":
            exit_code = 1
    except AssertionError as exc:
        print(f"\n[ERROR] Assertion failed: {exc}")
        exit_code = 1
        if report is not None:
            report["status"] = "FAIL"
            report["error"] = str(exc)
        else:
            report = {
                "status": "FAIL",
                "error": str(exc),
                "limitations": LIMITATIONS,
            }
    except Exception as exc:
        print(f"\n[ERROR] Execution failed: {exc}")
        exit_code = 1
        if report is not None:
            report["status"] = "FAIL"
            report["error"] = str(exc)
        else:
            report = {
                "status": "FAIL",
                "error": str(exc),
                "limitations": LIMITATIONS,
            }
    finally:
        if args.output and report is not None:
            try:
                out_dir = os.path.dirname(os.path.abspath(args.output))
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
                print(f"Summary report written to {args.output}")
            except Exception as e:
                print(f"[WARN] Failed to write output file: {e}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
