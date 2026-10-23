"""Output-limit verdicts and local HTTP/CLI integration; no external upstream."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from upstream_capability_probe.cli import main
from upstream_capability_probe.models import CapabilityProfile
from upstream_capability_probe.probes import (
    OUTPUT_LIMIT,
    OUTPUT_LIMIT_PROMPT,
    PROBE_SPECS,
    ProbeContext,
    probe_responses_function_calling,
    probe_responses_max_output_tokens,
    probe_responses_non_stream,
    probe_responses_stream,
    probe_responses_tool_result_continuation,
    probe_responses_vision_data_uri,
    run_probes,
)
from upstream_capability_probe.transport import (
    MAX_JSON_BODY_BYTES,
    JsonHttpResult,
    StreamObservation,
    Transport,
    normalize_base_url,
)


def response_body(**changes):
    body = {
        "object": "response",
        "id": "resp_PRIVATE_RESPONSE_ID",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "max_output_tokens": OUTPUT_LIMIT,
        "output": [{
            "type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "PRIVATE_COMPLETION_TEXT"}],
        }],
        "usage": {
            "input_tokens": 8,
            "output_tokens": OUTPUT_LIMIT,
            "total_tokens": 8 + OUTPUT_LIMIT,
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }
    body.update(changes)
    return body


def function_body():
    return {
        "object": "response", "status": "completed",
        "output": [{
            "type": "function_call", "call_id": "call_synthetic",
            "name": "probe_tool", "arguments": '{"value":"42"}',
        }],
    }


def json_result(body, **changes):
    return JsonHttpResult(
        status=changes.pop("status", 200), content_type="application/json",
        json_value=body, is_json=True, **changes,
    )


class OutputLimitVerdictTests(unittest.TestCase):
    def run_probe(self, result):
        transport = mock.Mock(spec=Transport)
        transport.endpoints = normalize_base_url("http://127.0.0.1:1")
        transport.post_json.return_value = result
        return probe_responses_max_output_tokens(ProbeContext(transport, "synthetic-model"))

    def test_limit_enforced_with_or_without_parameter_echo(self):
        for echo in (True, False):
            with self.subTest(echo=echo):
                body = response_body()
                if not echo:
                    del body["max_output_tokens"]
                result = self.run_probe(json_result(body))
                self.assertEqual(result.status, "supported")
                self.assertEqual(result.reason, "output_limit_enforced")
                self.assertTrue(result.evidence["usage_consistent"])
                self.assertTrue(result.evidence["output_tokens_within_limit"])
                self.assertEqual(result.evidence["parameter_echo_matches"], echo)

    def test_reasoning_only_output_can_consume_the_limit(self):
        body = response_body(output=[])
        body["usage"]["output_tokens_details"]["reasoning_tokens"] = OUTPUT_LIMIT
        self.assertEqual(self.run_probe(json_result(body)).status, "supported")

    def test_acceptance_and_echo_do_not_prove_enforcement(self):
        bodies = [
            response_body(status="completed", incomplete_details=None),
            response_body(incomplete_details=None),
            response_body(incomplete_details={"reason": "content_filter"}),
            response_body(status="completed"),
            response_body(status="failed"),
            response_body(object="chat.completion"),
            response_body(output=None),
            response_body(error={"message": "PRIVATE_ERROR_TEXT"}),
            {"max_output_tokens": OUTPUT_LIMIT},
            [],
        ]
        for index, body in enumerate(bodies):
            with self.subTest(case=index):
                result = self.run_probe(json_result(body))
                self.assertEqual(result.status, "inconclusive")
                self.assertIsNone(result.supported)

    def test_missing_or_invalid_usage_never_enables_the_limit(self):
        invalid = [None, {}, {"output_tokens": OUTPUT_LIMIT}]
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            for value in (True, False, -1, "16", 16.0, None):
                usage = copy.deepcopy(response_body()["usage"])
                usage[key] = value
                invalid.append(usage)
            usage = copy.deepcopy(response_body()["usage"])
            del usage[key]
            invalid.append(usage)
        invalid.extend([
            {"input_tokens": 8, "output_tokens": 0, "total_tokens": 8},
            {"input_tokens": 8, "output_tokens": OUTPUT_LIMIT, "total_tokens": 1},
        ])
        for index, usage in enumerate(invalid):
            with self.subTest(case=index):
                result = self.run_probe(json_result(response_body(usage=usage)))
                self.assertEqual(result.status, "inconclusive")
                self.assertFalse(result.evidence["usage_consistent"])

    def test_invalid_reasoning_usage_is_inconclusive(self):
        for details in ([], {}, {"reasoning_tokens": True}, {"reasoning_tokens": -1},
                        {"reasoning_tokens": OUTPUT_LIMIT + 1}):
            with self.subTest(details=details):
                body = response_body()
                body["usage"]["output_tokens_details"] = details
                self.assertEqual(self.run_probe(json_result(body)).status, "inconclusive")

    def test_optional_reasoning_usage_can_be_absent(self):
        body = response_body()
        del body["usage"]["output_tokens_details"]
        self.assertEqual(self.run_probe(json_result(body)).status, "supported")

    def test_consistent_usage_above_cap_is_unsupported(self):
        for status in ("completed", "incomplete"):
            with self.subTest(status=status):
                body = response_body(status=status)
                body["usage"].update(output_tokens=OUTPUT_LIMIT + 1, total_tokens=OUTPUT_LIMIT + 9)
                result = self.run_probe(json_result(body))
                self.assertEqual(result.status, "unsupported")
                self.assertEqual(result.reason, "output_limit_exceeded")

    def test_timeout_rate_limit_and_server_errors_are_inconclusive(self):
        for status, error in ((None, "timeout"), (429, "rate_limited"), (503, "service_unavailable")):
            with self.subTest(status=status):
                result = self.run_probe(JsonHttpResult(
                    status=status, error_class=error,
                    unsupported_parameter="max_output_tokens",
                ))
                self.assertEqual(result.status, "inconclusive")
                self.assertIsNone(result.supported)

    def test_authentication_is_an_error_not_a_capability_verdict(self):
        for status in (401, 403):
            result = self.run_probe(JsonHttpResult(status=status, error_class="unauthorized"))
            self.assertEqual(result.status, "error")
            self.assertIsNone(result.supported)

    def test_request_is_a_separate_responses_only_probe(self):
        transport = mock.Mock(spec=Transport)
        transport.endpoints = normalize_base_url("http://127.0.0.1:1")
        transport.post_json.return_value = json_result(response_body())
        probe_responses_max_output_tokens(ProbeContext(transport, "synthetic-model"))
        url, payload = transport.post_json.call_args.args
        self.assertEqual(url, transport.endpoints.responses_url)
        self.assertEqual(payload["max_output_tokens"], OUTPUT_LIMIT)
        self.assertEqual(payload["input"], OUTPUT_LIMIT_PROMPT)
        self.assertIs(payload["stream"], False)
        self.assertNotIn("max_tokens", payload)
        self.assertNotIn("max_completion_tokens", payload)

    def test_profiles_never_contain_prompt_response_text_or_ids(self):
        result = self.run_probe(json_result(response_body()))
        profile = CapabilityProfile(target={}, options={}, probes=[result])
        text = profile.to_json()
        for private in (OUTPUT_LIMIT_PROMPT, "PRIVATE_COMPLETION_TEXT", "PRIVATE_RESPONSE_ID"):
            self.assertNotIn(private, text)
        capability = profile.capabilities()["responses_max_output_tokens"]
        self.assertEqual(capability["status"], "supported")
        self.assertEqual(profile.assembly_hints()["responses_max_output_tokens"]["status"], "supported")

    def test_ordinary_responses_probes_omit_the_optional_limit(self):
        for run in (probe_responses_non_stream, probe_responses_stream,
                    probe_responses_function_calling, probe_responses_tool_result_continuation,
                    probe_responses_vision_data_uri):
            with self.subTest(probe=run.__name__):
                transport = mock.Mock(spec=Transport)
                transport.endpoints = normalize_base_url("http://127.0.0.1:1")
                transport.post_json.return_value = json_result(
                    function_body() if run is probe_responses_function_calling else response_body()
                )
                if run is probe_responses_tool_result_continuation:
                    transport.post_json.side_effect = [json_result(function_body()), json_result(response_body())]
                transport.post_stream.return_value = StreamObservation(
                    status=200, is_event_stream=True, data_event_count=2, json_event_count=2,
                    terminal_event_seen=True,
                )
                self.assertEqual(run(ProbeContext(transport, "m")).status, "supported")
                for call in transport.post_json.call_args_list + transport.post_stream.call_args_list:
                    self.assertNotIn("max_output_tokens", call.args[1])
                    self.assertIn("stream", call.args[1])

    def test_no_write_skips_limit_probe_without_generation_requests(self):
        transport = mock.Mock(spec=Transport)
        transport.endpoints = normalize_base_url("http://127.0.0.1:1")
        transport.get_json.return_value = json_result({"object": "list", "data": []})
        results = run_probes(ProbeContext(transport, "m"), allow_stateful=False)
        result = next(result for result in results if result.id == "responses.max_output_tokens")
        self.assertEqual(result.status, "skipped")
        transport.post_json.assert_not_called()
        transport.post_stream.assert_not_called()
        self.assertEqual(len(results), len(PROBE_SPECS))


class _LimitHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, content_type, body):
        data = body if isinstance(body, bytes) else (
            body.encode("utf-8") if isinstance(body, str) else json.dumps(body).encode("utf-8")
        )
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        self.server.requests.append(("GET", self.path, None))
        self.reply(200, "application/json", {"object": "list", "data": []})

    def do_POST(self):  # noqa: N802
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(("POST", self.path, payload))
        if self.path != "/v1/responses":
            self.reply(404, "application/json", {"error": "not found"})
        elif "max_output_tokens" in payload:
            self.reply(*self.server.limit_reply)
        elif payload.get("stream"):
            self.reply(200, "text/event-stream",
                       'event: response.created\ndata: {"type":"response.created"}\n\n'
                       'event: response.completed\ndata: {"type":"response.completed"}\n\n')
        elif payload.get("tools") and isinstance(payload.get("tool_choice"), dict):
            self.reply(200, "application/json", function_body())
        else:
            self.reply(200, "application/json", response_body(status="completed", incomplete_details=None))


class OutputLimitHttpTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _LimitHandler)
        self.server.requests = []
        self.server.limit_reply = (200, "application/json", response_body())
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        self.transport = Transport(self.base_url, timeout=2)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def probe(self):
        return probe_responses_max_output_tokens(ProbeContext(self.transport, "synthetic-model"))

    def test_actual_http_enforced_and_ignored_limits(self):
        self.assertEqual(self.probe().status, "supported")
        self.server.limit_reply = (200, "application/json", response_body(status="completed", incomplete_details=None))
        self.assertEqual(self.probe().status, "inconclusive")

    def test_explicit_rejection_is_detected_without_retaining_error_body(self):
        for status in (400, 422):
            for code in ("unsupported_parameter", "unknown_parameter", "unrecognized_parameter"):
                for wrapped in (True, False):
                    with self.subTest(status=status, code=code, wrapped=wrapped):
                        error = {"param": "max_output_tokens", "code": code, "message": "PRIVATE_PROMPT_AND_TOKEN"}
                        body = {"error": error} if wrapped else error
                        self.server.limit_reply = (status, "application/json", body)
                        result = self.transport.post_json(self.transport.endpoints.responses_url, {"max_output_tokens": OUTPUT_LIMIT})
                        self.assertIsNone(result.json_value)
                        self.assertEqual(result.unsupported_parameter, "max_output_tokens")
                        self.assertNotIn("PRIVATE_PROMPT_AND_TOKEN", repr(result))
                        probe = self.probe()
                        self.assertEqual(probe.status, "unsupported")
                        self.assertEqual(probe.reason, "max_output_tokens_rejected")
                        self.assertNotIn("PRIVATE_PROMPT_AND_TOKEN", json.dumps(probe.to_dict()))

    def test_generic_field_validation_and_unrelated_errors_are_inconclusive(self):
        errors = [
            {"param": "max_output_tokens", "code": "invalid_value", "message": "minimum is larger"},
            {"param": "model", "code": "unsupported_parameter", "message": "max_output_tokens"},
            {"code": "unsupported_parameter", "message": "max_output_tokens"},
            {"param": "max_output_tokens", "code": ["unsupported_parameter"]},
            {"param": ["max_output_tokens"], "code": "unsupported_parameter"},
        ]
        for error in errors:
            with self.subTest(error=error):
                self.server.limit_reply = (400, "application/json", {"error": error})
                self.assertEqual(self.probe().status, "inconclusive")

    def test_rate_limit_and_5xx_cannot_be_misread_as_field_rejection(self):
        for status in (429, 500, 503):
            self.server.limit_reply = (status, "application/json", {"error": {
                "param": "max_output_tokens", "code": "unsupported_parameter",
            }})
            self.assertEqual(self.probe().status, "inconclusive")

    def test_malformed_and_oversized_errors_do_not_yield_a_field_verdict(self):
        for body in (b"not JSON", b"x" * (MAX_JSON_BODY_BYTES + 1)):
            self.server.limit_reply = (400, "application/json", body)
            result = self.transport.post_json(self.transport.endpoints.responses_url, {"max_output_tokens": OUTPUT_LIMIT})
            self.assertIsNone(result.json_value)
            self.assertIsNone(result.unsupported_parameter)
            self.assertEqual(self.probe().status, "inconclusive")

    def test_full_cli_profiles_record_independent_limit_verdict(self):
        for supported in (True, False):
            with self.subTest(supported=supported):
                if not supported:
                    self.server.limit_reply = (400, "application/json", {"error": {
                        "param": "max_output_tokens", "code": "unsupported_parameter",
                        "message": "PRIVATE_UPSTREAM_ERROR",
                    }})
                self.server.requests.clear()
                with tempfile.TemporaryDirectory(dir=os.environ.get("PI_SCRATCH_DIR")) as tmp:
                    profile_path = Path(tmp) / "probe.json"
                    proxy_path = Path(tmp) / "proxy.json"
                    stdout, stderr = io.StringIO(), io.StringIO()
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        code = main([
                            "--base-url", self.base_url, "--model", "synthetic-model", "--timeout", "2",
                            "--output", str(profile_path), "--proxy-profile", str(proxy_path), "--channel", "mock_http",
                        ])
                    self.assertEqual(code, 0)
                    profile = json.loads(profile_path.read_text(encoding="utf-8"))
                    proxy = json.loads(proxy_path.read_text(encoding="utf-8"))
                    expected = "supported" if supported else "unsupported"
                    self.assertEqual(profile["summary"]["probes"]["total"], len(PROBE_SPECS))
                    self.assertEqual(profile["capabilities"]["responses_max_output_tokens"]["status"], expected)
                    self.assertEqual(proxy["capabilities"]["max_output_tokens"]["support"], expected)
                    self.assertEqual(proxy["capabilities"]["max_output_tokens"]["evidence"], "responses.max_output_tokens")
                    self.assertEqual(len(proxy["capabilities"]), 30)
                    for capability in ("responses_non_stream", "responses_stream", "responses_function_calling",
                                       "responses_tool_result_continuation", "responses_vision_data_uri"):
                        self.assertEqual(profile["capabilities"][capability]["status"], "supported", capability)
                    limited = [payload for method, path, payload in self.server.requests
                               if payload and "max_output_tokens" in payload]
                    self.assertEqual(len(limited), 1)
                    serialized = profile_path.read_text(encoding="utf-8") + proxy_path.read_text(encoding="utf-8") + stdout.getvalue() + stderr.getvalue()
                    for private in (OUTPUT_LIMIT_PROMPT, "PRIVATE_COMPLETION_TEXT", "PRIVATE_RESPONSE_ID", "PRIVATE_UPSTREAM_ERROR"):
                        self.assertNotIn(private, serialized)

    def test_no_write_cli_skips_the_new_probe_and_only_lists_models(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("PI_SCRATCH_DIR")) as tmp:
            output = Path(tmp) / "probe.json"
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = main(["--base-url", self.base_url, "--no-write-probes", "--output", str(output)])
            self.assertEqual(code, 0)
            profile = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(profile["capabilities"]["responses_max_output_tokens"]["status"], "skipped")
            self.assertEqual(self.server.requests, [("GET", "/v1/models", None)])


if __name__ == "__main__":
    unittest.main()
