"""Offline self-tests for the upstream capability probe.

Standard-library-only ``unittest`` suite.  Everything is local: tests use
loopback HTTP servers and pure functions; no external upstream is contacted.

Run from the tool directory::

    PYTHONPATH=src python -m unittest discover -s tests -v

On PowerShell::

    $env:PYTHONPATH = "src"; python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from upstream_capability_probe.cli import main
from upstream_capability_probe.models import (
    CapabilityProfile,
    ProbeResult,
    STATUS_ERROR,
    STATUS_INCONCLUSIVE,
    STATUS_SKIPPED,
    STATUS_SUPPORTED,
    STATUS_UNSUPPORTED,
)
from upstream_capability_probe.probes import PROBE_SPECS, _failure
from upstream_capability_probe.transport import (
    InvalidBaseUrl,
    Transport,
    normalize_base_url,
)


class BaseUrlTests(unittest.TestCase):
    def test_bare_host_gets_v1(self) -> None:
        endpoints = normalize_base_url("https://api.example.com")
        self.assertEqual(endpoints.api_root, "/v1")
        self.assertEqual(endpoints.chat_completions_url, "https://api.example.com/v1/chat/completions")
        self.assertEqual(endpoints.responses_url, "https://api.example.com/v1/responses")
        self.assertEqual(endpoints.models_url, "https://api.example.com/v1/models")

    def test_v1_and_trailing_slash(self) -> None:
        endpoints = normalize_base_url("https://api.example.com/v1/")
        self.assertEqual(endpoints.api_root, "/v1")
        self.assertEqual(endpoints.display_url, "https://api.example.com/v1")

    def test_full_endpoint_urls_normalize(self) -> None:
        for raw in (
            "https://api.example.com/v1/chat/completions",
            "https://api.example.com/v1/responses",
            "https://api.example.com/v1/models",
        ):
            endpoints = normalize_base_url(raw)
            self.assertEqual(endpoints.api_root, "/v1", raw)
            self.assertEqual(
                endpoints.chat_completions_url,
                "https://api.example.com/v1/chat/completions",
                raw,
            )

    def test_custom_prefix_preserved(self) -> None:
        endpoints = normalize_base_url("https://gateway.example.com/openai/v1")
        self.assertEqual(endpoints.display_url, "https://gateway.example.com/openai/v1")
        self.assertEqual(
            endpoints.responses_url, "https://gateway.example.com/openai/v1/responses"
        )

    def test_query_preserved_but_not_displayed(self) -> None:
        endpoints = normalize_base_url("https://api.example.com/v1?api-version=2024-01")
        self.assertEqual(endpoints.display_url, "https://api.example.com/v1")
        self.assertEqual(
            endpoints.chat_completions_url,
            "https://api.example.com/v1/chat/completions?api-version=2024-01",
        )

    def test_rejects_unsafe_urls(self) -> None:
        for raw in (
            "",
            "ftp://api.example.com",
            "file:///etc/passwd",
            "api.example.com",
            "https://user:secret@api.example.com",
            "https://api.example.com:notaport",
        ):
            with self.assertRaises(InvalidBaseUrl, msg=raw):
                normalize_base_url(raw)

    def test_display_url_has_no_userinfo(self) -> None:
        # Userinfo is rejected, so nothing secret can appear in display_url.
        endpoints = normalize_base_url("HTTPS://API.example.com/v1")
        self.assertEqual(endpoints.scheme, "https")
        self.assertNotIn("@", endpoints.display_url)


class ProfileRedactionTests(unittest.TestCase):
    def _result(self, status: str = STATUS_SUPPORTED, supported: bool | None = True) -> ProbeResult:
        return ProbeResult(
            id="chat.completions.stream",
            title="Chat Completions stream=true",
            capability="chat_completions_stream",
            kind="stateful",
            status=status,
            supported=supported,
            reason="sse_stream_observed",
            evidence={
                "is_event_stream": True,
                "event_count": 3,
                "sse_done_seen": True,
                "event_names": ["message"],
            },
        )

    def test_profile_has_no_secret_placeholders(self) -> None:
        profile = CapabilityProfile(
            target={"base_url": "https://api.example.com/v1", "auth": "bearer_env"},
            options={"model": "test-model"},
            probes=[self._result()],
        )
        serialized = profile.to_json()
        payload = json.loads(serialized)
        self.assertEqual(payload["security"]["contains_prompts"], False)
        self.assertEqual(payload["security"]["raw_response_retained"], False)
        self.assertNotIn("SECRET_TOKEN", serialized)
        self.assertNotIn("Bearer", serialized)
        self.assertNotIn("Authorization", serialized)

    def test_raw_body_is_not_serialized(self) -> None:
        # ProbeResult has no field capable of holding a raw body.
        result = self._result()
        self.assertFalse(hasattr(result, "raw_body"))
        self.assertFalse(hasattr(result, "response_body"))
        self.assertFalse(hasattr(result, "prompt"))

    def test_status_supported_consistency(self) -> None:
        with self.assertRaises(ValueError):
            ProbeResult(
                id="x", title="x", capability="x", kind="read_only",
                status=STATUS_SUPPORTED, supported=False, reason="bad",
            )
        with self.assertRaises(ValueError):
            ProbeResult(
                id="x", title="x", capability="x", kind="read_only",
                status=STATUS_SKIPPED, supported=True, reason="bad",
            )



class FailureClassificationTests(unittest.TestCase):
    """Direct coverage of the HTTP failure-to-status mapping."""

    def _fail(self, http_status=None, error_class="http_error", **kwargs):
        return _failure(
            "probe.x", "Probe X", "probe_x", "stateful",
            http_status=http_status, error_class=error_class, **kwargs,
        )

    def test_404_405_501_are_unsupported(self):
        for code in (404, 405, 501):
            result = self._fail(http_status=code)
            self.assertEqual(result.status, STATUS_UNSUPPORTED, code)
            self.assertFalse(result.supported)

    def test_auth_failures_are_error(self):
        for code, reason in ((401, "authentication_failed"), (403, "forbidden")):
            result = self._fail(http_status=code)
            self.assertEqual(result.status, STATUS_ERROR, code)
            self.assertEqual(result.reason, reason)
            self.assertIsNone(result.supported)

    def test_timeout_and_5xx_are_inconclusive(self):
        timeout = self._fail(http_status=None, error_class="timeout")
        server_error = self._fail(http_status=503)
        rate_limited = self._fail(http_status=429)
        for result, reason in ((timeout, "timeout"), (server_error, "upstream_server_error"), (rate_limited, "rate_limited")):
            self.assertEqual(result.status, STATUS_INCONCLUSIVE)
            self.assertEqual(result.reason, reason)
            self.assertIsNone(result.supported)

    def test_bad_request_defaults_inconclusive_but_opt_in_unsupported(self):
        default = self._fail(http_status=400)
        self.assertEqual(default.status, STATUS_INCONCLUSIVE)
        opted = self._fail(http_status=400, bad_request_unsupported=True, bad_request_reason="image_input_rejected")
        self.assertEqual(opted.status, STATUS_UNSUPPORTED)
        self.assertEqual(opted.reason, "image_input_rejected")

    def test_redirect_is_inconclusive_not_followed(self):
        result = self._fail(http_status=302)
        self.assertEqual(result.status, STATUS_INCONCLUSIVE)
        self.assertEqual(result.reason, "unexpected_redirect")

    def test_error_result_carries_no_body(self):
        result = self._fail(http_status=500)
        allowed = {"status", "supported", "reason", "evidence", "error_class", "http_status"}
        self.assertTrue(set(result.to_dict()) <= allowed | {"id", "title", "capability", "kind"})


class DerivedCapabilityTests(unittest.TestCase):
    def _profile(self, **evidence):
        probe = ProbeResult(
            id="chat.completions.stream", title="stream", capability="chat_completions_stream",
            kind="stateful", status=STATUS_SUPPORTED, supported=True,
            reason="sse_stream_observed", evidence=evidence,
        )
        return CapabilityProfile(target={}, options={}, probes=[probe])

    def test_sse_done_and_named_events_derived(self):
        caps = self._profile(sse_done_seen=True, event_names=["message"]).capabilities()
        self.assertEqual(caps["chat_completions_sse_done"]["status"], STATUS_SUPPORTED)
        self.assertEqual(caps["chat_completions_sse_named_events"]["status"], STATUS_SUPPORTED)

    def test_missing_done_and_anonymous_events(self):
        caps = self._profile(sse_done_seen=False, event_names=[]).capabilities()
        self.assertEqual(caps["chat_completions_sse_done"]["status"], STATUS_UNSUPPORTED)
        self.assertEqual(caps["chat_completions_sse_named_events"]["status"], STATUS_UNSUPPORTED)

    def test_usage_absent_is_unsupported(self):
        probe = ProbeResult(
            id="chat.completions.non_stream", title="chat", capability="chat_completions_non_stream",
            kind="stateful", status=STATUS_SUPPORTED, supported=True,
            reason="chat_completion_observed",
            evidence={"usage_present": False, "usage_fields": []},
        )
        caps = CapabilityProfile(target={}, options={}, probes=[probe]).capabilities()
        self.assertEqual(caps["chat_completions_usage"]["status"], STATUS_UNSUPPORTED)
        self.assertFalse(caps["chat_completions_usage"]["supported"])


class _Handler(BaseHTTPRequestHandler):
    server_version = "probe-test"
    routes: dict[str, Any] = {}

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        pass

    def _respond(self) -> None:
        path = self.path.split("?", 1)[0]
        route = self.routes.get(path)
        if route is None:
            self.send_response(404)
            self.end_headers()
            return
        if callable(route):
            status, content_type, body = route(self)
        else:
            status, content_type, body = route
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        self._respond()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        self._respond()


class _ServerCase(unittest.TestCase):
    routes: dict[str, Any] = {}

    def setUp(self) -> None:
        self.routes = dict(type(self).routes)
        handler = type("Handler", (_Handler,), {"routes": self.routes})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://127.0.0.1:{port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class HttpClassificationTests(_ServerCase):
    routes = {
        "/v1/chat/completions": (401, "application/json", '{"error":"unauthorized"}'),
        "/v1/responses": (404, "application/json", '{"error":"not found"}'),
        "/v1/models": (405, "application/json", '{"error":"method not allowed"}'),
    }

    def test_401_is_error(self) -> None:
        transport = Transport(self.base_url, timeout=5)
        result = transport.post_json(
            transport.endpoints.chat_completions_url, {"model": "m"}
        )
        self.assertEqual(result.status, 401)
        self.assertEqual(result.error_class, "unauthorized")

    def test_404_and_405_classified(self) -> None:
        transport = Transport(self.base_url, timeout=5)
        responses = transport.post_json(transport.endpoints.responses_url, {})
        models = transport.get_json(transport.endpoints.models_url)
        self.assertEqual(responses.error_class, "endpoint_not_found")
        self.assertEqual(models.error_class, "method_not_allowed")

    def test_probe_converts_http_errors_without_stopping(self) -> None:
        from upstream_capability_probe.probes import ProbeContext, run_probes

        transport = Transport(self.base_url, timeout=5)
        results = run_probes(ProbeContext(transport=transport, model="m"))
        self.assertEqual(len(results), len(PROBE_SPECS))
        # 404/405 responses map to unsupported rather than crash.
        responses = next(r for r in results if r.id == "responses.non_stream")
        models = next(r for r in results if r.id == "models.list")
        self.assertEqual(responses.status, STATUS_UNSUPPORTED)
        self.assertEqual(models.status, STATUS_UNSUPPORTED)


class StreamParsingTests(_ServerCase):
    routes = {
        "/v1/chat/completions": (
            200,
            "text/event-stream; charset=utf-8",
            'data: {"choices":[{"delta":{"content":"OK"}}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n",
        ),
        "/v1/responses": (
            200,
            "text/event-stream",
            'event: response.created\ndata: {"type":"response.created"}\n\n'
            'event: response.completed\ndata: {"type":"response.completed","response":{"status":"completed"}}\n\n',
        ),
    }

    def test_chat_stream_observation(self) -> None:
        transport = Transport(self.base_url, timeout=5)
        observation = transport.post_stream(
            transport.endpoints.chat_completions_url, {"stream": True}
        )
        self.assertTrue(observation.is_event_stream)
        self.assertEqual(observation.data_event_count, 3)
        self.assertTrue(observation.has_done_marker)

    def test_responses_stream_named_events_and_terminal(self) -> None:
        transport = Transport(self.base_url, timeout=5)
        observation = transport.post_stream(transport.endpoints.responses_url, {"stream": True})
        self.assertIn("response.created", observation.event_names)
        self.assertIn("response.completed", observation.event_names)
        self.assertTrue(observation.terminal_event_seen)


class NoWriteTests(_ServerCase):
    def test_no_write_skips_stateful(self) -> None:
        from upstream_capability_probe.probes import ProbeContext, run_probes

        transport = Transport(self.base_url, timeout=2)
        results = run_probes(
            ProbeContext(transport=transport, model="m"), allow_stateful=False
        )
        self.assertEqual(len(results), len(PROBE_SPECS))
        stateful = [r for r in results if r.kind == "stateful"]
        read_only = [r for r in results if r.kind == "read_only"]
        self.assertTrue(stateful)
        self.assertTrue(read_only)
        self.assertTrue(all(r.status == STATUS_SKIPPED for r in stateful))
        self.assertTrue(all(r.status != STATUS_SKIPPED for r in read_only))


class CliTests(unittest.TestCase):
    def test_help_exits_zero(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            main(["--help"])
        self.assertEqual(caught.exception.code, 0)

    def test_missing_key_env_is_refused(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            main(
                [
                    "--base-url",
                    "https://example.com",
                    "--api-key-env",
                    "UPSTREAM_CAPABILITY_PROBE_TEST_MISSING_ENV",
                ]
            )
        self.assertNotEqual(caught.exception.code, 0)

    def test_invalid_base_url_is_refused(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            main(["--base-url", "ftp://example.com"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
