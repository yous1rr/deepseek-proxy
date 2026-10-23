"""HTTP transport and structural observation for capability probes.

Design rules:

* Standard library only (``urllib``/``http``/``ssl``/``socket``).
* Only ``http`` and ``https`` base URLs are accepted; embedded credentials and
  non-HTTP schemes are rejected.
* Redirects are not followed, because forwarding an ``Authorization`` header
  to a different host would leak the credential.
* Response bodies are read with hard byte caps.  The transport returns only
  structural observations (status codes, content types, event names, field
  presence) and never exposes raw bodies to the profile serializer.
"""

from __future__ import annotations

import codecs
import json
import socket
import ssl
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_JSON_BODY_BYTES = 2 * 1024 * 1024
MAX_STREAM_BODY_BYTES = 256 * 1024
MAX_EVENT_NAMES = 24

HTTP_ERROR_CLASSES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "endpoint_not_found",
    405: "method_not_allowed",
    406: "not_acceptable",
    408: "request_timeout",
    409: "conflict",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "validation_failed",
    429: "rate_limited",
    500: "upstream_internal_error",
    501: "not_implemented",
    502: "bad_gateway",
    503: "service_unavailable",
    504: "gateway_timeout",
}


def classify_http_status(status: int) -> str:
    """Map an HTTP status code to a stable, non-secret error class."""
    return HTTP_ERROR_CLASSES.get(status, f"http_{status}")


class InvalidBaseUrl(ValueError):
    """Raised when --base-url is not a safe http(s) URL."""


@dataclass(frozen=True)
class EndpointSet:
    """Normalized endpoint builder derived from one base URL.

    ``display_url`` is safe to put in the profile: it omits query values and
    any userinfo (the latter is rejected outright).
    """

    scheme: str
    netloc: str
    api_root: str
    query: str

    @property
    def display_url(self) -> str:
        return urllib.parse.urlunsplit((self.scheme, self.netloc, self.api_root, "", ""))

    @property
    def has_query(self) -> bool:
        return bool(self.query)

    def endpoint(self, suffix: str) -> str:
        path = f"{self.api_root.rstrip('/')}/{suffix.lstrip('/')}"
        return urllib.parse.urlunsplit((self.scheme, self.netloc, path, self.query, ""))

    @property
    def chat_completions_url(self) -> str:
        return self.endpoint("chat/completions")

    @property
    def responses_url(self) -> str:
        return self.endpoint("responses")

    @property
    def models_url(self) -> str:
        return self.endpoint("models")


def normalize_base_url(raw: str) -> EndpointSet:
    """Normalize a user-supplied base URL into an API root.

    Accepted shapes (all produce an API root ending in ``/v1`` unless an
    explicit path prefix is supplied)::

        https://host
        https://host/
        https://host/v1
        https://host/v1/
        https://host/openai/v1
        https://host/v1/chat/completions   (full endpoint form)
        https://host/v1/responses          (full endpoint form)
        https://host/v1/models             (list endpoint form)

    Any other path is treated as the API root verbatim (this supports
    gateways mounted below a custom prefix).
    """
    if not isinstance(raw, str) or not raw.strip():
        raise InvalidBaseUrl("base URL is empty")
    candidate = raw.strip()
    parts = urllib.parse.urlsplit(candidate)
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        raise InvalidBaseUrl("base URL must use http or https")
    if not parts.hostname:
        raise InvalidBaseUrl("base URL has no host")
    if parts.username is not None or parts.password is not None:
        raise InvalidBaseUrl("base URL must not embed credentials; use --api-key-env")
    try:
        _ = parts.port
    except ValueError as exc:
        raise InvalidBaseUrl("base URL has an invalid port") from exc

    path = parts.path.rstrip("/")
    for suffix in ("/chat/completions", "/responses", "/models"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    if path.endswith("/v1"):
        api_root = path
    elif path in ("", "/"):
        api_root = "/v1"
    else:
        api_root = path
    return EndpointSet(
        scheme=scheme,
        netloc=parts.netloc,
        api_root=api_root,
        query=parts.query,
    )


@dataclass
class JsonHttpResult:
    """Structural result of a non-streaming HTTP request."""

    status: int | None
    content_type: str = ""
    error_class: str | None = None
    safe_message: str = ""
    json_value: Any = field(default=None, repr=False)
    body_bytes: int = 0
    body_truncated: bool = False
    is_json: bool = False
    # Fixed allowlisted identifier only; raw HTTP error bodies are discarded.
    unsupported_parameter: str | None = None

    @property
    def ok(self) -> bool:
        return (
            self.status is not None
            and 200 <= self.status < 300
            and self.error_class is None
            and not self.body_truncated
        )


@dataclass
class StreamObservation:
    """Structural, non-content observation of an SSE response."""

    status: int | None
    content_type: str = ""
    error_class: str | None = None
    safe_message: str = ""
    is_event_stream: bool = False
    event_count: int = 0
    data_event_count: int = 0
    json_event_count: int = 0
    event_names: list[str] = field(default_factory=list)
    has_done_marker: bool = False
    terminal_event_seen: bool = False
    error_event_seen: bool = False
    body_bytes: int = 0
    body_truncated: bool = False

    @property
    def ok(self) -> bool:
        return (
            self.status is not None
            and 200 <= self.status < 300
            and self.error_class is None
            and not self.body_truncated
        )


def _unsupported_parameter(body: Any) -> str | None:
    """Extract only explicit rejection of a known parameter, never error text."""
    if not isinstance(body, dict):
        return None
    error = body.get("error", body)
    if not isinstance(error, dict):
        return None
    code = error.get("code")
    if (
        error.get("param") == "max_output_tokens"
        and isinstance(code, str)
        and code in {"unsupported_parameter", "unknown_parameter", "unrecognized_parameter"}
    ):
        return "max_output_tokens"
    return None


def _safe_event_name(name: str) -> str | None:
    """Allow only short, identifier-like SSE event names into evidence."""
    candidate = name.strip()
    if not candidate or len(candidate) > 64:
        return None
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-")
    if any(ch not in allowed for ch in candidate):
        return None
    return candidate


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects so Authorization is never forwarded cross-origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None


class _SSEParser:
    """Incremental SSE parser that keeps only structural counters."""

    def __init__(self, max_names: int = MAX_EVENT_NAMES) -> None:
        self._buffer = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._event_name: str | None = None
        self._data_lines: list[str] = []
        self.event_count = 0
        self.data_event_count = 0
        self.json_event_count = 0
        self.event_names: list[str] = []
        self.has_done_marker = False
        self.terminal_event_seen = False
        self.error_event_seen = False
        self._max_names = max_names

    def feed(self, chunk: bytes) -> None:
        self._buffer += self._decoder.decode(chunk)
        while True:
            newline = self._buffer.find("\n")
            if newline < 0:
                return
            raw_line = self._buffer[:newline]
            self._buffer = self._buffer[newline + 1 :]
            if raw_line.endswith("\r"):
                raw_line = raw_line[:-1]
            self._consume_line(raw_line)

    def close(self) -> None:
        self._buffer += self._decoder.decode(b"", final=True)
        if self._buffer:
            raw_line = self._buffer[:-1] if self._buffer.endswith("\r") else self._buffer
            self._buffer = ""
            self._consume_line(raw_line)
        self._dispatch()

    def _consume_line(self, line: str) -> None:
        if line == "":
            self._dispatch()
            return
        if line.startswith(":"):
            return
        if line.startswith("event:"):
            self._event_name = line[6:].strip()
            return
        if line.startswith("data:"):
            value = line[5:]
            if value.startswith(" "):
                value = value[1:]
            self._data_lines.append(value)
            return
        # id:, retry:, and unknown fields are ignored.

    def _dispatch(self) -> None:
        if not self._data_lines and self._event_name is None:
            return
        self.event_count += 1
        data = "\n".join(self._data_lines).strip()
        self._data_lines = []
        event_name = self._event_name
        self._event_name = None

        if data:
            self.data_event_count += 1
            if data == "[DONE]":
                self.has_done_marker = True
            else:
                parsed: Any = None
                try:
                    parsed = json.loads(data)
                    self.json_event_count += 1
                except (ValueError, TypeError):
                    parsed = None
                declared_type = None
                if isinstance(parsed, dict):
                    declared_type = parsed.get("type")
                for type_value in (event_name, declared_type):
                    if isinstance(type_value, str):
                        if type_value in (
                            "response.completed",
                            "response.failed",
                            "response.incomplete",
                            "response.done",
                        ):
                            self.terminal_event_seen = True
                        if type_value in ("error", "response.error"):
                            self.error_event_seen = True
        if event_name:
            safe_name = _safe_event_name(event_name)
            if safe_name and safe_name not in self.event_names:
                if len(self.event_names) < self._max_names:
                    self.event_names.append(safe_name)


class Transport:
    """Tiny HTTP client with strict redaction and bounded reads."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        user_agent: str = "upstream-capability-probe/0.1.0",
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.endpoints = normalize_base_url(base_url)
        self._api_key = api_key or None
        self.timeout = float(timeout)
        self._opener = urllib.request.build_opener(_NoRedirect)
        self._user_agent = user_agent

    def _headers(self, accept: str) -> dict[str, str]:
        headers = {
            "Accept": accept,
            "User-Agent": self._user_agent,
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _open(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None,
        accept: str,
    ):
        data = None
        headers = self._headers(accept)
        if payload is not None:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        return self._opener.open(request, timeout=self.timeout)

    def _classify_exception(self, exc: BaseException) -> tuple[str, str]:
        if isinstance(exc, urllib.error.HTTPError):
            return classify_http_status(exc.code), f"HTTP {exc.code}"
        if isinstance(exc, (socket.timeout, TimeoutError)):
            return "timeout", "request timed out"
        if isinstance(exc, urllib.error.URLError):
            reason = exc.reason
            if isinstance(reason, (socket.timeout, TimeoutError)):
                return "timeout", "request timed out"
            if isinstance(reason, ssl.SSLError):
                return "tls_error", "TLS handshake or certificate error"
            if isinstance(reason, ConnectionRefusedError):
                return "connection_refused", "connection refused"
            if isinstance(reason, socket.gaierror):
                return "dns_error", "host name resolution failed"
            return "connection_error", "could not connect to upstream"
        if isinstance(exc, ssl.SSLError):
            return "tls_error", "TLS handshake or certificate error"
        return "transport_error", "transport failure"

    def get_json(self, url: str) -> JsonHttpResult:
        return self._read_json("GET", url, None)

    def post_json(self, url: str, payload: dict[str, Any]) -> JsonHttpResult:
        return self._read_json("POST", url, payload)

    def _read_json(
        self, method: str, url: str, payload: dict[str, Any] | None
    ) -> JsonHttpResult:
        try:
            response = self._open(method, url, payload, "application/json")
        except urllib.error.HTTPError as exc:
            status = exc.code
            if status in (400, 422):
                # Inspect only bounded JSON to recognize an explicit field
                # rejection. The parsed body is never retained in the result.
                response = exc
            else:
                try:
                    exc.close()
                except Exception:
                    pass
                return JsonHttpResult(
                    status=status,
                    error_class=classify_http_status(status),
                    safe_message=f"HTTP {status}",
                )
        except Exception as exc:  # noqa: BLE001 - normalized below
            error_class, message = self._classify_exception(exc)
            return JsonHttpResult(status=None, error_class=error_class, safe_message=message)

        try:
            with response:
                status = getattr(response, "status", None) or response.getcode()
                content_type = response.headers.get("Content-Type", "") or ""
                raw = response.read(MAX_JSON_BODY_BYTES + 1)
        except Exception as exc:  # noqa: BLE001
            error_class, message = self._classify_exception(exc)
            return JsonHttpResult(status=None, error_class=error_class, safe_message=message)

        truncated = len(raw) > MAX_JSON_BODY_BYTES
        body = raw[:MAX_JSON_BODY_BYTES]
        result = JsonHttpResult(
            status=status,
            content_type=content_type,
            body_bytes=len(body),
            body_truncated=truncated,
        )
        if truncated:
            result.error_class = "response_too_large"
            result.safe_message = "response exceeded the byte cap"
            return result
        if not (200 <= int(status) < 300):
            result.error_class = classify_http_status(int(status))
            result.safe_message = f"HTTP {status}"
            if status in (400, 422):
                try:
                    result.unsupported_parameter = _unsupported_parameter(
                        json.loads(body.decode("utf-8"))
                    )
                except (UnicodeDecodeError, ValueError):
                    pass
            return result
        try:
            result.json_value = json.loads(body.decode("utf-8"))
            result.is_json = True
        except (UnicodeDecodeError, ValueError):
            result.error_class = "invalid_json"
            result.safe_message = "response body was not valid JSON"
        return result

    def post_stream(self, url: str, payload: dict[str, Any]) -> StreamObservation:
        try:
            response = self._open("POST", url, payload, "text/event-stream")
        except urllib.error.HTTPError as exc:
            status = exc.code
            try:
                exc.close()
            except Exception:
                pass
            return StreamObservation(
                status=status,
                error_class=classify_http_status(status),
                safe_message=f"HTTP {status}",
            )
        except Exception as exc:  # noqa: BLE001
            error_class, message = self._classify_exception(exc)
            return StreamObservation(status=None, error_class=error_class, safe_message=message)

        parser = _SSEParser()
        total = 0
        truncated = False
        status = None
        content_type = ""
        try:
            with response:
                status = getattr(response, "status", None) or response.getcode()
                content_type = response.headers.get("Content-Type", "") or ""
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_STREAM_BODY_BYTES:
                        truncated = True
                        break
                    parser.feed(chunk)
        except Exception as exc:  # noqa: BLE001
            error_class, message = self._classify_exception(exc)
            return StreamObservation(
                status=status,
                content_type=content_type,
                error_class=error_class,
                safe_message=message,
                body_bytes=total,
            )
        parser.close()
        if truncated:
            error_class = "response_too_large"
            safe_message = "stream response exceeded the byte cap"
        elif status is not None and 200 <= int(status) < 300:
            error_class = None
            safe_message = ""
        elif status is None:
            error_class = "transport_error"
            safe_message = "no HTTP status observed"
        else:
            error_class = classify_http_status(int(status))
            safe_message = f"HTTP {status}"
        return StreamObservation(
            status=status,
            content_type=content_type,
            error_class=error_class,
            safe_message=safe_message,
            is_event_stream="text/event-stream" in content_type.lower(),
            event_count=parser.event_count,
            data_event_count=parser.data_event_count,
            json_event_count=parser.json_event_count,
            event_names=parser.event_names,
            has_done_marker=parser.has_done_marker,
            terminal_event_seen=parser.terminal_event_seen,
            error_event_seen=parser.error_event_seen,
            body_bytes=total,
            body_truncated=truncated,
        )
