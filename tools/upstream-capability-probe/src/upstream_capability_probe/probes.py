"""Probe definitions for OpenAI-style upstreams.

Every probe is intentionally tiny, synthetic, and structural.  A probe
observes status codes, content types, event names, and the presence of
documented JSON fields.  It never keeps response text, prompts, IDs, token
values, or headers.

Probe kinds:

``read_only``
    No model generation and no server-side state.  Safe to run unconditionally.

``stateful``
    Triggers model generation and may create provider-side history or cost.
    These probes are skipped with ``--no-write-probes``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from .models import (
    PROBE_KIND_READ_ONLY,
    PROBE_KIND_STATEFUL,
    STATUS_ERROR,
    STATUS_INCONCLUSIVE,
    STATUS_SKIPPED,
    STATUS_SUPPORTED,
    STATUS_UNSUPPORTED,
    ProbeResult,
)
from .transport import JsonHttpResult, StreamObservation, Transport

# A valid 1x1 PNG.  Synthetic and constant: this is not user data.
TINY_PNG_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)

CHAT_PROMPT = "Reply with the single word OK."
TOOL_PROMPT = "Call the probe_tool with value set to 42. Do not answer in text."
VISION_PROMPT = "This is a synthetic 1x1 red PNG pixel. Reply with the single word red."
OUTPUT_LIMIT = 16
OUTPUT_LIMIT_PROMPT = (
    "Write every integer from 1 through 200 separated by spaces. "
    "Do not summarize or stop early."
)

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "probe_tool",
        "description": "Synthetic capability probe tool. Return the supplied value.",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {"type": "string", "description": "Value to echo back."}
            },
            "required": ["value"],
            "additionalProperties": False,
        },
    },
}

RESPONSES_TOOL_DEFINITION = {
    "type": "function",
    "name": "probe_tool",
    "description": "Synthetic capability probe tool. Return the supplied value.",
    "parameters": {
        "type": "object",
        "properties": {"value": {"type": "string", "description": "Value to echo back."}},
        "required": ["value"],
        "additionalProperties": False,
    },
}

_TOP_LEVEL_FIELDS = {
    "id",
    "object",
    "created",
    "created_at",
    "model",
    "status",
    "choices",
    "output",
    "output_text",
    "usage",
    "error",
    "data",
    "type",
    "incomplete_details",
    "parallel_tool_calls",
    "tool_choice",
    "temperature",
    "top_p",
    "max_output_tokens",
    "previous_response_id",
    "instructions",
    "metadata",
}

_USAGE_FIELDS = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "input_tokens",
    "output_tokens",
    "prompt_tokens_details",
    "completion_tokens_details",
    "input_tokens_details",
    "output_tokens_details",
}

_TOKEN_COUNT_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "input_tokens",
    "output_tokens",
)

_FINISH_REASONS = {"stop", "length", "tool_calls", "function_call", "content_filter"}
_RESPONSE_OBJECTS = {"response", "error", "chat.completion", "list", "model"}
_RESPONSE_STATUSES = {
    "completed",
    "failed",
    "incomplete",
    "in_progress",
    "queued",
    "cancelled",
    "canceled",
}


@dataclass
class ProbeContext:
    """Everything a probe needs; no credentials beyond the transport."""

    transport: Transport
    model: str


# ---------------------------------------------------------------------------
# Structural evidence helpers (content-free by construction)
# ---------------------------------------------------------------------------


def _safe_identifier(value: Any, max_length: int = 64) -> str | None:
    """Accept only short token-like identifiers (no URLs, paths, or free text)."""
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate or len(candidate) > max_length:
        return None
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-")
    if any(ch not in allowed for ch in candidate):
        return None
    return candidate
    return candidate


def _safe_content_type(value: str) -> str:
    kind = (value or "").split(";", 1)[0].strip().lower()
    if not kind:
        return "absent"
    base = kind.split("/", 1)
    if len(base) != 2:
        return "other"
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789.+_-")
    if all(ch in allowed for ch in base[0]) and all(ch in allowed for ch in base[1]):
        return kind
    return "other"


def _value_kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        return "string_nonempty" if value else "string_empty"
    if isinstance(value, list):
        return "array_nonempty" if value else "array_empty"
    if isinstance(value, dict):
        return "object"
    return "other"


def _finish_reason(value: Any) -> str | None:
    if isinstance(value, str) and value in _FINISH_REASONS:
        return value
    if isinstance(value, str) and value:
        return "other"
    return None


def _object_value(value: Any) -> str:
    if isinstance(value, str) and value in _RESPONSE_OBJECTS:
        return value
    if isinstance(value, str) and value:
        return "other"
    return "absent"


def _status_value(value: Any) -> str:
    if isinstance(value, str) and value in _RESPONSE_STATUSES:
        return value
    if isinstance(value, str) and value:
        return "other"
    return "absent"


def _top_level_fields(body: Any) -> list[str]:
    if not isinstance(body, dict):
        return []
    return sorted(key for key in body if key in _TOP_LEVEL_FIELDS)


def _usage_evidence(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        return {"usage_present": False, "usage_fields": []}
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return {"usage_present": False, "usage_fields": []}
    fields = sorted(key for key in usage if key in _USAGE_FIELDS)
    token_fields = [
        key
        for key in _TOKEN_COUNT_FIELDS
        if isinstance(usage.get(key), int) and not isinstance(usage.get(key), bool)
    ]
    return {
        "usage_present": True,
        "usage_fields": fields,
        "usage_token_count_fields": token_fields,
        "usage_details_present": any(
            isinstance(usage.get(key), dict)
            for key in (
                "prompt_tokens_details",
                "completion_tokens_details",
                "input_tokens_details",
                "output_tokens_details",
            )
        ),
    }


def _chat_message(body: Any) -> dict[str, Any] | None:
    """Return the first message dict from a Chat Completions response."""
    if not isinstance(body, dict):
        return None
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    return message if isinstance(message, dict) else None


def _chat_choice_count(body: Any) -> int | None:
    if not isinstance(body, dict):
        return None
    choices = body.get("choices")
    return len(choices) if isinstance(choices, list) else None


def _first_choice(body: Any) -> dict[str, Any] | None:
    if not isinstance(body, dict):
        return None
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    return first if isinstance(first, dict) else None


def _chat_tool_calls(message: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(message, dict):
        return []
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return []
    return [call for call in calls if isinstance(call, dict)]


def _function_call_from_chat(message: dict[str, Any] | None) -> dict[str, Any] | None:
    """Extract the first chat tool call for an in-memory continuation.

    This value is used only to build the next synthetic request and is never
    serialized into the profile.
    """
    calls = _chat_tool_calls(message)
    if not calls:
        return None
    first = calls[0]
    function = first.get("function")
    if not isinstance(function, dict):
        return None
    return {
        "id": first.get("id") if isinstance(first.get("id"), str) else "",
        "name": function.get("name") if isinstance(function.get("name"), str) else "",
        "arguments": function.get("arguments")
        if isinstance(function.get("arguments"), str)
        else "",
    }


def _function_call_from_responses(body: Any) -> dict[str, Any] | None:
    """Extract the first Responses function_call item for in-memory reuse."""
    if not isinstance(body, dict):
        return None
    output = body.get("output")
    if not isinstance(output, list):
        return None
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "function_call":
            continue
        return {
            "id": item.get("id") if isinstance(item.get("id"), str) else "",
            "call_id": item.get("call_id") if isinstance(item.get("call_id"), str) else "",
            "name": item.get("name") if isinstance(item.get("name"), str) else "",
            "arguments": item.get("arguments")
            if isinstance(item.get("arguments"), str)
            else "",
        }
    return None


def _output_item_types(body: Any) -> list[str]:
    if not isinstance(body, dict):
        return []
    output = body.get("output")
    if not isinstance(output, list):
        return []
    kinds: list[str] = []
    for item in output:
        if isinstance(item, dict):
            kind = _safe_identifier(item.get("type"), max_length=32)
            if kind and kind not in kinds and len(kinds) < 8:
                kinds.append(kind)
    return kinds


def _assistant_message_evidence(message: dict[str, Any] | None) -> dict[str, Any]:
    if message is None:
        return {
            "assistant_message_present": False,
            "content_kind": "absent",
            "tool_call_count": 0,
        }
    return {
        "assistant_message_present": True,
        "content_kind": _value_kind(message.get("content")),
        "tool_call_count": len(_chat_tool_calls(message)),
    }


# ---------------------------------------------------------------------------
# Result construction and HTTP failure classification
# ---------------------------------------------------------------------------


_SUPPORTED_BY_STATUS = {
    STATUS_SUPPORTED: True,
    STATUS_UNSUPPORTED: False,
    STATUS_INCONCLUSIVE: None,
    STATUS_SKIPPED: None,
    STATUS_ERROR: None,
}


def _result(
    probe_id: str,
    title: str,
    capability: str,
    kind: str,
    status: str,
    reason: str,
    evidence: dict[str, Any] | None = None,
    error_class: str | None = None,
    http_status: int | None = None,
) -> ProbeResult:
    return ProbeResult(
        id=probe_id,
        title=title,
        capability=capability,
        kind=kind,
        status=status,
        supported=_SUPPORTED_BY_STATUS[status],
        reason=reason,
        evidence=evidence or {},
        error_class=error_class,
        http_status=http_status,
    )


def _failure(
    probe_id: str,
    title: str,
    capability: str,
    kind: str,
    *,
    http_status: int | None,
    error_class: str | None,
    evidence: dict[str, Any] | None = None,
    bad_request_unsupported: bool = False,
    bad_request_reason: str = "request_rejected",
) -> ProbeResult:
    """Turn an HTTP/transport failure into a safe probe result.

    No error body reaches this helper. The transport can extract an
    allowlisted unsupported-parameter marker, but discards all error text.
    """
    if http_status in (404, 405, 501):
        status = STATUS_UNSUPPORTED
        reason = {
            404: "endpoint_not_found",
            405: "method_not_allowed",
            501: "not_implemented",
        }[http_status]
    elif http_status in (401, 403):
        status = STATUS_ERROR
        reason = "authentication_failed" if http_status == 401 else "forbidden"
    elif error_class == "timeout":
        status = STATUS_INCONCLUSIVE
        reason = "timeout"
    elif error_class == "invalid_json":
        status = STATUS_INCONCLUSIVE
        reason = "invalid_json_response"
    elif error_class == "response_too_large":
        status = STATUS_INCONCLUSIVE
        reason = "response_too_large"
    elif http_status is not None and 300 <= http_status < 400:
        status = STATUS_INCONCLUSIVE
        reason = "unexpected_redirect"
    elif http_status is not None and 400 <= http_status < 500:
        if bad_request_unsupported and http_status in (400, 406, 415, 422):
            status = STATUS_UNSUPPORTED
            reason = bad_request_reason
        elif http_status == 429:
            status = STATUS_INCONCLUSIVE
            reason = "rate_limited"
        else:
            status = STATUS_INCONCLUSIVE
            reason = "request_rejected"
    elif http_status is not None and http_status >= 500:
        status = STATUS_INCONCLUSIVE
        reason = "upstream_server_error"
    else:
        status = STATUS_ERROR
        reason = "transport_failure"
    return _result(
        probe_id,
        title,
        capability,
        kind,
        status,
        reason,
        evidence=evidence,
        error_class=error_class,
        http_status=http_status,
    )


def _failure_from_json(
    probe_id: str,
    title: str,
    capability: str,
    kind: str,
    result: JsonHttpResult,
    **kwargs: Any,
) -> ProbeResult:
    return _failure(
        probe_id,
        title,
        capability,
        kind,
        http_status=result.status,
        error_class=result.error_class,
        **kwargs,
    )


def _failure_from_stream(
    probe_id: str,
    title: str,
    capability: str,
    kind: str,
    observation: StreamObservation,
    **kwargs: Any,
) -> ProbeResult:
    return _failure(
        probe_id,
        title,
        capability,
        kind,
        http_status=observation.status,
        error_class=observation.error_class,
        **kwargs,
    )


def _stream_evidence(observation: StreamObservation) -> dict[str, Any]:
    return {
        "is_event_stream": observation.is_event_stream,
        "event_count": observation.event_count,
        "data_event_count": observation.data_event_count,
        "json_event_count": observation.json_event_count,
        "sse_done_seen": observation.has_done_marker,
        "terminal_event_seen": observation.terminal_event_seen,
        "error_event_seen": observation.error_event_seen,
        "event_names": observation.event_names,
        "body_bytes": observation.body_bytes,
    }


# ---------------------------------------------------------------------------
# Read-only probes
# ---------------------------------------------------------------------------


def probe_models_list(ctx: ProbeContext) -> ProbeResult:
    probe_id = "models.list"
    title = "GET /models"
    capability = "models_list"
    kind = PROBE_KIND_READ_ONLY
    result = ctx.transport.get_json(ctx.transport.endpoints.models_url)
    if result.error_class is not None:
        return _failure_from_json(probe_id, title, capability, kind, result)
    body = result.json_value
    data = body.get("data") if isinstance(body, dict) else None
    data_is_list = isinstance(data, list)
    evidence = {
        "content_type": _safe_content_type(result.content_type),
        "object_value": _object_value(body.get("object")) if isinstance(body, dict) else "absent",
        "data_is_list": data_is_list,
        "model_count": len(data) if data_is_list else 0,
        "top_level_fields": _top_level_fields(body),
    }
    if data_is_list:
        return _result(
            probe_id, title, capability, kind, STATUS_SUPPORTED, "models_endpoint_responded",
            evidence=evidence, http_status=result.status,
        )
    return _result(
        probe_id, title, capability, kind, STATUS_UNSUPPORTED, "unrecognized_models_shape",
        evidence=evidence, http_status=result.status,
    )


# ---------------------------------------------------------------------------
# Chat Completions probes
# ---------------------------------------------------------------------------


def _chat_payload(ctx: ProbeContext, *, stream: bool) -> dict[str, Any]:
    return {
        "model": ctx.model,
        "messages": [{"role": "user", "content": CHAT_PROMPT}],
        "stream": stream,
        "max_tokens": 16,
    }


def probe_chat_non_stream(ctx: ProbeContext) -> ProbeResult:
    probe_id = "chat.completions.non_stream"
    title = "Chat Completions non-stream"
    capability = "chat_completions_non_stream"
    kind = PROBE_KIND_STATEFUL
    result = ctx.transport.post_json(
        ctx.transport.endpoints.chat_completions_url, _chat_payload(ctx, stream=False)
    )
    if result.error_class is not None:
        return _failure_from_json(probe_id, title, capability, kind, result)
    body = result.json_value
    message = _chat_message(body)
    choices_count = _chat_choice_count(body)
    evidence = {
        "content_type": _safe_content_type(result.content_type),
        "top_level_fields": _top_level_fields(body),
        "choices_count": choices_count if choices_count is not None else 0,
        **_assistant_message_evidence(message),
        **_usage_evidence(body),
    }
    if not isinstance(body, dict):
        return _result(
            probe_id, title, capability, kind, STATUS_UNSUPPORTED, "unrecognized_chat_shape",
            evidence=evidence, http_status=result.status,
        )
    if isinstance(body.get("error"), dict):
        return _result(
            probe_id, title, capability, kind, STATUS_UNSUPPORTED, "error_payload",
            evidence=evidence, http_status=result.status,
        )
    if choices_count is None:
        return _result(
            probe_id, title, capability, kind, STATUS_UNSUPPORTED, "unrecognized_chat_shape",
            evidence=evidence, http_status=result.status,
        )
    if choices_count == 0:
        return _result(
            probe_id, title, capability, kind, STATUS_UNSUPPORTED, "empty_choices",
            evidence=evidence, http_status=result.status,
        )
    if message is None:
        return _result(
            probe_id, title, capability, kind, STATUS_UNSUPPORTED, "missing_message",
            evidence=evidence, http_status=result.status,
        )
    first_choice = _first_choice(body)
    evidence["finish_reason"] = _finish_reason(
        first_choice.get("finish_reason") if first_choice else None
    )
    return _result(
        probe_id, title, capability, kind, STATUS_SUPPORTED, "chat_completion_observed",
        evidence=evidence, http_status=result.status,
    )


def probe_chat_stream(ctx: ProbeContext) -> ProbeResult:
    probe_id = "chat.completions.stream"
    title = "Chat Completions stream=true"
    capability = "chat_completions_stream"
    kind = PROBE_KIND_STATEFUL
    observation = ctx.transport.post_stream(
        ctx.transport.endpoints.chat_completions_url, _chat_payload(ctx, stream=True)
    )
    evidence = _stream_evidence(observation)
    if observation.error_class is not None:
        return _failure_from_stream(
            probe_id, title, capability, kind, observation, evidence=evidence
        )
    if not observation.is_event_stream:
        return _result(
            probe_id, title, capability, kind, STATUS_UNSUPPORTED, "non_sse_response",
            evidence=evidence, http_status=observation.status,
        )
    if observation.data_event_count == 0:
        return _result(
            probe_id, title, capability, kind, STATUS_UNSUPPORTED, "sse_without_data_events",
            evidence=evidence, http_status=observation.status,
        )
    if (
        observation.error_event_seen
        and observation.json_event_count <= 1
        and not observation.has_done_marker
        and not observation.terminal_event_seen
    ):
        return _result(
            probe_id, title, capability, kind, STATUS_UNSUPPORTED, "error_event_only",
            evidence=evidence, http_status=observation.status,
        )
    return _result(
        probe_id, title, capability, kind, STATUS_SUPPORTED, "sse_stream_observed",
        evidence=evidence, http_status=observation.status,
    )


def _chat_tool_payload(ctx: ProbeContext, messages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": ctx.model,
        "messages": messages,
        "tools": [TOOL_DEFINITION],
        "tool_choice": {"type": "function", "function": {"name": "probe_tool"}},
        "stream": False,
        "max_tokens": 64,
    }


def _tool_call_evidence(
    message: dict[str, Any] | None,
    *,
    body: Any = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    calls = _chat_tool_calls(message)
    first = calls[0] if calls else None
    function = first.get("function") if first and isinstance(first.get("function"), dict) else None
    name = function.get("name") if isinstance(function, dict) else None
    arguments = function.get("arguments") if isinstance(function, dict) else None
    arguments_json_valid = False
    arguments_is_object = False
    argument_keys_match = False
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except (ValueError, TypeError):
            parsed = None
        if parsed is not None:
            arguments_json_valid = True
            if isinstance(parsed, dict):
                arguments_is_object = True
                argument_keys_match = set(parsed.keys()) == {"value"}
    evidence = {
        "tool_calls_present": bool(calls),
        "tool_call_count": len(calls),
        "tool_name_matches": isinstance(name, str) and name == "probe_tool",
        "arguments_json_valid": arguments_json_valid,
        "arguments_is_object": arguments_is_object,
        "argument_keys_match": argument_keys_match,
        "finish_reason": _finish_reason(
            (_first_choice(body) or {}).get("finish_reason")
        ),
    }
    call = _function_call_from_chat(message)
    return evidence, call


def probe_chat_function_calling(ctx: ProbeContext) -> ProbeResult:
    probe_id = "chat.completions.function_calling"
    title = "Chat Completions function calling"
    capability = "chat_completions_function_calling"
    kind = PROBE_KIND_STATEFUL
    payload = _chat_tool_payload(
        ctx,
        [
            {"role": "user", "content": TOOL_PROMPT},
        ],
    )
    result = ctx.transport.post_json(ctx.transport.endpoints.chat_completions_url, payload)
    if result.error_class is not None:
        return _failure_from_json(probe_id, title, capability, kind, result)
    body = result.json_value
    message = _chat_message(body)
    evidence, call = _tool_call_evidence(message, body=body)
    evidence["content_type"] = _safe_content_type(result.content_type)
    if message is None:
        return _result(
            probe_id, title, capability, kind, STATUS_UNSUPPORTED, "missing_message",
            evidence=evidence, http_status=result.status,
        )
    if not evidence["tool_calls_present"]:
        return _result(
            probe_id, title, capability, kind, STATUS_UNSUPPORTED, "no_tool_call_observed",
            evidence=evidence, http_status=result.status,
        )
    if not evidence["tool_name_matches"]:
        return _result(
            probe_id, title, capability, kind, STATUS_INCONCLUSIVE, "tool_name_mismatch",
            evidence=evidence, http_status=result.status,
        )
    if not evidence["arguments_json_valid"] or not evidence["arguments_is_object"]:
        return _result(
            probe_id, title, capability, kind, STATUS_UNSUPPORTED, "invalid_tool_arguments",
            evidence=evidence, http_status=result.status,
        )
    return _result(
        probe_id, title, capability, kind, STATUS_SUPPORTED, "tool_call_observed",
        evidence=evidence, http_status=result.status,
    )


def probe_chat_tool_result_continuation(ctx: ProbeContext) -> ProbeResult:
    probe_id = "chat.completions.tool_result_continuation"
    title = "Chat Completions tool-result continuation"
    capability = "chat_completions_tool_result_continuation"
    kind = PROBE_KIND_STATEFUL
    first_payload = _chat_tool_payload(ctx, [{"role": "user", "content": TOOL_PROMPT}])
    first = ctx.transport.post_json(
        ctx.transport.endpoints.chat_completions_url, first_payload
    )
    if first.error_class is not None:
        return _failure_from_json(probe_id, title, capability, kind, first)
    first_message = _chat_message(first.json_value)
    first_evidence, call = _tool_call_evidence(first_message, body=first.json_value)
    base_evidence = {
        "initial_status": first.status,
        "initial_tool_call_present": bool(call),
        "initial_tool_name_matches": first_evidence["tool_name_matches"],
    }
    if call is None:
        return _result(
            probe_id, title, capability, kind, STATUS_UNSUPPORTED,
            "no_tool_call_to_continue", evidence=base_evidence, http_status=first.status,
        )
    if not call.get("id"):
        return _result(
            probe_id, title, capability, kind, STATUS_INCONCLUSIVE,
            "tool_call_id_missing", evidence=base_evidence, http_status=first.status,
        )
    assistant_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call["id"],
                "type": "function",
                "function": {"name": call["name"], "arguments": call["arguments"]},
            }
        ],
    }
    second_messages = [
        {"role": "user", "content": TOOL_PROMPT},
        assistant_message,
        {"role": "tool", "tool_call_id": call["id"], "content": '{"value": "84"}'},
    ]
    second_payload = {
        "model": ctx.model,
        "messages": second_messages,
        "tools": [TOOL_DEFINITION],
        "tool_choice": "auto",
        "stream": False,
        "max_tokens": 32,
    }
    second = ctx.transport.post_json(
        ctx.transport.endpoints.chat_completions_url, second_payload
    )
    if second.error_class is not None:
        failure = _failure_from_json(
            probe_id,
            title,
            capability,
            kind,
            second,
            evidence=base_evidence,
            bad_request_unsupported=True,
            bad_request_reason="tool_result_continuation_rejected",
        )
        return failure
    second_body = second.json_value
    second_message = _chat_message(second_body)
    evidence = {
        **base_evidence,
        "continuation_status": second.status,
        **_assistant_message_evidence(second_message),
        "continuation_finish_reason": _finish_reason(
            (_first_choice(second_body) or {}).get("finish_reason")
        ),
    }
    if second_message is None:
        return _result(
            probe_id, title, capability, kind, STATUS_INCONCLUSIVE,
            "continuation_missing_message", evidence=evidence, http_status=second.status,
        )
    if (
        not evidence["content_kind"].startswith("string_nonempty")
        and not evidence["content_kind"].startswith("array_nonempty")
        and evidence["tool_call_count"] == 0
    ):
        return _result(
            probe_id, title, capability, kind, STATUS_INCONCLUSIVE,
            "empty_continuation_message", evidence=evidence, http_status=second.status,
        )
    return _result(
        probe_id, title, capability, kind, STATUS_SUPPORTED,
        "tool_result_continuation_observed", evidence=evidence, http_status=second.status,
    )


def probe_chat_vision_data_uri(ctx: ProbeContext) -> ProbeResult:
    probe_id = "chat.completions.vision_data_uri"
    title = "Chat Completions vision data URI"
    capability = "chat_completions_vision_data_uri"
    kind = PROBE_KIND_STATEFUL
    payload = {
        "model": ctx.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": TINY_PNG_DATA_URI}},
                ],
            }
        ],
        "stream": False,
        "max_tokens": 16,
    }
    result = ctx.transport.post_json(ctx.transport.endpoints.chat_completions_url, payload)
    if result.error_class is not None:
        return _failure_from_json(
            probe_id,
            title,
            capability,
            kind,
            result,
            bad_request_unsupported=True,
            bad_request_reason="image_input_rejected",
        )
    body = result.json_value
    message = _chat_message(body)
    evidence = {
        "image_mime": "image/png",
        "content_parts": ["text", "image_url"],
        "top_level_fields": _top_level_fields(body),
        **_assistant_message_evidence(message),
    }
    if message is None:
        return _result(
            probe_id, title, capability, kind, STATUS_UNSUPPORTED, "missing_message",
            evidence=evidence, http_status=result.status,
        )
    return _result(
        probe_id, title, capability, kind, STATUS_SUPPORTED, "image_input_accepted",
        evidence=evidence, http_status=result.status,
    )


# ---------------------------------------------------------------------------
# Responses probes
# ---------------------------------------------------------------------------


def probe_responses_non_stream(ctx: ProbeContext) -> ProbeResult:
    probe_id = "responses.non_stream"
    title = "Responses non-stream"
    capability = "responses_non_stream"
    kind = PROBE_KIND_STATEFUL
    payload = {
        "model": ctx.model,
        "input": CHAT_PROMPT,
        "stream": False,
    }
    result = ctx.transport.post_json(ctx.transport.endpoints.responses_url, payload)
    if result.error_class is not None:
        return _failure_from_json(probe_id, title, capability, kind, result)
    body = result.json_value
    object_value = _object_value(body.get("object")) if isinstance(body, dict) else "absent"
    output = body.get("output") if isinstance(body, dict) else None
    evidence = {
        "content_type": _safe_content_type(result.content_type),
        "object_value": object_value,
        "status_value": _status_value(body.get("status")) if isinstance(body, dict) else "absent",
        "output_is_list": isinstance(output, list),
        "output_item_count": len(output) if isinstance(output, list) else 0,
        "output_item_types": _output_item_types(body),
        "top_level_fields": _top_level_fields(body),
        **_usage_evidence(body),
    }
    if not isinstance(body, dict):
        return _result(
            probe_id, title, capability, kind, STATUS_UNSUPPORTED,
            "unrecognized_response_shape", evidence=evidence, http_status=result.status,
        )
    if object_value == "error" or isinstance(body.get("error"), dict):
        return _result(
            probe_id, title, capability, kind, STATUS_UNSUPPORTED, "error_payload",
            evidence=evidence, http_status=result.status,
        )
    if object_value == "chat.completion":
        return _result(
            probe_id, title, capability, kind, STATUS_UNSUPPORTED,
            "chat_completion_shape_returned", evidence=evidence, http_status=result.status,
        )
    if object_value == "response":
        if isinstance(output, list) and output:
            return _result(
                probe_id, title, capability, kind, STATUS_SUPPORTED,
                "response_object_observed", evidence=evidence, http_status=result.status,
            )
        if isinstance(output, list) and not output:
            return _result(
                probe_id, title, capability, kind, STATUS_INCONCLUSIVE, "empty_output",
                evidence=evidence, http_status=result.status,
            )
    return _result(
        probe_id, title, capability, kind, STATUS_UNSUPPORTED,
        "unrecognized_response_shape", evidence=evidence, http_status=result.status,
    )


def probe_responses_stream(ctx: ProbeContext) -> ProbeResult:
    probe_id = "responses.stream"
    title = "Responses stream=true"
    capability = "responses_stream"
    kind = PROBE_KIND_STATEFUL
    payload = {
        "model": ctx.model,
        "input": CHAT_PROMPT,
        "stream": True,
    }
    observation = ctx.transport.post_stream(
        ctx.transport.endpoints.responses_url, payload
    )
    evidence = _stream_evidence(observation)
    if observation.error_class is not None:
        return _failure_from_stream(
            probe_id, title, capability, kind, observation, evidence=evidence
        )
    if not observation.is_event_stream:
        return _result(
            probe_id, title, capability, kind, STATUS_UNSUPPORTED, "non_sse_response",
            evidence=evidence, http_status=observation.status,
        )
    if observation.data_event_count == 0:
        return _result(
            probe_id, title, capability, kind, STATUS_UNSUPPORTED, "sse_without_data_events",
            evidence=evidence, http_status=observation.status,
        )
    if (
        observation.error_event_seen
        and observation.json_event_count <= 1
        and not observation.has_done_marker
        and not observation.terminal_event_seen
    ):
        return _result(
            probe_id, title, capability, kind, STATUS_UNSUPPORTED, "error_event_only",
            evidence=evidence, http_status=observation.status,
        )
    return _result(
        probe_id, title, capability, kind, STATUS_SUPPORTED, "sse_stream_observed",
        evidence=evidence, http_status=observation.status,
    )


def probe_responses_max_output_tokens(ctx: ProbeContext) -> ProbeResult:
    """Verify a limit was enforced, rather than merely accepted or echoed."""
    probe_id = "responses.max_output_tokens"
    title = "Responses max_output_tokens enforcement"
    capability = "responses_max_output_tokens"
    kind = PROBE_KIND_STATEFUL
    payload = {
        "model": ctx.model,
        "input": OUTPUT_LIMIT_PROMPT,
        "stream": False,
        "max_output_tokens": OUTPUT_LIMIT,
    }
    result = ctx.transport.post_json(ctx.transport.endpoints.responses_url, payload)
    rejected = (
        result.status in (400, 422)
        and result.unsupported_parameter == "max_output_tokens"
    )
    evidence = {
        "requested_max_output_tokens": OUTPUT_LIMIT,
        "parameter_explicitly_rejected": rejected,
    }
    if rejected:
        return _result(
            probe_id, title, capability, kind, STATUS_UNSUPPORTED,
            "max_output_tokens_rejected", evidence=evidence,
            error_class=result.error_class, http_status=result.status,
        )
    if not result.ok:
        return _failure_from_json(
            probe_id, title, capability, kind, result, evidence=evidence,
        )
    body = result.json_value
    response = body if isinstance(body, dict) else {}
    usage = response.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    total_tokens = usage.get("total_tokens")
    counts_valid = all(
        isinstance(count, int) and not isinstance(count, bool) and count >= 0
        for count in (input_tokens, output_tokens, total_tokens)
    )
    usage_consistent = (
        counts_valid
        and output_tokens > 0
        and total_tokens == input_tokens + output_tokens
    )
    details = usage.get("output_tokens_details")
    if details is not None:
        reasoning_tokens = details.get("reasoning_tokens") if isinstance(details, dict) else None
        usage_consistent = (
            usage_consistent
            and isinstance(reasoning_tokens, int)
            and not isinstance(reasoning_tokens, bool)
            and 0 <= reasoning_tokens <= output_tokens
        )
    incomplete = response.get("incomplete_details")
    limit_reason = (
        isinstance(incomplete, dict)
        and incomplete.get("reason") == "max_output_tokens"
    )
    object_value = _object_value(response.get("object"))
    status_value = _status_value(response.get("status"))
    response_shape = (
        object_value == "response"
        and isinstance(response.get("output"), list)
        and response.get("error") is None
    )
    within_limit = usage_consistent and output_tokens <= OUTPUT_LIMIT
    echo = response.get("max_output_tokens")
    evidence.update({
        "object_value": object_value,
        "status_value": status_value,
        "output_is_list": isinstance(response.get("output"), list),
        "parameter_echo_matches": type(echo) is int and echo == OUTPUT_LIMIT,
        "incomplete_reason_is_max_output_tokens": limit_reason,
        "usage_consistent": usage_consistent,
        "output_tokens_within_limit": within_limit,
    })
    if response_shape and usage_consistent and not within_limit:
        status, reason = STATUS_UNSUPPORTED, "output_limit_exceeded"
    elif response_shape and status_value == "incomplete" and limit_reason and within_limit:
        status, reason = STATUS_SUPPORTED, "output_limit_enforced"
    else:
        status, reason = STATUS_INCONCLUSIVE, "output_limit_not_verified"
    return _result(
        probe_id, title, capability, kind, status, reason,
        evidence=evidence, http_status=result.status,
    )


def probe_responses_function_calling(ctx: ProbeContext) -> ProbeResult:
    probe_id = "responses.function_calling"
    title = "Responses function calling"
    capability = "responses_function_calling"
    kind = PROBE_KIND_STATEFUL
    payload = {
        "model": ctx.model,
        "input": TOOL_PROMPT,
        "tools": [RESPONSES_TOOL_DEFINITION],
        "tool_choice": {"type": "function", "name": "probe_tool"},
        "stream": False,
    }
    result = ctx.transport.post_json(ctx.transport.endpoints.responses_url, payload)
    if result.error_class is not None:
        return _failure_from_json(probe_id, title, capability, kind, result)
    body = result.json_value
    call = _function_call_from_responses(body)
    evidence = {
        "content_type": _safe_content_type(result.content_type),
        "object_value": _object_value(body.get("object")) if isinstance(body, dict) else "absent",
        "output_is_list": isinstance(body.get("output"), list) if isinstance(body, dict) else False,
        "output_item_types": _output_item_types(body),
        "function_call_present": call is not None,
        "tool_name_matches": bool(call and call.get("name") == "probe_tool"),
    }
    arguments_json_valid = False
    arguments_is_object = False
    if call and isinstance(call.get("arguments"), str):
        try:
            parsed = json.loads(call["arguments"])
        except (ValueError, TypeError):
            parsed = None
        if parsed is not None:
            arguments_json_valid = True
            arguments_is_object = isinstance(parsed, dict)
    evidence["arguments_json_valid"] = arguments_json_valid
    evidence["arguments_is_object"] = arguments_is_object
    if not isinstance(body, dict):
        return _result(
            probe_id, title, capability, kind, STATUS_UNSUPPORTED,
            "unrecognized_response_shape", evidence=evidence, http_status=result.status,
        )
    if call is None:
        return _result(
            probe_id, title, capability, kind, STATUS_UNSUPPORTED,
            "no_function_call_observed", evidence=evidence, http_status=result.status,
        )
    if not evidence["tool_name_matches"]:
        return _result(
            probe_id, title, capability, kind, STATUS_INCONCLUSIVE, "tool_name_mismatch",
            evidence=evidence, http_status=result.status,
        )
    if not arguments_json_valid or not arguments_is_object:
        return _result(
            probe_id, title, capability, kind, STATUS_UNSUPPORTED, "invalid_tool_arguments",
            evidence=evidence, http_status=result.status,
        )
    return _result(
        probe_id, title, capability, kind, STATUS_SUPPORTED, "function_call_observed",
        evidence=evidence, http_status=result.status,
    )


def probe_responses_tool_result_continuation(ctx: ProbeContext) -> ProbeResult:
    probe_id = "responses.tool_result_continuation"
    title = "Responses tool-result continuation"
    capability = "responses_tool_result_continuation"
    kind = PROBE_KIND_STATEFUL
    first_payload = {
        "model": ctx.model,
        "input": TOOL_PROMPT,
        "tools": [RESPONSES_TOOL_DEFINITION],
        "tool_choice": {"type": "function", "name": "probe_tool"},
        "stream": False,
    }
    first = ctx.transport.post_json(ctx.transport.endpoints.responses_url, first_payload)
    if first.error_class is not None:
        return _failure_from_json(probe_id, title, capability, kind, first)
    call = _function_call_from_responses(first.json_value)
    base_evidence = {
        "initial_status": first.status,
        "initial_function_call_present": call is not None,
    }
    if call is None or not call.get("call_id"):
        return _result(
            probe_id, title, capability, kind, STATUS_UNSUPPORTED,
            "no_function_call_to_continue", evidence=base_evidence, http_status=first.status,
        )
    second_payload = {
        "model": ctx.model,
        "input": [
            {"role": "user", "content": TOOL_PROMPT},
            {
                "type": "function_call",
                "call_id": call["call_id"],
                "name": call["name"],
                "arguments": call["arguments"],
            },
            {
                "type": "function_call_output",
                "call_id": call["call_id"],
                "output": '{"value": "84"}',
            },
        ],
        "tools": [RESPONSES_TOOL_DEFINITION],
        "tool_choice": "auto",
        "stream": False,
    }
    second = ctx.transport.post_json(ctx.transport.endpoints.responses_url, second_payload)
    if second.error_class is not None:
        return _failure_from_json(
            probe_id,
            title,
            capability,
            kind,
            second,
            evidence=base_evidence,
            bad_request_unsupported=True,
            bad_request_reason="tool_result_continuation_rejected",
        )
    body = second.json_value
    output = body.get("output") if isinstance(body, dict) else None
    evidence = {
        **base_evidence,
        "continuation_status": second.status,
        "output_is_list": isinstance(output, list),
        "output_item_count": len(output) if isinstance(output, list) else 0,
        "output_item_types": _output_item_types(body),
    }
    if not isinstance(body, dict) or not isinstance(output, list):
        return _result(
            probe_id, title, capability, kind, STATUS_INCONCLUSIVE,
            "continuation_missing_output", evidence=evidence, http_status=second.status,
        )
    if body.get("error") is not None:
        return _result(
            probe_id, title, capability, kind, STATUS_UNSUPPORTED, "error_payload",
            evidence=evidence, http_status=second.status,
        )
    return _result(
        probe_id, title, capability, kind, STATUS_SUPPORTED,
        "tool_result_continuation_observed", evidence=evidence, http_status=second.status,
    )


def probe_responses_vision_data_uri(ctx: ProbeContext) -> ProbeResult:
    probe_id = "responses.vision_data_uri"
    title = "Responses vision data URI"
    capability = "responses_vision_data_uri"
    kind = PROBE_KIND_STATEFUL
    payload = {
        "model": ctx.model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": VISION_PROMPT},
                    {"type": "input_image", "image_url": TINY_PNG_DATA_URI},
                ],
            }
        ],
        "stream": False,
    }
    result = ctx.transport.post_json(ctx.transport.endpoints.responses_url, payload)
    if result.error_class is not None:
        return _failure_from_json(
            probe_id,
            title,
            capability,
            kind,
            result,
            bad_request_unsupported=True,
            bad_request_reason="image_input_rejected",
        )
    body = result.json_value
    evidence = {
        "image_mime": "image/png",
        "input_content_parts": ["input_text", "input_image"],
        "object_value": _object_value(body.get("object")) if isinstance(body, dict) else "absent",
        "output_is_list": isinstance(body.get("output"), list) if isinstance(body, dict) else False,
        "output_item_types": _output_item_types(body),
    }
    if not isinstance(body, dict):
        return _result(
            probe_id, title, capability, kind, STATUS_UNSUPPORTED,
            "unrecognized_response_shape", evidence=evidence, http_status=result.status,
        )
    output = body.get("output")
    if not isinstance(output, list) or not output:
        return _result(
            probe_id, title, capability, kind, STATUS_UNSUPPORTED, "empty_output",
            evidence=evidence, http_status=result.status,
        )
    return _result(
        probe_id, title, capability, kind, STATUS_SUPPORTED, "image_input_accepted",
        evidence=evidence, http_status=result.status,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeSpec:
    id: str
    title: str
    capability: str
    kind: str
    run: Callable[[ProbeContext], ProbeResult]


PROBE_SPECS: tuple[ProbeSpec, ...] = (
    ProbeSpec("models.list", "GET /models", "models_list", PROBE_KIND_READ_ONLY, probe_models_list),
    ProbeSpec(
        "chat.completions.non_stream", "Chat Completions non-stream",
        "chat_completions_non_stream", PROBE_KIND_STATEFUL, probe_chat_non_stream,
    ),
    ProbeSpec(
        "chat.completions.stream", "Chat Completions stream=true",
        "chat_completions_stream", PROBE_KIND_STATEFUL, probe_chat_stream,
    ),
    ProbeSpec(
        "chat.completions.function_calling", "Chat Completions function calling",
        "chat_completions_function_calling", PROBE_KIND_STATEFUL,
        probe_chat_function_calling,
    ),
    ProbeSpec(
        "chat.completions.tool_result_continuation",
        "Chat Completions tool-result continuation",
        "chat_completions_tool_result_continuation",
        PROBE_KIND_STATEFUL,
        probe_chat_tool_result_continuation,
    ),
    ProbeSpec(
        "chat.completions.vision_data_uri", "Chat Completions vision data URI",
        "chat_completions_vision_data_uri", PROBE_KIND_STATEFUL,
        probe_chat_vision_data_uri,
    ),
    ProbeSpec(
        "responses.non_stream", "Responses non-stream",
        "responses_non_stream", PROBE_KIND_STATEFUL, probe_responses_non_stream,
    ),
    ProbeSpec(
        "responses.stream", "Responses stream=true",
        "responses_stream", PROBE_KIND_STATEFUL, probe_responses_stream,
    ),
    ProbeSpec(
        "responses.max_output_tokens", "Responses max_output_tokens enforcement",
        "responses_max_output_tokens", PROBE_KIND_STATEFUL,
        probe_responses_max_output_tokens,
    ),
    ProbeSpec(
        "responses.function_calling", "Responses function calling",
        "responses_function_calling", PROBE_KIND_STATEFUL,
        probe_responses_function_calling,
    ),
    ProbeSpec(
        "responses.tool_result_continuation",
        "Responses tool-result continuation",
        "responses_tool_result_continuation",
        PROBE_KIND_STATEFUL,
        probe_responses_tool_result_continuation,
    ),
    ProbeSpec(
        "responses.vision_data_uri", "Responses vision data URI",
        "responses_vision_data_uri", PROBE_KIND_STATEFUL,
        probe_responses_vision_data_uri,
    ),
)

PROBE_BY_ID = {spec.id: spec for spec in PROBE_SPECS}


def run_probes(
    ctx: ProbeContext,
    *,
    allow_stateful: bool = True,
    on_progress: Callable[[ProbeResult], None] | None = None,
) -> list[ProbeResult]:
    """Run the registry in order, converting unexpected errors to safe results.

    A failure in one probe never stops the remaining probes.  ``on_progress``
    receives each result as soon as it is available (the CLI prints only id,
    status, and reason, never evidence content).
    """
    results: list[ProbeResult] = []
    for spec in PROBE_SPECS:
        if spec.kind == PROBE_KIND_STATEFUL and not allow_stateful:
            result = _result(
                spec.id,
                spec.title,
                spec.capability,
                spec.kind,
                STATUS_SKIPPED,
                "skipped_by_no_write_probes",
            )
        else:
            try:
                result = spec.run(ctx)
            except Exception as exc:  # noqa: BLE001 - must keep probing
                result = _result(
                    spec.id,
                    spec.title,
                    spec.capability,
                    spec.kind,
                    STATUS_ERROR,
                    "probe_internal_error",
                    evidence={"exception_type": type(exc).__name__},
                    error_class="internal_error",
                )
        results.append(result)
        if on_progress is not None:
            on_progress(result)
    return results
