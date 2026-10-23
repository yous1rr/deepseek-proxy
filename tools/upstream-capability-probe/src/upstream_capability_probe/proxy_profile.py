"""Convert an upstream probe profile into the Rust proxy capability profile.

The proxy consumes an OPTIONAL ``responses-capability-profile`` document whose
``capabilities`` map can only NARROW the channel adapter's built-in capability
set (effective = adapter built-in AND profile).  This module implements the
single, auditable conversion from an ``upstream-capability-probe`` profile to
that format.

Design rules:

* The converter is pure: it takes a probe profile mapping plus a channel id
  and returns a new dict.  It performs no I/O and no network access.
* Output contains no target URL, auth variable name, header, prompt, response
  body, token, or any other free-form input string.  The only input strings
  that can reach the output are the validated channel id, a validated
  ``generated_at`` timestamp, and validated probe ids used as evidence.
* Only the 30 capability ids defined by the proxy contract are emitted, always
  all of them, in contract order.
* Anything that is not an explicit ``supported`` or ``unsupported`` probe
  status is treated as ``unknown``, which disables the capability in the proxy
  (fail closed).
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import TOOL_NAME, __version__

PROXY_PROFILE_FORMAT = "responses-capability-profile"
PROXY_PROFILE_SCHEMA_VERSION = 1

SUPPORT_SUPPORTED = "supported"
SUPPORT_UNSUPPORTED = "unsupported"
SUPPORT_UNKNOWN = "unknown"

PROVIDER_UPSTREAM = "upstream"
PROVIDER_PROXY = "proxy"
PROVIDER_NONE = "none"

# The proxy contract requires exactly these ids, always all of them, in this
# order.  Do not add probe-only names here; unknown ids trigger proxy warnings.
CAPABILITY_IDS: tuple[str, ...] = (
    "text",
    "streaming",
    "non_streaming",
    "image_input",
    "file_input",
    "function_tools",
    "tool_choice",
    "custom_tools",
    "web_search",
    "file_search",
    "code_interpreter",
    "mcp",
    "image_generation",
    "computer_use",
    "shell",
    "reasoning",
    "structured_output",
    "verbosity",
    "sampling_controls",
    "max_output_tokens",
    "logprobs",
    "stored_responses",
    "previous_response",
    "conversations",
    "background",
    "stream_resume",
    "auto_truncation",
    "compaction",
    "prompt_templates",
    "exact_usage",
)

# Probe-profile capability name -> canonical probe id.  Used only as evidence
# strings (short, static, secret-free identifiers).
CANONICAL_PROBE_IDS: dict[str, str] = {
    "chat_completions_non_stream": "chat.completions.non_stream",
    "chat_completions_stream": "chat.completions.stream",
    "chat_completions_function_calling": "chat.completions.function_calling",
    "chat_completions_tool_result_continuation": "chat.completions.tool_result_continuation",
    "chat_completions_vision_data_uri": "chat.completions.vision_data_uri",
    "responses_non_stream": "responses.non_stream",
    "responses_stream": "responses.stream",
    "responses_max_output_tokens": "responses.max_output_tokens",
    "responses_function_calling": "responses.function_calling",
    "responses_tool_result_continuation": "responses.tool_result_continuation",
    "responses_vision_data_uri": "responses.vision_data_uri",
    "chat_completions_usage": "chat.completions.non_stream",
    "responses_usage": "responses.non_stream",
}

# Only these exact probe ids may appear in evidence.  Anything else coming from
# an input document is ignored and replaced by the canonical id.
_KNOWN_PROBE_IDS = frozenset(CANONICAL_PROBE_IDS.values()) | frozenset({"models.list"})


_CHANNEL_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_PROBE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")
_GENERATED_AT_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})T(?P<time>\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))?Z$"
)

_PROBE_STATUS_VALUES = {SUPPORT_SUPPORTED, SUPPORT_UNSUPPORTED}
_TEXT_PROBE_CAPABILITIES = (
    "chat_completions_non_stream",
    "chat_completions_stream",
    "responses_non_stream",
    "responses_stream",
)
_STREAM_PROBE_CAPABILITIES = ("responses_stream", "chat_completions_stream")
_IMAGE_PROBE_CAPABILITIES = (
    "responses_vision_data_uri",
    "chat_completions_vision_data_uri",
)
_FUNCTION_PAIR_RESPONSES = (
    "responses_function_calling",
    "responses_tool_result_continuation",
)
_FUNCTION_PAIR_CHAT = (
    "chat_completions_function_calling",
    "chat_completions_tool_result_continuation",
)
_USAGE_PROBE_CAPABILITIES = ("responses_usage", "chat_completions_usage")


class ProfileConversionError(ValueError):
    """Raised when a probe profile or channel id cannot be converted safely."""


def validate_channel_id(channel: str) -> str:
    """Validate and return a proxy channel id.

    Contract: 1-64 characters from ``[A-Za-z0-9_.-]``.  No spaces, slashes,
    URLs, or other free-form text may enter the proxy profile.
    """

    if not isinstance(channel, str) or not _CHANNEL_RE.fullmatch(channel):
        raise ProfileConversionError(
            "channel id must be 1-64 characters of letters, digits, '.', '_', or '-'"
        )
    return channel


def _now_utc_z() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


class _ProbeProfileView:
    """Read-only, defensive accessors over a probe profile mapping."""

    def __init__(self, profile: Mapping[str, Any]) -> None:
        capabilities = profile.get("capabilities")
        if capabilities is None:
            capabilities = {}
        if not isinstance(capabilities, Mapping):
            raise ProfileConversionError("probe profile 'capabilities' must be a JSON object")
        probes = profile.get("probes")
        if probes is None:
            probes = []
        if not isinstance(probes, list):
            raise ProfileConversionError("probe profile 'probes' must be a JSON array")

        self._capabilities: Mapping[str, Any] = capabilities
        self._probe_status: dict[str, str] = {}
        self._probe_ids: dict[str, str] = {}
        for probe in probes:
            if not isinstance(probe, Mapping):
                continue
            capability = probe.get("capability")
            if not isinstance(capability, str):
                continue
            status = probe.get("status")
            if isinstance(status, str) and status in _PROBE_STATUS_VALUES:
                self._probe_status.setdefault(capability, status)
            probe_id = probe.get("id")
            if isinstance(probe_id, str) and _PROBE_ID_RE.fullmatch(probe_id):
                self._probe_ids.setdefault(capability, probe_id)

    def status(self, capability: str) -> str:
        """Return supported/unsupported/unknown for a probe capability."""
        entry = self._capabilities.get(capability)
        if isinstance(entry, Mapping):
            raw = entry.get("status")
            if isinstance(raw, str) and raw in _PROBE_STATUS_VALUES:
                return raw
            if raw is not None:
                # inconclusive/skipped/error and any future status are unknown.
                return SUPPORT_UNKNOWN
        return self._probe_status.get(capability, SUPPORT_UNKNOWN)

    def probe_id(self, capability: str) -> str:
        canonical = CANONICAL_PROBE_IDS[capability]
        entry = self._capabilities.get(capability)
        if isinstance(entry, Mapping):
            candidate = entry.get("probe")
            if candidate in _KNOWN_PROBE_IDS:
                return candidate
        candidate = self._probe_ids.get(capability)
        if candidate in _KNOWN_PROBE_IDS:
            return candidate
        return canonical

    def evidence(self, capabilities: Iterable[str]) -> list[str]:
        ids: list[str] = []
        for capability in capabilities:
            probe_id = self.probe_id(capability)
            if probe_id not in ids:
                ids.append(probe_id)
        return ids


def _decision(
    support: str,
    provider: str,
    *,
    mode: str | None = None,
    evidence: Iterable[str] | None = None,
) -> dict[str, str]:
    entry: dict[str, str] = {"support": support, "provider": provider}
    if mode:
        entry["mode"] = mode
    if evidence:
        ids = list(dict.fromkeys(evidence))
        if ids:
            entry["evidence"] = ",".join(ids)
    return entry


def _evidence_where(
    view: _ProbeProfileView, capabilities: Iterable[str], status: str
) -> list[str]:
    """Probe ids whose status equals ``status`` (deduplicated, stable order)."""

    ids: list[str] = []
    for capability in capabilities:
        if view.status(capability) == status:
            probe_id = view.probe_id(capability)
            if probe_id not in ids:
                ids.append(probe_id)
    return ids


def _observed_evidence(view: _ProbeProfileView, capabilities: Iterable[str]) -> list[str]:
    """Probe ids that actually produced a supported/unsupported verdict."""

    ids: list[str] = []
    for status in (SUPPORT_SUPPORTED, SUPPORT_UNSUPPORTED):
        for probe_id in _evidence_where(view, capabilities, status):
            if probe_id not in ids:
                ids.append(probe_id)
    return ids


def _text(view: _ProbeProfileView) -> dict[str, str]:
    statuses = [view.status(capability) for capability in _TEXT_PROBE_CAPABILITIES]
    if any(status == SUPPORT_SUPPORTED for status in statuses):
        return _decision(
            SUPPORT_SUPPORTED,
            PROVIDER_UPSTREAM,
            evidence=_evidence_where(view, _TEXT_PROBE_CAPABILITIES, SUPPORT_SUPPORTED),
        )
    if all(status == SUPPORT_UNSUPPORTED for status in statuses):
        return _decision(
            SUPPORT_UNSUPPORTED,
            PROVIDER_UPSTREAM,
            evidence=_evidence_where(view, _TEXT_PROBE_CAPABILITIES, SUPPORT_UNSUPPORTED),
        )
    return _decision(
        SUPPORT_UNKNOWN,
        PROVIDER_UPSTREAM,
        evidence=_observed_evidence(view, _TEXT_PROBE_CAPABILITIES),
    )


def _streaming(view: _ProbeProfileView) -> dict[str, str]:
    responses = view.status("responses_stream")
    chat = view.status("chat_completions_stream")
    if responses == SUPPORT_SUPPORTED:
        return _decision(
            SUPPORT_SUPPORTED,
            PROVIDER_UPSTREAM,
            mode="native_responses_sse",
            evidence=[view.probe_id("responses_stream")],
        )
    if chat == SUPPORT_SUPPORTED:
        return _decision(
            SUPPORT_SUPPORTED,
            PROVIDER_PROXY,
            mode="chat_stream_reencoded",
            evidence=[view.probe_id("chat_completions_stream")],
        )
    if responses == SUPPORT_UNSUPPORTED and chat == SUPPORT_UNSUPPORTED:
        return _decision(
            SUPPORT_UNSUPPORTED,
            PROVIDER_NONE,
            evidence=_evidence_where(view, _STREAM_PROBE_CAPABILITIES, SUPPORT_UNSUPPORTED),
        )
    return _decision(
        SUPPORT_UNKNOWN,
        PROVIDER_NONE,
        evidence=_observed_evidence(view, _STREAM_PROBE_CAPABILITIES),
    )


def _non_streaming(view: _ProbeProfileView, streaming: Mapping[str, str]) -> dict[str, str]:
    responses = view.status("responses_non_stream")
    chat = view.status("chat_completions_non_stream")
    all_capabilities = (
        "responses_non_stream",
        "chat_completions_non_stream",
    ) + _STREAM_PROBE_CAPABILITIES
    if responses == SUPPORT_SUPPORTED:
        return _decision(
            SUPPORT_SUPPORTED,
            PROVIDER_UPSTREAM,
            mode="native",
            evidence=[view.probe_id("responses_non_stream")],
        )
    if chat == SUPPORT_SUPPORTED:
        return _decision(
            SUPPORT_SUPPORTED,
            PROVIDER_PROXY,
            mode="chat_reencoded",
            evidence=[view.probe_id("chat_completions_non_stream")],
        )
    if streaming.get("support") == SUPPORT_SUPPORTED:
        return _decision(
            SUPPORT_SUPPORTED,
            PROVIDER_PROXY,
            mode="collected_stream",
            evidence=[item for item in (streaming.get("evidence") or "").split(",") if item],
        )
    if (
        responses == SUPPORT_UNSUPPORTED
        and chat == SUPPORT_UNSUPPORTED
        and streaming.get("support") == SUPPORT_UNSUPPORTED
    ):
        return _decision(
            SUPPORT_UNSUPPORTED,
            PROVIDER_NONE,
            evidence=_evidence_where(view, all_capabilities, SUPPORT_UNSUPPORTED),
        )
    return _decision(
        SUPPORT_UNKNOWN,
        PROVIDER_NONE,
        evidence=_observed_evidence(view, all_capabilities),
    )


def _image_input(view: _ProbeProfileView) -> dict[str, str]:
    statuses = [view.status(capability) for capability in _IMAGE_PROBE_CAPABILITIES]
    if any(status == SUPPORT_SUPPORTED for status in statuses):
        return _decision(
            SUPPORT_SUPPORTED,
            PROVIDER_UPSTREAM,
            evidence=_evidence_where(view, _IMAGE_PROBE_CAPABILITIES, SUPPORT_SUPPORTED),
        )
    if all(status == SUPPORT_UNSUPPORTED for status in statuses):
        return _decision(
            SUPPORT_UNSUPPORTED,
            PROVIDER_NONE,
            evidence=_evidence_where(view, _IMAGE_PROBE_CAPABILITIES, SUPPORT_UNSUPPORTED),
        )
    return _decision(
        SUPPORT_UNKNOWN,
        PROVIDER_NONE,
        evidence=_observed_evidence(view, _IMAGE_PROBE_CAPABILITIES),
    )


def _function_tools(view: _ProbeProfileView) -> dict[str, str]:
    responses_pair = [view.status(capability) for capability in _FUNCTION_PAIR_RESPONSES]
    chat_pair = [view.status(capability) for capability in _FUNCTION_PAIR_CHAT]
    if all(status == SUPPORT_SUPPORTED for status in responses_pair):
        return _decision(
            SUPPORT_SUPPORTED,
            PROVIDER_UPSTREAM,
            evidence=view.evidence(_FUNCTION_PAIR_RESPONSES),
        )
    if all(status == SUPPORT_SUPPORTED for status in chat_pair):
        return _decision(
            SUPPORT_SUPPORTED,
            PROVIDER_UPSTREAM,
            evidence=view.evidence(_FUNCTION_PAIR_CHAT),
        )
    function_calling_capabilities = (
        "responses_function_calling",
        "chat_completions_function_calling",
    )
    function_calling_statuses = [
        view.status(capability) for capability in function_calling_capabilities
    ]
    if all(status == SUPPORT_UNSUPPORTED for status in function_calling_statuses):
        return _decision(
            SUPPORT_UNSUPPORTED,
            PROVIDER_NONE,
            evidence=_evidence_where(
                view, function_calling_capabilities, SUPPORT_UNSUPPORTED
            ),
        )
    return _decision(
        SUPPORT_UNKNOWN,
        PROVIDER_NONE,
        evidence=_observed_evidence(view, _FUNCTION_PAIR_RESPONSES + _FUNCTION_PAIR_CHAT),
    )


def _max_output_tokens(view: _ProbeProfileView) -> dict[str, str]:
    capability = "responses_max_output_tokens"
    support = view.status(capability)
    return _decision(
        support,
        PROVIDER_UPSTREAM if support == SUPPORT_SUPPORTED else PROVIDER_NONE,
        evidence=[CANONICAL_PROBE_IDS[capability]]
        if support in _PROBE_STATUS_VALUES else None,
    )


def _stored_responses(view: _ProbeProfileView, text: Mapping[str, str]) -> dict[str, str]:
    evidence = [item for item in (text.get("evidence") or "").split(",") if item]
    if text.get("support") == SUPPORT_SUPPORTED:
        return _decision(
            SUPPORT_SUPPORTED,
            PROVIDER_PROXY,
            mode="local_transcript",
            evidence=evidence,
        )
    return _decision(
        SUPPORT_UNKNOWN,
        PROVIDER_PROXY,
        mode="local_transcript",
        evidence=evidence,
    )


def _exact_usage(view: _ProbeProfileView) -> dict[str, str]:
    statuses = [view.status(capability) for capability in _USAGE_PROBE_CAPABILITIES]
    if any(status == SUPPORT_SUPPORTED for status in statuses):
        return _decision(
            SUPPORT_SUPPORTED,
            PROVIDER_UPSTREAM,
            evidence=_evidence_where(view, _USAGE_PROBE_CAPABILITIES, SUPPORT_SUPPORTED),
        )
    return _decision(
        SUPPORT_UNKNOWN,
        PROVIDER_UPSTREAM,
        evidence=_observed_evidence(view, _USAGE_PROBE_CAPABILITIES),
    )


def _generated_at(probe_profile: Mapping[str, Any]) -> str:
    value = probe_profile.get("generated_at")
    if isinstance(value, str):
        match = _GENERATED_AT_RE.fullmatch(value)
        if match:
            try:
                datetime.strptime(
                    f"{match.group('date')}T{match.group('time')}Z",
                    "%Y-%m-%dT%H:%M:%SZ",
                )
            except ValueError:
                pass
            else:
                return value
    return _now_utc_z()


def _validate_probe_profile(probe_profile: Mapping[str, Any]) -> None:
    if not isinstance(probe_profile, Mapping):
        raise ProfileConversionError("probe profile must be a JSON object")
    schema_version = probe_profile.get("schema_version")
    if schema_version is not None and schema_version != 1:
        raise ProfileConversionError(
            f"unsupported probe profile schema_version: {schema_version!r}"
        )
    tool = probe_profile.get("tool")
    if isinstance(tool, Mapping):
        name = tool.get("name")
        if isinstance(name, str) and name and name != TOOL_NAME:
            raise ProfileConversionError(
                "probe profile was not produced by upstream-capability-probe"
            )


def convert_probe_profile(probe_profile: Mapping[str, Any], channel: str) -> dict[str, Any]:
    """Convert a probe profile mapping into a proxy capability profile.

    Pure function.  Unknown or ambiguous probe statuses become ``unknown``,
    which the proxy treats as disabled.  All 30 contract capability ids are
    present in the returned ``capabilities`` map.
    """

    channel_id = validate_channel_id(channel)
    if not isinstance(probe_profile, Mapping):
        raise ProfileConversionError("probe profile must be a JSON object")
    _validate_probe_profile(probe_profile)
    view = _ProbeProfileView(probe_profile)

    text = _text(view)
    streaming = _streaming(view)
    non_streaming = _non_streaming(view, streaming)

    decisions: dict[str, dict[str, str]] = {
        "text": text,
        "streaming": streaming,
        "non_streaming": non_streaming,
        "image_input": _image_input(view),
        "function_tools": _function_tools(view),
        "max_output_tokens": _max_output_tokens(view),
        "stored_responses": _stored_responses(view, text),
        "previous_response": _stored_responses(view, text),
        "exact_usage": _exact_usage(view),
    }

    capabilities: dict[str, dict[str, str]] = {}
    for capability_id in CAPABILITY_IDS:
        capabilities[capability_id] = decisions.get(
            capability_id,
            {"support": SUPPORT_UNKNOWN, "provider": PROVIDER_NONE},
        )

    return {
        "format": PROXY_PROFILE_FORMAT,
        "schema_version": PROXY_PROFILE_SCHEMA_VERSION,
        "channel": channel_id,
        "generated_by": {"tool": TOOL_NAME, "version": __version__},
        "generated_at": _generated_at(probe_profile),
        "capabilities": capabilities,
    }


def load_probe_profile(path: str) -> dict[str, Any]:
    """Load a probe profile JSON document from a path (or '-' for stdin).

    Raises ``ProfileConversionError`` for unreadable or malformed documents.
    """

    try:
        if path == "-":
            raw = sys.stdin.read()
        else:
            raw = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        reason = getattr(exc, "strerror", None) or str(exc)
        raise ProfileConversionError(f"cannot read probe profile: {reason}") from exc
    try:
        document = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ProfileConversionError("probe profile is not valid JSON") from exc
    if not isinstance(document, dict):
        raise ProfileConversionError("probe profile must be a JSON object")
    return document


def proxy_profile_json(profile: Mapping[str, Any]) -> str:
    """Serialize a proxy profile deterministically with a trailing newline."""

    return json.dumps(profile, indent=2, ensure_ascii=False) + "\n"


def write_proxy_profile(profile: Mapping[str, Any], path: str) -> None:
    """Write a proxy profile to a path (or stdout for '-')."""

    text = proxy_profile_json(profile)
    if path == "-":
        sys.stdout.write(text)
        return
    destination = Path(path)
    if destination.parent and str(destination.parent) not in ("", "."):
        destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
