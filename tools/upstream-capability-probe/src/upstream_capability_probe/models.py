"""Data model and serialization for upstream capability profiles.

The profile is deliberately an allow-list of structural facts.  It never
contains request prompts, response text, response bodies, token values,
headers, or authentication material.  Everything here is JSON-serializable
so a proxy or an agent can consume it without a Python dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import TOOL_NAME, __version__

SCHEMA_VERSION = 1

STATUS_SUPPORTED = "supported"
STATUS_UNSUPPORTED = "unsupported"
STATUS_INCONCLUSIVE = "inconclusive"
STATUS_SKIPPED = "skipped"
STATUS_ERROR = "error"

ALL_STATUSES = (
    STATUS_SUPPORTED,
    STATUS_UNSUPPORTED,
    STATUS_INCONCLUSIVE,
    STATUS_SKIPPED,
    STATUS_ERROR,
)

PROBE_KIND_READ_ONLY = "read_only"
PROBE_KIND_STATEFUL = "stateful"
ALL_PROBE_KINDS = (PROBE_KIND_READ_ONLY, PROBE_KIND_STATEFUL)

DEFAULT_LIMITATIONS = [
    "This tool probes HTTP behaviour only. A successful probe does not prove that the upstream is semantically equivalent to the OpenAI API.",
    "Probes are synthetic, tiny, and single-sample. Results can vary with model, sampling, upstream version, and account entitlements.",
    "Stateful probes trigger model generation and can incur provider cost or create server-side history. Use --no-write-probes for a metadata-only run.",
    "The profile never stores raw response bodies, prompts, completion text, tokens, headers, or authentication values.",
    "The tool never calls destructive endpoints (delete, cancel, archive) and never retries automatically.",
    "Streaming probes read the response body to completion or up to a fixed size/time limit; they do not measure interactive latency.",
    "The Responses max_output_tokens probe requires an explicit limit-related incomplete status and consistent usage within the test cap; acceptance or parameter echo alone is inconclusive.",
]

SECURITY_BLOCK = {
    "contains_prompts": False,
    "contains_response_bodies": False,
    "contains_completion_text": False,
    "contains_tokens": False,
    "contains_headers": False,
    "contains_auth_values": False,
    "raw_response_retained": False,
}


@dataclass
class ProbeResult:
    """One probe execution result.

    ``supported`` is intentionally tri-state: True / False for a definite
    verdict, None when the probe was skipped or could not reach a verdict.
    """

    id: str
    title: str
    capability: str
    kind: str
    status: str
    supported: bool | None
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)
    error_class: str | None = None
    http_status: int | None = None

    def __post_init__(self) -> None:
        if self.status not in ALL_STATUSES:
            raise ValueError(f"invalid probe status: {self.status}")
        if self.kind not in ALL_PROBE_KINDS:
            raise ValueError(f"invalid probe kind: {self.kind}")
        expected = {
            STATUS_SUPPORTED: True,
            STATUS_UNSUPPORTED: False,
            STATUS_INCONCLUSIVE: None,
            STATUS_SKIPPED: None,
            STATUS_ERROR: None,
        }[self.status]
        if self.supported is not expected:
            raise ValueError(
                f"supported={self.supported!r} is inconsistent with status={self.status}"
            )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "capability": self.capability,
            "kind": self.kind,
            "status": self.status,
            "supported": self.supported,
            "reason": self.reason,
            "evidence": self.evidence,
        }
        if self.error_class is not None:
            result["error_class"] = self.error_class
        if self.http_status is not None:
            result["http_status"] = self.http_status
        return result


def _capability_entry(
    probe_id: str,
    status: str,
    supported: bool | None,
    reason: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "supported": supported,
        "reason": reason,
        "probe": probe_id,
    }


def _derived_capabilities(probe: ProbeResult) -> dict[str, dict[str, Any]]:
    """Capabilities inferred from a parent probe's structural evidence."""
    if probe.id == "chat.completions.stream":
        if probe.status != STATUS_SUPPORTED:
            prefix = {
                "chat_completions_sse_done": probe,
                "chat_completions_sse_named_events": probe,
            }
            return {
                name: _capability_entry(probe.id, probe.status, probe.supported, probe.reason)
                for name in prefix
            }
        evidence = probe.evidence
        done = evidence.get("sse_done_seen")
        names = evidence.get("event_names") or []
        return {
            "chat_completions_sse_done": _capability_entry(
                probe.id,
                STATUS_SUPPORTED if done is True else STATUS_UNSUPPORTED,
                True if done is True else False,
                "sse_done_marker_observed" if done is True else "sse_done_marker_missing",
            ),
            "chat_completions_sse_named_events": _capability_entry(
                probe.id,
                STATUS_SUPPORTED if names else STATUS_UNSUPPORTED,
                True if names else False,
                "named_sse_events_observed" if names else "only_anonymous_sse_events",
            ),
        }
    if probe.id == "responses.stream":
        if probe.status != STATUS_SUPPORTED:
            return {
                name: _capability_entry(probe.id, probe.status, probe.supported, probe.reason)
                for name in ("responses_sse_done", "responses_sse_named_events")
            }
        evidence = probe.evidence
        terminal = evidence.get("terminal_event_seen")
        names = evidence.get("event_names") or []
        return {
            "responses_sse_done": _capability_entry(
                probe.id,
                STATUS_SUPPORTED if terminal is True else STATUS_UNSUPPORTED,
                True if terminal is True else False,
                "terminal_stream_event_observed"
                if terminal is True
                else "terminal_stream_event_missing",
            ),
            "responses_sse_named_events": _capability_entry(
                probe.id,
                STATUS_SUPPORTED if names else STATUS_UNSUPPORTED,
                True if names else False,
                "named_sse_events_observed" if names else "only_anonymous_sse_events",
            ),
        }
    if probe.id == "chat.completions.non_stream":
        if probe.status != STATUS_SUPPORTED:
            return {
                "chat_completions_usage": _capability_entry(
                    probe.id, probe.status, probe.supported, probe.reason
                )
            }
        evidence = probe.evidence
        present = evidence.get("usage_present")
        fields = evidence.get("usage_fields") or []
        if present is True and fields:
            status, supported, reason = (
                STATUS_SUPPORTED,
                True,
                "usage_fields_observed",
            )
        elif present is True:
            status, supported, reason = (
                STATUS_INCONCLUSIVE,
                None,
                "usage_object_empty",
            )
        else:
            status, supported, reason = (
                STATUS_UNSUPPORTED,
                False,
                "usage_object_absent",
            )
        return {"chat_completions_usage": _capability_entry(probe.id, status, supported, reason)}
    if probe.id == "responses.non_stream":
        if probe.status != STATUS_SUPPORTED:
            return {
                "responses_usage": _capability_entry(
                    probe.id, probe.status, probe.supported, probe.reason
                )
            }
        evidence = probe.evidence
        present = evidence.get("usage_present")
        fields = evidence.get("usage_fields") or []
        if present is True and fields:
            status, supported, reason = (
                STATUS_SUPPORTED,
                True,
                "usage_fields_observed",
            )
        elif present is True:
            status, supported, reason = (
                STATUS_INCONCLUSIVE,
                None,
                "usage_object_empty",
            )
        else:
            status, supported, reason = (
                STATUS_UNSUPPORTED,
                False,
                "usage_object_absent",
            )
        return {"responses_usage": _capability_entry(probe.id, status, supported, reason)}
    return {}


ASSEMBLY_HINTS = {
    "native_responses_stream": "responses_stream",
    "native_responses_non_stream": "responses_non_stream",
    "chat_completions_stream": "chat_completions_stream",
    "chat_completions_non_stream": "chat_completions_non_stream",
    "responses_function_calling": "responses_function_calling",
    "chat_completions_function_calling": "chat_completions_function_calling",
    "responses_tool_result_continuation": "responses_tool_result_continuation",
    "chat_completions_tool_result_continuation": "chat_completions_tool_result_continuation",
    "responses_vision_data_uri": "responses_vision_data_uri",
    "chat_completions_vision_data_uri": "chat_completions_vision_data_uri",
    "chat_completions_usage": "chat_completions_usage",
    "responses_usage": "responses_usage",
    "responses_max_output_tokens": "responses_max_output_tokens",
}


@dataclass
class CapabilityProfile:
    """A complete, serializable capability profile."""

    target: dict[str, Any]
    options: dict[str, Any]
    probes: list[ProbeResult]
    limitations: list[str] = field(default_factory=lambda: list(DEFAULT_LIMITATIONS))
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    @property
    def tool(self) -> dict[str, str]:
        return {"name": TOOL_NAME, "version": __version__}

    def capabilities(self) -> dict[str, dict[str, Any]]:
        capabilities: dict[str, dict[str, Any]] = {}
        for probe in self.probes:
            capabilities[probe.capability] = _capability_entry(
                probe.id, probe.status, probe.supported, probe.reason
            )
            capabilities.update(_derived_capabilities(probe))
        return capabilities

    def assembly_hints(self) -> dict[str, dict[str, Any]]:
        capabilities = self.capabilities()
        hints: dict[str, dict[str, Any]] = {}
        for hint, capability in ASSEMBLY_HINTS.items():
            entry = capabilities.get(capability)
            if entry is None:
                hints[hint] = {
                    "status": "unknown",
                    "supported": None,
                    "capability": capability,
                }
            else:
                hints[hint] = {
                    "status": entry["status"],
                    "supported": entry["supported"],
                    "capability": capability,
                }
        return hints

    @staticmethod
    def _count_statuses(entries: Iterable[dict[str, Any]]) -> dict[str, int]:
        counts = {status: 0 for status in ALL_STATUSES}
        for entry in entries:
            status = entry.get("status", STATUS_ERROR)
            if status not in counts:
                status = STATUS_ERROR
            counts[status] += 1
        return counts

    def summary(self) -> dict[str, Any]:
        probe_counts = self._count_statuses(p.to_dict() for p in self.probes)
        capability_entries = list(self.capabilities().values())
        capability_counts = self._count_statuses(capability_entries)
        return {
            "probes": {"total": len(self.probes), **probe_counts},
            "capabilities": {"total": len(capability_entries), **capability_counts},
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "tool": self.tool,
            "generated_at": self.generated_at,
            "target": self.target,
            "options": self.options,
            "summary": self.summary(),
            "assembly_hints": self.assembly_hints(),
            "capabilities": self.capabilities(),
            "probes": [probe.to_dict() for probe in self.probes],
            "security": dict(SECURITY_BLOCK),
            "limitations": list(self.limitations),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        if destination.parent and str(destination.parent) not in ("", "."):
            destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json(), encoding="utf-8")
