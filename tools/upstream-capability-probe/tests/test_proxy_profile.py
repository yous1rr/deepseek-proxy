"""Offline tests for the probe -> Rust proxy capability profile converter.

All tests are pure/deterministic; no network is used.  Live-run CLI wiring is
tested with a mocked probe runner that produces synthetic ProbeResult values.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from typing import Any, Mapping
from unittest import mock

from upstream_capability_probe.cli import main
from upstream_capability_probe.proxy_profile import (
    CAPABILITY_IDS,
    PROXY_PROFILE_FORMAT,
    PROXY_PROFILE_SCHEMA_VERSION,
    ProfileConversionError,
    convert_probe_profile,
    validate_channel_id,
)

PROBE_IDS = {
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

ALL_PROBE_CAPABILITIES = tuple(PROBE_IDS)


def make_probe_profile(
    statuses: Mapping[str, str] | None = None,
    *,
    generated_at: str = "2025-01-02T03:04:05Z",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal probe profile with the given capability statuses."""

    statuses = dict(statuses or {})
    probes = []
    capabilities = {}
    for capability, status in statuses.items():
        probe_id = PROBE_IDS[capability]
        probes.append(
            {
                "id": probe_id,
                "title": capability,
                "capability": capability,
                "kind": "stateful",
                "status": status,
                "supported": True if status == "supported" else False if status == "unsupported" else None,
                "reason": "test",
                "evidence": {},
            }
        )
        capabilities[capability] = {
            "status": status,
            "supported": True if status == "supported" else False if status == "unsupported" else None,
            "reason": "test",
            "probe": probe_id,
        }
    document: dict[str, Any] = {
        "schema_version": 1,
        "tool": {"name": "upstream-capability-probe", "version": "0.1.0"},
        "generated_at": generated_at,
        "target": {"base_url": "https://api.example.com/v1", "auth": "none"},
        "options": {"model": "gpt-4o-mini"},
        "capabilities": capabilities,
        "probes": probes,
    }
    if extra:
        document.update(extra)
    return document


def convert(statuses: Mapping[str, str] | None = None, channel: str = "openai_style") -> dict[str, Any]:
    return convert_probe_profile(make_probe_profile(statuses), channel)


class FormatAndContractTests(unittest.TestCase):
    def test_exact_top_level_format(self) -> None:
        profile = convert()
        self.assertEqual(profile["format"], PROXY_PROFILE_FORMAT)
        self.assertEqual(profile["format"], "responses-capability-profile")
        self.assertEqual(profile["schema_version"], PROXY_PROFILE_SCHEMA_VERSION)
        self.assertEqual(profile["schema_version"], 1)
        self.assertEqual(profile["channel"], "openai_style")
        self.assertEqual(
            profile["generated_by"],
            {"tool": "upstream-capability-probe", "version": "0.1.0"},
        )
        self.assertEqual(profile["generated_at"], "2025-01-02T03:04:05Z")

    def test_every_contract_id_present_in_order(self) -> None:
        profile = convert()
        self.assertEqual(list(profile["capabilities"]), list(CAPABILITY_IDS))
        self.assertEqual(len(profile["capabilities"]), 30)

    def test_every_contract_id_present_even_for_empty_profile(self) -> None:
        profile = convert_probe_profile({}, "empty_channel")
        self.assertEqual(list(profile["capabilities"]), list(CAPABILITY_IDS))
        self.assertEqual(len(profile["capabilities"]), 30)
        for capability_id in CAPABILITY_IDS:
            entry = profile["capabilities"][capability_id]
            self.assertIn(entry["support"], ("unknown",), capability_id)

    def test_unknown_ids_do_not_leak_into_output(self) -> None:
        profile = convert({"chat_completions_stream": "supported"})
        self.assertNotIn("chat_completions_stream", profile["capabilities"])
        self.assertNotIn("responses_stream", profile["capabilities"])
        self.assertNotIn("models_list", profile["capabilities"])

    def test_no_probe_only_metadata_in_output(self) -> None:
        profile = convert()
        serialized = json.dumps(profile)
        for needle in ("base_url", "api_key", "api_key_env", "target", "options", "probes", "models_list"):
            self.assertNotIn(needle, serialized, needle)

    def test_generated_at_defaults_to_utc_z_when_invalid(self) -> None:
        profile = convert_probe_profile(make_probe_profile(generated_at="not-a-date"), "c")
        self.assertRegex(profile["generated_at"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

    def test_rejects_wrong_probe_schema_version(self) -> None:
        document = make_probe_profile()
        document["schema_version"] = 99
        with self.assertRaises(ProfileConversionError):
            convert_probe_profile(document, "c")

    def test_rejects_foreign_tool(self) -> None:
        document = make_probe_profile()
        document["tool"] = {"name": "another-tool", "version": "1"}
        with self.assertRaises(ProfileConversionError):
            convert_probe_profile(document, "c")

    def test_rejects_non_object_profiles(self) -> None:
        with self.assertRaises(ProfileConversionError):
            convert_probe_profile([], "c")  # type: ignore[arg-type]
        with self.assertRaises(ProfileConversionError):
            convert_probe_profile({"capabilities": []}, "c")  # type: ignore[dict-item]


class ChannelValidationTests(unittest.TestCase):
    def test_valid_channel_ids(self) -> None:
        for channel in ("a", "A", "0", "openai_style", "gpt-4o.mini", "x" * 64, "a-b_c.d"):
            self.assertEqual(validate_channel_id(channel), channel)

    def test_invalid_channel_ids(self) -> None:
        for channel in (
            "",
            "x" * 65,
            "with space",
            "with/slash",
            "with:colon",
            "https://api.example.com",
            "ümlaut",
            "with,comma",
        ):
            with self.assertRaises(ProfileConversionError, msg=channel):
                validate_channel_id(channel)

    def test_convert_rejects_invalid_channel(self) -> None:
        with self.assertRaises(ProfileConversionError):
            convert_probe_profile(make_probe_profile(), "bad channel")


class TextRuleTests(unittest.TestCase):
    def test_text_supported_if_any_protocol_works(self) -> None:
        for capability in PROBE_IDS:
            if capability in (
                "chat_completions_non_stream",
                "chat_completions_stream",
                "responses_non_stream",
                "responses_stream",
            ):
                profile = convert({capability: "supported"})
                self.assertEqual(profile["capabilities"]["text"]["support"], "supported", capability)
                self.assertEqual(profile["capabilities"]["text"]["provider"], "upstream")

    def test_text_unsupported_only_when_all_four_unsupported(self) -> None:
        statuses = {
            "chat_completions_non_stream": "unsupported",
            "chat_completions_stream": "unsupported",
            "responses_non_stream": "unsupported",
            "responses_stream": "unsupported",
        }
        profile = convert(statuses)
        self.assertEqual(profile["capabilities"]["text"]["support"], "unsupported")
        self.assertEqual(profile["capabilities"]["text"]["provider"], "upstream")

    def test_text_unknown_with_mixed_statuses(self) -> None:
        statuses = {
            "chat_completions_non_stream": "unsupported",
            "chat_completions_stream": "unsupported",
            "responses_non_stream": "unsupported",
            "responses_stream": "skipped",
        }
        profile = convert(statuses)
        self.assertEqual(profile["capabilities"]["text"]["support"], "unknown")


class StreamingRuleTests(unittest.TestCase):
    def test_responses_stream_is_native_upstream(self) -> None:
        profile = convert({"responses_stream": "supported"})
        entry = profile["capabilities"]["streaming"]
        self.assertEqual(entry["support"], "supported")
        self.assertEqual(entry["provider"], "upstream")
        self.assertEqual(entry["mode"], "native_responses_sse")
        self.assertEqual(entry["evidence"], "responses.stream")

    def test_chat_stream_is_proxy_reencoded(self) -> None:
        profile = convert({"chat_completions_stream": "supported"})
        entry = profile["capabilities"]["streaming"]
        self.assertEqual(entry["support"], "supported")
        self.assertEqual(entry["provider"], "proxy")
        self.assertEqual(entry["mode"], "chat_stream_reencoded")
        self.assertEqual(entry["evidence"], "chat.completions.stream")

    def test_responses_stream_wins_over_chat(self) -> None:
        profile = convert(
            {
                "responses_stream": "supported",
                "chat_completions_stream": "supported",
            }
        )
        self.assertEqual(profile["capabilities"]["streaming"]["provider"], "upstream")

    def test_streaming_unsupported_if_both_unsupported(self) -> None:
        profile = convert(
            {
                "responses_stream": "unsupported",
                "chat_completions_stream": "unsupported",
            }
        )
        self.assertEqual(profile["capabilities"]["streaming"]["support"], "unsupported")

    def test_streaming_unknown_if_one_inconclusive(self) -> None:
        profile = convert(
            {
                "responses_stream": "unsupported",
                "chat_completions_stream": "inconclusive",
            }
        )
        self.assertEqual(profile["capabilities"]["streaming"]["support"], "unknown")


class NonStreamingRuleTests(unittest.TestCase):
    def test_responses_non_stream_is_native(self) -> None:
        entry = convert({"responses_non_stream": "supported"})["capabilities"]["non_streaming"]
        self.assertEqual((entry["support"], entry["provider"], entry["mode"]), ("supported", "upstream", "native"))

    def test_chat_non_stream_is_proxy_reencoded(self) -> None:
        entry = convert({"chat_completions_non_stream": "supported"})["capabilities"]["non_streaming"]
        self.assertEqual((entry["support"], entry["provider"], entry["mode"]), ("supported", "proxy", "chat_reencoded"))

    def test_streaming_support_implies_collected_stream(self) -> None:
        entry = convert({"chat_completions_stream": "supported"})["capabilities"]["non_streaming"]
        self.assertEqual((entry["support"], entry["provider"], entry["mode"]), ("supported", "proxy", "collected_stream"))

    def test_non_streaming_unsupported_only_if_all_paths_unsupported(self) -> None:
        profile = convert(
            {
                "responses_non_stream": "unsupported",
                "chat_completions_non_stream": "unsupported",
                "responses_stream": "unsupported",
                "chat_completions_stream": "unsupported",
            }
        )
        self.assertEqual(profile["capabilities"]["non_streaming"]["support"], "unsupported")

    def test_non_streaming_unknown_when_streaming_unknown(self) -> None:
        profile = convert(
            {
                "responses_non_stream": "unsupported",
                "chat_completions_non_stream": "unsupported",
                "responses_stream": "skipped",
                "chat_completions_stream": "unsupported",
            }
        )
        self.assertEqual(profile["capabilities"]["non_streaming"]["support"], "unknown")


class ImageRuleTests(unittest.TestCase):
    def test_supported_by_either_vision_probe(self) -> None:
        for capability in ("responses_vision_data_uri", "chat_completions_vision_data_uri"):
            entry = convert({capability: "supported"})["capabilities"]["image_input"]
            self.assertEqual(entry["support"], "supported", capability)
            self.assertEqual(entry["provider"], "upstream")

    def test_unsupported_only_if_both_unsupported(self) -> None:
        profile = convert(
            {
                "responses_vision_data_uri": "unsupported",
                "chat_completions_vision_data_uri": "unsupported",
            }
        )
        self.assertEqual(profile["capabilities"]["image_input"]["support"], "unsupported")

    def test_unknown_on_mixed(self) -> None:
        profile = convert(
            {
                "responses_vision_data_uri": "supported",
                "chat_completions_vision_data_uri": "skipped",
            }
        )
        self.assertEqual(profile["capabilities"]["image_input"]["support"], "supported")
        profile = convert(
            {
                "responses_vision_data_uri": "unsupported",
                "chat_completions_vision_data_uri": "inconclusive",
            }
        )
        self.assertEqual(profile["capabilities"]["image_input"]["support"], "unknown")


class FunctionToolsRuleTests(unittest.TestCase):
    def test_responses_pair_both_supported(self) -> None:
        entry = convert(
            {
                "responses_function_calling": "supported",
                "responses_tool_result_continuation": "supported",
            }
        )["capabilities"]["function_tools"]
        self.assertEqual(entry["support"], "supported")
        self.assertEqual(entry["provider"], "upstream")
        self.assertIn("responses.function_calling", entry["evidence"])
        self.assertIn("responses.tool_result_continuation", entry["evidence"])

    def test_chat_pair_both_supported(self) -> None:
        entry = convert(
            {
                "chat_completions_function_calling": "supported",
                "chat_completions_tool_result_continuation": "supported",
            }
        )["capabilities"]["function_tools"]
        self.assertEqual(entry["support"], "supported")

    def test_cross_protocol_mix_is_not_supported(self) -> None:
        # function calling on Responses + continuation on Chat is not a pair.
        entry = convert(
            {
                "responses_function_calling": "supported",
                "chat_completions_tool_result_continuation": "supported",
            }
        )["capabilities"]["function_tools"]
        self.assertEqual(entry["support"], "unknown")

    def test_missing_continuation_is_unknown(self) -> None:
        entry = convert({"responses_function_calling": "supported"})["capabilities"]["function_tools"]
        self.assertEqual(entry["support"], "unknown")

    def test_unsupported_only_if_both_function_calling_probes_unsupported(self) -> None:
        profile = convert(
            {
                "responses_function_calling": "unsupported",
                "chat_completions_function_calling": "unsupported",
                "responses_tool_result_continuation": "skipped",
                "chat_completions_tool_result_continuation": "skipped",
            }
        )
        self.assertEqual(profile["capabilities"]["function_tools"]["support"], "unsupported")


class StoredResponseRuleTests(unittest.TestCase):
    def test_stored_and_previous_are_proxy_local_when_text_supported(self) -> None:
        profile = convert({"chat_completions_stream": "supported"})
        for capability in ("stored_responses", "previous_response"):
            entry = profile["capabilities"][capability]
            self.assertEqual(entry["support"], "supported", capability)
            self.assertEqual(entry["provider"], "proxy")
            self.assertEqual(entry["mode"], "local_transcript")

    def test_stored_responses_unknown_when_text_unknown(self) -> None:
        profile = convert({})
        for capability in ("stored_responses", "previous_response"):
            entry = profile["capabilities"][capability]
            self.assertEqual(entry["support"], "unknown", capability)
            self.assertEqual(entry["provider"], "proxy")
            self.assertEqual(entry["mode"], "local_transcript")

    def test_stored_responses_unknown_when_text_unsupported(self) -> None:
        statuses = {
            "chat_completions_non_stream": "unsupported",
            "chat_completions_stream": "unsupported",
            "responses_non_stream": "unsupported",
            "responses_stream": "unsupported",
        }
        profile = convert(statuses)
        self.assertEqual(profile["capabilities"]["stored_responses"]["support"], "unknown")


class MaxOutputTokensRuleTests(unittest.TestCase):
    def test_supported_uses_only_the_independent_responses_probe(self):
        entry = convert({"responses_max_output_tokens": "supported"})["capabilities"]["max_output_tokens"]
        self.assertEqual(entry, {
            "support": "supported", "provider": "upstream",
            "evidence": "responses.max_output_tokens",
        })

    def test_explicitly_unsupported_is_preserved(self):
        entry = convert({"responses_max_output_tokens": "unsupported"})["capabilities"]["max_output_tokens"]
        self.assertEqual(entry, {
            "support": "unsupported", "provider": "none",
            "evidence": "responses.max_output_tokens",
        })

    def test_non_verdict_statuses_are_unknown_without_evidence(self):
        for status in ("inconclusive", "skipped", "error", "unknown", "future_status"):
            with self.subTest(status=status):
                entry = convert({"responses_max_output_tokens": status})["capabilities"]["max_output_tokens"]
                self.assertEqual(entry, {"support": "unknown", "provider": "none"})

    def test_older_profile_without_new_probe_stays_unknown(self):
        statuses = {capability: "supported" for capability in PROBE_IDS if capability != "responses_max_output_tokens"}
        entry = convert(statuses)["capabilities"]["max_output_tokens"]
        self.assertEqual(entry, {"support": "unknown", "provider": "none"})

    def test_chat_token_parameters_cannot_enable_responses_limit(self):
        document = make_probe_profile({"chat_completions_non_stream": "supported", "chat_completions_usage": "supported"})
        for capability in ("chat_completions_max_tokens", "chat_completions_max_completion_tokens"):
            document["capabilities"][capability] = {"status": "supported"}
        entry = convert_probe_profile(document, "chat_only")["capabilities"]["max_output_tokens"]
        self.assertEqual(entry, {"support": "unknown", "provider": "none"})

    def test_probe_rows_can_supply_the_independent_verdict(self):
        document = make_probe_profile({"responses_max_output_tokens": "supported"})
        document["capabilities"] = {}
        entry = convert_probe_profile(document, "responses")["capabilities"]["max_output_tokens"]
        self.assertEqual(entry["support"], "supported")
        self.assertEqual(entry["evidence"], "responses.max_output_tokens")

    def test_inconclusive_capability_overrides_an_older_supported_probe_row(self):
        document = make_probe_profile({"responses_max_output_tokens": "supported"})
        document["capabilities"]["responses_max_output_tokens"]["status"] = "inconclusive"
        entry = convert_probe_profile(document, "responses")["capabilities"]["max_output_tokens"]
        self.assertEqual(entry, {"support": "unknown", "provider": "none"})


class ExactUsageRuleTests(unittest.TestCase):
    def test_supported_if_either_usage_probe_supported(self) -> None:
        for capability in ("responses_usage", "chat_completions_usage"):
            entry = convert({capability: "supported"})["capabilities"]["exact_usage"]
            self.assertEqual(entry["support"], "supported", capability)
            self.assertEqual(entry["provider"], "upstream")

    def test_unknown_when_usage_absent_or_unsupported(self) -> None:
        for statuses in ({}, {"responses_usage": "unsupported"}, {"chat_completions_usage": "skipped"}):
            entry = convert(statuses)["capabilities"]["exact_usage"]
            self.assertEqual(entry["support"], "unknown", statuses)


class UnknownAndDefaultRuleTests(unittest.TestCase):
    def test_all_other_ids_are_unknown_provider_none(self) -> None:
        profile = convert(
            {
                "chat_completions_stream": "supported",
                "responses_stream": "supported",
                "responses_vision_data_uri": "supported",
                "responses_function_calling": "supported",
                "responses_tool_result_continuation": "supported",
                "responses_usage": "supported",
                "responses_non_stream": "supported",
                "responses_max_output_tokens": "supported",
            }
        )
        expected_unknown = (
            "file_input",
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
            "logprobs",
            "conversations",
            "background",
            "stream_resume",
            "auto_truncation",
            "compaction",
            "prompt_templates",
        )
        for capability in expected_unknown:
            entry = profile["capabilities"][capability]
            self.assertEqual(entry["support"], "unknown", capability)
            self.assertEqual(entry["provider"], "none", capability)
            self.assertNotIn("evidence", entry, capability)

    def test_inconclusive_skipped_error_all_become_unknown(self) -> None:
        for status in ("inconclusive", "skipped", "error", "something_new"):
            profile = convert({"chat_completions_stream": status})
            self.assertEqual(profile["capabilities"]["text"]["support"], "unknown", status)
            self.assertEqual(profile["capabilities"]["streaming"]["support"], "unknown", status)

    def test_fail_closed_for_absent_capabilities(self) -> None:
        profile = convert({})
        for capability in ("text", "streaming", "non_streaming", "image_input", "function_tools", "exact_usage"):
            self.assertEqual(profile["capabilities"][capability]["support"], "unknown", capability)

    def test_dependency_ids_are_never_auto_enabled(self) -> None:
        # tool_choice/background/stream_resume must never be supported by the
        # converter even when their dependencies appear supported.
        profile = convert(
            {
                "chat_completions_stream": "supported",
                "chat_completions_function_calling": "supported",
                "chat_completions_tool_result_continuation": "supported",
            }
        )
        for capability in ("tool_choice", "background", "stream_resume", "conversations"):
            self.assertEqual(profile["capabilities"][capability]["support"], "unknown", capability)


class EvidenceTests(unittest.TestCase):
    def test_evidence_lists_driving_probe_ids(self) -> None:
        profile = convert({"chat_completions_stream": "supported"})
        self.assertEqual(profile["capabilities"]["streaming"]["evidence"], "chat.completions.stream")
        # text is supported by the chat stream probe alone; that is the driver.
        self.assertEqual(profile["capabilities"]["text"]["evidence"], "chat.completions.stream")

    def test_text_unsupported_evidence_lists_all_unsupported_probes(self) -> None:
        profile = convert(
            {
                "chat_completions_non_stream": "unsupported",
                "chat_completions_stream": "unsupported",
                "responses_non_stream": "unsupported",
                "responses_stream": "unsupported",
            }
        )
        self.assertEqual(
            profile["capabilities"]["text"]["evidence"],
            "chat.completions.non_stream,chat.completions.stream,responses.non_stream,responses.stream",
        )

    def test_evidence_never_contains_free_text_from_probe_profile(self) -> None:
        document = make_probe_profile({"chat_completions_stream": "supported"})
        document["probes"][0]["id"] = "chat.completions.stream"
        document["probes"][0]["evidence"] = {"note": "SECRET_PROMPT_TEXT"}
        profile = convert_probe_profile(document, "c")
        serialized = json.dumps(profile)
        self.assertNotIn("SECRET_PROMPT_TEXT", serialized)


class SecretRedactionTests(unittest.TestCase):
    def test_proxy_profile_contains_no_environment_or_url_material(self) -> None:
        document = make_probe_profile(
            {"chat_completions_stream": "supported"},
            extra={
                "target": {
                    "base_url": "https://user:password@api.example.com/v1?api_key=SECRET_QUERY",
                    "auth": "bearer_env",
                    "api_key_env": "UPSTREAM_SECRET_ENV",
                },
                "options": {"model": "SECRET_MODEL"},
            },
        )
        profile = convert_probe_profile(document, "openai_style")
        serialized = json.dumps(profile)
        for needle in (
            "SECRET_QUERY",
            "SECRET_MODEL",
            "UPSTREAM_SECRET_ENV",
            "user:password",
            "https://",
            "Bearer",
            "Authorization",
            "api_key",
            "base_url",
        ):
            self.assertNotIn(needle, serialized, needle)


class OfflineCliTests(unittest.TestCase):
    def _write(self, directory: str, name: str, payload: Mapping[str, Any]) -> str:
        path = os.path.join(directory, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return path

    def test_offline_conversion_writes_proxy_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = self._write(tmp, "probe.json", make_probe_profile({"chat_completions_stream": "supported"}))
            output = os.path.join(tmp, "proxy.json")
            exit_code = main(
                ["--from-profile", source, "--proxy-profile", output, "--channel", "openai_style"]
            )
            self.assertEqual(exit_code, 0)
            with open(output, encoding="utf-8") as handle:
                profile = json.load(handle)
            self.assertEqual(profile["format"], PROXY_PROFILE_FORMAT)
            self.assertEqual(profile["schema_version"], 1)
            self.assertEqual(profile["channel"], "openai_style")
            self.assertEqual(profile["capabilities"]["streaming"]["mode"], "chat_stream_reencoded")
            self.assertEqual(len(profile["capabilities"]), 30)

    def test_offline_conversion_never_touches_transport(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = self._write(tmp, "probe.json", make_probe_profile())
            output = os.path.join(tmp, "proxy.json")
            with mock.patch("upstream_capability_probe.cli.Transport") as transport_cls:
                exit_code = main(
                    ["--from-profile", source, "--proxy-profile", output, "--channel", "c"]
                )
            self.assertEqual(exit_code, 0)
            transport_cls.assert_not_called()

    def test_offline_conversion_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = self._write(tmp, "probe.json", make_probe_profile())
            with mock.patch("sys.stdout") as stdout:
                exit_code = main(
                    ["--from-profile", source, "--proxy-profile", "-", "--channel", "c"]
                )
            self.assertEqual(exit_code, 0)
            written = "".join(call.args[0] for call in stdout.write.call_args_list)
            self.assertEqual(json.loads(written)["format"], PROXY_PROFILE_FORMAT)

    def test_proxy_profile_requires_channel(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            main(["--base-url", "https://example.com", "--proxy-profile", "out.json"])
        self.assertEqual(caught.exception.code, 2)

    def test_channel_requires_proxy_profile(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            main(["--base-url", "https://example.com", "--channel", "c"])
        self.assertEqual(caught.exception.code, 2)

    def test_from_profile_requires_proxy_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = self._write(tmp, "probe.json", make_probe_profile())
            with self.assertRaises(SystemExit) as caught:
                main(["--from-profile", source])
            self.assertEqual(caught.exception.code, 2)

    def test_from_profile_rejects_base_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = self._write(tmp, "probe.json", make_probe_profile())
            output = os.path.join(tmp, "proxy.json")
            with self.assertRaises(SystemExit) as caught:
                main(
                    [
                        "--from-profile",
                        source,
                        "--proxy-profile",
                        output,
                        "--channel",
                        "c",
                        "--base-url",
                        "https://example.com",
                    ]
                )
            self.assertEqual(caught.exception.code, 2)

    def test_malformed_probe_profile_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "bad.json")
            with open(source, "w", encoding="utf-8") as handle:
                handle.write("{not json")
            output = os.path.join(tmp, "proxy.json")
            with self.assertRaises(SystemExit) as caught:
                main(["--from-profile", source, "--proxy-profile", output, "--channel", "c"])
            self.assertEqual(caught.exception.code, 2)

    def test_invalid_channel_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = self._write(tmp, "probe.json", make_probe_profile())
            output = os.path.join(tmp, "proxy.json")
            with self.assertRaises(SystemExit) as caught:
                main(["--from-profile", source, "--proxy-profile", output, "--channel", "bad channel"])
            self.assertEqual(caught.exception.code, 2)

    def test_help_mentions_proxy_options(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            main(["--help"])
        self.assertEqual(caught.exception.code, 0)

    def test_probe_run_writes_proxy_profile(self) -> None:
        from upstream_capability_probe.models import (
            DEFAULT_LIMITATIONS,
            CapabilityProfile,
            ProbeResult,
        )

        fake_result = ProbeResult(
            id="chat.completions.stream",
            title="Chat Completions stream=true",
            capability="chat_completions_stream",
            kind="stateful",
            status="supported",
            supported=True,
            reason="sse_stream_observed",
            evidence={"is_event_stream": True},
        )
        profile = CapabilityProfile(
            target={"base_url": "https://api.example.com/v1", "auth": "bearer_env",
                    "api_key_env": "SECRET_ENV_NAME"},
            options={"model": "secret-model"},
            probes=[fake_result],
            limitations=list(DEFAULT_LIMITATIONS),
        )
        with tempfile.TemporaryDirectory() as tmp:
            probe_out = os.path.join(tmp, "probe.json")
            proxy_out = os.path.join(tmp, "proxy.json")
            with mock.patch("upstream_capability_probe.cli.run_probes", return_value=[fake_result]), \
                 mock.patch("upstream_capability_probe.cli.Transport") as transport_cls:
                exit_code = main(
                    [
                        "--base-url", "https://api.example.com",
                        "--output", probe_out,
                        "--proxy-profile", proxy_out,
                        "--channel", "openai_compat",
                    ]
                )
            self.assertEqual(exit_code, 0)
            transport_cls.assert_called_once()
            with open(proxy_out, encoding="utf-8") as handle:
                proxy = json.load(handle)
            self.assertEqual(proxy["format"], PROXY_PROFILE_FORMAT)
            self.assertEqual(proxy["channel"], "openai_compat")
            self.assertEqual(proxy["capabilities"]["streaming"]["support"], "supported")
            serialized = json.dumps(proxy)
            self.assertNotIn("SECRET_ENV_NAME", serialized)
            self.assertNotIn("secret-model", serialized)
            self.assertNotIn("api.example.com", serialized)

    def test_both_stdout_outputs_rejected(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            main(
                [
                    "--base-url", "https://example.com",
                    "--output", "-",
                    "--proxy-profile", "-",
                    "--channel", "c",
                ]
            )
        self.assertEqual(caught.exception.code, 2)


class StatusMatrixTests(unittest.TestCase):
    def test_every_status_on_every_probe_capability_is_safe(self) -> None:
        for capability in PROBE_IDS:
            for status in (
                "supported",
                "unsupported",
                "inconclusive",
                "skipped",
                "error",
                "unknown",
            ):
                profile = convert({capability: status})
                self.assertEqual(len(profile["capabilities"]), 30)
                for capability_id, entry in profile["capabilities"].items():
                    self.assertIn(
                        entry["support"],
                        ("supported", "unsupported", "unknown"),
                        (capability, status, capability_id),
                    )
                    self.assertIn(
                        entry["provider"],
                        ("upstream", "proxy", "none"),
                        (capability, status, capability_id),
                    )


class ConservativeFallbackTests(unittest.TestCase):
    def test_inconclusive_capability_entry_overrides_supported_probe_row(self) -> None:
        # The derived `capabilities` entry is authoritative: if it says a
        # non-verdict status, a contradictory probe row must not promote it.
        document = make_probe_profile({"chat_completions_stream": "supported"})
        document["capabilities"]["chat_completions_stream"]["status"] = "inconclusive"
        profile = convert_probe_profile(document, "c")
        self.assertEqual(profile["capabilities"]["text"]["support"], "unknown")
        self.assertEqual(profile["capabilities"]["streaming"]["support"], "unknown")

    def test_unknown_evidence_lists_only_observed_verdicts(self) -> None:
        statuses = {
            "chat_completions_non_stream": "unsupported",
            "chat_completions_stream": "unsupported",
            "responses_non_stream": "unsupported",
            "responses_stream": "skipped",
        }
        profile = convert(statuses)
        self.assertEqual(
            profile["capabilities"]["text"]["evidence"],
            "chat.completions.non_stream,chat.completions.stream,responses.non_stream",
        )
        self.assertNotIn("responses.stream", profile["capabilities"]["text"]["evidence"])
        self.assertEqual(
            profile["capabilities"]["streaming"]["evidence"],
            "chat.completions.stream",
        )

    def test_unknown_decision_has_no_evidence_when_nothing_observed(self) -> None:
        profile = convert({"responses_stream": "skipped"})
        self.assertNotIn("evidence", profile["capabilities"]["streaming"])

    def test_generated_at_with_impossible_date_falls_back(self) -> None:
        profile = convert_probe_profile(
            make_probe_profile(generated_at="2025-13-40T99:99:99Z"), "c"
        )
        self.assertNotEqual(profile["generated_at"], "2025-13-40T99:99:99Z")
        self.assertRegex(
            profile["generated_at"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
        )

    def test_from_profile_rejects_output_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = self._write(tmp, "probe.json", make_probe_profile())
            with self.assertRaises(SystemExit) as caught:
                main(
                    [
                        "--from-profile", source,
                        "--proxy-profile", os.path.join(tmp, "proxy.json"),
                        "--channel", "c",
                        "--output", os.path.join(tmp, "probe-out.json"),
                    ]
                )
            self.assertEqual(caught.exception.code, 2)

    def _write(self, directory: str, name: str, payload: Mapping[str, Any]) -> str:
        path = os.path.join(directory, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return path


if __name__ == "__main__":
    unittest.main()
