"""Command-line entry point for the upstream capability probe.

Safety properties:

* The API key is read only from the environment variable named by
  ``--api-key-env``; it is never accepted as a command-line value and never
  printed, logged, or written to the output profile.
* Default output contains only structural probe facts.  Prompts, completion
  text, response bodies, headers, token values, and response identifiers are
  not retained.
* ``--no-write-probes`` skips every probe that can trigger model generation or
  server-side state.
* ``--proxy-profile`` converts the probe result to the Rust proxy's
  ``responses-capability-profile`` format.  That document never contains the
  base URL, header values, or credentials.
* ``--from-profile`` converts an existing probe profile offline: no network
  request is sent and no environment secret is read.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any, Mapping, Sequence

from . import TOOL_NAME, __version__
from .models import (
    DEFAULT_LIMITATIONS,
    PROBE_KIND_READ_ONLY,
    PROBE_KIND_STATEFUL,
    CapabilityProfile,
    ProbeResult,
)
from .probes import PROBE_SPECS, ProbeContext, run_probes
from .proxy_profile import (
    ProfileConversionError,
    convert_probe_profile,
    load_probe_profile,
    validate_channel_id,
    write_proxy_profile,
)
from .transport import (
    DEFAULT_TIMEOUT_SECONDS,
    InvalidBaseUrl,
    Transport,
    normalize_base_url,
)

SAFE_KEY_NAME_ALLOWED = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


def _positive_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(value) or value <= 0 or value > 3600:
        raise argparse.ArgumentTypeError("must be a finite number between 0 and 3600 seconds")
    return value


def _key_env_name(raw: str) -> str:
    if not raw or any(ch not in SAFE_KEY_NAME_ALLOWED for ch in raw):
        raise argparse.ArgumentTypeError(
            "must be an environment variable name (letters, digits, underscore)"
        )
    if raw[0].isdigit():
        raise argparse.ArgumentTypeError("must not start with a digit")
    return raw


def _channel_id(raw: str) -> str:
    try:
        return validate_channel_id(raw)
    except ProfileConversionError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=TOOL_NAME,
        description=(
            "Safely probe an OpenAI-style upstream and emit a redacted, "
            "machine-readable capability profile. Optionally convert the result "
            "to the Rust proxy's responses-capability-profile format."
        ),
        epilog=(
            "Only HTTP/HTTPS base URLs are accepted. The tool sends synthetic probes, "
            "never reads files, and never stores prompts, response bodies, completion text, "
            "tokens, headers, or authentication values. --from-profile converts an "
            "existing probe profile offline without sending any request."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "Upstream API root, e.g. https://api.example.com or https://api.example.com/v1. "
            "Full endpoint URLs ending in /chat/completions, /responses, or /models are also accepted. "
            "Required for a probe run; not used with --from-profile."
        ),
    )
    parser.add_argument(
        "--api-key-env",
        type=_key_env_name,
        default=None,
        metavar="ENV_NAME",
        help="Name of the environment variable holding the API key (value is never printed).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        metavar="MODEL",
        help="Model name to send in probe payloads (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--output",
        default="-",
        metavar="PATH",
        help="Write the probe JSON profile to PATH; '-' (default) writes to stdout.",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=DEFAULT_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help=f"Per-request timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS:g}).",
    )
    parser.add_argument(
        "--no-write-probes",
        action="store_true",
        help=(
            "Skip every stateful probe that can trigger model generation or server-side state. "
            "Only read-only probes run."
        ),
    )
    parser.add_argument(
        "--proxy-profile",
        default=None,
        metavar="PATH",
        help=(
            "Also write a converted Rust proxy profile (format responses-capability-profile). "
            "Requires --channel."
        ),
    )
    parser.add_argument(
        "--channel",
        type=_channel_id,
        default=None,
        metavar="ID",
        help=(
            "Channel id for the proxy profile: 1-64 characters of [A-Za-z0-9_.-]. "
            "Required with --proxy-profile."
        ),
    )
    parser.add_argument(
        "--from-profile",
        default=None,
        metavar="PATH",
        help=(
            "Offline mode: read an existing probe profile JSON from PATH ('-' for stdin) and "
            "convert it with --proxy-profile; no request is sent."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{TOOL_NAME} {__version__}",
    )
    return parser


def _resolve_api_key(env_name: str | None) -> str | None:
    if env_name is None:
        return None
    value = os.environ.get(env_name)
    if not value:
        raise SystemExit(
            f"{TOOL_NAME}: environment variable {env_name!r} is unset or empty; "
            "refusing to send unauthenticated probes"
        )
    return value


def _summary_line(result: ProbeResult) -> str:
    detail = result.status
    if result.error_class:
        detail += f"/{result.error_class}"
    return f"  {result.id:<42} {detail:<24} {result.reason}"


def _validate_mode_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.proxy_profile is not None and args.channel is None:
        parser.error("--channel is required with --proxy-profile")
    if args.channel is not None and args.proxy_profile is None:
        parser.error("--channel is only valid with --proxy-profile")
    if args.from_profile is not None:
        if args.proxy_profile is None:
            parser.error("--from-profile requires --proxy-profile")
        if args.output != "-":
            parser.error("--output is not used with --from-profile")
        if args.base_url is not None:
            parser.error("--base-url is not used with --from-profile")
        return
    if args.base_url is None:
        parser.error("--base-url is required for a probe run (or use --from-profile)")
    if args.output == "-" and args.proxy_profile == "-":
        parser.error("--output and --proxy-profile cannot both write to stdout")


def _convert(probe_profile: Mapping[str, Any], channel: str, parser: argparse.ArgumentParser) -> dict[str, Any]:
    try:
        return convert_probe_profile(probe_profile, channel)
    except ProfileConversionError as exc:
        parser.error(f"profile conversion failed: {exc}")
        raise AssertionError("unreachable") from exc


def _run_offline(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        probe_profile = load_probe_profile(args.from_profile)
    except ProfileConversionError as exc:
        parser.error(str(exc))
        return 2
    proxy_profile = _convert(probe_profile, args.channel, parser)
    try:
        write_proxy_profile(proxy_profile, args.proxy_profile)
    except OSError as exc:
        print(
            f"{TOOL_NAME}: cannot write {args.proxy_profile}: {exc.strerror}",
            file=sys.stderr,
        )
        return 1
    if args.proxy_profile != "-":
        print(f"{TOOL_NAME}: wrote {args.proxy_profile}", file=sys.stderr)
    return 0


def _run_probe(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        endpoints = normalize_base_url(args.base_url)
    except InvalidBaseUrl as exc:
        parser.error(f"--base-url: {exc}")
        return 2

    api_key = _resolve_api_key(args.api_key_env)

    transport = Transport(
        base_url=args.base_url,
        api_key=api_key,
        timeout=args.timeout,
        user_agent=f"{TOOL_NAME}/{__version__}",
    )
    ctx = ProbeContext(transport=transport, model=args.model)

    allow_stateful = not args.no_write_probes
    probe_descriptors = [
        {
            "id": spec.id,
            "title": spec.title,
            "capability": spec.capability,
            "kind": spec.kind,
        }
        for spec in PROBE_SPECS
    ]
    target = {
        "base_url": endpoints.display_url,
        "auth": "bearer_env" if args.api_key_env else "none",
        "api_key_env": args.api_key_env,
    }
    options = {
        "model": args.model,
        "timeout_seconds": args.timeout,
        "write_probes": allow_stateful,
        "read_only_probes_only": args.no_write_probes,
        "probe_kinds": {
            "read_only": sum(1 for p in PROBE_SPECS if p.kind == PROBE_KIND_READ_ONLY),
            "stateful": sum(1 for p in PROBE_SPECS if p.kind == PROBE_KIND_STATEFUL),
        },
        "planned_probes": probe_descriptors,
    }

    def on_progress(result: ProbeResult) -> None:
        if args.output == "-":
            print(_summary_line(result), file=sys.stderr, flush=True)

    if args.output == "-":
        print(
            f"{TOOL_NAME}: probing {endpoints.display_url} "
            f"(model={args.model}, write_probes={str(allow_stateful).lower()})",
            file=sys.stderr,
        )
    results = run_probes(ctx, allow_stateful=allow_stateful, on_progress=on_progress)

    profile = CapabilityProfile(
        target=target,
        options=options,
        probes=results,
        limitations=list(DEFAULT_LIMITATIONS),
    )
    serialized = profile.to_json()
    if args.output == "-":
        sys.stdout.write(serialized)
    else:
        try:
            profile.write(args.output)
        except OSError as exc:
            print(f"{TOOL_NAME}: cannot write {args.output}: {exc.strerror}", file=sys.stderr)
            return 1
        print(f"{TOOL_NAME}: wrote {args.output}", file=sys.stderr)

    if args.proxy_profile is not None:
        proxy_profile = _convert(profile.to_dict(), args.channel, parser)
        try:
            write_proxy_profile(proxy_profile, args.proxy_profile)
        except OSError as exc:
            print(
                f"{TOOL_NAME}: cannot write {args.proxy_profile}: {exc.strerror}",
                file=sys.stderr,
            )
            return 1
        if args.proxy_profile != "-":
            print(f"{TOOL_NAME}: wrote {args.proxy_profile}", file=sys.stderr)

    executed = [result for result in results if result.status != "skipped"]
    if executed and all(result.status == "error" for result in executed):
        return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_mode_args(parser, args)
    if args.from_profile is not None:
        return _run_offline(args, parser)
    return _run_probe(args, parser)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
