"""Safe, redacted capability probing for OpenAI-style upstream endpoints.

The package intentionally depends only on the Python standard library.  It
sends synthetic requests and emits a machine-readable capability profile that
contains no prompts, response bodies, tokens, headers or authentication
values.
"""

from __future__ import annotations

__version__ = "0.1.0"
TOOL_NAME = "upstream-capability-probe"

__all__ = ["__version__", "TOOL_NAME"]
