from __future__ import annotations

import json
import re
from typing import Any, Literal

FailureSubtype = Literal[
    "none",
    "infra_transient",
    "context_exhaustion",
    "max_tool_calls",
    "malformed_output",
]

FAILURE_SUBTYPE_NONE: FailureSubtype = "none"
FAILURE_SUBTYPE_INFRA_TRANSIENT: FailureSubtype = "infra_transient"
FAILURE_SUBTYPE_CONTEXT_EXHAUSTION: FailureSubtype = "context_exhaustion"
FAILURE_SUBTYPE_MAX_TOOL_CALLS: FailureSubtype = "max_tool_calls"
FAILURE_SUBTYPE_MALFORMED_OUTPUT: FailureSubtype = "malformed_output"

_CONTEXT_EXHAUSTION_PATTERNS = (
    re.compile(r"context[_ -]?length[_ -]?exceed", re.IGNORECASE),
    re.compile(r"maximum.{0,40}context.{0,20}(length|window)", re.IGNORECASE),
    re.compile(
        r"context.{0,20}(window|length).{0,40}(limit|exceed|overflow)",
        re.IGNORECASE,
    ),
    re.compile(
        r"input.{0,40}(too long|exceed).{0,40}(token|context)",
        re.IGNORECASE,
    ),
    re.compile(
        r"range of input length should be\s*\[\s*1\s*,\s*\d+\s*\]",
        re.IGNORECASE,
    ),
    re.compile(r"prompt.{0,30}(too long|exceed)", re.IGNORECASE),
    re.compile(r"too many tokens", re.IGNORECASE),
    re.compile(r"context[_ -]?window[_ -]?(exceeded|overflow)", re.IGNORECASE),
)

_MALFORMED_OUTPUT_PATTERNS = (
    "empty response without tool calls",
    "malformed output",
    "invalid json",
    "json decode error",
)


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, int, float)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def _exception_text(exc: BaseException, *, _depth: int = 0) -> str:
    parts = [
        str(exc),
        repr(exc),
        _to_text(getattr(exc, "code", None)),
        _to_text(getattr(exc, "param", None)),
        _to_text(getattr(exc, "body", None)),
    ]

    response = getattr(exc, "response", None)
    if response is not None:
        parts.append(_to_text(getattr(response, "status_code", None)))
        try:
            parts.append(_to_text(response.json()))
        except Exception:
            pass
        try:
            parts.append(_to_text(response.text))
        except Exception:
            pass

    if _depth < 1:
        cause = getattr(exc, "__cause__", None)
        if isinstance(cause, BaseException):
            parts.append(_exception_text(cause, _depth=_depth + 1))
        context = getattr(exc, "__context__", None)
        if isinstance(context, BaseException) and context is not cause:
            parts.append(_exception_text(context, _depth=_depth + 1))

    return "\n".join(part for part in parts if part).lower()


def is_context_exhaustion_error(exc: BaseException) -> bool:
    text = _exception_text(exc)
    return any(pattern.search(text) for pattern in _CONTEXT_EXHAUSTION_PATTERNS)


def is_malformed_output_error(exc: BaseException) -> bool:
    text = _exception_text(exc)
    return any(pattern in text for pattern in _MALFORMED_OUTPUT_PATTERNS)


def is_transient_infrastructure_error(
    exc: BaseException,
    *,
    retryable_exception_types: tuple[type[BaseException], ...] = (),
) -> bool:
    if is_context_exhaustion_error(exc):
        return False
    if retryable_exception_types and isinstance(exc, retryable_exception_types):
        return True
    if type(exc).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
    }:
        return True

    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int) and (status_code == 429 or status_code >= 500):
        return True

    return isinstance(exc, (ConnectionError, OSError, TimeoutError))


def classify_exception_failure_subtype(
    exc: BaseException,
    *,
    retryable_exception_types: tuple[type[BaseException], ...] = (),
) -> FailureSubtype:
    if is_context_exhaustion_error(exc):
        return FAILURE_SUBTYPE_CONTEXT_EXHAUSTION
    if is_malformed_output_error(exc):
        return FAILURE_SUBTYPE_MALFORMED_OUTPUT
    if is_transient_infrastructure_error(
        exc,
        retryable_exception_types=retryable_exception_types,
    ):
        return FAILURE_SUBTYPE_INFRA_TRANSIENT
    return FAILURE_SUBTYPE_NONE


def failure_subtype_from_stop_reason(stop_reason: str | None) -> FailureSubtype:
    if stop_reason == "max_steps_exhausted":
        return FAILURE_SUBTYPE_MAX_TOOL_CALLS
    return FAILURE_SUBTYPE_NONE
