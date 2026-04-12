from __future__ import annotations

import asyncio
import json
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SENSITIVE_HEADER_NAMES = {
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "cookie",
    "set-cookie",
}


def _isoformat_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds")


def _serialize_tool_call(tool_call: Any) -> dict[str, Any]:
    if isinstance(tool_call, dict):
        return {
            key: _to_jsonable(value)
            for key, value in tool_call.items()
            if value is not None
        }

    payload: dict[str, Any] = {}
    tool_call_id = getattr(tool_call, "id", None)
    if tool_call_id is not None:
        payload["id"] = tool_call_id

    tool_type = getattr(tool_call, "type", None)
    if tool_type is not None:
        payload["type"] = tool_type

    function = getattr(tool_call, "function", None)
    if function is not None:
        payload["function"] = {
            "name": getattr(function, "name", None),
            "arguments": getattr(function, "arguments", None),
        }

    return {key: value for key, value in payload.items() if value is not None}


def _serialize_usage(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None

    payload = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }
    return {key: value for key, value in payload.items() if value is not None}


def _coerce_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    raise TypeError(f"Unsupported timestamp value: {type(value)!r}")


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump(exclude_none=True))
    if hasattr(value, "__dict__"):
        return _to_jsonable(vars(value))
    return str(value)


def _redact_headers(headers: Any) -> dict[str, Any]:
    try:
        items = dict(headers.items())
    except Exception:
        try:
            items = dict(headers)
        except Exception:
            return {}

    payload: dict[str, Any] = {}
    for key, value in items.items():
        if str(key).lower() in SENSITIVE_HEADER_NAMES:
            payload[str(key)] = "<redacted>"
        else:
            payload[str(key)] = _to_jsonable(value)
    return payload


def _serialize_request(request: Any) -> dict[str, Any] | None:
    if request is None:
        return None

    payload = {
        "method": getattr(request, "method", None),
        "url": str(getattr(request, "url", "")) or None,
    }
    headers = _redact_headers(getattr(request, "headers", None))
    if headers:
        payload["headers"] = headers
    return {key: value for key, value in payload.items() if value is not None}


def _serialize_response_error(response: Any) -> dict[str, Any] | None:
    if response is None:
        return None

    payload: dict[str, Any] = {
        "status_code": getattr(response, "status_code", None),
        "reason_phrase": getattr(response, "reason_phrase", None),
        "url": str(getattr(response, "url", "")) or None,
    }
    headers = _redact_headers(getattr(response, "headers", None))
    if headers:
        payload["headers"] = headers

    try:
        json_body = response.json()
    except Exception:
        json_body = None
    if json_body is not None:
        payload["json_body"] = _to_jsonable(json_body)
    else:
        try:
            text_body = response.text
        except Exception:
            text_body = None
        if text_body:
            payload["text_body"] = text_body[:4000]

    request_payload = _serialize_request(getattr(response, "request", None))
    if request_payload is not None:
        payload["request"] = request_payload

    return {key: value for key, value in payload.items() if value is not None}


def serialize_exception(exc: BaseException) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": type(exc).__name__,
        "module": type(exc).__module__,
        "message": str(exc),
        "repr": repr(exc),
        "args": _to_jsonable(list(exc.args)),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ).strip(),
    }

    for attr_name in (
        "status_code",
        "model",
        "request_id",
        "code",
        "param",
        "max_retries",
        "num_retries",
    ):
        attr_value = getattr(exc, attr_name, None)
        if attr_value is not None:
            payload[attr_name] = _to_jsonable(attr_value)

    body = getattr(exc, "body", None)
    if body is not None:
        payload["body"] = _to_jsonable(body)

    request_payload = _serialize_request(getattr(exc, "request", None))
    if request_payload is not None:
        payload["request"] = request_payload

    response_payload = _serialize_response_error(getattr(exc, "response", None))
    if response_payload is not None:
        payload["response"] = response_payload

    if exc.__cause__ is not None:
        payload["cause"] = {
            "type": type(exc.__cause__).__name__,
            "message": str(exc.__cause__),
        }
    if exc.__context__ is not None and exc.__context__ is not exc.__cause__:
        payload["context"] = {
            "type": type(exc.__context__).__name__,
            "message": str(exc.__context__),
        }

    return payload


def serialize_message(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return _to_jsonable(message)

    payload: dict[str, Any] = {}
    for field_name in ("role", "content", "name", "tool_call_id", "reasoning_content"):
        field_value = getattr(message, field_name, None)
        if field_value is not None:
            payload[field_name] = field_value

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        payload["tool_calls"] = [
            _serialize_tool_call(tool_call) for tool_call in tool_calls
        ]

    return _to_jsonable(payload)


def serialize_messages(messages: list[Any]) -> list[dict[str, Any]]:
    return [serialize_message(message) for message in messages]


def serialize_response(response: Any) -> dict[str, Any]:
    if response is None:
        return {}
    if hasattr(response, "model_dump"):
        return _to_jsonable(response.model_dump(exclude_none=True))

    payload: dict[str, Any] = {}
    for field_name in ("id", "model", "system_fingerprint"):
        field_value = getattr(response, field_name, None)
        if field_value is not None:
            payload[field_name] = field_value

    choices = []
    for index, choice in enumerate(getattr(response, "choices", []) or []):
        choices.append(
            {
                "index": getattr(choice, "index", index),
                "finish_reason": getattr(choice, "finish_reason", None),
                "message": serialize_message(getattr(choice, "message", None)),
            }
        )
    payload["choices"] = choices

    usage_payload = _serialize_usage(getattr(response, "usage", None))
    if usage_payload is not None:
        payload["usage"] = usage_payload

    return _to_jsonable(payload)


@dataclass
class StructuredLogger:
    output_dir: Path | None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self) -> None:
        if self.output_dir is None:
            self.events_path = None
            self.results_path = None
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "agent_events.jsonl"
        self.results_path = self.output_dir / "task_results.jsonl"

    async def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.output_dir is None or self.events_path is None:
            return
        record = {
            "timestamp": _isoformat_utc(datetime.now(timezone.utc)),
            "event_type": event_type,
            **payload,
        }
        await self._append_jsonl(self.events_path, record)

    async def log_turn(
        self,
        *,
        domain: str,
        task_id: str,
        run_id: int,
        phase: str,
        turn_index: int,
        started_at: Any,
        ended_at: Any,
        request_messages: list[Any],
        raw_response: Any,
        parsed_tool_calls: list[dict[str, Any]],
        parse_warnings: list[str],
        tool_results: list[dict[str, Any]],
        prompt_tokens: int,
        completion_tokens: int,
        stop_reason: str,
        model_alias: str,
    ) -> None:
        started_at_dt = _coerce_datetime(started_at)
        ended_at_dt = _coerce_datetime(ended_at)
        duration_ms = int((ended_at_dt - started_at_dt).total_seconds() * 1000)
        await self.log_event(
            "executor_turn",
            {
                "domain": domain,
                "task_id": task_id,
                "run_id": run_id,
                "phase": phase,
                "turn_index": turn_index,
                "started_at": _isoformat_utc(started_at_dt),
                "ended_at": _isoformat_utc(ended_at_dt),
                "duration_ms": duration_ms,
                "request_messages": serialize_messages(request_messages),
                "raw_response": serialize_response(raw_response),
                "parsed_tool_calls": _to_jsonable(parsed_tool_calls),
                "parse_warnings": list(parse_warnings),
                "tool_results": _to_jsonable(tool_results),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "stop_reason": stop_reason,
                "model_alias": model_alias,
            },
        )

    async def log_result(self, payload: dict[str, Any]) -> None:
        if self.output_dir is None or self.results_path is None:
            return
        record = {
            "timestamp": _isoformat_utc(datetime.now(timezone.utc)),
            **payload,
        }
        await self._append_jsonl(self.results_path, record)

    async def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, default=str) + "\n"
        async with self._lock:
            await asyncio.to_thread(self._append_line, path, line)

    @staticmethod
    def _append_line(path: Path, line: str) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
