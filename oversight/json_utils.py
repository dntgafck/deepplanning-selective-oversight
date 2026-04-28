from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

JSON_OBJECT_ENVELOPE_KEYS: tuple[str, ...] = (
    "payload",
    "response",
    "result",
    "data",
    "json",
    "content",
    "object",
    "execution_contract",
    "task_checklist",
)
JSON_OBJECT_RESPONSE_FORMAT: dict[str, str] = {"type": "json_object"}
JSON_OBJECT_PARSE_ERROR = "Expected JSON object payload"


def extract_json_content(content: str) -> Any:
    """Parse JSON from common LLM response wrappers before giving up."""
    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    fences = re.finditer(
        r"```(?:json)?\s*(.*?)\s*```",
        content,
        re.DOTALL | re.IGNORECASE,
    )
    for fence in fences:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            continue

    decoder = json.JSONDecoder()
    for index, char in enumerate(content):
        if char not in "{[":
            continue
        try:
            payload, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        return payload

    return json.loads(content)


def extract_json_object_content(
    payload: str | dict[str, Any],
    *,
    expected_keys: Iterable[str] = (),
) -> dict[str, Any]:
    data = extract_json_content(payload) if isinstance(payload, str) else payload
    return _coerce_json_object(data, expected_keys=tuple(expected_keys))


def _coerce_json_object(data: Any, *, expected_keys: tuple[str, ...]) -> dict[str, Any]:
    if isinstance(data, dict):
        if not expected_keys or any(key in data for key in expected_keys):
            return data
        for key in JSON_OBJECT_ENVELOPE_KEYS:
            if key not in data:
                continue
            try:
                return _coerce_json_object(data[key], expected_keys=expected_keys)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        return data

    if isinstance(data, list):
        if len(data) == 1:
            item = data[0]
            if isinstance(item, dict):
                return _coerce_json_object(item, expected_keys=expected_keys)
            if isinstance(item, str) and _looks_jsonish(item):
                try:
                    return _coerce_json_object(item, expected_keys=expected_keys)
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass
        for item in data:
            if not isinstance(item, dict):
                continue
            if expected_keys and any(key in item for key in expected_keys):
                return item
        raise ValueError(f"{JSON_OBJECT_PARSE_ERROR}, got list")

    if isinstance(data, str):
        try:
            nested = extract_json_content(data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{JSON_OBJECT_PARSE_ERROR}, got string") from exc
        return _coerce_json_object(nested, expected_keys=expected_keys)

    raise ValueError(f"{JSON_OBJECT_PARSE_ERROR}, got {type(data).__name__}")


def _looks_jsonish(value: str) -> bool:
    stripped = value.strip()
    return stripped.startswith(("{", "[", "```")) or "{" in stripped or "[" in stripped
