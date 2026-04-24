from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

try:
    from ._bootstrap import ensure_repo_root_on_path
except ImportError:
    from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from llm import ProviderConfig, call_chat_completion, flush_langfuse

DEFAULT_PROMPT = "Reply with one short sentence confirming this smoke test works."

__all__ = [
    "DEFAULT_PROMPT",
    "main",
    "render_response",
    "run",
    "run_completion",
]


def _normalize_model_alias(model_alias: str) -> str:
    normalized_model_alias = str(model_alias).strip()
    if not normalized_model_alias:
        raise ValueError(
            "Model smoke runner requires a model alias as the first argument."
        )
    return normalized_model_alias


def _normalize_prompt(prompt_parts: list[str]) -> str:
    prompt = " ".join(str(part) for part in prompt_parts).strip()
    return prompt or DEFAULT_PROMPT


def _jsonify(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonify(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonify(value.model_dump(mode="json"))
    if hasattr(value, "__dict__"):
        return {
            str(key): _jsonify(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


def render_response(response: Any) -> str:
    return json.dumps(_jsonify(response), indent=2, ensure_ascii=False)


async def run_completion(model_alias: str, prompt: str) -> Any:
    provider = ProviderConfig.from_model_name(_normalize_model_alias(model_alias))
    return await call_chat_completion(
        provider=provider,
        messages=[{"role": "user", "content": prompt}],
        validate_nonempty=True,
    )


def run(model_alias: str, *prompt_parts: str) -> str:
    prompt = _normalize_prompt(list(prompt_parts))
    try:
        response = asyncio.run(run_completion(model_alias, prompt))
    finally:
        flush_langfuse()
    return render_response(response)


def main(argv: list[str] | None = None) -> None:
    args = list(argv if argv is not None else sys.argv[1:])
    if not args:
        raise ValueError(
            "Model smoke runner requires a model alias as the first argument."
        )
    print(run(args[0], *args[1:]))


if __name__ == "__main__":
    main()
