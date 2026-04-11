from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from deepplanning.config import load_model_config


def _is_reasoning_model(model_name: str) -> bool:
    lowered = model_name.lower()
    return any(token in lowered for token in ["o1", "o3", "o4-mini", "reasoner"])


def _merge_reasoning(
    extra_body: dict[str, Any], reasoning_enabled: bool | None
) -> dict[str, Any]:
    merged = dict(extra_body)
    if reasoning_enabled is None:
        return merged

    reasoning = dict(merged.get("reasoning") or {})
    reasoning["enabled"] = reasoning_enabled
    merged["reasoning"] = reasoning
    return merged


@dataclass(slots=True)
class ProviderConfig:
    alias: str
    model: str
    provider: str | None = None
    api_base: str | None = None
    api_key_env: str | None = None
    temperature: float | None = None
    max_retries: int = 1
    backoff: float = 1.0
    extra_body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_model_name(cls, model_name: str) -> "ProviderConfig":
        config = load_model_config(model_name)
        return cls(
            alias=model_name,
            model=config.get("model_name", model_name),
            provider=config.get("model_type"),
            api_base=config.get("base_url"),
            api_key_env=config.get("api_key_env"),
            temperature=config.get("temperature"),
            max_retries=max(int(config.get("max_retries", 1)), 1),
            backoff=float(config.get("backoff", 1.0)),
            extra_body=dict(config.get("extra_body") or {}),
        )


def extract_usage_tokens(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0

    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    return prompt_tokens, completion_tokens


def _validate_response(response: Any) -> None:
    message = response.choices[0].message
    has_content = bool((getattr(message, "content", None) or "").strip())
    has_tool_calls = bool(getattr(message, "tool_calls", None))
    if not has_content and not has_tool_calls:
        raise ValueError("Model returned an empty response without tool calls")


def _retryable_exception_types(litellm_module: Any) -> tuple[type[BaseException], ...]:
    exception_names = (
        "APIConnectionError",
        "APIError",
        "InternalServerError",
        "RateLimitError",
        "ServiceUnavailableError",
        "Timeout",
    )
    retryable: list[type[BaseException]] = [ConnectionError, OSError, TimeoutError]
    for name in exception_names:
        exc_type = getattr(litellm_module, name, None)
        if isinstance(exc_type, type) and issubclass(exc_type, BaseException):
            retryable.append(exc_type)
    return tuple(dict.fromkeys(retryable))


def _configure_litellm_callbacks() -> None:
    import litellm

    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        return

    # LiteLLM's legacy "langfuse" callback path still targets the older SDK.
    # For Langfuse 4.x, route directly to the OTEL integration instead.
    if not os.getenv("LANGFUSE_OTEL_HOST") and os.getenv("LANGFUSE_HOST"):
        os.environ["LANGFUSE_OTEL_HOST"] = os.environ["LANGFUSE_HOST"]

    callbacks = [
        callback
        for callback in list(getattr(litellm, "callbacks", []) or [])
        if callback != "langfuse"
    ]
    if "langfuse_otel" not in callbacks:
        callbacks.append("langfuse_otel")
    litellm.callbacks = callbacks


async def call_chat_completion(
    provider: ProviderConfig,
    messages: list[Any],
    tools: list[dict[str, Any]] | None = None,
    reasoning_enabled: bool | None = None,
    validate_nonempty: bool = False,
) -> Any:
    import litellm

    _configure_litellm_callbacks()

    api_key = os.getenv(provider.api_key_env) if provider.api_key_env else None
    if provider.api_key_env and not api_key:
        raise RuntimeError(
            f"API key not found for model '{provider.alias}'. Set {provider.api_key_env}."
        )

    params: dict[str, Any] = {
        "model": provider.model,
        "messages": messages,
        "api_key": api_key,
    }
    if provider.provider:
        params["custom_llm_provider"] = provider.provider
    if provider.api_base:
        params["api_base"] = provider.api_base
    if tools:
        params["tools"] = tools
    if provider.temperature is not None and not _is_reasoning_model(provider.model):
        params["temperature"] = provider.temperature

    extra_body = _merge_reasoning(provider.extra_body, reasoning_enabled)
    if extra_body:
        params["extra_body"] = extra_body

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max(provider.max_retries, 1)),
        wait=wait_exponential(
            multiplier=max(provider.backoff, 0.1), min=max(provider.backoff, 0.1)
        ),
        retry=retry_if_exception_type(_retryable_exception_types(litellm)),
        reraise=True,
    ):
        with attempt:
            response = await litellm.acompletion(**params)
            if validate_nonempty:
                _validate_response(response)
            return response

    raise RuntimeError("LLM call failed unexpectedly")
