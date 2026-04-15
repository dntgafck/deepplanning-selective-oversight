from __future__ import annotations

import contextlib
import inspect
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from langfuse import Langfuse, get_client, propagate_attributes
from langfuse.openai import AsyncOpenAI as LangfuseAsyncOpenAI
from openai import AsyncOpenAI as OpenAIAsyncOpenAI
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from deepplanning.config import load_model_config
from experiment.logging import serialize_exception, serialize_messages
from failure_subtypes import (
    classify_exception_failure_subtype,
    is_transient_infrastructure_error,
)

RetryErrorHandler = Callable[[dict[str, Any]], Awaitable[None] | None]
LANGFUSE_ID_PART_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
MAX_LANGFUSE_SESSION_ID_LENGTH = 200


def _langfuse_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


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
    logprobs: bool | None = None
    top_logprobs: int | None = None
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
            logprobs=config.get("logprobs"),
            top_logprobs=config.get("top_logprobs"),
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


def _retryable_exception_types(openai_module: Any) -> tuple[type[BaseException], ...]:
    exception_names = (
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
    )
    retryable: list[type[BaseException]] = [ConnectionError, OSError, TimeoutError]
    for name in exception_names:
        exc_type = getattr(openai_module, name, None)
        if isinstance(exc_type, type) and issubclass(exc_type, BaseException):
            retryable.append(exc_type)
    return tuple(dict.fromkeys(retryable))


def _should_retry_exception(
    exc: BaseException,
    *,
    retryable_exception_types: tuple[type[BaseException], ...],
) -> bool:
    return is_transient_infrastructure_error(
        exc,
        retryable_exception_types=retryable_exception_types,
    )


def _normalize_langfuse_id_parts(*parts: object) -> list[str]:
    normalized_parts: list[str] = []
    for part in parts:
        if part is None:
            continue
        value = LANGFUSE_ID_PART_PATTERN.sub("-", str(part).strip()).strip("-_.")
        if value:
            normalized_parts.append(value)
    return normalized_parts


def build_langfuse_session_id(*parts: object) -> str:
    session_id = "-".join(_normalize_langfuse_id_parts(*parts)) or "session"
    truncated = session_id[:MAX_LANGFUSE_SESSION_ID_LENGTH].rstrip("-_.")
    return truncated or "session"


def build_langfuse_trace_id(*parts: object) -> str:
    seed = "-".join(_normalize_langfuse_id_parts(*parts)) or "inference"
    return Langfuse.create_trace_id(seed=seed)


def flush_langfuse() -> None:
    if not _langfuse_enabled():
        return

    with contextlib.suppress(Exception):
        get_client().flush()


def _build_async_client(
    provider: ProviderConfig,
    api_key: str | None,
) -> Any:
    client_kwargs: dict[str, Any] = {"max_retries": 0}
    if api_key is not None:
        client_kwargs["api_key"] = api_key
    if provider.api_base:
        client_kwargs["base_url"] = provider.api_base

    client_cls = LangfuseAsyncOpenAI if _langfuse_enabled() else OpenAIAsyncOpenAI
    return client_cls(**client_kwargs)


async def call_chat_completion(
    provider: ProviderConfig,
    messages: list[Any],
    tools: list[dict[str, Any]] | None = None,
    reasoning_enabled: bool | None = None,
    validate_nonempty: bool = False,
    on_attempt_error: RetryErrorHandler | None = None,
    error_context: dict[str, Any] | None = None,
    trace_id: str | None = None,
    session_id: str | None = None,
) -> Any:
    import openai

    api_key = os.getenv(provider.api_key_env) if provider.api_key_env else None
    if provider.api_key_env and not api_key:
        raise RuntimeError(
            f"API key not found for model '{provider.alias}'. Set {provider.api_key_env}."
        )

    params: dict[str, Any] = {
        "model": provider.model,
        "messages": serialize_messages(messages),
    }
    if tools:
        params["tools"] = tools
    if provider.temperature is not None:
        params["temperature"] = provider.temperature
    if provider.logprobs is not None:
        params["logprobs"] = provider.logprobs
    if provider.top_logprobs is not None:
        if provider.logprobs is False:
            raise ValueError("top_logprobs requires logprobs to be enabled")
        params["logprobs"] = True
        params["top_logprobs"] = provider.top_logprobs

    extra_body = _merge_reasoning(provider.extra_body, reasoning_enabled)
    if extra_body:
        params["extra_body"] = extra_body

    if _langfuse_enabled():
        params["name"] = "deepplanning-chat-completion"
        params["metadata"] = {
            "model_alias": provider.alias,
            "provider": provider.provider or "openai",
        }
        if trace_id:
            params["trace_id"] = trace_id

    max_attempts = max(provider.max_retries, 1)
    retryable_exceptions = _retryable_exception_types(openai)
    retryer = AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=max(provider.backoff, 0.1), min=max(provider.backoff, 0.1)
        ),
        retry=retry_if_exception(
            lambda exc: _should_retry_exception(
                exc,
                retryable_exception_types=retryable_exceptions,
            )
        ),
        reraise=True,
    )

    async for attempt in retryer:
        with attempt:
            client = _build_async_client(provider, api_key)
            async with client:
                scope = (
                    propagate_attributes(session_id=session_id)
                    if _langfuse_enabled() and session_id
                    else contextlib.nullcontext()
                )
                with scope:
                    response = await client.chat.completions.create(**params)
                    if validate_nonempty:
                        _validate_response(response)

        outcome = attempt.retry_state.outcome
        if outcome is not None and outcome.failed:
            exc = outcome.exception()
            if exc is not None and on_attempt_error is not None:
                payload = {
                    "attempt": attempt.retry_state.attempt_number,
                    "max_attempts": max_attempts,
                    "will_retry": _should_retry_exception(
                        exc,
                        retryable_exception_types=retryable_exceptions,
                    )
                    and attempt.retry_state.attempt_number < max_attempts,
                    "failure_subtype": classify_exception_failure_subtype(
                        exc,
                        retryable_exception_types=retryable_exceptions,
                    ),
                    "error": serialize_exception(exc),
                }
                if error_context:
                    payload.update(error_context)
                maybe_awaitable = on_attempt_error(payload)
                if inspect.isawaitable(maybe_awaitable):
                    await maybe_awaitable
            continue

        return response

    raise RuntimeError("LLM call failed unexpectedly")
