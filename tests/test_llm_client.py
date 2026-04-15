from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import openai
import pytest

from experiment.logging import serialize_exception
from failure_subtypes import classify_exception_failure_subtype
from llm import client as llm_client


class FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class FakeToolFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.type = "function"
        self.function = FakeToolFunction(name, arguments)


class FakeMessage:
    def __init__(
        self,
        content: str,
        tool_calls: list[FakeToolCall] | None = None,
    ) -> None:
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls


class FakeChoice:
    def __init__(self, message: FakeMessage) -> None:
        self.index = 0
        self.finish_reason = "stop"
        self.message = message


class FakeResponse:
    def __init__(
        self,
        content: str,
        tool_calls: list[FakeToolCall] | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        self.id = "resp_1"
        self.model = "fake-model"
        self.system_fingerprint = "fp_test"
        self.choices = [FakeChoice(FakeMessage(content=content, tool_calls=tool_calls))]
        self.usage = FakeUsage(prompt_tokens, completion_tokens)


class FakeAsyncClient:
    def __init__(self, create_func) -> None:
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=create_func),
        )

    async def __aenter__(self) -> FakeAsyncClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


def _openai_status_error(
    exc_type: type[BaseException],
    *,
    status_code: int,
    message: str,
    error_type: str,
) -> BaseException:
    request = httpx.Request(
        "POST",
        "https://example.com/v1/chat/completions",
    )
    body = {
        "error": {
            "message": message,
            "type": error_type,
        }
    }
    response = httpx.Response(
        status_code,
        request=request,
        json=body,
        headers={"x-request-id": "req_123"},
    )
    return exc_type(message, response=response, body=body)


def test_build_langfuse_session_id_sanitizes_and_truncates():
    session_id = llm_client.build_langfuse_session_id(" thesis session ", "x" * 300)

    assert session_id.startswith("thesis-session-")
    assert len(session_id) <= 200
    assert " " not in session_id


def test_build_langfuse_trace_id_returns_deterministic_hex_trace_id():
    trace_id = llm_client.build_langfuse_trace_id(
        "experiment-session",
        "shopping",
        "qwen-plus",
        0,
        1,
    )

    assert trace_id == llm_client.build_langfuse_trace_id(
        "experiment-session",
        "shopping",
        "qwen-plus",
        0,
        1,
    )
    assert trace_id != llm_client.build_langfuse_trace_id(
        "experiment-session",
        "shopping",
        "qwen-plus",
        0,
        2,
    )
    assert len(trace_id) == 32
    int(trace_id, 16)


def test_call_chat_completion_returns_empty_no_tool_response_without_retry(
    monkeypatch,
):
    attempts = 0

    async def fake_create(**kwargs):
        nonlocal attempts
        attempts += 1
        return FakeResponse(content="", prompt_tokens=8, completion_tokens=0)

    monkeypatch.setattr(
        llm_client,
        "_build_async_client",
        lambda provider, api_key: FakeAsyncClient(fake_create),
    )
    provider = llm_client.ProviderConfig(alias="alias", model="test-model")

    response = asyncio.run(
        llm_client.call_chat_completion(
            provider=provider,
            messages=[{"role": "user", "content": "hello"}],
        )
    )

    assert response.choices[0].message.content == ""
    assert response.choices[0].message.tool_calls is None
    assert attempts == 1


def test_call_chat_completion_returns_tool_calls_unchanged(monkeypatch):
    tool_calls = [FakeToolCall("call_1", "lookup", '{"item":"tea"}')]

    async def fake_create(**kwargs):
        return FakeResponse(content="", tool_calls=tool_calls)

    monkeypatch.setattr(
        llm_client,
        "_build_async_client",
        lambda provider, api_key: FakeAsyncClient(fake_create),
    )
    provider = llm_client.ProviderConfig(alias="alias", model="test-model")

    response = asyncio.run(
        llm_client.call_chat_completion(
            provider=provider,
            messages=[{"role": "user", "content": "hello"}],
        )
    )

    assert response.choices[0].message.tool_calls == tool_calls


def test_call_chat_completion_normalizes_message_objects(monkeypatch):
    captured_params: list[dict[str, object]] = []
    tool_calls = [FakeToolCall("call_1", "lookup", '{"item":"tea"}')]

    async def fake_create(**kwargs):
        captured_params.append(kwargs)
        return FakeResponse(content="done")

    monkeypatch.setattr(
        llm_client,
        "_build_async_client",
        lambda provider, api_key: FakeAsyncClient(fake_create),
    )
    provider = llm_client.ProviderConfig(alias="alias", model="test-model")

    asyncio.run(
        llm_client.call_chat_completion(
            provider=provider,
            messages=[
                {"role": "user", "content": "hello"},
                FakeMessage(content="", tool_calls=tool_calls),
            ],
        )
    )

    assert captured_params[0]["messages"] == [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"item":"tea"}',
                    },
                }
            ],
        },
    ]


def test_call_chat_completion_forwards_temperature_for_explicit_model_config(
    monkeypatch,
):
    captured_params: list[dict[str, object]] = []

    async def fake_create(**kwargs):
        captured_params.append(kwargs)
        return FakeResponse(content="done")

    monkeypatch.setattr(
        llm_client,
        "_build_async_client",
        lambda provider, api_key: FakeAsyncClient(fake_create),
    )
    provider = llm_client.ProviderConfig(
        alias="alias",
        model="openai/o3",
        temperature=0.2,
    )

    asyncio.run(
        llm_client.call_chat_completion(
            provider=provider,
            messages=[{"role": "user", "content": "hello"}],
        )
    )

    assert captured_params[0]["temperature"] == 0.2


def test_call_chat_completion_preserves_model_config_reasoning_when_unset(
    monkeypatch,
):
    captured_params: list[dict[str, object]] = []

    async def fake_create(**kwargs):
        captured_params.append(kwargs)
        return FakeResponse(content="done")

    monkeypatch.setattr(
        llm_client,
        "_build_async_client",
        lambda provider, api_key: FakeAsyncClient(fake_create),
    )
    provider = llm_client.ProviderConfig(
        alias="qwen3.5-9b",
        model="qwen/qwen3.5-9b",
        provider="openai",
        extra_body={"reasoning": {"enabled": True}, "seed": 7},
    )

    asyncio.run(
        llm_client.call_chat_completion(
            provider=provider,
            messages=[{"role": "user", "content": "hello"}],
        )
    )

    assert captured_params[0]["extra_body"] == {
        "reasoning": {"enabled": True},
        "seed": 7,
    }


def test_provider_config_from_model_name_loads_logprobs_for_qwen():
    provider = llm_client.ProviderConfig.from_model_name("qwen3.5-9b")

    assert provider.logprobs is True
    assert provider.top_logprobs == 5


def test_call_chat_completion_forwards_logprobs_from_model_config(monkeypatch):
    captured_params: list[dict[str, object]] = []

    async def fake_create(**kwargs):
        captured_params.append(kwargs)
        return FakeResponse(content="done")

    monkeypatch.setattr(
        llm_client,
        "_build_async_client",
        lambda provider, api_key: FakeAsyncClient(fake_create),
    )
    provider = llm_client.ProviderConfig(
        alias="qwen3.5-9b",
        model="qwen/qwen3.5-9b",
        provider="openai",
        logprobs=True,
        top_logprobs=5,
    )

    asyncio.run(
        llm_client.call_chat_completion(
            provider=provider,
            messages=[{"role": "user", "content": "hello"}],
        )
    )

    assert captured_params[0]["logprobs"] is True
    assert captured_params[0]["top_logprobs"] == 5


def test_call_chat_completion_rejects_top_logprobs_without_logprobs():
    provider = llm_client.ProviderConfig(
        alias="alias",
        model="test-model",
        logprobs=False,
        top_logprobs=5,
    )

    with pytest.raises(ValueError, match="top_logprobs requires logprobs"):
        asyncio.run(
            llm_client.call_chat_completion(
                provider=provider,
                messages=[{"role": "user", "content": "hello"}],
            )
        )


def test_call_chat_completion_retries_only_transport_errors(monkeypatch):
    attempts = 0

    async def fake_create(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("temporary network failure")
        return FakeResponse(content="done", prompt_tokens=4, completion_tokens=2)

    monkeypatch.setattr(
        llm_client,
        "_build_async_client",
        lambda provider, api_key: FakeAsyncClient(fake_create),
    )
    provider = llm_client.ProviderConfig(
        alias="alias",
        model="test-model",
        max_retries=2,
        backoff=0.1,
    )

    response = asyncio.run(
        llm_client.call_chat_completion(
            provider=provider,
            messages=[{"role": "user", "content": "hello"}],
        )
    )

    assert response.choices[0].message.content == "done"
    assert attempts == 2


def test_call_chat_completion_reports_attempt_errors(monkeypatch):
    attempts = 0
    captured_errors: list[dict[str, object]] = []

    async def fake_create(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("temporary network failure")
        return FakeResponse(content="done", prompt_tokens=4, completion_tokens=2)

    async def capture_error(payload: dict[str, object]) -> None:
        captured_errors.append(payload)

    monkeypatch.setattr(
        llm_client,
        "_build_async_client",
        lambda provider, api_key: FakeAsyncClient(fake_create),
    )
    provider = llm_client.ProviderConfig(
        alias="alias",
        model="test-model",
        max_retries=2,
        backoff=0.1,
    )

    response = asyncio.run(
        llm_client.call_chat_completion(
            provider=provider,
            messages=[{"role": "user", "content": "hello"}],
            on_attempt_error=capture_error,
            error_context={"phase": "shopping"},
        )
    )

    assert response.choices[0].message.content == "done"
    assert len(captured_errors) == 1
    assert captured_errors[0]["attempt"] == 1
    assert captured_errors[0]["max_attempts"] == 2
    assert captured_errors[0]["will_retry"] is True
    assert captured_errors[0]["phase"] == "shopping"
    assert captured_errors[0]["error"]["type"] == "OSError"
    assert captured_errors[0]["error"]["message"] == "temporary network failure"
    assert "temporary network failure" in captured_errors[0]["error"]["traceback"]


def test_call_chat_completion_does_not_retry_context_exhaustion_provider_error(
    monkeypatch,
):
    attempts = 0
    captured_errors: list[dict[str, object]] = []
    overflow_error = _openai_status_error(
        openai.InternalServerError,
        status_code=500,
        message="This model's maximum context length is 40960 tokens, but your input exceeded that limit.",
        error_type="server_error",
    )

    async def fake_create(**kwargs):
        nonlocal attempts
        attempts += 1
        raise overflow_error

    async def capture_error(payload: dict[str, object]) -> None:
        captured_errors.append(payload)

    monkeypatch.setattr(
        llm_client,
        "_build_async_client",
        lambda provider, api_key: FakeAsyncClient(fake_create),
    )
    provider = llm_client.ProviderConfig(
        alias="alias",
        model="test-model",
        max_retries=3,
        backoff=0.1,
    )

    with pytest.raises(openai.InternalServerError):
        asyncio.run(
            llm_client.call_chat_completion(
                provider=provider,
                messages=[{"role": "user", "content": "hello"}],
                on_attempt_error=capture_error,
            )
        )

    assert attempts == 1
    assert len(captured_errors) == 1
    assert captured_errors[0]["will_retry"] is False
    assert captured_errors[0]["failure_subtype"] == "context_exhaustion"
    assert captured_errors[0]["error"]["type"] == "InternalServerError"


def test_classify_openrouter_input_length_range_error_as_context_exhaustion():
    request = httpx.Request(
        "POST",
        "https://openrouter.ai/api/v1/chat/completions",
    )
    body = {
        "message": "Provider returned error",
        "code": 400,
        "metadata": {
            "raw": '{"error":{"message":"<400> InternalError.Algo.InvalidParameter: Range of input length should be [1, 129024]","type":"invalid_request_error","param":null,"code":"invalid_parameter_error"}}',
            "provider_name": "Alibaba",
            "is_byok": False,
        },
    }
    response = httpx.Response(
        400,
        request=request,
        json={"error": body},
    )
    exc = openai.BadRequestError(
        "Error code: 400 - Provider returned error",
        response=response,
        body=body,
    )

    assert classify_exception_failure_subtype(exc) == "context_exhaustion"


def test_call_chat_completion_forwards_trace_id_when_langfuse_enabled(monkeypatch):
    captured_params: list[dict[str, object]] = []

    async def fake_create(**kwargs):
        captured_params.append(kwargs)
        return FakeResponse(content="done")

    monkeypatch.setattr(llm_client, "_langfuse_enabled", lambda: True)
    monkeypatch.setattr(
        llm_client,
        "_build_async_client",
        lambda provider, api_key: FakeAsyncClient(fake_create),
    )
    provider = llm_client.ProviderConfig(alias="alias", model="test-model")

    asyncio.run(
        llm_client.call_chat_completion(
            provider=provider,
            messages=[{"role": "user", "content": "hello"}],
            trace_id="shopping-inference-trace",
        )
    )

    assert captured_params[0]["trace_id"] == "shopping-inference-trace"
    assert captured_params[0]["name"] == "deepplanning-chat-completion"
    assert captured_params[0]["metadata"] == {
        "model_alias": "alias",
        "provider": "openai",
    }


def test_call_chat_completion_propagates_session_id_when_langfuse_enabled(monkeypatch):
    captured_params: list[dict[str, object]] = []
    propagation_calls: list[dict[str, object]] = []

    async def fake_create(**kwargs):
        captured_params.append(kwargs)
        return FakeResponse(content="done")

    class FakeScope:
        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    def fake_propagate_attributes(**kwargs):
        propagation_calls.append(kwargs)
        return FakeScope()

    monkeypatch.setattr(llm_client, "_langfuse_enabled", lambda: True)
    monkeypatch.setattr(
        llm_client,
        "_build_async_client",
        lambda provider, api_key: FakeAsyncClient(fake_create),
    )
    monkeypatch.setattr(
        llm_client,
        "propagate_attributes",
        fake_propagate_attributes,
    )
    provider = llm_client.ProviderConfig(alias="alias", model="test-model")

    asyncio.run(
        llm_client.call_chat_completion(
            provider=provider,
            messages=[{"role": "user", "content": "hello"}],
            session_id="experiment-session",
        )
    )

    assert propagation_calls == [{"session_id": "experiment-session"}]
    assert "session_id" not in captured_params[0]


def test_serialize_exception_includes_openai_error_details():
    request = httpx.Request(
        "POST",
        "https://example.com/v1/chat/completions",
        headers={"Authorization": "Bearer secret"},
    )
    response = httpx.Response(
        429,
        request=request,
        json={
            "error": {
                "message": "quota exceeded",
                "type": "rate_limit",
                "code": "rate_limit_exceeded",
            }
        },
        headers={"x-request-id": "req_123"},
    )
    body = {
        "error": {
            "message": "quota exceeded",
            "type": "rate_limit",
            "code": "rate_limit_exceeded",
        }
    }
    exc = openai.RateLimitError(
        "too many requests",
        response=response,
        body=body,
    )

    payload = serialize_exception(exc)

    assert payload["type"] == "RateLimitError"
    assert payload["status_code"] == 429
    assert payload["request_id"] == "req_123"
    assert payload["body"] == body
    assert payload["response"]["headers"]["x-request-id"] == "req_123"
    assert payload["response"]["reason_phrase"] == "Too Many Requests"


def test_serialize_exception_redacts_sensitive_request_headers():
    request = httpx.Request(
        "POST",
        "https://example.com/v1/chat/completions",
        headers={"Authorization": "Bearer secret", "X-Api-Key": "secret-key"},
    )

    class RequestBackedError(RuntimeError):
        def __init__(self) -> None:
            super().__init__("boom")
            self.request = request

    payload = serialize_exception(RequestBackedError())
    headers = payload["request"]["headers"]

    assert (
        headers.get("Authorization") == "<redacted>"
        or headers.get("authorization") == "<redacted>"
    )
    assert (
        headers.get("X-Api-Key") == "<redacted>"
        or headers.get("x-api-key") == "<redacted>"
    )


def test_flush_langfuse_noop_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    def fail_get_client():
        raise AssertionError("flush should not request a client without credentials")

    monkeypatch.setattr(llm_client, "get_client", fail_get_client)

    llm_client.flush_langfuse()


def test_flush_langfuse_flushes_active_client(monkeypatch):
    class FakeLangfuseClient:
        def __init__(self) -> None:
            self.flush_calls = 0

        def flush(self) -> None:
            self.flush_calls += 1

    client = FakeLangfuseClient()
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setattr(llm_client, "get_client", lambda: client)

    llm_client.flush_langfuse()

    assert client.flush_calls == 1
