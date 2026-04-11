from __future__ import annotations

import asyncio
import os

import litellm

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


def test_configure_litellm_callbacks_uses_langfuse_otel(monkeypatch):
    original_callbacks = list(getattr(litellm, "callbacks", []) or [])
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    monkeypatch.delenv("LANGFUSE_OTEL_HOST", raising=False)
    monkeypatch.setattr(litellm, "callbacks", ["langfuse"], raising=True)

    try:
        llm_client._configure_litellm_callbacks()

        assert "langfuse_otel" in litellm.callbacks
        assert "langfuse" not in litellm.callbacks
        assert os.environ["LANGFUSE_OTEL_HOST"] == "https://cloud.langfuse.com"
    finally:
        litellm.callbacks = original_callbacks


def test_configure_litellm_callbacks_noop_without_keys(monkeypatch):
    original_callbacks = list(getattr(litellm, "callbacks", []) or [])
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setattr(litellm, "callbacks", ["existing"], raising=True)

    try:
        llm_client._configure_litellm_callbacks()
        assert litellm.callbacks == ["existing"]
    finally:
        litellm.callbacks = original_callbacks


def test_call_chat_completion_returns_empty_no_tool_response_without_retry(
    monkeypatch,
):
    attempts = 0

    async def fake_acompletion(**kwargs):
        nonlocal attempts
        attempts += 1
        return FakeResponse(content="", prompt_tokens=8, completion_tokens=0)

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion, raising=True)
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

    async def fake_acompletion(**kwargs):
        return FakeResponse(content="", tool_calls=tool_calls)

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion, raising=True)
    provider = llm_client.ProviderConfig(alias="alias", model="test-model")

    response = asyncio.run(
        llm_client.call_chat_completion(
            provider=provider,
            messages=[{"role": "user", "content": "hello"}],
        )
    )

    assert response.choices[0].message.tool_calls == tool_calls


def test_call_chat_completion_retries_only_transport_errors(monkeypatch):
    attempts = 0

    async def fake_acompletion(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("temporary network failure")
        return FakeResponse(content="done", prompt_tokens=4, completion_tokens=2)

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion, raising=True)
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


def test_call_chat_completion_forces_reasoning_override(monkeypatch):
    captured_params: list[dict[str, object]] = []

    async def fake_acompletion(**kwargs):
        captured_params.append(kwargs)
        return FakeResponse(content="done")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion, raising=True)
    provider = llm_client.ProviderConfig(
        alias="qwen3-14b",
        model="openai/qwen3-14b",
        provider="openai",
        extra_body={"reasoning": {"enabled": True}, "seed": 7},
    )

    asyncio.run(
        llm_client.call_chat_completion(
            provider=provider,
            messages=[{"role": "user", "content": "hello"}],
            reasoning_enabled=False,
        )
    )

    assert captured_params[0]["extra_body"] == {
        "reasoning": {"enabled": False},
        "seed": 7,
    }
    assert captured_params[0]["custom_llm_provider"] == "openai"
