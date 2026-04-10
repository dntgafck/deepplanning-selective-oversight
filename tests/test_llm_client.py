from __future__ import annotations

import os

import litellm

from llm import client as llm_client


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
