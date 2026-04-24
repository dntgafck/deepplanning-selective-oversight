from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from scripts import run_model_completion as model_completion_script

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def model_dump(self, mode: str = "python") -> dict[str, object]:
        assert mode == "json"
        return self.payload


def _fake_response(content: str = "", tool_calls: list[object] | None = None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls)
            )
        ]
    )


def test_run_model_completion_uses_model_alias_and_prompt(monkeypatch):
    captured: dict[str, object] = {}

    class FakeProviderConfig:
        @classmethod
        def from_model_name(cls, model_name: str):
            captured["model_name"] = model_name
            return SimpleNamespace(alias=model_name, model="resolved-model")

    async def fake_call_chat_completion(*, provider, messages, validate_nonempty):
        captured["provider"] = provider
        captured["messages"] = messages
        captured["validate_nonempty"] = validate_nonempty
        return _fake_response("smoke ok")

    monkeypatch.setattr(model_completion_script, "ProviderConfig", FakeProviderConfig)
    monkeypatch.setattr(
        model_completion_script, "call_chat_completion", fake_call_chat_completion
    )
    monkeypatch.setattr(model_completion_script, "flush_langfuse", lambda: None)

    output = model_completion_script.run("qwen-plus", "Say", "hello")
    payload = model_completion_script.json.loads(output)

    assert payload["choices"][0]["message"]["content"] == "smoke ok"
    assert captured["model_name"] == "qwen-plus"
    assert captured["provider"] == SimpleNamespace(
        alias="qwen-plus", model="resolved-model"
    )
    assert captured["messages"] == [{"role": "user", "content": "Say hello"}]
    assert captured["validate_nonempty"] is True


def test_run_model_completion_uses_default_prompt_when_prompt_missing(monkeypatch):
    captured: dict[str, object] = {}

    class FakeProviderConfig:
        @classmethod
        def from_model_name(cls, model_name: str):
            return SimpleNamespace(alias=model_name, model="resolved-model")

    async def fake_call_chat_completion(*, provider, messages, validate_nonempty):
        captured["messages"] = messages
        return _fake_response("smoke ok")

    monkeypatch.setattr(model_completion_script, "ProviderConfig", FakeProviderConfig)
    monkeypatch.setattr(
        model_completion_script, "call_chat_completion", fake_call_chat_completion
    )
    monkeypatch.setattr(model_completion_script, "flush_langfuse", lambda: None)

    output = model_completion_script.run("qwen-plus")
    payload = model_completion_script.json.loads(output)

    assert payload["choices"][0]["message"]["content"] == "smoke ok"
    assert captured["messages"] == [
        {"role": "user", "content": model_completion_script.DEFAULT_PROMPT}
    ]


def test_render_response_serializes_full_model_dump_payload():
    rendered = model_completion_script.render_response(
        FakeResponse(
            {
                "id": "resp_1",
                "model": "resolved-model",
                "choices": [{"message": {"role": "assistant", "content": "smoke ok"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            }
        )
    )
    payload = model_completion_script.json.loads(rendered)

    assert payload["id"] == "resp_1"
    assert payload["model"] == "resolved-model"
    assert payload["usage"]["prompt_tokens"] == 7


def test_render_response_falls_back_to_object_serialization():
    tool_call = SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name="lookup", arguments='{"item":"tea"}'),
    )

    rendered = model_completion_script.render_response(
        _fake_response(tool_calls=[tool_call])
    )
    payload = model_completion_script.json.loads(rendered)

    assert payload["choices"][0]["message"]["tool_calls"][0]["id"] == "call_1"
    assert (
        payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
        == "lookup"
    )


def test_run_model_completion_script_requires_model_alias_without_import_failure():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_model_completion.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode != 0
    assert "ModuleNotFoundError" not in combined_output
    assert (
        "Model smoke runner requires a model alias as the first argument."
        in combined_output
    )
