from __future__ import annotations

import pytest

from experiment import build_system_config
from experiment.config import provider_identity_payload
from llm import ProviderConfig


def test_provider_identity_payload_infers_reasoning_from_deepseek_thinking_shape():
    payload = provider_identity_payload(
        ProviderConfig(
            alias="deepseek-v4-flash",
            model="deepseek-v4-flash",
            provider="openai",
            extra_body={"thinking": {"type": "enabled"}},
        )
    )

    assert payload is not None
    assert payload["sampling"]["reasoning_enabled"] is True


def test_build_system_config_defaults_streak_cap_to_five():
    config = build_system_config("C2", executor_model="qwen3.5-9b")

    assert config.max_consecutive_pre_tool_blocks == 5


def test_build_system_config_reads_explicit_b_streak_cap_from_yaml():
    config = build_system_config("B", executor_model="qwen3.5-9b")

    assert config.max_consecutive_pre_tool_blocks == 5


@pytest.mark.parametrize(
    ("system_name", "expected_profile", "expected_mode"),
    [
        ("A", "executor_only", "disabled"),
        ("B", "continuous_review", "always"),
        ("C1", "checkpoint_review", "checkpoint"),
        ("C2", "adaptive_risk", "adaptive"),
        ("C2-noretry", "adaptive_risk", "adaptive"),
        ("C2-nt", "adaptive_risk", "adaptive"),
    ],
)
def test_build_system_config_reads_oversight_profile_from_yaml(
    system_name: str,
    expected_profile: str,
    expected_mode: str,
) -> None:
    config = build_system_config(system_name, executor_model="qwen3.5-9b")

    assert config.oversight_profile == expected_profile
    assert config.oversight_mode == expected_mode


def test_build_system_config_rejects_mismatched_profile_and_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "experiment.config.load_system_defaults",
        lambda _system_name: {
            "name": "Mismatch",
            "oversight_enabled": True,
            "oversight_mode": "always",
            "oversight_profile": "adaptive_risk",
        },
    )

    with pytest.raises(ValueError, match="does not match oversight_profile"):
        build_system_config("Mismatch", executor_model="qwen3.5-9b")


def test_build_system_config_reads_oversight_domains():
    config = build_system_config("C2", executor_model="qwen3.5-9b")

    assert config.oversight_domains == ("shopping",)


def test_build_system_config_loads_frozen_shopping_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "experiment.config.load_system_defaults",
        lambda _system_name: {
            "name": "C2-test",
            "oversight_enabled": True,
            "oversight_profile": "adaptive_risk",
            "loop_similarity_threshold": 0.01,
            "loop_window": 1,
            "loop_repeat_count": 1,
            "coverage_threshold": 0.01,
            "oversight_domains": ["shopping"],
        },
    )

    config = build_system_config("C2-test", executor_model="qwen3.5-9b")

    assert config.loop_similarity_threshold == 0.92
    assert config.loop_window == 5
    assert config.loop_repeat_count == 3
    assert config.coverage_threshold == 0.50
