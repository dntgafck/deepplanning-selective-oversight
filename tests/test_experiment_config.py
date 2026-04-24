from __future__ import annotations

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
