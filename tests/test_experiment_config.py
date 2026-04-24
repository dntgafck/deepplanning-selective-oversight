from __future__ import annotations

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
