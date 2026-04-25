from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from experiment import build_system_config
import oversight as oversight_module
from oversight import ConversationState
from oversight.controllers import (
    AdaptiveRiskOversight,
    CheckpointReviewOversight,
    ContinuousReviewOversight,
    ExecutorOnlyOversight,
)
from oversight.factory import (
    build_oversight_controller,
    resolve_oversight_profile,
)


@pytest.mark.parametrize(
    (
        "system_name",
        "expected_profile",
        "expected_controller_name",
        "expected_runtime_dispatch",
        "expected_type",
    ),
    [
        (
            "A",
            "executor_only",
            "ExecutorOnlyOversight",
            "controller",
            ExecutorOnlyOversight,
        ),
        (
            "B",
            "continuous_review",
            "ContinuousReviewOversight",
            "controller",
            ContinuousReviewOversight,
        ),
        (
            "C1",
            "checkpoint_review",
            "CheckpointReviewOversight",
            "controller",
            CheckpointReviewOversight,
        ),
        (
            "C2",
            "adaptive_risk",
            "AdaptiveRiskOversight",
            "controller",
            AdaptiveRiskOversight,
        ),
        (
            "C2-nt",
            "adaptive_risk",
            "AdaptiveRiskOversight",
            "controller",
            AdaptiveRiskOversight,
        ),
    ],
)
def test_build_oversight_controller_resolves_semantic_profile_from_system_config(
    system_name: str,
    expected_profile: str,
    expected_controller_name: str,
    expected_runtime_dispatch: str,
    expected_type: type[object],
) -> None:
    system_config = build_system_config(system_name, executor_model="qwen3.5-9b")

    controller = build_oversight_controller(system_config)

    assert controller.profile == expected_profile
    assert controller.controller_name == expected_controller_name
    assert controller.runtime_dispatch == expected_runtime_dispatch
    assert isinstance(controller, expected_type)


def test_c2_and_c2_nt_share_adaptive_controller_with_config_only_differences() -> None:
    c2_config = build_system_config("C2", executor_model="qwen3.5-9b")
    c2_nt_config = build_system_config("C2-nt", executor_model="qwen3.5-9b")

    c2_controller = build_oversight_controller(c2_config)
    c2_nt_controller = build_oversight_controller(c2_nt_config)

    assert isinstance(c2_controller, AdaptiveRiskOversight)
    assert isinstance(c2_nt_controller, AdaptiveRiskOversight)
    assert type(c2_controller) is type(c2_nt_controller)
    assert c2_controller.profile == c2_nt_controller.profile == "adaptive_risk"
    assert c2_controller.active_hooks == c2_nt_controller.active_hooks
    assert c2_controller.runtime_dispatch == c2_nt_controller.runtime_dispatch == "controller"
    assert c2_config.overseer_thinking is True
    assert c2_nt_config.overseer_thinking is False


@pytest.mark.parametrize(
    ("enabled", "mode", "expected_profile"),
    [
        (False, "disabled", "executor_only"),
        (True, "always", "continuous_review"),
        (True, "checkpoint", "checkpoint_review"),
        (True, "adaptive", "adaptive_risk"),
    ],
)
def test_resolve_oversight_profile_falls_back_to_legacy_mode_when_profile_missing(
    enabled: bool,
    mode: str,
    expected_profile: str,
) -> None:
    system_config = SimpleNamespace(
        oversight_enabled=enabled,
        oversight_mode=mode,
    )

    assert resolve_oversight_profile(system_config) == expected_profile


def test_resolve_oversight_profile_rejects_unknown_profile() -> None:
    system_config = SimpleNamespace(
        oversight_enabled=True,
        oversight_mode="adaptive",
        oversight_profile="mystery_mode",
    )

    with pytest.raises(ValueError, match="Unknown oversight_profile"):
        resolve_oversight_profile(system_config)


def test_build_oversight_controller_derives_mode_alias_when_missing() -> None:
    system_config = SimpleNamespace(
        name="derived-profile",
        oversight_enabled=True,
        oversight_profile="adaptive_risk",
    )

    controller = build_oversight_controller(system_config)

    assert isinstance(controller, AdaptiveRiskOversight)
    assert controller.oversight_mode == "adaptive"


def test_legacy_mode_dispatch_impl_is_removed() -> None:
    assert not hasattr(oversight_module, "_evaluate_oversight_impl")


def test_evaluate_oversight_system_a_kwargs_return_executor_only_noop() -> None:
    state = ConversationState(
        task_id="shopping-task",
        domain="shopping",
        complexity=1,
        system_config_name="A",
    )
    system_config = build_system_config("A", executor_model="qwen3.5-9b")

    action = asyncio.run(
        oversight_module.evaluate_oversight(
            hook="pre_tool",
            state=state,
            system_config=system_config,
            phase="initial",
            task_query="buy a laptop under 1000",
            proposed_tool_calls=[
                {
                    "id": "call_1",
                    "name": "add_product_to_cart",
                    "arguments": '{"product_id":"1"}',
                }
            ],
            step_index=1,
        )
    )

    assert action.should_intervene is False
    assert action.overseer_invoked is False
    assert action.overseer_mode == "disabled"
    assert action.final_verification_result == "not_applicable"
    assert state.overseer_calls == 0


@pytest.mark.parametrize(
    ("system_name", "controller_type"),
    [
        ("B", ContinuousReviewOversight),
        ("C1", CheckpointReviewOversight),
        ("C2", AdaptiveRiskOversight),
        ("C2-nt", AdaptiveRiskOversight),
    ],
)
def test_evaluate_oversight_routes_class_backed_profiles_through_controller_instances(
    monkeypatch: pytest.MonkeyPatch,
    system_name: str,
    controller_type: type[object],
) -> None:
    recorded: dict[str, str] = {}

    async def fake_evaluate(self, context):
        recorded["profile"] = self.profile
        return oversight_module.OversightAction(
            should_intervene=False,
            trigger_type="controller_routed",
            intervention_type="approve",
        )

    monkeypatch.setattr(controller_type, "evaluate", fake_evaluate)

    state = ConversationState(
        task_id="shopping-task",
        domain="shopping",
        complexity=1,
        system_config_name=system_name,
    )
    system_config = build_system_config(system_name, executor_model="qwen3.5-9b")

    action = asyncio.run(
        oversight_module.evaluate_oversight(
            hook="midpoint",
            state=state,
            system_config=system_config,
            phase="initial",
            task_query="buy a laptop under 1000",
            step_index=1,
        )
    )

    assert recorded["profile"] == system_config.oversight_profile
    assert action.trigger_type == "controller_routed"


def test_evaluate_oversight_positional_adapter_remains_noop_for_adaptive_profiles() -> None:
    state = ConversationState(
        task_id="travel-task",
        domain="travel",
        complexity=1,
        system_config_name="C2",
    )
    system_config = build_system_config("C2", executor_model="qwen3.5-9b")

    action = oversight_module.evaluate_oversight(object(), [], state, system_config)

    assert action.should_intervene is False
    assert action.overseer_invoked is False
    assert action.overseer_mode == "thinking"
    assert action.final_verification_result == "not_applicable"
