from __future__ import annotations

from types import SimpleNamespace

import pytest

import oversight as oversight_module
from experiment import build_system_config
from oversight import (
    ConversationState,
    _oversight_active_for_hook,
    _oversight_active_for_task,
)


def _state(domain: str = "shopping") -> ConversationState:
    return ConversationState(
        task_id="T_test",
        domain=domain,
        complexity=1,
        system_config_name="dispatch-test",
    )


def test_legacy_hook_table_is_not_exported_anymore() -> None:
    assert not hasattr(oversight_module, "_HOOKS_BY_MODE")


@pytest.mark.parametrize(
    ("system_name", "hook", "expected"),
    [
        ("A", "pre_tool", False),
        ("A", "post_tool", False),
        ("A", "midpoint", False),
        ("A", "final", False),
        ("B", "pre_tool", True),
        ("B", "post_tool", True),
        ("B", "midpoint", True),
        ("B", "final", True),
        ("C1", "pre_tool", False),
        ("C1", "post_tool", False),
        ("C1", "midpoint", True),
        ("C1", "final", True),
    ],
)
def test_class_backed_profiles_use_controller_hook_activation(
    system_name: str,
    hook: str,
    expected: bool,
) -> None:
    system_config = build_system_config(system_name, executor_model="qwen3.5-9b")

    assert (
        _oversight_active_for_hook(
            state=_state(),
            system_config=system_config,
            hook=hook,
        )
        is expected
    )


def test_adaptive_profile_uses_controller_hook_activation() -> None:
    system_config = build_system_config("C2", executor_model="qwen3.5-9b")

    assert (
        _oversight_active_for_hook(
            state=_state(),
            system_config=system_config,
            hook="pre_tool",
        )
        is True
    )
    assert (
        _oversight_active_for_hook(
            state=_state(),
            system_config=system_config,
            hook="final",
        )
        is True
    )


def test_hook_activation_respects_oversight_enabled_false() -> None:
    system_config = SimpleNamespace(
        oversight_enabled=False,
        oversight_mode="always",
        oversight_profile="continuous_review",
    )

    assert (
        _oversight_active_for_hook(
            state=_state(),
            system_config=system_config,
            hook="pre_tool",
        )
        is False
    )


def test_hook_activation_is_shopping_only() -> None:
    system_config = build_system_config("B", executor_model="qwen3.5-9b")

    assert (
        _oversight_active_for_hook(
            state=_state(domain="travel"),
            system_config=system_config,
            hook="final",
        )
        is False
    )


def test_hook_activation_rejects_unknown_hook() -> None:
    system_config = build_system_config("C2", executor_model="qwen3.5-9b")

    with pytest.raises(ValueError, match="Unknown oversight hook"):
        _oversight_active_for_hook(
            state=_state(),
            system_config=system_config,
            hook="not_a_hook",
        )


def test_task_activation_uses_controller_profiles_for_class_backed_systems() -> None:
    assert not _oversight_active_for_task(
        state=_state(),
        system_config=build_system_config("A", executor_model="qwen3.5-9b"),
    )
    assert _oversight_active_for_task(
        state=_state(),
        system_config=build_system_config("B", executor_model="qwen3.5-9b"),
    )
    assert _oversight_active_for_task(
        state=_state(),
        system_config=build_system_config("C1", executor_model="qwen3.5-9b"),
    )


def test_task_activation_uses_controller_profiles_for_adaptive_systems() -> None:
    assert _oversight_active_for_task(
        state=_state(),
        system_config=build_system_config("C2", executor_model="qwen3.5-9b"),
    )
