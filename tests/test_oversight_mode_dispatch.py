from __future__ import annotations

from types import SimpleNamespace

import pytest

from oversight import (
    _HOOKS_BY_MODE,
    _oversight_active_for_hook,
    _oversight_active_for_task,
    ConversationState,
)


def _state(domain: str = "shopping") -> ConversationState:
    return ConversationState(
        task_id="T_test",
        domain=domain,
        complexity=1,
        system_config_name="dispatch-test",
    )


def _config(*, enabled: bool, mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        oversight_enabled=enabled,
        oversight_mode=mode,
    )


@pytest.mark.parametrize(
    ("mode", "hook", "expected"),
    [
        ("disabled", "pre_tool", False),
        ("disabled", "post_tool", False),
        ("disabled", "midpoint", False),
        ("disabled", "final", False),
        ("always", "pre_tool", True),
        ("always", "post_tool", True),
        ("always", "midpoint", True),
        ("always", "final", True),
        ("checkpoint", "pre_tool", False),
        ("checkpoint", "post_tool", False),
        ("checkpoint", "midpoint", True),
        ("checkpoint", "final", True),
        ("adaptive", "pre_tool", True),
        ("adaptive", "post_tool", True),
        ("adaptive", "midpoint", True),
        ("adaptive", "final", True),
    ],
)
def test_hook_activation_matrix(mode: str, hook: str, expected: bool) -> None:
    assert (
        _oversight_active_for_hook(
            state=_state(),
            system_config=_config(enabled=True, mode=mode),
            hook=hook,
        )
        is expected
    )


def test_hook_activation_respects_oversight_enabled_false() -> None:
    assert (
        _oversight_active_for_hook(
            state=_state(),
            system_config=_config(enabled=False, mode="always"),
            hook="pre_tool",
        )
        is False
    )


def test_hook_activation_is_shopping_only() -> None:
    assert (
        _oversight_active_for_hook(
            state=_state(domain="travel"),
            system_config=_config(enabled=True, mode="always"),
            hook="final",
        )
        is False
    )


def test_hook_activation_rejects_unknown_hook() -> None:
    with pytest.raises(ValueError, match="Unknown oversight hook"):
        _oversight_active_for_hook(
            state=_state(),
            system_config=_config(enabled=True, mode="adaptive"),
            hook="not_a_hook",
        )


def test_task_activation_true_for_enabled_modes_only() -> None:
    for mode in ("always", "checkpoint", "adaptive"):
        assert _oversight_active_for_task(
            state=_state(),
            system_config=_config(enabled=True, mode=mode),
        )
    assert not _oversight_active_for_task(
        state=_state(),
        system_config=_config(enabled=True, mode="disabled"),
    )
    assert not _oversight_active_for_task(
        state=_state(),
        system_config=_config(enabled=False, mode="adaptive"),
    )


def test_hooks_by_mode_is_canonical() -> None:
    assert _HOOKS_BY_MODE["disabled"] == frozenset()
    assert _HOOKS_BY_MODE["always"] == frozenset(
        {"pre_tool", "post_tool", "midpoint", "final"}
    )
    assert _HOOKS_BY_MODE["checkpoint"] == frozenset({"midpoint", "final"})
    assert _HOOKS_BY_MODE["adaptive"] == frozenset(
        {"pre_tool", "post_tool", "midpoint", "final"}
    )
