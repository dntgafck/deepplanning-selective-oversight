from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from .clients import make_noop_action
from .state import ConversationState

if TYPE_CHECKING:
    from . import OversightAction

OversightHook: TypeAlias = Literal["pre_tool", "post_tool", "midpoint", "final"]
OversightPhase: TypeAlias = Literal["initial", "cart_check"]
VALID_OVERSIGHT_HOOKS = frozenset({"pre_tool", "post_tool", "midpoint", "final"})


@dataclass(frozen=True, slots=True)
class OversightContext:
    hook: OversightHook
    state: ConversationState
    system_config: Any
    phase: OversightPhase
    task_query: str
    proposed_tool_calls: list[dict[str, Any]] | None = None
    latest_tool_result: Any | None = None
    draft_final_answer: str | None = None
    step_index: int = 0
    tool_index: int | None = None


class OversightController(ABC):
    profile: str
    controller_name: str
    runtime_dispatch: str = "controller"
    system_name: str
    oversight_mode: str
    active_hooks: frozenset[OversightHook] = frozenset()

    def __init__(self, *, system_name: str, oversight_mode: str) -> None:
        self.system_name = system_name
        self.oversight_mode = oversight_mode

    def is_active_for_hook(
        self,
        *,
        state: ConversationState,
        system_config: Any,
        hook: str,
    ) -> bool:
        if hook not in VALID_OVERSIGHT_HOOKS:
            raise ValueError(f"Unknown oversight hook: {hook!r}")
        if not bool(getattr(system_config, "oversight_enabled", False)):
            return False
        if state.domain != "shopping":
            return False
        return hook in self.active_hooks

    def is_active_for_task(
        self,
        *,
        state: ConversationState,
        system_config: Any,
    ) -> bool:
        if not bool(getattr(system_config, "oversight_enabled", False)):
            return False
        if state.domain != "shopping":
            return False
        return bool(self.active_hooks)

    @abstractmethod
    async def evaluate(self, context: OversightContext) -> OversightAction:
        """Return the oversight decision for the current evaluation context."""

    def noop_action(
        self,
        *,
        system_config: Any,
        final_result: str = "not_applicable",
    ) -> OversightAction:
        from . import OversightAction

        return make_noop_action(
            action_factory=OversightAction,
            system_config=system_config,
            final_result=final_result,
        )
