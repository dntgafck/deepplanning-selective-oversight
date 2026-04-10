from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .state import ConversationState


@dataclass(slots=True)
class OversightAction:
    should_intervene: bool = False
    trigger_reason: str | None = None
    intervention_type: str | None = None
    correction: str | None = None


def evaluate_oversight(
    response: Any,
    messages: list[Any],
    state: ConversationState,
    config: Any,
) -> OversightAction:
    return OversightAction(should_intervene=False)


async def apply_intervention(
    action: OversightAction,
    original_response: Any,
    messages: list[Any],
    state: ConversationState,
    config: Any,
) -> Any:
    return original_response


__all__ = [
    "ConversationState",
    "OversightAction",
    "apply_intervention",
    "evaluate_oversight",
]
