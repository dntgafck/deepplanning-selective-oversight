from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm import call_chat_completion, estimate_call_cost

from .base import OversightContext, OversightController
from .clients import make_noop_action as _make_noop_action
from .controllers.review import _cart_read_is_stale
from .factory import build_oversight_controller
from .notices import render_notice_from_action as _render_notice_from_action
from .parsing import parse_final_verifier_json, parse_runtime_overseer_json
from .policies import (
    H1Outcome,
    compute_h1_outcome,
)
from .state import ConversationState


@dataclass(slots=True)
class OversightAction:
    should_intervene: bool = False
    trigger_type: str | None = None
    trigger_reason: str | None = None
    intervention_type: str | None = None
    block_current_tool: bool = False
    guidance_lines: list[str] = field(default_factory=list)
    corrected_observation: str | None = None
    notice_text: str | None = None
    violated_contract_ids: list[str] = field(default_factory=list)
    unmet_checklist_keys: list[str] = field(default_factory=list)
    violation_confidence: str = "low"
    overseer_invoked: bool = False
    overseer_mode: str = "disabled"
    overseer_input_tokens: int = 0
    overseer_output_tokens: int = 0
    overseer_cost: float | None = None
    decision_summary: str | None = None
    final_verification_result: str = "not_applicable"
    loop_signature: str | None = None
    coverage_status: dict[str, Any] | None = None
    raw_overseer_text: str | None = None
    parsed_payload: dict[str, Any] | None = None
    notice_rendered: bool = False
    notice_source: str | None = None
    fallback_guidance_used: bool = False
    blocked_tool_name: str | None = None
    blocked_tool_arguments: Any | None = None
    blocked_tool_arguments_normalized: str | None = None
    blocked_mutation_repeat_count: int = 0
    terminate_phase: bool = False
    termination_reason: str | None = None
    h1_outcome: str | None = None


def _noop_action(
    *, system_config: Any, final_result: str = "not_applicable"
) -> OversightAction:
    return _make_noop_action(
        action_factory=OversightAction,
        system_config=system_config,
        final_result=final_result,
    )


def _controller_for(system_config: Any) -> OversightController:
    return build_oversight_controller(system_config)


def _oversight_active_for_hook(
    *,
    state: ConversationState,
    system_config: Any,
    hook: str,
) -> bool:
    """Return True iff the given hook should invoke oversight for this profile."""
    controller = _controller_for(system_config)
    return controller.is_active_for_hook(
        state=state,
        system_config=system_config,
        hook=hook,
    )


def _oversight_active_for_task(
    *,
    state: ConversationState,
    system_config: Any,
) -> bool:
    """Return True iff the task profile enables at least one oversight hook."""
    controller = _controller_for(system_config)
    return controller.is_active_for_task(
        state=state,
        system_config=system_config,
    )


def evaluate_oversight(*args: Any, **kwargs: Any) -> Any:
    if args and len(args) == 4 and not kwargs:
        _, _, state, system_config = args
        return _noop_action(system_config=system_config)
    controller = _controller_for(kwargs["system_config"])
    return controller.evaluate(OversightContext(**kwargs))


async def _apply_intervention_impl(
    *, state: ConversationState, action: OversightAction
) -> None:
    notice = _render_notice_from_action(action)
    if notice is not None:
        if (
            action.trigger_type == "final_checkpoint"
            or state.pending_executor_notice is None
        ):
            state.pending_executor_notice = notice


async def _apply_intervention_compat(
    *,
    original_response: Any | None,
    state: ConversationState,
    action: OversightAction,
) -> Any:
    await _apply_intervention_impl(state=state, action=action)
    return original_response


def apply_intervention(*args: Any, **kwargs: Any) -> Any:
    if "original_response" in kwargs:
        return _apply_intervention_compat(
            original_response=kwargs.get("original_response"),
            state=kwargs["state"],
            action=kwargs["action"],
        )
    return _apply_intervention_impl(**kwargs)


__all__ = [
    "_cart_read_is_stale",
    "_oversight_active_for_hook",
    "_oversight_active_for_task",
    "ConversationState",
    "H1Outcome",
    "OversightAction",
    "apply_intervention",
    "compute_h1_outcome",
    "evaluate_oversight",
    "parse_final_verifier_json",
    "parse_runtime_overseer_json",
]
