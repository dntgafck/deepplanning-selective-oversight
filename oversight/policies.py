from __future__ import annotations

from enum import Enum
from typing import Any

from .state import ConversationState, _hash_arguments


class H1Outcome(str, Enum):
    APPROVE_CONTINUE = "approve_continue"
    APPROVE_WITH_NUDGE = "approve_with_nudge"
    HARD_BLOCK = "hard_block"
    FORCED_APPROVE = "forced_approve"


def _tool_reversibility(tool_name: str, system_config: Any) -> str:
    irreversible = tuple(getattr(system_config, "irreversible_tools", ()) or ())
    if tool_name in irreversible:
        return "irreversible"
    mutating = tuple(getattr(system_config, "mutating_tools", ()) or ())
    if tool_name in mutating:
        return "reversible"
    return "unknown"


def compute_h1_outcome(
    *,
    action: Any,
    tool_name: str,
    arguments: Any,
    state: ConversationState,
    system_config: Any,
) -> H1Outcome:
    mode = str(getattr(system_config, "block_on_mutation_mode", "auto"))
    if mode == "never":
        if action.intervention_type == "provide_guidance":
            return H1Outcome.APPROVE_WITH_NUDGE
        return H1Outcome.APPROVE_CONTINUE

    max_streak = int(getattr(system_config, "max_consecutive_pre_tool_blocks", 5))
    if (
        max_streak > 0
        and state.consecutive_pre_tool_blocks >= max_streak
        and action.intervention_type == "provide_guidance"
    ):
        return H1Outcome.FORCED_APPROVE

    if mode == "always":
        if action.intervention_type == "provide_guidance":
            args_hash_key = _hash_arguments(arguments)
            prior_blocks = state.blocked_mutation_counts.get(
                (tool_name, args_hash_key), 0
            )
            max_blocks = max(
                int(getattr(system_config, "max_hard_blocks_per_args", 2)),
                1,
            )
            if prior_blocks >= max_blocks:
                return H1Outcome.FORCED_APPROVE
            return H1Outcome.HARD_BLOCK
        return H1Outcome.APPROVE_CONTINUE

    if action.intervention_type == "approve":
        return H1Outcome.APPROVE_CONTINUE

    args_hash_key = _hash_arguments(arguments)
    prior_blocks = state.blocked_mutation_counts.get((tool_name, args_hash_key), 0)
    max_blocks = max(int(getattr(system_config, "max_hard_blocks_per_args", 2)), 1)
    if prior_blocks >= max_blocks:
        return H1Outcome.FORCED_APPROVE

    has_cited_violation = bool(
        action.violated_contract_ids or action.unmet_checklist_keys
    )
    confidence = str(getattr(action, "violation_confidence", "low") or "low").lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "low"
    strong_confidence = confidence in {"medium", "high"}
    tool_reversibility = _tool_reversibility(tool_name, system_config)
    not_reversible = tool_reversibility in {"irreversible", "unknown"}
    require_cited = bool(
        getattr(system_config, "require_cited_violation_for_block", True)
    )
    block = (
        (not require_cited or has_cited_violation)
        and strong_confidence
        and not_reversible
    )
    return H1Outcome.HARD_BLOCK if block else H1Outcome.APPROVE_WITH_NUDGE


def increment_retry_and_check_cap(state: ConversationState, system_config: Any) -> bool:
    state.final_verification_retry_count += 1
    if state.final_verification_retry_count > int(
        getattr(system_config, "final_repair_retry_cap", 2)
    ):
        state.final_verification_result = "retry_cap_exhausted"
        return True
    return False


__all__ = [
    "H1Outcome",
    "compute_h1_outcome",
    "increment_retry_and_check_cap",
]
