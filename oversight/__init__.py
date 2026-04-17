from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from llm import call_chat_completion, estimate_call_cost

from .contracts import execution_contract_to_dict, task_checklist_to_dict
from .state import ConversationState
from .triggers import (
    build_authoritative_state_snapshot,
    classify_mutating_tool,
    compute_coverage_status,
    detect_loop,
    detect_tool_error,
    normalize_arguments,
    render_transient_notice,
)

P2_SYSTEM_PROMPT = """You are a selective execution overseer.

Evaluate the current executor step against:
1. the execution contract,
2. the task checklist,
3. the trigger-local trajectory state.

You are not the primary planner.
Do not solve the full task.

Your output has two purposes:
- diagnose whether the proposed action violates the contract or checklist, and
- suggest short corrective guidance if it does.

You do not decide whether the action is blocked or allowed. The runtime makes
that decision based on objective fields in your output.

Return approve for ambiguous or insufficient-evidence cases. Do not assert a
violation unless you can name the specific contract rule ID or unmet checklist
key that the proposed action contradicts.

Use only the allowed actions. Keep any corrective guidance short, specific, and
action-oriented. Output valid JSON only.

Return exactly one JSON object with this shape:
{
  "action": "approve" | "provide_guidance" | "correct_observation",
  "decision_summary": "string, one sentence",
  "violation_evidence": {
    "violated_contract_ids": ["string"],
    "unmet_checklist_keys": ["string"],
    "confidence": "low" | "medium" | "high"
  },
  "guidance_lines": ["string"],
  "corrected_observation": "string|null",
}

Rules:
- if action == "approve", violation_evidence.violated_contract_ids and
  unmet_checklist_keys MUST both be empty arrays
- if action == "provide_guidance" you MAY leave violation_evidence empty
  (a soft nudge); in that case set confidence to "low"
- if action == "provide_guidance", include usable corrective content in either
  guidance_lines or corrected_observation; never return provide_guidance with
  both empty
- "Insufficient evidence of correctness" is never grounds for intervention on
  a reversible mutation. Return approve in that case."""

P3_SYSTEM_PROMPT = """You are the final execution verifier.

The authoritative state snapshot is the source of truth.
Approve finalization only if the current state satisfies the execution contract and the task checklist.
Do not solve the task.
If finalization should be delayed, state the specific blockers and the next required executor actions.
Output valid JSON only."""

DEFAULT_FINAL_NOTICE = (
    "Call get_cart_info before finalizing. The cart state is the source of truth."
)


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
    action: "OversightAction",
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
    if mode == "always":
        if action.intervention_type == "provide_guidance":
            from .state import _hash_arguments

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

    from .state import _hash_arguments

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


def _overseer_mode(system_config: Any) -> str:
    provider = getattr(system_config, "overseer_provider", None)
    if provider is None:
        return "disabled"
    return (
        "thinking"
        if bool(getattr(system_config, "overseer_thinking", False))
        else "non-thinking"
    )


def _noop_action(
    *, system_config: Any, final_result: str = "not_applicable"
) -> OversightAction:
    return OversightAction(
        should_intervene=False,
        overseer_mode=_overseer_mode(system_config),
        final_verification_result=final_result,
    )


def _strict_json_object(payload: str | dict[str, Any]) -> dict[str, Any]:
    data = json.loads(payload) if isinstance(payload, str) else payload
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object payload")
    return data


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise ValueError("Expected a list of strings")
    return [str(item) for item in value if str(item).strip()]


def _humanize_identifier(value: str) -> str:
    return " ".join(
        token for token in value.replace(":", " ").replace("_", " ").split() if token
    )


def parse_runtime_overseer_json(payload: str | dict[str, Any]) -> dict[str, Any]:
    data = _strict_json_object(payload)
    action = str(data.get("action", "approve"))
    if action not in {"approve", "provide_guidance", "correct_observation"}:
        raise ValueError(f"Unsupported runtime overseer action: {action}")
    guidance_lines = _coerce_string_list(data.get("guidance_lines", []))
    corrected_observation = (
        None
        if data.get("corrected_observation") is None
        else str(data.get("corrected_observation")).strip() or None
    )
    evidence = data.get("violation_evidence") or {}
    if not isinstance(evidence, dict):
        evidence = {}
    violated_contract_ids = _coerce_string_list(
        evidence.get("violated_contract_ids", data.get("violated_contract_ids", []))
    )
    unmet_checklist_keys = _coerce_string_list(
        evidence.get("unmet_checklist_keys", data.get("unmet_checklist_keys", []))
    )
    confidence = str(evidence.get("confidence", "low")).strip().lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "low"
    if action == "approve":
        violated_contract_ids = []
        unmet_checklist_keys = []
        confidence = "low"
    return {
        "action": action,
        "decision_summary": str(data.get("decision_summary", "") or ""),
        "block_current_tool": bool(data.get("block_current_tool", False)),
        "guidance_lines": guidance_lines,
        "corrected_observation": corrected_observation,
        "violated_contract_ids": violated_contract_ids,
        "unmet_checklist_keys": unmet_checklist_keys,
        "violation_confidence": confidence,
        "missing_corrective_content": (
            action == "provide_guidance"
            and not guidance_lines
            and not corrected_observation
        ),
    }


def parse_final_verifier_json(payload: str | dict[str, Any]) -> dict[str, Any]:
    data = _strict_json_object(payload)
    action = str(data.get("action", "run_verification"))
    if action not in {"approve", "run_verification"}:
        raise ValueError(f"Unsupported final verifier action: {action}")
    blockers = data.get("blockers", [])
    if not isinstance(blockers, list):
        raise ValueError("Final verifier blockers must be a list")
    return {
        "action": action,
        "pass": bool(data.get("pass", False)),
        "decision_summary": str(data.get("decision_summary", "") or ""),
        "blockers": [dict(item) for item in blockers],
        "next_step_notice_lines": [
            str(line) for line in data.get("next_step_notice_lines", [])
        ],
        "violated_contract_ids": [
            str(item) for item in data.get("violated_contract_ids", [])
        ],
        "unmet_checklist_keys": [
            str(item) for item in data.get("unmet_checklist_keys", [])
        ],
    }


def _synthesize_guidance_lines(
    *,
    trigger_type: str,
    trigger_reason: str,
    trigger_evidence: dict[str, Any],
    violated_contract_ids: list[str],
    unmet_checklist_keys: list[str],
) -> list[str]:
    guidance_lines: list[str] = []
    tool_name = str(trigger_evidence.get("tool_name") or "").strip()

    for contract_id in violated_contract_ids[:2]:
        guidance_lines.append(
            f"Re-check contract constraint: {_humanize_identifier(contract_id)}."
        )
    for checklist_key in unmet_checklist_keys[:2]:
        guidance_lines.append(
            f"Re-check task requirement: {_humanize_identifier(checklist_key)}."
        )

    if trigger_type == "mutating_action":
        if tool_name:
            guidance_lines.append(
                f"Do not repeat {tool_name} until you verify a different candidate or cart state."
            )
        else:
            guidance_lines.append(
                "Do not repeat the blocked cart mutation until you verify a different candidate or cart state."
            )
    elif trigger_type == "loop_detection":
        guidance_lines.append(
            "The current proposal repeats a blocked pattern. Change strategy before using another cart mutation."
        )
    elif trigger_reason:
        guidance_lines.append(trigger_reason)

    return guidance_lines or [
        "Pause, verify the last blocked step, and change strategy before mutating the cart again."
    ]


def _render_notice_from_action(action: OversightAction) -> str | None:
    if action.notice_text:
        action.notice_rendered = True
        action.notice_source = action.notice_source or "preset_notice"
        return action.notice_text

    lines = list(action.guidance_lines)
    if action.corrected_observation:
        action.notice_source = (
            "corrected_observation_plus_guidance" if lines else "corrected_observation"
        )
        lines = lines + [action.corrected_observation]
    elif lines:
        action.notice_source = (
            "local_fallback" if action.fallback_guidance_used else "guidance_lines"
        )
    if action.intervention_type == "run_verification" and not lines:
        action.notice_source = "default_final_notice"
        lines = [DEFAULT_FINAL_NOTICE]
    if not lines or action.trigger_type is None:
        action.notice_rendered = False
        return None
    action.notice_rendered = True
    return render_transient_notice(trigger_type=action.trigger_type, lines=lines)


def _authoritative_snapshot(state: ConversationState) -> dict[str, Any] | None:
    if state.last_authoritative_cart_snapshot is not None:
        return state.last_authoritative_cart_snapshot
    return build_authoritative_state_snapshot(state.tool_calls_history)


def _recent_tool_trajectory(
    state: ConversationState, system_config: Any
) -> list[dict[str, Any]]:
    window = max(int(getattr(system_config, "recent_tool_window", 5)), 1)
    return list(state.tool_calls_history[-window:])


def _coverage_guidance_lines(
    state: ConversationState, missing_keys: list[str]
) -> list[str]:
    if state.task_checklist is None:
        return [f"Inspect checklist target: {key}" for key in missing_keys[:3]]
    descriptions: list[str] = []
    for key in missing_keys:
        for item in state.task_checklist.items:
            if str(item.get("key")) != key:
                continue
            description = str(item.get("description") or key).strip()
            descriptions.append(f"Inspect {description}.")
            break
        if len(descriptions) >= 3:
            break
    return descriptions or [
        f"Inspect checklist target: {key}" for key in missing_keys[:3]
    ]


def _activation_enabled(*, state: ConversationState, system_config: Any) -> bool:
    return bool(
        getattr(system_config, "oversight_enabled", False)
        and getattr(system_config, "oversight_mode", None) == "adaptive"
        and state.domain == "shopping"
    )


def _final_payload(
    *,
    task_query: str,
    state: ConversationState,
    system_config: Any,
    draft_final_answer: str,
) -> dict[str, Any]:
    if state.execution_contract is None or state.task_checklist is None:
        raise ValueError(
            "Adaptive oversight requires execution contract and task checklist"
        )
    return {
        "mode": "final_verification",
        "task_query": task_query,
        "execution_contract": execution_contract_to_dict(state.execution_contract),
        "task_checklist": task_checklist_to_dict(state.task_checklist),
        "recent_tool_trajectory": _recent_tool_trajectory(state, system_config),
        "authoritative_state_snapshot": _authoritative_snapshot(state) or {},
        "draft_final_answer": draft_final_answer,
        "finalization_retry_count": state.final_verification_retry_count,
    }


async def _invoke_runtime_overseer(
    *,
    trigger_type: str,
    allowed_actions: list[str],
    task_query: str,
    state: ConversationState,
    system_config: Any,
    phase: Literal["initial", "cart_check"],
    step_index: int,
    tool_index: int | None,
    proposed_tool_calls: list[dict[str, Any]] | None,
    latest_tool_result: Any | None,
    trigger_reason: str,
    trigger_evidence: dict[str, Any],
) -> OversightAction:
    provider = getattr(system_config, "overseer_provider", None)
    if provider is None:
        return _noop_action(system_config=system_config)

    raw_overseer_text: str | None = None
    try:
        response = await call_chat_completion(
            provider=provider,
            messages=[
                {"role": "system", "content": P2_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "mode": "runtime",
                            "trigger_type": trigger_type,
                            "allowed_actions": allowed_actions,
                            "task_query": task_query,
                            "execution_contract": execution_contract_to_dict(
                                state.execution_contract
                            ),
                            "task_checklist": task_checklist_to_dict(
                                state.task_checklist
                            ),
                            "phase": phase,
                            "step_index": step_index,
                            "tool_index": tool_index,
                            "recent_tool_trajectory": _recent_tool_trajectory(
                                state, system_config
                            ),
                            "current_proposed_tool_calls": proposed_tool_calls or [],
                            "latest_observation": latest_tool_result,
                            "authoritative_state_snapshot": _authoritative_snapshot(
                                state
                            )
                            or {},
                            "freshness": {
                                "last_authoritative_read_step": state.last_authoritative_read_step,
                                "last_mutation_step": state.last_mutation_step,
                            },
                            "trigger_evidence": trigger_evidence,
                            "response_schema": {
                                "action": allowed_actions,
                                "decision_summary": "string",
                                "violation_evidence": {
                                    "violated_contract_ids": ["string"],
                                    "unmet_checklist_keys": ["string"],
                                    "confidence": "low|medium|high",
                                },
                                "guidance_lines": ["string"],
                                "corrected_observation": "string|null",
                            },
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            reasoning_enabled=getattr(system_config, "overseer_thinking", None),
            validate_nonempty=True,
        )
        cost = estimate_call_cost(response=response, provider=provider)
        state.record_overseer_call(response, cost=cost)
        raw_overseer_text = str(
            getattr(response.choices[0].message, "content", "") or ""
        ).strip()
        parsed = parse_runtime_overseer_json(raw_overseer_text)
        prompt_tokens = int(
            getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0
        )
        completion_tokens = int(
            getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0
        )
        guidance_lines = list(parsed["guidance_lines"])
        corrected_observation = parsed["corrected_observation"]
        fallback_guidance_used = False
        decision_summary = parsed["decision_summary"]
        if (
            parsed["action"] == "provide_guidance"
            and parsed["missing_corrective_content"]
        ):
            guidance_lines = _synthesize_guidance_lines(
                trigger_type=trigger_type,
                trigger_reason=trigger_reason,
                trigger_evidence=trigger_evidence,
                violated_contract_ids=list(parsed["violated_contract_ids"]),
                unmet_checklist_keys=list(parsed["unmet_checklist_keys"]),
            )
            fallback_guidance_used = True
            decision_summary = (
                "Runtime overseer contract violation: provide_guidance missing usable "
                "corrective content."
                + (f" {decision_summary}" if decision_summary else "")
            )
        action = OversightAction(
            should_intervene=parsed["action"] != "approve",
            trigger_type=trigger_type,
            trigger_reason=trigger_reason,
            intervention_type=parsed["action"],
            block_current_tool=False,
            guidance_lines=guidance_lines,
            corrected_observation=corrected_observation,
            violated_contract_ids=list(parsed["violated_contract_ids"]),
            unmet_checklist_keys=list(parsed["unmet_checklist_keys"]),
            violation_confidence=str(parsed["violation_confidence"]),
            overseer_invoked=True,
            overseer_mode=_overseer_mode(system_config),
            overseer_input_tokens=prompt_tokens,
            overseer_output_tokens=completion_tokens,
            overseer_cost=cost,
            decision_summary=decision_summary,
            raw_overseer_text=raw_overseer_text or None,
            parsed_payload=parsed,
            fallback_guidance_used=fallback_guidance_used,
        )
        action.notice_text = _render_notice_from_action(action)
        return action
    except Exception as exc:
        return OversightAction(
            should_intervene=False,
            trigger_type=trigger_type,
            trigger_reason=trigger_reason,
            intervention_type="approve",
            overseer_invoked=True,
            overseer_mode=_overseer_mode(system_config),
            decision_summary=f"Runtime overseer fallback to approve: {exc}",
            raw_overseer_text=raw_overseer_text,
        )


def _increment_retry_and_check_cap(
    state: ConversationState, system_config: Any
) -> bool:
    state.final_verification_retry_count += 1
    if state.final_verification_retry_count > int(
        getattr(system_config, "final_repair_retry_cap", 2)
    ):
        state.final_verification_result = "retry_cap_exhausted"
        return True
    return False


async def _invoke_final_verifier(
    *,
    task_query: str,
    state: ConversationState,
    system_config: Any,
    phase: Literal["initial", "cart_check"],
    step_index: int,
    draft_final_answer: str,
) -> OversightAction:
    provider = getattr(system_config, "overseer_provider", None)
    if provider is None:
        return _noop_action(system_config=system_config)

    try:
        response = await call_chat_completion(
            provider=provider,
            messages=[
                {"role": "system", "content": P3_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        _final_payload(
                            task_query=task_query,
                            state=state,
                            system_config=system_config,
                            draft_final_answer=draft_final_answer,
                        ),
                        ensure_ascii=False,
                    ),
                },
            ],
            reasoning_enabled=getattr(system_config, "overseer_thinking", None),
            validate_nonempty=True,
        )
        cost = estimate_call_cost(response=response, provider=provider)
        state.record_overseer_call(response, cost=cost)
        parsed = parse_final_verifier_json(
            getattr(response.choices[0].message, "content", "") or ""
        )
        prompt_tokens = int(
            getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0
        )
        completion_tokens = int(
            getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0
        )

        if parsed["action"] == "approve" and parsed["pass"]:
            state.final_verification_result = "approved"
            return OversightAction(
                should_intervene=False,
                trigger_type="final_checkpoint",
                trigger_reason="final verifier approved finalization",
                intervention_type="approve",
                overseer_invoked=True,
                overseer_mode=_overseer_mode(system_config),
                overseer_input_tokens=prompt_tokens,
                overseer_output_tokens=completion_tokens,
                overseer_cost=cost,
                decision_summary=parsed["decision_summary"],
                final_verification_result="approved",
            )

        exhausted = _increment_retry_and_check_cap(state, system_config)
        if not exhausted:
            state.final_verification_result = "repair_requested"
        action = OversightAction(
            should_intervene=True,
            trigger_type="final_checkpoint",
            trigger_reason="final verifier requested more verification",
            intervention_type="run_verification",
            guidance_lines=list(parsed["next_step_notice_lines"])
            or [DEFAULT_FINAL_NOTICE],
            violated_contract_ids=list(parsed["violated_contract_ids"]),
            unmet_checklist_keys=list(parsed["unmet_checklist_keys"]),
            overseer_invoked=True,
            overseer_mode=_overseer_mode(system_config),
            overseer_input_tokens=prompt_tokens,
            overseer_output_tokens=completion_tokens,
            overseer_cost=cost,
            decision_summary=parsed["decision_summary"],
            final_verification_result=state.final_verification_result,
        )
        action.notice_text = _render_notice_from_action(action)
        if exhausted:
            action.final_verification_result = "retry_cap_exhausted"
        return action
    except Exception as exc:
        exhausted = _increment_retry_and_check_cap(state, system_config)
        if not exhausted:
            state.final_verification_result = "repair_requested"
        action = OversightAction(
            should_intervene=True,
            trigger_type="final_checkpoint",
            trigger_reason="final verifier fallback due to call or parse failure",
            intervention_type="run_verification",
            guidance_lines=[DEFAULT_FINAL_NOTICE],
            overseer_invoked=True,
            overseer_mode=_overseer_mode(system_config),
            decision_summary=f"Final verifier fallback: {exc}",
            final_verification_result=state.final_verification_result,
        )
        action.notice_text = _render_notice_from_action(action)
        if exhausted:
            action.final_verification_result = "retry_cap_exhausted"
        return action


async def _evaluate_oversight_impl(
    *,
    hook: Literal["pre_tool", "post_tool", "midpoint", "final"],
    state: ConversationState,
    system_config: Any,
    phase: Literal["initial", "cart_check"],
    task_query: str,
    proposed_tool_calls: list[dict[str, Any]] | None = None,
    latest_tool_result: Any | None = None,
    draft_final_answer: str | None = None,
    step_index: int = 0,
    tool_index: int | None = None,
) -> OversightAction:
    if not _activation_enabled(state=state, system_config=system_config):
        return _noop_action(system_config=system_config)

    if hook == "pre_tool":
        calls = proposed_tool_calls or []
        for index, call in enumerate(calls):
            classification = classify_mutating_tool(
                str(call.get("name") or ""),
                mutating_tools=getattr(system_config, "mutating_tools", ()),
            )
            if classification["is_mutating"]:
                action = await _invoke_runtime_overseer(
                    trigger_type="mutating_action",
                    allowed_actions=["approve", "provide_guidance"],
                    task_query=task_query,
                    state=state,
                    system_config=system_config,
                    phase=phase,
                    step_index=step_index,
                    tool_index=index,
                    proposed_tool_calls=calls,
                    latest_tool_result=None,
                    trigger_reason=f"mutating tool proposed: {classification['tool_name']}",
                    trigger_evidence={"tool_name": classification["tool_name"]},
                )
                action.blocked_tool_name = classification["tool_name"]
                action.blocked_tool_arguments = call.get("arguments")
                action.blocked_tool_arguments_normalized = normalize_arguments(
                    call.get("arguments")
                )
                return action

        for index, call in enumerate(calls):
            loop_result = detect_loop(
                current_tool_name=str(call.get("name") or ""),
                current_arguments=call.get("arguments"),
                recent_tool_history=state.tool_calls_history,
                similarity_threshold=float(
                    getattr(system_config, "loop_similarity_threshold", 0.92)
                ),
                window_size=int(getattr(system_config, "loop_window", 5)),
                repeat_threshold=int(getattr(system_config, "loop_repeat_count", 3)),
            )
            if not loop_result["would_trigger"]:
                continue
            action = await _invoke_runtime_overseer(
                trigger_type="loop_detection",
                allowed_actions=["approve", "provide_guidance"],
                task_query=task_query,
                state=state,
                system_config=system_config,
                phase=phase,
                step_index=step_index,
                tool_index=index,
                proposed_tool_calls=calls,
                latest_tool_result=None,
                trigger_reason="proposed tool call matches recent repeated tool pattern",
                trigger_evidence=loop_result,
            )
            action.loop_signature = loop_result["loop_signature"]
            action.blocked_tool_name = str(call.get("name") or "")
            action.blocked_tool_arguments = call.get("arguments")
            action.blocked_tool_arguments_normalized = normalize_arguments(
                call.get("arguments")
            )
            return action

        return _noop_action(system_config=system_config)

    if hook == "post_tool":
        if not detect_tool_error(latest_tool_result):
            return _noop_action(system_config=system_config)
        action = await _invoke_runtime_overseer(
            trigger_type="error_occurrence",
            allowed_actions=["provide_guidance", "correct_observation"],
            task_query=task_query,
            state=state,
            system_config=system_config,
            phase=phase,
            step_index=step_index,
            tool_index=tool_index,
            proposed_tool_calls=None,
            latest_tool_result=latest_tool_result,
            trigger_reason="latest tool result appears to be an error",
            trigger_evidence={"tool_result": latest_tool_result},
        )
        action.notice_text = _render_notice_from_action(action)
        return action

    if hook == "midpoint":
        if state.task_checklist is None:
            return _noop_action(system_config=system_config)
        coverage_status = compute_coverage_status(
            checklist=state.task_checklist,
            tool_history=state.tool_calls_history,
        )
        if (
            coverage_status["coverage_fraction"]
            >= float(getattr(system_config, "coverage_threshold", 0.50))
            or not coverage_status["missing_keys"]
        ):
            return OversightAction(
                should_intervene=False,
                trigger_type="coverage_deficit",
                intervention_type="approve",
                overseer_mode=_overseer_mode(system_config),
                coverage_status=coverage_status,
            )
        action = await _invoke_runtime_overseer(
            trigger_type="coverage_deficit",
            allowed_actions=["provide_guidance"],
            task_query=task_query,
            state=state,
            system_config=system_config,
            phase=phase,
            step_index=step_index,
            tool_index=None,
            proposed_tool_calls=None,
            latest_tool_result=None,
            trigger_reason="initial-phase coverage is below threshold",
            trigger_evidence={"coverage_status": coverage_status},
        )
        action.coverage_status = coverage_status
        if action.intervention_type == "approve" and not action.should_intervene:
            return action
        action.intervention_type = "provide_guidance"
        action.should_intervene = True
        action.guidance_lines = _coverage_guidance_lines(
            state, list(coverage_status["missing_keys"])
        )
        action.unmet_checklist_keys = list(coverage_status["missing_keys"])
        action.notice_text = None
        action.notice_text = _render_notice_from_action(action)
        return action

    if hook == "final":
        if phase != "cart_check":
            return _noop_action(system_config=system_config)

        if state.final_verification_result == "retry_cap_exhausted":
            return OversightAction(
                should_intervene=True,
                trigger_type="final_checkpoint",
                trigger_reason="retry cap already exhausted",
                intervention_type="run_verification",
                guidance_lines=[DEFAULT_FINAL_NOTICE],
                notice_text=render_transient_notice(
                    trigger_type="final_checkpoint",
                    lines=[DEFAULT_FINAL_NOTICE],
                ),
                overseer_mode=_overseer_mode(system_config),
                final_verification_result="retry_cap_exhausted",
            )

        if state.last_mutation_step is not None and (
            state.last_authoritative_read_step is None
            or state.last_mutation_step > state.last_authoritative_read_step
        ):
            exhausted = _increment_retry_and_check_cap(state, system_config)
            if not exhausted:
                state.final_verification_result = "stale_cart_notice"
            return OversightAction(
                should_intervene=True,
                trigger_type="final_checkpoint",
                trigger_reason="authoritative cart read is stale relative to latest mutation",
                intervention_type="run_verification",
                guidance_lines=[DEFAULT_FINAL_NOTICE],
                notice_text=render_transient_notice(
                    trigger_type="final_checkpoint",
                    lines=[DEFAULT_FINAL_NOTICE],
                ),
                overseer_mode=_overseer_mode(system_config),
                final_verification_result=state.final_verification_result,
            )

        return await _invoke_final_verifier(
            task_query=task_query,
            state=state,
            system_config=system_config,
            phase=phase,
            step_index=step_index,
            draft_final_answer=draft_final_answer or "",
        )

    return _noop_action(system_config=system_config)


def evaluate_oversight(*args: Any, **kwargs: Any) -> Any:
    if args and len(args) == 4 and not kwargs:
        _, _, state, system_config = args
        return _noop_action(system_config=system_config)
    return _evaluate_oversight_impl(**kwargs)


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
    "ConversationState",
    "H1Outcome",
    "OversightAction",
    "apply_intervention",
    "compute_h1_outcome",
    "evaluate_oversight",
    "parse_final_verifier_json",
    "parse_runtime_overseer_json",
]
