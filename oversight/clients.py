from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Literal

from .contracts import execution_contract_to_dict, task_checklist_to_dict
from .json_utils import JSON_OBJECT_RESPONSE_FORMAT
from .notices import (
    DEFAULT_FINAL_NOTICE,
    render_notice_from_action,
    synthesize_guidance_lines,
)
from .parsing import parse_final_verifier_json, parse_runtime_overseer_json
from .state import ConversationState
from .triggers import build_authoritative_state_snapshot


def overseer_mode(system_config: Any) -> str:
    provider = getattr(system_config, "overseer_provider", None)
    if provider is None:
        return "disabled"
    return (
        "thinking"
        if bool(getattr(system_config, "overseer_thinking", False))
        else "non-thinking"
    )


def make_noop_action(
    *,
    action_factory: Callable[..., Any],
    system_config: Any,
    final_result: str = "not_applicable",
) -> Any:
    return action_factory(
        should_intervene=False,
        overseer_mode=overseer_mode(system_config),
        final_verification_result=final_result,
    )


def authoritative_snapshot(
    state: ConversationState,
    system_config: Any | None = None,
) -> dict[str, Any] | None:
    if state.last_authoritative_cart_snapshot is not None:
        return state.last_authoritative_cart_snapshot
    return build_authoritative_state_snapshot(
        state.tool_calls_history,
        authority_tools=getattr(system_config, "state_authority_tools", None),
    )


def freshness_payload(state: ConversationState) -> dict[str, Any]:
    return {
        "tool_event_index": state.tool_event_index,
        "last_authoritative_read_step": state.last_authoritative_read_step,
        "last_mutation_step": state.last_mutation_step,
        "last_authoritative_read_event_index": state.last_authoritative_read_event_index,
        "last_mutation_event_index": state.last_mutation_event_index,
        "stale_cart_notice_count": state.stale_cart_notice_count,
    }


def recent_tool_trajectory(
    state: ConversationState, system_config: Any
) -> list[dict[str, Any]]:
    window = max(int(getattr(system_config, "recent_tool_window", 5)), 1)
    return list(state.tool_calls_history[-window:])


def final_payload(
    *,
    task_query: str,
    state: ConversationState,
    system_config: Any,
    draft_final_answer: str,
) -> dict[str, Any]:
    if state.execution_contract is None or state.task_checklist is None:
        raise ValueError("Oversight requires execution contract and task checklist")
    return {
        "mode": "final_verification",
        "task_query": task_query,
        "execution_contract": execution_contract_to_dict(state.execution_contract),
        "task_checklist": task_checklist_to_dict(state.task_checklist),
        "recent_tool_trajectory": recent_tool_trajectory(state, system_config),
        "authoritative_state_snapshot": authoritative_snapshot(state, system_config)
        or {},
        "draft_final_answer": draft_final_answer,
        "freshness": freshness_payload(state),
        "finalization_retry_count": state.final_verification_retry_count,
    }


async def invoke_runtime_overseer(
    *,
    action_factory: Callable[..., Any],
    call_chat_completion_fn: Callable[..., Any],
    estimate_call_cost_fn: Callable[..., float | None],
    prompt: str,
    task_query: str,
    state: ConversationState,
    system_config: Any,
    phase: Literal["initial", "cart_check"],
    step_index: int,
    tool_index: int | None,
    proposed_tool_calls: list[dict[str, Any]] | None,
    latest_tool_result: Any | None,
    trigger_type: str,
    allowed_actions: list[str],
    trigger_reason: str,
    trigger_evidence: dict[str, Any],
    render_notice_from_action_fn: Callable[
        [Any], str | None
    ] = render_notice_from_action,
    synthesize_guidance_lines_fn: Callable[..., list[str]] = synthesize_guidance_lines,
) -> Any:
    provider = getattr(system_config, "overseer_provider", None)
    if provider is None:
        return make_noop_action(
            action_factory=action_factory,
            system_config=system_config,
        )

    raw_overseer_text: str | None = None
    try:
        response = await call_chat_completion_fn(
            provider=provider,
            messages=[
                {"role": "system", "content": prompt},
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
                            "recent_tool_trajectory": recent_tool_trajectory(
                                state, system_config
                            ),
                            "current_proposed_tool_calls": proposed_tool_calls or [],
                            "latest_observation": latest_tool_result,
                            "authoritative_state_snapshot": authoritative_snapshot(
                                state,
                                system_config,
                            )
                            or {},
                            "freshness": freshness_payload(state),
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
            response_format=JSON_OBJECT_RESPONSE_FORMAT,
            validate_nonempty=True,
        )
        cost = estimate_call_cost_fn(response=response, provider=provider)
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
            guidance_lines = synthesize_guidance_lines_fn(
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
        action = action_factory(
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
            overseer_mode=overseer_mode(system_config),
            overseer_input_tokens=prompt_tokens,
            overseer_output_tokens=completion_tokens,
            overseer_cost=cost,
            decision_summary=decision_summary,
            raw_overseer_text=raw_overseer_text or None,
            parsed_payload=parsed,
            fallback_guidance_used=fallback_guidance_used,
        )
        action.notice_text = render_notice_from_action_fn(action)
        return action
    except Exception as exc:
        if raw_overseer_text is not None:
            state.runtime_overseer_parse_fallback_count += 1
        return action_factory(
            should_intervene=False,
            trigger_type=trigger_type,
            trigger_reason=trigger_reason,
            intervention_type="approve",
            overseer_invoked=True,
            overseer_mode=overseer_mode(system_config),
            decision_summary=f"Runtime overseer fallback to approve: {exc}",
            raw_overseer_text=raw_overseer_text,
        )


async def invoke_final_verifier(
    *,
    action_factory: Callable[..., Any],
    call_chat_completion_fn: Callable[..., Any],
    estimate_call_cost_fn: Callable[..., float | None],
    increment_retry_and_check_cap_fn: Callable[[ConversationState, Any], bool],
    prompt: str,
    task_query: str,
    state: ConversationState,
    system_config: Any,
    phase: Literal["initial", "cart_check"],
    step_index: int,
    draft_final_answer: str,
    render_notice_from_action_fn: Callable[
        [Any], str | None
    ] = render_notice_from_action,
) -> Any:
    provider = getattr(system_config, "overseer_provider", None)
    if provider is None:
        return make_noop_action(
            action_factory=action_factory,
            system_config=system_config,
        )

    raw_overseer_text: str | None = None
    try:
        response = await call_chat_completion_fn(
            provider=provider,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        final_payload(
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
            response_format=JSON_OBJECT_RESPONSE_FORMAT,
            validate_nonempty=True,
        )
        cost = estimate_call_cost_fn(response=response, provider=provider)
        state.record_overseer_call(response, cost=cost)
        raw_overseer_text = str(
            getattr(response.choices[0].message, "content", "") or ""
        ).strip()
        parsed = parse_final_verifier_json(raw_overseer_text)
        prompt_tokens = int(
            getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0
        )
        completion_tokens = int(
            getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0
        )

        if parsed["action"] == "approve" and parsed["pass"]:
            state.final_verification_result = "approved"
            return action_factory(
                should_intervene=False,
                trigger_type="final_checkpoint",
                trigger_reason="final verifier approved finalization",
                intervention_type="approve",
                overseer_invoked=True,
                overseer_mode=overseer_mode(system_config),
                overseer_input_tokens=prompt_tokens,
                overseer_output_tokens=completion_tokens,
                overseer_cost=cost,
                decision_summary=parsed["decision_summary"],
                final_verification_result="approved",
                raw_overseer_text=raw_overseer_text or None,
                parsed_payload=parsed,
            )

        exhausted = increment_retry_and_check_cap_fn(state, system_config)
        if not exhausted:
            state.final_verification_result = "repair_requested"
        action = action_factory(
            should_intervene=True,
            trigger_type="final_checkpoint",
            trigger_reason="final verifier requested more verification",
            intervention_type="run_verification",
            guidance_lines=list(parsed["next_step_notice_lines"])
            or [DEFAULT_FINAL_NOTICE],
            violated_contract_ids=list(parsed["violated_contract_ids"]),
            unmet_checklist_keys=list(parsed["unmet_checklist_keys"]),
            overseer_invoked=True,
            overseer_mode=overseer_mode(system_config),
            overseer_input_tokens=prompt_tokens,
            overseer_output_tokens=completion_tokens,
            overseer_cost=cost,
            decision_summary=parsed["decision_summary"],
            final_verification_result=state.final_verification_result,
            raw_overseer_text=raw_overseer_text or None,
            parsed_payload=parsed,
        )
        action.notice_text = render_notice_from_action_fn(action)
        if exhausted:
            action.final_verification_result = "retry_cap_exhausted"
        return action
    except Exception as exc:
        if raw_overseer_text is not None:
            state.final_verifier_parse_fallback_count += 1
        exhausted = increment_retry_and_check_cap_fn(state, system_config)
        if not exhausted:
            state.final_verification_result = "repair_requested"
        action = action_factory(
            should_intervene=True,
            trigger_type="final_checkpoint",
            trigger_reason="final verifier fallback due to call or parse failure",
            intervention_type="run_verification",
            guidance_lines=[DEFAULT_FINAL_NOTICE],
            overseer_invoked=True,
            overseer_mode=overseer_mode(system_config),
            decision_summary=f"Final verifier fallback: {exc}",
            final_verification_result=state.final_verification_result,
            raw_overseer_text=raw_overseer_text,
        )
        action.notice_text = render_notice_from_action_fn(action)
        if exhausted:
            action.final_verification_result = "retry_cap_exhausted"
        return action


__all__ = [
    "authoritative_snapshot",
    "final_payload",
    "freshness_payload",
    "invoke_final_verifier",
    "invoke_runtime_overseer",
    "make_noop_action",
    "overseer_mode",
    "recent_tool_trajectory",
]
