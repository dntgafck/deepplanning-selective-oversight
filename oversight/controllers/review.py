from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..base import OversightContext, OversightController
from ..clients import invoke_final_verifier, invoke_runtime_overseer, overseer_mode
from ..notices import (
    DEFAULT_FINAL_NOTICE,
    coverage_guidance_lines,
    render_notice_from_action,
    render_transient_notice,
)
from ..policies import increment_retry_and_check_cap
from ..prompts import P2_SYSTEM_PROMPT, P3_SYSTEM_PROMPT
from ..triggers import (
    classify_mutating_tool,
    compute_coverage_status,
    detect_loop,
    detect_tool_error,
    normalize_arguments,
)

if TYPE_CHECKING:
    from .. import OversightAction
    from ..state import ConversationState


def _runtime_dependencies() -> tuple[type[OversightAction], Any, Any]:
    import oversight as oversight_module

    return (
        oversight_module.OversightAction,
        oversight_module.call_chat_completion,
        oversight_module.estimate_call_cost,
    )


def _cart_read_is_stale(state: ConversationState) -> bool:
    if state.last_mutation_event_index is None:
        return False
    if state.last_authoritative_read_event_index is None:
        return True
    return state.last_mutation_event_index > state.last_authoritative_read_event_index


async def _evaluate_pre_tool(
    context: OversightContext,
    *,
    always_on: bool,
) -> OversightAction:
    action_factory, call_chat_completion_fn, estimate_call_cost_fn = (
        _runtime_dependencies()
    )
    calls = context.proposed_tool_calls or []
    for index, call in enumerate(calls):
        classification = classify_mutating_tool(
            str(call.get("name") or ""),
            mutating_tools=getattr(context.system_config, "mutating_tools", ()),
        )
        if classification["is_mutating"]:
            action = await invoke_runtime_overseer(
                action_factory=action_factory,
                call_chat_completion_fn=call_chat_completion_fn,
                estimate_call_cost_fn=estimate_call_cost_fn,
                prompt=P2_SYSTEM_PROMPT,
                trigger_type="mutating_action",
                allowed_actions=["approve", "provide_guidance"],
                task_query=context.task_query,
                state=context.state,
                system_config=context.system_config,
                phase=context.phase,
                step_index=context.step_index,
                tool_index=index,
                proposed_tool_calls=calls,
                latest_tool_result=None,
                trigger_reason=(
                    f"mutating tool proposed: {classification['tool_name']}"
                ),
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
            recent_tool_history=context.state.tool_calls_history,
            similarity_threshold=float(
                getattr(context.system_config, "loop_similarity_threshold", 0.92)
            ),
            window_size=int(getattr(context.system_config, "loop_window", 5)),
            repeat_threshold=int(
                getattr(context.system_config, "loop_repeat_count", 3)
            ),
        )
        if not loop_result["would_trigger"]:
            continue
        action = await invoke_runtime_overseer(
            action_factory=action_factory,
            call_chat_completion_fn=call_chat_completion_fn,
            estimate_call_cost_fn=estimate_call_cost_fn,
            prompt=P2_SYSTEM_PROMPT,
            trigger_type="loop_detection",
            allowed_actions=["approve", "provide_guidance"],
            task_query=context.task_query,
            state=context.state,
            system_config=context.system_config,
            phase=context.phase,
            step_index=context.step_index,
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

    if not always_on or not calls:
        return action_factory(
            should_intervene=False,
            overseer_mode=overseer_mode(context.system_config),
            final_verification_result="not_applicable",
        )

    action = await invoke_runtime_overseer(
        action_factory=action_factory,
        call_chat_completion_fn=call_chat_completion_fn,
        estimate_call_cost_fn=estimate_call_cost_fn,
        prompt=P2_SYSTEM_PROMPT,
        trigger_type="always_on_pre_tool",
        allowed_actions=["approve", "provide_guidance"],
        task_query=context.task_query,
        state=context.state,
        system_config=context.system_config,
        phase=context.phase,
        step_index=context.step_index,
        tool_index=0,
        proposed_tool_calls=calls,
        latest_tool_result=None,
        trigger_reason="always-on pre-tool oversight (System B)",
        trigger_evidence={
            "mode": "always",
            "tool_name": str(calls[0].get("name") or ""),
            "proposed_call_count": len(calls),
        },
    )
    action.blocked_tool_name = str(calls[0].get("name") or "")
    action.blocked_tool_arguments = calls[0].get("arguments")
    action.blocked_tool_arguments_normalized = normalize_arguments(
        calls[0].get("arguments")
    )
    return action


async def _evaluate_post_tool(
    context: OversightContext,
    *,
    always_on: bool,
) -> OversightAction:
    action_factory, call_chat_completion_fn, estimate_call_cost_fn = (
        _runtime_dependencies()
    )
    has_error = detect_tool_error(context.latest_tool_result)
    if always_on:
        action = await invoke_runtime_overseer(
            action_factory=action_factory,
            call_chat_completion_fn=call_chat_completion_fn,
            estimate_call_cost_fn=estimate_call_cost_fn,
            prompt=P2_SYSTEM_PROMPT,
            trigger_type="always_on_post_tool",
            allowed_actions=[
                "approve",
                "provide_guidance",
                "correct_observation",
            ],
            task_query=context.task_query,
            state=context.state,
            system_config=context.system_config,
            phase=context.phase,
            step_index=context.step_index,
            tool_index=context.tool_index,
            proposed_tool_calls=None,
            latest_tool_result=context.latest_tool_result,
            trigger_reason="always-on post-tool oversight (System B)",
            trigger_evidence={
                "mode": "always",
                "tool_error_detected": has_error,
            },
        )
        action.notice_text = render_notice_from_action(action)
        return action

    if not has_error:
        return action_factory(
            should_intervene=False,
            overseer_mode=overseer_mode(context.system_config),
            final_verification_result="not_applicable",
        )

    action = await invoke_runtime_overseer(
        action_factory=action_factory,
        call_chat_completion_fn=call_chat_completion_fn,
        estimate_call_cost_fn=estimate_call_cost_fn,
        prompt=P2_SYSTEM_PROMPT,
        trigger_type="error_occurrence",
        allowed_actions=["provide_guidance", "correct_observation"],
        task_query=context.task_query,
        state=context.state,
        system_config=context.system_config,
        phase=context.phase,
        step_index=context.step_index,
        tool_index=context.tool_index,
        proposed_tool_calls=None,
        latest_tool_result=context.latest_tool_result,
        trigger_reason="latest tool result appears to be an error",
        trigger_evidence={"tool_result": context.latest_tool_result},
    )
    action.notice_text = render_notice_from_action(action)
    return action


async def _evaluate_midpoint(context: OversightContext) -> OversightAction:
    action_factory, call_chat_completion_fn, estimate_call_cost_fn = (
        _runtime_dependencies()
    )
    if context.state.task_checklist is None:
        return action_factory(
            should_intervene=False,
            overseer_mode=overseer_mode(context.system_config),
            final_verification_result="not_applicable",
        )

    coverage_status = compute_coverage_status(
        checklist=context.state.task_checklist,
        tool_history=context.state.tool_calls_history,
        role_map=getattr(context.system_config, "tool_role_map", None),
    )
    if (
        coverage_status["coverage_fraction"]
        >= float(getattr(context.system_config, "coverage_threshold", 0.50))
        or not coverage_status["missing_keys"]
    ):
        return action_factory(
            should_intervene=False,
            trigger_type="coverage_deficit",
            intervention_type="approve",
            overseer_mode=overseer_mode(context.system_config),
            coverage_status=coverage_status,
        )

    action = await invoke_runtime_overseer(
        action_factory=action_factory,
        call_chat_completion_fn=call_chat_completion_fn,
        estimate_call_cost_fn=estimate_call_cost_fn,
        prompt=P2_SYSTEM_PROMPT,
        trigger_type="coverage_deficit",
        allowed_actions=["provide_guidance"],
        task_query=context.task_query,
        state=context.state,
        system_config=context.system_config,
        phase=context.phase,
        step_index=context.step_index,
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
    action.guidance_lines = coverage_guidance_lines(
        context.state, list(coverage_status["missing_keys"])
    )
    action.unmet_checklist_keys = list(coverage_status["missing_keys"])
    action.notice_text = None
    action.notice_text = render_notice_from_action(action)
    return action


async def _evaluate_final(context: OversightContext) -> OversightAction:
    action_factory, call_chat_completion_fn, estimate_call_cost_fn = (
        _runtime_dependencies()
    )
    if context.phase != "cart_check":
        return action_factory(
            should_intervene=False,
            overseer_mode=overseer_mode(context.system_config),
            final_verification_result="not_applicable",
        )

    if context.state.final_verification_result == "retry_cap_exhausted":
        return action_factory(
            should_intervene=True,
            trigger_type="final_checkpoint",
            trigger_reason="retry cap already exhausted",
            intervention_type="run_verification",
            guidance_lines=[DEFAULT_FINAL_NOTICE],
            notice_text=render_transient_notice(
                trigger_type="final_checkpoint",
                lines=[DEFAULT_FINAL_NOTICE],
            ),
            overseer_mode=overseer_mode(context.system_config),
            final_verification_result="retry_cap_exhausted",
        )

    if _cart_read_is_stale(context.state):
        max_stale_cart_notices = max(
            int(getattr(context.system_config, "max_stale_cart_notices", 1)),
            0,
        )
        if context.state.stale_cart_notice_count < max_stale_cart_notices:
            context.state.stale_cart_notice_count += 1
            context.state.final_verification_result = "stale_cart_notice"
            return action_factory(
                should_intervene=True,
                trigger_type="final_checkpoint",
                trigger_reason=(
                    "authoritative cart read is stale relative to latest mutation"
                ),
                intervention_type="run_verification",
                guidance_lines=[DEFAULT_FINAL_NOTICE],
                notice_text=render_transient_notice(
                    trigger_type="final_checkpoint",
                    lines=[DEFAULT_FINAL_NOTICE],
                ),
                overseer_mode=overseer_mode(context.system_config),
                final_verification_result=context.state.final_verification_result,
            )

    return await invoke_final_verifier(
        action_factory=action_factory,
        call_chat_completion_fn=call_chat_completion_fn,
        estimate_call_cost_fn=estimate_call_cost_fn,
        increment_retry_and_check_cap_fn=increment_retry_and_check_cap,
        prompt=P3_SYSTEM_PROMPT,
        task_query=context.task_query,
        state=context.state,
        system_config=context.system_config,
        phase=context.phase,
        step_index=context.step_index,
        draft_final_answer=context.draft_final_answer or "",
    )


class ContinuousReviewOversight(OversightController):
    profile = "continuous_review"
    controller_name = "ContinuousReviewOversight"
    active_hooks = frozenset({"pre_tool", "post_tool", "midpoint", "final"})

    async def evaluate(self, context: OversightContext) -> OversightAction:
        if not self.is_active_for_hook(
            state=context.state,
            system_config=context.system_config,
            hook=context.hook,
        ):
            return self.inactive_action(context)
        if context.hook == "pre_tool":
            return await _evaluate_pre_tool(context, always_on=True)
        if context.hook == "post_tool":
            return await _evaluate_post_tool(context, always_on=True)
        if context.hook == "midpoint":
            return await _evaluate_midpoint(context)
        if context.hook == "final":
            return await _evaluate_final(context)
        return self.noop_action(system_config=context.system_config)


class CheckpointReviewOversight(OversightController):
    profile = "checkpoint_review"
    controller_name = "CheckpointReviewOversight"
    active_hooks = frozenset({"midpoint", "final"})

    async def evaluate(self, context: OversightContext) -> OversightAction:
        if not self.is_active_for_hook(
            state=context.state,
            system_config=context.system_config,
            hook=context.hook,
        ):
            return self.inactive_action(context)
        if context.hook == "midpoint":
            return await _evaluate_midpoint(context)
        if context.hook == "final":
            return await _evaluate_final(context)
        return self.noop_action(system_config=context.system_config)
