from __future__ import annotations

import asyncio
import json
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from deepplanning.config import OUTPUT_ROOT
from experiment import StructuredLogger, build_system_config, run_experiment
from experiment.config import system_model_identities
from experiment.logging import serialize_exception
from experiment.progress import InferenceProgressReporter
from failure_subtypes import (
    FAILURE_SUBTYPE_INFRA_TRANSIENT,
    classify_exception_failure_subtype,
    failure_subtype_from_stop_reason,
    observation_valid_for_failure_subtype,
)
from llm import (
    build_langfuse_trace_id,
    call_chat_completion,
    estimate_call_cost,
    extract_usage_tokens,
    flush_langfuse,
)
from oversight import (
    ConversationState,
    H1Outcome,
    apply_intervention,
    compute_h1_outcome,
    evaluate_oversight,
)
from oversight.contracts import (
    load_or_build_execution_contract_with_metadata,
    load_or_build_task_checklist_with_metadata,
)

from .base import TaskResult
from .vendor import (
    clear_vendored_tool_module_cache,
    load_shopping_agent_class,
    load_shopping_prompt,
)

VendorShoppingFnAgent = load_shopping_agent_class()
DEFAULT_INFRA_RETRY_LIMIT = 2
DEBUG_MESSAGES_ROOT = OUTPUT_ROOT / "_debug_messages" / "shopping"
TRANSIENT_NOTICE_ROLE = "user"


def get_system_prompt(level: int) -> str:
    return load_shopping_prompt(level)


def _collect_tool_call_parse_warnings(assistant_message: Any) -> list[str]:
    warnings: list[str] = []
    for index, tool_call in enumerate(
        getattr(assistant_message, "tool_calls", None) or []
    ):
        try:
            function = tool_call.function
            _ = function.name
            _ = function.arguments
        except Exception as exc:
            warnings.append(f"tool_calls[{index}] skipped: {exc}")
    return warnings


def _assistant_message_to_dict(msg: Any, calls: list[dict[str, Any]]) -> dict[str, Any]:
    message = {
        "role": "assistant",
        "content": getattr(msg, "content", None) or "",
    }
    reasoning_content = getattr(msg, "reasoning_content", None)
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    if calls:
        message["tool_calls"] = [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": call["arguments"],
                },
            }
            for call in calls
        ]
    return message


def _extract_final_output(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def _build_transient_notice_message(notice_text: str) -> dict[str, str]:
    # Keep executor prompt semantics stable by transporting oversight notices
    # through a transient request-side user turn instead of a system-role edit.
    return {
        "role": TRANSIENT_NOTICE_ROLE,
        "content": notice_text,
    }


def _build_request_messages(
    messages: list[Any],
    state: ConversationState,
    system_config: Any,
) -> list[Any]:
    request_messages = list(messages)
    if state.pending_executor_notice and getattr(
        system_config, "inject_transient_notice", True
    ):
        request_messages.append(
            _build_transient_notice_message(str(state.pending_executor_notice))
        )
    return request_messages


def _overseer_budget_remaining(state: ConversationState, system_config: Any) -> bool:
    budget = int(getattr(system_config, "overseer_call_budget_per_task", 8))
    if budget < 0:
        return True
    return state.overseer_invocation_count < budget


async def _maybe_log_budget_exhausted(
    *,
    logger: StructuredLogger | None,
    state: ConversationState,
    run_id: int,
    phase: str,
    step_index: int,
    system_config: Any,
) -> None:
    if logger is None or state._budget_exhaustion_logged:
        state._budget_exhaustion_logged = True
        return
    await logger.log_event(
        "oversight_budget_exhausted",
        {
            "domain": "shopping",
            "task_id": state.task_id,
            "run_id": run_id,
            "phase": phase,
            "step_index": step_index,
            "overseer_invocation_count": state.overseer_invocation_count,
            "budget": int(getattr(system_config, "overseer_call_budget_per_task", 8)),
        },
    )
    state._budget_exhaustion_logged = True


async def _log_oversight_artifact(
    *,
    logger: StructuredLogger | None,
    task_id: str,
    run_id: int,
    artifact_type: str,
    cache_key: str,
    cache_status: str,
    compiler_signature: str,
    overseer_mode: str,
    model_identities: dict[str, Any] | None = None,
) -> None:
    if logger is None:
        return
    payload = {
        "domain": "shopping",
        "task_id": task_id,
        "run_id": run_id,
        "artifact_type": artifact_type,
        "cache_key": cache_key,
        "cache_status": cache_status,
        "compiler_signature": compiler_signature,
        "overseer_mode": overseer_mode,
    }
    if model_identities is not None:
        payload["model_identities"] = model_identities
    await logger.log_event("oversight_artifact", payload)


async def _log_notice_injection(
    *,
    logger: StructuredLogger | None,
    state: ConversationState,
    run_id: int,
    phase: str,
    step_index: int,
    notice_text: str,
    notice_role: str,
) -> None:
    if logger is None:
        return
    await logger.log_event(
        "oversight_notice_injected",
        {
            "domain": "shopping",
            "task_id": state.task_id,
            "run_id": run_id,
            "phase": phase,
            "step_index": step_index,
            "notice_text": notice_text,
            "notice_role": notice_role,
        },
    )


async def _log_oversight_step(
    *,
    logger: StructuredLogger | None,
    state: ConversationState,
    run_id: int,
    phase: str,
    step_index: int,
    tool_index: int | None,
    action: Any,
    executor_input_tokens: int,
    executor_output_tokens: int,
    executor_cost: float | None,
    failure_subtype: str | None = None,
) -> None:
    if logger is None:
        return
    overseer_cost = getattr(action, "overseer_cost", None)
    total_cost = (
        None
        if executor_cost is None or overseer_cost is None
        else executor_cost + overseer_cost
    )
    await logger.log_event(
        "oversight_step",
        {
            "domain": "shopping",
            "task_id": state.task_id,
            "run_id": run_id,
            "phase": phase,
            "step_index": step_index,
            "tool_index": tool_index,
            "trigger_type": getattr(action, "trigger_type", None),
            "trigger_reason": getattr(action, "trigger_reason", None),
            "intervention_type": getattr(action, "intervention_type", None),
            "overseer_invoked": bool(getattr(action, "overseer_invoked", False)),
            "overseer_mode": getattr(action, "overseer_mode", "disabled"),
            "executor_input_tokens": executor_input_tokens,
            "executor_output_tokens": executor_output_tokens,
            "overseer_input_tokens": getattr(action, "overseer_input_tokens", 0),
            "overseer_output_tokens": getattr(action, "overseer_output_tokens", 0),
            "executor_cost": executor_cost,
            "overseer_cost": overseer_cost,
            "total_cost": total_cost,
            "failure_subtype": failure_subtype,
            "loop_signature": getattr(action, "loop_signature", None),
            "coverage_status": getattr(action, "coverage_status", None),
            "raw_overseer_text": getattr(action, "raw_overseer_text", None),
            "parsed_payload": getattr(action, "parsed_payload", None),
            "notice_rendered": bool(getattr(action, "notice_rendered", False)),
            "notice_source": getattr(action, "notice_source", None),
            "notice_text": getattr(action, "notice_text", None),
            "fallback_guidance_used": bool(
                getattr(action, "fallback_guidance_used", False)
            ),
            "h1_outcome": getattr(action, "h1_outcome", None),
            "blocked_tool_name": getattr(action, "blocked_tool_name", None),
            "blocked_mutation_repeat_count": getattr(
                action,
                "blocked_mutation_repeat_count",
                0,
            ),
            "termination_reason": getattr(action, "termination_reason", None),
            "final_verification_result": getattr(
                action,
                "final_verification_result",
                state.final_verification_result,
            ),
        },
    )


async def _maybe_log_overseer_error(
    *,
    logger: StructuredLogger | None,
    state: ConversationState,
    run_id: int,
    phase: str,
    step_index: int,
    tool_index: int | None,
    action: Any,
) -> None:
    if logger is None:
        return
    decision_summary = str(getattr(action, "decision_summary", "") or "")
    if "fallback" not in decision_summary.lower():
        return
    await logger.log_event(
        "overseer_error",
        {
            "domain": "shopping",
            "task_id": state.task_id,
            "run_id": run_id,
            "phase": phase,
            "step_index": step_index,
            "tool_index": tool_index,
            "trigger_type": getattr(action, "trigger_type", None),
            "decision_summary": decision_summary,
            "overseer_mode": getattr(action, "overseer_mode", "disabled"),
        },
    )


async def _evaluate_oversight_with_budget(
    *,
    logger: StructuredLogger | None,
    budget_state: ConversationState,
    run_id: int,
    budget_phase: str,
    budget_step_index: int,
    budget_system_config: Any,
    **kwargs: Any,
) -> Any | None:
    if _overseer_budget_remaining(budget_state, budget_system_config):
        return await evaluate_oversight(**kwargs)
    await _maybe_log_budget_exhausted(
        logger=logger,
        state=budget_state,
        run_id=run_id,
        phase=budget_phase,
        step_index=budget_step_index,
        system_config=budget_system_config,
    )
    return None


def _adaptive_shopping_oversight_active(
    state: ConversationState,
    system_config: Any,
) -> bool:
    return bool(
        system_config.oversight_enabled
        and system_config.oversight_mode == "adaptive"
        and state.domain == "shopping"
    )


def _resolve_run_database_dir(
    base_database_dir: Path,
    run_id: int,
    database_dir_by_run: dict[int, Path] | None,
    runs: int,
) -> Path:
    if database_dir_by_run is not None:
        return database_dir_by_run[run_id]
    if runs > 1:
        raise ValueError(
            "Shopping multi-run execution requires per-run database directories."
        )
    return base_database_dir


def _resolve_run_output_dir(
    base_output_dir: Path | None,
    run_id: int,
    output_dir_by_run: dict[int, Path] | None,
    runs: int,
) -> Path | None:
    if output_dir_by_run is not None:
        return output_dir_by_run[run_id]
    if base_output_dir is None:
        return None
    if runs > 1:
        return base_output_dir / f"run_{run_id}"
    return base_output_dir


def _error_summary(error_payload: dict[str, Any]) -> str:
    error_type = error_payload.get("type", "Error")
    message = str(error_payload.get("message", "")).strip()
    return f"{error_type}: {message}" if message else str(error_type)


def _prepare_case_retry_snapshot(
    run_database_dir: Path,
    sample_id: str,
) -> tuple[Path, Path] | None:
    case_dir = run_database_dir / f"case_{sample_id}"
    if not case_dir.exists():
        return None

    snapshot_root = run_database_dir.parent / ".retry_snapshots"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    snapshot_dir = snapshot_root / f"case_{sample_id}"
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    shutil.copytree(case_dir, snapshot_dir)
    return case_dir, snapshot_dir


def _restore_case_retry_snapshot(snapshot: tuple[Path, Path] | None) -> None:
    if snapshot is None:
        return

    case_dir, snapshot_dir = snapshot
    if case_dir.exists():
        shutil.rmtree(case_dir)
    shutil.copytree(snapshot_dir, case_dir)


def _cleanup_case_retry_snapshot(snapshot: tuple[Path, Path] | None) -> None:
    if snapshot is None:
        return

    _, snapshot_dir = snapshot
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir, ignore_errors=True)


class ShoppingAgentRunner(VendorShoppingFnAgent):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        clear_vendored_tool_module_cache()
        super().__init__(*args, **kwargs)

    async def run_task(
        self,
        user_query: str,
        system_prompt: str | None,
        state: ConversationState,
        system_config: Any,
        logger: StructuredLogger | None = None,
        run_id: int = 0,
        save_messages: bool = True,
        sample_id: str | None = None,
        messages_output_dir: str | None = None,
        shared_oversight_cache_root: Path | None = None,
        trace_id: str | None = None,
        session_id: str | None = None,
    ) -> TaskResult:
        state.begin()

        messages_file: Path | None = None
        if save_messages:
            if sample_id:
                db_case_dir = self.database_base_path / f"case_{sample_id}"
                db_case_dir.mkdir(parents=True, exist_ok=True)
                messages_file = db_case_dir / "messages.json"
            else:
                msg_dir = Path(messages_output_dir or DEBUG_MESSAGES_ROOT)
                msg_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                messages_file = msg_dir / f"messages_{timestamp}.json"

        messages: list[Any] = (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        ) + [{"role": "user", "content": user_query}]
        if save_messages and messages_file is not None:
            self._save_messages(messages, messages_file, 0, "Initial messages")

        oversight_active = _adaptive_shopping_oversight_active(state, system_config)
        model_identities = system_model_identities(system_config)
        if oversight_active:
            cache_root = shared_oversight_cache_root
            if cache_root is None:
                cache_root = Path(self.database_base_path).parent / "_oversight_cache"
            cache_root.mkdir(parents=True, exist_ok=True)
            contract, contract_cache_key, contract_cache_status = (
                await load_or_build_execution_contract_with_metadata(
                    domain="shopping",
                    executor_system_prompt=system_prompt or "",
                    tool_schema=self.openai_tools,
                    system_config=system_config,
                    cache_root=cache_root,
                )
            )
            state.execution_contract = contract
            await _log_oversight_artifact(
                logger=logger,
                task_id=state.task_id,
                run_id=run_id,
                artifact_type="execution_contract",
                cache_key=contract_cache_key,
                cache_status=contract_cache_status,
                compiler_signature=contract.compiler_signature,
                overseer_mode=(
                    "thinking" if system_config.overseer_thinking else "non-thinking"
                ),
                model_identities=model_identities,
            )
            checklist, checklist_cache_key, checklist_cache_status = (
                await load_or_build_task_checklist_with_metadata(
                    task_id=state.task_id,
                    task_query=user_query,
                    execution_contract=contract,
                    system_config=system_config,
                    cache_root=cache_root,
                )
            )
            state.task_checklist = checklist
            await _log_oversight_artifact(
                logger=logger,
                task_id=state.task_id,
                run_id=run_id,
                artifact_type="task_checklist",
                cache_key=checklist_cache_key,
                cache_status=checklist_cache_status,
                compiler_signature=checklist.compiler_signature,
                overseer_mode=(
                    "thinking" if system_config.overseer_thinking else "non-thinking"
                ),
                model_identities=model_identities,
            )

        messages, initial_phase_stop_reason, initial_last_step = await self._run_phase(
            messages=messages,
            state=state,
            system_config=system_config,
            logger=logger,
            run_id=run_id,
            phase_name="initial",
            stop_on_no_calls=False,
            save_messages=save_messages,
            messages_file=messages_file,
            task_query=user_query,
            trace_id=trace_id,
            session_id=session_id,
        )
        if oversight_active:
            midpoint_action = await _evaluate_oversight_with_budget(
                logger=logger,
                budget_state=state,
                run_id=run_id,
                budget_phase="initial",
                budget_step_index=initial_last_step,
                budget_system_config=system_config,
                hook="midpoint",
                state=state,
                system_config=system_config,
                phase="initial",
                task_query=user_query,
                step_index=initial_last_step,
            )
            if midpoint_action is not None:
                state.record_oversight_decision(initial_last_step, midpoint_action)
                await _log_oversight_step(
                    logger=logger,
                    state=state,
                    run_id=run_id,
                    phase="initial",
                    step_index=initial_last_step,
                    tool_index=None,
                    action=midpoint_action,
                    executor_input_tokens=0,
                    executor_output_tokens=0,
                    executor_cost=None,
                )
                await _maybe_log_overseer_error(
                    logger=logger,
                    state=state,
                    run_id=run_id,
                    phase="initial",
                    step_index=initial_last_step,
                    tool_index=None,
                    action=midpoint_action,
                )
                if midpoint_action.should_intervene:
                    await apply_intervention(state=state, action=midpoint_action)

        messages = self._add_to_cart(messages)
        messages, phase_stop_reason, _ = await self._run_phase(
            messages=messages,
            state=state,
            system_config=system_config,
            logger=logger,
            run_id=run_id,
            phase_name="cart_check",
            stop_on_no_calls=True,
            save_messages=save_messages,
            messages_file=messages_file,
            task_query=user_query,
            trace_id=trace_id,
            session_id=session_id,
        )

        final_output = (
            ""
            if state.final_verification_result == "retry_cap_exhausted"
            else _extract_final_output(messages)
        )
        final_stop_reason = (
            phase_stop_reason
            if phase_stop_reason == "no_tool_calls"
            else "max_steps_exhausted"
        )
        state.record_final_outcome(
            stop_reason=final_stop_reason,
            output=final_output,
            max_steps_hit=phase_stop_reason == "max_steps_exhausted",
        )
        state.finish()
        if logger is not None and oversight_active:
            await logger.log_event(
                "oversight_run_summary",
                {
                    **state.to_metrics(),
                    "domain": "shopping",
                    "task_id": state.task_id,
                    "run_id": run_id,
                    "model_identities": model_identities,
                },
            )
        return TaskResult(
            task_id=state.task_id,
            run_id=run_id,
            output=final_output,
            messages=messages,
            state=state,
        )

    async def _run_phase(
        self,
        messages: list[Any],
        state: ConversationState,
        system_config: Any,
        logger: StructuredLogger | None,
        run_id: int,
        phase_name: str,
        stop_on_no_calls: bool,
        save_messages: bool,
        messages_file: Path | None,
        task_query: str,
        trace_id: str | None,
        session_id: str | None,
    ) -> tuple[list[Any], str, int]:
        last_step_count = 0
        for step_count in range(1, system_config.max_steps + 1):
            last_step_count = step_count
            request_messages = _build_request_messages(messages, state, system_config)
            notice_injected = len(request_messages) > len(messages)
            started_at = time.time()
            if notice_injected:
                await _log_notice_injection(
                    logger=logger,
                    state=state,
                    run_id=run_id,
                    phase=phase_name,
                    step_index=step_count,
                    notice_text=str(state.pending_executor_notice or ""),
                    notice_role=TRANSIENT_NOTICE_ROLE,
                )

            async def log_attempt_error(
                payload: dict[str, Any],
                *,
                step: int = step_count,
                phase: str = phase_name,
                task_id: str = state.task_id,
                model_alias: str = system_config.executor_provider.alias,
            ) -> None:
                error_payload = payload.get("error", {})
                retry_note = (
                    "retrying" if payload.get("will_retry") else "no retries left"
                )
                print(
                    "⚠️  Shopping LLM call failed "
                    f"(sample={task_id}, run={run_id}, phase={phase}, "
                    f"step={step}, attempt={payload.get('attempt')}/{payload.get('max_attempts')}, "
                    f"{retry_note}): {_error_summary(error_payload)}"
                )
                if logger is None:
                    return
                await logger.log_event(
                    "executor_llm_error",
                    {
                        "domain": "shopping",
                        "task_id": task_id,
                        "run_id": run_id,
                        "phase": phase,
                        "step": step,
                        "model_alias": model_alias,
                        **payload,
                    },
                )

            response = await call_chat_completion(
                provider=system_config.executor_provider,
                messages=request_messages,
                tools=self.openai_tools,
                on_attempt_error=log_attempt_error,
                trace_id=trace_id,
                session_id=session_id,
            )
            ended_at = time.time()
            if notice_injected:
                state.pending_executor_notice = None
            executor_cost = estimate_call_cost(
                response=response,
                provider=system_config.executor_provider,
            )
            state.record_executor_call(response, cost=executor_cost)

            prompt_tokens, completion_tokens = extract_usage_tokens(response)
            assistant_message = response.choices[0].message
            calls = self._detect_tool_calls(assistant_message)
            parse_warnings = _collect_tool_call_parse_warnings(assistant_message)
            assistant_payload = _assistant_message_to_dict(assistant_message, calls)

            if calls and _adaptive_shopping_oversight_active(state, system_config):
                pre_tool_action = await _evaluate_oversight_with_budget(
                    logger=logger,
                    budget_state=state,
                    run_id=run_id,
                    budget_phase=phase_name,
                    budget_step_index=step_count,
                    budget_system_config=system_config,
                    hook="pre_tool",
                    state=state,
                    system_config=system_config,
                    phase=phase_name,
                    task_query=task_query,
                    proposed_tool_calls=calls,
                    step_index=step_count,
                )
                if pre_tool_action is not None:
                    h1_outcome = H1Outcome.APPROVE_CONTINUE
                    if (
                        pre_tool_action.should_intervene
                        and pre_tool_action.blocked_tool_name
                    ):
                        h1_outcome = compute_h1_outcome(
                            action=pre_tool_action,
                            tool_name=str(pre_tool_action.blocked_tool_name),
                            arguments=pre_tool_action.blocked_tool_arguments,
                            state=state,
                            system_config=system_config,
                        )

                    pre_tool_action.h1_outcome = h1_outcome.value
                    if h1_outcome == H1Outcome.HARD_BLOCK:
                        pre_tool_action.block_current_tool = True
                        repeat_count = state.record_blocked_mutation_attempt(
                            tool_name=str(pre_tool_action.blocked_tool_name),
                            arguments=pre_tool_action.blocked_tool_arguments,
                            similarity_threshold=float(
                                getattr(
                                    system_config,
                                    "loop_similarity_threshold",
                                    0.92,
                                )
                            ),
                        )
                        pre_tool_action.blocked_mutation_repeat_count = repeat_count
                    elif h1_outcome == H1Outcome.FORCED_APPROVE:
                        pre_tool_action.block_current_tool = False
                        pre_tool_action.intervention_type = "overseer_override_forced"
                        pre_tool_action.notice_text = None
                        pre_tool_action.notice_rendered = False
                        pre_tool_action.notice_source = None
                    elif h1_outcome == H1Outcome.APPROVE_WITH_NUDGE:
                        pre_tool_action.block_current_tool = False
                    else:
                        pre_tool_action.block_current_tool = False
                        pre_tool_action.notice_text = None
                        pre_tool_action.notice_rendered = False
                        pre_tool_action.notice_source = None

                    state.record_oversight_decision(step_count, pre_tool_action)
                    await _log_oversight_step(
                        logger=logger,
                        state=state,
                        run_id=run_id,
                        phase=phase_name,
                        step_index=step_count,
                        tool_index=0 if calls else None,
                        action=pre_tool_action,
                        executor_input_tokens=prompt_tokens,
                        executor_output_tokens=completion_tokens,
                        executor_cost=executor_cost,
                    )
                    await _maybe_log_overseer_error(
                        logger=logger,
                        state=state,
                        run_id=run_id,
                        phase=phase_name,
                        step_index=step_count,
                        tool_index=0 if calls else None,
                        action=pre_tool_action,
                    )
                    if pre_tool_action.notice_text:
                        await apply_intervention(state=state, action=pre_tool_action)
                    if h1_outcome == H1Outcome.HARD_BLOCK:
                        if save_messages and messages_file is not None:
                            self._save_messages(
                                messages,
                                messages_file,
                                step_count,
                                f"Oversight blocked tool batch ({phase_name})",
                            )
                        if logger is not None:
                            await logger.log_turn(
                                domain="shopping",
                                task_id=state.task_id,
                                run_id=run_id,
                                phase=phase_name,
                                turn_index=step_count,
                                started_at=started_at,
                                ended_at=ended_at,
                                request_messages=request_messages,
                                raw_response=response,
                                parsed_tool_calls=calls,
                                parse_warnings=parse_warnings,
                                tool_results=[],
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                                stop_reason="oversight_blocked",
                                model_alias=system_config.executor_provider.alias,
                            )
                        continue

            tool_results: list[dict[str, Any]] = []
            if not calls:
                if phase_name == "cart_check" and _adaptive_shopping_oversight_active(
                    state, system_config
                ):
                    final_action = await _evaluate_oversight_with_budget(
                        logger=logger,
                        budget_state=state,
                        run_id=run_id,
                        budget_phase=phase_name,
                        budget_step_index=step_count,
                        budget_system_config=system_config,
                        hook="final",
                        state=state,
                        system_config=system_config,
                        phase="cart_check",
                        task_query=task_query,
                        draft_final_answer=str(assistant_payload.get("content") or ""),
                        step_index=step_count,
                    )
                    if final_action is not None:
                        state.record_oversight_decision(step_count, final_action)
                        await _log_oversight_step(
                            logger=logger,
                            state=state,
                            run_id=run_id,
                            phase=phase_name,
                            step_index=step_count,
                            tool_index=None,
                            action=final_action,
                            executor_input_tokens=prompt_tokens,
                            executor_output_tokens=completion_tokens,
                            executor_cost=executor_cost,
                        )
                        await _maybe_log_overseer_error(
                            logger=logger,
                            state=state,
                            run_id=run_id,
                            phase=phase_name,
                            step_index=step_count,
                            tool_index=None,
                            action=final_action,
                        )
                        if final_action.should_intervene:
                            await apply_intervention(state=state, action=final_action)
                            if (
                                final_action.final_verification_result
                                == "retry_cap_exhausted"
                            ):
                                return messages, "no_tool_calls", step_count
                            continue

                messages.append(assistant_payload)
                if save_messages and messages_file is not None:
                    self._save_messages(
                        messages,
                        messages_file,
                        step_count,
                        f"LLM response ({phase_name}) - 0 tool calls",
                    )
                if logger is not None:
                    await logger.log_turn(
                        domain="shopping",
                        task_id=state.task_id,
                        run_id=run_id,
                        phase=phase_name,
                        turn_index=step_count,
                        started_at=started_at,
                        ended_at=ended_at,
                        request_messages=request_messages,
                        raw_response=response,
                        parsed_tool_calls=calls,
                        parse_warnings=parse_warnings,
                        tool_results=tool_results,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        stop_reason="no_tool_calls",
                        model_alias=system_config.executor_provider.alias,
                    )
                return (
                    messages,
                    "no_tool_calls" if stop_on_no_calls else "phase_break",
                    step_count,
                )

            messages.append(assistant_payload)
            if save_messages and messages_file is not None:
                self._save_messages(
                    messages,
                    messages_file,
                    step_count,
                    f"LLM response ({phase_name}) - {len(calls)} tool calls",
                )

            turn_has_pending_notice = False
            for tool_index, call in enumerate(calls):
                tool_result = self._exec_tool(call["name"], call["arguments"])
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": tool_result,
                    }
                )
                state.record_tool_call(
                    call,
                    tool_result,
                    phase=phase_name,
                    step_index=step_count,
                    tool_index=tool_index,
                    mutating_tools=tuple(system_config.mutating_tools),
                )
                tool_results.append(
                    {
                        "tool_name": call["name"],
                        "tool_call_id": call["id"],
                        "content": tool_result,
                    }
                )
                if not turn_has_pending_notice and _adaptive_shopping_oversight_active(
                    state, system_config
                ):
                    post_tool_action = await _evaluate_oversight_with_budget(
                        logger=logger,
                        budget_state=state,
                        run_id=run_id,
                        budget_phase=phase_name,
                        budget_step_index=step_count,
                        budget_system_config=system_config,
                        hook="post_tool",
                        state=state,
                        system_config=system_config,
                        phase=phase_name,
                        task_query=task_query,
                        latest_tool_result=tool_result,
                        step_index=step_count,
                        tool_index=tool_index,
                    )
                    if post_tool_action is not None:
                        state.record_oversight_decision(
                            step_count,
                            post_tool_action,
                            tool_index=tool_index,
                        )
                        await _log_oversight_step(
                            logger=logger,
                            state=state,
                            run_id=run_id,
                            phase=phase_name,
                            step_index=step_count,
                            tool_index=tool_index,
                            action=post_tool_action,
                            executor_input_tokens=prompt_tokens,
                            executor_output_tokens=completion_tokens,
                            executor_cost=executor_cost,
                        )
                        await _maybe_log_overseer_error(
                            logger=logger,
                            state=state,
                            run_id=run_id,
                            phase=phase_name,
                            step_index=step_count,
                            tool_index=tool_index,
                            action=post_tool_action,
                        )
                        if post_tool_action.should_intervene:
                            await apply_intervention(
                                state=state,
                                action=post_tool_action,
                            )
                            turn_has_pending_notice = (
                                state.pending_executor_notice is not None
                            )
            if save_messages and messages_file is not None:
                self._save_messages(
                    messages,
                    messages_file,
                    step_count,
                    f"Tool execution ({phase_name}) - {len(calls)} tools",
                )
            if logger is not None:
                await logger.log_turn(
                    domain="shopping",
                    task_id=state.task_id,
                    run_id=run_id,
                    phase=phase_name,
                    turn_index=step_count,
                    started_at=started_at,
                    ended_at=ended_at,
                    request_messages=request_messages,
                    raw_response=response,
                    parsed_tool_calls=calls,
                    parse_warnings=parse_warnings,
                    tool_results=tool_results,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    stop_reason="tool_calls",
                    model_alias=system_config.executor_provider.alias,
                )

        return messages, "max_steps_exhausted", last_step_count


async def run_agent_inference_async(
    model: str,
    test_data_path: Path,
    database_dir: Path,
    tool_schema_path: Path,
    system_prompt: str,
    overseer_model: str = "deepseek-v3.2",
    output_dir: Path | None = None,
    workers: int = 10,
    max_llm_calls: int = 100,
    runs: int = 1,
    infra_retry_limit: int = DEFAULT_INFRA_RETRY_LIMIT,
    rerun_ids: list[int] | None = None,
    system: str = "A",
    database_dir_by_run: dict[int, Path] | None = None,
    output_dir_by_run: dict[int, Path] | None = None,
    shared_oversight_cache_root: Path | None = None,
    trace_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    with test_data_path.open("r", encoding="utf-8") as fh:
        test_data = json.load(fh)

    if rerun_ids is not None:
        rerun_ids_set = {str(sample_id) for sample_id in rerun_ids}
        original_count = len(test_data)
        test_data = [
            sample for sample in test_data if str(sample.get("id")) in rerun_ids_set
        ]
        print(
            f"  🔄 Filtered {original_count} samples to {len(test_data)} samples for rerun"
        )
        if not test_data:
            return {
                "total": 0,
                "success": 0,
                "failed": 0,
                "invalid": 0,
                "elapsed_time": 0,
                "results": [],
            }

    print(f"\n{'=' * 80}")
    print("Agent Inference")
    print(f"{'=' * 80}")
    print(f"Model: {model}")
    print(f"Samples: {len(test_data)}")
    print(f"Runs: {runs}")
    print(f"Workers: {workers}")
    print(f"Infra retry limit: {infra_retry_limit}")
    progress = InferenceProgressReporter(
        domain="shopping",
        samples_per_run=len(test_data),
        runs=runs,
    )
    print(f"Execution mode: {progress.execution_mode(workers)}")
    print(f"{'=' * 80}\n")

    system_config = build_system_config(
        system_name=system,
        executor_model=model,
        overseer_model=overseer_model,
        max_steps=max_llm_calls,
        num_runs=runs,
    )
    model_identities = system_model_identities(system_config)
    started_at = time.time()
    loggers: dict[int, StructuredLogger] = {}

    def resolve_run_database_dir(run_id: int) -> Path:
        return _resolve_run_database_dir(
            database_dir, run_id, database_dir_by_run, runs
        )

    def resolve_run_output_dir(run_id: int) -> Path | None:
        run_output_dir = _resolve_run_output_dir(
            output_dir,
            run_id,
            output_dir_by_run,
            runs,
        )
        if run_output_dir is not None:
            run_output_dir.mkdir(parents=True, exist_ok=True)
        return run_output_dir

    def get_logger(run_id: int) -> StructuredLogger:
        if run_id not in loggers:
            loggers[run_id] = StructuredLogger(resolve_run_output_dir(run_id))
        return loggers[run_id]

    async def process_sample(sample: object, run_id: int) -> dict[str, Any]:
        sample = dict(sample)
        sample_id = str(sample.get("id", "unknown"))
        query = str(sample.get("query", ""))
        run_database_dir = resolve_run_database_dir(run_id)
        logger = get_logger(run_id)
        sample_trace_id = trace_id or build_langfuse_trace_id(
            session_id,
            "shopping",
            model,
            run_id,
            sample_id,
        )
        case_snapshot = (
            _prepare_case_retry_snapshot(run_database_dir, sample_id)
            if infra_retry_limit > 0
            else None
        )

        try:
            max_attempts = max(int(infra_retry_limit), 0) + 1
            for attempt_number in range(1, max_attempts + 1):
                if attempt_number > 1:
                    _restore_case_retry_snapshot(case_snapshot)

                state = ConversationState(
                    task_id=sample_id,
                    domain="shopping",
                    complexity=int(sample.get("level", 0) or 0) or None,
                    system_config_name=system,
                )

                try:
                    runner = ShoppingAgentRunner(
                        model=system_config.executor_provider.alias,
                        sample_id=sample_id,
                        database_base_path=str(run_database_dir),
                        tool_schema_path=str(tool_schema_path),
                    )
                    result = await runner.run_task(
                        user_query=query,
                        system_prompt=system_prompt,
                        state=state,
                        system_config=system_config,
                        logger=logger,
                        run_id=run_id,
                        save_messages=True,
                        sample_id=sample_id,
                        shared_oversight_cache_root=shared_oversight_cache_root,
                        trace_id=sample_trace_id,
                        session_id=session_id,
                    )
                    failure_subtype = failure_subtype_from_stop_reason(
                        result.state.final_stop_reason
                    )
                    observation_valid = observation_valid_for_failure_subtype(
                        failure_subtype
                    )
                    await logger.log_result(
                        {
                            **result.state.to_metrics(),
                            "task_id": sample_id,
                            "run_id": run_id,
                            "system": system,
                            "domain": "shopping",
                            "success": True,
                            "failure_subtype": failure_subtype,
                            "observation_valid": observation_valid,
                            "final_output": result.output,
                            "model_identities": model_identities,
                        }
                    )
                    await progress.record_completion(
                        run_id=run_id,
                        sample_id=sample_id,
                        success=True,
                        elapsed_seconds=state.wall_time_seconds,
                    )
                    return {
                        "id": sample_id,
                        "query": query,
                        "model": model,
                        "messages": result.messages,
                        "elapsed_time": state.wall_time_seconds,
                        "success": True,
                        "failure_subtype": failure_subtype,
                        "observation_valid": observation_valid,
                        "model_identities": model_identities,
                    }
                except Exception as exc:
                    if not state.end_time:
                        state.finish()
                    error_payload = serialize_exception(exc)
                    failure_subtype = classify_exception_failure_subtype(exc)
                    observation_valid = observation_valid_for_failure_subtype(
                        failure_subtype
                    )
                    should_retry = (
                        failure_subtype == FAILURE_SUBTYPE_INFRA_TRANSIENT
                        and attempt_number < max_attempts
                    )
                    if should_retry:
                        print(
                            "⚠️  Shopping sample invalid due to transient infrastructure error "
                            f"(sample={sample_id}, run={run_id}, attempt={attempt_number}/{max_attempts}); "
                            "retrying same seed/config."
                        )
                        await logger.log_event(
                            "task_invalid_attempt",
                            {
                                "domain": "shopping",
                                "task_id": sample_id,
                                "run_id": run_id,
                                "attempt": attempt_number,
                                "max_attempts": max_attempts,
                                "failure_subtype": failure_subtype,
                                "observation_valid": False,
                                "error": error_payload,
                            },
                        )
                        continue

                    if failure_subtype == FAILURE_SUBTYPE_INFRA_TRANSIENT:
                        _restore_case_retry_snapshot(case_snapshot)

                    await logger.log_event(
                        "task_error",
                        {
                            "domain": "shopping",
                            "task_id": sample_id,
                            "run_id": run_id,
                            "failure_subtype": failure_subtype,
                            "observation_valid": observation_valid,
                            "error": error_payload,
                            "model_identities": model_identities,
                        },
                    )
                    await logger.log_result(
                        {
                            **state.to_metrics(),
                            "task_id": sample_id,
                            "run_id": run_id,
                            "system": system,
                            "domain": "shopping",
                            "success": False,
                            "failure_subtype": failure_subtype,
                            "observation_valid": observation_valid,
                            "final_output": None,
                            "error": error_payload,
                            "model_identities": model_identities,
                        }
                    )
                    await progress.record_completion(
                        run_id=run_id,
                        sample_id=sample_id,
                        success=False,
                        error_summary=_error_summary(error_payload),
                    )
                    traceback.print_exc()
                    return {
                        "id": sample_id,
                        "query": query,
                        "success": False,
                        "failure_subtype": failure_subtype,
                        "observation_valid": observation_valid,
                        "error": error_payload.get("message", str(exc)),
                        "model_identities": model_identities,
                    }
        finally:
            _cleanup_case_retry_snapshot(case_snapshot)

    raw_results = await run_experiment(
        test_data, process_sample, parallel=workers, runs=runs
    )
    results: list[dict[str, Any]] = []
    for item in raw_results:
        if isinstance(item, Exception):
            results.append(
                {
                    "success": False,
                    "failure_subtype": "none",
                    "observation_valid": True,
                    "error": str(item),
                }
            )
        else:
            results.append(item)

    observed_results = [
        result for result in results if result.get("observation_valid", True)
    ]
    success_count = sum(1 for result in observed_results if result.get("success"))
    return {
        "total": len(observed_results),
        "success": success_count,
        "failed": len(observed_results) - success_count,
        "invalid": len(results) - len(observed_results),
        "elapsed_time": time.time() - started_at,
        "runs": runs,
        "results": results,
    }


def run_agent_inference(
    model: str,
    test_data_path: Path,
    database_dir: Path,
    tool_schema_path: Path,
    system_prompt: str,
    overseer_model: str = "deepseek-v3.2",
    output_dir: Path | None = None,
    workers: int = 10,
    max_llm_calls: int = 100,
    runs: int = 1,
    infra_retry_limit: int = DEFAULT_INFRA_RETRY_LIMIT,
    rerun_ids: list[int] | None = None,
    system: str = "A",
    database_dir_by_run: dict[int, Path] | None = None,
    output_dir_by_run: dict[int, Path] | None = None,
    shared_oversight_cache_root: Path | None = None,
    trace_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    async def _run_with_cleanup() -> dict[str, Any]:
        try:
            return await run_agent_inference_async(
                model=model,
                test_data_path=test_data_path,
                database_dir=database_dir,
                tool_schema_path=tool_schema_path,
                system_prompt=system_prompt,
                overseer_model=overseer_model,
                output_dir=output_dir,
                workers=workers,
                max_llm_calls=max_llm_calls,
                runs=runs,
                infra_retry_limit=infra_retry_limit,
                rerun_ids=rerun_ids,
                system=system,
                database_dir_by_run=database_dir_by_run,
                output_dir_by_run=output_dir_by_run,
                shared_oversight_cache_root=shared_oversight_cache_root,
                trace_id=trace_id,
                session_id=session_id,
            )
        finally:
            flush_langfuse()

    return asyncio.run(_run_with_cleanup())
