from __future__ import annotations

import difflib
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from .contracts import ExecutionContract, TaskChecklist
from .triggers import classify_mutating_tool, normalize_arguments


def _hash_arguments(arguments: Any) -> str:
    normalized = normalize_arguments(arguments)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _summarize_result(result: Any, limit: int = 160) -> str:
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _parse_result_payload(result: Any) -> Any:
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return result
    return result


def _increment_counter(counter: dict[str, int], key: str | None) -> None:
    if not key:
        return
    counter[key] = counter.get(key, 0) + 1


@dataclass(slots=True)
class ConversationState:
    task_id: str
    domain: str
    complexity: int | None
    system_config_name: str
    executor_calls: int = 0
    executor_tokens_in: int = 0
    executor_tokens_out: int = 0
    overseer_calls: int = 0
    overseer_tokens_in: int = 0
    overseer_tokens_out: int = 0
    tool_calls_history: list[dict[str, Any]] = field(default_factory=list)
    tool_call_count: int = 0
    triggers_fired: list[dict[str, Any]] = field(default_factory=list)
    final_stop_reason: str | None = None
    final_output_present: bool = False
    max_steps_hit: bool = False
    start_time: float = 0.0
    end_time: float = 0.0
    execution_contract: ExecutionContract | None = None
    task_checklist: TaskChecklist | None = None
    pending_executor_notice: str | None = None
    last_authoritative_cart_snapshot: dict[str, Any] | None = None
    last_authoritative_read_step: int | None = None
    last_mutation_step: int | None = None
    final_verification_retry_count: int = 0
    final_verification_result: str = "not_applicable"
    blocked_mutation_count: int = 0
    blocked_mutation_repeat_count: int = 0
    last_blocked_mutation_tool_name: str | None = None
    last_blocked_mutation_arguments_normalized: str | None = None
    overseer_invocation_count_by_trigger: dict[str, int] = field(default_factory=dict)
    intervention_count_by_action: dict[str, int] = field(default_factory=dict)
    executor_cost_usd: float = 0.0
    overseer_cost_usd: float = 0.0
    _executor_cost_known: bool = field(default=False, init=False, repr=False)
    _overseer_cost_known: bool = field(default=False, init=False, repr=False)

    def begin(self) -> None:
        if not self.start_time:
            self.start_time = time.time()

    def finish(self) -> None:
        self.end_time = time.time()

    def record_executor_call(self, response: Any, *, cost: float | None = None) -> None:
        self.executor_calls += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.executor_tokens_in += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.executor_tokens_out += int(getattr(usage, "completion_tokens", 0) or 0)
        if cost is not None:
            self._executor_cost_known = True
            self.executor_cost_usd += cost

    def record_overseer_call(self, response: Any, *, cost: float | None = None) -> None:
        self.overseer_calls += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.overseer_tokens_in += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.overseer_tokens_out += int(getattr(usage, "completion_tokens", 0) or 0)
        if cost is not None:
            self._overseer_cost_known = True
            self.overseer_cost_usd += cost

    def record_blocked_mutation_attempt(
        self,
        *,
        tool_name: str,
        arguments: Any,
        similarity_threshold: float = 0.92,
    ) -> int:
        arguments_normalized = normalize_arguments(arguments)
        prior_arguments = self.last_blocked_mutation_arguments_normalized
        prior_tool_name = self.last_blocked_mutation_tool_name
        similarity = 0.0
        if prior_arguments:
            similarity = (
                1.0
                if prior_arguments == arguments_normalized
                else difflib.SequenceMatcher(
                    None,
                    prior_arguments,
                    arguments_normalized,
                ).ratio()
            )

        repeated = (
            prior_tool_name == tool_name
            and prior_arguments is not None
            and similarity >= similarity_threshold
        )
        self.blocked_mutation_repeat_count = (
            self.blocked_mutation_repeat_count + 1 if repeated else 1
        )
        self.blocked_mutation_count += 1
        self.last_blocked_mutation_tool_name = tool_name
        self.last_blocked_mutation_arguments_normalized = arguments_normalized
        return self.blocked_mutation_repeat_count

    def record_oversight_decision(
        self,
        step: int,
        action: Any,
        *,
        tool_index: int | None = None,
    ) -> None:
        trigger_type = getattr(action, "trigger_type", None)
        intervention_type = getattr(action, "intervention_type", None)
        if getattr(action, "overseer_invoked", False):
            _increment_counter(self.overseer_invocation_count_by_trigger, trigger_type)
        if intervention_type is not None:
            _increment_counter(self.intervention_count_by_action, intervention_type)
        if trigger_type is None and intervention_type is None:
            return
        self.triggers_fired.append(
            {
                "step": step,
                "tool_index": tool_index,
                "trigger_type": trigger_type,
                "trigger_reason": getattr(action, "trigger_reason", None),
                "intervention_type": intervention_type,
                "should_intervene": bool(getattr(action, "should_intervene", False)),
                "overseer_invoked": bool(getattr(action, "overseer_invoked", False)),
            }
        )

    def record_tool_call(
        self,
        tool_call: dict[str, Any],
        result: Any,
        *,
        phase: str = "",
        step_index: int = 0,
        tool_index: int = 0,
        mutating_tools: tuple[str, ...] = (),
    ) -> None:
        tool_name = str(tool_call.get("name") or "")
        arguments_raw = tool_call.get("arguments")
        arguments_normalized = normalize_arguments(arguments_raw)
        result_payload = _parse_result_payload(result)
        is_mutating = classify_mutating_tool(
            tool_name,
            mutating_tools=mutating_tools,
        )["is_mutating"]

        self.tool_call_count += 1
        record = {
            "phase": phase,
            "step_index": step_index,
            "tool_index": tool_index,
            "tool_name": tool_name,
            "arguments_raw": arguments_raw,
            "arguments_normalized": arguments_normalized,
            "args_hash": _hash_arguments(arguments_raw),
            "is_mutating": is_mutating,
            "result_summary": _summarize_result(result),
            "result_payload": result_payload,
        }
        self.tool_calls_history.append(record)

        if tool_name == "get_cart_info" and isinstance(result_payload, dict):
            self.last_authoritative_cart_snapshot = result_payload
            self.last_authoritative_read_step = step_index
        if is_mutating:
            self.last_mutation_step = step_index
            self.blocked_mutation_repeat_count = 0
            self.last_blocked_mutation_tool_name = None
            self.last_blocked_mutation_arguments_normalized = None

    def record_final_outcome(
        self, *, stop_reason: str, output: str | None, max_steps_hit: bool
    ) -> None:
        self.final_stop_reason = stop_reason
        self.final_output_present = bool(output)
        self.max_steps_hit = max_steps_hit

    @property
    def wall_time_seconds(self) -> float:
        if not self.start_time:
            return 0.0
        end_time = self.end_time or time.time()
        return max(end_time - self.start_time, 0.0)

    def _total_cost_usd(self) -> float | None:
        if not self._executor_cost_known or not self._overseer_cost_known:
            return None
        return self.executor_cost_usd + self.overseer_cost_usd

    def to_metrics(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "complexity": self.complexity,
            "system": self.system_config_name,
            "executor_calls": self.executor_calls,
            "executor_tokens_in": self.executor_tokens_in,
            "executor_tokens_out": self.executor_tokens_out,
            "overseer_calls": self.overseer_calls,
            "overseer_tokens_in": self.overseer_tokens_in,
            "overseer_tokens_out": self.overseer_tokens_out,
            "executor_cost_usd": (
                self.executor_cost_usd if self._executor_cost_known else None
            ),
            "overseer_cost_usd": (
                self.overseer_cost_usd if self._overseer_cost_known else None
            ),
            "total_cost_usd": self._total_cost_usd(),
            "tool_call_count": self.tool_call_count,
            "tool_calls_history": self.tool_calls_history,
            "triggers_fired": self.triggers_fired,
            "final_stop_reason": self.final_stop_reason,
            "final_output_present": self.final_output_present,
            "max_steps_hit": self.max_steps_hit,
            "final_verification_result": self.final_verification_result,
            "final_verification_retry_count": self.final_verification_retry_count,
            "blocked_mutation_count": self.blocked_mutation_count,
            "blocked_mutation_repeat_count": self.blocked_mutation_repeat_count,
            "overseer_invocation_count_by_trigger": self.overseer_invocation_count_by_trigger,
            "intervention_count_by_action": self.intervention_count_by_action,
            "wall_time_seconds": self.wall_time_seconds,
        }
