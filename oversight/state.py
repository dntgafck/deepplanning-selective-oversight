from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any


def _hash_arguments(arguments: str | dict[str, Any] | None) -> str:
    if arguments is None:
        return ""
    if isinstance(arguments, dict):
        raw = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
    else:
        raw = arguments
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _summarize_result(result: Any, limit: int = 160) -> str:
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


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
    triggers_fired: list[dict[str, Any]] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    def begin(self) -> None:
        if not self.start_time:
            self.start_time = time.time()

    def finish(self) -> None:
        self.end_time = time.time()

    def record_executor_call(self, response: Any) -> None:
        self.executor_calls += 1
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.executor_tokens_in += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.executor_tokens_out += int(getattr(usage, "completion_tokens", 0) or 0)

    def record_overseer_call(self, response: Any) -> None:
        self.overseer_calls += 1
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.overseer_tokens_in += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.overseer_tokens_out += int(getattr(usage, "completion_tokens", 0) or 0)

    def record_oversight_decision(self, step: int, action: Any) -> None:
        self.triggers_fired.append(
            {
                "step": step,
                "trigger_reason": getattr(action, "trigger_reason", None),
                "intervention_type": getattr(action, "intervention_type", None),
                "should_intervene": bool(getattr(action, "should_intervene", False)),
            }
        )

    def record_tool_call(self, tool_call: dict[str, Any], result: Any) -> None:
        self.tool_calls_history.append(
            {
                "tool_name": tool_call.get("name"),
                "args_hash": _hash_arguments(tool_call.get("arguments")),
                "result_summary": _summarize_result(result),
            }
        )

    @property
    def wall_time_seconds(self) -> float:
        if not self.start_time:
            return 0.0
        end_time = self.end_time or time.time()
        return max(end_time - self.start_time, 0.0)

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
            "tool_calls_history": self.tool_calls_history,
            "triggers_fired": self.triggers_fired,
            "wall_time_seconds": self.wall_time_seconds,
        }
