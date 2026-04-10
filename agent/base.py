from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from oversight import ConversationState


@dataclass(slots=True)
class TaskResult:
    task_id: str
    run_id: int
    output: str
    messages: list[Any]
    state: ConversationState
