from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class StructuredLogger:
    output_dir: Path | None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self) -> None:
        if self.output_dir is None:
            self.events_path = None
            self.results_path = None
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "agent_events.jsonl"
        self.results_path = self.output_dir / "task_results.jsonl"

    async def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.output_dir is None or self.events_path is None:
            return
        record = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "event_type": event_type,
            **payload,
        }
        await self._append_jsonl(self.events_path, record)

    async def log_result(self, payload: dict[str, Any]) -> None:
        if self.output_dir is None or self.results_path is None:
            return
        record = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            **payload,
        }
        await self._append_jsonl(self.results_path, record)

    async def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, default=str) + "\n"
        async with self._lock:
            await asyncio.to_thread(self._append_line, path, line)

    @staticmethod
    def _append_line(path: Path, line: str) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
