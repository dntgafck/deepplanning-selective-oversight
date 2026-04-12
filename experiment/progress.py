from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass(slots=True)
class RunProgress:
    completed: int = 0
    success: int = 0
    failed: int = 0


class InferenceProgressReporter:
    def __init__(self, *, domain: str, samples_per_run: int, runs: int) -> None:
        self.domain = domain
        self.samples_per_run = max(int(samples_per_run), 0)
        self.runs = max(int(runs), 1)
        self.overall_total = self.samples_per_run * self.runs
        self._lock = asyncio.Lock()
        self._runs = {run_id: RunProgress() for run_id in range(self.runs)}

    def execution_mode(self, workers: int) -> str:
        if self.runs > 1:
            return (
                f"concurrent across runs ({self.runs} runs, shared worker pool: {workers}, "
                f"{self.samples_per_run} tasks/run, {self.overall_total} total tasks)"
            )
        return f"single run ({self.samples_per_run} tasks, worker pool: {workers})"

    async def record_completion(
        self,
        *,
        run_id: int,
        sample_id: str,
        success: bool,
        elapsed_seconds: float | None = None,
        error_summary: str | None = None,
    ) -> None:
        async with self._lock:
            run_progress = self._runs[run_id]
            run_progress.completed += 1
            if success:
                run_progress.success += 1
            else:
                run_progress.failed += 1

            overall_completed = sum(
                progress.completed for progress in self._runs.values()
            )
            overall_success = sum(progress.success for progress in self._runs.values())
            overall_failed = sum(progress.failed for progress in self._runs.values())
            run_remaining = max(self.samples_per_run - run_progress.completed, 0)
            overall_remaining = max(self.overall_total - overall_completed, 0)
            run_label = f"run {run_id + 1}/{self.runs}"

            status = "completed" if success else "failed"
            parts = [f"{sample_id} {status}"]
            if elapsed_seconds is not None:
                parts.append(f"{elapsed_seconds:.2f}s")
            if error_summary:
                parts.append(error_summary)
            detail = " | ".join(parts)

            prefix = "✅" if success else "❌"
            print(
                f"{prefix} {self.domain.capitalize()} progress | {run_label} | {detail} | "
                f"run: {run_progress.completed}/{self.samples_per_run} done, {run_remaining} left "
                f"(ok={run_progress.success}, failed={run_progress.failed}) | "
                f"overall: {overall_completed}/{self.overall_total} done, {overall_remaining} left "
                f"(ok={overall_success}, failed={overall_failed})"
            )
