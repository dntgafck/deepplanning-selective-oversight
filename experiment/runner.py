from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import TypeVar

T = TypeVar("T")


async def run_experiment(
    tasks: Iterable[object],
    worker: Callable[[object, int], Awaitable[T]],
    parallel: int = 20,
    runs: int = 1,
) -> list[T | Exception]:
    semaphore = asyncio.Semaphore(parallel)

    async def run_with_limit(task: object, run_id: int) -> T:
        async with semaphore:
            return await worker(task, run_id)

    coroutines = [
        run_with_limit(task, run_id) for run_id in range(runs) for task in tasks
    ]
    return await asyncio.gather(*coroutines, return_exceptions=True)
