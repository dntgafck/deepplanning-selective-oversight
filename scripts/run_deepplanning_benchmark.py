from __future__ import annotations

import fire

from deepplanning.aggregation import aggregate_results
from deepplanning.orchestration import run_benchmark_compat as run

__all__ = ["aggregate_results", "run"]


def main() -> None:
    fire.Fire(run)


if __name__ == "__main__":
    main()
