from __future__ import annotations

import fire

from deepplanning import shopping_runner as shopping_runtime
from deepplanning.orchestration import run_shopping_compat as run

SHOPPING_ROOT = shopping_runtime.SHOPPING_ROOT
SHOPPING_DATA_ROOT = shopping_runtime.SHOPPING_DATA_ROOT
SHOPPING_OUTPUT_ROOT = shopping_runtime.SHOPPING_OUTPUT_ROOT
evaluate_database = shopping_runtime.evaluate_database
import_modules = shopping_runtime.import_modules
load_dotenv = shopping_runtime.load_dotenv
load_model_config = shopping_runtime.load_model_config
prepare_run_inputs = shopping_runtime.prepare_run_inputs
shopping_runner = shopping_runtime.shopping_agent_runner
write_statistics = shopping_runtime.write_statistics

__all__ = [
    "SHOPPING_DATA_ROOT",
    "SHOPPING_OUTPUT_ROOT",
    "SHOPPING_ROOT",
    "evaluate_database",
    "import_modules",
    "load_dotenv",
    "load_model_config",
    "prepare_run_inputs",
    "run",
    "shopping_runner",
    "write_statistics",
]


def main() -> None:
    fire.Fire(run)


if __name__ == "__main__":
    main()
