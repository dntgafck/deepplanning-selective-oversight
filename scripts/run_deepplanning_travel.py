from __future__ import annotations

import fire

from deepplanning import travel_runner as travel_runtime
from deepplanning.orchestration import run_travel_compat as run

GENERATED_DATA_ONLY_DIRNAME = travel_runtime.GENERATED_DATA_ONLY_DIRNAME
GENERATED_DATA_ONLY_FILENAME = travel_runtime.GENERATED_DATA_ONLY_FILENAME
TRAVEL_DATA_ROOT = travel_runtime.TRAVEL_DATA_ROOT
TRAVEL_OUTPUT_ROOT = travel_runtime.TRAVEL_OUTPUT_ROOT
TRAVEL_ROOT = travel_runtime.TRAVEL_ROOT
TRAVEL_RUN_STATUS_FILENAME = travel_runtime.TRAVEL_RUN_STATUS_FILENAME
import_modules = travel_runtime.import_modules
load_dotenv = travel_runtime.load_dotenv
load_model_config = travel_runtime.load_model_config
prepare_test_data = travel_runtime.prepare_test_data
run_language = travel_runtime.run_language
travel_runner = travel_runtime.travel_agent_runner

__all__ = [
    "GENERATED_DATA_ONLY_DIRNAME",
    "GENERATED_DATA_ONLY_FILENAME",
    "TRAVEL_DATA_ROOT",
    "TRAVEL_OUTPUT_ROOT",
    "TRAVEL_ROOT",
    "TRAVEL_RUN_STATUS_FILENAME",
    "import_modules",
    "load_dotenv",
    "load_model_config",
    "prepare_test_data",
    "run",
    "run_language",
    "travel_runner",
]


def main() -> None:
    fire.Fire(run)


if __name__ == "__main__":
    main()
