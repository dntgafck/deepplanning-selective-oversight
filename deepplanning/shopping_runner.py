from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path
from typing import Any

from .config import (
    BENCHMARK_ROOT,
    DATA_ROOT,
    OUTPUT_ROOT,
    clear_imported_modules,
    ensure_directory,
    ensure_repo_imports,
    filter_samples_by_ids,
    load_dotenv,
    load_json_file,
    load_model_config,
    parse_id_list,
    parse_int_list,
    parse_space_separated,
    resolve_repo_path,
    write_json_file,
)

SHOPPING_ROOT = BENCHMARK_ROOT / "shoppingplanning"
SHOPPING_DATA_ROOT = DATA_ROOT / "shopping"
SHOPPING_OUTPUT_ROOT = OUTPUT_ROOT / "shopping"

ensure_repo_imports()

from agent import shopping as shopping_agent_runner


def import_modules() -> tuple[object, object]:
    clear_imported_modules(["evaluation", "score_statistics"])
    sys.path.insert(0, str(SHOPPING_ROOT))

    import evaluation.evaluation_pipeline as evaluation_pipeline
    import evaluation.score_statistics as score_statistics

    return evaluation_pipeline, score_statistics


def evaluate_database(
    database_dir: Path, report_dir: Path, evaluation_pipeline: object
) -> None:
    case_dirs = sorted(
        path
        for path in database_dir.iterdir()
        if path.is_dir() and path.name.startswith("case_")
    )
    if not case_dirs:
        raise RuntimeError(f"No shopping case directories found in {database_dir}")

    ensure_directory(report_dir)
    all_results = []
    for case_dir in case_dirs:
        result = evaluation_pipeline.evaluate_single_case(case_dir)
        all_results.append(result)
        if result.get("success"):
            evaluation_pipeline.generate_case_report(result, report_dir)

    evaluation_pipeline.generate_summary_report(all_results, report_dir)


def write_statistics(
    model: str, result_report_root: Path, score_statistics: object
) -> None:
    statistics = score_statistics.calculate_model_statistics(model, result_report_root)
    if statistics is None:
        raise RuntimeError(f"Failed to calculate shopping statistics for model {model}")

    output_file = result_report_root / f"{model}_statistics.json"
    write_json_file(output_file, statistics)
    print(f"✅ Shopping statistics saved to: {output_file}")


def prepare_run_inputs(
    level: int,
    source_database_dir: Path,
    run_database_dir: Path,
    sample_ids: list[str] | None,
) -> Path:
    source_test_data_path = SHOPPING_ROOT / "data" / f"level_{level}_query_meta.json"
    if sample_ids is None:
        shutil.copytree(source_database_dir, run_database_dir)
        return source_test_data_path

    all_samples = load_json_file(source_test_data_path)
    selected_samples = filter_samples_by_ids(all_samples, sample_ids)
    if not selected_samples:
        raise ValueError(f"No shopping samples matched ids: {', '.join(sample_ids)}")

    ensure_directory(run_database_dir)
    for sample in selected_samples:
        case_dir = source_database_dir / f"case_{sample['id']}"
        if not case_dir.exists():
            raise FileNotFoundError(f"Shopping case directory missing: {case_dir}")
        shutil.copytree(case_dir, run_database_dir / case_dir.name)

    filtered_test_data_path = (
        run_database_dir.parent / f"{run_database_dir.name}_query_meta.json"
    )
    write_json_file(filtered_test_data_path, selected_samples)
    return filtered_test_data_path


def run(
    *,
    models: str | list[str] | None = None,
    levels: int | str | list[int] | list[str] | None = None,
    sample_ids: int | str | list[int] | list[str] | None = None,
    system: str = "A",
    workers: int = 50,
    max_llm_calls: int = 400,
    runs: int = 4,
    output_root: str | Path | None = None,
) -> None:
    load_dotenv()
    evaluation_pipeline, score_statistics = import_modules()

    model_names = parse_space_separated(models, ["qwen3-14b"])
    level_numbers = parse_int_list(levels, [1, 2, 3])
    selected_sample_ids = parse_id_list(sample_ids)
    run_count = int(runs)
    output_root_path = resolve_repo_path(output_root or SHOPPING_OUTPUT_ROOT)

    tool_schema_path = SHOPPING_ROOT / "tools" / "shopping_tool_schema.json"

    for model in model_names:
        load_model_config(model)
        model_output_root = ensure_directory(output_root_path / model)

        for level in level_numbers:
            source_database_dir = SHOPPING_DATA_ROOT / f"database_level{level}"
            if not source_database_dir.exists():
                raise FileNotFoundError(
                    f"Shopping data missing at {source_database_dir}. Ensure DVC data bootstrap is present under data/deepplanning/."
                )

            timestamp = time.strftime("%Y%m%d%H%M")
            output_name = f"database_{model}_level{level}_{timestamp}"
            run_database_dirs: dict[int, Path] = {}
            run_output_dirs: dict[int, Path] = {}
            run_report_dirs: dict[int, Path] = {}
            test_data_path: Path | None = None

            for run_id in range(run_count):
                run_root = ensure_directory(model_output_root / f"run_{run_id}")
                database_output_root = ensure_directory(run_root / "database_infered")
                log_output_root = ensure_directory(run_root / "logs")
                result_report_root = ensure_directory(run_root / "result_report")

                run_database_dir = database_output_root / output_name
                run_output_dir = ensure_directory(log_output_root / output_name)

                if run_database_dir.exists():
                    shutil.rmtree(run_database_dir)

                current_test_data_path = prepare_run_inputs(
                    level=level,
                    source_database_dir=source_database_dir,
                    run_database_dir=run_database_dir,
                    sample_ids=selected_sample_ids,
                )
                if test_data_path is None:
                    test_data_path = current_test_data_path

                run_database_dirs[run_id] = run_database_dir
                run_output_dirs[run_id] = run_output_dir
                run_report_dirs[run_id] = result_report_root / output_name

            print("\n🚀 Shopping benchmark")
            print(f"   Model: {model}")
            print(f"   Level: {level}")
            print(f"   Runs: {run_count}")
            print(f"   Output root: {model_output_root}")
            if selected_sample_ids is not None:
                print(f"   Sample IDs: {', '.join(selected_sample_ids)}")

            system_prompt = shopping_agent_runner.get_system_prompt(level)
            results = shopping_agent_runner.run_agent_inference(
                model=model,
                test_data_path=(
                    test_data_path
                    if test_data_path is not None
                    else SHOPPING_ROOT / "data" / f"level_{level}_query_meta.json"
                ),
                database_dir=run_database_dirs[0],
                tool_schema_path=tool_schema_path,
                output_dir=run_output_dirs[0],
                system_prompt=system_prompt,
                workers=int(workers),
                max_llm_calls=int(max_llm_calls),
                runs=run_count,
                system=str(system),
                database_dir_by_run=run_database_dirs,
                output_dir_by_run=run_output_dirs,
            )
            print(
                f"✅ Shopping inference complete: {results['success']}/{results['total']} succeeded"
            )

            for run_id in range(run_count):
                evaluate_database(
                    database_dir=run_database_dirs[run_id],
                    report_dir=run_report_dirs[run_id],
                    evaluation_pipeline=evaluation_pipeline,
                )

        for run_id in range(run_count):
            write_statistics(
                model,
                model_output_root / f"run_{run_id}" / "result_report",
                score_statistics,
            )
