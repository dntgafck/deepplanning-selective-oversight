from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import fire

try:
    from deepplanning_common import (
        BENCHMARK_ROOT,
        DATA_ROOT,
        OUTPUT_ROOT,
        clear_imported_modules,
        compose_config,
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
except ModuleNotFoundError:
    from scripts.deepplanning_common import (
        BENCHMARK_ROOT,
        DATA_ROOT,
        OUTPUT_ROOT,
        clear_imported_modules,
        compose_config,
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

from agent import shopping as shopping_runner


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


def main() -> None:
    run()


def run(
    models: str | None = None,
    levels: str | None = None,
    sample_ids: str | None = None,
    system: str | None = None,
    workers: int | None = None,
    max_llm_calls: int | None = None,
    output_root: str | None = None,
) -> None:
    load_dotenv()
    evaluation_pipeline, score_statistics = import_modules()

    cfg = compose_config(
        "shopping",
        {
            "models": models,
            "levels": levels,
            "sample_ids": sample_ids,
            "system": system,
            "workers": workers,
            "max_llm_calls": max_llm_calls,
            "output_root": output_root,
        },
    )

    model_names = parse_space_separated(cfg.models, ["qwen-plus"])
    level_numbers = parse_int_list(cfg.levels, [1, 2, 3])
    selected_sample_ids = parse_id_list(cfg.sample_ids)
    output_root_path = resolve_repo_path(cfg.output_root)

    tool_schema_path = SHOPPING_ROOT / "tools" / "shopping_tool_schema.json"
    database_output_root = ensure_directory(output_root_path / "database_infered")
    result_report_root = ensure_directory(output_root_path / "result_report")

    for model in model_names:
        load_model_config(model)

        for level in level_numbers:
            source_database_dir = SHOPPING_DATA_ROOT / f"database_level{level}"
            if not source_database_dir.exists():
                raise FileNotFoundError(
                    f"Shopping data missing at {source_database_dir}. Ensure DVC data bootstrap is present under data/deepplanning/."
                )

            timestamp = time.strftime("%Y%m%d%H%M")
            output_name = f"database_{model}_level{level}_{timestamp}"
            run_database_dir = database_output_root / output_name

            if run_database_dir.exists():
                shutil.rmtree(run_database_dir)
            test_data_path = prepare_run_inputs(
                level=level,
                source_database_dir=source_database_dir,
                run_database_dir=run_database_dir,
                sample_ids=selected_sample_ids,
            )

            print("\n🚀 Shopping benchmark")
            print(f"   Model: {model}")
            print(f"   Level: {level}")
            print(f"   Database: {run_database_dir}")
            if selected_sample_ids is not None:
                print(f"   Sample IDs: {', '.join(selected_sample_ids)}")

            system_prompt = shopping_runner.get_system_prompt(level)
            results = shopping_runner.run_agent_inference(
                model=model,
                test_data_path=test_data_path,
                database_dir=run_database_dir,
                tool_schema_path=tool_schema_path,
                system_prompt=system_prompt,
                workers=int(cfg.workers),
                max_llm_calls=int(cfg.max_llm_calls),
                system=str(cfg.system),
            )
            print(
                f"✅ Shopping inference complete: {results['success']}/{results['total']} succeeded"
            )

            evaluate_database(
                database_dir=run_database_dir,
                report_dir=result_report_root / output_name,
                evaluation_pipeline=evaluation_pipeline,
            )

        write_statistics(model, result_report_root, score_statistics)


if __name__ == "__main__":
    fire.Fire(run)
