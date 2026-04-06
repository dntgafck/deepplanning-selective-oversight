from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import fire
from deepplanning_common import (
    BENCHMARK_ROOT,
    DATA_ROOT,
    OUTPUT_ROOT,
    compose_config,
    ensure_directory,
    load_dotenv,
    load_model_config,
    parse_int_list,
    parse_space_separated,
    resolve_repo_path,
)

SHOPPING_ROOT = BENCHMARK_ROOT / "shoppingplanning"
SHOPPING_DATA_ROOT = DATA_ROOT / "shopping"
SHOPPING_OUTPUT_ROOT = OUTPUT_ROOT / "shopping"


def import_modules() -> tuple[object, object, object, object, object]:
    sys.path.insert(0, str(SHOPPING_ROOT))

    import agent.call_llm as shopping_call_llm
    import agent.shopping_agent as shopping_agent
    import evaluation.evaluation_pipeline as evaluation_pipeline
    import evaluation.score_statistics as score_statistics
    from agent.prompts import prompt_lib

    shopping_call_llm.load_model_config = load_model_config
    return (
        shopping_call_llm,
        prompt_lib,
        shopping_agent,
        evaluation_pipeline,
        score_statistics,
    )


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
    output_file.write_text(
        json.dumps(statistics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"✅ Shopping statistics saved to: {output_file}")


def main() -> None:
    run()


def run(
    models: str | None = None,
    levels: str | None = None,
    workers: int | None = None,
    max_llm_calls: int | None = None,
    output_root: str | None = None,
) -> None:
    load_dotenv()
    _, prompt_lib, shopping_agent, evaluation_pipeline, score_statistics = (
        import_modules()
    )

    cfg = compose_config(
        "shopping",
        {
            "models": models,
            "levels": levels,
            "workers": workers,
            "max_llm_calls": max_llm_calls,
            "output_root": output_root,
        },
    )

    model_names = parse_space_separated(cfg.models, ["qwen-plus"])
    level_numbers = parse_int_list(cfg.levels, [1, 2, 3])
    output_root_path = resolve_repo_path(cfg.output_root)

    test_data_dir = SHOPPING_ROOT / "data"
    tool_schema_path = SHOPPING_ROOT / "tools" / "shopping_tool_schema.json"
    database_output_root = ensure_directory(output_root_path / "database_infered")
    result_report_root = ensure_directory(output_root_path / "result_report")

    for model in model_names:
        load_model_config(model)

        for level in level_numbers:
            source_database_dir = SHOPPING_DATA_ROOT / f"database_level{level}"
            if not source_database_dir.exists():
                raise FileNotFoundError(
                    f"Shopping data missing at {source_database_dir}. Run `pixi exec dvc repro deepplanning_data` first."
                )

            timestamp = time.strftime("%Y%m%d%H%M")
            output_name = f"database_{model}_level{level}_{timestamp}"
            run_database_dir = database_output_root / output_name

            if run_database_dir.exists():
                shutil.rmtree(run_database_dir)
            shutil.copytree(source_database_dir, run_database_dir)

            print("\n🚀 Shopping benchmark")
            print(f"   Model: {model}")
            print(f"   Level: {level}")
            print(f"   Database: {run_database_dir}")

            system_prompt = getattr(prompt_lib, f"SYSTEM_PROMPT_level{level}")
            results = shopping_agent.run_agent_inference(
                model=model,
                test_data_path=test_data_dir / f"level_{level}_query_meta.json",
                database_dir=run_database_dir,
                tool_schema_path=tool_schema_path,
                system_prompt=system_prompt,
                workers=int(cfg.workers),
                max_llm_calls=int(cfg.max_llm_calls),
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
