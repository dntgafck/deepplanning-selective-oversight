from __future__ import annotations

import sys
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
    parse_space_separated,
    resolve_repo_path,
)

TRAVEL_ROOT = BENCHMARK_ROOT / "travelplanning"
TRAVEL_DATA_ROOT = DATA_ROOT / "travel" / "database"
TRAVEL_OUTPUT_ROOT = OUTPUT_ROOT / "travel"


def import_modules() -> tuple[object, object, object]:
    sys.path.insert(0, str(TRAVEL_ROOT))

    import agent.call_llm as travel_call_llm
    import evaluation.convert_report as convert_report
    import evaluation.eval_converted as eval_converted
    import run as travel_run

    travel_call_llm.load_model_config = load_model_config
    return convert_report, eval_converted, travel_run


def run_language(
    model: str,
    language: str,
    cfg: object,
    convert_report: object,
    eval_converted: object,
    travel_run: object,
) -> None:
    database_dir = TRAVEL_DATA_ROOT / f"database_{language}"
    if not database_dir.exists():
        raise FileNotFoundError(
            f"Travel data missing at {database_dir}. Run `pixi exec dvc repro deepplanning_data` first."
        )

    output_base = ensure_directory(resolve_repo_path(cfg.output_root))
    output_dir = ensure_directory(output_base / f"{model}_{language}")
    ensure_directory(output_dir / "trajectories")
    ensure_directory(output_dir / "reports")
    ensure_directory(output_dir / "converted_plans")
    ensure_directory(output_dir / "evaluation")

    test_data_path = TRAVEL_ROOT / "data" / f"travelplanning_query_{language}.json"
    tool_schema_path = TRAVEL_ROOT / "tools" / f"tool_schema_{language}.json"

    print("\n🚀 Travel benchmark")
    print(f"   Model: {model}")
    print(f"   Language: {language}")
    print(f"   Database: {database_dir}")
    print(f"   Output: {output_dir}")

    if cfg.start_from == "inference":
        inference_results = travel_run.run_agent_inference(
            model=model,
            language=language,
            test_data_path=test_data_path,
            database_dir=database_dir,
            tool_schema_path=tool_schema_path,
            output_dir=output_dir,
            workers=int(cfg.workers),
            max_llm_calls=int(cfg.max_llm_calls),
        )
        print(
            f"✅ Travel inference complete: {inference_results['success']}/{inference_results['total']} succeeded"
        )

    if cfg.start_from in {"inference", "conversion"}:
        conversion_results = convert_report.convert_reports(
            result_dir=output_dir,
            language=language,
            workers=int(cfg.workers),
            skip_existing=True,
            verbose=bool(cfg.verbose),
        )
        print(
            f"✅ Travel conversion complete: {conversion_results['converted']} converted, {conversion_results['skipped']} skipped"
        )

    evaluation_results = eval_converted.evaluate_plans(
        result_dir=output_dir,
        test_data_path=test_data_path,
        database_dir=database_dir,
        verbose=bool(cfg.verbose),
    )
    print(
        f"✅ Travel evaluation complete: composite={evaluation_results['metrics']['composite_score']:.2%}, case_acc={evaluation_results['metrics']['case_acc']:.2%}"
    )


def main() -> None:
    run()


def run(
    models: str | None = None,
    language: str | None = None,
    workers: int | None = None,
    max_llm_calls: int | None = None,
    start_from: str | None = None,
    output_root: str | None = None,
    verbose: bool | None = None,
    debug: bool | None = None,
) -> None:
    load_dotenv()
    convert_report, eval_converted, travel_run = import_modules()

    cfg = compose_config(
        "travel",
        {
            "models": models,
            "language": language,
            "workers": workers,
            "max_llm_calls": max_llm_calls,
            "start_from": start_from,
            "output_root": output_root,
            "verbose": verbose,
            "debug": debug,
        },
    )

    model_names = parse_space_separated(cfg.models, ["qwen-plus"])
    language_value = cfg.language or ""

    languages = [language_value] if language_value else ["zh", "en"]
    for model in model_names:
        load_model_config(model)
        for language in languages:
            run_language(
                model, language, cfg, convert_report, eval_converted, travel_run
            )


if __name__ == "__main__":
    fire.Fire(run)
