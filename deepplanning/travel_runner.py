from __future__ import annotations

import json
import sys
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
    parse_space_separated,
    resolve_repo_path,
    write_json_file,
)

TRAVEL_ROOT = BENCHMARK_ROOT / "travelplanning"
TRAVEL_DATA_ROOT = DATA_ROOT / "travel" / "database"
TRAVEL_OUTPUT_ROOT = OUTPUT_ROOT / "travel"
GENERATED_DATA_ONLY_DIRNAME = "generated_data_only_evaluation"
GENERATED_DATA_ONLY_FILENAME = "generated_data_only_summary.json"
TRAVEL_RUN_STATUS_FILENAME = "travel_run_status.json"

ensure_repo_imports()

from agent import travel as travel_agent_runner


def import_modules() -> tuple[object, object]:
    clear_imported_modules(["evaluation", "prompts", "call_llm"])
    sys.path.insert(0, str(TRAVEL_ROOT))

    import evaluation.convert_report as convert_report
    import evaluation.eval_converted as eval_converted

    convert_report.load_model_config = load_model_config
    return convert_report, eval_converted


def prepare_test_data(
    language: str, output_dir: Path, sample_ids: list[str] | None
) -> Path:
    source_test_data_path = (
        TRAVEL_ROOT / "data" / f"travelplanning_query_{language}.json"
    )
    if sample_ids is None:
        return source_test_data_path

    all_samples = load_json_file(source_test_data_path)
    selected_samples = filter_samples_by_ids(all_samples, sample_ids)
    if not selected_samples:
        raise ValueError(f"No travel samples matched ids: {', '.join(sample_ids)}")

    filtered_test_data_path = (
        output_dir / f"travelplanning_query_{language}_subset.json"
    )
    write_json_file(filtered_test_data_path, selected_samples)
    return filtered_test_data_path


def _load_test_samples(test_data_path: Path) -> list[dict[str, Any]]:
    payload = load_json_file(test_data_path)
    if isinstance(payload, dict):
        return [payload]
    return list(payload)


def _task_id(sample_id: str) -> str:
    return f"id_{sample_id}" if sample_id.isdigit() else sample_id


def _load_task_results_by_task_id(results_path: Path) -> dict[str, dict[str, Any]]:
    if not results_path.exists():
        return {}

    results: dict[str, dict[str, Any]] = {}
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        task_id = str(record.get("task_id", "")).strip()
        if task_id:
            results[task_id] = record
    return results


def _build_sample_statuses(
    *,
    test_data_path: Path,
    run_output_dir: Path,
    conversion_results: dict[str, Any] | None,
    evaluation_results: dict[str, Any] | None,
    fallback_active: bool,
) -> list[dict[str, Any]]:
    task_results = _load_task_results_by_task_id(run_output_dir / "task_results.jsonl")
    conversion_by_id = {
        str(item.get("sample_id")): item
        for item in (conversion_results or {}).get("results", [])
        if item.get("sample_id") is not None
    }
    evaluation_by_id = {
        str(item.get("sample_id")): item
        for item in (evaluation_results or {}).get("results", [])
        if item.get("sample_id") is not None
    }

    sample_statuses: list[dict[str, Any]] = []
    for sample in _load_test_samples(test_data_path):
        sample_id = str(sample.get("id"))
        task_key = _task_id(sample_id)
        trajectory_path = run_output_dir / "trajectories" / f"{task_key}.json"
        report_path = run_output_dir / "reports" / f"{task_key}.txt"
        converted_path = (
            run_output_dir / "converted_plans" / f"{task_key}_converted.json"
        )
        evaluation_path = run_output_dir / "evaluation" / f"{task_key}_score.json"

        task_result = task_results.get(task_key)
        conversion_entry = conversion_by_id.get(sample_id)
        evaluation_entry = evaluation_by_id.get(sample_id)

        inference_complete = trajectory_path.exists() or task_result is not None
        inference_success = (
            bool(task_result.get("success"))
            if task_result is not None and "success" in task_result
            else trajectory_path.exists()
        )
        observation_valid = (
            bool(task_result.get("observation_valid", True))
            if task_result is not None
            else trajectory_path.exists()
        )
        report_generated = report_path.exists()
        final_output_present = (
            bool(task_result.get("final_output_present"))
            if task_result is not None
            else report_generated
        )
        failure_subtype = (
            str(task_result.get("failure_subtype", "none"))
            if task_result is not None
            else "none"
        )

        if converted_path.exists():
            conversion_status = "complete"
            conversion_error = None
        elif report_generated:
            if conversion_entry is not None and not conversion_entry.get(
                "success", False
            ):
                conversion_status = "failed"
                conversion_error = str(conversion_entry.get("error", ""))
            else:
                conversion_status = "missing"
                conversion_error = None
        else:
            conversion_status = (
                "missing_generated_report" if inference_complete else "not_started"
            )
            conversion_error = None

        official_evaluation_complete = evaluation_path.exists()
        evaluation_error = None
        if evaluation_entry is not None and not evaluation_entry.get("success", False):
            evaluation_error = str(evaluation_entry.get("error", ""))

        if official_evaluation_complete:
            evaluation_status = "complete"
        elif fallback_active and inference_complete:
            evaluation_status = "generated_data_only"
        elif conversion_status == "complete":
            evaluation_status = "failed_or_missing"
        else:
            evaluation_status = "skipped"

        metrics_excerpt = None
        if task_result is not None:
            metrics_excerpt = {
                "success": task_result.get("success"),
                "failure_subtype": task_result.get("failure_subtype"),
                "observation_valid": task_result.get("observation_valid"),
                "executor_calls": task_result.get("executor_calls"),
                "tool_call_count": task_result.get("tool_call_count"),
                "final_stop_reason": task_result.get("final_stop_reason"),
                "final_output_present": task_result.get("final_output_present"),
            }

        sample_statuses.append(
            {
                "sample_id": sample_id,
                "task_id": task_key,
                "inference_complete": inference_complete,
                "inference_success": inference_success,
                "failure_subtype": failure_subtype,
                "observation_valid": observation_valid,
                "report_generated": report_generated,
                "final_output_present": final_output_present,
                "conversion_status": conversion_status,
                "conversion_error": conversion_error,
                "official_evaluation_complete": official_evaluation_complete,
                "evaluation_status": evaluation_status,
                "evaluation_error": evaluation_error,
                "fallback_eval_only": fallback_active
                and not official_evaluation_complete
                and inference_complete,
                "paths": {
                    "trajectory": (
                        str(trajectory_path) if trajectory_path.exists() else None
                    ),
                    "report": str(report_path) if report_path.exists() else None,
                    "converted_plan": (
                        str(converted_path) if converted_path.exists() else None
                    ),
                    "official_evaluation": (
                        str(evaluation_path) if evaluation_path.exists() else None
                    ),
                },
                "task_metrics": metrics_excerpt,
            }
        )

    return sample_statuses


def _write_generated_data_only_summary(
    *,
    run_output_dir: Path,
    model: str,
    language: str,
    run_id: int,
    reason: str,
    sample_statuses: list[dict[str, Any]],
) -> Path:
    summary_dir = ensure_directory(run_output_dir / GENERATED_DATA_ONLY_DIRNAME)
    summary_path = summary_dir / GENERATED_DATA_ONLY_FILENAME
    payload = {
        "mode": "generated_data_only",
        "label": "Unofficial generated-data-only summary. Do not treat this as official benchmark evaluation.",
        "model": model,
        "language": language,
        "run_id": run_id,
        "reason": reason,
        "total_samples": len(sample_statuses),
        "inference_complete_count": sum(
            1 for status in sample_statuses if status["inference_complete"]
        ),
        "report_generated_count": sum(
            1 for status in sample_statuses if status["report_generated"]
        ),
        "conversion_complete_count": sum(
            1 for status in sample_statuses if status["conversion_status"] == "complete"
        ),
        "sample_statuses": sample_statuses,
    }
    write_json_file(summary_path, payload)
    return summary_path


def _write_travel_run_status(
    *,
    run_output_dir: Path,
    model: str,
    language: str,
    run_id: int,
    start_from: str,
    evaluation_mode: str,
    conversion_results: dict[str, Any] | None,
    evaluation_results: dict[str, Any] | None,
    sample_statuses: list[dict[str, Any]],
    fallback_summary_path: Path | None,
) -> Path:
    official_summary_path = run_output_dir / "evaluation" / "evaluation_summary.json"
    payload = {
        "model": model,
        "language": language,
        "run_id": run_id,
        "start_from": start_from,
        "evaluation_mode": evaluation_mode,
        "inference_complete": all(
            status["inference_complete"] for status in sample_statuses
        ),
        "conversion_complete": all(
            status["conversion_status"] == "complete" for status in sample_statuses
        ),
        "full_eval_complete": all(
            status["official_evaluation_complete"] for status in sample_statuses
        ),
        "fallback_eval_only": fallback_summary_path is not None
        and not official_summary_path.exists(),
        "official_evaluation_present": official_summary_path.exists(),
        "official_evaluation_summary_path": (
            str(official_summary_path) if official_summary_path.exists() else None
        ),
        "generated_data_only_summary_path": (
            str(fallback_summary_path) if fallback_summary_path is not None else None
        ),
        "conversion_summary": conversion_results,
        "evaluation_summary": evaluation_results,
        "sample_statuses": sample_statuses,
    }
    status_path = run_output_dir / TRAVEL_RUN_STATUS_FILENAME
    write_json_file(status_path, payload)
    return status_path


def run_language(
    model: str,
    language: str,
    sample_ids: list[str] | None,
    system: str,
    runs: int,
    cfg: Any,
    convert_report: object,
    eval_converted: object,
    langfuse_session_id: str | None = None,
) -> None:
    database_dir = TRAVEL_DATA_ROOT / f"database_{language}"
    if not database_dir.exists():
        raise FileNotFoundError(
            f"Travel data missing at {database_dir}. Ensure DVC data bootstrap is present under data/deepplanning/."
        )

    output_base = ensure_directory(resolve_repo_path(cfg.output_root))
    output_dir = ensure_directory(output_base / f"{model}_{language}")
    run_output_dirs = {
        run_id: ensure_directory(output_dir / f"run_{run_id}") for run_id in range(runs)
    }
    for run_output_dir in run_output_dirs.values():
        ensure_directory(run_output_dir / "trajectories")
        ensure_directory(run_output_dir / "reports")
        ensure_directory(run_output_dir / "converted_plans")
        ensure_directory(run_output_dir / "evaluation")

    test_data_path = prepare_test_data(language, output_dir, sample_ids)
    tool_schema_path = TRAVEL_ROOT / "tools" / f"tool_schema_{language}.json"

    print("\n🚀 Travel benchmark")
    print(f"   Model: {model}")
    print(f"   Language: {language}")
    print(f"   Database: {database_dir}")
    print(f"   Output: {output_dir}")
    print(f"   Runs: {runs}")
    if sample_ids is not None:
        print(f"   Sample IDs: {', '.join(sample_ids)}")

    evaluation_mode = str(getattr(cfg, "evaluation_mode", "auto") or "auto")
    if evaluation_mode not in {"auto", "generated_data_only"}:
        raise ValueError(
            f"Unsupported travel evaluation_mode '{evaluation_mode}'. Expected 'auto' or 'generated_data_only'."
        )

    inference_results: dict[str, Any] | None = None
    if cfg.start_from == "inference":
        inference_results = travel_agent_runner.run_agent_inference(
            model=model,
            language=language,
            test_data_path=test_data_path,
            database_dir=database_dir,
            tool_schema_path=tool_schema_path,
            output_dir=output_dir,
            workers=int(cfg.workers),
            max_llm_calls=int(cfg.max_llm_calls),
            infra_retry_limit=int(getattr(cfg, "infra_retry_limit", 2)),
            runs=runs,
            system=system,
            output_dir_by_run=run_output_dirs,
            session_id=langfuse_session_id,
        )
        print(
            f"✅ Travel inference complete: {inference_results['success']}/{inference_results['total']} succeeded"
        )

    for run_id, run_output_dir in sorted(run_output_dirs.items()):
        conversion_results: dict[str, Any] | None = None
        evaluation_results: dict[str, Any] | None = None
        if (
            cfg.start_from in {"inference", "conversion"}
            and evaluation_mode != "generated_data_only"
        ):
            conversion_results = convert_report.convert_reports(
                result_dir=run_output_dir,
                language=language,
                workers=int(cfg.workers),
                skip_existing=True,
                verbose=bool(cfg.verbose),
            )
            print(
                f"✅ Travel conversion complete for run {run_id}: {conversion_results['converted']} converted, {conversion_results['failed']} failed, {conversion_results['skipped']} skipped"
            )

        if evaluation_mode != "generated_data_only":
            evaluation_results = eval_converted.evaluate_plans(
                result_dir=run_output_dir,
                test_data_path=test_data_path,
                database_dir=database_dir,
                verbose=bool(cfg.verbose),
            )
            metrics = evaluation_results.get("metrics", {})
            if metrics:
                print(
                    f"✅ Travel evaluation complete for run {run_id}: composite={metrics['composite_score']:.2%}, case_acc={metrics['case_acc']:.2%}"
                )
            else:
                print(
                    f"⚠️  Travel evaluation incomplete for run {run_id}: no official metrics produced"
                )

        sample_statuses = _build_sample_statuses(
            test_data_path=test_data_path,
            run_output_dir=run_output_dir,
            conversion_results=conversion_results,
            evaluation_results=evaluation_results,
            fallback_active=False,
        )

        official_summary_path = (
            run_output_dir / "evaluation" / "evaluation_summary.json"
        )
        full_eval_complete = official_summary_path.exists() and all(
            status["official_evaluation_complete"] for status in sample_statuses
        )

        fallback_summary_path: Path | None = None
        if evaluation_mode == "generated_data_only" or not full_eval_complete:
            fallback_reason = (
                "generated-data-only mode requested"
                if evaluation_mode == "generated_data_only"
                else "official full evaluation unavailable or incomplete"
            )
            sample_statuses = _build_sample_statuses(
                test_data_path=test_data_path,
                run_output_dir=run_output_dir,
                conversion_results=conversion_results,
                evaluation_results=evaluation_results,
                fallback_active=True,
            )
            fallback_summary_path = _write_generated_data_only_summary(
                run_output_dir=run_output_dir,
                model=model,
                language=language,
                run_id=run_id,
                reason=fallback_reason,
                sample_statuses=sample_statuses,
            )
            print(
                f"⚠️  Travel generated-data-only summary written for run {run_id}: {fallback_summary_path}"
            )

        status_path = _write_travel_run_status(
            run_output_dir=run_output_dir,
            model=model,
            language=language,
            run_id=run_id,
            start_from=str(cfg.start_from),
            evaluation_mode=evaluation_mode,
            conversion_results=conversion_results,
            evaluation_results=evaluation_results,
            sample_statuses=sample_statuses,
            fallback_summary_path=fallback_summary_path,
        )
        print(f"ℹ️  Travel run status written to: {status_path}")


def run(
    *,
    models: str | list[str] | None = None,
    language: str | list[str] | None = None,
    sample_ids: int | str | list[int] | list[str] | None = None,
    system: str = "A",
    workers: int = 50,
    max_llm_calls: int = 400,
    infra_retry_limit: int = 2,
    runs: int = 4,
    start_from: str = "inference",
    evaluation_mode: str = "auto",
    output_root: str | Path | None = None,
    verbose: bool = False,
    debug: bool = False,
    langfuse_session_id: str | None = None,
) -> None:
    load_dotenv()
    convert_report, eval_converted = import_modules()

    model_names = parse_space_separated(models, ["qwen3-14b"])
    language_value = language or ""
    selected_sample_ids = parse_id_list(sample_ids)

    languages = [str(language_value)] if str(language_value) else ["zh", "en"]
    cfg = type(
        "TravelRuntimeConfig",
        (),
        {
            "output_root": str(output_root or TRAVEL_OUTPUT_ROOT),
            "start_from": start_from,
            "workers": workers,
            "max_llm_calls": max_llm_calls,
            "infra_retry_limit": infra_retry_limit,
            "verbose": verbose,
            "debug": debug,
            "evaluation_mode": evaluation_mode,
        },
    )()

    for model in model_names:
        load_model_config(model)
        for selected_language in languages:
            run_language(
                model,
                selected_language,
                selected_sample_ids,
                str(system),
                int(runs),
                cfg,
                convert_report,
                eval_converted,
                langfuse_session_id=langfuse_session_id,
            )
