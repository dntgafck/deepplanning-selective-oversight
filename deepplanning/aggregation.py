from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import OUTPUT_ROOT, ensure_directory


def _parse_run_id(path: Path) -> int | None:
    if not path.is_dir() or not path.name.startswith("run_"):
        return None
    suffix = path.name.removeprefix("run_")
    return int(suffix) if suffix.isdigit() else None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_shopping_run_stats(
    model: str, shopping_output_root: Path
) -> dict[int, dict[str, Any]]:
    legacy_stats_path = (
        shopping_output_root / "result_report" / f"{model}_statistics.json"
    )
    if legacy_stats_path.exists():
        return {0: _load_json(legacy_stats_path)}

    stats_by_run: dict[int, dict[str, Any]] = {}
    shopping_model_root = shopping_output_root / model
    for run_dir in sorted(shopping_model_root.glob("run_*")):
        run_id = _parse_run_id(run_dir)
        if run_id is None:
            continue
        stats_path = run_dir / "result_report" / f"{model}_statistics.json"
        if stats_path.exists():
            stats_by_run[run_id] = _load_json(stats_path)
    return stats_by_run


def _collect_travel_run_summaries(
    model: str, travel_output_root: Path
) -> dict[int, dict[str, Any]]:
    summaries_by_run: dict[int, dict[str, Any]] = {}

    for language in ["zh", "en"]:
        legacy_summary_path = (
            travel_output_root
            / f"{model}_{language}"
            / "evaluation"
            / "evaluation_summary.json"
        )
        if legacy_summary_path.exists():
            summaries_by_run.setdefault(0, {})[language] = _load_json(
                legacy_summary_path
            )

    for language_root in sorted(travel_output_root.glob(f"{model}_*")):
        if not language_root.is_dir():
            continue
        language = language_root.name.removeprefix(f"{model}_")
        if not language:
            continue
        for run_dir in sorted(language_root.glob("run_*")):
            run_id = _parse_run_id(run_dir)
            if run_id is None:
                continue
            summary_path = run_dir / "evaluation" / "evaluation_summary.json"
            if summary_path.exists():
                summaries_by_run.setdefault(run_id, {})[language] = _load_json(
                    summary_path
                )

    return summaries_by_run


def _collect_travel_run_statuses(
    model: str, travel_output_root: Path
) -> dict[int, dict[str, Any]]:
    statuses_by_run: dict[int, dict[str, Any]] = {}

    for language_root in sorted(travel_output_root.glob(f"{model}_*")):
        if not language_root.is_dir():
            continue
        language = language_root.name.removeprefix(f"{model}_")
        if not language:
            continue
        for run_dir in sorted(language_root.glob("run_*")):
            run_id = _parse_run_id(run_dir)
            if run_id is None:
                continue
            status_path = run_dir / "travel_run_status.json"
            if not status_path.exists():
                continue
            statuses_by_run.setdefault(run_id, {})[language] = _load_json(status_path)

    return statuses_by_run


def aggregate_results(model: str, benchmark_output_root: Path | None = None) -> None:
    root = ensure_directory(benchmark_output_root or OUTPUT_ROOT)
    aggregated_output_root = ensure_directory(root / "aggregated_results")
    shopping_stats_by_run = _collect_shopping_run_stats(model, root / "shopping")
    travel_summaries_by_run = _collect_travel_run_summaries(model, root / "travel")
    travel_statuses_by_run = _collect_travel_run_statuses(model, root / "travel")
    run_ids = sorted(
        set(shopping_stats_by_run)
        | set(travel_summaries_by_run)
        | set(travel_statuses_by_run)
    )

    for run_id in run_ids:
        shopping_stats = shopping_stats_by_run.get(run_id)
        travel_languages = travel_summaries_by_run.get(run_id, {})
        travel_statuses = travel_statuses_by_run.get(run_id, {})
        aggregated: dict[str, Any] = {
            "model_name": model,
            "run_id": run_id,
            "aggregation_time": datetime.now().isoformat(),
            "domains": {},
            "overall": {"domains_completed": [], "num_domains": 0},
        }

        if shopping_stats is not None:
            total = shopping_stats["total"]
            aggregated["domains"]["shopping"] = total
            aggregated["overall"]["domains_completed"].append("shopping")
            aggregated["overall"]["shopping_match_rate"] = total["match_rate"]
            aggregated["overall"]["shopping_weighted_average_case_score"] = total[
                "weighted_average_case_score"
            ]

        if travel_languages:
            metrics = [payload["metrics"] for payload in travel_languages.values()]
            total_samples = sum(
                payload.get("total_test_samples", 0)
                for payload in travel_languages.values()
            )
            successful_cases = sum(
                payload.get("evaluation_success_count", 0)
                for payload in travel_languages.values()
            )
            composite_score = sum(
                item.get("composite_score", 0.0) for item in metrics
            ) / len(metrics)
            case_acc = sum(item.get("case_acc", 0.0) for item in metrics) / len(metrics)
            commonsense_score = sum(
                item.get("commonsense_score", 0.0) for item in metrics
            ) / len(metrics)
            personalized_score = sum(
                item.get("personalized_score", 0.0) for item in metrics
            ) / len(metrics)
            aggregated["domains"]["travel"] = {
                "total_cases": total_samples,
                "successful_cases": successful_cases,
                "successful_rate": (
                    successful_cases / total_samples if total_samples else 0.0
                ),
                "composite_score": composite_score,
                "case_acc": case_acc,
                "commonsense_score": commonsense_score,
                "personalized_score": personalized_score,
                "languages_completed": sorted(travel_languages),
                "language_details": {
                    language: payload["metrics"]
                    for language, payload in sorted(travel_languages.items())
                },
            }
            aggregated["overall"]["domains_completed"].append("travel")
            aggregated["overall"]["travel_composite_score"] = composite_score
            aggregated["overall"]["travel_case_acc"] = case_acc
            aggregated["overall"]["travel_commonsense_score"] = commonsense_score
            aggregated["overall"]["travel_personalized_score"] = personalized_score

        if travel_statuses:
            aggregated["domains"]["travel_artifacts"] = {
                "languages": {
                    language: {
                        "inference_complete": payload.get("inference_complete"),
                        "conversion_complete": payload.get("conversion_complete"),
                        "full_eval_complete": payload.get("full_eval_complete"),
                        "fallback_eval_only": payload.get("fallback_eval_only"),
                        "official_evaluation_present": payload.get(
                            "official_evaluation_present"
                        ),
                        "official_evaluation_summary_path": payload.get(
                            "official_evaluation_summary_path"
                        ),
                        "generated_data_only_summary_path": payload.get(
                            "generated_data_only_summary_path"
                        ),
                    }
                    for language, payload in sorted(travel_statuses.items())
                }
            }
            aggregated["overall"]["travel_official_metrics_available"] = bool(
                travel_languages
            )
            aggregated["overall"]["travel_generated_data_only_available"] = any(
                payload.get("generated_data_only_summary_path")
                for payload in travel_statuses.values()
            )

        if not aggregated["domains"]:
            continue

        completed = aggregated["overall"]["domains_completed"]
        aggregated["overall"]["num_domains"] = len(completed)
        if (
            "shopping_match_rate" in aggregated["overall"]
            and "travel_composite_score" in aggregated["overall"]
        ):
            aggregated["overall"]["avg_acc"] = (
                aggregated["overall"]["shopping_weighted_average_case_score"]
                + aggregated["overall"]["travel_case_acc"]
            ) / 2

        output_path = aggregated_output_root / f"{model}_run_{run_id}_aggregated.json"
        output_path.write_text(
            json.dumps(aggregated, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"✅ Aggregated results saved to: {output_path}")
