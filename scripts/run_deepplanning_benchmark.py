from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import fire
from deepplanning_common import (
    OUTPUT_ROOT,
    REPO_ROOT,
    compose_config,
    ensure_directory,
    parse_int_list,
    parse_space_separated,
)


def run_subprocess(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def aggregate_results(model: str) -> None:
    shopping_stats_path = (
        OUTPUT_ROOT / "shopping" / "result_report" / f"{model}_statistics.json"
    )
    travel_output_root = OUTPUT_ROOT / "travel"
    aggregated_output_root = ensure_directory(OUTPUT_ROOT / "aggregated_results")

    shopping_stats: dict[str, Any] | None = None
    if shopping_stats_path.exists():
        shopping_stats = json.loads(shopping_stats_path.read_text(encoding="utf-8"))

    travel_languages: dict[str, Any] = {}
    for language in ["zh", "en"]:
        summary_path = (
            travel_output_root
            / f"{model}_{language}"
            / "evaluation"
            / ("evaluation_summary.json")
        )
        if summary_path.exists():
            travel_languages[language] = json.loads(
                summary_path.read_text(encoding="utf-8")
            )

    if shopping_stats is None and not travel_languages:
        return

    aggregated: dict[str, Any] = {
        "model_name": model,
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

    output_path = aggregated_output_root / f"{model}_aggregated.json"
    output_path.write_text(
        json.dumps(aggregated, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"✅ Aggregated results saved to: {output_path}")


def main() -> None:
    run()


def run(
    domains: str | None = None,
    models: str | None = None,
    shopping_levels: str | None = None,
    workers: int | None = None,
    max_llm_calls: int | None = None,
    travel_language: str | None = None,
    travel_start_from: str | None = None,
) -> None:
    cfg = compose_config(
        "benchmark",
        {
            "domains": domains,
            "models": models,
            "shopping_levels": shopping_levels,
            "workers": workers,
            "max_llm_calls": max_llm_calls,
            "travel_language": travel_language,
            "travel_start_from": travel_start_from,
        },
    )

    domain_names = parse_space_separated(cfg.domains, ["travel", "shopping"])
    model_names = parse_space_separated(cfg.models, ["qwen-plus"])
    shopping_level_numbers = parse_int_list(cfg.shopping_levels, [1, 2, 3])

    if "shopping" in domain_names:
        run_subprocess(
            [
                sys.executable,
                "scripts/run_deepplanning_shopping.py",
                f"--models={' '.join(model_names)}",
                f"--levels={' '.join(str(level) for level in shopping_level_numbers)}",
                f"--workers={cfg.workers}",
                f"--max_llm_calls={cfg.max_llm_calls}",
            ]
        )

    if "travel" in domain_names:
        command = [
            sys.executable,
            "scripts/run_deepplanning_travel.py",
            f"--models={' '.join(model_names)}",
            f"--workers={cfg.workers}",
            f"--max_llm_calls={cfg.max_llm_calls}",
            f"--start_from={cfg.travel_start_from}",
        ]
        if cfg.travel_language:
            command.append(f"--language={cfg.travel_language}")
        run_subprocess(command)

    for model in model_names:
        aggregate_results(model)


if __name__ == "__main__":
    fire.Fire(run)
