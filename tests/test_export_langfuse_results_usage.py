from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts import export_langfuse_results_usage as export_usage


def _write_session(
    root: Path,
    *,
    experiment_name: str,
    timestamp: str,
    system: str,
    runs: int = 1,
    executor: str = "qwen3.5-9b",
) -> Path:
    session_root = root / experiment_name / timestamp
    (session_root / "aggregated_results").mkdir(parents=True)
    metadata = {
        "experiment": {"name": experiment_name},
        "timestamp": timestamp,
        "parameters": {
            "name": experiment_name,
            "domains": ["shopping"],
            "models": {"executor": executor, "overseer": "deepseek-v4-flash"},
            "system": {"name": system},
            "runtime": {"runs": runs},
            "shopping": {"split": "all"},
        },
    }
    (session_root / "experiment_session.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    for run_id in range(runs):
        aggregate = {
            "model_name": executor,
            "run_id": run_id,
            "domains": {
                "shopping": {
                    "total_cases": 120,
                    "successful_cases": 40 + run_id,
                    "failed_cases": 80 - run_id,
                    "total_matched_products": 390 + run_id,
                    "total_expected_products": 546,
                    "total_extra_products": 100,
                    "successful_rate": (40 + run_id) / 120,
                    "match_rate": 0.70 + run_id / 100,
                    "weighted_average_case_score": (40 + run_id) / 120,
                    "incomplete_cases": 1,
                    "incomplete_rate": 1 / 120,
                    "valid": True,
                    "levels_completed": [1, 2, 3],
                }
            },
            "overall": {
                "num_domains": 1,
                "shopping_match_rate": 0.70 + run_id / 100,
                "shopping_weighted_average_case_score": (40 + run_id) / 120,
            },
        }
        (session_root / "aggregated_results" / f"{executor}_run_{run_id}_aggregated.json").write_text(
            json.dumps(aggregate),
            encoding="utf-8",
        )
    return session_root


def test_discover_result_sessions_excludes_shopping_b_and_system_b(tmp_path):
    outputs_root = tmp_path / "outputs"
    included = _write_session(
        outputs_root,
        experiment_name="shopping-a",
        timestamp="2026-04-29_21-02-41",
        system="A",
    )
    _write_session(
        outputs_root,
        experiment_name="shopping-b",
        timestamp="2026-05-01_13-56-53",
        system="B",
    )
    _write_session(
        outputs_root,
        experiment_name="system-b",
        timestamp="2026-05-01_14-00-00",
        system="B",
    )
    _write_session(
        outputs_root,
        experiment_name="threshold-tuning-run",
        timestamp="2026-04-28_00-00-00",
        system="C2",
    )

    sessions = export_usage.discover_result_sessions(outputs_root)

    assert sessions == [included]


def test_build_report_rows_joins_langfuse_usage_and_aggregate_metrics(tmp_path):
    session_root = _write_session(
        tmp_path / "outputs",
        experiment_name="shopping-c2",
        timestamp="2026-04-30_09-18-02",
        system="C2",
        runs=2,
    )

    def fake_usage_fetcher(session_id: str):
        assert session_id == "2026-04-30_09-18-02"
        return (
            pd.DataFrame(
                [
                    {
                        "model": "qwen3.5-9b",
                        "input": 10,
                        "input_cached_tokens": 2,
                        "output": 3,
                        "total": 13,
                    },
                    {
                        "model": "deepseek-v4-flash",
                        "input": 20,
                        "input_cached_tokens": 5,
                        "output": 7,
                        "total": 27,
                    },
                ]
            ),
            8,
        )

    rows = export_usage.build_report_rows(
        [session_root],
        usage_fetcher=fake_usage_fetcher,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["experiment_name"] == "shopping-c2"
    assert row["system"] == "C2"
    assert row["configured_runs"] == 2
    assert row["run_count"] == 2
    assert row["langfuse_session_id"] == "2026-04-30_09-18-02"
    assert row["langfuse_observation_count"] == 8
    assert row["langfuse_models"] == "qwen3.5-9b,deepseek-v4-flash"
    assert row["langfuse_input_tokens"] == 30
    assert row["langfuse_input_cached_tokens"] == 7
    assert row["langfuse_output_tokens"] == 10
    assert row["langfuse_total_tokens"] == 40
    assert row["aggregate_domains_shopping_total_cases_sum"] == 240
    assert row["aggregate_domains_shopping_successful_cases_sum"] == 81
    assert row["aggregate_domains_shopping_match_rate_mean"] == 0.705


def test_write_report_creates_csv(tmp_path):
    output_path = tmp_path / "usage.csv"

    export_usage.write_report(
        [{"experiment_name": "shopping-a", "langfuse_total_tokens": 12}],
        output_path,
    )

    frame = pd.read_csv(output_path)
    assert frame.to_dict("records") == [
        {"experiment_name": "shopping-a", "langfuse_total_tokens": 12}
    ]
