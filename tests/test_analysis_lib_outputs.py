from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import analysis_lib as al  # noqa: E402


def _write_session(
    root: Path,
    *,
    experiment_name: str,
    timestamp: str,
    system: str,
    run_id: int = 0,
) -> Path:
    session_root = root / experiment_name / timestamp
    model = "qwen3.5-9b"
    result_root = (
        session_root
        / "shopping"
        / model
        / f"run_{run_id}"
        / "result_report"
        / f"database_{model}_level1_202605130001"
    )
    result_root.mkdir(parents=True)
    (session_root / "aggregated_results").mkdir()

    metadata = {
        "experiment": {"name": experiment_name},
        "timestamp": timestamp,
        "parameters": {
            "name": experiment_name,
            "models": {"executor": model, "overseer": "deepseek-v4-flash"},
            "system": {"name": system},
        },
    }
    (session_root / "experiment_session.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    (result_root / "summary_report.json").write_text(
        json.dumps(
            {
                "case_results": [
                    {
                        "case_name": "case_1",
                        "case_score": 1.0,
                        "score": 1.0,
                        "success": True,
                        "matched_count": 2,
                        "expected_count": 2,
                        "extra_products_count": 0,
                        "is_completed": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (
        session_root / "aggregated_results" / f"{model}_run_{run_id}_aggregated.json"
    ).write_text(
        json.dumps(
            {
                "run_id": run_id,
                "model_name": model,
                "domains": {
                    "shopping": {
                        "total_cases": 1,
                        "successful_cases": 1,
                        "successful_rate": 1.0,
                        "match_rate": 1.0,
                        "weighted_average_case_score": 1.0,
                        "incomplete_cases": 0,
                        "incomplete_rate": 0.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return session_root


def test_discover_output_sessions_prefers_system_dirs_and_ignores_prefixes(tmp_path):
    experiments_root = tmp_path / "experiments"
    included = _write_session(
        experiments_root,
        experiment_name="system-a",
        timestamp="2026-05-13_00-00-00",
        system="A",
    )
    _write_session(
        experiments_root,
        experiment_name="system-b-smoke",
        timestamp="2026-05-13_00-01-00",
        system="B",
    )
    _write_session(
        experiments_root,
        experiment_name="shopping-a",
        timestamp="2026-05-13_00-02-00",
        system="A",
    )

    sessions = al.discover_output_sessions(experiments_root)

    assert sessions == [included]


def test_discover_output_sessions_falls_back_to_shopping_dirs(tmp_path):
    experiments_root = tmp_path / "experiments"
    fallback = _write_session(
        experiments_root,
        experiment_name="shopping-a",
        timestamp="2026-05-13_00-02-00",
        system="A",
    )

    sessions = al.discover_output_sessions(experiments_root)

    assert sessions == [fallback]


def test_load_output_sessions_labels_variants_and_offsets_duplicate_runs(tmp_path):
    experiments_root = tmp_path / "experiments"
    first = _write_session(
        experiments_root,
        experiment_name="shopping-c2-deepseek-pro",
        timestamp="2026-05-13_00-00-00",
        system="C2",
    )
    second = _write_session(
        experiments_root,
        experiment_name="shopping-c2-deepseek-pro",
        timestamp="2026-05-13_00-01-00",
        system="C2",
    )

    per_case = al.load_per_case([first, second])
    aggregated = al.load_aggregated([first, second])

    assert per_case["system"].unique().tolist() == ["C2-deepseek-pro"]
    assert per_case["run"].tolist() == [0, 1]
    assert aggregated["run"].tolist() == [0, 1]
