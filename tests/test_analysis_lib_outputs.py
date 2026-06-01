from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

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


def test_cost_dollars_uses_flat_pricing_from_models_config(tmp_path):
    models_config = tmp_path / "models.yaml"
    models_config.write_text(
        """
models:
  qwen3.5-9b:
    model_name: Qwen/Qwen3.5-9B
    pricing:
      calculator: flat_input_output_v1
      prices:
        input_per_million_usd: 0.10
        output_per_million_usd: 0.20
""".lstrip(),
        encoding="utf-8",
    )
    lf = pd.DataFrame(
        [
            {
                "system": "C2-lora",
                "role": "executor",
                "model": "Qwen/Qwen3.5-9B",
                "input_uncached": 1_000_000,
                "input_cached": 500_000,
                "output": 2_000_000,
                "runs": 2,
            }
        ]
    )

    costs = al.cost_dollars(lf, models_config_path=models_config)

    assert costs.loc[0, "total_usd_as_billed"] == pytest.approx(0.55)
    assert costs.loc[0, "total_usd_uncached"] == pytest.approx(0.55)
    assert costs.loc[0, "per_run_usd_as_billed"] == pytest.approx(0.275)
    assert costs.loc[0, "per_run_usd_uncached"] == pytest.approx(0.275)


def test_cost_dollars_uses_cached_pricing_from_models_config(tmp_path):
    models_config = tmp_path / "models.yaml"
    models_config.write_text(
        """
models:
  deepseek-v4-flash-nt:
    model_name: deepseek-v4-flash
    pricing:
      calculator: cached_input_output_v1
      prices:
        input_cache_hit_per_million_usd: 0.01
        input_cache_miss_per_million_usd: 0.20
        output_per_million_usd: 0.30
""".lstrip(),
        encoding="utf-8",
    )
    lf = pd.DataFrame(
        [
            {
                "system": "C2-deepseek-nt",
                "role": "combined",
                "model": "deepseek-v4-flash",
                "input_uncached": 1_000_000,
                "input_cached": 2_000_000,
                "output": 3_000_000,
                "runs": 1,
            }
        ]
    )

    costs = al.cost_dollars(lf, models_config_path=models_config)

    assert costs.loc[0, "total_usd_as_billed"] == pytest.approx(1.12)
    assert costs.loc[0, "total_usd_uncached"] == pytest.approx(1.50)


def test_load_trace_tokens_filters_sessions_and_sums_runs(tmp_path):
    summary_path = tmp_path / "token_summary.csv"
    pd.DataFrame(
        [
            {
                "system": "C2-deepseek",
                "experiment_name": "shopping-c2-deepseek",
                "session_id": "session-a",
                "split": "hold_out",
                "role": "executor",
                "model": "deepseek-v4-flash",
                "input_uncached_tokens": 10,
                "input_cached_tokens": 90,
                "output_total_tokens": 5,
                "total_tokens": 105,
            },
            {
                "system": "C2-deepseek",
                "experiment_name": "shopping-c2-deepseek",
                "session_id": "session-b",
                "split": "non_hold_out",
                "role": "executor",
                "model": "deepseek-v4-flash",
                "input_uncached_tokens": 20,
                "input_cached_tokens": 80,
                "output_total_tokens": 10,
                "total_tokens": 110,
            },
            {
                "system": "C2-deepseek",
                "experiment_name": "shopping-c2-deepseek",
                "session_id": "ignored-session",
                "split": "hold_out",
                "role": "executor",
                "model": "deepseek-v4-flash",
                "input_uncached_tokens": 999,
                "input_cached_tokens": 999,
                "output_total_tokens": 999,
                "total_tokens": 999,
            },
            {
                "system": "C2-deepseek",
                "experiment_name": "shopping-c2-deepseek",
                "session_id": "session-a",
                "split": "hold_out",
                "role": "overseer",
                "model": "overseer",
                "input_uncached_tokens": 7,
                "input_cached_tokens": 0,
                "output_total_tokens": 3,
                "total_tokens": 10,
            },
        ]
    ).to_csv(summary_path, index=False)

    lf = al.load_trace_tokens(
        summary_path,
        session_runs={"session-a": 2, "session-b": 3},
        session_ids={"session-a", "session-b"},
    )

    executor = lf[(lf["role"] == "executor")].iloc[0]
    assert executor["runs"] == 5
    assert executor["input_uncached"] == 30
    assert executor["input_cached"] == 170
    assert executor["output"] == 15
    assert executor["total"] == 215
    assert executor["per_run_total"] == pytest.approx(43)

    overseer = lf[(lf["role"] == "overseer")].iloc[0]
    assert overseer["runs"] == 5
    assert overseer["session_key"] == "session-a"
    assert overseer["total"] == 10
