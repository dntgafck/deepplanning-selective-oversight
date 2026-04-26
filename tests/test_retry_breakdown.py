from __future__ import annotations

import json

from experiment import build_system_config
from scripts.summarize_retry_breakdown import (
    load_task_result_records,
    summarize_retry_breakdown,
)


def test_retry_breakdown_counts_first_attempt_success_correctly(tmp_path):
    path = tmp_path / "task_results.jsonl"
    path.write_text(
        json.dumps(
            {
                "domain": "shopping",
                "success": True,
                "final_verification_retry_count": 0,
                "final_verification_result": "approved",
                "total_cost_usd": 0.10,
                "overseer_calls": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_retry_breakdown(load_task_result_records([path]))

    assert summary["total_records"] == 1
    assert summary["first_attempt_success_count"] == 1
    assert summary["first_attempt_success_rate"] == 1.0
    assert summary["recovered_after_retry_count"] == 0


def test_retry_breakdown_counts_recovered_success_correctly(tmp_path):
    path = tmp_path / "task_results.jsonl"
    path.write_text(
        json.dumps(
            {
                "domain": "shopping",
                "success": True,
                "final_verification_retry_count": 2,
                "final_verification_result": "approved",
                "total_cost_usd": 0.20,
                "overseer_calls": 3,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_retry_breakdown(load_task_result_records([path]))

    assert summary["recovered_after_retry_count"] == 1
    assert summary["recovered_after_retry_rate"] == 1.0
    assert summary["mean_retries_among_successful_repaired_cases"] == 2.0


def test_retry_breakdown_counts_retry_cap_exhaustion_correctly(tmp_path):
    path = tmp_path / "task_results.jsonl"
    rows = [
        {
            "domain": "shopping",
            "success": False,
            "final_verification_retry_count": 2,
            "final_verification_result": "retry_cap_exhausted",
            "total_cost_usd": 0.30,
            "overseer_calls": 4,
        },
        {
            "domain": "travel",
            "success": True,
            "final_verification_retry_count": 0,
            "final_verification_result": "approved",
            "total_cost_usd": 0.40,
            "overseer_calls": 0,
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    summary = summarize_retry_breakdown(load_task_result_records([path]))

    assert summary["total_records"] == 1
    assert summary["retry_cap_exhausted_count"] == 1
    assert summary["retry_cap_exhausted_rate"] == 1.0
    assert summary["total_cost_usd"] == 0.30
    assert summary["overseer_calls"] == 4


def test_c2_noretry_config_sets_retry_cap_to_zero():
    config = build_system_config("C2-noretry", executor_model="qwen3.5-9b")

    assert config.name == "C2-noretry"
    assert config.final_repair_retry_cap == 0
    assert config.overseer_prompt_version == "c2-lite-v1.4-frozen"
    assert config.oversight_domains == ("shopping",)
