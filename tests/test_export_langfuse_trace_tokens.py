from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts import export_langfuse_trace_tokens as trace_tokens


def _write_session(root: Path) -> Path:
    session_root = root / "shopping-c2-lora" / "2026-05-15_15-13-58"
    log_dir = (
        session_root
        / "shopping"
        / "qwen3.5-9b"
        / "run_0"
        / "logs"
        / "database_qwen3.5-9b_level1_202605151513"
    )
    (session_root / "aggregated_results").mkdir(parents=True)
    log_dir.mkdir(parents=True)
    metadata = {
        "experiment": {"name": "shopping-c2-lora"},
        "timestamp": "2026-05-15_15-13-58",
        "parameters": {
            "name": "shopping-c2-lora",
            "domains": ["shopping"],
            "models": {
                "executor": "qwen3.5-9b",
                "overseer": "qwen2.5-7b-overseer-lora",
            },
            "system": {"name": "C2"},
            "runtime": {"runs": 1},
            "shopping": {"split": "hold_out"},
        },
    }
    (session_root / "experiment_session.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    (
        session_root / "aggregated_results" / "qwen3.5-9b_run_0_aggregated.json"
    ).write_text(json.dumps({"run_id": 0}), encoding="utf-8")
    events = [
        {
            "timestamp": "2026-05-15T15:00:00+00:00",
            "event_type": "executor_turn",
            "domain": "shopping",
            "task_id": "3",
            "run_id": 0,
            "phase": "initial",
            "turn_index": 1,
            "prompt_tokens": 100,
            "completion_tokens": 11,
            "model_alias": "qwen3.5-9b",
            "raw_response": {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 11,
                    "total_tokens": 111,
                }
            },
        },
        {
            "timestamp": "2026-05-15T15:00:01+00:00",
            "event_type": "oversight_step",
            "domain": "shopping",
            "task_id": "8",
            "run_id": 0,
            "phase": "initial",
            "step_index": 2,
            "tool_index": 0,
            "trigger_type": "mutating_action",
            "overseer_invoked": True,
            "overseer_input_tokens": 30,
            "overseer_output_tokens": 7,
            "raw_overseer_text": '{"action":"approve"}',
        },
        {
            "timestamp": "2026-05-15T15:00:01+00:00",
            "event_type": "oversight_step",
            "domain": "shopping",
            "task_id": "8",
            "run_id": 0,
            "phase": "initial",
            "step_index": 2,
            "tool_index": 0,
            "trigger_type": "mutating_action",
            "overseer_invoked": True,
            "overseer_input_tokens": 30,
            "overseer_output_tokens": 7,
            "raw_overseer_text": '{"action":"approve"}',
        },
    ]
    (log_dir / "agent_events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    return session_root


def _observation(
    observation_id: str,
    *,
    actor: str,
    task_id: str,
    level: int,
    step_index: int,
    trace_id: str = "collapsed-trace",
    usage: dict[str, object] | None = None,
) -> dict[str, object]:
    name = (
        f"executor.initial.step_{step_index:03d}"
        if actor == "executor"
        else f"overseer.pre_tool.step_{step_index:03d}.mutating_action"
    )
    metadata = {
        "actor": actor,
        "domain": "shopping",
        "task_id": task_id,
        "level": level,
        "run_id": 0,
        "system": "C2",
        "phase": "initial",
        "step_index": step_index,
    }
    if actor == "overseer":
        metadata.update(
            {
                "tool_index": 0,
                "hook": "pre_tool",
                "trigger_type": "mutating_action",
            }
        )
    return {
        "id": observation_id,
        "traceId": trace_id,
        "sessionId": "2026-05-15_15-13-58",
        "name": name,
        "model": "qwen3.5-9b" if actor == "executor" else "overseer",
        "metadata": metadata,
        "usageDetails": usage or {},
    }


def test_token_export_unravels_collapsed_trace_ids_with_metadata_join(tmp_path):
    session_root = _write_session(tmp_path / "outputs")
    descriptor = trace_tokens.session_descriptor(session_root)
    observations = [
        trace_tokens.normalize_observation(
            _observation(
                "exec-1",
                actor="executor",
                task_id="3",
                level=1,
                step_index=1,
                usage={
                    "input": 60,
                    "input_cached_tokens": 40,
                    "output": 9,
                    "output_reasoning_tokens": 2,
                },
            ),
            descriptor=descriptor,
        ),
        trace_tokens.normalize_observation(
            _observation(
                "overseer-1",
                actor="overseer",
                task_id="8",
                level=1,
                step_index=2,
                usage={
                    "input": 20,
                    "input_cached_tokens": 10,
                    "output": 4,
                    "output_reasoning_tokens": 1,
                },
            ),
            descriptor=descriptor,
        ),
    ]
    local_records, local_diagnostics = trace_tokens.load_local_call_records([session_root])

    rows, diagnostics = trace_tokens.build_token_rows(
        observations,
        local_records,
        split_lookup={"level_1:3": "hold_out", "level_1:8": "hold_out"},
    )

    assert local_diagnostics["local_duplicate_event_count"] == 1
    assert diagnostics["counts"]["matched_local_records"] == 2
    assert diagnostics["counts"]["collapsed_trace_groups"] == 1
    assert len(rows) == 2
    assert {row["case_name"] for row in rows} == {"case_3", "case_8"}
    exec_row = next(row for row in rows if row["role"] == "executor")
    assert exec_row["input_total_tokens"] == 100
    assert exec_row["input_uncached_tokens"] == 60
    assert exec_row["input_cached_tokens"] == 40
    assert exec_row["output_total_tokens"] == 11
    assert exec_row["split"] == "hold_out"


def test_local_fallback_preserves_deepseek_cache_counts():
    local = {
        "source": "local_agent_log",
        "source_session_root": "session",
        "experiment_name": "shopping-c2-deepseek-lora",
        "system": "C2-deepseek-lora",
        "executor_model": "deepseek-v4-flash",
        "overseer_model": "qwen2.5-7b-overseer-lora",
        "session_id": "2026-05-15_14-56-51",
        "trace_id": "trace",
        "actor": "executor",
        "model": "deepseek-v4-flash",
        "domain": "shopping",
        "level": "2",
        "task_id": "13",
        "case_name": "case_13",
        "run_id": 2,
        "phase": "initial",
        "step_index": 1,
        "tool_index": None,
        "hook": "",
        "trigger_type": "",
        "source_event_path": "agent_events.jsonl",
        "source_event_line": 1,
        "call_key": trace_tokens.CallKey(
            "executor",
            "2026-05-15_14-56-51",
            "13",
            2,
            "initial",
            1,
            None,
            "",
            "",
        ),
        "event": {
            "timestamp": "2026-05-15T14:00:00+00:00",
            "prompt_tokens": 3701,
            "completion_tokens": 173,
            "raw_response": {
                "usage": {
                    "prompt_tokens": 3701,
                    "completion_tokens": 173,
                    "total_tokens": 3874,
                    "prompt_cache_hit_tokens": 3456,
                    "prompt_cache_miss_tokens": 245,
                    "completion_tokens_details": {"reasoning_tokens": 30},
                }
            },
        },
    }

    rows, diagnostics = trace_tokens.build_token_rows(
        [],
        [local],
        split_lookup={"level_2:13": "hold_out"},
    )

    assert diagnostics["counts"]["local_fallback_rows"] == 1
    row = rows[0]
    assert row["source"] == "local_agent_log"
    assert row["input_total_tokens"] == 3701
    assert row["input_cached_tokens"] == 3456
    assert row["input_uncached_tokens"] == 245
    assert row["output_reasoning_tokens"] == 30
    assert row["total_tokens"] == 3874


def test_write_outputs_creates_held_out_summary_files(tmp_path):
    output_dir = tmp_path / "trace_tokens"
    rows = [
        {
            "source": "langfuse",
            "match_source": "langfuse_observation",
            "system": "C2-lora",
            "experiment_name": "shopping-c2-lora",
            "session_id": "2026-05-15_15-13-58",
            "split": "hold_out",
            "role": "executor",
            "model": "qwen3.5-9b",
            "run_id": 0,
            "level": 1,
            "case_name": "case_3",
            "input_total_tokens": 10,
            "input_uncached_tokens": 10,
            "input_cached_tokens": 0,
            "output_total_tokens": 5,
            "output_text_tokens": 5,
            "output_reasoning_tokens": 0,
            "total_tokens": 15,
            "usage_reported_total_tokens": 15,
        }
    ]

    trace_tokens.write_outputs(
        output_dir=output_dir,
        observation_records=[],
        token_rows=rows,
        diagnostics={"counts": {"token_rows": 1}},
        manifest={"counts": {"token_rows": 1}},
    )

    summary = pd.read_csv(output_dir / "held_out_token_summary.csv")
    assert summary.to_dict("records") == [
        {
            "system": "C2-lora",
            "experiment_name": "shopping-c2-lora",
            "session_id": "2026-05-15_15-13-58",
            "split": "hold_out",
            "role": "executor",
            "model": "qwen3.5-9b",
            "observation_count": 1,
            "case_count": 1,
            "input_total_tokens": 10,
            "input_uncached_tokens": 10,
            "input_cached_tokens": 0,
            "output_total_tokens": 5,
            "output_text_tokens": 5,
            "output_reasoning_tokens": 0,
            "total_tokens": 15,
        }
    ]
