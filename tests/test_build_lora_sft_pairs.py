from __future__ import annotations

import json

from scripts import build_lora_sft_pairs as builder


def _first_task(split: dict[str, object], split_name: str) -> tuple[str, str]:
    levels = split["splits"][split_name]
    level_key = sorted(levels)[0]
    return level_key.removeprefix("level_"), str(levels[level_key][0])


def _local_record(
    *,
    session_id: str = "2026-04-30_13-35-38",
    task_id: str = "1",
    level: str = "1",
    task_query: str | None = None,
) -> dict[str, object]:
    key = builder.JoinKey(
        session_id=session_id,
        task_id=task_id,
        run_id=0,
        phase="initial",
        step_index=1,
        tool_index=0,
        hook="pre_tool",
        trigger_type="mutating_action",
    )
    return {
        "source_system": "C2-nt",
        "session_id": session_id,
        "trace_id": "trace-local",
        "level": level,
        "task_query": task_query,
        "hook": "pre_tool",
        "source_event_path": "agent_events.jsonl",
        "join_key": key,
        "event": {
            "overseer_mode": "non-thinking",
            "raw_overseer_text": '{"action":"approve","decision_summary":"","violation_evidence":{"violated_contract_ids":[],"unmet_checklist_keys":[],"confidence":"low"},"guidance_lines":[],"corrected_observation":null}',
            "overseer_input_tokens": 10,
            "overseer_output_tokens": 4,
        },
    }


def _export_record(
    *,
    session_id: str = "2026-04-30_13-35-38",
    task_id: str = "1",
    trace_id: str = "trace-langfuse",
    include_level_metadata: bool = True,
    messages: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    metadata = {
        "actor": "overseer",
        "task_id": task_id,
        "run_id": 0,
        "phase": "initial",
        "step_index": 1,
        "tool_index": 0,
        "hook": "pre_tool",
        "trigger_type": "mutating_action",
        "overseer_mode": "non-thinking",
    }
    if include_level_metadata:
        metadata["level"] = 1
    return {
        "source_system": "C2-nt",
        "langfuse_session_id": session_id,
        "trace_id": trace_id,
        "observation_id": "obs-1",
        "name": "overseer.pre_tool.step_001.mutating_action",
        "metadata": metadata,
        "input_messages": messages
        or [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "{}"},
        ],
        "output": {
            "choices": [
                {
                    "message": {
                        "content": '{"action":"approve","decision_summary":"","violation_evidence":{"violated_contract_ids":[],"unmet_checklist_keys":[],"confidence":"low"},"guidance_lines":[],"corrected_observation":null}'
                    }
                }
            ]
        },
        "usage": {"input": 12, "output": 5},
        "source_export_path": "export.jsonl",
    }


def _messages_for_task_query(query: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "prompt"},
        {"role": "user", "content": json.dumps({"task_query": query})},
    ]


def test_lora_task_split_is_task_level_disjoint_and_has_expected_counts():
    split = builder.build_lora_task_split()
    builder.assert_split_disjoint_at_task_level(split)
    lookup = builder.split_lookup(split)

    assert len(lookup) == 120
    assert sum(1 for value in lookup.values() if value == "train") == 80
    assert sum(1 for value in lookup.values() if value == "val") == 16
    assert sum(1 for value in lookup.values() if value == "held_out") == 24
    assert split["counts"]["1"] == {"train": 35, "val": 7, "held_out": 8}
    assert split["counts"]["3"] == {"train": 10, "val": 2, "held_out": 8}


def test_pair_id_includes_session_id_to_avoid_multi_root_collisions():
    common = {
        "source_system": "C2-deepseek",
        "run_id": 0,
        "task_id": "4",
        "level": "1",
        "phase": "initial",
        "step_index": 1,
        "tool_index": 0,
        "hook": "pre_tool",
        "trigger_type": "mutating_action",
    }

    assert builder.make_pair_id(session_id="session-a", **common) != builder.make_pair_id(
        session_id="session-b",
        **common,
    )
    assert builder.make_pair_id(session_id="session-a", **common) == builder.make_pair_id(
        session_id="session-a",
        **common,
    )


def test_build_pairs_constructs_schema_and_parse_diagnostics():
    split = builder.build_lora_task_split()
    level, task_id = _first_task(split, "train")
    local = _local_record(task_id=task_id, level=level)
    export = _export_record(task_id=task_id)

    pairs, diagnostics = builder.build_pairs(
        local_records=[local],
        export_records=[export],
        split=split,
    )

    assert diagnostics["counts"]["matched"] == 1
    assert diagnostics["counts"]["parse_clean"] == 1
    pair = pairs[0]
    assert pair["source_system"] == "C2-nt"
    assert pair["split"] == "train"
    assert pair["decision_label"] == "approve"
    assert pair["input_messages"] == export["input_messages"]
    assert pair["input_messages_source"] == "langfuse"
    assert pair["target_text_source"] == "langfuse"
    assert pair["trace_id"] == "trace-langfuse"
    assert pair["input_tokens"] == 12
    assert pair["output_tokens"] == 5


def test_build_pairs_keeps_duplicate_langfuse_rows_from_level_collisions():
    split = builder.build_lora_task_split()
    task_id = "1"
    local_level_1 = _local_record(
        task_id=task_id,
        level="1",
        task_query="level one query",
    )
    local_level_2 = _local_record(
        task_id=task_id,
        level="2",
        task_query="level two query",
    )
    export_level_1 = _export_record(
        task_id=task_id,
        trace_id="collapsed-trace",
        include_level_metadata=False,
        messages=_messages_for_task_query("level one query"),
    )
    export_level_2 = _export_record(
        task_id=task_id,
        trace_id="collapsed-trace",
        include_level_metadata=False,
        messages=_messages_for_task_query("level two query"),
    )
    export_level_2["observation_id"] = "obs-2"

    pairs, diagnostics = builder.build_pairs(
        local_records=[local_level_1, local_level_2],
        export_records=[export_level_1, export_level_2],
        split=split,
    )

    assert diagnostics["counts"]["exported_runtime_final"] == 2
    assert diagnostics["counts"]["matched"] == 2
    assert diagnostics["counts"]["duplicate_langfuse"] == 1
    assert diagnostics["counts"]["langfuse_trace_query_collision_groups"] == 1
    assert {pair["level"] for pair in pairs} == {1, 2}
    assert {pair["match_resolution"] for pair in pairs} == {"task_query_match"}


def test_reasoning_content_is_stripped_from_target():
    text, stripped = builder.extract_target_text(
        {
            "output": {
                "choices": [
                    {
                        "message": {
                            "reasoning_content": "hidden reasoning",
                            "content": '{"action":"approve"}',
                        }
                    }
                ]
            }
        }
    )

    assert text == '{"action":"approve"}'
    assert stripped is True


def test_output_json_string_message_content_is_used_as_target():
    text, stripped = builder.extract_target_text(
        {
            "output": json.dumps(
                {
                    "role": "assistant",
                    "content": '{"action":"approve","decision_summary":""}',
                }
            )
        }
    )

    assert text == '{"action":"approve","decision_summary":""}'
    assert stripped is False
