from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import fire
from tqdm.auto import tqdm

try:
    from ._bootstrap import ensure_repo_root_on_path
    from .export_langfuse_overseer_observations import (
        SOURCE_SESSION_ROOTS,
        observation_metadata,
        observation_name,
        session_descriptor,
    )
except ImportError:
    from _bootstrap import ensure_repo_root_on_path
    from export_langfuse_overseer_observations import (
        SOURCE_SESSION_ROOTS,
        observation_metadata,
        observation_name,
        session_descriptor,
    )

ensure_repo_root_on_path()

from llm import build_langfuse_trace_id
from oversight.parsing import parse_final_verifier_json, parse_runtime_overseer_json

DEFAULT_OUTPUT_DIR = Path("outputs/deepplanning/lora_sft")
DEFAULT_EXPORT_GLOB = "outputs/deepplanning/langfuse_exports/*.jsonl"
SHOPPING_DATA_ROOT = (
    Path("external")
    / "qwen-agent"
    / "benchmark"
    / "deepplanning"
    / "shoppingplanning"
    / "data"
)
SPLIT_SEED = "lora-sft-v1"
SPLIT_COUNTS = {
    "1": {"train": 35, "val": 7, "held_out": 8},
    "2": {"train": 35, "val": 7, "held_out": 8},
    "3": {"train": 10, "val": 2, "held_out": 8},
}

@dataclass(frozen=True, slots=True)
class JoinKey:
    session_id: str
    task_id: str
    run_id: int
    phase: str
    step_index: int
    tool_index: int | None
    hook: str
    trigger_type: str

    def as_tuple(self) -> tuple[Any, ...]:
        return (
            self.session_id,
            self.task_id,
            self.run_id,
            self.phase,
            self.step_index,
            self.tool_index,
            self.hook,
            self.trigger_type,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "phase": self.phase,
            "step_index": self.step_index,
            "tool_index": self.tool_index,
            "hook": self.hook,
            "trigger_type": self.trigger_type,
        }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _json_dumps(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, indent=indent, sort_keys=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_state() -> dict[str, Any]:
    def run_git(args: list[str]) -> str:
        try:
            return subprocess.check_output(
                ["git", *args],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return ""

    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "dirty": bool(run_git(["status", "--porcelain"])),
    }


def _nested(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _level_from_path(path: Path) -> str | None:
    match = re.search(r"level[_-]?([123])", str(path))
    return match.group(1) if match else None


def _normalize_level(value: Any) -> str | None:
    if value is None or value == "":
        return None
    match = re.search(r"([123])", str(value))
    return match.group(1) if match else None


def _hook_from_trigger(trigger_type: str | None) -> str:
    trigger = str(trigger_type or "")
    if trigger == "final_checkpoint":
        return "final"
    if trigger == "coverage_deficit":
        return "midpoint"
    if trigger in {"error_occurrence", "always_on_post_tool"}:
        return "post_tool"
    if trigger in {"mutating_action", "loop_detection", "always_on_pre_tool"}:
        return "pre_tool"
    return ""


def _hook_from_export(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    hook = str(metadata.get("hook") or "")
    if hook:
        return hook
    name = str(record.get("name") or observation_name(record) or "")
    parts = name.split(".")
    return parts[1] if len(parts) >= 3 and parts[0] == "overseer" else ""


def _tool_index(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def make_pair_id(
    *,
    source_system: str,
    session_id: str,
    run_id: int,
    task_id: str,
    level: str,
    phase: str,
    step_index: int,
    tool_index: int | None,
    hook: str,
    trigger_type: str,
) -> str:
    tool_component = "none" if tool_index is None else str(tool_index)
    seed = (
        f"{source_system}|{session_id}|{run_id}|{task_id}|{level}|{phase}|"
        f"{step_index}|{tool_component}|{hook}|{trigger_type}"
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _task_key(level: str, task_id: str) -> str:
    return f"level_{level}:{task_id}"


def _hash_text(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _split_sort_key(level: str, task_id: str) -> str:
    return hashlib.sha256(
        f"{SPLIT_SEED}|level_{level}|{task_id}".encode("utf-8")
    ).hexdigest()


def build_lora_task_split(data_root: Path = SHOPPING_DATA_ROOT) -> dict[str, Any]:
    split: dict[str, Any] = {
        "version": "lora_task_split_v1",
        "seed": SPLIT_SEED,
        "counts": SPLIT_COUNTS,
        "policy": (
            "Held-out is level-balanced at 8 tasks per level; train/val split "
            "the remaining tasks 5:1 within each level."
        ),
        "splits": {"train": {}, "val": {}, "held_out": {}},
    }
    for level, counts in SPLIT_COUNTS.items():
        path = data_root / f"level_{level}_query_meta.json"
        records = json.loads(path.read_text(encoding="utf-8"))
        task_ids = sorted(
            (str(record["id"]) for record in records),
            key=lambda task_id: _split_sort_key(level, task_id),
        )
        expected = counts["train"] + counts["val"] + counts["held_out"]
        if len(task_ids) != expected:
            raise ValueError(
                f"Level {level} has {len(task_ids)} tasks; split expects {expected}."
            )
        train_end = counts["train"]
        val_end = train_end + counts["val"]
        split["splits"]["train"][f"level_{level}"] = sorted(
            task_ids[:train_end],
            key=lambda item: int(item) if item.isdigit() else item,
        )
        split["splits"]["val"][f"level_{level}"] = sorted(
            task_ids[train_end:val_end],
            key=lambda item: int(item) if item.isdigit() else item,
        )
        split["splits"]["held_out"][f"level_{level}"] = sorted(
            task_ids[val_end:],
            key=lambda item: int(item) if item.isdigit() else item,
        )
    split["digest"] = hashlib.sha256(
        json.dumps(split["splits"], sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return split


def split_lookup(split: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for split_name, levels in split["splits"].items():
        for level_key, task_ids in levels.items():
            level = str(level_key).removeprefix("level_")
            for task_id in task_ids:
                key = _task_key(level, str(task_id))
                if key in lookup:
                    raise ValueError(f"Task {key} appears in multiple LoRA splits.")
                lookup[key] = split_name
    return lookup


@lru_cache(maxsize=512)
def _task_query_lookup(level: str) -> dict[str, str]:
    path = SHOPPING_DATA_ROOT / f"level_{level}_query_meta.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    return {str(record["id"]): str(record.get("query") or "") for record in records}


def _task_query_for_level_task(level: str, task_id: str) -> str | None:
    query = _task_query_lookup(level).get(str(task_id))
    return query or None


def assert_split_disjoint_at_task_level(split: dict[str, Any]) -> None:
    counts = Counter(split_lookup(split).values())
    expected = {
        split_name: sum(level_counts[split_name] for level_counts in SPLIT_COUNTS.values())
        for split_name in ("train", "val", "held_out")
    }
    if dict(counts) != expected:
        raise ValueError(f"Unexpected LoRA split counts: {dict(counts)} != {expected}")


def load_export_records(paths: list[Path], *, progress: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in tqdm(
        paths,
        desc="load exports",
        unit="file",
        dynamic_ncols=True,
        disable=not progress,
    ):
        with path.open(encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    payload.setdefault("source_export_path", str(path))
                    payload.setdefault("source_export_line", line_number)
                    records.append(payload)
    return records


def export_join_key(record: dict[str, Any]) -> JoinKey | None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = observation_metadata(record)
    hook = _hook_from_export({**record, "metadata": metadata})
    trigger_type = str(metadata.get("trigger_type") or "")
    if not hook or not trigger_type:
        return None
    session_id = str(
        record.get("langfuse_session_id")
        or record.get("session_id")
        or record.get("sessionId")
        or ""
    )
    if not session_id:
        return None
    return JoinKey(
        session_id=session_id,
        task_id=str(metadata.get("task_id") or ""),
        run_id=int(metadata.get("run_id")),
        phase=str(metadata.get("phase") or ""),
        step_index=int(metadata.get("step_index")),
        tool_index=_tool_index(metadata.get("tool_index")),
        hook=hook,
        trigger_type=trigger_type,
    )


def _export_level(record: dict[str, Any]) -> str | None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = observation_metadata(record)
    return _normalize_level(metadata.get("level"))


def _messages_task_query(messages: Any) -> str | None:
    if not isinstance(messages, list):
        return None
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("task_query"), str):
            return payload["task_query"]
    return None


def _session_metadata(session_root: Path) -> dict[str, Any]:
    return _read_json(session_root / "experiment_session.json")


def _iter_event_file(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                payload["_line_number"] = line_number
                yield payload


def _session_local_records(
    session_root: Path,
    *,
    wanted_keys: set[tuple[Any, ...]] | None = None,
    progress: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    descriptor = session_descriptor(session_root)
    metadata = _session_metadata(session_root)
    records: list[dict[str, Any]] = []
    event_paths = sorted(session_root.rglob("agent_events.jsonl"))
    for event_path in tqdm(
        event_paths,
        desc=f"scan {descriptor['langfuse_session_id']}",
        unit="file",
        dynamic_ncols=True,
        disable=not progress,
    ):
        level = _level_from_path(event_path)
        if level is None:
            continue
        for event in tqdm(
            _iter_event_file(event_path),
            desc=event_path.parent.name,
            unit="event",
            leave=False,
            dynamic_ncols=True,
            disable=not progress,
        ):
            if not (
                event.get("event_type") == "oversight_step"
                and event.get("overseer_invoked") is True
            ):
                continue
            trigger_type = str(event.get("trigger_type") or "")
            hook = _hook_from_trigger(trigger_type)
            if not hook:
                continue
            event["_source_event_path"] = str(event_path)
            task_id = str(event.get("task_id") or "")
            run_id = int(event.get("run_id") or 0)
            trace_id = build_langfuse_trace_id(
                descriptor["langfuse_session_id"],
                "shopping",
                _nested(metadata, "parameters", "models", "executor"),
                _nested(metadata, "parameters", "system", "name"),
                f"run_{run_id}",
                f"level_{level}",
                f"sample_{task_id}",
            )
            join_key = JoinKey(
                session_id=str(descriptor["langfuse_session_id"]),
                task_id=task_id,
                run_id=run_id,
                phase=str(event.get("phase") or ""),
                step_index=int(event.get("step_index") or 0),
                tool_index=_tool_index(event.get("tool_index")),
                hook=hook,
                trigger_type=trigger_type,
            )
            if wanted_keys is not None and join_key.as_tuple() not in wanted_keys:
                continue
            record = {
                "source_system": descriptor["source_system"],
                "session_id": descriptor["langfuse_session_id"],
                "trace_id": trace_id,
                "level": level,
                "task_query": _task_query_for_level_task(level, task_id),
                "hook": hook,
                "source_event_path": str(event_path),
                "event": event,
                "join_key": join_key,
            }
            records.append(record)
    return records, descriptor


def load_local_records(
    session_roots: list[Path],
    *,
    wanted_keys: set[tuple[Any, ...]] | None = None,
    progress: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    descriptors: list[dict[str, Any]] = []
    for session_root in tqdm(
        session_roots,
        desc="local sessions",
        unit="session",
        dynamic_ncols=True,
        disable=not progress,
    ):
        session_records, descriptor = _session_local_records(
            session_root,
            wanted_keys=wanted_keys,
            progress=progress,
        )
        records.extend(session_records)
        descriptors.append(descriptor)
    return records, descriptors


def _message_from_output_payload(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if "choices" in value and isinstance(value["choices"], list) and value["choices"]:
            choice = value["choices"][0]
            if isinstance(choice, dict) and isinstance(choice.get("message"), dict):
                return choice["message"]
        if isinstance(value.get("message"), dict):
            return value["message"]
        if "content" in value or "reasoning_content" in value:
            return value
    return None


def extract_target_text(export_record: dict[str, Any]) -> tuple[str | None, bool]:
    output = export_record.get("output")
    if isinstance(export_record.get("output_text"), str):
        output = export_record["output_text"]
    if isinstance(output, str):
        try:
            parsed_output = json.loads(output)
        except json.JSONDecodeError:
            parsed_output = None
        if parsed_output is not None:
            message = _message_from_output_payload(parsed_output)
            if message is not None:
                content = message.get("content")
                reasoning = message.get("reasoning_content")
                if isinstance(content, str) and content.strip():
                    return content.strip(), bool(reasoning)
        text = output.strip()
        stripped = False
        if "<think>" in text and "</think>" in text:
            text = re.sub(r"(?s)<think>.*?</think>", "", text).strip()
            stripped = True
        return text or None, stripped
    message = _message_from_output_payload(output)
    if message is not None:
        content = message.get("content")
        reasoning = message.get("reasoning_content")
        if isinstance(content, str) and content.strip():
            return content.strip(), bool(reasoning)
    return None, False


def _parse_target(target_text: str, hook: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        if hook == "final":
            return parse_final_verifier_json(target_text), None
        return parse_runtime_overseer_json(target_text), None
    except Exception as exc:
        return None, str(exc)


def _decision_label(parsed_payload: dict[str, Any] | None) -> str | None:
    if not parsed_payload:
        return None
    return str(parsed_payload.get("action") or "") or None


def _usage(record: dict[str, Any]) -> dict[str, Any]:
    usage = record.get("usage")
    return usage if isinstance(usage, dict) else {}


def _usage_value(usage: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key in usage and usage[key] not in (None, ""):
            return int(float(usage[key]))
    return None


def _local_record_id(local: dict[str, Any]) -> tuple[str, Any]:
    event = local.get("event") if isinstance(local.get("event"), dict) else {}
    return (str(local.get("source_event_path") or ""), event.get("_line_number") or id(local))


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _time_delta_seconds(export_record: dict[str, Any], local: dict[str, Any]) -> float | None:
    export_time = _parse_timestamp(
        export_record.get("started_at")
        or export_record.get("startTime")
        or export_record.get("ended_at")
        or export_record.get("endTime")
    )
    event = local.get("event") if isinstance(local.get("event"), dict) else {}
    local_time = _parse_timestamp(event.get("timestamp"))
    if export_time is None or local_time is None:
        return None
    return abs((export_time - local_time).total_seconds())


def _json_text_equivalent(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    if left.strip() == right.strip():
        return True
    try:
        left_payload = json.loads(left)
        right_payload = json.loads(right)
    except json.JSONDecodeError:
        return False
    return json.dumps(left_payload, sort_keys=True, separators=(",", ":")) == json.dumps(
        right_payload,
        sort_keys=True,
        separators=(",", ":"),
    )


def _trace_query_collision_groups(
    export_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in export_records:
        trace_id = str(record.get("trace_id") or "")
        if not trace_id:
            continue
        session_id = str(
            record.get("langfuse_session_id")
            or record.get("session_id")
            or record.get("sessionId")
            or ""
        )
        task_query = _messages_task_query(record.get("input_messages"))
        if task_query:
            grouped[(session_id, trace_id)].append(record | {"_task_query": task_query})

    collisions: list[dict[str, Any]] = []
    for (session_id, trace_id), records in grouped.items():
        query_hashes = sorted(
            {
                str(_hash_text(str(record["_task_query"])))
                for record in records
                if record.get("_task_query")
            }
        )
        if len(query_hashes) <= 1:
            continue
        collisions.append(
            {
                "session_id": session_id,
                "trace_id": trace_id,
                "distinct_task_query_count": len(query_hashes),
                "observation_count": len(records),
                "task_ids": sorted(
                    {
                        str((record.get("metadata") or {}).get("task_id") or "")
                        for record in records
                    }
                ),
                "source_observation_ids": [
                    record.get("observation_id") for record in records[:20]
                ],
            }
        )
    return collisions


def _select_local_candidate(
    *,
    export_record: dict[str, Any],
    candidates: list[dict[str, Any]],
    used_local_ids: set[tuple[str, Any]],
    input_messages: list[dict[str, Any]] | None,
    target_text: str | None,
) -> dict[str, Any] | None:
    export_level = _export_level(export_record)
    export_query = _messages_task_query(input_messages)
    export_trace_id = str(export_record.get("trace_id") or "")
    evaluations: list[dict[str, Any]] = []

    for index, candidate in enumerate(candidates):
        local_id = _local_record_id(candidate)
        if local_id in used_local_ids:
            continue
        local_level = str(candidate.get("level") or "")
        if export_level is not None and export_level != local_level:
            continue
        local_query = candidate.get("task_query")
        if export_query and local_query and export_query != local_query:
            continue

        score = 0
        reasons: list[str] = []
        if export_level is not None and export_level == local_level:
            score += 500
            reasons.append("metadata_level_match")
        if export_query and local_query and export_query == local_query:
            score += 1000
            reasons.append("task_query_match")

        event = candidate.get("event") if isinstance(candidate.get("event"), dict) else {}
        if _json_text_equivalent(target_text, str(event.get("raw_overseer_text") or "")):
            score += 800
            reasons.append("target_text_match")

        if export_trace_id and export_trace_id == str(candidate.get("trace_id") or ""):
            score += 300
            reasons.append("trace_id_match")

        delta_seconds = _time_delta_seconds(export_record, candidate)
        if delta_seconds is not None:
            score += max(0, 100 - min(int(delta_seconds), 100))
            reasons.append("timestamp_nearest")

        evaluations.append(
            {
                "candidate": candidate,
                "index": index,
                "score": score,
                "reasons": reasons or ["first_candidate"],
                "time_delta_seconds": delta_seconds,
            }
        )

    if not evaluations:
        return None

    def sort_key(evaluation: dict[str, Any]) -> tuple[int, float, int]:
        delta = evaluation["time_delta_seconds"]
        delta_score = -float(delta) if delta is not None else float("-inf")
        return (int(evaluation["score"]), delta_score, -int(evaluation["index"]))

    selected = max(evaluations, key=sort_key)
    selected["candidate_count"] = len(evaluations)
    return selected


def _index_export_records(
    export_records: list[dict[str, Any]],
    *,
    progress: bool,
) -> tuple[list[tuple[JoinKey, dict[str, Any]]], list[dict[str, Any]]]:
    export_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keyed_export_records: list[tuple[JoinKey, dict[str, Any]]] = []
    for record in tqdm(
        export_records,
        desc="index exports",
        unit="record",
        dynamic_ncols=True,
        disable=not progress,
    ):
        key = export_join_key(record)
        if key is None:
            continue
        keyed_export_records.append((key, record))
        export_by_key[key.as_tuple()].append(record)
    return keyed_export_records, _duplicate_export_groups(export_by_key)


def _duplicate_export_groups(
    export_by_key: dict[tuple[Any, ...], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    return [
        JoinKey(*key_tuple).as_dict()
        | {
            "record_count": len(records),
            "source_observation_ids": [
                record.get("observation_id") for record in records[:20]
            ],
        }
        for key_tuple, records in export_by_key.items()
        if len(records) > 1
    ]


def _index_local_records(
    local_records: list[dict[str, Any]],
    *,
    progress: bool,
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    local_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for local in tqdm(
        local_records,
        desc="index locals",
        unit="record",
        dynamic_ncols=True,
        disable=not progress,
    ):
        key: JoinKey = local["join_key"]
        local_by_key[key.as_tuple()].append(local)
    return local_by_key


def _input_messages(export_record: dict[str, Any]) -> list[dict[str, Any]] | None:
    messages = export_record.get("input_messages")
    return messages if isinstance(messages, list) else None


def _unmatched_langfuse_entry(
    key: JoinKey, export_record: dict[str, Any]
) -> dict[str, Any]:
    return key.as_dict() | {"source_observation_id": export_record.get("observation_id")}


def _collision_entry(
    *,
    key: JoinKey,
    local: dict[str, Any],
    export_record: dict[str, Any],
    candidate_count: int,
    reasons: list[str],
) -> dict[str, Any]:
    return key.as_dict() | {
        "candidate_count": candidate_count,
        "selected_source_event_path": local["source_event_path"],
        "selected_level": local["level"],
        "source_observation_id": export_record.get("observation_id"),
        "resolution_reasons": reasons,
    }


def _has_strong_resolution(reasons: list[str]) -> bool:
    strong_reasons = {
        "metadata_level_match",
        "task_query_match",
        "target_text_match",
        "trace_id_match",
    }
    return any(reason in strong_reasons for reason in reasons)


def _target_info(
    *,
    key: JoinKey,
    local: dict[str, Any],
    langfuse_target_text: str | None,
    reasoning_stripped: bool,
) -> dict[str, Any]:
    target_text = langfuse_target_text
    target_source = "langfuse"
    if target_text is None:
        target_text = str(local["event"].get("raw_overseer_text") or "").strip()
        target_source = "local_fallback"

    parsed_payload, parse_error = (
        _parse_target(target_text, key.hook) if target_text else (None, "empty target")
    )
    return {
        "text": target_text,
        "source": target_source,
        "payload": parsed_payload,
        "parse_error": parse_error,
        "reasoning_stripped": reasoning_stripped,
    }


def _parse_failure_entry(
    *,
    key: JoinKey,
    local: dict[str, Any],
    export_record: dict[str, Any],
    error: str | None,
) -> dict[str, Any]:
    return key.as_dict() | {
        "source_event_path": local["source_event_path"],
        "source_observation_id": export_record.get("observation_id"),
        "error": error,
    }


def _pair_record(
    *,
    key: JoinKey,
    local: dict[str, Any],
    export_record: dict[str, Any],
    input_messages: list[dict[str, Any]] | None,
    target: dict[str, Any],
    match_resolution: str,
    split_by_task: dict[str, str],
) -> dict[str, Any]:
    usage = _usage(export_record)
    level = str(local["level"])
    export_task_query = _messages_task_query(input_messages)
    local_task_query = local.get("task_query")
    parsed_payload = target["payload"]
    return {
        "pair_id": make_pair_id(
            source_system=str(local["source_system"]),
            session_id=key.session_id,
            run_id=key.run_id,
            task_id=key.task_id,
            level=level,
            phase=key.phase,
            step_index=key.step_index,
            tool_index=key.tool_index,
            hook=key.hook,
            trigger_type=key.trigger_type,
        ),
        "source_system": local["source_system"],
        "session_id": key.session_id,
        "trace_id": export_record.get("trace_id") or local.get("trace_id"),
        "task_id": key.task_id,
        "level": int(level),
        "task_key": _task_key(level, key.task_id),
        "split": split_by_task.get(_task_key(level, key.task_id), "unknown"),
        "run_id": key.run_id,
        "phase": key.phase,
        "step_index": key.step_index,
        "tool_index": key.tool_index,
        "hook": key.hook,
        "trigger_type": key.trigger_type,
        "overseer_mode": local["event"].get("overseer_mode")
        or (export_record.get("metadata") or {}).get("overseer_mode"),
        "input_messages": input_messages,
        "input_messages_source": "langfuse" if input_messages is not None else "missing",
        "target_text": target["text"],
        "target_text_source": target["source"],
        "parsed_payload": parsed_payload,
        "decision_label": _decision_label(parsed_payload),
        "reasoning_stripped": target["reasoning_stripped"],
        "parse_clean": parsed_payload is not None,
        "match_resolution": match_resolution,
        "langfuse_task_query_hash": _hash_text(export_task_query),
        "local_task_query_hash": _hash_text(
            str(local_task_query) if local_task_query else None
        ),
        "source_event_path": local["source_event_path"],
        "source_observation_id": export_record.get("observation_id"),
        "source_export_path": export_record.get("source_export_path"),
        "local_raw_overseer_text": local["event"].get("raw_overseer_text"),
        "langfuse_model": export_record.get("model"),
        "input_tokens": _usage_value(
            usage,
            "input",
            "prompt_tokens",
            "input_tokens",
        )
        or local["event"].get("overseer_input_tokens"),
        "output_tokens": _usage_value(
            usage,
            "output",
            "completion_tokens",
            "output_tokens",
        )
        or local["event"].get("overseer_output_tokens"),
        "usage": usage,
    }


def _unmatched_local_entries(
    local_records: list[dict[str, Any]],
    used_local_ids: set[tuple[str, Any]],
) -> list[dict[str, Any]]:
    return [
        local["join_key"].as_dict()
        | {
            "source_event_path": local["source_event_path"],
            "level": local.get("level"),
        }
        for local in local_records
        if _local_record_id(local) not in used_local_ids
    ]


def _diagnostics(
    *,
    pairs: list[dict[str, Any]],
    local_records: list[dict[str, Any]],
    keyed_export_records: list[tuple[JoinKey, dict[str, Any]]],
    unmatched_local: list[dict[str, Any]],
    unmatched_langfuse: list[dict[str, Any]],
    duplicate_exports: list[dict[str, Any]],
    local_collision_candidates: list[dict[str, Any]],
    ambiguous_order_matches: list[dict[str, Any]],
    parse_failures: list[dict[str, Any]],
) -> dict[str, Any]:
    trace_query_collisions = _trace_query_collision_groups(
        [record for _, record in keyed_export_records]
    )
    return {
        "counts": {
            "local_invoked": len(local_records),
            "exported_runtime_final": len(keyed_export_records),
            "matched": len(pairs),
            "unmatched_local": len(unmatched_local),
            "unmatched_langfuse": len(unmatched_langfuse),
            "duplicate_langfuse": sum(item["record_count"] - 1 for item in duplicate_exports),
            "duplicate_langfuse_groups": len(duplicate_exports),
            "local_collision_candidates": len(local_collision_candidates),
            "ambiguous_order_matches": len(ambiguous_order_matches),
            "langfuse_trace_query_collision_groups": len(trace_query_collisions),
            "parse_clean": sum(1 for pair in pairs if pair["parse_clean"]),
            "input_messages_present": sum(
                1 for pair in pairs if pair["input_messages"] is not None
            ),
            "fallback_or_error": len(parse_failures),
        },
        "by_source_system": _counts_by_source(
            pairs,
            local_records,
            [record for _, record in keyed_export_records],
        ),
        "unmatched_local": unmatched_local[:1000],
        "unmatched_langfuse": unmatched_langfuse[:1000],
        "duplicate_langfuse": duplicate_exports[:1000],
        "local_collision_candidates": local_collision_candidates[:1000],
        "ambiguous_order_matches": ambiguous_order_matches[:1000],
        "langfuse_trace_query_collision_groups": trace_query_collisions[:1000],
        "parse_failures": parse_failures[:1000],
    }


def build_pairs(
    *,
    local_records: list[dict[str, Any]],
    export_records: list[dict[str, Any]],
    split: dict[str, Any],
    progress: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    keyed_export_records, duplicate_exports = _index_export_records(
        export_records, progress=progress
    )
    local_by_key = _index_local_records(local_records, progress=progress)

    pairs: list[dict[str, Any]] = []
    unmatched_langfuse: list[dict[str, Any]] = []
    parse_failures: list[dict[str, Any]] = []
    local_collision_candidates: list[dict[str, Any]] = []
    ambiguous_order_matches: list[dict[str, Any]] = []
    split_by_task = split_lookup(split)
    used_local_ids: set[tuple[str, Any]] = set()

    for key, export_record in tqdm(
        keyed_export_records,
        desc="build pairs",
        unit="pair",
        dynamic_ncols=True,
        disable=not progress,
    ):
        candidates = local_by_key.get(key.as_tuple(), [])
        input_messages = _input_messages(export_record)
        target_text, reasoning_stripped = extract_target_text(export_record)
        selected = _select_local_candidate(
            export_record=export_record,
            candidates=candidates,
            used_local_ids=used_local_ids,
            input_messages=input_messages,
            target_text=target_text,
        )
        if selected is None:
            unmatched_langfuse.append(_unmatched_langfuse_entry(key, export_record))
            continue

        local = selected["candidate"]
        used_local_ids.add(_local_record_id(local))
        reasons = selected["reasons"]
        if len(candidates) > 1:
            collision = _collision_entry(
                key=key,
                local=local,
                export_record=export_record,
                candidate_count=len(candidates),
                reasons=reasons,
            )
            local_collision_candidates.append(collision)
            if not _has_strong_resolution(reasons):
                ambiguous_order_matches.append(collision)

        target = _target_info(
            key=key,
            local=local,
            langfuse_target_text=target_text,
            reasoning_stripped=reasoning_stripped,
        )
        if target["payload"] is None:
            parse_failures.append(
                _parse_failure_entry(
                    key=key,
                    local=local,
                    export_record=export_record,
                    error=target["parse_error"],
                )
            )

        pairs.append(
            _pair_record(
                key=key,
                local=local,
                export_record=export_record,
                input_messages=input_messages,
                target=target,
                match_resolution=str(reasons[0]),
                split_by_task=split_by_task,
            )
        )

    diagnostics = _diagnostics(
        pairs=pairs,
        local_records=local_records,
        keyed_export_records=keyed_export_records,
        unmatched_local=_unmatched_local_entries(local_records, used_local_ids),
        unmatched_langfuse=unmatched_langfuse,
        duplicate_exports=duplicate_exports,
        local_collision_candidates=local_collision_candidates,
        ambiguous_order_matches=ambiguous_order_matches,
        parse_failures=parse_failures,
    )
    return pairs, diagnostics


def _counts_by_source(
    pairs: list[dict[str, Any]],
    local_records: list[dict[str, Any]],
    export_records: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    systems = sorted(
        {
            str(record.get("source_system"))
            for record in local_records
            if record.get("source_system")
        }
        | {
            str(record.get("source_system"))
            for record in export_records
            if record.get("source_system")
        }
    )
    counts: dict[str, dict[str, int]] = {}
    for system in systems:
        local_count = sum(1 for record in local_records if record["source_system"] == system)
        export_count = sum(
            1 for record in export_records if record.get("source_system") == system
        )
        system_pairs = [pair for pair in pairs if pair["source_system"] == system]
        counts[system] = {
            "local_invoked": local_count,
            "exported_runtime_final": export_count,
            "matched": len(system_pairs),
            "parse_clean": sum(1 for pair in system_pairs if pair["parse_clean"]),
            "input_messages_present": sum(
                1 for pair in system_pairs if pair["input_messages"] is not None
            ),
            "fallback_or_error": sum(1 for pair in system_pairs if not pair["parse_clean"]),
        }
    return counts


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_dumps(payload, indent=2) + "\n", encoding="utf-8")


def _check_a_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    smoke_systems = {"C2-nt", "C2-deepseek-nt"}
    smoke_pairs = [pair for pair in pairs if pair["source_system"] in smoke_systems]
    systems_present = sorted({str(pair["source_system"]) for pair in smoke_pairs})
    joined = len(smoke_pairs)
    parse_clean = sum(1 for pair in smoke_pairs if pair["parse_clean"])
    input_messages_present = sum(
        1 for pair in smoke_pairs if pair["input_messages"] is not None
    )
    parse_rate = parse_clean / joined if joined else 0.0
    input_messages_rate = input_messages_present / joined if joined else 0.0
    langfuse_authoritative_pass = (
        smoke_systems <= set(systems_present)
        and joined > 0
        and parse_rate >= 0.99
        and input_messages_rate >= 0.99
    )
    return {
        "policy": "langfuse_authoritative_input",
        "systems": sorted(smoke_systems),
        "joined": joined,
        "systems_present": systems_present,
        "parse_clean": parse_clean,
        "input_messages_present": input_messages_present,
        "parse_clean_rate": parse_rate,
        "input_messages_rate": input_messages_rate,
        "langfuse_authoritative_input_pass": langfuse_authoritative_pass,
        "pass": langfuse_authoritative_pass,
    }


def write_outputs(
    *,
    pairs: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    split: dict[str, Any],
    descriptors: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_pairs = [
        pair
        for pair in pairs
        if pair["parse_clean"] and pair["input_messages"] is not None and pair["split"] != "unknown"
    ]
    by_split = {
        "train": [pair for pair in clean_pairs if pair["split"] == "train"],
        "val": [pair for pair in clean_pairs if pair["split"] == "val"],
        "held_out": [pair for pair in clean_pairs if pair["split"] == "held_out"],
    }

    _write_jsonl(output_dir / "all_pairs.jsonl", pairs)
    _write_jsonl(output_dir / "train.jsonl", by_split["train"])
    _write_jsonl(output_dir / "val.jsonl", by_split["val"])
    _write_jsonl(output_dir / "held_out.jsonl", by_split["held_out"])
    _write_json(output_dir / "diagnostics.json", diagnostics)
    _write_json(output_dir / "task_split_v1.json", split)

    manifest = {
        "created_at": _utc_now(),
        "source_sessions": descriptors,
        "split_digest": split["digest"],
        "counts": diagnostics["counts"]
        | {
            "all_pairs": len(pairs),
            "train_pairs": len(by_split["train"]),
            "val_pairs": len(by_split["val"]),
            "held_out_pairs": len(by_split["held_out"]),
        },
        "check_a": _check_a_summary(pairs),
        "held_out_semantics": (
            "held_out.jsonl contains SFT pairs for final offline audit/evaluation "
            "only. It must not be loaded by training, early stopping, "
            "hyperparameter sweeps, or model selection."
        ),
        "git": _git_state(),
    }
    _write_json(output_dir / "manifest.json", manifest)
    report = [
        "# LoRA SFT Pair Build",
        "",
        f"- All joined pairs: {len(pairs)}",
        f"- Train pairs: {len(by_split['train'])}",
        f"- Validation pairs: {len(by_split['val'])}",
        f"- Held-out pairs: {len(by_split['held_out'])}",
        f"- Parse-clean pairs: {diagnostics['counts']['parse_clean']}",
        f"- Langfuse input-message pairs: {diagnostics['counts']['input_messages_present']}",
        "",
        "Held-out pairs are written for final offline audit/evaluation only and must "
        "not be used for training, early stopping, hyperparameter sweeps, or model "
        "selection.",
        "",
        f"Check A policy: {manifest['check_a']['policy']}",
        f"Check A PASS: {manifest['check_a']['pass']}",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return manifest


def run(
    *,
    session_roots: list[Path],
    export_paths: list[Path],
    output_dir: Path,
    progress: bool = False,
) -> dict[str, Any]:
    split = build_lora_task_split()
    assert_split_disjoint_at_task_level(split)
    export_records = load_export_records(export_paths, progress=progress)
    wanted_keys = {
        key.as_tuple()
        for key in (export_join_key(record) for record in export_records)
        if key is not None
    }
    local_records, descriptors = load_local_records(
        session_roots,
        wanted_keys=wanted_keys,
        progress=progress,
    )
    pairs, diagnostics = build_pairs(
        local_records=local_records,
        export_records=export_records,
        split=split,
        progress=progress,
    )
    return write_outputs(
        pairs=pairs,
        diagnostics=diagnostics,
        split=split,
        descriptors=descriptors,
        output_dir=output_dir,
    )


def _parse_export_paths(values: list[str] | None) -> list[Path]:
    if values:
        return [Path(value) for value in values]
    return sorted(Path().glob(DEFAULT_EXPORT_GLOB))


def _parse_session_roots(values: list[str] | None) -> list[Path]:
    if values:
        return [Path(value) for value in values]
    return list(SOURCE_SESSION_ROOTS)


def _coerce_path_values(value: Any) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, Path):
        return [value]
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",") if part.strip()]
        return [Path(part) for part in values]
    return [Path(item) for item in value]


def main(
    session_root: Any = None,
    session_roots: Any = None,
    export: Any = None,
    exports: Any = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    progress: bool = True,
) -> None:
    """Build LoRA SFT overseer pairs by joining local logs to Langfuse exports.

    Fire examples:
      pixi run python scripts/build_lora_sft_pairs.py \
        --session_root='["outputs/.../session-a","outputs/.../session-b"]' \
        --export='["outputs/.../export-a.jsonl","outputs/.../export-b.jsonl"]'
    """
    selected_exports = _coerce_path_values(exports) + _coerce_path_values(export)
    export_paths = _parse_export_paths([str(path) for path in selected_exports] or None)
    if not export_paths:
        print(
            "error: no Langfuse export JSONL files found; run "
            "export_langfuse_overseer_observations.py first or pass --export.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    selected_session_roots = _coerce_path_values(session_roots) + _coerce_path_values(
        session_root
    )
    manifest = run(
        session_roots=_parse_session_roots(
            [str(path) for path in selected_session_roots] or None
        ),
        export_paths=export_paths,
        output_dir=Path(output_dir),
        progress=progress,
    )
    print(
        f"Wrote {manifest['counts']['all_pairs']} pairs to {Path(output_dir)}; "
        f"Check A pass={manifest['check_a']['pass']}"
    )


if __name__ == "__main__":
    fire.Fire(main)
