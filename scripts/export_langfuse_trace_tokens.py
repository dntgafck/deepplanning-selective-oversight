from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import yaml
from tqdm.auto import tqdm

try:
    from ._bootstrap import ensure_repo_root_on_path
    from .export_langfuse_overseer_observations import (
        DEFAULT_FIELDS,
        fetch_observations,
        observation_metadata,
        observation_name,
    )
    from .fetch_langfuse_session_usage import DEFAULT_LANGFUSE_HOST
except ImportError:
    from _bootstrap import ensure_repo_root_on_path
    from export_langfuse_overseer_observations import (
        DEFAULT_FIELDS,
        fetch_observations,
        observation_metadata,
        observation_name,
    )
    from fetch_langfuse_session_usage import DEFAULT_LANGFUSE_HOST

ensure_repo_root_on_path()

from deepplanning.config import load_dotenv
from llm import build_langfuse_trace_id

DEFAULT_EXPERIMENTS_ROOT = Path("outputs/deepplanning/experiments")
DEFAULT_OUTPUT_DIR = Path("outputs/deepplanning/langfuse_trace_tokens")
DEFAULT_SPLIT_PATH = Path("configs/shopping/splits.yaml")
DEFAULT_EXPERIMENTS = (
    "shopping-c2",
    "shopping-c2-nt",
    "shopping-c2-deepseek",
    "shopping-c2-deepseek-nt",
    "shopping-c2-lora",
    "shopping-c2-deepseek-lora",
)
SETUP_TRIGGER_TYPES = {"compile_contract", "compile_checklist"}
SETUP_HOOKS = {"setup"}
TOKEN_COLUMNS = [
    "input_total_tokens",
    "input_uncached_tokens",
    "input_cached_tokens",
    "output_total_tokens",
    "output_text_tokens",
    "output_reasoning_tokens",
    "total_tokens",
    "usage_reported_total_tokens",
]
CACHED_INPUT_KEYS = (
    "input_cached_tokens",
    "cached_tokens",
    "cache_read_input_tokens",
    "cached_input_tokens",
    "prompt_cache_hit_tokens",
    "input_cache_hit_tokens",
    "prompt_tokens_details.cached_tokens",
)
INPUT_UNCACHED_KEYS = (
    "input",
    "input_uncached_tokens",
    "cache_miss_input_tokens",
    "prompt_cache_miss_tokens",
    "input_cache_miss_tokens",
)
INPUT_TOTAL_KEYS = ("input_tokens", "prompt_tokens", "prompt")
OUTPUT_TEXT_KEYS = ("output", "output_text_tokens")
OUTPUT_TOTAL_KEYS = ("output_tokens", "completion_tokens", "completion")
OUTPUT_REASONING_KEYS = (
    "output_reasoning_tokens",
    "reasoning_tokens",
    "completion_tokens_details.reasoning_tokens",
)
TOTAL_KEYS = ("total_tokens", "total")


@dataclass(frozen=True, slots=True)
class CallKey:
    actor: str
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
            self.actor,
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
            "actor": self.actor,
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


def _dotted(mapping: dict[str, Any], key: str) -> Any:
    if "." not in key:
        return mapping.get(key)
    return _nested(mapping, *key.split("."))


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _usage_value(usage: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = _dotted(usage, key)
        coerced = _coerce_int(value)
        if coerced is not None:
            return coerced
    return None


def _has_any_usage_key(usage: dict[str, Any], keys: Iterable[str]) -> bool:
    return any(_dotted(usage, key) not in (None, "") for key in keys)


def _normalize_slug(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def normalize_system_label(experiment_name: str, metadata_system: str = "") -> str:
    slug = _normalize_slug(experiment_name)
    for prefix in ("shopping-", "system-"):
        if slug.startswith(prefix):
            slug = slug[len(prefix) :]
            break
    parts = [part for part in slug.split("-") if part]
    if parts:
        head = parts[0]
        if head in {"a", "b", "c1", "c2", "d"}:
            label = head.upper()
            tail = "-".join(parts[1:])
            return f"{label}-{tail}" if tail else label
    if metadata_system:
        return metadata_system.upper()
    return experiment_name


def _metadata_experiment_name(metadata: dict[str, Any], session_root: Path) -> str:
    return str(
        _nested(metadata, "experiment", "name")
        or _nested(metadata, "parameters", "name")
        or session_root.parent.name
    )


def _metadata_system_name(metadata: dict[str, Any]) -> str:
    return str(_nested(metadata, "parameters", "system", "name") or "")


def _metadata_model_name(metadata: dict[str, Any], key: str) -> str:
    return str(_nested(metadata, "parameters", "models", key) or "")


def _langfuse_session_id(metadata: dict[str, Any], session_root: Path) -> str:
    return str(metadata.get("timestamp") or session_root.name)


def session_descriptor(session_root: Path) -> dict[str, Any]:
    metadata = _read_json(session_root / "experiment_session.json")
    experiment_name = _metadata_experiment_name(metadata, session_root)
    metadata_system = _metadata_system_name(metadata)
    return {
        "session_root": str(session_root),
        "experiment_name": experiment_name,
        "system": normalize_system_label(experiment_name, metadata_system),
        "metadata_system": metadata_system,
        "executor_model": _metadata_model_name(metadata, "executor"),
        "overseer_model": _metadata_model_name(metadata, "overseer"),
        "langfuse_session_id": _langfuse_session_id(metadata, session_root),
        "shopping_split": str(_nested(metadata, "parameters", "shopping", "split") or ""),
    }


def discover_session_roots(
    experiments_root: Path,
    *,
    experiment_names: Iterable[str] = DEFAULT_EXPERIMENTS,
) -> list[Path]:
    roots: list[Path] = []
    for experiment_name in experiment_names:
        experiment_dir = experiments_root / experiment_name
        if not experiment_dir.exists():
            continue
        for metadata_path in sorted(experiment_dir.glob("*/experiment_session.json")):
            session_root = metadata_path.parent
            if list((session_root / "aggregated_results").glob("*_aggregated.json")):
                roots.append(session_root)
    return roots


def load_split_lookup(split_path: Path = DEFAULT_SPLIT_PATH) -> dict[str, str]:
    if not split_path.exists():
        return {}
    payload = yaml.safe_load(split_path.read_text(encoding="utf-8")) or {}
    lookup: dict[str, str] = {}
    if not isinstance(payload, dict):
        return lookup
    for split_name, levels in payload.items():
        if not isinstance(levels, dict):
            continue
        for level_key, task_ids in levels.items():
            level = str(level_key).removeprefix("level_")
            for task_id in task_ids or []:
                lookup[f"level_{level}:{task_id}"] = str(split_name)
    return lookup


def _case_split(split_lookup: dict[str, str], level: Any, task_id: Any) -> str:
    normalized_level = _normalize_level(level)
    if normalized_level is None or task_id in (None, ""):
        return ""
    return split_lookup.get(f"level_{normalized_level}:{task_id}", "non_hold_out")


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


def _hook_from_name(name: str) -> str:
    parts = name.split(".")
    return parts[1] if len(parts) >= 3 and parts[0] == "overseer" else ""


def _actor_from_name(name: str) -> str:
    head = name.split(".", 1)[0].strip()
    return head if head in {"executor", "overseer"} else ""


def _step_from_name(name: str) -> int | None:
    match = re.search(r"step[_-](\d+)", name)
    return int(match.group(1)) if match else None


def _usage_payload(observation: dict[str, Any]) -> dict[str, Any]:
    for key in ("usageDetails", "usage", "usage_details"):
        value = observation.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _extract_input_messages(observation: dict[str, Any]) -> list[dict[str, Any]] | None:
    candidate = observation.get("input")
    if isinstance(candidate, str):
        try:
            candidate = json.loads(candidate)
        except json.JSONDecodeError:
            return None
    if isinstance(candidate, dict) and isinstance(candidate.get("messages"), list):
        return [dict(item) for item in candidate["messages"] if isinstance(item, dict)]
    if isinstance(candidate, list) and all(isinstance(item, dict) for item in candidate):
        return [dict(item) for item in candidate]
    if isinstance(candidate, dict) and isinstance(candidate.get("kwargs"), dict):
        messages = candidate["kwargs"].get("messages")
        if isinstance(messages, list):
            return [dict(item) for item in messages if isinstance(item, dict)]
    return None


def normalize_observation(
    observation: dict[str, Any],
    *,
    descriptor: dict[str, Any],
) -> dict[str, Any]:
    metadata = observation_metadata(observation)
    name = observation_name(observation)
    actor = str(metadata.get("actor") or _actor_from_name(name) or "")
    session_id = str(
        observation.get("sessionId")
        or observation.get("session_id")
        or descriptor["langfuse_session_id"]
    )
    return {
        "source": "langfuse",
        "source_session_root": descriptor["session_root"],
        "experiment_name": descriptor["experiment_name"],
        "system": descriptor["system"],
        "executor_model": descriptor["executor_model"],
        "overseer_model": descriptor["overseer_model"],
        "session_id": session_id,
        "trace_id": observation.get("traceId") or observation.get("trace_id"),
        "observation_id": observation.get("id"),
        "name": name,
        "type": observation.get("type"),
        "started_at": observation.get("startTime") or observation.get("startedAt"),
        "ended_at": observation.get("endTime") or observation.get("endedAt"),
        "model": (
            observation.get("providedModelName")
            or observation.get("model")
            or observation.get("internalModelId")
        ),
        "actor": actor,
        "usage": _usage_payload(observation),
        "metadata": metadata,
        "input_messages": _extract_input_messages(observation),
    }


def _is_setup_observation(record: dict[str, Any]) -> bool:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    hook = str(metadata.get("hook") or _hook_from_name(str(record.get("name") or "")))
    trigger_type = str(metadata.get("trigger_type") or "")
    return hook in SETUP_HOOKS or trigger_type in SETUP_TRIGGER_TYPES


def _observation_call_key(record: dict[str, Any]) -> CallKey | None:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    actor = str(metadata.get("actor") or record.get("actor") or "").strip()
    if actor not in {"executor", "overseer"}:
        return None

    name = str(record.get("name") or "")
    step_index = _coerce_int(metadata.get("step_index")) or _step_from_name(name)
    if step_index is None:
        return None
    session_id = str(record.get("session_id") or record.get("langfuse_session_id") or "")
    task_id = str(metadata.get("task_id") or "")
    run_id = _coerce_int(metadata.get("run_id"))
    if not session_id or not task_id or run_id is None:
        return None

    trigger_type = "" if actor == "executor" else str(metadata.get("trigger_type") or "")
    hook = "" if actor == "executor" else str(metadata.get("hook") or _hook_from_name(name))
    tool_index = (
        None if actor == "executor" else _coerce_int(metadata.get("tool_index"))
    )
    return CallKey(
        actor=actor,
        session_id=session_id,
        task_id=task_id,
        run_id=run_id,
        phase=str(metadata.get("phase") or ""),
        step_index=step_index,
        tool_index=tool_index,
        hook=hook,
        trigger_type=trigger_type,
    )


def token_counts_from_usage(
    usage: dict[str, Any],
    *,
    fallback_input_total: int | None = None,
    fallback_output_total: int | None = None,
    assume_uncached_when_cache_missing: bool = True,
) -> dict[str, int | None]:
    input_cached = _usage_value(usage, *CACHED_INPUT_KEYS) or 0
    input_uncached = _usage_value(usage, *INPUT_UNCACHED_KEYS)
    input_total = _usage_value(usage, *INPUT_TOTAL_KEYS)
    if input_total is None:
        if input_uncached is not None:
            input_total = input_uncached + input_cached
        else:
            input_total = fallback_input_total
    if input_uncached is None and input_total is not None:
        if input_cached:
            input_uncached = max(input_total - input_cached, 0)
        elif assume_uncached_when_cache_missing:
            input_uncached = input_total

    output_reasoning = _usage_value(usage, *OUTPUT_REASONING_KEYS) or 0
    output_text = _usage_value(usage, *OUTPUT_TEXT_KEYS)
    output_total = _usage_value(usage, *OUTPUT_TOTAL_KEYS)
    if output_total is None:
        if output_text is not None:
            output_total = output_text + output_reasoning
        else:
            output_total = fallback_output_total
    if output_text is None and output_total is not None:
        output_text = max(output_total - output_reasoning, 0)

    usage_total = _usage_value(usage, *TOTAL_KEYS)
    total_tokens = None
    if input_total is not None or output_total is not None:
        total_tokens = (input_total or 0) + (output_total or 0)
    elif usage_total is not None:
        total_tokens = usage_total

    return {
        "input_total_tokens": input_total,
        "input_uncached_tokens": input_uncached,
        "input_cached_tokens": input_cached,
        "output_total_tokens": output_total,
        "output_text_tokens": output_text,
        "output_reasoning_tokens": output_reasoning,
        "total_tokens": total_tokens,
        "usage_reported_total_tokens": usage_total,
    }


def _cache_split_status(usage: dict[str, Any], counts: dict[str, int | None]) -> str:
    if _has_any_usage_key(usage, CACHED_INPUT_KEYS) or _has_any_usage_key(
        usage, INPUT_UNCACHED_KEYS
    ):
        return "provided"
    if counts["input_total_tokens"] is None:
        return "missing"
    if counts["input_uncached_tokens"] == counts["input_total_tokens"]:
        return "assumed_uncached"
    return "unknown_total_only"


def _local_trace_id(
    descriptor: dict[str, Any],
    *,
    level: str | None,
    task_id: str,
    run_id: int,
) -> str:
    return build_langfuse_trace_id(
        descriptor["langfuse_session_id"],
        "shopping",
        descriptor["executor_model"],
        descriptor["metadata_system"],
        f"run_{run_id}",
        f"level_{level or 'unknown'}",
        f"sample_{task_id}",
    )


def _iter_event_file(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                payload["_line_number"] = line_number
                yield payload


def _local_record_from_event(
    *,
    event: dict[str, Any],
    event_path: Path,
    level: str | None,
    descriptor: dict[str, Any],
) -> dict[str, Any] | None:
    event_type = str(event.get("event_type") or "")
    if event_type == "executor_turn":
        actor = "executor"
        step_index = _coerce_int(event.get("turn_index"))
        tool_index = None
        hook = ""
        trigger_type = ""
        model = str(event.get("model_alias") or descriptor["executor_model"])
    elif event_type == "oversight_step" and event.get("overseer_invoked") is True:
        actor = "overseer"
        step_index = _coerce_int(event.get("step_index"))
        tool_index = _coerce_int(event.get("tool_index"))
        trigger_type = str(event.get("trigger_type") or "")
        hook = _hook_from_trigger(trigger_type)
        model = descriptor["overseer_model"]
    else:
        return None

    task_id = str(event.get("task_id") or "")
    run_id = _coerce_int(event.get("run_id"))
    if not task_id or run_id is None or step_index is None:
        return None
    key = CallKey(
        actor=actor,
        session_id=str(descriptor["langfuse_session_id"]),
        task_id=task_id,
        run_id=run_id,
        phase=str(event.get("phase") or ""),
        step_index=step_index,
        tool_index=tool_index,
        hook=hook,
        trigger_type=trigger_type,
    )
    return {
        "source": "local_agent_log",
        "source_session_root": descriptor["session_root"],
        "experiment_name": descriptor["experiment_name"],
        "system": descriptor["system"],
        "executor_model": descriptor["executor_model"],
        "overseer_model": descriptor["overseer_model"],
        "session_id": descriptor["langfuse_session_id"],
        "trace_id": _local_trace_id(
            descriptor,
            level=level,
            task_id=task_id,
            run_id=run_id,
        ),
        "actor": actor,
        "model": model,
        "domain": str(event.get("domain") or "shopping"),
        "level": level,
        "task_id": task_id,
        "case_name": f"case_{task_id}",
        "run_id": run_id,
        "phase": key.phase,
        "step_index": step_index,
        "tool_index": tool_index,
        "hook": hook,
        "trigger_type": trigger_type,
        "source_event_path": str(event_path),
        "source_event_line": event.get("_line_number"),
        "event": event,
        "call_key": key,
    }


def _local_dedupe_key(record: dict[str, Any]) -> str:
    event = record.get("event") if isinstance(record.get("event"), dict) else {}
    usage = _local_usage(record)
    payload = {
        "call_key": record["call_key"].as_tuple(),
        "input": event.get("prompt_tokens")
        if record["actor"] == "executor"
        else event.get("overseer_input_tokens"),
        "output": event.get("completion_tokens")
        if record["actor"] == "executor"
        else event.get("overseer_output_tokens"),
        "usage": usage,
        "raw_overseer_text": event.get("raw_overseer_text")
        if record["actor"] == "overseer"
        else None,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _local_usage(record: dict[str, Any]) -> dict[str, Any]:
    event = record.get("event") if isinstance(record.get("event"), dict) else {}
    if record.get("actor") == "executor":
        raw_response = event.get("raw_response") if isinstance(event, dict) else {}
        usage = raw_response.get("usage") if isinstance(raw_response, dict) else {}
        usage = dict(usage) if isinstance(usage, dict) else {}
        if event.get("prompt_tokens") is not None:
            usage.setdefault("prompt_tokens", event.get("prompt_tokens"))
        if event.get("completion_tokens") is not None:
            usage.setdefault("completion_tokens", event.get("completion_tokens"))
        return usage
    return {}


def load_local_call_records(
    session_roots: Iterable[Path],
    *,
    progress: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    duplicate_count = 0
    seen: set[str] = set()
    for session_root in tqdm(
        list(session_roots),
        desc="local sessions",
        unit="session",
        dynamic_ncols=True,
        disable=not progress,
    ):
        descriptor = session_descriptor(Path(session_root))
        event_paths = sorted(Path(session_root).rglob("agent_events.jsonl"))
        for event_path in tqdm(
            event_paths,
            desc=Path(session_root).name,
            unit="file",
            leave=False,
            dynamic_ncols=True,
            disable=not progress,
        ):
            level = _level_from_path(event_path)
            for event in _iter_event_file(event_path):
                record = _local_record_from_event(
                    event=event,
                    event_path=event_path,
                    level=level,
                    descriptor=descriptor,
                )
                if record is None:
                    continue
                dedupe_key = _local_dedupe_key(record)
                if dedupe_key in seen:
                    duplicate_count += 1
                    continue
                seen.add(dedupe_key)
                records.append(record)
    diagnostics = {
        "local_record_count": len(records),
        "local_duplicate_event_count": duplicate_count,
    }
    return records, diagnostics


def _index_local_records(
    records: Iterable[dict[str, Any]],
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    indexed: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = record.get("call_key")
        if isinstance(key, CallKey):
            indexed[key.as_tuple()].append(record)
    return indexed


def _select_local_match(
    record: dict[str, Any],
    candidates: list[dict[str, Any]],
    used_local_ids: set[tuple[str, int]],
) -> dict[str, Any] | None:
    if not candidates:
        return None
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    export_level = _normalize_level(metadata.get("level"))
    trace_id = str(record.get("trace_id") or "")
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for index, candidate in enumerate(candidates):
        local_id = (
            str(candidate.get("source_event_path") or ""),
            int(candidate.get("source_event_line") or 0),
        )
        if local_id in used_local_ids:
            continue
        score = 0
        if export_level and export_level == str(candidate.get("level") or ""):
            score += 100
        if trace_id and trace_id == str(candidate.get("trace_id") or ""):
            score += 50
        scored.append((score, -index, candidate))
    if not scored:
        return None
    return max(scored, key=lambda item: (item[0], item[1]))[2]


def _base_row_from_record(
    record: dict[str, Any],
    *,
    split_lookup: dict[str, str],
    local: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    actor = str(metadata.get("actor") or record.get("actor") or "")
    level = _normalize_level(metadata.get("level"))
    task_id = str(metadata.get("task_id") or "")
    if local is not None:
        actor = str(local.get("actor") or actor)
        level = str(local.get("level") or level or "")
        task_id = str(local.get("task_id") or task_id)
    step_index = _coerce_int(metadata.get("step_index")) or _step_from_name(
        str(record.get("name") or "")
    )
    if local is not None:
        step_index = _coerce_int(local.get("step_index")) or step_index
    run_id = _coerce_int(metadata.get("run_id"))
    if local is not None and local.get("run_id") is not None:
        run_id = int(local["run_id"])
    hook = str(metadata.get("hook") or _hook_from_name(str(record.get("name") or "")))
    trigger_type = str(metadata.get("trigger_type") or "")
    if local is not None:
        hook = str(local.get("hook") or hook)
        trigger_type = str(local.get("trigger_type") or trigger_type)
    return {
        "source": "langfuse",
        "match_source": "langfuse_observation+local_join"
        if local is not None
        else "langfuse_observation",
        "experiment_name": record.get("experiment_name"),
        "system": record.get("system"),
        "session_id": record.get("session_id"),
        "source_session_root": record.get("source_session_root"),
        "executor_model": record.get("executor_model"),
        "overseer_model": record.get("overseer_model"),
        "trace_id": record.get("trace_id"),
        "observation_id": record.get("observation_id"),
        "observation_name": record.get("name"),
        "observation_type": record.get("type"),
        "started_at": record.get("started_at"),
        "ended_at": record.get("ended_at"),
        "role": actor,
        "model": record.get("model"),
        "domain": str(metadata.get("domain") or (local or {}).get("domain") or ""),
        "level": int(level) if str(level).isdigit() else None,
        "task_id": task_id,
        "case_name": f"case_{task_id}" if task_id else "",
        "run_id": run_id,
        "phase": str(metadata.get("phase") or (local or {}).get("phase") or ""),
        "step_index": step_index,
        "tool_index": _coerce_int(metadata.get("tool_index"))
        if local is None
        else local.get("tool_index"),
        "hook": hook,
        "trigger_type": trigger_type,
        "split": _case_split(split_lookup, level, task_id),
        "source_event_path": (local or {}).get("source_event_path"),
        "source_event_line": (local or {}).get("source_event_line"),
    }


def _token_row_from_observation(
    record: dict[str, Any],
    *,
    split_lookup: dict[str, str],
    local: dict[str, Any] | None,
) -> dict[str, Any]:
    usage = record.get("usage") if isinstance(record.get("usage"), dict) else {}
    counts = token_counts_from_usage(usage)
    row = _base_row_from_record(record, split_lookup=split_lookup, local=local)
    row.update(counts)
    row["input_cache_split"] = _cache_split_status(usage, counts)
    return row


def _token_row_from_local(
    record: dict[str, Any],
    *,
    split_lookup: dict[str, str],
) -> dict[str, Any]:
    event = record.get("event") if isinstance(record.get("event"), dict) else {}
    usage = _local_usage(record)
    if record.get("actor") == "executor":
        counts = token_counts_from_usage(
            usage,
            fallback_input_total=_coerce_int(event.get("prompt_tokens")),
            fallback_output_total=_coerce_int(event.get("completion_tokens")),
            assume_uncached_when_cache_missing=True,
        )
    else:
        counts = token_counts_from_usage(
            usage,
            fallback_input_total=_coerce_int(event.get("overseer_input_tokens")),
            fallback_output_total=_coerce_int(event.get("overseer_output_tokens")),
            assume_uncached_when_cache_missing=False,
        )
    level = record.get("level")
    task_id = str(record.get("task_id") or "")
    row = {
        "source": "local_agent_log",
        "match_source": "local_agent_log_unmatched",
        "experiment_name": record.get("experiment_name"),
        "system": record.get("system"),
        "session_id": record.get("session_id"),
        "source_session_root": record.get("source_session_root"),
        "executor_model": record.get("executor_model"),
        "overseer_model": record.get("overseer_model"),
        "trace_id": record.get("trace_id"),
        "observation_id": "",
        "observation_name": "",
        "observation_type": "",
        "started_at": event.get("started_at") or event.get("timestamp"),
        "ended_at": event.get("ended_at"),
        "role": record.get("actor"),
        "model": record.get("model"),
        "domain": record.get("domain"),
        "level": int(level) if str(level).isdigit() else None,
        "task_id": task_id,
        "case_name": record.get("case_name"),
        "run_id": record.get("run_id"),
        "phase": record.get("phase"),
        "step_index": record.get("step_index"),
        "tool_index": record.get("tool_index"),
        "hook": record.get("hook"),
        "trigger_type": record.get("trigger_type"),
        "split": _case_split(split_lookup, level, task_id),
        "source_event_path": record.get("source_event_path"),
        "source_event_line": record.get("source_event_line"),
    }
    row.update(counts)
    row["input_cache_split"] = _cache_split_status(usage, counts)
    return row


def _row_has_tokens(row: dict[str, Any]) -> bool:
    return any(row.get(column) not in (None, 0, "") for column in TOKEN_COLUMNS)


def collapsed_trace_groups(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        trace_id = str(row.get("trace_id") or "")
        session_id = str(row.get("session_id") or "")
        if trace_id and session_id:
            grouped[(session_id, trace_id)].append(row)

    collisions: list[dict[str, Any]] = []
    for (session_id, trace_id), records in grouped.items():
        case_keys = sorted(
            {
                f"level_{record.get('level')}:{record.get('task_id')}"
                for record in records
                if record.get("level") not in (None, "")
                and record.get("task_id") not in (None, "")
            }
        )
        if len(case_keys) <= 1:
            continue
        collisions.append(
            {
                "session_id": session_id,
                "trace_id": trace_id,
                "distinct_case_count": len(case_keys),
                "observation_count": len(records),
                "systems": sorted({str(record.get("system") or "") for record in records}),
                "roles": sorted({str(record.get("role") or "") for record in records}),
                "case_keys": case_keys[:50],
                "source_observation_ids": [
                    record.get("observation_id") for record in records[:50]
                ],
            }
        )
    return sorted(
        collisions,
        key=lambda item: (item["session_id"], -int(item["distinct_case_count"])),
    )


def duplicate_observation_key_groups(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = _observation_call_key(record)
        if key is not None:
            grouped[key.as_tuple()].append(record)
    return [
        CallKey(*key_tuple).as_dict()
        | {
            "record_count": len(items),
            "source_observation_ids": [item.get("observation_id") for item in items[:50]],
        }
        for key_tuple, items in grouped.items()
        if len(items) > 1
    ]


def build_token_rows(
    observation_records: list[dict[str, Any]],
    local_records: list[dict[str, Any]],
    *,
    split_lookup: dict[str, str],
    include_local_fallback: bool = True,
    include_setup: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    local_by_key = _index_local_records(local_records)
    rows: list[dict[str, Any]] = []
    skipped_setup = 0
    skipped_no_key = 0
    skipped_no_tokens = 0
    matched_local = 0
    used_local_ids: set[tuple[str, int]] = set()

    for record in observation_records:
        if not include_setup and _is_setup_observation(record):
            skipped_setup += 1
            continue
        key = _observation_call_key(record)
        local = None
        if key is not None:
            local = _select_local_match(
                record,
                local_by_key.get(key.as_tuple(), []),
                used_local_ids,
            )
        else:
            skipped_no_key += 1
        if local is not None:
            matched_local += 1
            used_local_ids.add(
                (
                    str(local.get("source_event_path") or ""),
                    int(local.get("source_event_line") or 0),
                )
            )
        row = _token_row_from_observation(
            record,
            split_lookup=split_lookup,
            local=local,
        )
        if not _row_has_tokens(row):
            skipped_no_tokens += 1
            continue
        rows.append(row)

    local_fallback_count = 0
    if include_local_fallback:
        for local in local_records:
            local_id = (
                str(local.get("source_event_path") or ""),
                int(local.get("source_event_line") or 0),
            )
            if local_id in used_local_ids:
                continue
            row = _token_row_from_local(local, split_lookup=split_lookup)
            if not _row_has_tokens(row):
                continue
            rows.append(row)
            local_fallback_count += 1

    diagnostics = {
        "counts": {
            "langfuse_observation_records": len(observation_records),
            "local_records": len(local_records),
            "token_rows": len(rows),
            "matched_local_records": matched_local,
            "local_fallback_rows": local_fallback_count,
            "skipped_setup_observations": skipped_setup,
            "skipped_no_key_observations": skipped_no_key,
            "skipped_no_token_observations": skipped_no_tokens,
            "duplicate_observation_key_groups": len(
                duplicate_observation_key_groups(observation_records)
            ),
            "collapsed_trace_groups": len(collapsed_trace_groups(rows)),
        },
        "duplicate_observation_key_groups": duplicate_observation_key_groups(
            observation_records
        )[:1000],
        "collapsed_trace_groups": collapsed_trace_groups(rows)[:1000],
    }
    return rows, diagnostics


def summarize_token_rows(rows: list[dict[str, Any]], group_cols: list[str]) -> pd.DataFrame:
    columns = group_cols + [
        "observation_count",
        "case_count",
        "input_total_tokens",
        "input_uncached_tokens",
        "input_cached_tokens",
        "output_total_tokens",
        "output_text_tokens",
        "output_reasoning_tokens",
        "total_tokens",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    for column in TOKEN_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    level_values = frame.get("level", pd.Series("", index=frame.index)).fillna("")
    case_values = frame.get("case_name", pd.Series("", index=frame.index)).fillna("")
    frame["_level_case_key"] = [
        f"{level}:{case}" if str(level) and str(case) else ""
        for level, case in zip(level_values, case_values, strict=False)
    ]
    available_group_cols = [column for column in group_cols if column in frame.columns]
    summary = (
        frame.groupby(available_group_cols, dropna=False)
        .agg(
            observation_count=("source", "size"),
            case_count=(
                "_level_case_key",
                lambda values: values.replace("", pd.NA).nunique(),
            ),
            input_total_tokens=("input_total_tokens", "sum"),
            input_uncached_tokens=("input_uncached_tokens", "sum"),
            input_cached_tokens=("input_cached_tokens", "sum"),
            output_total_tokens=("output_total_tokens", "sum"),
            output_text_tokens=("output_text_tokens", "sum"),
            output_reasoning_tokens=("output_reasoning_tokens", "sum"),
            total_tokens=("total_tokens", "sum"),
        )
        .reset_index()
        .sort_values(available_group_cols, kind="stable")
        .reset_index(drop=True)
    )
    return summary


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_outputs(
    *,
    output_dir: Path,
    observation_records: list[dict[str, Any]],
    token_rows: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    manifest: dict[str, Any],
    write_raw_observations: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if write_raw_observations:
        write_jsonl(output_dir / "langfuse_observations.jsonl", observation_records)
    token_frame = pd.DataFrame(token_rows)
    token_frame.to_csv(output_dir / "token_observations.csv", index=False)
    held_out = (
        token_frame[token_frame["split"] == "hold_out"].copy()
        if not token_frame.empty and "split" in token_frame.columns
        else token_frame.copy()
    )
    held_out.to_csv(output_dir / "held_out_token_observations.csv", index=False)
    summarize_token_rows(
        token_rows,
        ["system", "experiment_name", "session_id", "split", "role", "model"],
    ).to_csv(output_dir / "token_summary.csv", index=False)
    summarize_token_rows(
        held_out.to_dict("records"),
        ["system", "experiment_name", "session_id", "split", "role", "model"],
    ).to_csv(output_dir / "held_out_token_summary.csv", index=False)
    summarize_token_rows(
        token_rows,
        ["system", "session_id", "run_id", "level", "case_name", "split", "role"],
    ).to_csv(output_dir / "token_summary_by_case.csv", index=False)
    summarize_token_rows(
        held_out.to_dict("records"),
        ["system", "session_id", "run_id", "level", "case_name", "split", "role"],
    ).to_csv(output_dir / "held_out_token_summary_by_case.csv", index=False)
    (output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def export_trace_tokens(
    *,
    session_roots: list[Path],
    output_dir: Path,
    split_path: Path,
    host: str,
    public_key: str,
    secret_key: str,
    fields: str = DEFAULT_FIELDS,
    limit: int = 1000,
    timeout: float = 30.0,
    max_records_per_session: int | None = None,
    include_local_fallback: bool = False,
    include_setup: bool = False,
    write_raw_observations: bool = False,
    continue_on_error: bool = False,
    progress: bool = False,
) -> dict[str, Any]:
    split_lookup = load_split_lookup(split_path)
    observation_records: list[dict[str, Any]] = []
    session_entries: list[dict[str, Any]] = []
    exported_at = _utc_now()

    for session_root in tqdm(
        session_roots,
        desc="fetch sessions",
        unit="session",
        dynamic_ncols=True,
        disable=not progress,
    ):
        descriptor = session_descriptor(session_root)
        try:
            observations = fetch_observations(
                session_id=str(descriptor["langfuse_session_id"]),
                host=host,
                public_key=public_key,
                secret_key=secret_key,
                fields=fields,
                limit=limit,
                timeout=timeout,
                max_records=max_records_per_session,
                progress=progress,
            )
            normalized = [
                normalize_observation(observation, descriptor=descriptor)
                for observation in observations
            ]
            observation_records.extend(normalized)
            error = ""
        except Exception as exc:
            if not continue_on_error:
                raise
            observations = []
            normalized = []
            error = str(exc)
        session_entries.append(
            {
                **descriptor,
                "raw_observation_count": len(observations),
                "normalized_observation_count": len(normalized),
                "langfuse_error": error,
            }
        )

    if include_local_fallback:
        local_records, local_diagnostics = load_local_call_records(
            session_roots,
            progress=progress,
        )
    else:
        local_records = []
        local_diagnostics = {
            "local_record_count": 0,
            "local_duplicate_event_count": 0,
        }
    token_rows, diagnostics = build_token_rows(
        observation_records,
        local_records,
        split_lookup=split_lookup,
        include_local_fallback=include_local_fallback,
        include_setup=include_setup,
    )
    diagnostics["local"] = local_diagnostics
    manifest = {
        "exported_at": exported_at,
        "host": host,
        "fields": fields,
        "output_dir": str(output_dir),
        "split_path": str(split_path),
        "filters": {
            "include_setup": include_setup,
            "include_local_fallback": include_local_fallback,
            "write_raw_observations": write_raw_observations,
        },
        "git": _git_state(),
        "sessions": session_entries,
        "counts": diagnostics["counts"],
    }
    write_outputs(
        output_dir=output_dir,
        observation_records=observation_records,
        token_rows=token_rows,
        diagnostics=diagnostics,
        manifest=manifest,
        write_raw_observations=write_raw_observations,
    )
    return manifest


def _coerce_path_values(values: list[str] | None) -> list[Path]:
    return [Path(value) for value in values or []]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export observation-level Langfuse token usage and join it to local "
            "agent logs so collapsed trace ids do not collapse per-case usage."
        )
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=DEFAULT_EXPERIMENTS_ROOT,
        help=f"Experiment output root. Defaults to {DEFAULT_EXPERIMENTS_ROOT}.",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        dest="experiments",
        help=(
            "Exact experiment directory name to include. Repeatable. Defaults to "
            "the six held-out LoRA analysis systems."
        ),
    )
    parser.add_argument(
        "--session-root",
        action="append",
        dest="session_roots",
        help="Explicit session root to include. Repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=DEFAULT_SPLIT_PATH,
        help=f"Shopping split config. Defaults to {DEFAULT_SPLIT_PATH}.",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("LANGFUSE_HOST", DEFAULT_LANGFUSE_HOST),
        help=f"Langfuse host. Defaults to LANGFUSE_HOST or {DEFAULT_LANGFUSE_HOST}.",
    )
    parser.add_argument(
        "--public-key",
        default=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
        help="Langfuse public key. Defaults to LANGFUSE_PUBLIC_KEY.",
    )
    parser.add_argument(
        "--secret-key",
        default=os.getenv("LANGFUSE_SECRET_KEY", ""),
        help="Langfuse secret key. Defaults to LANGFUSE_SECRET_KEY.",
    )
    parser.add_argument(
        "--fields",
        default=DEFAULT_FIELDS,
        help=f"Langfuse observation fields. Defaults to {DEFAULT_FIELDS}.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Observations per Langfuse page. Must be 1-1000.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--max-records-per-session",
        type=int,
        default=None,
        help="Optional debug cap on fetched observations per session.",
    )
    parser.add_argument(
        "--join-local-logs",
        action="store_true",
        help=(
            "Scan agent_events.jsonl and join by metadata key. This can be slow "
            "on full baseline sessions; off by default because Langfuse metadata "
            "already carries level/task/run ids."
        ),
    )
    parser.add_argument(
        "--write-raw-observations",
        action="store_true",
        help=(
            "Also write langfuse_observations.jsonl. Off by default because raw "
            "Langfuse inputs can be several GB."
        ),
    )
    parser.add_argument(
        "--include-setup",
        action="store_true",
        help="Include overseer setup/compile observations. Defaults to runtime only.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep writing local fallback rows if a Langfuse session fetch fails.",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="Print selected sessions and exit.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)
    explicit_session_roots = _coerce_path_values(args.session_roots)
    session_roots = explicit_session_roots or discover_session_roots(
        args.experiments_root,
        experiment_names=args.experiments or DEFAULT_EXPERIMENTS,
    )
    if args.list_sessions:
        print(
            json.dumps(
                [session_descriptor(session_root) for session_root in session_roots],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return
    if not session_roots:
        parser.exit(1, "error: no matching experiment sessions found\n")

    try:
        manifest = export_trace_tokens(
            session_roots=session_roots,
            output_dir=args.output_dir,
            split_path=args.split_path,
            host=args.host,
            public_key=args.public_key,
            secret_key=args.secret_key,
            fields=args.fields,
            limit=args.limit,
            timeout=args.timeout,
            max_records_per_session=args.max_records_per_session,
            include_local_fallback=args.join_local_logs,
            include_setup=args.include_setup,
            write_raw_observations=args.write_raw_observations,
            continue_on_error=args.continue_on_error,
            progress=args.progress,
        )
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")

    print(
        "Wrote "
        f"{manifest['counts']['token_rows']} token rows for "
        f"{len(manifest['sessions'])} sessions to {args.output_dir}"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
