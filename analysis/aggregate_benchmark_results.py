from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "deepplanning-matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_OUTPUT_ROOT = Path("outputs/benchmark_report")
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260429
SYSTEM_ORDER = ("A", "C2-noretry", "C2", "B", "C1", "C2-nt")
FIGURE_FILENAMES = {
    "forest": "system_comparison_forest.png",
    "pareto": "cost_accuracy_pareto.png",
    "per_level": "per_level_case_accuracy.png",
    "retry": "retry_decomposition.png",
    "trigger": "trigger_mix.png",
    "infra": "infra_failure_heatmap.png",
}


@dataclass(frozen=True)
class ReportArtifacts:
    output_root: Path
    summary_csv: Path
    summary_json: Path
    report_md: Path
    figures: dict[str, Path]


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            yield payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _as_path_list(paths: Sequence[str | Path]) -> list[Path]:
    return [Path(path) for path in paths]


def _discover_files(paths: Sequence[Path], filename: str) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob(filename)))
        elif path.name == filename:
            files.append(path)
    return sorted(dict.fromkeys(files))


def _find_session_root(path: Path) -> Path:
    for parent in [path.parent, *path.parents]:
        if (parent / "experiment_session.json").exists():
            return parent
    return path.parent


def _load_metadata(session_root: Path) -> dict[str, Any]:
    metadata_path = session_root / "experiment_session.json"
    if not metadata_path.exists():
        return {}
    return _load_json(metadata_path)


def _parse_run_id(path: Path) -> int | None:
    for part in reversed(path.parts):
        if part.startswith("run_") and part.removeprefix("run_").isdigit():
            return int(part.removeprefix("run_"))
    return None


def _parse_level(path: Path) -> int | None:
    for part in reversed(path.parts):
        match = re.search(r"level[_-]?(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def _task_id_from_case_name(value: Any) -> str:
    text = str(value or "").strip()
    return text.removeprefix("case_")


def _metadata_value(
    metadata: dict[str, Any],
    path: Sequence[str],
    default: Any = None,
) -> Any:
    value: Any = metadata
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _experiment_name(metadata: dict[str, Any], session_root: Path) -> str:
    return str(
        _metadata_value(metadata, ("experiment", "name"))
        or _metadata_value(metadata, ("parameters", "name"))
        or session_root.parent.name
    )


def _system_name(record: dict[str, Any], metadata: dict[str, Any]) -> str:
    return str(
        record.get("system")
        or record.get("system_name")
        or _metadata_value(metadata, ("parameters", "system", "name"))
        or "unknown"
    )


def _split_name(record: dict[str, Any], metadata: dict[str, Any]) -> str:
    return str(
        record.get("split")
        or _metadata_value(metadata, ("parameters", "shopping", "split"))
        or "unknown"
    )


def _system_config_field(metadata: dict[str, Any], key: str) -> Any:
    return _metadata_value(metadata, ("parameters", "system", key))


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if pd.isna(value) if not isinstance(value, (dict, list, tuple, str)) else False:
        return None
    return value


def _counter_from_value(value: Any) -> Counter[str]:
    counter: Counter[str] = Counter()
    if isinstance(value, dict):
        for key, count in value.items():
            try:
                counter[str(key)] += int(count or 0)
            except (TypeError, ValueError):
                counter[str(key)] += 1
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                trigger = item.get("trigger_type") or item.get("type")
                if trigger:
                    counter[str(trigger)] += 1
            elif item is not None:
                counter[str(item)] += 1
    return counter


def _merge_counters(values: Iterable[Any]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for value in values:
        counter.update(_counter_from_value(value))
    return dict(sorted(counter.items()))


def _column(frame: pd.DataFrame, name: str, default: Any = None) -> pd.Series:
    if name in frame:
        return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def load_task_result_records(paths: Sequence[str | Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result_path in _discover_files(_as_path_list(paths), "task_results.jsonl"):
        session_root = _find_session_root(result_path)
        metadata = _load_metadata(session_root)
        run_id_from_path = _parse_run_id(result_path)
        level_from_path = _parse_level(result_path)
        for record in _iter_jsonl(result_path):
            run_id = _coerce_int(record.get("run_id"))
            level = _coerce_int(record.get("level") or record.get("complexity"))
            rows.append(
                {
                    **record,
                    "source_path": str(result_path),
                    "session_root": str(session_root),
                    "experiment_name": _experiment_name(metadata, session_root),
                    "system": _system_name(record, metadata),
                    "split": _split_name(record, metadata),
                    "run_id": run_id if run_id is not None else run_id_from_path,
                    "level": level if level is not None else level_from_path,
                    "task_id": str(record.get("task_id") or record.get("id") or ""),
                    "loop_similarity_threshold": record.get(
                        "loop_similarity_threshold",
                        _system_config_field(metadata, "loop_similarity_threshold"),
                    ),
                    "loop_window": record.get(
                        "loop_window",
                        _system_config_field(metadata, "loop_window"),
                    ),
                    "loop_repeat_count": record.get(
                        "loop_repeat_count",
                        _system_config_field(metadata, "loop_repeat_count"),
                    ),
                    "coverage_threshold": record.get(
                        "coverage_threshold",
                        _system_config_field(metadata, "coverage_threshold"),
                    ),
                    "final_repair_retry_cap": record.get(
                        "final_repair_retry_cap",
                        _system_config_field(metadata, "final_repair_retry_cap"),
                    ),
                    "overseer_prompt_version": record.get(
                        "overseer_prompt_version",
                        _system_config_field(metadata, "overseer_prompt_version"),
                    ),
                }
            )
    return pd.DataFrame(rows)


def load_evaluation_records(paths: Sequence[str | Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary_path in _discover_files(_as_path_list(paths), "summary_report.json"):
        session_root = _find_session_root(summary_path)
        metadata = _load_metadata(session_root)
        run_id = _parse_run_id(summary_path)
        level = _parse_level(summary_path)
        payload = _load_json(summary_path)
        case_results = payload.get("case_results") or []
        if not isinstance(case_results, list):
            continue
        for case in case_results:
            if not isinstance(case, dict):
                continue
            case_score = _coerce_float(case.get("case_score"))
            success = _coerce_bool(case.get("success"))
            case_accuracy = case_score
            if case_accuracy is None and success is not None:
                case_accuracy = 1.0 if success else 0.0
            rows.append(
                {
                    "session_root": str(session_root),
                    "experiment_name": _experiment_name(metadata, session_root),
                    "system": _system_name(case, metadata),
                    "split": _split_name(case, metadata),
                    "run_id": run_id,
                    "level": level,
                    "task_id": _task_id_from_case_name(case.get("case_name")),
                    "eval_case_accuracy": case_accuracy,
                    "eval_match_score": _coerce_float(
                        case.get("match_score", case.get("score"))
                    ),
                    "eval_success": success,
                    "eval_source_path": str(summary_path),
                }
            )
    return pd.DataFrame(rows)


def load_agent_event_counts(paths: Sequence[str | Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for events_path in _discover_files(_as_path_list(paths), "agent_events.jsonl"):
        session_root = _find_session_root(events_path)
        metadata = _load_metadata(session_root)
        level_from_path = _parse_level(events_path)
        counters: dict[tuple[str, int | None, int | None, str], Counter[str]] = {}
        for record in _iter_jsonl(events_path):
            trigger = record.get("trigger_type")
            if not trigger:
                continue
            if record.get("event_type") == "oversight_step" and not record.get(
                "overseer_invoked", True
            ):
                continue
            run_id = _coerce_int(record.get("run_id"))
            level = _coerce_int(record.get("level") or record.get("complexity"))
            task_id = str(record.get("task_id") or "")
            key = (
                str(session_root),
                run_id,
                level if level is not None else level_from_path,
                task_id,
            )
            counters.setdefault(key, Counter())[str(trigger)] += 1
        for (session, run_id, level, task_id), counter in counters.items():
            rows.append(
                {
                    "session_root": session,
                    "experiment_name": _experiment_name(metadata, session_root),
                    "run_id": run_id,
                    "level": level,
                    "task_id": task_id,
                    "event_trigger_counts": dict(sorted(counter.items())),
                }
            )
    return pd.DataFrame(rows)


def load_benchmark_records(paths: Sequence[str | Path]) -> pd.DataFrame:
    task_df = load_task_result_records(paths)
    eval_df = load_evaluation_records(paths)
    event_df = load_agent_event_counts(paths)
    keys = ["session_root", "run_id", "level", "task_id"]

    if task_df.empty and eval_df.empty:
        return pd.DataFrame()
    if task_df.empty:
        merged = eval_df.copy()
    elif eval_df.empty:
        merged = task_df.copy()
    else:
        merged = task_df.merge(
            eval_df[keys + ["eval_case_accuracy", "eval_match_score", "eval_success"]],
            on=keys,
            how="outer",
        )

    if not event_df.empty:
        merged = merged.merge(
            event_df[keys + ["event_trigger_counts"]], on=keys, how="left"
        )

    for column in ("system", "split", "experiment_name"):
        fallback = f"{column}_y"
        primary = f"{column}_x"
        if primary in merged or fallback in merged:
            merged[column] = _column(merged, primary).combine_first(
                _column(merged, column)
            )
            merged[column] = merged[column].combine_first(_column(merged, fallback))

    explicit_case_accuracy = pd.to_numeric(
        _column(
            merged,
            "case_accuracy",
        ).combine_first(_column(merged, "case_score")),
        errors="coerce",
    )
    merged["case_accuracy"] = pd.to_numeric(
        _column(merged, "eval_case_accuracy"),
        errors="coerce",
    ).combine_first(explicit_case_accuracy)
    merged["match_score"] = pd.to_numeric(
        _column(merged, "eval_match_score"),
        errors="coerce",
    ).combine_first(
        pd.to_numeric(
            _column(merged, "match_score").combine_first(_column(merged, "score")),
            errors="coerce",
        )
    )
    merged["total_cost_usd"] = pd.to_numeric(
        _column(merged, "total_cost_usd"), errors="coerce"
    )
    overseer_col = (
        "overseer_invocation_count"
        if "overseer_invocation_count" in merged
        else "overseer_calls"
    )
    merged["overseer_invocation_count"] = pd.to_numeric(
        _column(merged, overseer_col), errors="coerce"
    )
    merged["final_verification_retry_count"] = pd.to_numeric(
        _column(merged, "final_verification_retry_count"), errors="coerce"
    )
    merged["observation_valid"] = _column(merged, "observation_valid", True)
    merged["failure_subtype"] = _column(merged, "failure_subtype", "none")
    merged["task_unit"] = (
        merged["level"].astype("Int64").astype(str)
        + ":"
        + merged["task_id"].astype(str)
    )
    merged["trigger_counts"] = [
        _merge_counters(
            [
                row.get("overseer_invocation_count_by_trigger"),
                row.get("event_trigger_counts"),
                row.get("triggers_fired"),
            ]
        )
        for row in merged.to_dict(orient="records")
    ]
    return merged


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float | None, float | None, float | None]:
    arr = np.asarray([value for value in values if pd.notna(value)], dtype=float)
    if arr.size == 0:
        return (None, None, None)
    if arr.size == 1 or resamples <= 0:
        value = float(arr.mean())
        return (value, value, value)
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(int(resamples), arr.size), replace=True).mean(axis=1)
    return (
        float(arr.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def _system_sort_key(system: str) -> tuple[int, str]:
    try:
        return (SYSTEM_ORDER.index(system), system)
    except ValueError:
        return (len(SYSTEM_ORDER), system)


def _retry_distribution(frame: pd.DataFrame) -> dict[str, int]:
    retry_counts = frame["final_verification_retry_count"].dropna().astype(int)
    return {
        str(key): int(value)
        for key, value in retry_counts.value_counts().sort_index().items()
    }


def _infra_mask(frame: pd.DataFrame) -> pd.Series:
    invalid = frame["observation_valid"].map(_coerce_bool).eq(False)
    infra_subtype = frame["failure_subtype"].astype(str).str.contains("infra", na=False)
    return invalid | infra_subtype


def _task_level_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (system, split, level, task_id), group in frame.groupby(
        ["system", "split", "level", "task_id"], dropna=False
    ):
        retry_known = group["final_verification_retry_count"].notna().any()
        case_accuracy = pd.to_numeric(group["case_accuracy"], errors="coerce")
        retry_counts = pd.to_numeric(
            group["final_verification_retry_count"], errors="coerce"
        )
        if retry_known:
            first_attempt_values = case_accuracy.where(
                retry_counts.fillna(0).eq(0), 0.0
            )
            recovered_values = case_accuracy.where(retry_counts.fillna(0).gt(0), 0.0)
        else:
            first_attempt_values = pd.Series(dtype=float)
            recovered_values = pd.Series(dtype=float)
        rows.append(
            {
                "system": system,
                "split": split,
                "level": level,
                "task_id": task_id,
                "case_accuracy": case_accuracy.mean(),
                "match_score": pd.to_numeric(
                    group["match_score"], errors="coerce"
                ).mean(),
                "total_cost_usd": pd.to_numeric(
                    group["total_cost_usd"], errors="coerce"
                ).mean(),
                "overseer_invocation_count": pd.to_numeric(
                    group["overseer_invocation_count"], errors="coerce"
                ).mean(),
                "overseer_invoked": bool(
                    pd.to_numeric(group["overseer_invocation_count"], errors="coerce")
                    .fillna(0)
                    .gt(0)
                    .any()
                ),
                "first_attempt_case_accuracy": (
                    first_attempt_values.mean() if retry_known else np.nan
                ),
                "recovered_after_retry_accuracy": (
                    recovered_values.mean() if retry_known else np.nan
                ),
                "infra_failure": bool(_infra_mask(group).any()),
            }
        )
    return pd.DataFrame(rows)


def build_per_level_breakdown(task_frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if task_frame.empty:
        return rows
    for (system, split, level), group in task_frame.groupby(
        ["system", "split", "level"], dropna=False
    ):
        rows.append(
            {
                "system": system,
                "split": split,
                "level": f"L{int(level)}" if pd.notna(level) else "unknown",
                "task_count": int(len(group)),
                "case_accuracy": _coerce_float(group["case_accuracy"].mean()),
                "match_score": _coerce_float(group["match_score"].mean()),
                "infra_failure_count": int(group["infra_failure"].sum()),
                "infra_failure_rate": _coerce_float(group["infra_failure"].mean()),
            }
        )
    rows.sort(key=lambda row: (_system_sort_key(str(row["system"])), str(row["level"])))
    return rows


def build_summary(
    records: pd.DataFrame,
    *,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if records.empty:
        return ([], [])

    task_frame = _task_level_frame(records)
    per_level = build_per_level_breakdown(task_frame)
    summary_rows: list[dict[str, Any]] = []
    for (system, split), group in records.groupby(["system", "split"], dropna=False):
        task_group = task_frame[
            task_frame["system"].eq(system) & task_frame["split"].eq(split)
        ]
        case_mean, case_low, case_high = bootstrap_mean_ci(
            task_group["case_accuracy"].tolist(),
            resamples=bootstrap_resamples,
            seed=seed,
        )
        match_mean, match_low, match_high = bootstrap_mean_ci(
            task_group["match_score"].tolist(),
            resamples=bootstrap_resamples,
            seed=seed + 1,
        )
        trigger_counts = _merge_counters(group["trigger_counts"].tolist())
        trigger_total = sum(trigger_counts.values())
        retry_known = group["final_verification_retry_count"].notna().any()
        with_retry = _coerce_float(task_group["case_accuracy"].mean())
        retry_cap_values = pd.to_numeric(
            _column(group, "final_repair_retry_cap"), errors="coerce"
        ).dropna()
        no_final_repair_retry = str(system) == "C2-noretry" or (
            not retry_cap_values.empty and retry_cap_values.eq(0).all()
        )
        if no_final_repair_retry:
            first_attempt = with_retry
            recovered = 0.0 if with_retry is not None else None
            retry_added = 0.0 if with_retry is not None else None
            retry_basis = "no_final_repair_retry"
        else:
            first_attempt = _coerce_float(
                task_group["first_attempt_case_accuracy"].mean()
            )
            recovered = _coerce_float(
                task_group["recovered_after_retry_accuracy"].mean()
            )
            retry_added = (
                _coerce_float((with_retry or 0.0) - (first_attempt or 0.0))
                if retry_known and with_retry is not None and first_attempt is not None
                else None
            )
            retry_basis = (
                "proxy_from_final_verification_retry_count"
                if retry_known
                else "unavailable"
            )
        summary_rows.append(
            {
                "system": system,
                "split": split,
                "task_count": int(
                    task_group[["level", "task_id"]].drop_duplicates().shape[0]
                ),
                "run_count": int(
                    group[["session_root", "run_id"]].drop_duplicates().shape[0]
                ),
                "case_accuracy": case_mean,
                "case_accuracy_ci_low": case_low,
                "case_accuracy_ci_high": case_high,
                "match_score": match_mean,
                "match_score_ci_low": match_low,
                "match_score_ci_high": match_high,
                "mean_total_cost_usd": _coerce_float(
                    task_group["total_cost_usd"].mean()
                ),
                "mean_overseer_invocation_count": _coerce_float(
                    task_group["overseer_invocation_count"].mean()
                ),
                "overseer_invocation_rate": _coerce_float(
                    task_group["overseer_invoked"].mean()
                ),
                "final_verification_retry_distribution": _retry_distribution(group),
                "first_attempt_metric_basis": retry_basis,
                "first_attempt_case_accuracy": (
                    first_attempt if retry_known or no_final_repair_retry else None
                ),
                "with_retry_case_accuracy": with_retry,
                "retry_added_case_accuracy": retry_added,
                "recovered_after_retry_accuracy": (
                    recovered if retry_known or no_final_repair_retry else None
                ),
                "trigger_counts": trigger_counts,
                "trigger_rates": {
                    key: value / trigger_total if trigger_total else 0.0
                    for key, value in trigger_counts.items()
                },
                "infra_failure_count": int(_infra_mask(group).sum()),
                "infra_failure_rate": _coerce_float(_infra_mask(group).mean()),
            }
        )
    summary_rows.sort(
        key=lambda row: (_system_sort_key(str(row["system"])), row["split"])
    )
    return (summary_rows, per_level)


def pareto_frontier(summary_rows: Sequence[dict[str, Any]]) -> set[str]:
    candidates = [
        row
        for row in summary_rows
        if _coerce_float(row.get("case_accuracy")) is not None
        and _coerce_float(row.get("mean_total_cost_usd")) is not None
    ]
    frontier: set[str] = set()
    for row in candidates:
        accuracy = float(row["case_accuracy"])
        cost = float(row["mean_total_cost_usd"])
        dominated = False
        for other in candidates:
            if other is row:
                continue
            other_accuracy = float(other["case_accuracy"])
            other_cost = float(other["mean_total_cost_usd"])
            if (
                other_accuracy >= accuracy
                and other_cost <= cost
                and (other_accuracy > accuracy or other_cost < cost)
            ):
                dominated = True
                break
        if not dominated:
            frontier.add(str(row["system"]))
    return frontier


def _save_empty(path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, "No data", ha="center", va="center")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_forest(summary: list[dict[str, Any]], path: Path) -> None:
    rows = [row for row in summary if row.get("case_accuracy") is not None]
    if not rows:
        _save_empty(path, "System comparison")
        return
    labels = [str(row["system"]) for row in rows]
    y = np.arange(len(rows))
    values = np.array([float(row["case_accuracy"]) for row in rows])
    lows = np.array(
        [float(row.get("case_accuracy_ci_low") or row["case_accuracy"]) for row in rows]
    )
    highs = np.array(
        [
            float(row.get("case_accuracy_ci_high") or row["case_accuracy"])
            for row in rows
        ]
    )
    fig, ax = plt.subplots(figsize=(7, 0.7 * len(rows) + 2))
    ax.errorbar(values, y, xerr=[values - lows, highs - values], fmt="o", capsize=4)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Case Accuracy")
    ax.set_xlim(0, 1)
    ax.set_title("Shopping Held-out Case Accuracy")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_pareto(summary: list[dict[str, Any]], path: Path) -> None:
    rows = [
        row
        for row in summary
        if row.get("case_accuracy") is not None
        and row.get("mean_total_cost_usd") is not None
    ]
    if not rows:
        _save_empty(path, "Cost-accuracy Pareto")
        return
    frontier = pareto_frontier(rows)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for row in rows:
        system = str(row["system"])
        is_frontier = system in frontier
        ax.scatter(
            float(row["mean_total_cost_usd"]),
            float(row["case_accuracy"]),
            s=95 if is_frontier else 55,
            marker="D" if is_frontier else "o",
        )
        ax.annotate(
            system,
            (float(row["mean_total_cost_usd"]), float(row["case_accuracy"])),
            xytext=(5, 5),
            textcoords="offset points",
        )
    frontier_rows = sorted(
        [row for row in rows if str(row["system"]) in frontier],
        key=lambda row: float(row["mean_total_cost_usd"]),
    )
    if len(frontier_rows) > 1:
        ax.plot(
            [float(row["mean_total_cost_usd"]) for row in frontier_rows],
            [float(row["case_accuracy"]) for row in frontier_rows],
            linewidth=1,
            alpha=0.7,
        )
    ax.set_xlabel("Mean cost/task (USD)")
    ax.set_ylabel("Case Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Cost vs Case Accuracy")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_per_level(per_level: list[dict[str, Any]], path: Path) -> None:
    rows = [row for row in per_level if row.get("case_accuracy") is not None]
    if not rows:
        _save_empty(path, "Per-level Case Accuracy")
        return
    df = pd.DataFrame(rows)
    systems = sorted(df["system"].unique(), key=_system_sort_key)
    levels = sorted(df["level"].unique())
    x = np.arange(len(levels))
    width = 0.8 / max(len(systems), 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for index, system in enumerate(systems):
        values = []
        for level in levels:
            subset = df[df["system"].eq(system) & df["level"].eq(level)]
            values.append(
                float(subset["case_accuracy"].iloc[0]) if not subset.empty else 0.0
            )
        ax.bar(
            x + (index - (len(systems) - 1) / 2) * width, values, width, label=system
        )
    ax.set_xticks(x, levels)
    ax.set_ylabel("Case Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Case Accuracy by Shopping Level")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_retry(summary: list[dict[str, Any]], path: Path) -> None:
    rows = [
        row
        for row in summary
        if row.get("first_attempt_case_accuracy") is not None
        and row.get("with_retry_case_accuracy") is not None
    ]
    if not rows:
        _save_empty(path, "Retry decomposition")
        return
    labels = [str(row["system"]) for row in rows]
    first = np.array([float(row["first_attempt_case_accuracy"]) for row in rows])
    added = np.array(
        [float(row.get("retry_added_case_accuracy") or 0.0) for row in rows]
    )
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x, first, label="First-attempt success")
    ax.bar(x, added, bottom=first, label="Added after final-verifier retry")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Case Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("First-attempt vs With-retry Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_trigger(summary: list[dict[str, Any]], path: Path) -> None:
    trigger_names = sorted(
        {trigger for row in summary for trigger in row.get("trigger_counts", {})}
    )
    if not summary or not trigger_names:
        _save_empty(path, "Trigger mix")
        return
    labels = [str(row["system"]) for row in summary]
    x = np.arange(len(summary))
    bottoms = np.zeros(len(summary))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for trigger in trigger_names:
        values = np.array(
            [float(row.get("trigger_counts", {}).get(trigger, 0)) for row in summary]
        )
        ax.bar(x, values, bottom=bottoms, label=trigger)
        bottoms += values
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Overseer trigger count")
    ax.set_title("Trigger Mix")
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_infra(per_level: list[dict[str, Any]], path: Path) -> None:
    rows = [row for row in per_level if row.get("infra_failure_rate") is not None]
    if not rows:
        _save_empty(path, "Infra-failure heatmap")
        return
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(
        index="system",
        columns="level",
        values="infra_failure_rate",
        aggfunc="mean",
    )
    pivot = pivot.reindex(sorted(pivot.index, key=_system_sort_key))
    fig, ax = plt.subplots(figsize=(6.5, max(3, 0.5 * len(pivot.index) + 1.5)))
    image = ax.imshow(pivot.fillna(0).values, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), list(pivot.columns))
    ax.set_yticks(range(len(pivot.index)), list(pivot.index))
    ax.set_title("Infra-failure Rate")
    for row_index, system in enumerate(pivot.index):
        for col_index, level in enumerate(pivot.columns):
            value = pivot.loc[system, level]
            label = "NA" if pd.isna(value) else f"{value:.2f}"
            ax.text(
                col_index, row_index, label, ha="center", va="center", color="white"
            )
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_figures(
    summary_rows: list[dict[str, Any]],
    per_level_rows: list[dict[str, Any]],
    figure_root: Path,
) -> dict[str, Path]:
    figure_root.mkdir(parents=True, exist_ok=True)
    paths = {
        name: figure_root / filename for name, filename in FIGURE_FILENAMES.items()
    }
    _plot_forest(summary_rows, paths["forest"])
    _plot_pareto(summary_rows, paths["pareto"])
    _plot_per_level(per_level_rows, paths["per_level"])
    _plot_retry(summary_rows, paths["retry"])
    _plot_trigger(summary_rows, paths["trigger"])
    _plot_infra(per_level_rows, paths["infra"])
    return paths


def _csv_ready_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ready: list[dict[str, Any]] = []
    for row in rows:
        out: dict[str, Any] = {}
        for key, value in row.items():
            safe = _json_safe(value)
            if isinstance(safe, (dict, list)):
                out[key] = json.dumps(safe, sort_keys=True)
            else:
                out[key] = safe
        ready.append(out)
    return ready


def _write_report(
    path: Path,
    summary_rows: list[dict[str, Any]],
    per_level_rows: list[dict[str, Any]],
    figure_paths: dict[str, Path],
    *,
    bootstrap_resamples: int,
    seed: int,
) -> None:
    lines = [
        "# Shopping Held-out Benchmark Report",
        "",
        f"Bootstrap resamples: `{bootstrap_resamples}`",
        f"Bootstrap seed: `{seed}`",
        "",
        "First-attempt decomposition is exact for `C2-noretry`; for full `C2`, "
        "it uses a proxy derived from `final_verification_retry_count` when "
        "available. Otherwise fields are reported as unavailable.",
        "",
        "## Summary",
        "",
    ]
    if summary_rows:
        lines.extend(
            [
                "| System | Split | Tasks | Runs | Case Accuracy | 95% CI | Match Score | Mean Cost | Overseer Rate | Infra Rate |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in summary_rows:
            ci = (
                "NA"
                if row.get("case_accuracy_ci_low") is None
                else f"{row['case_accuracy_ci_low']:.3f}-{row['case_accuracy_ci_high']:.3f}"
            )
            lines.append(
                "| {system} | {split} | {tasks} | {runs} | {acc} | {ci} | {match} | {cost} | {overseer} | {infra} |".format(
                    system=row["system"],
                    split=row["split"],
                    tasks=row["task_count"],
                    runs=row["run_count"],
                    acc=(
                        "NA"
                        if row["case_accuracy"] is None
                        else f"{row['case_accuracy']:.3f}"
                    ),
                    ci=ci,
                    match=(
                        "NA"
                        if row["match_score"] is None
                        else f"{row['match_score']:.3f}"
                    ),
                    cost=(
                        "NA"
                        if row["mean_total_cost_usd"] is None
                        else f"{row['mean_total_cost_usd']:.4f}"
                    ),
                    overseer=(
                        "NA"
                        if row["overseer_invocation_rate"] is None
                        else f"{row['overseer_invocation_rate']:.3f}"
                    ),
                    infra=(
                        "NA"
                        if row["infra_failure_rate"] is None
                        else f"{row['infra_failure_rate']:.3f}"
                    ),
                )
            )
    else:
        lines.append("No benchmark records were found.")

    lines.extend(["", "## Figures", ""])
    for name, figure_path in figure_paths.items():
        lines.append(f"- `{name}`: `{figure_path}`")
    lines.extend(["", "## Per-level Breakdown", ""])
    if per_level_rows:
        lines.extend(
            [
                "| System | Split | Level | Tasks | Case Accuracy | Match Score | Infra Rate |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in per_level_rows:
            lines.append(
                "| {system} | {split} | {level} | {tasks} | {acc} | {match} | {infra} |".format(
                    system=row["system"],
                    split=row["split"],
                    level=row["level"],
                    tasks=row["task_count"],
                    acc=(
                        "NA"
                        if row["case_accuracy"] is None
                        else f"{row['case_accuracy']:.3f}"
                    ),
                    match=(
                        "NA"
                        if row["match_score"] is None
                        else f"{row['match_score']:.3f}"
                    ),
                    infra=(
                        "NA"
                        if row["infra_failure_rate"] is None
                        else f"{row['infra_failure_rate']:.3f}"
                    ),
                )
            )
    else:
        lines.append("No per-level records were found.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_paths(
    paths: Sequence[str | Path],
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> ReportArtifacts:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    figure_root = output_root / "figures"
    records = load_benchmark_records(paths)
    summary_rows, per_level_rows = build_summary(
        records,
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
    )
    figure_paths = write_figures(summary_rows, per_level_rows, figure_root)

    summary_csv = output_root / "summary.csv"
    summary_json = output_root / "summary.json"
    report_md = output_root / "report.md"
    pd.DataFrame(_csv_ready_rows(summary_rows)).to_csv(summary_csv, index=False)
    summary_json.write_text(
        json.dumps(
            {
                "bootstrap": {
                    "resamples": int(bootstrap_resamples),
                    "seed": int(seed),
                    "unit": "shopping task",
                },
                "summary": _json_safe(summary_rows),
                "per_level": _json_safe(per_level_rows),
                "figures": {key: str(value) for key, value in figure_paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(
        report_md,
        summary_rows,
        per_level_rows,
        figure_paths,
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
    )
    return ReportArtifacts(
        output_root=output_root,
        summary_csv=summary_csv,
        summary_json=summary_json,
        report_md=report_md,
        figures=figure_paths,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate Shopping held-out task_results.jsonl, agent_events.jsonl, "
            "and evaluation summaries into report artifacts."
        )
    )
    parser.add_argument(
        "paths", nargs="+", help="Run/session directories or JSONL files."
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    artifacts = aggregate_paths(
        args.paths,
        output_root=args.output_root,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    print(f"Wrote {artifacts.summary_csv}")
    print(f"Wrote {artifacts.summary_json}")
    print(f"Wrote {artifacts.report_md}")
    for figure_path in artifacts.figures.values():
        print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
