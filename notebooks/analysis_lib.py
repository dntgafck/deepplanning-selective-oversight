"""
Analysis library for selective-oversight thesis experiments.

Loads output session directories or legacy result archives plus Langfuse cost
CSV, then produces per-system / per-level summary tables, per-case dataframes
for head-to-heads, and cost rollups.

Designed to be re-runnable from a notebook with minimal path edits.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Result directory discovery. Primary runs now use system-* names. The shopping-*
# fallback keeps this notebook usable with the existing v1 output directories.
DEFAULT_RESULT_PREFIXES = ("system-",)
DEFAULT_FALLBACK_RESULT_PREFIXES = ("shopping-",)
DEFAULT_IGNORE_PREFIXES = ("system-b", "shopping-b")

# Map from legacy archive folder name -> canonical system label used everywhere.
# The CSV labels shopping-d as "A" in its `system` column — we override that
# here so D is always D.
ARCHIVE_TO_SYSTEM = {
    "shopping-a-results": "A",
    "shopping-c2-results": "C2",
    "shopping-c2-nt-results": "C2-nt",
    "shopping-c2-noretry-results": "C2-noretry",
    "shopping-c2-deepseek-results": "C2-deepseek",
    "shopping-d-results": "D",
}

# Display order for tables / plots.
SYSTEM_ORDER = [
    "A",
    "B",
    "C1",
    "C2",
    "C2-nt",
    "C2-noretry",
    "C2-deepseek",
    "C2-deepseek-nt",
    "C2-deepseek-pro",
    "D",
]

# Per-system metadata: executor / overseer / how to read it.
SYSTEM_META = {
    "A": {"executor": "qwen3.5-9b", "overseer": None, "kind": "exec-only"},
    "C2": {
        "executor": "qwen3.5-9b",
        "overseer": "deepseek-v4-flash",
        "kind": "selective",
    },
    "C2-nt": {
        "executor": "qwen3.5-9b",
        "overseer": "deepseek-v4-flash",
        "kind": "selective-no-think",
    },
    "C2-noretry": {
        "executor": "qwen3.5-9b",
        "overseer": "deepseek-v4-flash",
        "kind": "selective-no-retry",
    },
    "C2-deepseek": {
        "executor": "deepseek-v4-flash",
        "overseer": "deepseek-v4-flash",
        "kind": "selective-ds",
    },
    "D": {"executor": "deepseek-v4-flash", "overseer": None, "kind": "monolithic"},
}

# DeepSeek pricing per 1M tokens (deepseek-v4-flash list price; user can edit).
# Source: deepseek API current pricing tier; we report both as-billed and uncached
# variants so any pricing drift only changes magnitude, not the qualitative story.
PRICE_DS_INPUT_UNCACHED = 0.14  # USD / 1M tokens
PRICE_DS_INPUT_CACHED = 0.0028  # USD / 1M tokens (cache hit)
PRICE_DS_OUTPUT = 0.28  # USD / 1M tokens

# Qwen3.5-9B via Together.ai (open-weight tier; user can edit).
PRICE_QWEN_INPUT = 0.10  # USD / 1M tokens
PRICE_QWEN_OUTPUT = 0.15  # USD / 1M tokens
# Qwen does not have a separate cached price in our Langfuse data (0% cached).


# -----------------------------------------------------------------------------
# Archive loading
# -----------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _nested(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


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


def _normalize_slug(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _optional_str(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _has_prefix(value: str, prefixes: Iterable[str]) -> bool:
    normalized = _normalize_slug(value)
    return any(normalized.startswith(_normalize_slug(prefix)) for prefix in prefixes)


def normalize_system_label(experiment_name: str, metadata_system: str = "") -> str:
    """Derive a stable display label from an experiment directory/name.

    Examples:
      shopping-a -> A
      system-c2-noretry -> C2-noretry
      shopping-c2-deepseek-pro -> C2-deepseek-pro
    """
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


def ordered_systems(systems: Iterable[str]) -> list[str]:
    """Return known systems in thesis order, followed by new variants by name."""
    present = {str(system) for system in systems if pd.notna(system)}
    ordered = [system for system in SYSTEM_ORDER if system in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def set_system_order(systems: Iterable[str]) -> list[str]:
    """Update global plot/table order from the systems loaded in this notebook."""
    global SYSTEM_ORDER
    SYSTEM_ORDER = ordered_systems(systems)
    return SYSTEM_ORDER


def discover_output_sessions(
    experiments_root: Path,
    *,
    include_prefixes: Iterable[str] = DEFAULT_RESULT_PREFIXES,
    fallback_include_prefixes: Iterable[str] = DEFAULT_FALLBACK_RESULT_PREFIXES,
    ignore_prefixes: Iterable[str] = DEFAULT_IGNORE_PREFIXES,
    include_system_b: bool = False,
) -> list[Path]:
    """Discover timestamped experiment sessions under outputs.

    The primary prefix list is tried first. If it finds no sessions, the fallback
    prefix list is tried so older shopping-* output directories still work.
    """

    def discover_with_prefixes(prefixes: Iterable[str]) -> list[Path]:
        session_roots: list[Path] = []
        for metadata_path in sorted(experiments_root.rglob("experiment_session.json")):
            session_root = metadata_path.parent
            metadata = _read_json(metadata_path)
            experiment_name = _metadata_experiment_name(metadata, session_root)
            directory_name = session_root.parent.name
            system_name = _metadata_system_name(metadata)
            system_label = normalize_system_label(experiment_name, system_name)

            if prefixes and not (
                _has_prefix(experiment_name, prefixes)
                or _has_prefix(directory_name, prefixes)
            ):
                continue
            if ignore_prefixes and (
                _has_prefix(experiment_name, ignore_prefixes)
                or _has_prefix(directory_name, ignore_prefixes)
            ):
                continue
            if not include_system_b and (
                system_name.upper() == "B" or system_label.upper() == "B"
            ):
                continue
            if not list(
                (session_root / "aggregated_results").glob("*_aggregated.json")
            ):
                continue
            session_roots.append(session_root)
        return session_roots

    sessions = discover_with_prefixes(tuple(include_prefixes))
    if sessions or not fallback_include_prefixes:
        return sessions
    return discover_with_prefixes(tuple(fallback_include_prefixes))


def describe_sessions(session_roots: Iterable[Path]) -> pd.DataFrame:
    """Return a small table describing discovered sessions."""
    rows = []
    for session_root in session_roots:
        metadata = _read_json(session_root / "experiment_session.json")
        experiment_name = _metadata_experiment_name(metadata, session_root)
        system_name = _metadata_system_name(metadata)
        rows.append(
            {
                "experiment_name": experiment_name,
                "system": normalize_system_label(experiment_name, system_name),
                "metadata_system": system_name,
                "timestamp": session_root.name,
                "session_root": str(session_root),
            }
        )
    return pd.DataFrame(rows)


def _level_from_dirname(dirname: str) -> int:
    """Extract level number from folder name like database_qwen3.5-9b_level1_..."""
    m = re.search(r"level(\d+)", dirname)
    if not m:
        raise ValueError(f"Cannot parse level from {dirname!r}")
    return int(m.group(1))


def _find_session_root(archive_dir: Path) -> Path:
    """Each archive has shape: shopping-X-results/shopping-X/<timestamp>/..."""
    inner = next(p for p in archive_dir.iterdir() if p.is_dir())
    timestamp_dir = next(p for p in inner.iterdir() if p.is_dir())
    return timestamp_dir


def _session_context(session_root: Path) -> dict[str, str]:
    metadata = _read_json(session_root / "experiment_session.json")
    experiment_name = _metadata_experiment_name(metadata, session_root)
    system_name = _metadata_system_name(metadata)
    return {
        "experiment_name": experiment_name,
        "system": normalize_system_label(experiment_name, system_name),
        "executor_model": _metadata_model_name(metadata, "executor"),
        "overseer_model": _metadata_model_name(metadata, "overseer"),
        "timestamp": session_root.name,
    }


def _run_id_from_run_dir(run_dir: Path) -> int:
    return int(run_dir.name.split("_")[1])


def _load_per_case_from_sessions(session_roots: Iterable[Path]) -> pd.DataFrame:
    rows = []
    next_run_by_system: dict[str, int] = {}

    for session_root in sorted(Path(root) for root in session_roots):
        context = _session_context(session_root)
        system = context["system"]
        run_map: dict[int, int] = {}
        for run_dir in sorted(session_root.glob("shopping/*/run_*")):
            original_run_id = _run_id_from_run_dir(run_dir)
            if original_run_id not in run_map:
                run_map[original_run_id] = next_run_by_system.get(system, 0)
                next_run_by_system[system] = run_map[original_run_id] + 1
            run_id = run_map[original_run_id]
            rr = run_dir / "result_report"
            if not rr.exists():
                continue
            for level_dir in sorted(rr.iterdir()):
                if not level_dir.is_dir():
                    continue
                level = _level_from_dirname(level_dir.name)
                summary = level_dir / "summary_report.json"
                if not summary.exists():
                    print(f"[warn] missing summary: {summary}")
                    continue
                data = _read_json(summary)
                for c in data.get("case_results", []):
                    rows.append(
                        {
                            **context,
                            "run": run_id,
                            "original_run": original_run_id,
                            "level": level,
                            "case_name": c["case_name"],
                            "case_score": c.get("case_score", 0.0),
                            "score": c.get("score", 0.0),
                            "success": bool(c.get("success", False)),
                            "matched_count": c.get("matched_count", 0),
                            "expected_count": c.get("expected_count", 0),
                            "extra_products_count": c.get("extra_products_count", 0),
                            "is_completed": bool(c.get("is_completed", False)),
                        }
                    )
    return pd.DataFrame(rows)


def _load_per_case_from_archives(archives_root: Path) -> pd.DataFrame:
    rows = []
    for archive_name, system in ARCHIVE_TO_SYSTEM.items():
        archive_dir = archives_root / archive_name
        if not archive_dir.exists():
            print(f"[warn] missing archive: {archive_dir}")
            continue
        session_root = _find_session_root(archive_dir)
        # Walk shopping/<model>/run_*/result_report/database_*_levelN_*/summary_report.json
        for run_dir in sorted(session_root.glob("shopping/*/run_*")):
            run_id = int(run_dir.name.split("_")[1])
            rr = run_dir / "result_report"
            for level_dir in sorted(rr.iterdir()):
                if not level_dir.is_dir():
                    continue
                level = _level_from_dirname(level_dir.name)
                summary = level_dir / "summary_report.json"
                if not summary.exists():
                    print(f"[warn] missing summary: {summary}")
                    continue
                with summary.open() as f:
                    data = json.load(f)
                for c in data.get("case_results", []):
                    rows.append(
                        {
                            "system": system,
                            "run": run_id,
                            "level": level,
                            "case_name": c["case_name"],
                            "case_score": c.get("case_score", 0.0),
                            "score": c.get("score", 0.0),
                            "success": bool(c.get("success", False)),
                            "matched_count": c.get("matched_count", 0),
                            "expected_count": c.get("expected_count", 0),
                            "extra_products_count": c.get("extra_products_count", 0),
                            "is_completed": bool(c.get("is_completed", False)),
                        }
                    )
    return pd.DataFrame(rows)


def load_per_case(source: Path | Iterable[Path]) -> pd.DataFrame:
    """Load every (system, run, level, case) row from result_report folders.

    Returns columns:
      system, run, level, case_name, case_score, score, success,
      matched_count, expected_count, extra_products_count, is_completed
    """
    if isinstance(source, Path):
        if (source / "experiment_session.json").exists():
            return _load_per_case_from_sessions([source])
        return _load_per_case_from_archives(source)
    return _load_per_case_from_sessions(source)


def _load_aggregated_from_sessions(session_roots: Iterable[Path]) -> pd.DataFrame:
    rows = []
    next_run_by_system: dict[str, int] = {}

    for session_root in sorted(Path(root) for root in session_roots):
        context = _session_context(session_root)
        system = context["system"]
        run_map: dict[int, int] = {}
        for agg in sorted(
            (session_root / "aggregated_results").glob("*_aggregated.json")
        ):
            data = _read_json(agg)
            original_run_id = int(data["run_id"])
            if original_run_id not in run_map:
                run_map[original_run_id] = next_run_by_system.get(system, 0)
                next_run_by_system[system] = run_map[original_run_id] + 1
            shop = (data.get("domains") or {}).get("shopping")
            if not shop:
                continue
            rows.append(
                {
                    **context,
                    "run": run_map[original_run_id],
                    "original_run": original_run_id,
                    "model_name": data["model_name"],
                    "total_cases": shop["total_cases"],
                    "successful_cases": shop["successful_cases"],
                    "successful_rate": shop["successful_rate"],
                    "match_rate": shop["match_rate"],
                    "weighted_avg_case_score": shop["weighted_average_case_score"],
                    "incomplete_cases": shop["incomplete_cases"],
                    "incomplete_rate": shop["incomplete_rate"],
                }
            )
    return pd.DataFrame(rows)


def _load_aggregated_from_archives(archives_root: Path) -> pd.DataFrame:
    rows = []
    for archive_name, system in ARCHIVE_TO_SYSTEM.items():
        archive_dir = archives_root / archive_name
        if not archive_dir.exists():
            continue
        session_root = _find_session_root(archive_dir)
        for agg in sorted(
            (session_root / "aggregated_results").glob("*_aggregated.json")
        ):
            with agg.open() as f:
                data = json.load(f)
            shop = data["domains"]["shopping"]
            rows.append(
                {
                    "system": system,
                    "run": data["run_id"],
                    "model_name": data["model_name"],
                    "total_cases": shop["total_cases"],
                    "successful_cases": shop["successful_cases"],
                    "successful_rate": shop["successful_rate"],
                    "match_rate": shop["match_rate"],
                    "weighted_avg_case_score": shop["weighted_average_case_score"],
                    "incomplete_cases": shop["incomplete_cases"],
                    "incomplete_rate": shop["incomplete_rate"],
                }
            )
    return pd.DataFrame(rows)


def load_aggregated(source: Path | Iterable[Path]) -> pd.DataFrame:
    """Load aggregated_results/*_run_N_aggregated.json — one row per (system, run)."""
    if isinstance(source, Path):
        if (source / "experiment_session.json").exists():
            return _load_aggregated_from_sessions([source])
        return _load_aggregated_from_archives(source)
    return _load_aggregated_from_sessions(source)


# -----------------------------------------------------------------------------
# Langfuse cost loading
# -----------------------------------------------------------------------------


def load_langfuse(csv_path: Path) -> pd.DataFrame:
    """Load Langfuse usage CSV, normalize the system label, expand per-model rows.

    Returns long-format frame:
      experiment_name, system, role, model, input, input_cached, output, total,
      runs, per_run_total, cache_hit_rate
    """
    raw = pd.read_csv(csv_path)

    rows = []
    for _, r in raw.iterrows():
        experiment_name = _optional_str(r["experiment_name"])
        system = normalize_system_label(experiment_name, _optional_str(r.get("system")))
        runs = int(r["run_count"])
        usage = json.loads(r["langfuse_usage_by_model_json"])
        meta = SYSTEM_META.get(system, {})
        executor_model = _optional_str(r.get("executor_model")) or _optional_str(
            meta.get("executor")
        )
        overseer_model = _optional_str(r.get("overseer_model")) or _optional_str(
            meta.get("overseer")
        )
        session_key = _optional_str(
            r.get("langfuse_session_id") or f"{experiment_name}-{r.name}"
        )

        for u in usage:
            model = u["model"].lower()
            # Edge case: when executor and overseer use the same model, Langfuse's
            # usage_by_model rolls them up into a single row that we cannot split.
            # Tag this as 'combined' so the breakdown plot can color it distinctly.
            executor_l = executor_model.lower()
            overseer_l = overseer_model.lower()
            if (
                executor_l
                and overseer_l
                and executor_l == overseer_l
                and executor_l in model
            ):
                role = "combined"
            elif executor_l and executor_l in model:
                role = "executor"
            elif overseer_l and overseer_l in model:
                role = "overseer"
            else:
                role = "other"

            rows.append(
                {
                    "experiment_name": experiment_name,
                    "session_key": session_key,
                    "system": system,
                    "runs": runs,
                    "role": role,
                    "model": u["model"],
                    "input_uncached": u["input"],
                    "input_cached": u["input_cached_tokens"],
                    "output": u["output"],
                    "total": u["total"],
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    runs_by_system = (
        df[["system", "session_key", "runs"]]
        .drop_duplicates()
        .groupby("system")["runs"]
        .sum()
    )
    df = (
        df.groupby(["system", "role", "model"], as_index=False)
        .agg(
            experiment_name=(
                "experiment_name",
                lambda values: ",".join(sorted(set(values))),
            ),
            session_key=("session_key", lambda values: ",".join(sorted(set(values)))),
            input_uncached=("input_uncached", "sum"),
            input_cached=("input_cached", "sum"),
            output=("output", "sum"),
            total=("total", "sum"),
        )
        .reset_index(drop=True)
    )
    df["runs"] = df["system"].map(runs_by_system).astype(int)
    df["input_total"] = df["input_uncached"] + df["input_cached"]
    df["per_run_input_uncached"] = df["input_uncached"] / df["runs"]
    df["per_run_input_cached"] = df["input_cached"] / df["runs"]
    df["per_run_output"] = df["output"] / df["runs"]
    df["per_run_total"] = df["total"] / df["runs"]
    df["cache_hit_rate"] = np.where(
        df["input_total"] > 0,
        df["input_cached"] / df["input_total"],
        np.nan,
    )
    return df


def system_token_summary(lf: pd.DataFrame) -> pd.DataFrame:
    """One row per system with tokens summed across roles, per-run averaged."""
    g = lf.groupby("system", as_index=False).agg(
        total_tokens=("total", "sum"),
        input_uncached=("input_uncached", "sum"),
        input_cached=("input_cached", "sum"),
        output=("output", "sum"),
        runs=("runs", "first"),
    )
    g["per_run_total"] = g["total_tokens"] / g["runs"]
    g["per_run_input_uncached"] = g["input_uncached"] / g["runs"]
    g["per_run_input_cached"] = g["input_cached"] / g["runs"]
    g["per_run_output"] = g["output"] / g["runs"]
    g["input_total"] = g["input_uncached"] + g["input_cached"]
    g["cache_hit_rate"] = np.where(
        g["input_total"] > 0,
        g["input_cached"] / g["input_total"],
        np.nan,
    )
    g["system"] = pd.Categorical(
        g["system"], categories=ordered_systems(g["system"]), ordered=True
    )
    return g.sort_values("system").reset_index(drop=True)


def cost_dollars(lf: pd.DataFrame) -> pd.DataFrame:
    """Compute per-system $ as-billed and at uncached-input rates.

    `as_billed` honors cache hit pricing.
    `uncached` prices all input tokens at the uncached rate (a robustness frame
    that strips out the cache subsidy — useful for comparing token volume).
    """
    # Approximate per-system pricing — assume each system's total is dominated by
    # one provider for input; for mixed systems (C2 family) we split by role/model
    # so prices map correctly.
    rows = []
    for system, grp in lf.groupby("system"):
        as_billed_usd = 0.0
        uncached_usd = 0.0
        runs = int(grp["runs"].iloc[0])
        for _, r in grp.iterrows():
            m = r["model"].lower()
            if "deepseek" in m:
                as_billed_usd += (
                    r["input_uncached"] / 1e6 * PRICE_DS_INPUT_UNCACHED
                    + r["input_cached"] / 1e6 * PRICE_DS_INPUT_CACHED
                    + r["output"] / 1e6 * PRICE_DS_OUTPUT
                )
                uncached_usd += (
                    r["input_uncached"] + r["input_cached"]
                ) / 1e6 * PRICE_DS_INPUT_UNCACHED + r["output"] / 1e6 * PRICE_DS_OUTPUT
            elif "qwen" in m:
                qwen_cost = (
                    r["input_uncached"] / 1e6 * PRICE_QWEN_INPUT
                    + r["output"] / 1e6 * PRICE_QWEN_OUTPUT
                )
                as_billed_usd += qwen_cost
                uncached_usd += qwen_cost  # qwen has no cache split in our data
        rows.append(
            {
                "system": system,
                "runs": runs,
                "total_usd_as_billed": as_billed_usd,
                "total_usd_uncached": uncached_usd,
                "per_run_usd_as_billed": as_billed_usd / runs,
                "per_run_usd_uncached": uncached_usd / runs,
            }
        )
    out = pd.DataFrame(rows)
    out["system"] = pd.Categorical(
        out["system"], categories=ordered_systems(out["system"]), ordered=True
    )
    return out.sort_values("system").reset_index(drop=True)


# -----------------------------------------------------------------------------
# Accuracy summaries
# -----------------------------------------------------------------------------


def per_run_per_level(per_case: pd.DataFrame) -> pd.DataFrame:
    """One row per (system, run, level): case_accuracy, mean_score, n."""
    g = per_case.groupby(["system", "run", "level"], as_index=False).agg(
        case_accuracy=("case_score", "mean"),
        mean_score=("score", "mean"),
        n=("case_score", "size"),
        successes=("case_score", lambda s: int((s == 1.0).sum())),
    )
    return g


def per_run_overall(per_case: pd.DataFrame) -> pd.DataFrame:
    """One row per (system, run): overall case_accuracy."""
    g = per_case.groupby(["system", "run"], as_index=False).agg(
        case_accuracy=("case_score", "mean"),
        mean_score=("score", "mean"),
        successes=("case_score", lambda s: int((s == 1.0).sum())),
        total=("case_score", "size"),
        match_rate=("matched_count", "sum"),
    )
    # match_rate post-fix: matched / expected across all cases in the run.
    expected = (
        per_case.groupby(["system", "run"])["expected_count"]
        .sum()
        .rename("expected_total")
    )
    g = g.merge(expected, on=["system", "run"])
    g["match_rate"] = g["match_rate"] / g["expected_total"]
    return g


def system_summary(per_case: pd.DataFrame) -> pd.DataFrame:
    """Per-system: mean ± std of case_accuracy across runs, plus level breakdown."""
    overall = per_run_overall(per_case)
    overall_summary = overall.groupby("system", as_index=False).agg(
        case_acc_mean=("case_accuracy", "mean"),
        case_acc_std=("case_accuracy", "std"),
        match_rate_mean=("match_rate", "mean"),
        runs=("case_accuracy", "size"),
    )

    per_level = per_run_per_level(per_case)
    level_pivot = per_level.groupby(["system", "level"], as_index=False).agg(
        case_acc_mean=("case_accuracy", "mean"), case_acc_std=("case_accuracy", "std")
    )
    level_wide_mean = level_pivot.pivot(
        index="system", columns="level", values="case_acc_mean"
    )
    level_wide_mean.columns = [f"L{c}_mean" for c in level_wide_mean.columns]
    level_wide_std = level_pivot.pivot(
        index="system", columns="level", values="case_acc_std"
    )
    level_wide_std.columns = [f"L{c}_std" for c in level_wide_std.columns]
    out = overall_summary.merge(
        level_wide_mean, left_on="system", right_index=True
    ).merge(level_wide_std, left_on="system", right_index=True)
    out["system"] = pd.Categorical(
        out["system"], categories=ordered_systems(out["system"]), ordered=True
    )
    return out.sort_values("system").reset_index(drop=True)


# -----------------------------------------------------------------------------
# Head-to-head per-case comparison
# -----------------------------------------------------------------------------


def head_to_head(per_case: pd.DataFrame, sys_a: str, sys_b: str) -> pd.DataFrame:
    """Per-case mean across runs, joined system A vs system B.

    Returns columns: case_name, level, score_A, score_B, delta (=A-B), winner
    """
    pa = (
        per_case[per_case.system == sys_a]
        .groupby(["level", "case_name"], as_index=False)["case_score"]
        .mean()
        .rename(columns={"case_score": f"score_{sys_a}"})
    )
    pb = (
        per_case[per_case.system == sys_b]
        .groupby(["level", "case_name"], as_index=False)["case_score"]
        .mean()
        .rename(columns={"case_score": f"score_{sys_b}"})
    )
    j = pa.merge(pb, on=["level", "case_name"])
    j["delta"] = j[f"score_{sys_a}"] - j[f"score_{sys_b}"]
    j["winner"] = np.where(
        j["delta"] > 0, sys_a, np.where(j["delta"] < 0, sys_b, "tie")
    )
    return j


def head_to_head_summary(j: pd.DataFrame, sys_a: str, sys_b: str) -> dict:
    counts = j["winner"].value_counts().to_dict()
    return {
        f"{sys_a}_wins": counts.get(sys_a, 0),
        f"{sys_b}_wins": counts.get(sys_b, 0),
        "ties": counts.get("tie", 0),
        f"mean_score_{sys_a}": j[f"score_{sys_a}"].mean(),
        f"mean_score_{sys_b}": j[f"score_{sys_b}"].mean(),
        "mean_delta": j["delta"].mean(),
        "n_cases": len(j),
    }


# -----------------------------------------------------------------------------
# Pretty-printers
# -----------------------------------------------------------------------------


def fmt_pct(x: float, nd: int = 2) -> str:
    return f"{x*100:.{nd}f}%" if pd.notna(x) else "—"


def fmt_pp(x: float, nd: int = 2) -> str:
    return f"{x*100:+.{nd}f}pp" if pd.notna(x) else "—"
