"""
Analysis library for selective-oversight thesis experiments.

Loads result archives + Langfuse cost CSV and produces per-system / per-level
summary tables, per-case dataframes for head-to-heads, and cost rollups.

Designed to be re-runnable from a notebook with minimal path edits.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Map from archive folder name -> canonical system label used everywhere.
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
SYSTEM_ORDER = ["A", "C2", "C2-nt", "C2-noretry", "C2-deepseek", "D"]

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


def load_per_case(archives_root: Path) -> pd.DataFrame:
    """Load every (system, run, level, case) row from result_report folders.

    Returns columns:
      system, run, level, case_name, case_score, score, success,
      matched_count, expected_count, extra_products_count, is_completed
    """
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
    df = pd.DataFrame(rows)
    return df


def load_aggregated(archives_root: Path) -> pd.DataFrame:
    """Load aggregated_results/*_run_N_aggregated.json — one row per (system, run)."""
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
    # The CSV mislabels shopping-d as system="A" — fix from experiment_name.
    name_to_system = {
        v_name: v_sys
        for v_name, v_sys in zip(
            [
                "shopping-a",
                "shopping-c2",
                "shopping-c2-deepseek",
                "shopping-c2-noretry",
                "shopping-c2-nt",
                "shopping-d",
            ],
            ["A", "C2", "C2-deepseek", "C2-noretry", "C2-nt", "D"],
        )
    }
    raw["system"] = raw["experiment_name"].map(name_to_system).fillna(raw["system"])

    rows = []
    for _, r in raw.iterrows():
        system = r["system"]
        runs = int(r["run_count"])
        usage = json.loads(r["langfuse_usage_by_model_json"])
        executor_model = SYSTEM_META[system]["executor"]
        overseer_model = SYSTEM_META[system]["overseer"]

        for u in usage:
            model = u["model"].lower()
            # Edge case: when executor and overseer use the same model, Langfuse's
            # usage_by_model rolls them up into a single row that we cannot split.
            # Tag this as 'combined' so the breakdown plot can color it distinctly.
            if (
                executor_model
                and executor_model.lower() in model
                and overseer_model
                and overseer_model.lower() in model
                and executor_model == overseer_model
            ):
                role = "combined"
            elif executor_model and executor_model.lower() in model:
                role = "executor"
            elif overseer_model and overseer_model.lower() in model:
                role = "overseer"
            else:
                role = "other"

            rows.append(
                {
                    "experiment_name": r["experiment_name"],
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
    g["system"] = pd.Categorical(g["system"], categories=SYSTEM_ORDER, ordered=True)
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
    out["system"] = pd.Categorical(out["system"], categories=SYSTEM_ORDER, ordered=True)
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
    out["system"] = pd.Categorical(out["system"], categories=SYSTEM_ORDER, ordered=True)
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
