from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

try:
    from ._bootstrap import ensure_repo_root_on_path
    from .fetch_langfuse_session_usage import (
        DEFAULT_LANGFUSE_HOST,
        fetch_session_observations,
        summarize_usage,
        usage_row_from_observation,
    )
except ImportError:
    from _bootstrap import ensure_repo_root_on_path
    from fetch_langfuse_session_usage import (
        DEFAULT_LANGFUSE_HOST,
        fetch_session_observations,
        summarize_usage,
        usage_row_from_observation,
    )

ensure_repo_root_on_path()

from deepplanning.config import load_dotenv

DEFAULT_OUTPUTS_ROOT = Path("outputs")
DEFAULT_OUTPUT_PATH = Path("outputs/deepplanning/langfuse-results-usage.csv")
DEFAULT_INCLUDE_PREFIXES = ("shopping-", "system-")
DEFAULT_EXCLUDED_EXPERIMENTS = ("shopping-b", "system-b")
COUNT_METRIC_SUFFIXES = ("_cases", "_products")

UsageFetcher = Callable[[str], tuple[pd.DataFrame, int]]


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


def _metadata_domains(metadata: dict[str, Any]) -> str:
    domains = _nested(metadata, "parameters", "domains") or []
    if isinstance(domains, list):
        return ",".join(str(domain) for domain in domains)
    return str(domains)


def _metadata_split(metadata: dict[str, Any]) -> str:
    return str(_nested(metadata, "parameters", "shopping", "split") or "")


def _metadata_configured_runs(metadata: dict[str, Any]) -> int | None:
    value = _nested(metadata, "parameters", "runtime", "runs")
    return int(value) if value is not None else None


def _langfuse_session_id(metadata: dict[str, Any], session_root: Path) -> str:
    return str(metadata.get("timestamp") or session_root.name)


def _has_prefix(value: str, prefixes: Iterable[str]) -> bool:
    return any(value.startswith(prefix) for prefix in prefixes)


def discover_result_sessions(
    outputs_root: Path,
    *,
    include_prefixes: Iterable[str] = DEFAULT_INCLUDE_PREFIXES,
    excluded_experiments: Iterable[str] = DEFAULT_EXCLUDED_EXPERIMENTS,
    include_system_b: bool = False,
) -> list[Path]:
    include_prefix_list = tuple(include_prefixes)
    excluded = set(excluded_experiments)
    session_roots: list[Path] = []

    for metadata_path in sorted(outputs_root.rglob("experiment_session.json")):
        session_root = metadata_path.parent
        metadata = _read_json(metadata_path)
        experiment_name = _metadata_experiment_name(metadata, session_root)
        directory_name = session_root.parent.name
        system_name = _metadata_system_name(metadata)

        if include_prefix_list and not (
            _has_prefix(experiment_name, include_prefix_list)
            or _has_prefix(directory_name, include_prefix_list)
        ):
            continue
        if experiment_name in excluded or directory_name in excluded:
            continue
        if not include_system_b and system_name.upper() == "B":
            continue
        if not list((session_root / "aggregated_results").glob("*_aggregated.json")):
            continue

        session_roots.append(session_root)

    return session_roots


def _flatten_numeric_metrics(
    value: Any,
    *,
    prefix: str,
    metrics: dict[str, float],
) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}_{key}" if prefix else str(key)
            _flatten_numeric_metrics(item, prefix=next_prefix, metrics=metrics)
        return
    if isinstance(value, bool):
        return
    if isinstance(value, int | float):
        metrics[prefix] = float(value)


def _is_count_metric(metric_name: str) -> bool:
    return metric_name.endswith(COUNT_METRIC_SUFFIXES)


def aggregate_metrics(session_root: Path) -> dict[str, Any]:
    aggregate_paths = sorted((session_root / "aggregated_results").glob("*_aggregated.json"))
    metric_values: dict[str, list[float]] = {}
    run_ids: set[int] = set()
    model_names: set[str] = set()

    for aggregate_path in aggregate_paths:
        payload = _read_json(aggregate_path)
        run_id = payload.get("run_id")
        if run_id is not None:
            run_ids.add(int(run_id))
        model_name = payload.get("model_name")
        if model_name:
            model_names.add(str(model_name))

        flattened: dict[str, float] = {}
        _flatten_numeric_metrics(
            payload.get("domains") or {},
            prefix="domains",
            metrics=flattened,
        )
        _flatten_numeric_metrics(
            payload.get("overall") or {},
            prefix="overall",
            metrics=flattened,
        )
        for key, value in flattened.items():
            metric_values.setdefault(key, []).append(value)

    metrics: dict[str, Any] = {
        "aggregate_files_count": len(aggregate_paths),
        "run_count": len(run_ids) if run_ids else len(aggregate_paths),
        "aggregate_run_ids": ",".join(str(run_id) for run_id in sorted(run_ids)),
        "aggregate_model_names": ",".join(sorted(model_names)),
    }
    for key, values in sorted(metric_values.items()):
        metrics[f"aggregate_{key}_mean"] = sum(values) / len(values)
        if _is_count_metric(key):
            metrics[f"aggregate_{key}_sum"] = sum(values)

    return metrics


def make_langfuse_fetcher(
    *,
    host: str,
    public_key: str,
    secret_key: str,
    limit: int,
    timeout: float,
) -> UsageFetcher:
    def fetcher(session_id: str) -> tuple[pd.DataFrame, int]:
        observations = fetch_session_observations(
            session_id=session_id,
            host=host,
            public_key=public_key,
            secret_key=secret_key,
            limit=limit,
            timeout=timeout,
        )
        rows = [usage_row_from_observation(observation) for observation in observations]
        return summarize_usage(rows), len(observations)

    return fetcher


def _usage_totals(usage_summary: pd.DataFrame) -> dict[str, Any]:
    if usage_summary.empty:
        return {
            "langfuse_models": "",
            "langfuse_input_tokens": 0,
            "langfuse_input_cached_tokens": 0,
            "langfuse_output_tokens": 0,
            "langfuse_total_tokens": 0,
            "langfuse_usage_by_model_json": "[]",
        }

    records = usage_summary.to_dict("records")
    return {
        "langfuse_models": ",".join(str(model) for model in usage_summary["model"]),
        "langfuse_input_tokens": int(usage_summary["input"].sum()),
        "langfuse_input_cached_tokens": int(
            usage_summary["input_cached_tokens"].sum()
        ),
        "langfuse_output_tokens": int(usage_summary["output"].sum()),
        "langfuse_total_tokens": int(usage_summary["total"].sum()),
        "langfuse_usage_by_model_json": json.dumps(records, sort_keys=True),
    }


def build_report_rows(
    session_roots: Iterable[Path],
    *,
    usage_fetcher: UsageFetcher,
    continue_on_error: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for session_root in session_roots:
        metadata = _read_json(session_root / "experiment_session.json")
        session_id = _langfuse_session_id(metadata, session_root)
        row: dict[str, Any] = {
            "experiment_name": _metadata_experiment_name(metadata, session_root),
            "system": _metadata_system_name(metadata),
            "executor_model": _metadata_model_name(metadata, "executor"),
            "overseer_model": _metadata_model_name(metadata, "overseer"),
            "domains": _metadata_domains(metadata),
            "shopping_split": _metadata_split(metadata),
            "configured_runs": _metadata_configured_runs(metadata),
            "langfuse_session_id": session_id,
            "session_root": str(session_root),
        }
        row.update(aggregate_metrics(session_root))

        try:
            usage_summary, observation_count = usage_fetcher(session_id)
            row["langfuse_observation_count"] = observation_count
            row["langfuse_error"] = ""
            row.update(_usage_totals(usage_summary))
        except Exception as exc:
            if not continue_on_error:
                raise
            row["langfuse_observation_count"] = None
            row["langfuse_error"] = str(exc)
            row.update(_usage_totals(pd.DataFrame()))

        rows.append(row)

    return rows


def write_report(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def run(
    *,
    outputs_root: Path,
    output_path: Path,
    host: str,
    public_key: str,
    secret_key: str,
    include_prefixes: Iterable[str],
    excluded_experiments: Iterable[str],
    include_system_b: bool = False,
    continue_on_error: bool = False,
    limit: int = 1000,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    session_roots = discover_result_sessions(
        outputs_root,
        include_prefixes=include_prefixes,
        excluded_experiments=excluded_experiments,
        include_system_b=include_system_b,
    )
    usage_fetcher = make_langfuse_fetcher(
        host=host,
        public_key=public_key,
        secret_key=secret_key,
        limit=limit,
        timeout=timeout,
    )
    rows = build_report_rows(
        session_roots,
        usage_fetcher=usage_fetcher,
        continue_on_error=continue_on_error,
    )
    write_report(rows, output_path)
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Join benchmark aggregate metrics with Langfuse token usage for "
            "timestamp-named experiment sessions."
        )
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=DEFAULT_OUTPUTS_ROOT,
        help=f"Root to scan for experiment_session.json. Defaults to {DEFAULT_OUTPUTS_ROOT}.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"CSV output path. Defaults to {DEFAULT_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--include-prefix",
        action="append",
        dest="include_prefixes",
        help=(
            "Experiment directory/name prefix to include. Repeatable. Defaults "
            "to shopping- and system-."
        ),
    )
    parser.add_argument(
        "--exclude-experiment",
        action="append",
        dest="excluded_experiments",
        help="Experiment directory/name to exclude. Repeatable.",
    )
    parser.add_argument(
        "--include-system-b",
        action="store_true",
        help="Include metadata system=B sessions. By default they are skipped.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Write rows with langfuse_error instead of failing on a fetch error.",
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
        "--limit",
        type=int,
        default=1000,
        help="Observations per Langfuse page. Must be 1-1000.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP request timeout in seconds.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)
    include_prefixes = args.include_prefixes or list(DEFAULT_INCLUDE_PREFIXES)
    excluded_experiments = list(DEFAULT_EXCLUDED_EXPERIMENTS) + (
        args.excluded_experiments or []
    )

    try:
        rows = run(
            outputs_root=args.outputs_root,
            output_path=args.output,
            host=args.host,
            public_key=args.public_key,
            secret_key=args.secret_key,
            include_prefixes=include_prefixes,
            excluded_experiments=excluded_experiments,
            include_system_b=args.include_system_b,
            continue_on_error=args.continue_on_error,
            limit=args.limit,
            timeout=args.timeout,
        )
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
