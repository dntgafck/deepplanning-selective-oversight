from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _iter_jsonl_records(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            yield payload


def _task_result_paths(paths: Iterable[Path]) -> list[Path]:
    resolved: list[Path] = []
    for path in paths:
        if path.is_dir():
            resolved.extend(sorted(path.rglob("task_results.jsonl")))
        elif path.name == "task_results.jsonl":
            resolved.append(path)
        else:
            raise ValueError(f"Expected task_results.jsonl file or directory: {path}")
    return resolved


def load_task_result_records(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    result_paths = _task_result_paths([Path(path) for path in paths])
    records: list[dict[str, Any]] = []
    for path in result_paths:
        records.extend(_iter_jsonl_records(path))
    return records


def summarize_retry_breakdown(
    records: Iterable[dict[str, Any]],
    *,
    domain: str = "shopping",
) -> dict[str, Any]:
    selected = [
        record
        for record in records
        if str(record.get("domain") or domain).strip() == domain
    ]
    total = len(selected)
    first_attempt_success = 0
    recovered_after_retry = 0
    retry_cap_exhausted = 0
    repaired_retry_counts: list[int] = []
    total_cost = 0.0
    total_cost_count = 0
    overseer_calls = 0

    for record in selected:
        retry_count = int(record.get("final_verification_retry_count") or 0)
        success = bool(record.get("success", False))
        final_result = str(record.get("final_verification_result") or "")

        if success and retry_count == 0:
            first_attempt_success += 1
        elif success and retry_count > 0:
            recovered_after_retry += 1
            repaired_retry_counts.append(retry_count)

        if final_result == "retry_cap_exhausted":
            retry_cap_exhausted += 1

        if record.get("total_cost_usd") is not None:
            total_cost += float(record["total_cost_usd"])
            total_cost_count += 1
        overseer_calls += int(record.get("overseer_calls") or 0)

    def rate(count: int) -> float:
        return count / total if total else 0.0

    mean_retries = (
        sum(repaired_retry_counts) / len(repaired_retry_counts)
        if repaired_retry_counts
        else 0.0
    )

    return {
        "domain": domain,
        "total_records": total,
        "first_attempt_success_count": first_attempt_success,
        "first_attempt_success_rate": rate(first_attempt_success),
        "recovered_after_retry_count": recovered_after_retry,
        "recovered_after_retry_rate": rate(recovered_after_retry),
        "retry_cap_exhausted_count": retry_cap_exhausted,
        "retry_cap_exhausted_rate": rate(retry_cap_exhausted),
        "mean_retries_among_successful_repaired_cases": mean_retries,
        "total_cost_usd": total_cost if total_cost_count else None,
        "mean_cost_usd": total_cost / total_cost_count if total_cost_count else None,
        "overseer_calls": overseer_calls,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Shopping final-verifier retry outcomes from task_results.jsonl."
    )
    parser.add_argument(
        "paths", nargs="+", help="task_results.jsonl files or run directories"
    )
    parser.add_argument("--domain", default="shopping")
    args = parser.parse_args(argv)

    records = load_task_result_records(args.paths)
    summary = summarize_retry_breakdown(records, domain=args.domain)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
