from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

try:
    from ._bootstrap import ensure_repo_root_on_path
except ImportError:
    from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from deepplanning.config import load_dotenv

DEFAULT_LANGFUSE_HOST = "https://cloud.langfuse.com"
DEFAULT_OUTPUT_PATH = Path("langfuse-session-usage-by-model.csv")
OBSERVATION_FIELDS = "core,basic,model,usage"
CACHED_INPUT_USAGE_KEYS = (
    "input_cached_tokens",
    "cached_tokens",
    "cache_read_input_tokens",
    "cached_input_tokens",
)


def _normalize_host(host: str) -> str:
    normalized = host.strip().rstrip("/")
    if not normalized:
        raise ValueError("Langfuse host cannot be empty.")
    return normalized


def _session_filter(session_id: str) -> str:
    return json.dumps(
        [
            {
                "column": "sessionId",
                "operator": "=",
                "value": session_id,
                "type": "string",
            }
        ],
        separators=(",", ":"),
    )


def _coerce_token_count(value: Any) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def _usage_count(usage: dict[str, Any], *keys: str) -> int:
    for key in keys:
        if key in usage:
            return _coerce_token_count(usage.get(key))
    return 0


def usage_row_from_observation(observation: dict[str, Any]) -> dict[str, Any]:
    usage = observation.get("usageDetails") or {}
    if not isinstance(usage, dict):
        usage = {}

    input_tokens = _usage_count(usage, "input")
    output_tokens = _usage_count(usage, "output")
    total_tokens = _usage_count(usage, "total") or input_tokens + output_tokens

    return {
        "model": (
            observation.get("providedModelName")
            or observation.get("model")
            or observation.get("internalModelId")
            or "unknown"
        ),
        "input": input_tokens,
        "input_cached_tokens": _usage_count(usage, *CACHED_INPUT_USAGE_KEYS),
        "output": output_tokens,
        "total": total_tokens,
    }


def summarize_usage(rows: list[dict[str, Any]]) -> pd.DataFrame:
    columns = ["model", "input", "input_cached_tokens", "output", "total"]
    if not rows:
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(rows)
    return (
        frame.groupby("model", dropna=False)[
            ["input", "input_cached_tokens", "output", "total"]
        ]
        .sum()
        .reset_index()
        .sort_values("model", kind="stable")
        .reset_index(drop=True)
    )


def fetch_session_observations(
    *,
    session_id: str,
    host: str,
    public_key: str,
    secret_key: str,
    limit: int = 1000,
    timeout: float = 30.0,
    client: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    if not session_id.strip():
        raise ValueError("Session ID cannot be empty.")
    if not public_key.strip() or not secret_key.strip():
        raise ValueError(
            "Langfuse credentials are required. Set LANGFUSE_PUBLIC_KEY and "
            "LANGFUSE_SECRET_KEY, or pass --public-key/--secret-key."
        )
    if limit < 1 or limit > 1000:
        raise ValueError("Limit must be between 1 and 1000.")

    observations: list[dict[str, Any]] = []
    cursor: str | None = None
    owns_client = client is None
    active_client = client or httpx.Client(
        auth=(public_key, secret_key),
        timeout=timeout,
    )

    try:
        while True:
            params: dict[str, str | int] = {
                "fields": OBSERVATION_FIELDS,
                "limit": limit,
                "filter": _session_filter(session_id),
            }
            if cursor:
                params["cursor"] = cursor

            response = active_client.get(
                f"{_normalize_host(host)}/api/public/v2/observations",
                params=params,
            )
            response.raise_for_status()
            payload = response.json()
            page_data = payload.get("data", [])
            if not isinstance(page_data, list):
                raise RuntimeError("Langfuse response field 'data' is not a list.")
            observations.extend(page_data)

            meta = payload.get("meta") or {}
            cursor = meta.get("cursor") if isinstance(meta, dict) else None
            if not cursor:
                break
    finally:
        if owns_client:
            active_client.close()

    return observations


def write_summary(summary: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)


def run(
    *,
    session_id: str,
    host: str,
    public_key: str,
    secret_key: str,
    output_path: Path,
    limit: int = 1000,
    timeout: float = 30.0,
) -> pd.DataFrame:
    observations = fetch_session_observations(
        session_id=session_id,
        host=host,
        public_key=public_key,
        secret_key=secret_key,
        limit=limit,
        timeout=timeout,
    )
    summary = summarize_usage(
        [usage_row_from_observation(observation) for observation in observations]
    )
    write_summary(summary, output_path)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch Langfuse token usage for one session and summarize by model."
    )
    parser.add_argument("session_id", help="Langfuse sessionId to query.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"CSV output path. Defaults to {DEFAULT_OUTPUT_PATH}.",
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

    try:
        summary = run(
            session_id=args.session_id,
            host=args.host,
            public_key=args.public_key,
            secret_key=args.secret_key,
            output_path=args.output,
            limit=args.limit,
            timeout=args.timeout,
        )
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")

    if summary.empty:
        print("No observations found.")
    else:
        print(summary.to_string(index=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
