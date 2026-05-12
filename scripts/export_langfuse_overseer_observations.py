from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fire
import httpx
from tqdm.auto import tqdm

try:
    from ._bootstrap import ensure_repo_root_on_path
    from .fetch_langfuse_session_usage import DEFAULT_LANGFUSE_HOST, _normalize_host
except ImportError:
    from _bootstrap import ensure_repo_root_on_path
    from fetch_langfuse_session_usage import DEFAULT_LANGFUSE_HOST, _normalize_host

ensure_repo_root_on_path()

from deepplanning.config import load_dotenv

DEFAULT_OUTPUT_DIR = Path("outputs/deepplanning/langfuse_exports")
DEFAULT_FIELDS = "core,basic,model,usage,metadata,io"
MAX_OBSERVATION_BYTES = 10 * 1024 * 1024

SOURCE_SESSION_ROOTS = (
    Path("outputs/deepplanning/experiments/shopping-c2/2026-04-30_09-18-02"),
    Path("outputs/deepplanning/experiments/shopping-c2-nt/2026-04-30_13-35-38"),
    Path("outputs/deepplanning/experiments/shopping-c2-noretry/2026-04-29_21-07-58"),
    Path("outputs/deepplanning/experiments/shopping-c2-deepseek/2026-05-01_18-54-47"),
    Path("outputs/deepplanning/experiments/shopping-c2-deepseek/2026-05-10_18-11-57"),
    Path(
        "outputs/deepplanning/experiments/shopping-c2-deepseek-nt/2026-05-10_18-08-50"
    ),
)

EXCLUDED_SETUP_NAMES = {
    "overseer.compile_contract",
    "overseer.compile_checklist",
}
EXCLUDED_TRIGGER_TYPES = {"compile_contract", "compile_checklist"}
INCLUDED_HOOKS = {"pre_tool", "midpoint", "post_tool", "final"}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _nested(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _source_system_name(metadata: dict[str, Any], session_root: Path) -> str:
    experiment_name = str(
        _nested(metadata, "experiment", "name")
        or _nested(metadata, "parameters", "name")
        or session_root.parent.name
    )
    mapping = {
        "shopping-c2": "C2",
        "shopping-c2-nt": "C2-nt",
        "shopping-c2-noretry": "C2-noretry",
        "shopping-c2-deepseek": "C2-deepseek",
        "shopping-c2-deepseek-nt": "C2-deepseek-nt",
    }
    return mapping.get(experiment_name, experiment_name.removeprefix("shopping-"))


def _langfuse_session_id(metadata: dict[str, Any], session_root: Path) -> str:
    return str(metadata.get("timestamp") or session_root.name)


def _normalized_config_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    parameters = metadata.get("parameters") or {}
    if not isinstance(parameters, dict):
        parameters = {}
    return {
        "name": parameters.get("name") or _nested(metadata, "experiment", "name"),
        "domains": parameters.get("domains"),
        "models": parameters.get("models"),
        "system": parameters.get("system"),
        "runtime": {
            "workers": _nested(parameters, "runtime", "workers"),
            "max_llm_calls": _nested(parameters, "runtime", "max_llm_calls"),
            "infra_retry_limit": _nested(parameters, "runtime", "infra_retry_limit"),
            "runs": _nested(parameters, "runtime", "runs"),
        },
        "shopping": parameters.get("shopping"),
    }


def session_descriptor(session_root: Path) -> dict[str, Any]:
    metadata_path = session_root / "experiment_session.json"
    metadata = _read_json(metadata_path)
    config_payload = _normalized_config_payload(metadata)
    return {
        "session_root": str(session_root),
        "source_system": _source_system_name(metadata, session_root),
        "experiment_name": str(
            _nested(metadata, "experiment", "name")
            or _nested(metadata, "parameters", "name")
            or session_root.parent.name
        ),
        "langfuse_session_id": _langfuse_session_id(metadata, session_root),
        "config_digest": _sha256_text(
            json.dumps(config_payload, sort_keys=True, separators=(",", ":"))
        ),
        "config_payload": config_payload,
    }


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


def fetch_observations(
    *,
    session_id: str,
    host: str,
    public_key: str,
    secret_key: str,
    fields: str = DEFAULT_FIELDS,
    limit: int = 1000,
    timeout: float = 30.0,
    max_records: int | None = None,
    client: httpx.Client | None = None,
    progress: bool = False,
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

    progress_bar = tqdm(
        total=max_records,
        desc=f"fetch {session_id}",
        unit="obs",
        dynamic_ncols=True,
        disable=not progress,
    )
    try:
        while True:
            params: dict[str, str | int] = {
                "fields": fields,
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
            for item in page_data:
                if isinstance(item, dict):
                    observations.append(item)
                    progress_bar.update(1)
                    if max_records is not None and len(observations) >= max_records:
                        return observations

            meta = payload.get("meta") or {}
            cursor = meta.get("cursor") if isinstance(meta, dict) else None
            if not cursor:
                break
    finally:
        progress_bar.close()
        if owns_client:
            active_client.close()

    return observations


def observation_metadata(observation: dict[str, Any]) -> dict[str, Any]:
    metadata = observation.get("metadata") or {}
    return metadata if isinstance(metadata, dict) else {}


def observation_name(observation: dict[str, Any]) -> str:
    return str(observation.get("name") or observation.get("observationName") or "")


def _observation_hook(metadata: dict[str, Any], name: str) -> str:
    hook = str(metadata.get("hook") or "").strip()
    if hook:
        return hook
    parts = name.split(".")
    return parts[1] if len(parts) >= 3 and parts[0] == "overseer" else ""


def is_runtime_or_final_overseer_observation(observation: dict[str, Any]) -> bool:
    metadata = observation_metadata(observation)
    name = observation_name(observation)
    if metadata.get("actor") != "overseer":
        return False
    if name in EXCLUDED_SETUP_NAMES:
        return False
    if str(metadata.get("trigger_type") or "") in EXCLUDED_TRIGGER_TYPES:
        return False
    hook = _observation_hook(metadata, name)
    return hook in INCLUDED_HOOKS


def _usage_payload(observation: dict[str, Any]) -> dict[str, Any]:
    for key in ("usageDetails", "usage", "usage_details"):
        value = observation.get(key)
        if isinstance(value, dict):
            return value
    return {}


def extract_input_messages(observation: dict[str, Any]) -> list[dict[str, Any]] | None:
    candidate = observation.get("input")
    if isinstance(candidate, str):
        try:
            candidate = json.loads(candidate)
        except json.JSONDecodeError:
            return None
    if isinstance(candidate, dict) and isinstance(candidate.get("messages"), list):
        return [dict(item) for item in candidate["messages"] if isinstance(item, dict)]
    if isinstance(candidate, list) and all(
        isinstance(item, dict) for item in candidate
    ):
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
    usage = _usage_payload(observation)
    trace_id = observation.get("traceId") or observation.get("trace_id")
    session_id = (
        observation.get("sessionId")
        or observation.get("session_id")
        or descriptor["langfuse_session_id"]
    )
    return {
        "source_system": descriptor["source_system"],
        "source_session_root": descriptor["session_root"],
        "langfuse_session_id": descriptor["langfuse_session_id"],
        "session_id": session_id,
        "trace_id": trace_id,
        "source_config_digest": descriptor["config_digest"],
        "observation_id": observation.get("id"),
        "name": observation_name(observation),
        "type": observation.get("type"),
        "started_at": observation.get("startTime") or observation.get("startedAt"),
        "ended_at": observation.get("endTime") or observation.get("endedAt"),
        "model": (
            observation.get("providedModelName")
            or observation.get("model")
            or observation.get("internalModelId")
        ),
        "usage": usage,
        "metadata": metadata,
        "input": observation.get("input"),
        "input_messages": extract_input_messages(observation),
        "output": observation.get("output"),
        "raw_observation": observation,
    }


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_sessions(
    *,
    session_roots: list[Path],
    output_dir: Path,
    host: str,
    public_key: str,
    secret_key: str,
    fields: str = DEFAULT_FIELDS,
    limit: int = 1000,
    timeout: float = 30.0,
    max_records: int | None = None,
    client: httpx.Client | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_at = _utc_now()
    descriptors = [session_descriptor(session_root) for session_root in session_roots]
    config_consistency: dict[str, dict[str, Any]] = {}
    for descriptor in descriptors:
        source_system = str(descriptor["source_system"])
        entry = config_consistency.setdefault(
            source_system,
            {"config_digests": [], "session_roots": [], "compatible": True},
        )
        entry["config_digests"].append(descriptor["config_digest"])
        entry["session_roots"].append(descriptor["session_root"])
    for entry in config_consistency.values():
        unique_digests = sorted(set(entry["config_digests"]))
        entry["config_digests"] = unique_digests
        entry["compatible"] = len(unique_digests) == 1
    incompatible = [
        source_system
        for source_system, entry in config_consistency.items()
        if not entry["compatible"]
    ]
    if incompatible:
        raise ValueError(
            "Source systems have incompatible multi-root configs: "
            + ", ".join(sorted(incompatible))
        )
    manifest: dict[str, Any] = {
        "exported_at": exported_at,
        "host": _normalize_host(host),
        "fields": fields,
        "config_consistency": config_consistency,
        "filters": {
            "actor": "overseer",
            "excluded_hooks": ["setup"],
            "excluded_trigger_types": sorted(EXCLUDED_TRIGGER_TYPES),
            "included_hooks": sorted(INCLUDED_HOOKS),
        },
        "git": _git_state(),
        "sessions": [],
    }

    for descriptor in tqdm(
        descriptors,
        desc="export sessions",
        unit="session",
        dynamic_ncols=True,
        disable=not progress,
    ):
        observations = fetch_observations(
            session_id=str(descriptor["langfuse_session_id"]),
            host=host,
            public_key=public_key,
            secret_key=secret_key,
            fields=fields,
            limit=limit,
            timeout=timeout,
            max_records=max_records,
            client=client,
            progress=progress,
        )

        selected: list[dict[str, Any]] = []
        excluded_oversize = 0
        max_raw_bytes = 0
        for observation in tqdm(
            observations,
            desc=f"filter {descriptor['langfuse_session_id']}",
            unit="obs",
            leave=False,
            dynamic_ncols=True,
            disable=not progress,
        ):
            raw_size = len(_json_bytes(observation))
            max_raw_bytes = max(max_raw_bytes, raw_size)
            if raw_size > MAX_OBSERVATION_BYTES:
                excluded_oversize += 1
                continue
            if is_runtime_or_final_overseer_observation(observation):
                selected.append(
                    normalize_observation(observation, descriptor=descriptor)
                )

        output_path = (
            output_dir
            / f"overseer_observations_{descriptor['langfuse_session_id']}.jsonl"
        )
        write_jsonl(output_path, selected)
        manifest["sessions"].append(
            {
                "source_system": descriptor["source_system"],
                "session_root": descriptor["session_root"],
                "langfuse_session_id": descriptor["langfuse_session_id"],
                "config_digest": descriptor["config_digest"],
                "raw_observation_count": len(observations),
                "exported_observation_count": len(selected),
                "excluded_oversize_count": excluded_oversize,
                "max_raw_observation_bytes": max_raw_bytes,
                "jsonl_path": str(output_path),
                "jsonl_sha256": _sha256_file(output_path),
                "jsonl_bytes": output_path.stat().st_size,
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parse_session_roots(values: list[str] | None) -> list[Path]:
    if not values:
        return list(SOURCE_SESSION_ROOTS)
    return [Path(value) for value in values]


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
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    fields: str = DEFAULT_FIELDS,
    host: str | None = None,
    public_key: str | None = None,
    secret_key: str | None = None,
    limit: int = 1000,
    timeout: float = 30.0,
    max_records: int | None = None,
    list_sessions: bool = False,
    progress: bool = True,
) -> None:
    """Export raw runtime/final overseer observations from Langfuse.

    Fire examples:
      pixi run python scripts/export_langfuse_overseer_observations.py --list-sessions
      pixi run python scripts/export_langfuse_overseer_observations.py \
        --session-root='["outputs/.../session-a","outputs/.../session-b"]'
    """
    load_dotenv()
    selected_session_roots = _coerce_path_values(session_roots) + _coerce_path_values(
        session_root
    )
    selected_session_roots = _parse_session_roots(
        [str(path) for path in selected_session_roots] or None
    )
    if list_sessions:
        print(
            json.dumps(
                [
                    session_descriptor(session_root)
                    for session_root in selected_session_roots
                ],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return
    try:
        manifest = export_sessions(
            session_roots=selected_session_roots,
            output_dir=Path(output_dir),
            host=host or os.getenv("LANGFUSE_HOST", DEFAULT_LANGFUSE_HOST),
            public_key=public_key or os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=secret_key or os.getenv("LANGFUSE_SECRET_KEY", ""),
            fields=fields,
            limit=int(limit),
            timeout=float(timeout),
            max_records=max_records,
            progress=progress,
        )
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    count = sum(
        int(session["exported_observation_count"]) for session in manifest["sessions"]
    )
    print(f"Wrote {count} overseer observations to {Path(output_dir)}")


if __name__ == "__main__":
    fire.Fire(main)
