from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .aggregation import aggregate_results
from .config import (
    compose_config,
    ensure_directory,
    extract_experiment_key,
    hydra_value,
    named_experiment_path,
    normalize_int_list,
    normalize_public_overrides,
    normalize_string_list,
    persist_config_yaml,
    persist_overrides,
    resolve_repo_path,
)
from .shopping_runner import run as run_shopping
from .travel_runner import run as run_travel

DEFAULT_OUTPUT_ROOT = "outputs/deepplanning/experiments"
DEFAULT_METADATA_FILENAME = "experiment_session.json"
MAX_LANGFUSE_SESSION_ID_LENGTH = 200


def _has_value(value: object | None) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


def _session_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-")
    if not cleaned:
        raise ValueError(
            "Experiment name must contain at least one path-safe character."
        )
    return cleaned


def _build_langfuse_session_id(session_root: Path) -> str:
    session_id = _safe_name(session_root.name)
    truncated = session_id[:MAX_LANGFUSE_SESSION_ID_LENGTH].rstrip("-_.")
    return truncated or "session"


def _legacy_kwargs_to_public_overrides(kwargs: dict[str, Any]) -> list[str]:
    if not kwargs:
        return []

    resolved = dict(kwargs)
    overrides: list[str] = []

    experiment = resolved.pop("experiment", None)
    if _has_value(experiment):
        overrides.append(f"experiment={experiment}")

    name = resolved.pop("name", None)
    if _has_value(name):
        overrides.append(f"name={hydra_value(name)}")

    domain = resolved.pop("domain", None)
    domains = resolved.pop("domains", None)
    domain_value = domains if _has_value(domains) else domain
    if _has_value(domain_value):
        overrides.append(
            f"domains={hydra_value(normalize_string_list(domain_value) or [str(domain_value)])}"
        )

    models = resolved.pop("models", None)
    if _has_value(models):
        selected_models = normalize_string_list(models) or [str(models)]
        model_value: Any = (
            selected_models[0] if len(selected_models) == 1 else selected_models
        )
        overrides.append(f"models.executor={hydra_value(model_value)}")

    system = resolved.pop("system", None)
    if _has_value(system):
        overrides.append(f"system={hydra_value(system, quote_strings=False)}")

    parallel = resolved.pop("parallel", None)
    workers = resolved.pop("workers", None)
    runtime_workers = workers if _has_value(workers) else parallel
    if _has_value(runtime_workers):
        overrides.append(f"runtime.workers={hydra_value(int(runtime_workers))}")

    max_llm_calls = resolved.pop("max_llm_calls", None)
    if _has_value(max_llm_calls):
        overrides.append(f"runtime.max_llm_calls={hydra_value(int(max_llm_calls))}")

    runs = resolved.pop("runs", None)
    if _has_value(runs):
        overrides.append(f"runtime.runs={hydra_value(int(runs))}")

    shopping_levels = resolved.pop("shopping_levels", None)
    if _has_value(shopping_levels):
        overrides.append(
            f"shopping.levels={hydra_value(normalize_int_list(shopping_levels))}"
        )

    sample_ids = resolved.pop("sample_ids", None)
    shopping_sample_ids = resolved.pop("shopping_sample_ids", None)
    travel_sample_ids = resolved.pop("travel_sample_ids", None)
    if _has_value(sample_ids) and not _has_value(shopping_sample_ids):
        shopping_sample_ids = sample_ids
    if _has_value(sample_ids) and not _has_value(travel_sample_ids):
        travel_sample_ids = sample_ids

    if _has_value(shopping_sample_ids):
        overrides.append(
            f"shopping.sample_ids={hydra_value(normalize_string_list(shopping_sample_ids))}"
        )

    travel_language = resolved.pop("travel_language", None)
    if _has_value(travel_language):
        overrides.append(f"travel.language={hydra_value(travel_language)}")

    travel_start_from = resolved.pop("travel_start_from", None)
    if _has_value(travel_start_from):
        overrides.append(f"travel.start_from={hydra_value(travel_start_from)}")

    travel_evaluation_mode = resolved.pop("travel_evaluation_mode", None)
    if _has_value(travel_evaluation_mode):
        overrides.append(
            f"travel.evaluation_mode={hydra_value(travel_evaluation_mode)}"
        )

    if _has_value(travel_sample_ids):
        overrides.append(
            f"travel.sample_ids={hydra_value(normalize_string_list(travel_sample_ids))}"
        )

    travel_verbose = resolved.pop("travel_verbose", None)
    if _has_value(travel_verbose):
        overrides.append(f"travel.verbose={hydra_value(bool(travel_verbose))}")

    travel_debug = resolved.pop("travel_debug", None)
    if _has_value(travel_debug):
        overrides.append(f"travel.debug={hydra_value(bool(travel_debug))}")

    output_root = resolved.pop("output_root", None)
    if _has_value(output_root):
        overrides.append(f"output_root={hydra_value(output_root)}")

    session_root = resolved.pop("session_root", None)
    if _has_value(session_root):
        overrides.append(f"session_root={hydra_value(session_root)}")

    metadata_filename = resolved.pop("metadata_filename", None)
    if _has_value(metadata_filename):
        overrides.append(f"metadata_filename={hydra_value(metadata_filename)}")

    return overrides


def _public_command_preview(overrides: list[str]) -> list[str]:
    command = ["pixi", "run", "deepplanning-experiment", "--"]
    command.extend(overrides)
    return command


def _require_name(cfg: Any, experiment_key: str | None) -> str:
    name = str(getattr(cfg, "name", "") or "").strip()
    if not name:
        if experiment_key:
            raise ValueError(
                f"Experiment config '{experiment_key}' must define an explicit non-empty 'name'."
            )
        raise ValueError("Experiment runs require an explicit 'name'.")
    return name


def _resolve_session_root(cfg: Any, timestamp: str, experiment_name: str) -> Path:
    configured_session_root = str(getattr(cfg, "session_root", "") or "").strip()
    if configured_session_root:
        return ensure_directory(resolve_repo_path(configured_session_root))

    output_root = resolve_repo_path(
        str(getattr(cfg, "output_root", DEFAULT_OUTPUT_ROOT) or DEFAULT_OUTPUT_ROOT)
    )
    return ensure_directory(output_root / _safe_name(experiment_name) / timestamp)


def _normalized_parameters(cfg: Any, session_root: Path) -> dict[str, Any]:
    return {
        "name": str(cfg.name),
        "domains": list(OmegaConf.to_container(cfg.domains, resolve=True) or []),
        "models": OmegaConf.to_container(cfg.models, resolve=True),
        "system": OmegaConf.to_container(cfg.system, resolve=True),
        "runtime": OmegaConf.to_container(cfg.runtime, resolve=True),
        "shopping": OmegaConf.to_container(cfg.shopping, resolve=True),
        "travel": OmegaConf.to_container(cfg.travel, resolve=True),
        "session_root": str(session_root),
        "output_root": str(session_root),
    }


def _write_session_metadata(
    *,
    metadata_path: Path,
    experiment_key: str | None,
    experiment_name: str,
    config_path: Path | None,
    timestamp: str,
    session_root: Path,
    raw_overrides: list[str],
    normalized_overrides: list[str],
    parameters: dict[str, Any],
) -> None:
    payload = {
        "experiment": {
            "key": experiment_key,
            "name": experiment_name,
            "config_path": str(config_path) if config_path is not None else None,
        },
        "timestamp": timestamp,
        "session_root": str(session_root),
        "artifacts": {
            "config_yaml": str(session_root / "config.yaml"),
            "overrides_txt": str(session_root / "overrides.txt"),
        },
        "overrides": {
            "raw": raw_overrides,
            "normalized": normalized_overrides,
        },
        "parameters": parameters,
        "launched_command": _public_command_preview(raw_overrides),
    }
    metadata_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _executor_models(cfg: Any) -> list[str]:
    value = (
        OmegaConf.to_container(cfg.models.executor, resolve=True)
        if OmegaConf.is_config(cfg.models.executor)
        else cfg.models.executor
    )
    return normalize_string_list(value) or ["qwen3-14b"]


def _domain_names(cfg: Any) -> list[str]:
    value = (
        OmegaConf.to_container(cfg.domains, resolve=True)
        if OmegaConf.is_config(cfg.domains)
        else cfg.domains
    )
    return normalize_string_list(value) or ["travel", "shopping"]


def run_benchmark_from_cfg(cfg: Any, benchmark_output_root: Path) -> None:
    output_root = ensure_directory(benchmark_output_root)
    runtime_cfg = OmegaConf.to_container(cfg.runtime, resolve=True) or {}
    shopping_cfg = OmegaConf.to_container(cfg.shopping, resolve=True) or {}
    travel_cfg = OmegaConf.to_container(cfg.travel, resolve=True) or {}
    system_name = str(cfg.system.name)
    langfuse_session_id = _build_langfuse_session_id(output_root)

    domain_names = _domain_names(cfg)
    executor_models = _executor_models(cfg)

    if "shopping" in domain_names:
        run_shopping(
            models=executor_models,
            levels=shopping_cfg.get("levels"),
            sample_ids=shopping_cfg.get("sample_ids"),
            system=system_name,
            workers=int(runtime_cfg.get("workers", 20)),
            max_llm_calls=int(runtime_cfg.get("max_llm_calls", 400)),
            runs=int(runtime_cfg.get("runs", 1)),
            output_root=output_root / "shopping",
            langfuse_session_id=langfuse_session_id,
        )

    if "travel" in domain_names:
        run_travel(
            models=executor_models,
            language=travel_cfg.get("language"),
            sample_ids=travel_cfg.get("sample_ids"),
            system=system_name,
            workers=int(runtime_cfg.get("workers", 20)),
            max_llm_calls=int(runtime_cfg.get("max_llm_calls", 400)),
            runs=int(runtime_cfg.get("runs", 1)),
            start_from=str(travel_cfg.get("start_from", "inference")),
            evaluation_mode=str(travel_cfg.get("evaluation_mode", "auto")),
            output_root=output_root / "travel",
            verbose=bool(travel_cfg.get("verbose", False)),
            debug=bool(travel_cfg.get("debug", False)),
            langfuse_session_id=langfuse_session_id,
        )

    for model in executor_models:
        aggregate_results(model, benchmark_output_root=output_root)


def run(*overrides: str, **legacy_kwargs: Any) -> Path:
    raw_overrides = list(overrides)
    if legacy_kwargs:
        raw_overrides.extend(_legacy_kwargs_to_public_overrides(legacy_kwargs))

    normalized_overrides = normalize_public_overrides(raw_overrides)
    experiment_key = extract_experiment_key(raw_overrides)
    cfg = compose_config("experiment", raw_overrides)
    experiment_name = _require_name(cfg, experiment_key)
    timestamp = _session_timestamp()
    session_root = _resolve_session_root(cfg, timestamp, experiment_name)

    persist_config_yaml(cfg, session_root / "config.yaml")
    persist_overrides(raw_overrides, session_root / "overrides.txt")

    metadata_filename = str(
        getattr(cfg, "metadata_filename", DEFAULT_METADATA_FILENAME)
        or DEFAULT_METADATA_FILENAME
    )
    _write_session_metadata(
        metadata_path=session_root / metadata_filename,
        experiment_key=experiment_key,
        experiment_name=experiment_name,
        config_path=named_experiment_path(experiment_key),
        timestamp=timestamp,
        session_root=session_root,
        raw_overrides=raw_overrides,
        normalized_overrides=normalized_overrides,
        parameters=_normalized_parameters(cfg, session_root),
    )

    run_benchmark_from_cfg(cfg, session_root)
    return session_root


def main(argv: list[str] | None = None) -> None:
    run(*(argv if argv is not None else sys.argv[1:]))
