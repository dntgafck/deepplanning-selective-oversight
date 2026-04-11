from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import fire
from omegaconf import OmegaConf

try:
    from deepplanning_common import (
        CONFIG_ROOT,
        compose_config,
        ensure_directory,
        resolve_repo_path,
    )
    from run_deepplanning_benchmark import run as run_benchmark
except ModuleNotFoundError:
    from scripts.deepplanning_common import (
        CONFIG_ROOT,
        compose_config,
        ensure_directory,
        resolve_repo_path,
    )
    from scripts.run_deepplanning_benchmark import run as run_benchmark

EXPERIMENTS_CONFIG_ROOT = CONFIG_ROOT / "experiments"
DEFAULT_OUTPUT_ROOT = "outputs/deepplanning/experiments"
DEFAULT_METADATA_FILENAME = "experiment_session.json"


def _has_value(value: object | None) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


def _session_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _space_join(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value if _has_value(item))
    return str(value)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-")
    if not cleaned:
        raise ValueError(
            "Experiment name must contain at least one path-safe character."
        )
    return cleaned


def _load_named_experiment_config(
    experiment: str | None,
) -> tuple[dict[str, Any], Path | None]:
    if not _has_value(experiment):
        return {}, None

    config_path = EXPERIMENTS_CONFIG_ROOT / f"{experiment}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")

    data = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"Experiment config must be a mapping: {config_path}")
    return data, config_path


def _normalize_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(overrides)
    parallel = normalized.pop("parallel", None)
    sample_ids = normalized.pop("sample_ids", None)
    domain = normalized.pop("domain", None)
    domains = normalized.get("domains")

    if not _has_value(domains) and _has_value(domain):
        normalized["domains"] = domain
    if not _has_value(normalized.get("workers")) and _has_value(parallel):
        normalized["workers"] = parallel
    if _has_value(sample_ids):
        if not _has_value(normalized.get("travel_sample_ids")):
            normalized["travel_sample_ids"] = sample_ids
        if not _has_value(normalized.get("shopping_sample_ids")):
            normalized["shopping_sample_ids"] = sample_ids

    return {key: value for key, value in normalized.items() if value is not None}


def _compose_experiment_config(
    experiment: str | None,
    overrides: dict[str, Any],
) -> tuple[Any, Path | None]:
    base_cfg = compose_config("experiment")
    experiment_overrides, config_path = _load_named_experiment_config(experiment)
    merged = OmegaConf.merge(
        base_cfg,
        OmegaConf.create(experiment_overrides),
        OmegaConf.create(_normalize_overrides(overrides)),
    )
    return merged, config_path


def _required_name(cfg: Any, experiment: str | None) -> str:
    name = str(getattr(cfg, "name", "") or "").strip()
    if not name:
        if _has_value(experiment):
            raise ValueError(
                f"Experiment config '{experiment}' must define an explicit non-empty 'name'."
            )
        raise ValueError("Experiment runs require an explicit 'name'.")
    return name


def _command_preview(parameters: dict[str, Any]) -> list[str]:
    command = ["pixi", "run", "deepplanning-benchmark", "--"]
    for key, value in parameters.items():
        if not _has_value(value):
            continue
        command.append(f"--{key}={value}")
    return command


def _write_session_metadata(
    *,
    metadata_path: Path,
    experiment_key: str | None,
    experiment_name: str,
    config_path: Path | None,
    timestamp: str,
    session_root: Path,
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
        "parameters": parameters,
        "launched_command": _command_preview(parameters),
    }
    metadata_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run(
    experiment: str | None = None,
    name: str | None = None,
    domain: str | None = None,
    domains: str | None = None,
    models: str | None = None,
    system: str | None = None,
    parallel: int | None = None,
    workers: int | None = None,
    max_llm_calls: int | None = None,
    runs: int | None = None,
    shopping_levels: str | None = None,
    shopping_sample_ids: str | None = None,
    travel_language: str | None = None,
    travel_start_from: str | None = None,
    travel_evaluation_mode: str | None = None,
    travel_sample_ids: str | None = None,
    sample_ids: str | None = None,
    output_root: str | None = None,
    session_root: str | None = None,
) -> None:
    cfg, config_path = _compose_experiment_config(
        experiment,
        {
            "name": name,
            "domain": domain,
            "domains": domains,
            "models": models,
            "system": system,
            "parallel": parallel,
            "workers": workers,
            "max_llm_calls": max_llm_calls,
            "runs": runs,
            "shopping_levels": shopping_levels,
            "shopping_sample_ids": shopping_sample_ids,
            "travel_language": travel_language,
            "travel_start_from": travel_start_from,
            "travel_evaluation_mode": travel_evaluation_mode,
            "travel_sample_ids": travel_sample_ids,
            "sample_ids": sample_ids,
            "output_root": output_root,
            "session_root": session_root,
        },
    )

    experiment_name = _required_name(cfg, experiment)
    timestamp_value = _session_timestamp()
    configured_session_root = str(getattr(cfg, "session_root", "") or "").strip()
    if configured_session_root:
        resolved_session_root = resolve_repo_path(configured_session_root)
    else:
        root = resolve_repo_path(
            str(getattr(cfg, "output_root", DEFAULT_OUTPUT_ROOT) or DEFAULT_OUTPUT_ROOT)
        )
        resolved_session_root = root / _safe_name(experiment_name) / timestamp_value

    session_dir = ensure_directory(resolved_session_root)
    metadata_filename = str(
        getattr(cfg, "metadata_filename", DEFAULT_METADATA_FILENAME)
        or DEFAULT_METADATA_FILENAME
    )

    normalized_parameters = {
        "domains": _space_join(
            getattr(cfg, "domains", "") or getattr(cfg, "domain", "")
        ),
        "models": _space_join(cfg.models),
        "system": str(cfg.system),
        "workers": int(getattr(cfg, "workers", 0) or getattr(cfg, "parallel", 0)),
        "max_llm_calls": int(cfg.max_llm_calls),
        "runs": int(getattr(cfg, "runs", 1)),
        "shopping_levels": _space_join(getattr(cfg, "shopping_levels", "") or ""),
        "shopping_sample_ids": _space_join(
            getattr(cfg, "shopping_sample_ids", "") or ""
        ),
        "travel_language": _space_join(getattr(cfg, "travel_language", "") or ""),
        "travel_start_from": _space_join(getattr(cfg, "travel_start_from", "") or ""),
        "travel_evaluation_mode": _space_join(
            getattr(cfg, "travel_evaluation_mode", "") or ""
        ),
        "travel_sample_ids": _space_join(getattr(cfg, "travel_sample_ids", "") or ""),
        "output_root": str(session_dir),
    }

    _write_session_metadata(
        metadata_path=session_dir / metadata_filename,
        experiment_key=experiment,
        experiment_name=experiment_name,
        config_path=config_path,
        timestamp=timestamp_value,
        session_root=session_dir,
        parameters=normalized_parameters,
    )

    run_benchmark(
        domains=normalized_parameters["domains"],
        models=cfg.models,
        system=str(cfg.system),
        shopping_levels=getattr(cfg, "shopping_levels", None),
        shopping_sample_ids=getattr(cfg, "shopping_sample_ids", None),
        workers=int(normalized_parameters["workers"]),
        max_llm_calls=int(cfg.max_llm_calls),
        runs=int(getattr(cfg, "runs", 1)),
        travel_language=getattr(cfg, "travel_language", None),
        travel_start_from=getattr(cfg, "travel_start_from", None),
        travel_evaluation_mode=getattr(cfg, "travel_evaluation_mode", None),
        travel_sample_ids=getattr(cfg, "travel_sample_ids", None),
        output_root=str(session_dir),
    )


if __name__ == "__main__":
    fire.Fire(run)
