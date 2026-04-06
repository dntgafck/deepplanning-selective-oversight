from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = REPO_ROOT / "external" / "qwen-agent" / "benchmark" / "deepplanning"
DATA_ROOT = REPO_ROOT / "data" / "deepplanning"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "deepplanning"
CONFIG_ROOT = REPO_ROOT / "conf" / "deepplanning"
MODELS_CONFIG_PATH = REPO_ROOT / "models_config.json"
DOTENV_PATH = REPO_ROOT / ".env"


def load_dotenv(path: Path = DOTENV_PATH) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_model_config(
    model_name: str, config_path: Path = MODELS_CONFIG_PATH
) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(
            f"models_config.json not found at {config_path}. "
            "Keep benchmark config at the repo root to avoid modifying the submodule."
        )

    with config_path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)

    models = config.get("models", {})
    if model_name not in models:
        available = ", ".join(sorted(models))
        raise ValueError(
            f"Model '{model_name}' not found in {config_path}. Available models: {available}"
        )

    return models[model_name]


def compose_config(
    config_name: str, overrides: dict[str, Any] | None = None
) -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
        cfg = compose(config_name=config_name)

    if overrides:
        filtered = {key: value for key, value in overrides.items() if value is not None}
        if filtered:
            cfg = OmegaConf.merge(cfg, OmegaConf.create(filtered))

    return cfg


def parse_space_separated(
    value: str | list[str] | None, default: list[str]
) -> list[str]:
    if value is None:
        return default
    if isinstance(value, list):
        return value or default
    items = [item for item in value.split() if item]
    return items or default


def parse_int_list(
    value: str | list[int] | list[str] | None, default: list[int]
) -> list[int]:
    if value is None:
        return default
    if isinstance(value, list):
        return [int(item) for item in value] or default
    items = [item for item in value.split() if item]
    return [int(item) for item in items] or default


def resolve_repo_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return REPO_ROOT / path_obj


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
