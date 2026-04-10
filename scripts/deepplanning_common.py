from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = REPO_ROOT / "external" / "qwen-agent" / "benchmark" / "deepplanning"
DATA_ROOT = REPO_ROOT / "data" / "deepplanning"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "deepplanning"
CONFIG_ROOT = REPO_ROOT / "conf" / "deepplanning"
MODELS_CONFIG_PATH = CONFIG_ROOT / "models.yaml"
DOTENV_PATH = REPO_ROOT / ".env"


def ensure_repo_imports() -> None:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


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
            f"Hydra model config not found at {config_path}. "
            "Keep benchmark model definitions under conf/deepplanning/."
        )

    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

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
    value: int | str | list[int] | list[str] | None, default: list[int]
) -> list[int]:
    if value is None:
        return default
    if isinstance(value, int):
        return [value]
    if isinstance(value, list):
        return [int(item) for item in value] or default
    items = [item for item in value.split() if item]
    return [int(item) for item in items] or default


def parse_id_list(
    value: int | str | list[int] | list[str] | None,
) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, int):
        return [str(value)]
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items or None

    items = [item for item in re.split(r"[\s,]+", value.strip()) if item]
    return items or None


def filter_samples_by_ids(
    samples: list[dict[str, Any]], sample_ids: Iterable[str] | None
) -> list[dict[str, Any]]:
    if sample_ids is None:
        return samples

    sample_id_set = {str(sample_id) for sample_id in sample_ids}
    return [sample for sample in samples if str(sample.get("id")) in sample_id_set]


def load_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json_file(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def clear_imported_modules(prefixes: Iterable[str]) -> None:
    for module_name in list(sys.modules):
        if any(
            module_name == prefix or module_name.startswith(f"{prefix}.")
            for prefix in prefixes
        ):
            sys.modules.pop(module_name, None)


def resolve_repo_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return REPO_ROOT / path_obj


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
