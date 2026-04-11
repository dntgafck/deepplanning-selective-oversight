from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf, open_dict

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = REPO_ROOT / "external" / "qwen-agent" / "benchmark" / "deepplanning"
DATA_ROOT = REPO_ROOT / "data" / "deepplanning"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "deepplanning"
CONFIG_ROOT = REPO_ROOT / "configs"
MODELS_CONFIG_PATH = CONFIG_ROOT / "models.yaml"
DOTENV_PATH = REPO_ROOT / ".env"

PUBLIC_EXPERIMENT_OVERRIDE = "experiment"
HYDRA_EXPERIMENT_GROUP = "experiments"


def ensure_repo_imports() -> None:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_repo_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return REPO_ROOT / path_obj


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


def parse_space_separated(
    value: str | Sequence[str] | None,
    default: list[str],
) -> list[str]:
    if value is None:
        return default
    if isinstance(value, Sequence) and not isinstance(value, str):
        items = [str(item) for item in value if str(item).strip()]
        return items or default
    items = [item for item in str(value).split() if item]
    return items or default


def parse_int_list(
    value: int | str | Sequence[int | str] | None,
    default: list[int],
) -> list[int]:
    if value is None:
        return default
    if isinstance(value, int):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [int(item) for item in value] or default
    items = [item for item in str(value).split() if item]
    return [int(item) for item in items] or default


def parse_id_list(
    value: int | str | Sequence[int | str] | None,
) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, int):
        return [str(value)]
    if isinstance(value, Sequence) and not isinstance(value, str):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items or None

    items = [item for item in re.split(r"[\s,]+", str(value).strip()) if item]
    return items or None


def filter_samples_by_ids(
    samples: list[dict[str, Any]], sample_ids: Iterable[str] | None
) -> list[dict[str, Any]]:
    if sample_ids is None:
        return samples

    sample_id_set = {str(sample_id) for sample_id in sample_ids}
    return [sample for sample in samples if str(sample.get("id")) in sample_id_set]


def load_model_config(
    model_name: str, config_path: Path = MODELS_CONFIG_PATH
) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Hydra model config not found at {config_path}. "
            "Keep benchmark model definitions under configs/."
        )

    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

    models = config.get("models", {})
    if model_name not in models:
        available = ", ".join(sorted(models))
        raise ValueError(
            f"Model '{model_name}' not found in {config_path}. Available models: {available}"
        )

    return dict(models[model_name])


def available_system_names() -> list[str]:
    system_root = CONFIG_ROOT / "system"
    if not system_root.exists():
        return []
    return sorted(path.stem for path in system_root.glob("*.yaml"))


def load_system_defaults(system_name: str) -> dict[str, Any]:
    system_path = CONFIG_ROOT / "system" / f"{system_name}.yaml"
    if not system_path.exists():
        available = ", ".join(available_system_names())
        raise ValueError(f"Unknown system '{system_name}'. Available: {available}")

    payload = OmegaConf.to_container(OmegaConf.load(system_path), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"System config must be a mapping: {system_path}")
    return payload


def _normalize_public_override(override: str) -> str:
    key, separator, value = override.partition("=")
    if separator and key == PUBLIC_EXPERIMENT_OVERRIDE:
        return f"{HYDRA_EXPERIMENT_GROUP}={value}"
    return override


def normalize_public_overrides(overrides: Sequence[str] | None) -> list[str]:
    if not overrides:
        return []
    return [_normalize_public_override(override) for override in overrides]


def extract_experiment_key(overrides: Sequence[str] | None) -> str | None:
    for override in overrides or []:
        key, separator, value = override.partition("=")
        if separator and key in {PUBLIC_EXPERIMENT_OVERRIDE, HYDRA_EXPERIMENT_GROUP}:
            selected = value.strip().strip('"').strip("'")
            return selected or None
    return None


def compose_config(
    config_name: str = "experiment",
    overrides: Sequence[str] | None = None,
) -> DictConfig:
    normalized_overrides = normalize_public_overrides(overrides)
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=normalized_overrides)

    model_registry = OmegaConf.load(MODELS_CONFIG_PATH)
    with open_dict(cfg):
        cfg.model_registry = OmegaConf.to_container(
            model_registry.get("models", {}),
            resolve=True,
        )
    return cfg


def persist_config_yaml(cfg: DictConfig, output_path: Path) -> None:
    output_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")


def persist_overrides(overrides: Sequence[str], output_path: Path) -> None:
    text = "\n".join(overrides) if overrides else "# No overrides"
    output_path.write_text(text + "\n", encoding="utf-8")


def hydra_value(value: Any, *, quote_strings: bool = True) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(hydra_value(item) for item in value) + "]"
    text = str(value)
    return json.dumps(text) if quote_strings else text


def normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if " " in text:
        return [item for item in text.split() if item]
    return [text]


def normalize_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [int(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    if " " in text:
        return [int(item) for item in text.split() if item]
    return [int(text)]


def named_experiment_path(experiment_key: str | None) -> Path | None:
    if not experiment_key:
        return None
    return CONFIG_ROOT / HYDRA_EXPERIMENT_GROUP / f"{experiment_key}.yaml"
