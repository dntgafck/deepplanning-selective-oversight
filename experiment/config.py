from __future__ import annotations

from dataclasses import dataclass

from deepplanning.config import available_system_names, load_system_defaults
from llm import ProviderConfig


@dataclass(slots=True)
class SystemConfig:
    name: str
    executor_provider: ProviderConfig
    oversight_enabled: bool
    oversight_mode: str
    overseer_provider: ProviderConfig | None = None
    overseer_thinking: bool | None = None
    max_steps: int = 400
    num_runs: int = 1


def build_system_config(
    system_name: str,
    executor_model: str,
    overseer_model: str = "deepseek-v3.2",
    max_steps: int = 400,
    num_runs: int = 1,
) -> SystemConfig:
    try:
        defaults = load_system_defaults(system_name)
    except ValueError as exc:
        available = ", ".join(available_system_names())
        raise ValueError(
            f"Unknown system '{system_name}'. Available: {available}"
        ) from exc

    oversight_enabled = bool(defaults["oversight_enabled"])
    overseer_provider = None
    if oversight_enabled:
        overseer_provider = ProviderConfig.from_model_name(overseer_model)

    return SystemConfig(
        name=str(defaults["name"]),
        executor_provider=ProviderConfig.from_model_name(executor_model),
        oversight_enabled=oversight_enabled,
        oversight_mode=str(defaults["oversight_mode"]),
        overseer_provider=overseer_provider,
        overseer_thinking=defaults.get("overseer_thinking"),
        max_steps=max_steps,
        num_runs=num_runs,
    )
