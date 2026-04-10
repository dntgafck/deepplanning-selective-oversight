from __future__ import annotations

from dataclasses import dataclass

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


SYSTEM_DEFAULTS = {
    "A": {
        "oversight_enabled": False,
        "oversight_mode": "disabled",
        "overseer_thinking": None,
    },
    "B": {
        "oversight_enabled": True,
        "oversight_mode": "always",
        "overseer_thinking": True,
    },
    "C1": {
        "oversight_enabled": True,
        "oversight_mode": "checkpoint",
        "overseer_thinking": True,
    },
    "C2": {
        "oversight_enabled": True,
        "oversight_mode": "adaptive",
        "overseer_thinking": True,
    },
    "C2-nt": {
        "oversight_enabled": True,
        "oversight_mode": "adaptive",
        "overseer_thinking": False,
    },
}


def build_system_config(
    system_name: str,
    executor_model: str,
    overseer_model: str = "deepseek-v3.2",
    max_steps: int = 400,
    num_runs: int = 1,
) -> SystemConfig:
    if system_name not in SYSTEM_DEFAULTS:
        available = ", ".join(sorted(SYSTEM_DEFAULTS))
        raise ValueError(f"Unknown system '{system_name}'. Available: {available}")

    defaults = SYSTEM_DEFAULTS[system_name]
    overseer_provider = None
    if defaults["oversight_enabled"]:
        overseer_provider = ProviderConfig.from_model_name(overseer_model)

    return SystemConfig(
        name=system_name,
        executor_provider=ProviderConfig.from_model_name(executor_model),
        oversight_enabled=bool(defaults["oversight_enabled"]),
        oversight_mode=str(defaults["oversight_mode"]),
        overseer_provider=overseer_provider,
        overseer_thinking=defaults["overseer_thinking"],
        max_steps=max_steps,
        num_runs=num_runs,
    )
