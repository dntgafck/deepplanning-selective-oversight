from __future__ import annotations

from typing import Any

__all__ = ["StructuredLogger", "SystemConfig", "build_system_config", "run_experiment"]


def __getattr__(name: str) -> Any:
    if name == "StructuredLogger":
        from .logging import StructuredLogger

        return StructuredLogger
    if name in {"SystemConfig", "build_system_config"}:
        from .config import SystemConfig, build_system_config

        if name == "SystemConfig":
            return SystemConfig
        return build_system_config
    if name == "run_experiment":
        from .runner import run_experiment

        return run_experiment
    raise AttributeError(f"module 'experiment' has no attribute {name!r}")
