from __future__ import annotations

from typing import Any

from .base import OversightController
from .controllers import (
    AdaptiveRiskOversight,
    CheckpointReviewOversight,
    ContinuousReviewOversight,
    ExecutorOnlyOversight,
)

_PROFILE_BY_MODE = {
    "disabled": "executor_only",
    "always": "continuous_review",
    "checkpoint": "checkpoint_review",
    "adaptive": "adaptive_risk",
}
_CONTROLLER_BY_PROFILE: dict[str, type[OversightController]] = {
    "executor_only": ExecutorOnlyOversight,
    "continuous_review": ContinuousReviewOversight,
    "checkpoint_review": CheckpointReviewOversight,
    "adaptive_risk": AdaptiveRiskOversight,
}
_VALID_PROFILES = frozenset(_CONTROLLER_BY_PROFILE)
_MODE_BY_PROFILE = {profile: mode for mode, profile in _PROFILE_BY_MODE.items()}


def resolve_oversight_profile(system_config: Any) -> str:
    explicit_profile = getattr(system_config, "oversight_profile", None)
    if explicit_profile is not None:
        normalized = str(explicit_profile).strip()
        if normalized not in _VALID_PROFILES:
            available = ", ".join(sorted(_VALID_PROFILES))
            raise ValueError(
                f"Unknown oversight_profile '{normalized}'. Available: {available}"
            )
        return normalized

    if not bool(getattr(system_config, "oversight_enabled", False)):
        return "executor_only"

    oversight_mode = str(getattr(system_config, "oversight_mode", "disabled"))
    if oversight_mode in _PROFILE_BY_MODE:
        return _PROFILE_BY_MODE[oversight_mode]

    available = ", ".join(sorted(_PROFILE_BY_MODE))
    raise ValueError(
        f"Cannot resolve oversight profile from oversight_mode '{oversight_mode}'. "
        f"Available modes: {available}"
    )


def resolve_oversight_mode_alias(
    system_config: Any, *, profile: str | None = None
) -> str:
    resolved_profile = profile or resolve_oversight_profile(system_config)
    explicit_mode = getattr(system_config, "oversight_mode", None)
    if explicit_mode is None:
        return _MODE_BY_PROFILE[resolved_profile]

    normalized = str(explicit_mode).strip()
    if normalized not in _PROFILE_BY_MODE:
        available = ", ".join(sorted(_PROFILE_BY_MODE))
        raise ValueError(
            f"Cannot resolve oversight mode alias from '{normalized}'. "
            f"Available modes: {available}"
        )

    expected_profile = _PROFILE_BY_MODE[normalized]
    if expected_profile != resolved_profile:
        raise ValueError(
            "oversight_mode "
            f"'{normalized}' does not match oversight_profile '{resolved_profile}'."
        )
    return normalized


def build_oversight_controller(
    system_config: Any,
) -> OversightController:
    profile = resolve_oversight_profile(system_config)
    system_name = str(getattr(system_config, "name", ""))
    oversight_mode = resolve_oversight_mode_alias(
        system_config,
        profile=profile,
    )
    controller_cls = _CONTROLLER_BY_PROFILE[profile]
    return controller_cls(
        system_name=system_name,
        oversight_mode=oversight_mode,
    )
