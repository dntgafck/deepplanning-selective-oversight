from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from deepplanning.config import available_system_names, load_system_defaults
from llm import ProviderConfig

_OVERSIGHT_PROFILE_BY_MODE = {
    "disabled": "executor_only",
    "always": "continuous_review",
    "checkpoint": "checkpoint_review",
    "adaptive": "adaptive_risk",
}
_OVERSIGHT_MODE_BY_PROFILE = {
    profile: mode for mode, profile in _OVERSIGHT_PROFILE_BY_MODE.items()
}
_VALID_OVERSIGHT_PROFILES = frozenset(_OVERSIGHT_MODE_BY_PROFILE)
_VALID_OVERSIGHT_MODES = frozenset(_OVERSIGHT_PROFILE_BY_MODE)


@dataclass(slots=True)
class SystemConfig:
    name: str
    executor_provider: ProviderConfig
    oversight_enabled: bool
    oversight_mode: str
    oversight_profile: str
    overseer_provider: ProviderConfig | None = None
    overseer_thinking: bool | None = None
    max_steps: int = 400
    num_runs: int = 1
    overseer_prompt_version: str = "c2-lite-v1.3"
    loop_similarity_threshold: float = 0.92
    loop_window: int = 5
    loop_repeat_count: int = 3
    coverage_threshold: float = 0.50
    final_repair_retry_cap: int = 2
    max_stale_cart_notices: int = 1
    recent_tool_window: int = 5
    inject_transient_notice: bool = True
    mutating_tools: tuple[str, ...] = (
        "add_product_to_cart",
        "delete_product_from_cart",
        "add_coupon_to_cart",
        "delete_coupon_from_cart",
    )
    irreversible_tools: tuple[str, ...] = ()
    block_on_mutation_mode: str = "auto"
    max_hard_blocks_per_args: int = 2
    max_consecutive_pre_tool_blocks: int = 5
    require_cited_violation_for_block: bool = True
    overseer_call_budget_per_task: int = 8


def provider_identity_payload(
    provider: ProviderConfig | None,
    *,
    reasoning_enabled: bool | None = None,
) -> dict[str, Any] | None:
    if provider is None:
        return None
    resolved_reasoning_enabled = reasoning_enabled
    if resolved_reasoning_enabled is None:
        resolved_reasoning_enabled = provider.configured_reasoning_enabled()
    return {
        "requested_model": provider.alias,
        "resolved_provider": provider.provider,
        "resolved_model": provider.model,
        "sampling": {
            "temperature": provider.temperature,
            "top_p": provider.top_p,
            "seed": provider.seed,
            "reasoning_enabled": resolved_reasoning_enabled,
        },
    }


def system_model_identities(system_config: SystemConfig) -> dict[str, Any]:
    return {
        "executor": provider_identity_payload(system_config.executor_provider),
        "overseer": provider_identity_payload(
            system_config.overseer_provider,
            reasoning_enabled=system_config.overseer_thinking,
        ),
    }


def system_config_with_seed_override(
    system_config: SystemConfig,
    seed_override: int | None,
) -> SystemConfig:
    if seed_override is None:
        return system_config

    executor_provider = replace(
        system_config.executor_provider,
        seed=int(seed_override),
    )
    overseer_provider = (
        replace(system_config.overseer_provider, seed=int(seed_override))
        if system_config.overseer_provider is not None
        else None
    )
    return replace(
        system_config,
        executor_provider=executor_provider,
        overseer_provider=overseer_provider,
    )


def _resolve_oversight_profile(defaults: dict[str, Any]) -> str:
    explicit_profile = defaults.get("oversight_profile")
    if explicit_profile is not None:
        normalized = str(explicit_profile).strip()
        if normalized not in _VALID_OVERSIGHT_PROFILES:
            available = ", ".join(sorted(_VALID_OVERSIGHT_PROFILES))
            raise ValueError(
                f"Unknown oversight_profile '{normalized}'. Available: {available}"
            )
        return normalized

    if not bool(defaults.get("oversight_enabled", False)):
        return "executor_only"

    mode = str(defaults.get("oversight_mode", "disabled")).strip()
    if mode in _OVERSIGHT_PROFILE_BY_MODE:
        return _OVERSIGHT_PROFILE_BY_MODE[mode]

    available = ", ".join(sorted(_OVERSIGHT_PROFILE_BY_MODE))
    raise ValueError(f"Unknown oversight_mode '{mode}'. Available: {available}")


def _resolve_oversight_mode(defaults: dict[str, Any], *, profile: str) -> str:
    explicit_mode = defaults.get("oversight_mode")
    if explicit_mode is None:
        return _OVERSIGHT_MODE_BY_PROFILE[profile]

    normalized = str(explicit_mode).strip()
    if normalized not in _VALID_OVERSIGHT_MODES:
        available = ", ".join(sorted(_VALID_OVERSIGHT_MODES))
        raise ValueError(
            f"Unknown oversight_mode '{normalized}'. Available: {available}"
        )

    expected_profile = _OVERSIGHT_PROFILE_BY_MODE[normalized]
    if expected_profile != profile:
        raise ValueError(
            "Config oversight_mode "
            f"'{normalized}' does not match oversight_profile '{profile}'."
        )
    return normalized


def build_system_config(
    system_name: str,
    executor_model: str,
    overseer_model: str = "deepseek-v4-flash",
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
    oversight_profile = _resolve_oversight_profile(defaults)
    oversight_mode = _resolve_oversight_mode(
        defaults,
        profile=oversight_profile,
    )

    return SystemConfig(
        name=str(defaults["name"]),
        executor_provider=ProviderConfig.from_model_name(executor_model),
        oversight_enabled=oversight_enabled,
        oversight_mode=oversight_mode,
        oversight_profile=oversight_profile,
        overseer_provider=overseer_provider,
        overseer_thinking=defaults.get("overseer_thinking"),
        max_steps=max_steps,
        num_runs=num_runs,
        overseer_prompt_version=str(
            defaults.get("overseer_prompt_version", "c2-lite-v1.3")
        ),
        loop_similarity_threshold=float(
            defaults.get("loop_similarity_threshold", 0.92)
        ),
        loop_window=int(defaults.get("loop_window", 5)),
        loop_repeat_count=int(defaults.get("loop_repeat_count", 3)),
        coverage_threshold=float(defaults.get("coverage_threshold", 0.50)),
        final_repair_retry_cap=int(defaults.get("final_repair_retry_cap", 2)),
        max_stale_cart_notices=int(defaults.get("max_stale_cart_notices", 1)),
        recent_tool_window=int(defaults.get("recent_tool_window", 5)),
        inject_transient_notice=bool(defaults.get("inject_transient_notice", True)),
        mutating_tools=tuple(
            str(tool_name)
            for tool_name in defaults.get(
                "mutating_tools",
                (
                    "add_product_to_cart",
                    "delete_product_from_cart",
                    "add_coupon_to_cart",
                    "delete_coupon_from_cart",
                ),
            )
        ),
        irreversible_tools=tuple(
            str(tool_name) for tool_name in defaults.get("irreversible_tools", ())
        ),
        block_on_mutation_mode=str(defaults.get("block_on_mutation_mode", "auto")),
        max_hard_blocks_per_args=int(defaults.get("max_hard_blocks_per_args", 2)),
        max_consecutive_pre_tool_blocks=int(
            defaults.get("max_consecutive_pre_tool_blocks", 5)
        ),
        require_cited_violation_for_block=bool(
            defaults.get("require_cited_violation_for_block", True)
        ),
        overseer_call_budget_per_task=int(
            defaults.get("overseer_call_budget_per_task", 8)
        ),
    )
