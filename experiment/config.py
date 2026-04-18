from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    overseer_prompt_version: str = "c2-lite-v1.3"
    loop_similarity_threshold: float = 0.92
    loop_window: int = 5
    loop_repeat_count: int = 3
    coverage_threshold: float = 0.50
    final_repair_retry_cap: int = 2
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
    require_cited_violation_for_block: bool = True
    overseer_call_budget_per_task: int = 8


def provider_identity_payload(
    provider: ProviderConfig | None,
) -> dict[str, Any] | None:
    if provider is None:
        return None
    return {
        "requested_model": provider.alias,
        "resolved_provider": provider.provider,
        "resolved_model": provider.model,
    }


def system_model_identities(system_config: SystemConfig) -> dict[str, Any]:
    return {
        "executor": provider_identity_payload(system_config.executor_provider),
        "overseer": provider_identity_payload(system_config.overseer_provider),
    }


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
        require_cited_violation_for_block=bool(
            defaults.get("require_cited_violation_for_block", True)
        ),
        overseer_call_budget_per_task=int(
            defaults.get("overseer_call_budget_per_task", 8)
        ),
    )
