from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from deepplanning.config import CONFIG_ROOT


@dataclass(frozen=True, slots=True)
class OversightDomainConfig:
    domain: str
    mutating_tools: tuple[str, ...]
    irreversible_tools: tuple[str, ...]
    role_map: dict[str, tuple[str, ...]]
    state_authority_tools: tuple[str, ...]
    state_authority_state: str
    default_final_notice_template: str
    blocked_mutation_template: str
    blocked_strategy_template: str
    product_type_hints_enabled: bool
    product_type_hint_files: tuple[str, ...]


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        cleaned = value.strip()
        return (cleaned,) if cleaned else ()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    cleaned = str(value).strip()
    return (cleaned,) if cleaned else ()


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Oversight config must be a mapping: {path}")
    return payload


@lru_cache(maxsize=None)
def load_oversight_domain_config(domain: str = "shopping") -> OversightDomainConfig:
    normalized_domain = str(domain).strip()
    config_path = CONFIG_ROOT / normalized_domain / "oversight_domain.yaml"
    payload = _load_yaml_mapping(config_path)

    role_map_payload = payload.get("role_map", {})
    if not isinstance(role_map_payload, dict):
        raise ValueError(f"role_map must be a mapping in {config_path}")

    authority_tools = _string_tuple(
        payload.get("state_authority_tools") or payload.get("state_authority_tool")
    )
    if not authority_tools:
        raise ValueError(f"state authority tool is required in {config_path}")

    return OversightDomainConfig(
        domain=normalized_domain,
        mutating_tools=_string_tuple(payload.get("mutating_tools")),
        irreversible_tools=_string_tuple(payload.get("irreversible_tools")),
        role_map={
            str(role): _string_tuple(tools)
            for role, tools in role_map_payload.items()
        },
        state_authority_tools=authority_tools,
        state_authority_state=str(payload.get("state_authority_state") or "state"),
        default_final_notice_template=str(
            payload.get("default_final_notice_template") or ""
        ),
        blocked_mutation_template=str(payload.get("blocked_mutation_template") or ""),
        blocked_strategy_template=str(payload.get("blocked_strategy_template") or ""),
        product_type_hints_enabled=bool(
            payload.get("product_type_hints_enabled", True)
        ),
        product_type_hint_files=_string_tuple(payload.get("product_type_hint_files")),
    )


def render_template(template: str, domain_config: OversightDomainConfig) -> str:
    authority_tool = (
        domain_config.state_authority_tools[0]
        if domain_config.state_authority_tools
        else "authoritative_state"
    )
    return template.format(
        state_authority_tool=authority_tool,
        state_authority_state=domain_config.state_authority_state,
    )


def default_final_notice(domain: str = "shopping") -> str:
    domain_config = load_oversight_domain_config(domain)
    return render_template(domain_config.default_final_notice_template, domain_config)


@lru_cache(maxsize=None)
def load_product_type_hints(domain: str = "shopping") -> tuple[tuple[str, str], ...]:
    domain_config = load_oversight_domain_config(domain)
    if not domain_config.product_type_hints_enabled:
        return ()

    hints: list[tuple[str, str]] = []
    for relative_path in domain_config.product_type_hint_files:
        path = CONFIG_ROOT / relative_path
        payload = _load_yaml_mapping(path)
        raw_hints = payload.get("hints", [])
        if not isinstance(raw_hints, list):
            raise ValueError(f"hints must be a list in {path}")
        for item in raw_hints:
            if not isinstance(item, dict):
                raise ValueError(f"hint entries must be mappings in {path}")
            needle = str(item.get("needle") or "").strip()
            label = str(item.get("label") or "").strip()
            if needle and label:
                hints.append((needle, label))
    return tuple(hints)


__all__ = [
    "OversightDomainConfig",
    "default_final_notice",
    "load_oversight_domain_config",
    "load_product_type_hints",
    "render_template",
]
