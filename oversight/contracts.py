from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llm import call_chat_completion

if TYPE_CHECKING:
    from experiment import SystemConfig

P0_SYSTEM_PROMPT = """You convert an executor's task instructions and tool schema into a compact execution contract for selective oversight.

Output valid JSON only.
Do not solve tasks.
Do not rewrite the task instructions.
Extract only policy-relevant structure needed for runtime monitoring and verification.

Required behavior:
- Identify the primary objective and objective priority.
- Extract explicit hard rules.
- Identify what state is authoritative and which tool reads it.
- Classify tools into mutating, read-only, search, and verification roles.
- Capture any level-specific policy differences such as budget priority or coupon logic.
- Keep the contract concise and normalized."""

P1_SYSTEM_PROMPT = """You convert a task query into an instance-specific checklist under the provided execution contract.

Output valid JSON only.
Do not solve the task.
Do not invent preferences or constraints.
If information is ambiguous, record the ambiguity.
Separate coverage targets from final verification constraints.

Required fidelity rules:
- Preserve explicit product-type or category constraints in machine-usable form.
- Preserve named item requirements, review or quality constraints, and shipping constraints.
- Do not promote global task framing (e.g., "footwear collection", "summer outfit", "running gear") into per-item hard constraints. Each item's `value.product_type` must be supported by that item's own description or source span. If an item's product type is unclear or only implied by the overall task theme, leave `value.product_type` null and record the item key in `ambiguities`.
- For any hard item-level semantic field you emit, include item-local grounding metadata under `value.support` with field-specific entries such as `support.product_type = {"scope": "item_local", "spans": [...], "strength": "explicit"}`.
- Coverage targets must be keyed to the corresponding checklist item keys rather than synthetic placeholder keys."""

CHECKLIST_NORMALIZER_VERSION = "shopping-hardening-v3"


@dataclass(slots=True)
class ExecutionContract:
    contract_id: str
    domain: str
    primary_objective: str
    objective_priority: list[str]
    hard_rules: list[dict[str, Any]]
    state_authority_rules: list[dict[str, Any]]
    level_policy: dict[str, Any]
    tool_semantics: dict[str, list[str]]
    final_output_requirements: list[str]
    compiler_signature: str


@dataclass(slots=True)
class CoverageTarget:
    key: str
    category: str
    aliases: list[str]
    tool_roles: list[str]


@dataclass(slots=True)
class TaskChecklist:
    checklist_id: str
    items: list[dict[str, Any]]
    coverage_targets: list[CoverageTarget]
    final_verification_only_keys: list[str]
    ambiguities: list[str]
    compiler_signature: str


@dataclass(slots=True)
class CoverageIndex:
    targets: list[CoverageTarget]


class ChecklistInvariantError(ValueError):
    """Raised when a normalized checklist violates semantic invariants."""


PRODUCT_TYPE_HINTS: tuple[tuple[str, str], ...] = (
    ("trail shoe", "trail shoe"),
    ("trail shoes", "trail shoes"),
    ("running shoe", "running shoe"),
    ("running shoes", "running shoes"),
    ("sneaker", "sneakers"),
    ("sneakers", "sneakers"),
    ("shoe", "shoes"),
    ("shoes", "shoes"),
    ("boot", "boots"),
    ("boots", "boots"),
    ("sandal", "sandals"),
    ("sandals", "sandals"),
)

ITEM_LOCAL_SUPPORT_SPANS: tuple[str, ...] = (
    "description",
    "source_text",
    "value_name",
)


def execution_contract_to_dict(contract: ExecutionContract) -> dict[str, Any]:
    return asdict(contract)


def task_checklist_to_dict(checklist: TaskChecklist) -> dict[str, Any]:
    return asdict(checklist)


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value).strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return deduped


def _stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def _item_local_text_sources(item: dict[str, Any]) -> dict[str, str]:
    value = item.get("value")
    value_name = ""
    if isinstance(value, dict):
        value_name = str(value.get("name") or "")

    return {
        "description": str(item.get("description") or ""),
        "source_text": str(item.get("source_text") or ""),
        "value_name": value_name,
    }


def _normalize_field_support(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    support = value.get("support")
    if not isinstance(support, dict):
        return {}

    normalized: dict[str, dict[str, Any]] = {}
    for field_name, metadata in support.items():
        if not isinstance(metadata, dict):
            continue
        normalized_metadata = dict(metadata)
        raw_spans = normalized_metadata.get("spans", [])
        if isinstance(raw_spans, str):
            raw_spans = [raw_spans]
        spans = _dedupe_strings([str(span) for span in raw_spans])
        if spans:
            normalized_metadata["spans"] = spans
        normalized[str(field_name)] = normalized_metadata
    return normalized


def _support_metadata_has_item_local_grounding(
    *,
    item: dict[str, Any],
    field_support: dict[str, Any] | None,
    candidates: list[str],
) -> bool:
    if not isinstance(field_support, dict):
        return False

    scope = str(field_support.get("scope") or "").strip().lower()
    if scope != "item_local":
        return False

    span_names = _dedupe_strings([str(span) for span in field_support.get("spans", [])])
    if not span_names:
        return False

    sources = {
        key: value.lower()
        for key, value in _item_local_text_sources(item).items()
        if key in ITEM_LOCAL_SUPPORT_SPANS and value
    }
    for span_name in span_names:
        text = sources.get(span_name.lower())
        if not text:
            continue
        if any(candidate and candidate in text for candidate in candidates):
            return True
    return False


def _has_item_local_text_support(
    *,
    item: dict[str, Any],
    candidates: list[str],
) -> bool:
    combined_local = " ".join(
        value.lower() for value in _item_local_text_sources(item).values() if value
    )
    return any(candidate and candidate in combined_local for candidate in candidates)


def _derive_local_product_type_hints(
    *,
    task_query: str,
    item: dict[str, Any],
) -> tuple[list[str], dict[str, list[str]]]:
    """Infer item-local product-type hints and their supporting spans."""
    del task_query

    category = str(item.get("category") or "")
    if category != "required_product":
        return [], {}

    sources = {
        key: value.lower()
        for key, value in _item_local_text_sources(item).items()
        if value
    }

    evidence_by_hint: dict[str, list[str]] = {}
    for needle, label in PRODUCT_TYPE_HINTS:
        for span_name, text in sources.items():
            if not text:
                continue
            if needle in text:
                evidence_by_hint.setdefault(label, [])
                if span_name not in evidence_by_hint[label]:
                    evidence_by_hint[label].append(span_name)

    hints = _dedupe_strings(list(evidence_by_hint.keys()))
    return hints, evidence_by_hint


def _normalize_checklist_item(
    *,
    item: dict[str, Any],
    task_query: str,
) -> tuple[dict[str, Any], str | None]:
    normalized = dict(item)
    original_value = normalized.get("value")
    value = (
        dict(original_value)
        if isinstance(original_value, dict)
        else ({} if original_value is None else {"raw_value": original_value})
    )
    aliases = _dedupe_strings([str(alias) for alias in normalized.get("aliases", [])])
    hints, evidence_by_hint = _derive_local_product_type_hints(
        task_query=task_query,
        item=normalized,
    )
    normalized_support = _normalize_field_support(value)
    if normalized_support:
        value["support"] = normalized_support
    else:
        value.pop("support", None)

    author_provided_type = str(value.get("product_type") or "").strip()
    ambiguity_entry: str | None = None

    if author_provided_type:
        value["product_type_hints_soft"] = _dedupe_strings(
            [hint for hint in hints if hint.lower() != author_provided_type.lower()]
        )
        aliases = _dedupe_strings(
            aliases + hints + [author_provided_type, str(value.get("name") or "")]
        )
    elif not hints:
        pass
    else:
        value["product_type_hints_soft"] = _dedupe_strings(hints)
        soft_aliases = [
            str(value.get("name") or ""),
            str(normalized.get("description") or ""),
        ]
        for hint in hints:
            if len(evidence_by_hint.get(hint, [])) >= 2:
                soft_aliases.append(hint)
        aliases = _dedupe_strings(aliases + soft_aliases)
        key = str(normalized.get("key") or "").strip()
        if key:
            ambiguity_entry = (
                f"item {key!r} has product-type hints "
                f"({', '.join(hints)}) but no compiler-issued hard product_type"
            )

    normalized["aliases"] = aliases
    if isinstance(original_value, dict) or value:
        normalized["value"] = value
    return normalized, ambiguity_entry


def _coverage_category(item: dict[str, Any]) -> str:
    category = str(item.get("category") or "")
    return {
        "required_product": "product",
        "product_attribute": "attribute",
        "budget": "budget",
        "shipping": "shipping",
        "coupon": "coupon",
    }.get(category, "product")


def _coverage_tool_roles(item: dict[str, Any]) -> list[str]:
    category = str(item.get("category") or "")
    value = item.get("value")
    text = " ".join(
        part
        for part in (
            str(item.get("description") or ""),
            str(item.get("source_text") or ""),
            _stringify_value(value),
        )
        if part
    ).lower()
    roles: list[str] = []
    if category == "coupon":
        roles.append("coupon")
    elif category == "shipping":
        roles.append("shipping")
    else:
        roles.extend(["search", "details"])
        if "transport time" in text or "arrive" in text or "shipping" in text:
            roles.append("shipping")
    return _dedupe_strings(roles)


def _coverage_aliases(item: dict[str, Any]) -> list[str]:
    value = item.get("value")
    payload = dict(value) if isinstance(value, dict) else {}
    aliases = [str(alias) for alias in item.get("aliases", [])]
    aliases.extend(
        [
            str(payload.get("name") or ""),
            str(payload.get("brand") or ""),
            str(payload.get("color") or ""),
            str(payload.get("product_type") or ""),
            str(item.get("description") or ""),
        ]
    )
    aliases.extend(str(alias) for alias in payload.get("product_type_aliases", []))
    return _dedupe_strings(aliases)


def _validate_checklist_invariants(checklist: TaskChecklist) -> None:
    """Enforce semantic invariants for normalized shopping checklists."""
    item_keys = {str(item.get("key") or "") for item in checklist.items}
    ambiguity_keys: set[str] = set()
    for entry in checklist.ambiguities:
        text = str(entry)
        if text.startswith("item '") and "'" in text[6:]:
            closing = text.index("'", 6)
            ambiguity_keys.add(text[6:closing])

    for item in checklist.items:
        key = str(item.get("key") or "")
        value = item.get("value") if isinstance(item.get("value"), dict) else {}
        hard_type = str(value.get("product_type") or "").strip().lower()
        if not hard_type:
            continue

        if key in ambiguity_keys:
            raise ChecklistInvariantError(
                f"Checklist item {key!r} has both a hard product_type "
                f"({hard_type!r}) and an ambiguity entry."
            )

        aliases = [
            str(alias).lower() for alias in value.get("product_type_aliases", [])
        ]
        candidates = [hard_type] + aliases
        support = _normalize_field_support(value)
        product_type_support = support.get("product_type")
        if not (
            _support_metadata_has_item_local_grounding(
                item=item,
                field_support=product_type_support,
                candidates=candidates,
            )
            or _has_item_local_text_support(
                item=item,
                candidates=candidates,
            )
        ):
            raise ChecklistInvariantError(
                f"Checklist item {key!r} has hard product_type={hard_type!r} "
                "but no item-local support metadata or local textual support. "
                "Global task framing does not count."
            )

    for target in checklist.coverage_targets:
        if target.key not in item_keys:
            raise ChecklistInvariantError(
                f"Coverage target {target.key!r} has no matching checklist item."
            )


def normalize_task_checklist(
    checklist: TaskChecklist,
    *,
    task_query: str,
) -> TaskChecklist:
    normalization_results = [
        _normalize_checklist_item(item=item, task_query=task_query)
        for item in checklist.items
    ]
    normalized_items = [result[0] for result in normalization_results]
    new_ambiguities = [result[1] for result in normalization_results if result[1]]

    combined_ambiguities = _dedupe_strings(
        list(checklist.ambiguities) + new_ambiguities
    )
    result = TaskChecklist(
        checklist_id=checklist.checklist_id,
        items=normalized_items,
        coverage_targets=list(checklist.coverage_targets),
        final_verification_only_keys=list(checklist.final_verification_only_keys),
        ambiguities=combined_ambiguities,
        compiler_signature=checklist.compiler_signature,
    )
    _validate_checklist_invariants(result)
    return result


def build_coverage_index(checklist: TaskChecklist) -> CoverageIndex:
    items_by_key = {
        str(item.get("key") or ""): item
        for item in checklist.items
        if str(item.get("key") or "")
    }

    base_targets = list(checklist.coverage_targets)
    if not base_targets:
        base_targets = [
            CoverageTarget(
                key=str(item["key"]),
                category=_coverage_category(item),
                aliases=[],
                tool_roles=[],
            )
            for item in checklist.items
            if item.get("coverage_relevant")
            and not item.get("final_verify_only")
            and str(item.get("key") or "")
        ]

    targets: list[CoverageTarget] = []
    for target in base_targets:
        item = items_by_key.get(target.key)
        if item is None:
            continue
        targets.append(
            CoverageTarget(
                key=target.key,
                category=target.category or _coverage_category(item),
                aliases=_dedupe_strings(list(target.aliases) + _coverage_aliases(item)),
                tool_roles=_dedupe_strings(
                    list(target.tool_roles) + _coverage_tool_roles(item)
                ),
            )
        )
    return CoverageIndex(targets=targets)


def parse_execution_contract_json(payload: str | dict[str, Any]) -> ExecutionContract:
    data = json.loads(payload) if isinstance(payload, str) else payload
    if not isinstance(data, dict):
        raise ValueError("Execution contract payload must be a JSON object")
    return ExecutionContract(
        contract_id=str(data["contract_id"]),
        domain=str(data["domain"]),
        primary_objective=str(data["primary_objective"]),
        objective_priority=[str(item) for item in data.get("objective_priority", [])],
        hard_rules=[dict(item) for item in data.get("hard_rules", [])],
        state_authority_rules=[
            dict(item) for item in data.get("state_authority_rules", [])
        ],
        level_policy=dict(data.get("level_policy", {})),
        tool_semantics={
            str(key): [str(item) for item in value]
            for key, value in dict(data.get("tool_semantics", {})).items()
        },
        final_output_requirements=[
            str(item) for item in data.get("final_output_requirements", [])
        ],
        compiler_signature=str(data["compiler_signature"]),
    )


def parse_task_checklist_json(
    payload: str | dict[str, Any],
    *,
    task_query: str | None = None,
) -> TaskChecklist:
    data = json.loads(payload) if isinstance(payload, str) else payload
    if not isinstance(data, dict):
        raise ValueError("Task checklist payload must be a JSON object")
    checklist = TaskChecklist(
        checklist_id=str(data["checklist_id"]),
        items=[dict(item) for item in data.get("items", [])],
        coverage_targets=[
            CoverageTarget(
                key=str(item["key"]),
                category=str(item["category"]),
                aliases=[str(alias) for alias in item.get("aliases", [])],
                tool_roles=[str(role) for role in item.get("tool_roles", [])],
            )
            for item in data.get("coverage_targets", [])
        ],
        final_verification_only_keys=[
            str(item) for item in data.get("final_verification_only_keys", [])
        ],
        ambiguities=[str(item) for item in data.get("ambiguities", [])],
        compiler_signature=str(data["compiler_signature"]),
    )
    if task_query is not None:
        return normalize_task_checklist(checklist, task_query=task_query)
    return checklist


def make_contract_cache_key(
    *,
    domain: str,
    executor_system_prompt: str,
    tool_schema: Any,
    compiler_signature: str,
) -> str:
    payload = {
        "domain": domain,
        "executor_system_prompt": executor_system_prompt,
        "tool_schema": tool_schema,
        "compiler_signature": compiler_signature,
    }
    return sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def make_checklist_cache_key(
    *,
    task_id: str,
    task_query: str,
    contract_id: str,
    compiler_signature: str,
) -> str:
    payload = {
        "task_id": task_id,
        "task_query": task_query,
        "contract_id": contract_id,
        "compiler_signature": compiler_signature,
    }
    return sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _compiler_signature(system_config: SystemConfig) -> str:
    model_alias = "disabled"
    resolved_provider = "disabled"
    resolved_model = "disabled"
    if system_config.overseer_provider is not None:
        model_alias = system_config.overseer_provider.alias
        resolved_provider = str(system_config.overseer_provider.provider or "unknown")
        resolved_model = system_config.overseer_provider.model
    thinking_mode = "thinking" if system_config.overseer_thinking else "non-thinking"
    return (
        f"overseer={model_alias}|provider={resolved_provider}|"
        f"model={resolved_model}|mode={thinking_mode}|"
        f"prompt={system_config.overseer_prompt_version}|"
        f"checklist={CHECKLIST_NORMALIZER_VERSION}"
    )


def _strict_json_content(response: Any) -> dict[str, Any]:
    message = response.choices[0].message
    content = str(getattr(message, "content", "") or "").strip()
    if not content:
        raise ValueError("Overseer returned empty JSON content")
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("Overseer JSON payload must be an object")
    return payload


def _cache_path(cache_root: Path, subdir: str, cache_key: str) -> Path:
    directory = cache_root / subdir
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{cache_key}.json"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    temp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if path.exists():
        temp_path.unlink(missing_ok=True)
        return False
    os.replace(temp_path, path)
    return True


async def _compile_execution_contract(
    *,
    domain: str,
    executor_system_prompt: str,
    tool_schema: Any,
    system_config: SystemConfig,
) -> ExecutionContract:
    if system_config.overseer_provider is None:
        raise ValueError("Adaptive oversight requires an overseer provider")

    compiler_signature = _compiler_signature(system_config)
    response = await call_chat_completion(
        provider=system_config.overseer_provider,
        messages=[
            {"role": "system", "content": P0_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "domain": domain,
                        "executor_system_prompt": executor_system_prompt,
                        "tool_schema": tool_schema,
                        "response_schema": {
                            "contract_id": "string",
                            "domain": "shopping",
                            "primary_objective": "string",
                            "objective_priority": ["string"],
                            "hard_rules": [{"id": "string", "text": "string"}],
                            "state_authority_rules": [
                                {
                                    "state": "cart",
                                    "tool": "get_cart_info",
                                    "authoritative": True,
                                }
                            ],
                            "level_policy": {
                                "budget_priority": "primary|secondary|none",
                                "coupon_reasoning_required": True,
                                "allow_over_budget_explanation": False,
                            },
                            "tool_semantics": {
                                "mutating_tools": ["string"],
                                "read_only_tools": ["string"],
                                "search_tools": ["string"],
                                "verification_tools": ["string"],
                            },
                            "final_output_requirements": ["string"],
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        reasoning_enabled=system_config.overseer_thinking,
        validate_nonempty=True,
    )
    payload = _strict_json_content(response)
    payload["compiler_signature"] = compiler_signature
    return parse_execution_contract_json(payload)


async def _compile_task_checklist(
    *,
    task_id: str,
    task_query: str,
    execution_contract: ExecutionContract,
    system_config: SystemConfig,
) -> TaskChecklist:
    if system_config.overseer_provider is None:
        raise ValueError("Adaptive oversight requires an overseer provider")

    response = await call_chat_completion(
        provider=system_config.overseer_provider,
        messages=[
            {"role": "system", "content": P1_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "execution_contract": execution_contract_to_dict(
                            execution_contract
                        ),
                        "task_query": task_query,
                        "response_schema": {
                            "checklist_id": "string",
                            "items": [
                                {
                                    "key": "string",
                                    "category": "required_product|product_attribute|budget|shipping|coupon|ambiguity|final_requirement",
                                    "description": "string",
                                    "value": {
                                        "product_type": "string|null",
                                        "name": "string|null",
                                        "brand": "string|null",
                                        "color": "string|null",
                                        "support": {
                                            "product_type": {
                                                "scope": "item_local",
                                                "spans": [
                                                    "description|source_text|value_name"
                                                ],
                                                "strength": "explicit|derived",
                                            }
                                        },
                                        "review_criteria": "object|null",
                                        "shipping_constraints": "object|null",
                                        "other_constraints": "object|null",
                                    },
                                    "required": True,
                                    "explicit": True,
                                    "coverage_relevant": True,
                                    "final_verify_only": False,
                                    "aliases": ["string"],
                                    "source_text": "string|null",
                                }
                            ],
                            "coverage_targets": [
                                {
                                    "key": "string",
                                    "category": "product|attribute|budget|shipping|coupon",
                                    "aliases": ["string"],
                                    "tool_roles": [
                                        "search|details|shipping|coupon|user_info"
                                    ],
                                }
                            ],
                            "final_verification_only_keys": ["string"],
                            "ambiguities": ["string"],
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        reasoning_enabled=system_config.overseer_thinking,
        validate_nonempty=True,
    )
    payload = _strict_json_content(response)
    payload["compiler_signature"] = execution_contract.compiler_signature
    return parse_task_checklist_json(payload, task_query=task_query)


async def load_or_build_execution_contract_with_metadata(
    *,
    domain: str,
    executor_system_prompt: str,
    tool_schema: Any,
    system_config: SystemConfig,
    cache_root: Path,
) -> tuple[ExecutionContract, str, str]:
    compiler_signature = _compiler_signature(system_config)
    cache_key = make_contract_cache_key(
        domain=domain,
        executor_system_prompt=executor_system_prompt,
        tool_schema=tool_schema,
        compiler_signature=compiler_signature,
    )
    path = _cache_path(cache_root, "contracts", cache_key)
    if path.exists():
        return (
            parse_execution_contract_json(path.read_text(encoding="utf-8")),
            cache_key,
            "hit",
        )

    contract = await _compile_execution_contract(
        domain=domain,
        executor_system_prompt=executor_system_prompt,
        tool_schema=tool_schema,
        system_config=system_config,
    )
    wrote_file = _write_json_atomic(path, execution_contract_to_dict(contract))
    status = "built" if wrote_file else "hit"
    return (
        parse_execution_contract_json(path.read_text(encoding="utf-8")),
        cache_key,
        status,
    )


async def load_or_build_task_checklist_with_metadata(
    *,
    task_id: str,
    task_query: str,
    execution_contract: ExecutionContract,
    system_config: SystemConfig,
    cache_root: Path,
) -> tuple[TaskChecklist, str, str]:
    cache_key = make_checklist_cache_key(
        task_id=task_id,
        task_query=task_query,
        contract_id=execution_contract.contract_id,
        compiler_signature=execution_contract.compiler_signature,
    )
    path = _cache_path(cache_root, "checklists", cache_key)
    if path.exists():
        return (
            parse_task_checklist_json(
                path.read_text(encoding="utf-8"),
                task_query=task_query,
            ),
            cache_key,
            "hit",
        )

    checklist = await _compile_task_checklist(
        task_id=task_id,
        task_query=task_query,
        execution_contract=execution_contract,
        system_config=system_config,
    )
    wrote_file = _write_json_atomic(path, task_checklist_to_dict(checklist))
    status = "built" if wrote_file else "hit"
    return (
        parse_task_checklist_json(
            path.read_text(encoding="utf-8"),
            task_query=task_query,
        ),
        cache_key,
        status,
    )


async def load_or_build_execution_contract(
    *,
    domain: str,
    executor_system_prompt: str,
    tool_schema: Any,
    system_config: SystemConfig,
    cache_root: Path,
) -> ExecutionContract:
    contract, _, _ = await load_or_build_execution_contract_with_metadata(
        domain=domain,
        executor_system_prompt=executor_system_prompt,
        tool_schema=tool_schema,
        system_config=system_config,
        cache_root=cache_root,
    )
    return contract


async def load_or_build_task_checklist(
    *,
    task_id: str,
    task_query: str,
    execution_contract: ExecutionContract,
    system_config: SystemConfig,
    cache_root: Path,
) -> TaskChecklist:
    checklist, _, _ = await load_or_build_task_checklist_with_metadata(
        task_id=task_id,
        task_query=task_query,
        execution_contract=execution_contract,
        system_config=system_config,
        cache_root=cache_root,
    )
    return checklist
