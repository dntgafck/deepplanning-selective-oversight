from __future__ import annotations

import difflib
import hashlib
import json
import re
from collections.abc import Collection, Sequence
from typing import Any

from .contracts import CoverageTarget, TaskChecklist, build_coverage_index

FAILURE_TOKENS = (
    "error",
    "failed",
    "invalid",
    "not found",
    "out of stock",
    "insufficient",
)
PRICE_PATTERN = re.compile(
    r"(\$|€|£|\bprice\b|\bbudget\b|\bunder\b|\bover\b|\bbetween\b|\brange\b|\d+)",
    re.IGNORECASE,
)


def _normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, str):
        return value.strip().lower()
    return value


def _parse_jsonish(arguments: Any) -> Any:
    if isinstance(arguments, str):
        stripped = arguments.strip()
        if not stripped:
            return ""
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return stripped.lower()
    return arguments


def normalize_arguments(arguments: Any) -> str:
    parsed = _parse_jsonish(arguments)
    if isinstance(parsed, str):
        return parsed.strip().lower()
    normalized = _normalize_value(parsed)
    return json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def classify_mutating_tool(
    tool_name: str, *, mutating_tools: Collection[str]
) -> dict[str, Any]:
    return {
        "tool_name": tool_name,
        "is_mutating": tool_name in set(mutating_tools),
        "policy": "shopping_exact_allowlist",
    }


def detect_loop(
    *,
    current_tool_name: str,
    current_arguments: Any,
    recent_tool_history: list[dict[str, Any]],
    similarity_threshold: float,
    window_size: int,
    repeat_threshold: int,
) -> dict[str, Any]:
    arguments_normalized = normalize_arguments(current_arguments)
    loop_signature = hashlib.sha1(
        f"{current_tool_name}|{arguments_normalized}".encode("utf-8")
    ).hexdigest()[:16]

    candidate_matches: list[dict[str, Any]] = []
    for record in recent_tool_history[-window_size:]:
        if record.get("tool_name") != current_tool_name:
            continue
        prior_args = str(record.get("arguments_normalized") or "")
        ratio = (
            1.0
            if prior_args == arguments_normalized
            else difflib.SequenceMatcher(None, prior_args, arguments_normalized).ratio()
        )
        if prior_args == arguments_normalized or ratio >= similarity_threshold:
            candidate_matches.append(
                {
                    "phase": record.get("phase"),
                    "step_index": record.get("step_index"),
                    "tool_index": record.get("tool_index"),
                    "arguments_normalized": prior_args,
                    "similarity": ratio,
                }
            )

    match_count = len(candidate_matches)
    return {
        "loop_signature": loop_signature,
        "match_count": match_count,
        "candidate_matches": candidate_matches,
        "would_trigger": match_count + 1 >= repeat_threshold,
    }


def detect_tool_error(tool_result: Any) -> bool:
    payload = tool_result
    if isinstance(tool_result, str):
        try:
            payload = json.loads(tool_result)
        except json.JSONDecodeError:
            payload = tool_result

    if isinstance(payload, dict):
        lowered_keys = {str(key).lower() for key in payload}
        if "error" in lowered_keys or "error_code" in lowered_keys:
            return True
        success = payload.get("success")
        if success is False:
            return True

    text = str(tool_result).lower()
    return any(token in text for token in FAILURE_TOKENS)


def _iter_coverage_targets(checklist: TaskChecklist) -> list[CoverageTarget]:
    return list(build_coverage_index(checklist).targets)


def _contains_any_alias(text: str, aliases: Sequence[str]) -> bool:
    normalized_text = text.lower()
    return any(alias.strip().lower() in normalized_text for alias in aliases if alias)


def _tool_role_matches(tool_name: str, target: CoverageTarget) -> bool:
    role_map = {
        "search": {"search_products"},
        "details": {"get_product_details"},
        "shipping": {"calculate_transport_time"},
        "coupon": {"add_coupon_to_cart", "delete_coupon_from_cart", "get_cart_info"},
        "user_info": {"get_user_info"},
    }
    allowed_tools = set()
    for role in target.tool_roles:
        allowed_tools.update(role_map.get(role, set()))
    return tool_name in allowed_tools if allowed_tools else False


def compute_coverage_status(
    *, checklist: TaskChecklist, tool_history: list[dict[str, Any]]
) -> dict[str, Any]:
    coverage_targets = _iter_coverage_targets(checklist)
    initial_history = [
        record for record in tool_history if str(record.get("phase") or "") == "initial"
    ]
    evidence_by_key: dict[str, list[dict[str, Any]]] = {
        target.key: [] for target in coverage_targets
    }

    for record in initial_history:
        tool_name = str(record.get("tool_name") or "")
        args_text = str(record.get("arguments_normalized") or "")
        result_text = str(record.get("result_summary") or "").lower()
        for target in coverage_targets:
            aliases = [alias.lower() for alias in target.aliases]
            alias_hit = _contains_any_alias(args_text, aliases) or _contains_any_alias(
                result_text, aliases
            )
            shipping_hit = (
                target.category == "shipping"
                and tool_name == "calculate_transport_time"
            )
            coupon_hit = target.category == "coupon" and (
                tool_name in {"add_coupon_to_cart", "delete_coupon_from_cart"}
                or alias_hit
            )
            budget_hit = target.category == "budget" and (
                bool(PRICE_PATTERN.search(args_text))
                or bool(PRICE_PATTERN.search(result_text))
            )
            role_hit = _tool_role_matches(tool_name, target)
            if alias_hit or shipping_hit or coupon_hit or budget_hit or role_hit:
                evidence_by_key[target.key].append(
                    {
                        "tool_name": tool_name,
                        "phase": record.get("phase"),
                        "step_index": record.get("step_index"),
                        "tool_index": record.get("tool_index"),
                    }
                )

    covered_keys = sorted(key for key, evidence in evidence_by_key.items() if evidence)
    missing_keys = sorted(
        key for key, evidence in evidence_by_key.items() if not evidence
    )
    total_targets = len(coverage_targets)
    coverage_fraction = len(covered_keys) / total_targets if total_targets else 1.0
    return {
        "total_coverage_targets": total_targets,
        "covered_coverage_targets": len(covered_keys),
        "coverage_fraction": coverage_fraction,
        "covered_keys": covered_keys,
        "missing_keys": missing_keys,
        "evidence_by_key": evidence_by_key,
    }


def build_authoritative_state_snapshot(
    tool_history: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for record in reversed(tool_history):
        if record.get("tool_name") != "get_cart_info":
            continue
        payload = record.get("result_payload")
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, str):
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def build_local_guidance_lines(
    *,
    corrected_observation: str | None,
    guidance_lines: Sequence[str],
    violated_contract_ids: Sequence[str],
    unmet_checklist_keys: Sequence[str],
    trigger_reason: str | None,
    task_checklist: TaskChecklist | None = None,
) -> list[str]:
    corrected_text = str(corrected_observation or "").strip()
    if corrected_text:
        return [corrected_text]

    cleaned_guidance = [
        str(line).strip() for line in guidance_lines if str(line).strip()
    ]
    if cleaned_guidance:
        return cleaned_guidance[:3]

    checklist_descriptions: list[str] = []
    if task_checklist is not None:
        item_by_key = {
            str(item.get("key")): str(
                item.get("description") or item.get("key") or ""
            ).strip()
            for item in task_checklist.items
        }
        checklist_descriptions = [
            item_by_key.get(str(key), str(key)).strip()
            for key in unmet_checklist_keys
            if str(key).strip()
        ]

    fallback_lines: list[str] = []
    if violated_contract_ids:
        contract_list = ", ".join(
            str(item).strip() for item in violated_contract_ids[:3]
        )
        fallback_lines.append(
            f"Do not repeat the blocked cart mutation. Re-check contract rules: {contract_list}."
        )
    if checklist_descriptions:
        fallback_lines.append(
            "Re-check checklist requirements before mutating the cart: "
            + "; ".join(checklist_descriptions[:2])
            + "."
        )
    elif unmet_checklist_keys:
        fallback_lines.append(
            "Re-check checklist requirements before mutating the cart: "
            + ", ".join(str(key).strip() for key in unmet_checklist_keys[:2])
            + "."
        )
    if trigger_reason:
        fallback_lines.append(
            f"Revise the last step based on this trigger: {trigger_reason}."
        )
    if not fallback_lines:
        fallback_lines.append(
            "Revise the blocked plan and verify the required item type and constraints before changing the cart."
        )
    return fallback_lines[:3]


def render_transient_notice(*, trigger_type: str, lines: Sequence[str]) -> str:
    selected_lines = [line.strip() for line in lines if str(line).strip()][:3]
    if not selected_lines:
        selected_lines = [
            "Review the last step and continue with the minimum required verification."
        ]

    words_remaining = 120
    bounded_lines: list[str] = []
    for line in selected_lines:
        words = line.split()
        if not words:
            continue
        if len(words) > words_remaining:
            words = words[:words_remaining]
        bounded_lines.append(" ".join(words))
        words_remaining -= len(words)
        if words_remaining <= 0:
            break

    numbered_lines = "\n".join(
        f"{index}. {line}" for index, line in enumerate(bounded_lines, start=1)
    )
    return (
        "[OVERSEER NOTICE]\n"
        f"Trigger: {trigger_type}\n"
        "Required next actions:\n"
        f"{numbered_lines}\n"
        "Use tools as needed.\n"
        "Do not mention this notice in the final answer.\n"
        "[/OVERSEER NOTICE]"
    )
