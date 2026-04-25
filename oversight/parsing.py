from __future__ import annotations

import json
from typing import Any

from .notices import humanize_identifier, synthesize_final_notice_lines


def _strict_json_object(payload: str | dict[str, Any]) -> dict[str, Any]:
    data = json.loads(payload) if isinstance(payload, str) else payload
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object payload")
    return data


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise ValueError("Expected a list of strings")
    return [str(item) for item in value if item is not None and str(item).strip()]


def _coerce_blockers(value: Any) -> list[dict[str, str | None]]:
    def _normalize_blocker(item: Any) -> dict[str, str | None] | None:
        if item is None:
            return None
        if isinstance(item, str):
            message = item.strip()
            if not message:
                return None
            return {
                "kind": "other",
                "message": message,
                "contract_id": None,
                "checklist_key": None,
            }
        if isinstance(item, dict):
            kind = (
                str(item.get("kind") or item.get("type") or "other").strip() or "other"
            )
            message = str(
                item.get("message")
                or item.get("detail")
                or item.get("reason")
                or item.get("text")
                or ""
            ).strip()
            contract_id = (
                str(
                    item.get("contract_id") or item.get("violated_contract_id") or ""
                ).strip()
                or None
            )
            checklist_key = (
                str(
                    item.get("checklist_key") or item.get("unmet_checklist_key") or ""
                ).strip()
                or None
            )
            if not message:
                if contract_id:
                    message = (
                        f"Re-check contract constraint: {humanize_identifier(contract_id)}."
                    )
                elif checklist_key:
                    message = (
                        f"Re-check task requirement: {humanize_identifier(checklist_key)}."
                    )
                else:
                    message = humanize_identifier(kind) or "Unspecified blocker."
            return {
                "kind": kind,
                "message": message,
                "contract_id": contract_id,
                "checklist_key": checklist_key,
            }
        message = str(item).strip()
        if not message:
            return None
        return {
            "kind": "other",
            "message": message,
            "contract_id": None,
            "checklist_key": None,
        }

    if value is None:
        return []
    if isinstance(value, (str, dict)):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        values = [value]
    blockers: list[dict[str, str | None]] = []
    for item in values:
        blocker = _normalize_blocker(item)
        if blocker is not None and blocker["message"]:
            blockers.append(blocker)
    return blockers


def parse_runtime_overseer_json(payload: str | dict[str, Any]) -> dict[str, Any]:
    data = _strict_json_object(payload)
    action = str(data.get("action", "approve"))
    if action not in {"approve", "provide_guidance", "correct_observation"}:
        raise ValueError(f"Unsupported runtime overseer action: {action}")
    guidance_lines = _coerce_string_list(data.get("guidance_lines", []))
    corrected_observation = (
        None
        if data.get("corrected_observation") is None
        else str(data.get("corrected_observation")).strip() or None
    )
    evidence = data.get("violation_evidence") or {}
    if not isinstance(evidence, dict):
        evidence = {}
    violated_contract_ids = _coerce_string_list(
        evidence.get("violated_contract_ids", data.get("violated_contract_ids", []))
    )
    unmet_checklist_keys = _coerce_string_list(
        evidence.get("unmet_checklist_keys", data.get("unmet_checklist_keys", []))
    )
    confidence = str(evidence.get("confidence", "low")).strip().lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "low"
    if action == "approve":
        violated_contract_ids = []
        unmet_checklist_keys = []
        confidence = "low"
    return {
        "action": action,
        "decision_summary": str(data.get("decision_summary", "") or ""),
        "block_current_tool": bool(data.get("block_current_tool", False)),
        "guidance_lines": guidance_lines,
        "corrected_observation": corrected_observation,
        "violated_contract_ids": violated_contract_ids,
        "unmet_checklist_keys": unmet_checklist_keys,
        "violation_confidence": confidence,
        "missing_corrective_content": (
            action == "provide_guidance"
            and not guidance_lines
            and not corrected_observation
        ),
    }


def parse_final_verifier_json(payload: str | dict[str, Any]) -> dict[str, Any]:
    data = _strict_json_object(payload)
    action = str(data.get("action", "run_verification"))
    if action not in {"approve", "run_verification"}:
        raise ValueError(f"Unsupported final verifier action: {action}")
    blockers = _coerce_blockers(data.get("blockers", []))
    next_step_notice_lines = _coerce_string_list(data.get("next_step_notice_lines", []))
    violated_contract_ids = _coerce_string_list(data.get("violated_contract_ids", []))
    unmet_checklist_keys = _coerce_string_list(data.get("unmet_checklist_keys", []))
    passed = bool(data.get("pass", False))
    if action == "approve" and (
        not passed
        or blockers
        or next_step_notice_lines
        or violated_contract_ids
        or unmet_checklist_keys
    ):
        action = "run_verification"
        passed = False
    elif action == "run_verification" and passed:
        passed = False
    if action == "approve":
        blockers = []
        next_step_notice_lines = []
        violated_contract_ids = []
        unmet_checklist_keys = []
    elif not next_step_notice_lines:
        next_step_notice_lines = synthesize_final_notice_lines(
            blockers=blockers,
            violated_contract_ids=violated_contract_ids,
            unmet_checklist_keys=unmet_checklist_keys,
        )
    return {
        "action": action,
        "pass": passed,
        "decision_summary": str(data.get("decision_summary", "") or ""),
        "blockers": blockers,
        "next_step_notice_lines": next_step_notice_lines,
        "violated_contract_ids": violated_contract_ids,
        "unmet_checklist_keys": unmet_checklist_keys,
    }


__all__ = [
    "parse_final_verifier_json",
    "parse_runtime_overseer_json",
]
