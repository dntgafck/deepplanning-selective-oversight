from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .contracts import TaskChecklist

DEFAULT_FINAL_NOTICE = (
    "Call get_cart_info before finalizing. The cart state is the source of truth."
)


def humanize_identifier(value: str) -> str:
    return " ".join(
        token for token in value.replace(":", " ").replace("_", " ").split() if token
    )


def synthesize_final_notice_lines(
    *,
    blockers: list[dict[str, str | None]],
    violated_contract_ids: list[str],
    unmet_checklist_keys: list[str],
) -> list[str]:
    if blockers:
        blocker = blockers[0]
        message = str(blocker.get("message") or "").strip()
        if message:
            return [message]
    if violated_contract_ids:
        return [
            f"Re-check contract constraint: {humanize_identifier(violated_contract_ids[0])}."
        ]
    if unmet_checklist_keys:
        return [
            f"Re-check task requirement: {humanize_identifier(unmet_checklist_keys[0])}."
        ]
    return [DEFAULT_FINAL_NOTICE]


def synthesize_guidance_lines(
    *,
    trigger_type: str,
    trigger_reason: str,
    trigger_evidence: dict[str, Any],
    violated_contract_ids: list[str],
    unmet_checklist_keys: list[str],
) -> list[str]:
    guidance_lines: list[str] = []
    tool_name = str(trigger_evidence.get("tool_name") or "").strip()

    for contract_id in violated_contract_ids[:2]:
        guidance_lines.append(
            f"Re-check contract constraint: {humanize_identifier(contract_id)}."
        )
    for checklist_key in unmet_checklist_keys[:2]:
        guidance_lines.append(
            f"Re-check task requirement: {humanize_identifier(checklist_key)}."
        )

    if trigger_type == "mutating_action":
        if tool_name:
            guidance_lines.append(
                f"Do not repeat {tool_name} until you verify a different candidate or cart state."
            )
        else:
            guidance_lines.append(
                "Do not repeat the blocked cart mutation until you verify a different candidate or cart state."
            )
    elif trigger_type == "always_on_pre_tool":
        if tool_name:
            guidance_lines.append(f"Review {tool_name} before continuing.")
        else:
            guidance_lines.append("Review the proposed tool call before continuing.")
    elif trigger_type == "always_on_post_tool":
        guidance_lines.append("Review the latest tool result before continuing.")
    elif trigger_type == "loop_detection":
        guidance_lines.append(
            "The current proposal repeats a blocked pattern. Change strategy before using another cart mutation."
        )
    elif trigger_reason:
        guidance_lines.append(trigger_reason)

    return guidance_lines or [
        "Pause, verify the last blocked step, and change strategy before mutating the cart again."
    ]


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


def coverage_guidance_lines(state: Any, missing_keys: list[str]) -> list[str]:
    if getattr(state, "task_checklist", None) is None:
        return [f"Inspect checklist target: {key}" for key in missing_keys[:3]]
    descriptions: list[str] = []
    for key in missing_keys:
        for item in state.task_checklist.items:
            if str(item.get("key")) != key:
                continue
            description = str(item.get("description") or key).strip()
            descriptions.append(f"Inspect {description}.")
            break
        if len(descriptions) >= 3:
            break
    return descriptions or [
        f"Inspect checklist target: {key}" for key in missing_keys[:3]
    ]


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


def render_notice_from_action(action: Any) -> str | None:
    if action.notice_text:
        action.notice_rendered = True
        action.notice_source = action.notice_source or "preset_notice"
        return action.notice_text

    lines = list(action.guidance_lines)
    if action.corrected_observation:
        action.notice_source = (
            "corrected_observation_plus_guidance" if lines else "corrected_observation"
        )
        lines = lines + [action.corrected_observation]
    elif lines:
        action.notice_source = (
            "local_fallback" if action.fallback_guidance_used else "guidance_lines"
        )
    if action.intervention_type == "run_verification" and not lines:
        action.notice_source = "default_final_notice"
        lines = [DEFAULT_FINAL_NOTICE]
    if not lines or action.trigger_type is None:
        action.notice_rendered = False
        return None
    action.notice_rendered = True
    return render_transient_notice(trigger_type=action.trigger_type, lines=lines)


__all__ = [
    "DEFAULT_FINAL_NOTICE",
    "build_local_guidance_lines",
    "coverage_guidance_lines",
    "humanize_identifier",
    "render_notice_from_action",
    "render_transient_notice",
    "synthesize_final_notice_lines",
    "synthesize_guidance_lines",
]
