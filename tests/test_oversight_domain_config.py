from __future__ import annotations

from oversight.contracts import CoverageTarget, TaskChecklist
from oversight.domain_config import default_final_notice, load_oversight_domain_config
from oversight.triggers import (
    build_authoritative_state_snapshot,
    compute_coverage_status,
)


def _checklist() -> TaskChecklist:
    return TaskChecklist(
        checklist_id="checklist-1",
        items=[
            {
                "key": "product:laptop",
                "category": "required_product",
                "description": "Find a laptop",
                "required": True,
                "explicit": True,
                "coverage_relevant": True,
                "final_verify_only": False,
                "aliases": ["laptop"],
            }
        ],
        coverage_targets=[
            CoverageTarget(
                key="product:laptop",
                category="product",
                aliases=["laptop"],
                tool_roles=["search"],
            )
        ],
        final_verification_only_keys=[],
        ambiguities=[],
        compiler_signature="sig",
    )


def test_shopping_domain_config_preserves_authoritative_snapshot_behavior():
    domain_config = load_oversight_domain_config("shopping")
    snapshot = build_authoritative_state_snapshot(
        [
            {
                "tool_name": domain_config.state_authority_tools[0],
                "result_payload": {"items": ["old"]},
            },
            {
                "tool_name": "unrelated_tool",
                "result_payload": {"items": ["ignored"]},
            },
            {
                "tool_name": domain_config.state_authority_tools[0],
                "result_payload": {"items": ["new"]},
            },
        ],
        authority_tools=domain_config.state_authority_tools,
    )

    assert snapshot == {"items": ["new"]}
    assert default_final_notice("shopping") == (
        "Call get_cart_info before finalizing. "
        "The cart state is the source of truth."
    )


def test_coverage_status_uses_configured_role_map():
    domain_config = load_oversight_domain_config("shopping")
    coverage = compute_coverage_status(
        checklist=_checklist(),
        tool_history=[
            {
                "phase": "initial",
                "tool_name": domain_config.role_map["search"][0],
                "arguments_normalized": '{"query":"laptop"}',
                "result_summary": "[]",
            }
        ],
        role_map=domain_config.role_map,
    )

    assert coverage["covered_coverage_targets"] == 1
    assert coverage["missing_keys"] == []
