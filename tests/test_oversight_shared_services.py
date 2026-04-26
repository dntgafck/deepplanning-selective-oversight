from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from oversight import ConversationState, OversightAction
from oversight import notices as notices_module
from oversight import triggers as triggers_module
from oversight.clients import invoke_final_verifier, invoke_runtime_overseer
from oversight.contracts import CoverageTarget, ExecutionContract, TaskChecklist
from oversight.notices import DEFAULT_FINAL_NOTICE, render_notice_from_action
from oversight.parsing import parse_final_verifier_json, parse_runtime_overseer_json
from oversight.policies import (
    H1Outcome,
    compute_h1_outcome,
    increment_retry_and_check_cap,
)


class FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.role = "assistant"
        self.content = content


class FakeChoice:
    def __init__(self, message: FakeMessage) -> None:
        self.index = 0
        self.finish_reason = "stop"
        self.message = message


class FakeResponse:
    def __init__(
        self,
        content: str,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        self.id = "resp_1"
        self.model = "fake-model"
        self.system_fingerprint = "fp_test"
        self.choices = [FakeChoice(FakeMessage(content))]
        self.usage = FakeUsage(prompt_tokens, completion_tokens)


def _execution_contract() -> ExecutionContract:
    return ExecutionContract(
        contract_id="contract-shopping",
        domain="shopping",
        primary_objective="Build the correct cart.",
        objective_priority=["requirements", "budget"],
        hard_rules=[{"id": "rule-1", "text": "Stay within budget."}],
        state_authority_rules=[
            {"state": "cart", "tool": "get_cart_info", "authoritative": True}
        ],
        level_policy={"budget_priority": "primary"},
        tool_semantics={
            "mutating_tools": ["add_product_to_cart"],
            "read_only_tools": ["get_cart_info"],
            "search_tools": ["search_products"],
            "verification_tools": ["get_cart_info"],
        },
        final_output_requirements=["Use the authoritative cart state."],
        compiler_signature="sig",
    )


def _task_checklist() -> TaskChecklist:
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
        final_verification_only_keys=["final:fresh-cart"],
        ambiguities=[],
        compiler_signature="sig",
    )


def _state() -> ConversationState:
    state = ConversationState(
        task_id="1",
        domain="shopping",
        complexity=1,
        system_config_name="shared-services-test",
    )
    state.execution_contract = _execution_contract()
    state.task_checklist = _task_checklist()
    return state


def _runtime_config() -> SimpleNamespace:
    return SimpleNamespace(
        overseer_provider=SimpleNamespace(alias="fake-overseer"),
        overseer_thinking=True,
        recent_tool_window=5,
        final_repair_retry_cap=2,
        mutating_tools=("add_product_to_cart",),
        irreversible_tools=(),
        block_on_mutation_mode="always",
        max_hard_blocks_per_args=2,
        max_consecutive_pre_tool_blocks=5,
        require_cited_violation_for_block=True,
    )


def test_parsing_helpers_still_normalize_runtime_and_final_payloads():
    runtime = parse_runtime_overseer_json(
        {
            "action": "provide_guidance",
            "decision_summary": "Need more evidence.",
            "violation_evidence": {
                "violated_contract_ids": ["rule-1"],
                "unmet_checklist_keys": ["product:laptop"],
                "confidence": "medium",
            },
            "guidance_lines": [],
            "corrected_observation": None,
        }
    )
    final = parse_final_verifier_json(
        {
            "action": "run_verification",
            "pass": False,
            "decision_summary": "Need a fresh cart read.",
            "blockers": [],
            "next_step_notice_lines": [],
            "violated_contract_ids": ["rule_1"],
            "unmet_checklist_keys": [],
        }
    )

    assert runtime["missing_corrective_content"] is True
    assert runtime["violation_confidence"] == "medium"
    assert final["next_step_notice_lines"] == ["Re-check contract constraint: rule 1."]


def test_notice_helpers_render_default_final_notice_and_compat_reexports():
    action = OversightAction(
        trigger_type="final_checkpoint",
        intervention_type="run_verification",
    )

    notice = render_notice_from_action(action)

    assert DEFAULT_FINAL_NOTICE in notice
    assert action.notice_source == "default_final_notice"
    assert (
        triggers_module.render_transient_notice
        is notices_module.render_transient_notice
    )
    assert (
        triggers_module.build_local_guidance_lines
        is notices_module.build_local_guidance_lines
    )


def test_policy_helpers_preserve_streak_cap_and_retry_cap_behavior():
    state = _state()
    state.consecutive_pre_tool_blocks = 5
    action = OversightAction(
        intervention_type="provide_guidance",
        violated_contract_ids=["rule-1"],
        violation_confidence="high",
    )
    system_config = _runtime_config()

    outcome = compute_h1_outcome(
        action=action,
        tool_name="add_product_to_cart",
        arguments={"product_id": "p1"},
        state=state,
        system_config=system_config,
    )

    retry_state = _state()
    retry_state.final_verification_retry_count = 2

    assert outcome == H1Outcome.FORCED_APPROVE
    assert increment_retry_and_check_cap(retry_state, system_config) is True
    assert retry_state.final_verification_result == "retry_cap_exhausted"


def test_clients_runtime_helper_uses_missing_content_fallback_notice():
    async def fake_call_chat_completion(**kwargs):
        return FakeResponse(
            json.dumps(
                {
                    "action": "provide_guidance",
                    "decision_summary": "Blocked pending correction.",
                    "guidance_lines": [],
                    "corrected_observation": None,
                    "violated_contract_ids": ["rule-1"],
                    "unmet_checklist_keys": ["product:laptop"],
                }
            ),
            prompt_tokens=12,
            completion_tokens=4,
        )

    state = _state()
    action = asyncio.run(
        invoke_runtime_overseer(
            action_factory=OversightAction,
            call_chat_completion_fn=fake_call_chat_completion,
            estimate_call_cost_fn=lambda **kwargs: 0.25,
            prompt="runtime prompt",
            task_query="buy a laptop",
            state=state,
            system_config=_runtime_config(),
            phase="initial",
            step_index=1,
            tool_index=0,
            proposed_tool_calls=[
                {
                    "id": "call_1",
                    "name": "add_product_to_cart",
                    "arguments": '{"product_id":"1"}',
                }
            ],
            latest_tool_result=None,
            trigger_type="mutating_action",
            allowed_actions=["approve", "provide_guidance"],
            trigger_reason="mutating tool proposed: add_product_to_cart",
            trigger_evidence={"tool_name": "add_product_to_cart"},
        )
    )

    assert action.fallback_guidance_used is True
    assert action.notice_source == "local_fallback"
    assert "Re-check task requirement: product laptop." in action.notice_text
    assert state.overseer_calls == 1


def test_clients_final_helper_counts_parse_fallback_and_requests_retry():
    async def fake_call_chat_completion(**kwargs):
        return FakeResponse("{not valid json", prompt_tokens=10, completion_tokens=3)

    state = _state()
    action = asyncio.run(
        invoke_final_verifier(
            action_factory=OversightAction,
            call_chat_completion_fn=fake_call_chat_completion,
            estimate_call_cost_fn=lambda **kwargs: 0.5,
            increment_retry_and_check_cap_fn=increment_retry_and_check_cap,
            prompt="final prompt",
            task_query="buy a laptop",
            state=state,
            system_config=_runtime_config(),
            phase="cart_check",
            step_index=2,
            draft_final_answer="Draft answer.",
        )
    )

    assert action.should_intervene is True
    assert action.final_verification_result == "repair_requested"
    assert DEFAULT_FINAL_NOTICE in action.notice_text
    assert state.final_verification_retry_count == 1
    assert state.final_verifier_parse_fallback_count == 1
