from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from experiment import build_system_config
from llm import client as llm_client
from oversight import (
    ConversationState,
    H1Outcome,
    OversightAction,
    apply_intervention,
    compute_h1_outcome,
    evaluate_oversight,
    parse_final_verifier_json,
    parse_runtime_overseer_json,
)
from oversight.contracts import (
    CoverageTarget,
    ExecutionContract,
    TaskChecklist,
    execution_contract_to_dict,
    make_checklist_cache_key,
    make_contract_cache_key,
    parse_execution_contract_json,
    parse_task_checklist_json,
    task_checklist_to_dict,
)
from oversight.triggers import (
    build_authoritative_state_snapshot,
    classify_mutating_tool,
    compute_coverage_status,
    detect_loop,
    detect_tool_error,
    normalize_arguments,
    render_transient_notice,
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
        self.tool_calls = None


class FakeChoice:
    def __init__(self, message: FakeMessage) -> None:
        self.index = 0
        self.finish_reason = "stop"
        self.message = message


class FakeResponse:
    def __init__(
        self,
        content: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        self.id = "resp_1"
        self.model = "fake-model"
        self.system_fingerprint = "fp_test"
        self.choices = [FakeChoice(FakeMessage(content))]
        self.usage = FakeUsage(prompt_tokens, completion_tokens)


class FakeAsyncClient:
    def __init__(self, create_func) -> None:
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=create_func),
        )

    async def __aenter__(self) -> FakeAsyncClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


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
        level_policy={
            "budget_priority": "primary",
            "coupon_reasoning_required": True,
            "allow_over_budget_explanation": False,
        },
        tool_semantics={
            "mutating_tools": ["add_product_to_cart", "delete_product_from_cart"],
            "read_only_tools": ["get_cart_info"],
            "search_tools": ["search_products"],
            "verification_tools": ["get_cart_info"],
        },
        final_output_requirements=["Use the authoritative cart state."],
        compiler_signature="overseer=deepseek-v3.2|mode=thinking|prompt=c2-lite-v1.2",
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
                "aliases": ["laptop", "notebook"],
            },
            {
                "key": "budget:under-1000",
                "category": "budget",
                "description": "Budget under 1000",
                "required": True,
                "explicit": True,
                "coverage_relevant": True,
                "final_verify_only": False,
                "aliases": ["under 1000", "$1000", "1000"],
            },
            {
                "key": "final:fresh-cart",
                "category": "final_requirement",
                "description": "Fresh cart read",
                "required": True,
                "explicit": False,
                "coverage_relevant": False,
                "final_verify_only": True,
                "aliases": [],
            },
        ],
        coverage_targets=[
            CoverageTarget(
                key="product:laptop",
                category="product",
                aliases=["laptop", "notebook"],
                tool_roles=["search"],
            ),
            CoverageTarget(
                key="budget:under-1000",
                category="budget",
                aliases=["under 1000", "$1000", "1000"],
                tool_roles=[],
            ),
        ],
        final_verification_only_keys=["final:fresh-cart"],
        ambiguities=[],
        compiler_signature="overseer=deepseek-v3.2|mode=thinking|prompt=c2-lite-v1.2",
    )


def test_normalize_arguments_canonicalizes_order_case_and_whitespace():
    arguments = ' { "Budget" : " Under 1000 ", "Query": " LAPTOP " } '

    assert normalize_arguments(arguments) == '{"Budget":"under 1000","Query":"laptop"}'


def test_detect_loop_fires_on_third_similar_call_in_last_five():
    recent_tool_history = [
        {
            "tool_name": "search_products",
            "arguments_normalized": '{"query":"laptop","sort":"price"}',
            "phase": "initial",
            "step_index": 1,
            "tool_index": 0,
        },
        {
            "tool_name": "search_products",
            "arguments_normalized": '{"query":" laptop ","sort":"price"}'.replace(
                " laptop ", "laptop"
            ),
            "phase": "initial",
            "step_index": 2,
            "tool_index": 0,
        },
    ]

    result = detect_loop(
        current_tool_name="search_products",
        current_arguments='{"sort":"price","query":"LAPTOP"}',
        recent_tool_history=recent_tool_history,
        similarity_threshold=0.92,
        window_size=5,
        repeat_threshold=3,
    )

    assert result["would_trigger"] is True
    assert result["match_count"] == 2
    assert len(result["candidate_matches"]) == 2


def test_detect_loop_does_not_fire_for_other_tool_names():
    result = detect_loop(
        current_tool_name="search_products",
        current_arguments='{"query":"laptop"}',
        recent_tool_history=[
            {
                "tool_name": "get_product_details",
                "arguments_normalized": '{"query":"laptop"}',
                "phase": "initial",
                "step_index": 1,
                "tool_index": 0,
            }
        ],
        similarity_threshold=0.92,
        window_size=5,
        repeat_threshold=3,
    )

    assert result["would_trigger"] is False
    assert result["match_count"] == 0


def test_classify_mutating_tool_matches_exact_allowlist():
    mutating_tools = {
        "add_product_to_cart",
        "delete_product_from_cart",
        "add_coupon_to_cart",
        "delete_coupon_from_cart",
    }

    assert (
        classify_mutating_tool(
            "add_product_to_cart",
            mutating_tools=mutating_tools,
        )["is_mutating"]
        is True
    )
    assert (
        classify_mutating_tool(
            "get_cart_info",
            mutating_tools=mutating_tools,
        )["is_mutating"]
        is False
    )


def test_detect_tool_error_on_error_dict_and_error_string():
    assert detect_tool_error('{"error":"out of stock"}') is True
    assert detect_tool_error("FAILED: invalid coupon") is True
    assert detect_tool_error('{"success":true,"items":[]}') is False


def test_build_authoritative_state_snapshot_uses_latest_get_cart_info():
    snapshot = build_authoritative_state_snapshot(
        [
            {
                "tool_name": "get_cart_info",
                "result_payload": {"items": ["old"]},
            },
            {
                "tool_name": "add_product_to_cart",
                "result_payload": {"success": True},
            },
            {
                "tool_name": "get_cart_info",
                "result_payload": {"items": ["new"]},
            },
        ]
    )

    assert snapshot == {"items": ["new"]}


def test_compute_coverage_status_counts_only_coverage_targets():
    checklist = _task_checklist()
    coverage = compute_coverage_status(
        checklist=checklist,
        tool_history=[
            {
                "phase": "initial",
                "tool_name": "search_products",
                "arguments_normalized": '{"query":"laptop under 1000"}',
                "result_summary": '["Budget laptop"]',
                "step_index": 1,
                "tool_index": 0,
            },
            {
                "phase": "cart_check",
                "tool_name": "get_cart_info",
                "arguments_normalized": "{}",
                "result_summary": '{"items":[]}',
                "step_index": 2,
                "tool_index": 0,
            },
        ],
    )

    assert coverage["total_coverage_targets"] == 2
    assert coverage["covered_coverage_targets"] == 2
    assert coverage["missing_keys"] == []


def test_render_transient_notice_is_deterministic_and_bounded():
    notice = render_transient_notice(
        trigger_type="coverage_deficit",
        lines=["Check laptop", "Check budget", "Check shipping", "Ignore extra"],
    )

    assert notice == render_transient_notice(
        trigger_type="coverage_deficit",
        lines=["Check laptop", "Check budget", "Check shipping", "Ignore extra"],
    )
    assert "Ignore extra" not in notice
    assert notice.count("\n1. ") == 1
    assert len(notice.split()) <= 140


def test_contract_parser_accepts_valid_p0_json():
    contract = parse_execution_contract_json(
        execution_contract_to_dict(_execution_contract())
    )

    assert contract.domain == "shopping"
    assert contract.compiler_signature.endswith("c2-lite-v1.2")


def test_checklist_parser_accepts_valid_p1_json():
    payload = task_checklist_to_dict(_task_checklist())
    payload["coverage_targets"] = [
        {
            "key": "product:laptop",
            "category": "product",
            "aliases": ["laptop"],
            "tool_roles": ["search"],
        }
    ]
    checklist = parse_task_checklist_json(payload)

    assert checklist.checklist_id == "checklist-1"
    assert checklist.coverage_targets[0].key == "product:laptop"


def test_runtime_overseer_parser_accepts_valid_p2_json():
    payload_v13 = {
        "action": "provide_guidance",
        "decision_summary": "Premature mutation",
        "violation_evidence": {
            "violated_contract_ids": ["mut.precondition.1"],
            "unmet_checklist_keys": ["item:nike_orange_footwear"],
            "confidence": "medium",
        },
        "guidance_lines": ["Verify product category before adding to cart."],
        "corrected_observation": None,
    }
    parsed = parse_runtime_overseer_json(payload_v13)

    assert parsed["action"] == "provide_guidance"
    assert parsed["violated_contract_ids"] == ["mut.precondition.1"]
    assert parsed["unmet_checklist_keys"] == ["item:nike_orange_footwear"]
    assert parsed["violation_confidence"] == "medium"
    assert parsed["missing_corrective_content"] is False

    payload_v12 = {
        "action": "provide_guidance",
        "decision_summary": "Premature mutation",
        "block_current_tool": True,
        "violated_contract_ids": ["mut.precondition.1"],
        "unmet_checklist_keys": ["item:nike_orange_footwear"],
        "guidance_lines": ["Verify product category before adding to cart."],
        "corrected_observation": None,
    }
    parsed_v12 = parse_runtime_overseer_json(payload_v12)

    assert parsed_v12["action"] == "provide_guidance"
    assert parsed_v12["block_current_tool"] is True
    assert parsed_v12["violated_contract_ids"] == parsed["violated_contract_ids"]
    assert parsed_v12["unmet_checklist_keys"] == parsed["unmet_checklist_keys"]
    assert parsed_v12["violation_confidence"] == "low"


def test_h1_gate_approves_reversible_mutation_by_default():
    action = OversightAction(
        intervention_type="provide_guidance",
        violated_contract_ids=[],
        unmet_checklist_keys=[],
        violation_confidence="low",
    )
    state = ConversationState(
        task_id="t1",
        domain="shopping",
        complexity=1,
        system_config_name="C2",
    )
    system_cfg = SimpleNamespace(
        mutating_tools=("add_product_to_cart",),
        irreversible_tools=(),
        block_on_mutation_mode="auto",
        max_hard_blocks_per_args=2,
        require_cited_violation_for_block=True,
        overseer_call_budget_per_task=8,
    )

    outcome = compute_h1_outcome(
        action=action,
        tool_name="add_product_to_cart",
        arguments={"product_id": "p1"},
        state=state,
        system_config=system_cfg,
    )

    assert outcome == H1Outcome.APPROVE_WITH_NUDGE


def test_h1_gate_escalates_to_forced_approve_after_cap():
    action = OversightAction(
        intervention_type="provide_guidance",
        violated_contract_ids=["c.1"],
        unmet_checklist_keys=[],
        violation_confidence="high",
    )
    state = ConversationState(
        task_id="t1",
        domain="shopping",
        complexity=1,
        system_config_name="C2",
    )
    args = {"product_id": "p1"}

    from oversight.state import _hash_arguments

    state.blocked_mutation_counts[
        ("irreversible_book_flight", _hash_arguments(args))
    ] = 2
    system_cfg = SimpleNamespace(
        mutating_tools=("irreversible_book_flight",),
        irreversible_tools=("irreversible_book_flight",),
        block_on_mutation_mode="auto",
        max_hard_blocks_per_args=2,
        require_cited_violation_for_block=True,
        overseer_call_budget_per_task=8,
    )

    outcome = compute_h1_outcome(
        action=action,
        tool_name="irreversible_book_flight",
        arguments=args,
        state=state,
        system_config=system_cfg,
    )

    assert outcome == H1Outcome.FORCED_APPROVE


def test_h1_gate_always_mode_restores_v12_semantics():
    action = OversightAction(
        intervention_type="provide_guidance",
        violated_contract_ids=[],
        unmet_checklist_keys=[],
        violation_confidence="low",
    )
    state = ConversationState(
        task_id="t1",
        domain="shopping",
        complexity=1,
        system_config_name="C2",
    )
    system_cfg = SimpleNamespace(
        mutating_tools=("add_product_to_cart",),
        irreversible_tools=(),
        block_on_mutation_mode="always",
        max_hard_blocks_per_args=2,
        require_cited_violation_for_block=True,
        overseer_call_budget_per_task=8,
    )

    outcome = compute_h1_outcome(
        action=action,
        tool_name="add_product_to_cart",
        arguments={"product_id": "p1"},
        state=state,
        system_config=system_cfg,
    )

    assert outcome == H1Outcome.HARD_BLOCK


def test_runtime_overseer_empty_guidance_uses_local_fallback_notice(monkeypatch):
    async def fake_call_chat_completion(**kwargs):
        return FakeResponse(
            json.dumps(
                {
                    "action": "provide_guidance",
                    "decision_summary": "Blocked pending correction.",
                    "block_current_tool": True,
                    "guidance_lines": [],
                    "corrected_observation": None,
                    "violated_contract_ids": ["rule-1"],
                    "unmet_checklist_keys": ["product:footwear"],
                }
            ),
            prompt_tokens=12,
            completion_tokens=4,
        )

    import oversight as oversight_module

    monkeypatch.setattr(
        oversight_module, "call_chat_completion", fake_call_chat_completion
    )

    state = ConversationState(
        task_id="1",
        domain="shopping",
        complexity=1,
        system_config_name="C2",
    )
    state.execution_contract = _execution_contract()
    state.task_checklist = _task_checklist()
    system_config = build_system_config("C2", executor_model="qwen3.5-9b", max_steps=2)

    action = asyncio.run(
        evaluate_oversight(
            hook="pre_tool",
            state=state,
            system_config=system_config,
            phase="initial",
            task_query="build a footwear collection",
            proposed_tool_calls=[
                {
                    "id": "call_1",
                    "name": "add_product_to_cart",
                    "arguments": '{"product_id":"1"}',
                }
            ],
            step_index=1,
        )
    )
    asyncio.run(apply_intervention(state=state, action=action))

    assert action.should_intervene is True
    assert action.notice_rendered is True
    assert action.notice_source == "local_fallback"
    assert action.fallback_guidance_used is True
    assert state.pending_executor_notice is not None
    assert (
        "Re-check task requirement: product footwear." in state.pending_executor_notice
    )
    assert action.h1_outcome in (None, "approve_with_nudge")


def test_final_verifier_parser_accepts_valid_p3_json():
    parsed = parse_final_verifier_json(
        {
            "action": "run_verification",
            "pass": False,
            "decision_summary": "Budget not satisfied.",
            "blockers": [{"type": "budget", "detail": "Over budget"}],
            "next_step_notice_lines": ["Call get_cart_info."],
            "violated_contract_ids": ["rule-1"],
            "unmet_checklist_keys": ["budget:under-1000"],
        }
    )

    assert parsed["action"] == "run_verification"
    assert parsed["pass"] is False
    assert parsed["blockers"][0]["type"] == "budget"


def test_runtime_overseer_invalid_json_falls_back_to_approve(monkeypatch):
    async def fake_call_chat_completion(**kwargs):
        return FakeResponse("{not valid json", prompt_tokens=12, completion_tokens=4)

    import oversight as oversight_module

    monkeypatch.setattr(
        oversight_module, "call_chat_completion", fake_call_chat_completion
    )

    state = ConversationState(
        task_id="1",
        domain="shopping",
        complexity=1,
        system_config_name="C2",
    )
    state.execution_contract = _execution_contract()
    state.task_checklist = _task_checklist()
    system_config = build_system_config("C2", executor_model="qwen3.5-9b", max_steps=2)

    action = asyncio.run(
        evaluate_oversight(
            hook="pre_tool",
            state=state,
            system_config=system_config,
            phase="initial",
            task_query="buy a laptop under 1000",
            proposed_tool_calls=[
                {
                    "id": "call_1",
                    "name": "add_product_to_cart",
                    "arguments": '{"product_id":"1"}',
                }
            ],
            step_index=1,
        )
    )

    assert action.should_intervene is False
    assert action.overseer_mode == "thinking"
    assert state.overseer_calls == 1


def test_checklist_parser_does_not_promote_global_framing_to_item_product_type():
    checklist = parse_task_checklist_json(
        {
            "checklist_id": "checklist-1",
            "items": [
                {
                    "key": "nike_orange_product",
                    "category": "required_product",
                    "description": "A Nike product in orange with strong reviews",
                    "value": {"brand": "Nike", "color": "orange"},
                    "required": True,
                    "explicit": True,
                    "coverage_relevant": True,
                    "final_verify_only": False,
                    "aliases": ["Nike orange item"],
                    "source_text": "I need something from Nike in orange.",
                }
            ],
            "coverage_targets": [
                {
                    "key": "coverage_nike",
                    "category": "product",
                    "aliases": ["coverage_nike"],
                    "tool_roles": ["search"],
                }
            ],
            "final_verification_only_keys": [],
            "ambiguities": [],
            "compiler_signature": "sig",
        },
        task_query="I'm putting together a complete footwear collection and need something from Nike in orange.",
    )

    item = checklist.items[0]
    assert item["value"].get("product_type") in (None, "")
    assert "footwear" not in item["value"].get("product_type_hints_soft", [])
    assert "footwear" not in [alias.lower() for alias in item.get("aliases", [])]
    assert checklist.coverage_targets[0].key == "nike_orange_product"
    assert "footwear" not in [
        alias.lower() for alias in checklist.coverage_targets[0].aliases
    ]


def test_infer_product_types_uses_item_local_text_only():
    from oversight.contracts import _infer_product_types

    hints, evidence = _infer_product_types(
        task_query="I'm putting together a complete footwear collection.",
        item={
            "category": "required_product",
            "description": "Nike Pro Dri-FIT Long-Sleeve Training Top",
            "source_text": "A Nike training top.",
            "value": {"name": "Training Top", "brand": "Nike"},
        },
    )

    assert hints == []
    assert evidence == {}


def test_infer_product_types_promotes_hard_only_with_multiple_local_spans():
    from oversight.contracts import _normalize_checklist_item

    item = {
        "key": "generic_shoe",
        "category": "required_product",
        "description": "Casual weekend shoe",
        "source_text": "I want something casual.",
        "value": {"name": "Casual pick"},
        "aliases": [],
    }

    normalized, ambiguity_entry = _normalize_checklist_item(
        item=item,
        task_query="anything you want",
    )

    assert normalized["value"].get("product_type") in (None, "")
    assert "shoes" in normalized["value"].get("product_type_hints_soft", [])
    assert ambiguity_entry is not None
    assert "generic_shoe" in ambiguity_entry


def test_checklist_invariants_reject_hard_type_without_local_support():
    from oversight.contracts import (
        ChecklistInvariantError,
        _validate_checklist_invariants,
    )

    bad = TaskChecklist(
        checklist_id="c-1",
        items=[
            {
                "key": "bad_item",
                "category": "required_product",
                "description": "Nike training top",
                "source_text": "I need a Nike item.",
                "value": {
                    "name": "Training Top",
                    "brand": "Nike",
                    "product_type": "footwear",
                    "product_type_aliases": ["footwear"],
                },
                "aliases": [],
            }
        ],
        coverage_targets=[
            CoverageTarget(
                key="bad_item",
                category="product",
                aliases=[],
                tool_roles=["search"],
            )
        ],
        final_verification_only_keys=[],
        ambiguities=[],
        compiler_signature="sig",
    )

    try:
        _validate_checklist_invariants(bad)
    except ChecklistInvariantError as exc:
        assert "local textual support" in str(exc)
    else:
        raise AssertionError(
            "Expected ChecklistInvariantError for hard type without local support"
        )


def test_cache_keys_differ_between_thinking_and_non_thinking_compilers():
    contract_key_thinking = make_contract_cache_key(
        domain="shopping",
        executor_system_prompt="prompt",
        tool_schema=[{"type": "function"}],
        compiler_signature="overseer=deepseek-v3.2|mode=thinking|prompt=c2-lite-v1.2",
    )
    contract_key_non_thinking = make_contract_cache_key(
        domain="shopping",
        executor_system_prompt="prompt",
        tool_schema=[{"type": "function"}],
        compiler_signature="overseer=deepseek-v3.2|mode=non-thinking|prompt=c2-lite-v1.2",
    )
    checklist_key_thinking = make_checklist_cache_key(
        task_id="1",
        task_query="buy laptop",
        contract_id="contract-shopping",
        compiler_signature="overseer=deepseek-v3.2|mode=thinking|prompt=c2-lite-v1.2",
    )
    checklist_key_non_thinking = make_checklist_cache_key(
        task_id="1",
        task_query="buy laptop",
        contract_id="contract-shopping",
        compiler_signature="overseer=deepseek-v3.2|mode=non-thinking|prompt=c2-lite-v1.2",
    )

    assert contract_key_thinking != contract_key_non_thinking
    assert checklist_key_thinking != checklist_key_non_thinking


def test_call_chat_completion_overrides_deepseek_thinking_flag(monkeypatch):
    captured_params: list[dict[str, object]] = []

    async def fake_create(**kwargs):
        captured_params.append(kwargs)
        return FakeResponse(content="done")

    monkeypatch.setattr(
        llm_client,
        "_build_async_client",
        lambda provider, api_key: FakeAsyncClient(fake_create),
    )
    provider = llm_client.ProviderConfig(
        alias="deepseek-v3.2",
        model="deepseek-ai/deepseek-v3.2",
        provider="openai",
        extra_body={"chat_template_kwargs": {"thinking": True}},
    )

    asyncio.run(
        llm_client.call_chat_completion(
            provider=provider,
            messages=[{"role": "user", "content": "hello"}],
            reasoning_enabled=False,
        )
    )

    assert captured_params[0]["extra_body"]["reasoning"]["enabled"] is False
    assert captured_params[0]["extra_body"]["chat_template_kwargs"]["thinking"] is False
