from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

import oversight as oversight_module
from agent import shopping as shopping_module
from agent import travel as travel_module
from deepplanning import aggregation as aggregation_module
from deepplanning import orchestration as orchestration_module
from deepplanning import shopping_runner as shopping_runtime_module
from deepplanning import travel_runner as travel_runtime_module
from experiment import StructuredLogger, build_system_config
from oversight import ConversationState
from oversight import contracts as oversight_contracts
from scripts import run_experiment as experiment_script
from scripts.deepplanning_common import (
    filter_samples_by_ids,
    load_json_file,
    parse_id_list,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SHOPPING_FIXTURE_ROOT = (
    REPO_ROOT / "data" / "deepplanning" / "shopping" / "database_level1"
)
SHOPPING_SCHEMA_PATH = (
    REPO_ROOT
    / "external"
    / "qwen-agent"
    / "benchmark"
    / "deepplanning"
    / "shoppingplanning"
    / "tools"
    / "shopping_tool_schema.json"
)
TRAVEL_DATABASE_DIR = (
    REPO_ROOT / "data" / "deepplanning" / "travel" / "database" / "database_en"
)
TRAVEL_SCHEMA_PATH = (
    REPO_ROOT
    / "external"
    / "qwen-agent"
    / "benchmark"
    / "deepplanning"
    / "travelplanning"
    / "tools"
    / "tool_schema_en.json"
)


class FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class FakeToolFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.type = "function"
        self.function = FakeToolFunction(name, arguments)


class FakeMessage:
    def __init__(
        self,
        content: str,
        tool_calls: list[FakeToolCall] | None = None,
        reasoning_content: str | None = None,
    ) -> None:
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content


class FakeChoice:
    def __init__(self, message: FakeMessage, finish_reason: str) -> None:
        self.index = 0
        self.finish_reason = finish_reason
        self.message = message


class FakeResponse:
    def __init__(
        self,
        content: str,
        tool_calls: list[FakeToolCall] | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        finish_reason: str = "stop",
        reasoning_content: str | None = None,
    ) -> None:
        self.id = "resp_1"
        self.model = "fake-model"
        self.system_fingerprint = "fp_test"
        self.choices = [
            FakeChoice(
                FakeMessage(
                    content=content,
                    tool_calls=tool_calls,
                    reasoning_content=reasoning_content,
                ),
                finish_reason=finish_reason,
            )
        ]
        self.usage = FakeUsage(prompt_tokens, completion_tokens)


def _fake_json_response(
    payload: dict[str, object],
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> FakeResponse:
    return FakeResponse(
        content=json.dumps(payload),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _shopping_contract_payload() -> dict[str, object]:
    return {
        "contract_id": "contract-shopping",
        "domain": "shopping",
        "primary_objective": "Build the correct cart.",
        "objective_priority": ["requirements", "budget"],
        "hard_rules": [{"id": "rule-1", "text": "Stay within budget."}],
        "state_authority_rules": [
            {"state": "cart", "tool": "get_cart_info", "authoritative": True}
        ],
        "level_policy": {
            "budget_priority": "primary",
            "coupon_reasoning_required": True,
            "allow_over_budget_explanation": False,
        },
        "tool_semantics": {
            "mutating_tools": [
                "add_product_to_cart",
                "delete_product_from_cart",
                "add_coupon_to_cart",
                "delete_coupon_from_cart",
            ],
            "read_only_tools": ["get_cart_info", "search_products"],
            "search_tools": ["search_products"],
            "verification_tools": ["get_cart_info"],
        },
        "final_output_requirements": ["Use the authoritative cart state."],
    }


def _shopping_checklist_payload(
    *,
    coverage_targets: list[dict[str, object]] | None = None,
    items: list[dict[str, object]] | None = None,
    final_verification_only_keys: list[str] | None = None,
) -> dict[str, object]:
    return {
        "checklist_id": "checklist-1",
        "items": items
        or [
            {
                "key": "final:fresh-cart",
                "category": "final_requirement",
                "description": "Fresh cart read",
                "value": None,
                "required": True,
                "explicit": False,
                "coverage_relevant": False,
                "final_verify_only": True,
                "aliases": [],
                "source_text": None,
            }
        ],
        "coverage_targets": coverage_targets or [],
        "final_verification_only_keys": final_verification_only_keys
        or ["final:fresh-cart"],
        "ambiguities": [],
    }


def _fake_completion_factory(
    responses: list[FakeResponse],
    captured_calls: list[dict[str, object]] | None = None,
):
    queue = list(responses)

    async def _fake_call(*args, **kwargs):
        if captured_calls is not None:
            captured_calls.append(kwargs)
        return queue.pop(0)

    return _fake_call


def _install_c2_lite_mocks(
    monkeypatch,
    *,
    executor_responses: list[FakeResponse],
    compiler_responses: list[FakeResponse] | None = None,
    overseer_responses: list[FakeResponse] | None = None,
    captured_executor_calls: list[dict[str, object]] | None = None,
) -> None:
    monkeypatch.setattr(
        shopping_module,
        "call_chat_completion",
        _fake_completion_factory(executor_responses, captured_executor_calls),
    )
    monkeypatch.setattr(
        oversight_contracts,
        "call_chat_completion",
        _fake_completion_factory(
            compiler_responses
            or [
                _fake_json_response(_shopping_contract_payload()),
                _fake_json_response(_shopping_checklist_payload()),
            ]
        ),
    )
    monkeypatch.setattr(
        oversight_module,
        "call_chat_completion",
        _fake_completion_factory(overseer_responses or []),
    )


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_travel_runner_matches_benchmark_loop_and_logs_turns(monkeypatch, tmp_path):
    captured_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        travel_module,
        "call_chat_completion",
        _fake_completion_factory(
            [
                FakeResponse(
                    content="",
                    tool_calls=[
                        FakeToolCall(
                            "call_1",
                            "query_train_info",
                            '{"departure":"A","arrival":"B"}',
                        )
                    ],
                    prompt_tokens=20,
                    completion_tokens=7,
                    finish_reason="tool_calls",
                ),
                FakeResponse(
                    content="<think>reasoning</think><plan>Day 1 itinerary</plan>",
                    prompt_tokens=11,
                    completion_tokens=5,
                ),
            ],
            captured_calls,
        ),
    )
    monkeypatch.setattr(
        travel_module.TravelAgentRunner,
        "_exec_tool",
        lambda self, name, arguments: '{"status":"ok"}',
    )

    runner = travel_module.TravelAgentRunner(
        model="qwen-plus",
        sample_id="0",
        database_base_path=str(TRAVEL_DATABASE_DIR),
        tool_schema_path=str(TRAVEL_SCHEMA_PATH),
        language="en",
    )
    state = ConversationState(
        task_id="id_0",
        domain="travel",
        complexity=2,
        system_config_name="A",
    )
    logger = StructuredLogger(tmp_path / "travel_logs")
    system_config = build_system_config("A", executor_model="qwen-plus", max_steps=2)

    result = asyncio.run(
        runner.run_task(
            user_query="test travel query",
            system_prompt=travel_module.get_system_prompt("en"),
            state=state,
            system_config=system_config,
            logger=logger,
        )
    )

    assert result.output == "Day 1 itinerary"
    assert state.executor_calls == 2
    assert state.final_stop_reason == "no_tool_calls"
    assert state.final_output_present is True
    assert "reasoning_enabled" not in captured_calls[0]
    assert "reasoning_enabled" not in captured_calls[1]
    assistant_message = captured_calls[1]["messages"][2]
    assert assistant_message.role == "assistant"
    assert assistant_message.content == ""
    assert assistant_message.tool_calls[0].id == "call_1"
    assert assistant_message.tool_calls[0].function.name == "query_train_info"
    assert assistant_message.tool_calls[0].function.arguments == (
        '{"departure":"A","arrival":"B"}'
    )
    assert any(
        isinstance(message, dict)
        and message.get("role") == "tool"
        and message.get("name") == "query_train_info"
        for message in result.messages
    )

    records = _load_jsonl(tmp_path / "travel_logs" / "agent_events.jsonl")
    assert len(records) == 2
    assert records[0]["event_type"] == "executor_turn"
    assert records[0]["parsed_tool_calls"] == [
        {
            "id": "call_1",
            "name": "query_train_info",
            "arguments": '{"departure":"A","arrival":"B"}',
        }
    ]
    assert records[0]["tool_results"] == [
        {
            "tool_name": "query_train_info",
            "tool_call_id": "call_1",
            "content": '{"status":"ok"}',
        }
    ]
    assert records[1]["stop_reason"] == "no_tool_calls"


def test_travel_runner_loads_vendored_tool_instances():
    runner = travel_module.TravelAgentRunner(
        model="qwen-plus",
        sample_id="0",
        database_base_path=str(TRAVEL_DATABASE_DIR),
        tool_schema_path=str(TRAVEL_SCHEMA_PATH),
        language="en",
    )

    assert "query_train_info" in runner.tool_instances
    assert "query_hotel_info" in runner.tool_instances
    assert "recommend_restaurants" in runner.tool_instances
    assert "recommend_around_restaurants" in runner.tool_instances
    assert "recommend_around_restaurants" in [
        tool["function"]["name"] for tool in runner.openai_tools
    ]


def test_cross_domain_agent_initialization_keeps_tool_registries_isolated():
    shopping_runner = shopping_module.ShoppingAgentRunner(
        model="qwen-plus",
        sample_id="1",
        database_base_path=str(SHOPPING_FIXTURE_ROOT),
        tool_schema_path=str(SHOPPING_SCHEMA_PATH),
    )
    travel_runner = travel_module.TravelAgentRunner(
        model="qwen-plus",
        sample_id="0",
        database_base_path=str(TRAVEL_DATABASE_DIR),
        tool_schema_path=str(TRAVEL_SCHEMA_PATH),
        language="en",
    )

    assert "search_products" in shopping_runner.tool_instances
    assert "query_train_info" in travel_runner.tool_instances
    assert "query_hotel_info" in travel_runner.tool_instances
    assert "recommend_around_restaurants" in travel_runner.tool_instances


def test_travel_runner_stops_immediately_on_no_tool_response(monkeypatch):
    monkeypatch.setattr(
        travel_module,
        "call_chat_completion",
        _fake_completion_factory(
            [
                FakeResponse(
                    content="<think>reasoning</think><plan>Day 1 itinerary</plan>",
                    prompt_tokens=20,
                    completion_tokens=7,
                )
            ]
        ),
    )

    runner = travel_module.TravelAgentRunner(
        model="qwen-plus",
        sample_id="0",
        database_base_path=str(TRAVEL_DATABASE_DIR),
        tool_schema_path=str(TRAVEL_SCHEMA_PATH),
        language="en",
    )
    state = ConversationState(
        task_id="id_0",
        domain="travel",
        complexity=2,
        system_config_name="A",
    )
    system_config = build_system_config("A", executor_model="qwen-plus", max_steps=2)

    result = asyncio.run(
        runner.run_task(
            user_query="test travel query",
            system_prompt=travel_module.get_system_prompt("en"),
            state=state,
            system_config=system_config,
        )
    )

    assert result.output == "Day 1 itinerary"
    assert state.executor_calls == 1
    assert state.final_stop_reason == "no_tool_calls"
    assert result.messages[-1].role == "assistant"
    assert result.messages[-1].content == (
        "<think>reasoning</think><plan>Day 1 itinerary</plan>"
    )


def test_shopping_runner_breaks_phase_one_and_adds_cart_message(monkeypatch, tmp_path):
    run_database_dir = tmp_path / "shopping_db"
    shutil.copytree(SHOPPING_FIXTURE_ROOT, run_database_dir)

    captured_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        shopping_module,
        "call_chat_completion",
        _fake_completion_factory(
            [
                FakeResponse(
                    content="Phase one complete.",
                    prompt_tokens=10,
                    completion_tokens=4,
                ),
                FakeResponse(
                    content="Cart verified.",
                    prompt_tokens=11,
                    completion_tokens=5,
                ),
            ],
            captured_calls,
        ),
    )

    runner = shopping_module.ShoppingAgentRunner(
        model="qwen-plus",
        sample_id="1",
        database_base_path=str(run_database_dir),
        tool_schema_path=str(SHOPPING_SCHEMA_PATH),
    )
    state = ConversationState(
        task_id="1",
        domain="shopping",
        complexity=1,
        system_config_name="A",
    )
    logger = StructuredLogger(tmp_path / "shopping_logs")
    system_config = build_system_config("A", executor_model="qwen-plus", max_steps=2)

    result = asyncio.run(
        runner.run_task(
            user_query="test shopping query",
            system_prompt=shopping_module.get_system_prompt(1),
            state=state,
            system_config=system_config,
            logger=logger,
            sample_id="1",
        )
    )

    assert result.output == "Cart verified."
    assert state.executor_calls == 2
    assert state.final_stop_reason == "no_tool_calls"
    assert "reasoning_enabled" not in captured_calls[0]
    assert "reasoning_enabled" not in captured_calls[1]
    assert any(
        isinstance(message, dict)
        and message.get("role") == "user"
        and "Check whether the items in the shopping cart meet the requirements."
        in message.get("content", "")
        for message in captured_calls[1]["messages"]
    )
    assert (run_database_dir / "case_1" / "messages.json").exists()

    records = _load_jsonl(tmp_path / "shopping_logs" / "agent_events.jsonl")
    assert [record["phase"] for record in records] == ["initial", "cart_check"]
    assert records[1]["request_messages"][-1]["role"] == "user"


def test_shopping_runner_gives_phase_two_a_fresh_budget_and_omits_tool_name(
    monkeypatch, tmp_path
):
    run_database_dir = tmp_path / "shopping_db"
    shutil.copytree(SHOPPING_FIXTURE_ROOT, run_database_dir)

    captured_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        shopping_module,
        "call_chat_completion",
        _fake_completion_factory(
            [
                FakeResponse(
                    content="",
                    tool_calls=[FakeToolCall("call_1", "get_cart_info", "{}")],
                    prompt_tokens=10,
                    completion_tokens=4,
                    finish_reason="tool_calls",
                ),
                FakeResponse(
                    content="Cart verified.",
                    prompt_tokens=11,
                    completion_tokens=5,
                ),
            ],
            captured_calls,
        ),
    )
    monkeypatch.setattr(
        shopping_module.ShoppingAgentRunner,
        "_exec_tool",
        lambda self, name, arguments: '{"cart":"ok"}',
    )

    runner = shopping_module.ShoppingAgentRunner(
        model="qwen-plus",
        sample_id="1",
        database_base_path=str(run_database_dir),
        tool_schema_path=str(SHOPPING_SCHEMA_PATH),
    )
    state = ConversationState(
        task_id="1",
        domain="shopping",
        complexity=1,
        system_config_name="A",
    )
    system_config = build_system_config("A", executor_model="qwen-plus", max_steps=1)

    result = asyncio.run(
        runner.run_task(
            user_query="test shopping query",
            system_prompt=shopping_module.get_system_prompt(1),
            state=state,
            system_config=system_config,
            sample_id="1",
        )
    )

    assert result.output == "Cart verified."
    assert state.executor_calls == 2
    assert state.tool_call_count == 1
    assert "reasoning_enabled" not in captured_calls[0]
    assert "reasoning_enabled" not in captured_calls[1]
    assert any(
        isinstance(message, dict)
        and message.get("role") == "tool"
        and message.get("tool_call_id") == "call_1"
        and "name" not in message
        for message in result.messages
    )


def test_system_a_behavior_unchanged_when_oversight_disabled(monkeypatch, tmp_path):
    run_database_dir = tmp_path / "shopping_db"
    shutil.copytree(SHOPPING_FIXTURE_ROOT, run_database_dir)

    captured_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        shopping_module,
        "call_chat_completion",
        _fake_completion_factory(
            [
                FakeResponse(
                    content="Phase one complete.",
                    prompt_tokens=10,
                    completion_tokens=4,
                ),
                FakeResponse(
                    content="Cart verified.",
                    prompt_tokens=11,
                    completion_tokens=5,
                ),
            ],
            captured_calls,
        ),
    )

    runner = shopping_module.ShoppingAgentRunner(
        model="qwen-plus",
        sample_id="1",
        database_base_path=str(run_database_dir),
        tool_schema_path=str(SHOPPING_SCHEMA_PATH),
    )
    state = ConversationState(
        task_id="1",
        domain="shopping",
        complexity=1,
        system_config_name="A",
    )
    logger = StructuredLogger(tmp_path / "shopping_logs")
    system_config = build_system_config("A", executor_model="qwen-plus", max_steps=2)

    result = asyncio.run(
        runner.run_task(
            user_query="test shopping query",
            system_prompt=shopping_module.get_system_prompt(1),
            state=state,
            system_config=system_config,
            logger=logger,
            sample_id="1",
        )
    )

    assert result.output == "Cart verified."
    assert state.executor_calls == 2
    assert state.final_stop_reason == "no_tool_calls"
    assert any(
        isinstance(message, dict)
        and message.get("role") == "user"
        and "Check whether the items in the shopping cart meet the requirements."
        in message.get("content", "")
        for message in captured_calls[1]["messages"]
    )
    records = _load_jsonl(tmp_path / "shopping_logs" / "agent_events.jsonl")
    assert [record["event_type"] for record in records] == [
        "executor_turn",
        "executor_turn",
    ]
    assert not any(record["event_type"] == "oversight_step" for record in records)


def test_shopping_runner_saves_debug_messages_outside_agent_dir_without_sample_id(
    monkeypatch, tmp_path
):
    run_database_dir = tmp_path / "shopping_db"
    shutil.copytree(SHOPPING_FIXTURE_ROOT, run_database_dir)

    monkeypatch.setattr(
        shopping_module,
        "call_chat_completion",
        _fake_completion_factory(
            [
                FakeResponse(
                    content="Phase one complete.",
                    prompt_tokens=10,
                    completion_tokens=4,
                ),
                FakeResponse(
                    content="Cart verified.",
                    prompt_tokens=11,
                    completion_tokens=5,
                ),
            ]
        ),
    )
    monkeypatch.setattr(
        shopping_module,
        "DEBUG_MESSAGES_ROOT",
        tmp_path / "debug_messages" / "shopping",
    )

    runner = shopping_module.ShoppingAgentRunner(
        model="qwen-plus",
        database_base_path=str(run_database_dir),
        tool_schema_path=str(SHOPPING_SCHEMA_PATH),
    )
    state = ConversationState(
        task_id="debug-run",
        domain="shopping",
        complexity=1,
        system_config_name="A",
    )
    system_config = build_system_config("A", executor_model="qwen-plus", max_steps=2)

    result = asyncio.run(
        runner.run_task(
            user_query="test shopping query",
            system_prompt=shopping_module.get_system_prompt(1),
            state=state,
            system_config=system_config,
            save_messages=True,
            sample_id=None,
        )
    )

    assert result.output == "Cart verified."
    debug_messages = sorted(
        (tmp_path / "debug_messages" / "shopping").glob("messages_*.json")
    )
    assert len(debug_messages) == 1
    assert REPO_ROOT / "agent" not in debug_messages[0].resolve().parents


def test_pre_tool_blocked_mutation_does_not_persist_rejected_tool_call_message(
    monkeypatch, tmp_path
):
    run_database_dir = tmp_path / "shopping_db"
    shutil.copytree(SHOPPING_FIXTURE_ROOT, run_database_dir)

    captured_calls: list[dict[str, object]] = []
    _install_c2_lite_mocks(
        monkeypatch,
        executor_responses=[
            FakeResponse(
                content="",
                tool_calls=[
                    FakeToolCall(
                        "call_1",
                        "add_product_to_cart",
                        '{"product_id":"1"}',
                    )
                ],
                prompt_tokens=10,
                completion_tokens=4,
                finish_reason="tool_calls",
            ),
            FakeResponse(
                content="Phase one complete.",
                prompt_tokens=11,
                completion_tokens=5,
            ),
            FakeResponse(
                content="Final cart answer.",
                prompt_tokens=12,
                completion_tokens=6,
            ),
        ],
        overseer_responses=[
            _fake_json_response(
                {
                    "action": "provide_guidance",
                    "decision_summary": "Read the cart before mutating it.",
                    "block_current_tool": True,
                    "guidance_lines": ["Call get_cart_info before mutating the cart."],
                    "corrected_observation": None,
                    "violated_contract_ids": ["rule-1"],
                    "unmet_checklist_keys": ["final:fresh-cart"],
                }
            ),
            _fake_json_response(
                {
                    "action": "approve",
                    "pass": True,
                    "decision_summary": "Final answer approved.",
                    "blockers": [],
                    "next_step_notice_lines": [],
                    "violated_contract_ids": [],
                    "unmet_checklist_keys": [],
                }
            ),
        ],
        captured_executor_calls=captured_calls,
    )

    runner = shopping_module.ShoppingAgentRunner(
        model="qwen3.5-9b",
        sample_id="1",
        database_base_path=str(run_database_dir),
        tool_schema_path=str(SHOPPING_SCHEMA_PATH),
    )
    state = ConversationState(
        task_id="1",
        domain="shopping",
        complexity=1,
        system_config_name="C2",
    )
    logger = StructuredLogger(tmp_path / "shopping_logs")
    system_config = build_system_config("C2", executor_model="qwen3.5-9b", max_steps=3)

    result = asyncio.run(
        runner.run_task(
            user_query="buy a laptop",
            system_prompt=shopping_module.get_system_prompt(1),
            state=state,
            system_config=system_config,
            logger=logger,
            sample_id="1",
            shared_oversight_cache_root=tmp_path / "cache",
        )
    )

    assert result.output == "Final cart answer."
    assert state.blocked_mutation_count == 1
    assert not any(
        isinstance(message, dict)
        and message.get("role") == "assistant"
        and message.get("tool_calls")
        for message in result.messages
    )
    follow_up_messages = captured_calls[1]["messages"]
    assert follow_up_messages[-1]["role"] == "user"
    assert (
        "Call get_cart_info before mutating the cart."
        in follow_up_messages[-1]["content"]
    )
    records = _load_jsonl(tmp_path / "shopping_logs" / "agent_events.jsonl")
    oversight_steps = [
        record
        for record in records
        if record["event_type"] == "oversight_step"
        and record.get("trigger_type") == "mutating_action"
    ]
    assert oversight_steps[0]["parsed_payload"]["action"] == "provide_guidance"
    assert oversight_steps[0]["raw_overseer_text"] is not None
    assert oversight_steps[0]["notice_rendered"] is True
    assert any(
        record["event_type"] == "oversight_notice_injected"
        and record.get("notice_role") == "user"
        and "Call get_cart_info before mutating the cart." in record["notice_text"]
        for record in records
    )


def test_pre_tool_empty_guidance_uses_local_fallback_and_clears_notice_after_injection(
    monkeypatch, tmp_path
):
    run_database_dir = tmp_path / "shopping_db"
    shutil.copytree(SHOPPING_FIXTURE_ROOT, run_database_dir)

    captured_calls: list[dict[str, object]] = []
    _install_c2_lite_mocks(
        monkeypatch,
        executor_responses=[
            FakeResponse(
                content="",
                tool_calls=[
                    FakeToolCall(
                        "call_1",
                        "add_product_to_cart",
                        '{"product_id":"1"}',
                    )
                ],
                prompt_tokens=10,
                completion_tokens=4,
                finish_reason="tool_calls",
            ),
            FakeResponse(
                content="Phase one complete.",
                prompt_tokens=11,
                completion_tokens=5,
            ),
            FakeResponse(
                content="Final cart answer.",
                prompt_tokens=12,
                completion_tokens=6,
            ),
        ],
        overseer_responses=[
            _fake_json_response(
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
            _fake_json_response(
                {
                    "action": "approve",
                    "pass": True,
                    "decision_summary": "Final answer approved.",
                    "blockers": [],
                    "next_step_notice_lines": [],
                    "violated_contract_ids": [],
                    "unmet_checklist_keys": [],
                }
            ),
        ],
        captured_executor_calls=captured_calls,
    )

    runner = shopping_module.ShoppingAgentRunner(
        model="qwen3.5-9b",
        sample_id="1",
        database_base_path=str(run_database_dir),
        tool_schema_path=str(SHOPPING_SCHEMA_PATH),
    )
    state = ConversationState(
        task_id="1",
        domain="shopping",
        complexity=1,
        system_config_name="C2",
    )
    system_config = build_system_config("C2", executor_model="qwen3.5-9b", max_steps=3)

    result = asyncio.run(
        runner.run_task(
            user_query="build a footwear collection",
            system_prompt=shopping_module.get_system_prompt(1),
            state=state,
            system_config=system_config,
            sample_id="1",
            shared_oversight_cache_root=tmp_path / "cache",
        )
    )

    follow_up_messages = captured_calls[1]["messages"]
    assert follow_up_messages[-1]["role"] == "user"
    assert (
        "Re-check task requirement: product footwear."
        in follow_up_messages[-1]["content"]
    )
    assert not any(
        isinstance(message, dict)
        and "[OVERSEER NOTICE]" in str(message.get("content", ""))
        for message in result.messages
    )
    assert state.pending_executor_notice is None


def test_post_tool_error_occurrence_sets_pending_notice_without_rewriting_history(
    monkeypatch, tmp_path
):
    run_database_dir = tmp_path / "shopping_db"
    shutil.copytree(SHOPPING_FIXTURE_ROOT, run_database_dir)

    captured_calls: list[dict[str, object]] = []
    _install_c2_lite_mocks(
        monkeypatch,
        executor_responses=[
            FakeResponse(
                content="",
                tool_calls=[
                    FakeToolCall(
                        "call_1",
                        "search_products",
                        '{"query":"laptop"}',
                    )
                ],
                prompt_tokens=10,
                completion_tokens=4,
                finish_reason="tool_calls",
            ),
            FakeResponse(
                content="Phase one complete.",
                prompt_tokens=11,
                completion_tokens=5,
            ),
            FakeResponse(
                content="Final cart answer.",
                prompt_tokens=12,
                completion_tokens=6,
            ),
        ],
        overseer_responses=[
            _fake_json_response(
                {
                    "action": "provide_guidance",
                    "decision_summary": "Tool failed.",
                    "block_current_tool": False,
                    "guidance_lines": [
                        "Retry with a safer search or inspect cart state."
                    ],
                    "corrected_observation": None,
                    "violated_contract_ids": [],
                    "unmet_checklist_keys": [],
                }
            ),
            _fake_json_response(
                {
                    "action": "approve",
                    "pass": True,
                    "decision_summary": "Final answer approved.",
                    "blockers": [],
                    "next_step_notice_lines": [],
                    "violated_contract_ids": [],
                    "unmet_checklist_keys": [],
                }
            ),
        ],
        captured_executor_calls=captured_calls,
    )
    monkeypatch.setattr(
        shopping_module.ShoppingAgentRunner,
        "_exec_tool",
        lambda self, name, arguments: "FAILED: invalid query",
    )

    runner = shopping_module.ShoppingAgentRunner(
        model="qwen3.5-9b",
        sample_id="1",
        database_base_path=str(run_database_dir),
        tool_schema_path=str(SHOPPING_SCHEMA_PATH),
    )
    state = ConversationState(
        task_id="1",
        domain="shopping",
        complexity=1,
        system_config_name="C2",
    )
    system_config = build_system_config("C2", executor_model="qwen3.5-9b", max_steps=3)

    result = asyncio.run(
        runner.run_task(
            user_query="buy a laptop",
            system_prompt=shopping_module.get_system_prompt(1),
            state=state,
            system_config=system_config,
            sample_id="1",
            shared_oversight_cache_root=tmp_path / "cache",
        )
    )

    assert any(
        isinstance(message, dict)
        and message.get("role") == "tool"
        and message.get("content") == "FAILED: invalid query"
        for message in result.messages
    )
    follow_up_messages = captured_calls[1]["messages"]
    assert follow_up_messages[-1]["role"] == "user"
    assert "Retry with a safer search" in follow_up_messages[-1]["content"]


def test_repeated_blocked_mutation_terminates_cleanly(monkeypatch, tmp_path):
    run_database_dir = tmp_path / "shopping_db"
    shutil.copytree(SHOPPING_FIXTURE_ROOT, run_database_dir)

    captured_calls: list[dict[str, object]] = []
    _install_c2_lite_mocks(
        monkeypatch,
        executor_responses=[
            FakeResponse(
                content="",
                tool_calls=[
                    FakeToolCall(
                        "call_1",
                        "add_product_to_cart",
                        '{"product_id":"1"}',
                    )
                ],
                prompt_tokens=10,
                completion_tokens=4,
                finish_reason="tool_calls",
            ),
            FakeResponse(
                content="",
                tool_calls=[
                    FakeToolCall(
                        "call_2",
                        "add_product_to_cart",
                        '{"product_id":"1"}',
                    )
                ],
                prompt_tokens=11,
                completion_tokens=4,
                finish_reason="tool_calls",
            ),
            FakeResponse(
                content="",
                tool_calls=[
                    FakeToolCall(
                        "call_3",
                        "add_product_to_cart",
                        '{"product_id":"1"}',
                    )
                ],
                prompt_tokens=12,
                completion_tokens=4,
                finish_reason="tool_calls",
            ),
        ],
        overseer_responses=[
            _fake_json_response(
                {
                    "action": "provide_guidance",
                    "decision_summary": "Blocked cart mutation.",
                    "block_current_tool": True,
                    "guidance_lines": ["Verify a different candidate before mutating."],
                    "corrected_observation": None,
                    "violated_contract_ids": ["rule-1"],
                    "unmet_checklist_keys": [],
                }
            ),
            _fake_json_response(
                {
                    "action": "provide_guidance",
                    "decision_summary": "Blocked cart mutation again.",
                    "block_current_tool": True,
                    "guidance_lines": ["Verify a different candidate before mutating."],
                    "corrected_observation": None,
                    "violated_contract_ids": ["rule-1"],
                    "unmet_checklist_keys": [],
                }
            ),
            _fake_json_response(
                {
                    "action": "provide_guidance",
                    "decision_summary": "Blocked cart mutation again.",
                    "block_current_tool": True,
                    "guidance_lines": ["Verify a different candidate before mutating."],
                    "corrected_observation": None,
                    "violated_contract_ids": ["rule-1"],
                    "unmet_checklist_keys": [],
                }
            ),
        ],
        captured_executor_calls=captured_calls,
    )

    runner = shopping_module.ShoppingAgentRunner(
        model="qwen3.5-9b",
        sample_id="1",
        database_base_path=str(run_database_dir),
        tool_schema_path=str(SHOPPING_SCHEMA_PATH),
    )
    state = ConversationState(
        task_id="1",
        domain="shopping",
        complexity=1,
        system_config_name="C2",
    )
    logger = StructuredLogger(tmp_path / "shopping_logs")
    system_config = build_system_config("C2", executor_model="qwen3.5-9b", max_steps=5)

    result = asyncio.run(
        runner.run_task(
            user_query="buy a laptop",
            system_prompt=shopping_module.get_system_prompt(1),
            state=state,
            system_config=system_config,
            logger=logger,
            sample_id="1",
            shared_oversight_cache_root=tmp_path / "cache",
        )
    )

    assert result.output == ""
    assert state.final_stop_reason == shopping_module.REPEATED_BLOCK_STOP_REASON
    assert state.final_output_present is False
    assert state.max_steps_hit is False
    assert state.blocked_mutation_count == 3
    assert len(captured_calls) == 3
    records = _load_jsonl(tmp_path / "shopping_logs" / "agent_events.jsonl")
    assert any(
        record["event_type"] == "executor_turn"
        and record["stop_reason"] == shopping_module.REPEATED_BLOCK_STOP_REASON
        for record in records
    )


def test_midpoint_coverage_notice_is_consumed_on_first_cart_check_turn(
    monkeypatch, tmp_path
):
    run_database_dir = tmp_path / "shopping_db"
    shutil.copytree(SHOPPING_FIXTURE_ROOT, run_database_dir)

    captured_calls: list[dict[str, object]] = []
    _install_c2_lite_mocks(
        monkeypatch,
        executor_responses=[
            FakeResponse(
                content="Phase one complete.",
                prompt_tokens=10,
                completion_tokens=4,
            ),
            FakeResponse(
                content="",
                tool_calls=[FakeToolCall("call_1", "get_cart_info", "{}")],
                prompt_tokens=11,
                completion_tokens=5,
                finish_reason="tool_calls",
            ),
            FakeResponse(
                content="Final cart answer.",
                prompt_tokens=12,
                completion_tokens=6,
            ),
        ],
        compiler_responses=[
            _fake_json_response(_shopping_contract_payload()),
            _fake_json_response(
                _shopping_checklist_payload(
                    coverage_targets=[
                        {
                            "key": "product:laptop",
                            "category": "product",
                            "aliases": ["laptop"],
                            "tool_roles": ["search"],
                        }
                    ],
                    items=[
                        {
                            "key": "product:laptop",
                            "category": "required_product",
                            "description": "laptop",
                            "value": "laptop",
                            "required": True,
                            "explicit": True,
                            "coverage_relevant": True,
                            "final_verify_only": False,
                            "aliases": ["laptop"],
                            "source_text": "laptop",
                        }
                    ],
                    final_verification_only_keys=[],
                )
            ),
        ],
        overseer_responses=[
            _fake_json_response(
                {
                    "action": "provide_guidance",
                    "decision_summary": "Missing product coverage.",
                    "block_current_tool": False,
                    "guidance_lines": ["Inspect laptop options before cart check."],
                    "corrected_observation": None,
                    "violated_contract_ids": [],
                    "unmet_checklist_keys": ["product:laptop"],
                }
            ),
            _fake_json_response(
                {
                    "action": "approve",
                    "pass": True,
                    "decision_summary": "Final answer approved.",
                    "blockers": [],
                    "next_step_notice_lines": [],
                    "violated_contract_ids": [],
                    "unmet_checklist_keys": [],
                }
            ),
        ],
        captured_executor_calls=captured_calls,
    )
    monkeypatch.setattr(
        shopping_module.ShoppingAgentRunner,
        "_exec_tool",
        lambda self, name, arguments: '{"items":[]}',
    )

    runner = shopping_module.ShoppingAgentRunner(
        model="qwen3.5-9b",
        sample_id="1",
        database_base_path=str(run_database_dir),
        tool_schema_path=str(SHOPPING_SCHEMA_PATH),
    )
    state = ConversationState(
        task_id="1",
        domain="shopping",
        complexity=1,
        system_config_name="C2",
    )
    system_config = build_system_config("C2", executor_model="qwen3.5-9b", max_steps=3)

    asyncio.run(
        runner.run_task(
            user_query="buy a laptop",
            system_prompt=shopping_module.get_system_prompt(1),
            state=state,
            system_config=system_config,
            sample_id="1",
            shared_oversight_cache_root=tmp_path / "cache",
        )
    )

    first_cart_check_messages = captured_calls[1]["messages"]
    second_cart_check_messages = captured_calls[2]["messages"]
    assert first_cart_check_messages[-1]["role"] == "user"
    assert "Inspect laptop." in first_cart_check_messages[-1]["content"]
    assert not any(
        isinstance(message, dict) and "[OVERSEER NOTICE]" in message.get("content", "")
        for message in second_cart_check_messages[-2:]
    )


def test_final_checkpoint_blocks_commit_until_approved(monkeypatch, tmp_path):
    run_database_dir = tmp_path / "shopping_db"
    shutil.copytree(SHOPPING_FIXTURE_ROOT, run_database_dir)

    captured_calls: list[dict[str, object]] = []
    _install_c2_lite_mocks(
        monkeypatch,
        executor_responses=[
            FakeResponse(
                content="Phase one complete.", prompt_tokens=10, completion_tokens=4
            ),
            FakeResponse(
                content="Draft final answer.", prompt_tokens=11, completion_tokens=5
            ),
            FakeResponse(
                content="Approved final answer.", prompt_tokens=12, completion_tokens=6
            ),
        ],
        overseer_responses=[
            _fake_json_response(
                {
                    "action": "run_verification",
                    "pass": False,
                    "decision_summary": "Need a fresh verification step.",
                    "blockers": [
                        {"type": "missing_item", "detail": "Need cart verification"}
                    ],
                    "next_step_notice_lines": ["Call get_cart_info before finalizing."],
                    "violated_contract_ids": [],
                    "unmet_checklist_keys": ["final:fresh-cart"],
                }
            ),
            _fake_json_response(
                {
                    "action": "approve",
                    "pass": True,
                    "decision_summary": "Approved.",
                    "blockers": [],
                    "next_step_notice_lines": [],
                    "violated_contract_ids": [],
                    "unmet_checklist_keys": [],
                }
            ),
        ],
        captured_executor_calls=captured_calls,
    )

    runner = shopping_module.ShoppingAgentRunner(
        model="qwen3.5-9b",
        sample_id="1",
        database_base_path=str(run_database_dir),
        tool_schema_path=str(SHOPPING_SCHEMA_PATH),
    )
    state = ConversationState(
        task_id="1",
        domain="shopping",
        complexity=1,
        system_config_name="C2",
    )
    system_config = build_system_config("C2", executor_model="qwen3.5-9b", max_steps=3)

    result = asyncio.run(
        runner.run_task(
            user_query="buy a laptop",
            system_prompt=shopping_module.get_system_prompt(1),
            state=state,
            system_config=system_config,
            sample_id="1",
            shared_oversight_cache_root=tmp_path / "cache",
        )
    )

    assert result.output == "Approved final answer."
    assert not any(
        isinstance(message, dict)
        and message.get("role") == "assistant"
        and message.get("content") == "Draft final answer."
        for message in result.messages
    )
    assert any(
        isinstance(message, dict)
        and message.get("role") == "assistant"
        and message.get("content") == "Approved final answer."
        for message in result.messages
    )
    assert captured_calls[2]["messages"][-1]["role"] == "user"
    assert (
        "Call get_cart_info before finalizing."
        in captured_calls[2]["messages"][-1]["content"]
    )


def test_final_checkpoint_stale_cart_precheck_forces_get_cart_info_first(
    monkeypatch, tmp_path
):
    run_database_dir = tmp_path / "shopping_db"
    shutil.copytree(SHOPPING_FIXTURE_ROOT, run_database_dir)

    captured_calls: list[dict[str, object]] = []
    _install_c2_lite_mocks(
        monkeypatch,
        executor_responses=[
            FakeResponse(
                content="Phase one complete.", prompt_tokens=10, completion_tokens=4
            ),
            FakeResponse(
                content="",
                tool_calls=[
                    FakeToolCall("call_1", "add_product_to_cart", '{"product_id":"1"}')
                ],
                prompt_tokens=11,
                completion_tokens=5,
                finish_reason="tool_calls",
            ),
            FakeResponse(
                content="Draft final answer.", prompt_tokens=12, completion_tokens=6
            ),
            FakeResponse(
                content="",
                tool_calls=[FakeToolCall("call_2", "get_cart_info", "{}")],
                prompt_tokens=13,
                completion_tokens=5,
                finish_reason="tool_calls",
            ),
            FakeResponse(
                content="Approved final answer.", prompt_tokens=14, completion_tokens=6
            ),
        ],
        overseer_responses=[
            _fake_json_response(
                {
                    "action": "approve",
                    "decision_summary": "Mutation approved.",
                    "block_current_tool": False,
                    "guidance_lines": [],
                    "corrected_observation": None,
                    "violated_contract_ids": [],
                    "unmet_checklist_keys": [],
                }
            ),
            _fake_json_response(
                {
                    "action": "approve",
                    "pass": True,
                    "decision_summary": "Approved.",
                    "blockers": [],
                    "next_step_notice_lines": [],
                    "violated_contract_ids": [],
                    "unmet_checklist_keys": [],
                }
            ),
        ],
        captured_executor_calls=captured_calls,
    )
    monkeypatch.setattr(
        shopping_module.ShoppingAgentRunner,
        "_exec_tool",
        lambda self, name, arguments: (
            '{"success": true}' if name == "add_product_to_cart" else '{"items":[]}'
        ),
    )

    runner = shopping_module.ShoppingAgentRunner(
        model="qwen3.5-9b",
        sample_id="1",
        database_base_path=str(run_database_dir),
        tool_schema_path=str(SHOPPING_SCHEMA_PATH),
    )
    state = ConversationState(
        task_id="1",
        domain="shopping",
        complexity=1,
        system_config_name="C2",
    )
    system_config = build_system_config("C2", executor_model="qwen3.5-9b", max_steps=5)

    result = asyncio.run(
        runner.run_task(
            user_query="buy a laptop",
            system_prompt=shopping_module.get_system_prompt(1),
            state=state,
            system_config=system_config,
            sample_id="1",
            shared_oversight_cache_root=tmp_path / "cache",
        )
    )

    assert result.output == "Approved final answer."
    assert captured_calls[3]["messages"][-1]["role"] == "user"
    assert (
        "Call get_cart_info before finalizing. The cart state is the source of truth."
        in captured_calls[3]["messages"][-1]["content"]
    )


def test_final_checkpoint_retry_cap_exhaustion_returns_empty_final_output(
    monkeypatch, tmp_path
):
    run_database_dir = tmp_path / "shopping_db"
    shutil.copytree(SHOPPING_FIXTURE_ROOT, run_database_dir)

    _install_c2_lite_mocks(
        monkeypatch,
        executor_responses=[
            FakeResponse(
                content="Phase one complete.", prompt_tokens=10, completion_tokens=4
            ),
            FakeResponse(content="Draft 1", prompt_tokens=11, completion_tokens=5),
            FakeResponse(content="Draft 2", prompt_tokens=12, completion_tokens=5),
            FakeResponse(content="Draft 3", prompt_tokens=13, completion_tokens=5),
        ],
        overseer_responses=[
            _fake_json_response(
                {
                    "action": "run_verification",
                    "pass": False,
                    "decision_summary": "Need more verification.",
                    "blockers": [
                        {"type": "missing_item", "detail": "Need cart verification"}
                    ],
                    "next_step_notice_lines": ["Call get_cart_info before finalizing."],
                    "violated_contract_ids": [],
                    "unmet_checklist_keys": ["final:fresh-cart"],
                }
            ),
            _fake_json_response(
                {
                    "action": "run_verification",
                    "pass": False,
                    "decision_summary": "Still not verified.",
                    "blockers": [
                        {"type": "missing_item", "detail": "Need cart verification"}
                    ],
                    "next_step_notice_lines": ["Call get_cart_info before finalizing."],
                    "violated_contract_ids": [],
                    "unmet_checklist_keys": ["final:fresh-cart"],
                }
            ),
            _fake_json_response(
                {
                    "action": "run_verification",
                    "pass": False,
                    "decision_summary": "Retry cap reached.",
                    "blockers": [
                        {"type": "missing_item", "detail": "Need cart verification"}
                    ],
                    "next_step_notice_lines": ["Call get_cart_info before finalizing."],
                    "violated_contract_ids": [],
                    "unmet_checklist_keys": ["final:fresh-cart"],
                }
            ),
        ],
    )

    runner = shopping_module.ShoppingAgentRunner(
        model="qwen3.5-9b",
        sample_id="1",
        database_base_path=str(run_database_dir),
        tool_schema_path=str(SHOPPING_SCHEMA_PATH),
    )
    state = ConversationState(
        task_id="1",
        domain="shopping",
        complexity=1,
        system_config_name="C2",
    )
    system_config = build_system_config("C2", executor_model="qwen3.5-9b", max_steps=4)

    result = asyncio.run(
        runner.run_task(
            user_query="buy a laptop",
            system_prompt=shopping_module.get_system_prompt(1),
            state=state,
            system_config=system_config,
            sample_id="1",
            shared_oversight_cache_root=tmp_path / "cache",
        )
    )

    assert result.output == ""
    assert state.final_verification_result == "retry_cap_exhausted"


def test_shopping_run_agent_inference_still_writes_logs_under_output_dir(
    monkeypatch, tmp_path
):
    run_database_dir = tmp_path / "shopping_db"
    shutil.copytree(SHOPPING_FIXTURE_ROOT, run_database_dir)
    output_dir = tmp_path / "shopping_logs"
    test_data_path = tmp_path / "shopping_samples.json"
    test_data_path.write_text(
        json.dumps([{"id": 1, "query": "test shopping query", "level": 1}]),
        encoding="utf-8",
    )

    _install_c2_lite_mocks(
        monkeypatch,
        executor_responses=[
            FakeResponse(
                content="Phase one complete.", prompt_tokens=10, completion_tokens=4
            ),
            FakeResponse(
                content="Cart verified.", prompt_tokens=11, completion_tokens=5
            ),
        ],
        overseer_responses=[
            _fake_json_response(
                {
                    "action": "approve",
                    "pass": True,
                    "decision_summary": "Approved.",
                    "blockers": [],
                    "next_step_notice_lines": [],
                    "violated_contract_ids": [],
                    "unmet_checklist_keys": [],
                }
            )
        ],
    )

    results = shopping_module.run_agent_inference(
        model="qwen3.5-9b",
        test_data_path=test_data_path,
        database_dir=run_database_dir,
        tool_schema_path=SHOPPING_SCHEMA_PATH,
        system_prompt=shopping_module.get_system_prompt(1),
        output_dir=output_dir,
        workers=1,
        max_llm_calls=2,
        system="C2",
        shared_oversight_cache_root=tmp_path / "cache",
    )

    assert results["success"] == 1
    assert (output_dir / "agent_events.jsonl").exists()
    assert (output_dir / "task_results.jsonl").exists()
    assert not (run_database_dir / "agent_events.jsonl").exists()


def test_shopping_run_agent_inference_logs_under_output_dir(monkeypatch, tmp_path):
    run_database_dir = tmp_path / "shopping_db"
    shutil.copytree(SHOPPING_FIXTURE_ROOT, run_database_dir)
    output_dir = tmp_path / "shopping_logs"
    test_data_path = tmp_path / "shopping_samples.json"
    test_data_path.write_text(
        json.dumps([{"id": 1, "query": "test shopping query", "level": 1}]),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        shopping_module,
        "call_chat_completion",
        _fake_completion_factory(
            [
                FakeResponse(
                    content="Phase one complete.",
                    prompt_tokens=10,
                    completion_tokens=4,
                ),
                FakeResponse(
                    content="Cart verified.",
                    prompt_tokens=11,
                    completion_tokens=5,
                ),
            ]
        ),
    )

    results = shopping_module.run_agent_inference(
        model="qwen-plus",
        test_data_path=test_data_path,
        database_dir=run_database_dir,
        tool_schema_path=SHOPPING_SCHEMA_PATH,
        system_prompt=shopping_module.get_system_prompt(1),
        output_dir=output_dir,
        workers=1,
        max_llm_calls=2,
        system="A",
    )

    assert results["success"] == 1
    assert (output_dir / "agent_events.jsonl").exists()
    assert (output_dir / "task_results.jsonl").exists()
    assert not (run_database_dir / "agent_events.jsonl").exists()

    records = _load_jsonl(output_dir / "agent_events.jsonl")
    assert records[0]["request_messages"][0]["role"] == "system"
    assert "raw_response" in records[0]
    assert "parsed_tool_calls" in records[0]
    assert "tool_results" in records[0]


def test_shopping_run_agent_inference_scopes_trace_per_case(monkeypatch, tmp_path):
    run_database_dir = tmp_path / "shopping_db"
    shutil.copytree(SHOPPING_FIXTURE_ROOT, run_database_dir)
    test_data_path = tmp_path / "shopping_samples.json"
    test_data_path.write_text(
        json.dumps(
            [
                {"id": 1, "query": "test shopping query 1", "level": 1},
                {"id": 10, "query": "test shopping query 10", "level": 1},
            ]
        ),
        encoding="utf-8",
    )

    captured_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        shopping_module,
        "build_langfuse_trace_id",
        lambda *parts: "|".join(str(part) for part in parts),
    )
    monkeypatch.setattr(
        shopping_module,
        "call_chat_completion",
        _fake_completion_factory(
            [
                FakeResponse(
                    content="Phase one complete.",
                    prompt_tokens=10,
                    completion_tokens=4,
                ),
                FakeResponse(
                    content="Cart verified.",
                    prompt_tokens=11,
                    completion_tokens=5,
                ),
                FakeResponse(
                    content="Phase one complete.",
                    prompt_tokens=12,
                    completion_tokens=4,
                ),
                FakeResponse(
                    content="Cart verified.",
                    prompt_tokens=13,
                    completion_tokens=5,
                ),
            ],
            captured_calls,
        ),
    )

    results = shopping_module.run_agent_inference(
        model="qwen-plus",
        test_data_path=test_data_path,
        database_dir=run_database_dir,
        tool_schema_path=SHOPPING_SCHEMA_PATH,
        system_prompt=shopping_module.get_system_prompt(1),
        output_dir=tmp_path / "shopping_logs",
        workers=1,
        max_llm_calls=2,
        system="A",
        session_id="bench-session",
    )

    assert results["success"] == 2
    assert [call["trace_id"] for call in captured_calls] == [
        "bench-session|shopping|qwen-plus|0|1",
        "bench-session|shopping|qwen-plus|0|1",
        "bench-session|shopping|qwen-plus|0|10",
        "bench-session|shopping|qwen-plus|0|10",
    ]
    assert {call["session_id"] for call in captured_calls} == {"bench-session"}


def test_shopping_run_agent_inference_logs_structured_task_error(monkeypatch, tmp_path):
    test_data_path = tmp_path / "shopping_samples.json"
    test_data_path.write_text(
        json.dumps([{"id": 1, "query": "test shopping query", "level": 1}]),
        encoding="utf-8",
    )

    class FakeShoppingRunner:
        def __init__(self, **kwargs) -> None:
            pass

        async def run_task(self, **kwargs):
            raise RuntimeError("shopping boom")

    monkeypatch.setattr(shopping_module, "ShoppingAgentRunner", FakeShoppingRunner)

    output_dir = tmp_path / "shopping_logs"
    results = shopping_module.run_agent_inference(
        model="qwen-plus",
        test_data_path=test_data_path,
        database_dir=tmp_path / "shopping_db",
        tool_schema_path=tmp_path / "shopping_tool_schema.json",
        system_prompt=shopping_module.get_system_prompt(1),
        output_dir=output_dir,
        workers=1,
        max_llm_calls=2,
        system="A",
    )

    assert results["failed"] == 1
    records = _load_jsonl(output_dir / "agent_events.jsonl")
    task_results = _load_jsonl(output_dir / "task_results.jsonl")
    assert records[0]["event_type"] == "task_error"
    assert records[0]["failure_subtype"] == "none"
    assert records[0]["error"]["type"] == "RuntimeError"
    assert records[0]["error"]["message"] == "shopping boom"
    assert "shopping boom" in records[0]["error"]["traceback"]
    assert records[0]["observation_valid"] is True
    assert task_results[0]["success"] is False
    assert task_results[0]["failure_subtype"] == "none"
    assert task_results[0]["observation_valid"] is True
    assert task_results[0]["error"]["message"] == "shopping boom"


def test_shopping_run_agent_inference_records_max_tool_calls_failure_subtype(
    monkeypatch, tmp_path
):
    test_data_path = tmp_path / "shopping_samples.json"
    test_data_path.write_text(
        json.dumps([{"id": 1, "query": "test shopping query", "level": 1}]),
        encoding="utf-8",
    )

    class FakeShoppingRunner:
        def __init__(self, **kwargs) -> None:
            pass

        async def run_task(self, **kwargs):
            state = kwargs["state"]
            state.begin()
            state.record_final_outcome(
                stop_reason="max_steps_exhausted",
                output=None,
                max_steps_hit=True,
            )
            state.finish()
            return shopping_module.TaskResult(
                task_id=state.task_id,
                run_id=kwargs["run_id"],
                output="",
                messages=[],
                state=state,
            )

    monkeypatch.setattr(shopping_module, "ShoppingAgentRunner", FakeShoppingRunner)

    output_dir = tmp_path / "shopping_logs"
    results = shopping_module.run_agent_inference(
        model="qwen-plus",
        test_data_path=test_data_path,
        database_dir=tmp_path / "shopping_db",
        tool_schema_path=tmp_path / "shopping_tool_schema.json",
        system_prompt=shopping_module.get_system_prompt(1),
        output_dir=output_dir,
        workers=1,
        max_llm_calls=2,
        system="A",
    )

    assert results["success"] == 1
    task_results = _load_jsonl(output_dir / "task_results.jsonl")
    assert task_results[0]["success"] is True
    assert task_results[0]["failure_subtype"] == "max_tool_calls"
    assert task_results[0]["observation_valid"] is True
    assert task_results[0]["final_stop_reason"] == "max_steps_exhausted"


def test_shopping_run_agent_inference_retries_infra_and_restores_case_state(
    monkeypatch, tmp_path
):
    test_data_path = tmp_path / "shopping_samples.json"
    test_data_path.write_text(
        json.dumps([{"id": 1, "query": "test shopping query", "level": 1}]),
        encoding="utf-8",
    )

    run_database_dir = tmp_path / "shopping_db"
    case_dir = run_database_dir / "case_1"
    case_dir.mkdir(parents=True)
    (case_dir / "cart.json").write_text("{}", encoding="utf-8")

    marker_presence: list[bool] = []

    class FakeShoppingRunner:
        def __init__(self, **kwargs) -> None:
            self.database_base_path = Path(kwargs["database_base_path"])

        async def run_task(self, **kwargs):
            sample_case_dir = self.database_base_path / "case_1"
            marker = sample_case_dir / "attempt_marker.txt"
            marker_presence.append(marker.exists())
            if len(marker_presence) == 1:
                marker.write_text("mutated", encoding="utf-8")
                raise OSError("socket reset by peer")

            state = kwargs["state"]
            state.begin()
            state.record_final_outcome(
                stop_reason="no_tool_calls",
                output="Recovered cart",
                max_steps_hit=False,
            )
            state.finish()
            return shopping_module.TaskResult(
                task_id=state.task_id,
                run_id=kwargs["run_id"],
                output="Recovered cart",
                messages=[],
                state=state,
            )

    monkeypatch.setattr(shopping_module, "ShoppingAgentRunner", FakeShoppingRunner)

    output_dir = tmp_path / "shopping_logs"
    results = shopping_module.run_agent_inference(
        model="qwen-plus",
        test_data_path=test_data_path,
        database_dir=run_database_dir,
        tool_schema_path=tmp_path / "shopping_tool_schema.json",
        system_prompt=shopping_module.get_system_prompt(1),
        output_dir=output_dir,
        workers=1,
        max_llm_calls=2,
        infra_retry_limit=1,
        system="A",
    )

    assert marker_presence == [False, False]
    assert results["total"] == 1
    assert results["success"] == 1
    assert results["failed"] == 0
    assert results["invalid"] == 0
    records = _load_jsonl(output_dir / "agent_events.jsonl")
    task_results = _load_jsonl(output_dir / "task_results.jsonl")
    assert [record["event_type"] for record in records] == ["task_invalid_attempt"]
    assert records[0]["failure_subtype"] == "infra_transient"
    assert records[0]["observation_valid"] is False
    assert len(task_results) == 1
    assert task_results[0]["success"] is True
    assert task_results[0]["observation_valid"] is True
    assert not (case_dir / "attempt_marker.txt").exists()


def test_shopping_run_agent_inference_prints_per_run_progress(
    monkeypatch, tmp_path, capsys
):
    test_data_path = tmp_path / "shopping_samples.json"
    test_data_path.write_text(
        json.dumps([{"id": 1, "query": "test shopping query", "level": 1}]),
        encoding="utf-8",
    )

    class FakeShoppingRunner:
        def __init__(self, **kwargs) -> None:
            pass

        async def run_task(self, **kwargs):
            state = kwargs["state"]
            run_id = kwargs["run_id"]
            state.begin()
            state.record_final_outcome(
                stop_reason="no_tool_calls",
                output=f"run-{run_id}",
                max_steps_hit=False,
            )
            state.finish()
            return shopping_module.TaskResult(
                task_id=state.task_id,
                run_id=run_id,
                output=f"run-{run_id}",
                messages=[],
                state=state,
            )

    monkeypatch.setattr(shopping_module, "ShoppingAgentRunner", FakeShoppingRunner)

    database_dir_by_run = {
        0: tmp_path / "shopping_db_run_0",
        1: tmp_path / "shopping_db_run_1",
    }
    for path in database_dir_by_run.values():
        path.mkdir(parents=True)

    results = shopping_module.run_agent_inference(
        model="qwen-plus",
        test_data_path=test_data_path,
        database_dir=database_dir_by_run[0],
        tool_schema_path=SHOPPING_SCHEMA_PATH,
        system_prompt=shopping_module.get_system_prompt(1),
        output_dir=tmp_path / "shopping_logs",
        workers=1,
        max_llm_calls=2,
        runs=2,
        system="A",
        database_dir_by_run=database_dir_by_run,
    )

    assert results["success"] == 2
    output = capsys.readouterr().out
    assert "Execution mode: concurrent across runs" in output
    assert "Shopping progress | run 1/2" in output
    assert "Shopping progress | run 2/2" in output
    assert "overall: 1/2 done, 1 left" in output
    assert "overall: 2/2 done, 0 left" in output


def test_shopping_run_agent_inference_flushes_langfuse(monkeypatch, tmp_path):
    cleanup_calls: list[str] = []

    async def fake_run_agent_inference_async(**kwargs):
        return {
            "total": 0,
            "success": 0,
            "failed": 0,
            "elapsed_time": 0,
            "results": [],
        }

    def fake_flush() -> None:
        cleanup_calls.append("shopping")

    monkeypatch.setattr(
        shopping_module, "run_agent_inference_async", fake_run_agent_inference_async
    )
    monkeypatch.setattr(shopping_module, "flush_langfuse", fake_flush)

    results = shopping_module.run_agent_inference(
        model="qwen-plus",
        test_data_path=tmp_path / "shopping_samples.json",
        database_dir=tmp_path / "shopping_db",
        tool_schema_path=SHOPPING_SCHEMA_PATH,
        system_prompt="prompt",
        output_dir=tmp_path / "shopping_output",
        workers=1,
        max_llm_calls=1,
        runs=1,
        system="A",
    )

    assert results["total"] == 0
    assert cleanup_calls == ["shopping"]


def test_travel_run_agent_inference_isolates_multi_run_outputs(monkeypatch, tmp_path):
    test_data_path = tmp_path / "travel_samples.json"
    test_data_path.write_text(
        json.dumps(
            [
                {
                    "id": 0,
                    "query": "test travel query",
                    "meta_info": {"days": 2},
                }
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        travel_module,
        "call_chat_completion",
        _fake_completion_factory(
            [
                FakeResponse(
                    content="<think>reasoning</think><plan>Run 0 plan</plan>",
                    prompt_tokens=8,
                    completion_tokens=3,
                ),
                FakeResponse(
                    content="<think>reasoning</think><plan>Run 1 plan</plan>",
                    prompt_tokens=8,
                    completion_tokens=3,
                ),
            ]
        ),
    )

    output_root = tmp_path / "travel_runs"
    results = travel_module.run_agent_inference(
        model="qwen-plus",
        language="en",
        test_data_path=test_data_path,
        database_dir=TRAVEL_DATABASE_DIR,
        tool_schema_path=TRAVEL_SCHEMA_PATH,
        output_dir=output_root,
        workers=1,
        max_llm_calls=1,
        runs=2,
        system="A",
    )

    assert results["success"] == 2
    assert results["runs"] == 2

    run_zero_records = _load_jsonl(output_root / "run_0" / "task_results.jsonl")
    run_one_records = _load_jsonl(output_root / "run_1" / "task_results.jsonl")
    assert run_zero_records[0]["run_id"] == 0
    assert run_one_records[0]["run_id"] == 1
    assert (output_root / "run_0" / "reports" / "id_0.txt").exists()
    assert (output_root / "run_1" / "reports" / "id_0.txt").exists()


def test_travel_run_agent_inference_prints_per_run_progress(
    monkeypatch, tmp_path, capsys
):
    test_data_path = tmp_path / "travel_samples.json"
    test_data_path.write_text(
        json.dumps(
            [
                {
                    "id": 0,
                    "query": "test travel query",
                    "meta_info": {"days": 2},
                }
            ]
        ),
        encoding="utf-8",
    )

    class FakeTravelRunner:
        def __init__(self, **kwargs) -> None:
            pass

        async def run_task(self, **kwargs):
            state = kwargs["state"]
            run_id = kwargs["run_id"]
            state.begin()
            state.record_final_outcome(
                stop_reason="no_tool_calls",
                output=f"plan-{run_id}",
                max_steps_hit=False,
            )
            state.finish()
            return travel_module.TaskResult(
                task_id=state.task_id,
                run_id=run_id,
                output=f"plan-{run_id}",
                messages=[],
                state=state,
            )

    monkeypatch.setattr(travel_module, "TravelAgentRunner", FakeTravelRunner)

    results = travel_module.run_agent_inference(
        model="qwen-plus",
        language="en",
        test_data_path=test_data_path,
        database_dir=TRAVEL_DATABASE_DIR,
        tool_schema_path=TRAVEL_SCHEMA_PATH,
        output_dir=tmp_path / "travel_output",
        workers=1,
        max_llm_calls=1,
        runs=2,
        system="A",
    )

    assert results["success"] == 2
    output = capsys.readouterr().out
    assert "Execution mode: concurrent across runs" in output
    assert "Travel progress | run 1/2" in output
    assert "Travel progress | run 2/2" in output
    assert "overall: 1/2 done, 1 left" in output
    assert "overall: 2/2 done, 0 left" in output


def test_travel_run_agent_inference_flushes_langfuse(monkeypatch, tmp_path):
    cleanup_calls: list[str] = []

    async def fake_run_agent_inference_async(**kwargs):
        return {
            "total": 0,
            "success": 0,
            "failed": 0,
            "elapsed_time": 0,
            "results": [],
        }

    def fake_flush() -> None:
        cleanup_calls.append("travel")

    monkeypatch.setattr(
        travel_module, "run_agent_inference_async", fake_run_agent_inference_async
    )
    monkeypatch.setattr(travel_module, "flush_langfuse", fake_flush)

    results = travel_module.run_agent_inference(
        model="qwen-plus",
        language="en",
        test_data_path=tmp_path / "travel_samples.json",
        database_dir=TRAVEL_DATABASE_DIR,
        tool_schema_path=TRAVEL_SCHEMA_PATH,
        output_dir=tmp_path / "travel_output",
        workers=1,
        max_llm_calls=1,
        runs=1,
        system="A",
    )

    assert results["total"] == 0
    assert cleanup_calls == ["travel"]


def test_travel_run_agent_inference_logs_structured_task_error(monkeypatch, tmp_path):
    test_data_path = tmp_path / "travel_samples.json"
    test_data_path.write_text(
        json.dumps(
            [
                {
                    "id": 0,
                    "query": "test travel query",
                    "meta_info": {"days": 2},
                }
            ]
        ),
        encoding="utf-8",
    )

    class FakeTravelRunner:
        def __init__(self, **kwargs) -> None:
            pass

        async def run_task(self, **kwargs):
            raise RuntimeError("travel boom")

    monkeypatch.setattr(travel_module, "TravelAgentRunner", FakeTravelRunner)

    output_dir = tmp_path / "travel_output"
    results = travel_module.run_agent_inference(
        model="qwen-plus",
        language="en",
        test_data_path=test_data_path,
        database_dir=TRAVEL_DATABASE_DIR,
        tool_schema_path=TRAVEL_SCHEMA_PATH,
        output_dir=output_dir,
        workers=1,
        max_llm_calls=1,
        system="A",
    )

    assert results["failed"] == 1
    records = _load_jsonl(output_dir / "agent_events.jsonl")
    task_results = _load_jsonl(output_dir / "task_results.jsonl")
    assert records[0]["event_type"] == "task_error"
    assert records[0]["failure_subtype"] == "none"
    assert records[0]["error"]["type"] == "RuntimeError"
    assert records[0]["error"]["message"] == "travel boom"
    assert "travel boom" in records[0]["error"]["traceback"]
    assert records[0]["observation_valid"] is True
    assert task_results[0]["success"] is False
    assert task_results[0]["failure_subtype"] == "none"
    assert task_results[0]["observation_valid"] is True
    assert task_results[0]["error"]["message"] == "travel boom"


def test_travel_run_agent_inference_marks_exhausted_infra_as_invalid(
    monkeypatch, tmp_path
):
    test_data_path = tmp_path / "travel_samples.json"
    test_data_path.write_text(
        json.dumps(
            [
                {
                    "id": 0,
                    "query": "test travel query",
                    "meta_info": {"days": 2},
                }
            ]
        ),
        encoding="utf-8",
    )

    attempts = 0

    class FakeTravelRunner:
        def __init__(self, **kwargs) -> None:
            pass

        async def run_task(self, **kwargs):
            nonlocal attempts
            attempts += 1
            raise OSError("provider disconnected")

    monkeypatch.setattr(travel_module, "TravelAgentRunner", FakeTravelRunner)

    output_dir = tmp_path / "travel_output"
    results = travel_module.run_agent_inference(
        model="qwen-plus",
        language="en",
        test_data_path=test_data_path,
        database_dir=TRAVEL_DATABASE_DIR,
        tool_schema_path=TRAVEL_SCHEMA_PATH,
        output_dir=output_dir,
        workers=1,
        max_llm_calls=1,
        infra_retry_limit=1,
        system="A",
    )

    assert attempts == 2
    assert results["total"] == 0
    assert results["success"] == 0
    assert results["failed"] == 0
    assert results["invalid"] == 1
    assert results["results"][0]["failure_subtype"] == "infra_transient"
    assert results["results"][0]["observation_valid"] is False
    records = _load_jsonl(output_dir / "agent_events.jsonl")
    task_results = _load_jsonl(output_dir / "task_results.jsonl")
    assert [record["event_type"] for record in records] == [
        "task_invalid_attempt",
        "task_error",
    ]
    assert task_results[0]["failure_subtype"] == "infra_transient"
    assert task_results[0]["observation_valid"] is False


def test_travel_run_agent_inference_does_not_retry_context_exhaustion(
    monkeypatch, tmp_path
):
    test_data_path = tmp_path / "travel_samples.json"
    test_data_path.write_text(
        json.dumps(
            [
                {
                    "id": 0,
                    "query": "test travel query",
                    "meta_info": {"days": 2},
                }
            ]
        ),
        encoding="utf-8",
    )

    attempts = 0

    class FakeTravelRunner:
        def __init__(self, **kwargs) -> None:
            pass

        async def run_task(self, **kwargs):
            nonlocal attempts
            attempts += 1
            raise RuntimeError("maximum context length exceeded for this model")

    monkeypatch.setattr(travel_module, "TravelAgentRunner", FakeTravelRunner)

    output_dir = tmp_path / "travel_output"
    results = travel_module.run_agent_inference(
        model="qwen-plus",
        language="en",
        test_data_path=test_data_path,
        database_dir=TRAVEL_DATABASE_DIR,
        tool_schema_path=TRAVEL_SCHEMA_PATH,
        output_dir=output_dir,
        workers=1,
        max_llm_calls=1,
        infra_retry_limit=3,
        system="A",
    )

    assert attempts == 1
    assert results["total"] == 1
    assert results["failed"] == 1
    assert results["invalid"] == 0
    task_results = _load_jsonl(output_dir / "task_results.jsonl")
    assert task_results[0]["failure_subtype"] == "context_exhaustion"
    assert task_results[0]["observation_valid"] is True


def test_travel_run_agent_inference_serializes_tool_calls_in_trajectory(
    monkeypatch, tmp_path
):
    test_data_path = tmp_path / "travel_samples.json"
    test_data_path.write_text(
        json.dumps(
            [
                {
                    "id": 0,
                    "query": "test travel query",
                    "meta_info": {"days": 2},
                }
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        travel_module,
        "call_chat_completion",
        _fake_completion_factory(
            [
                FakeResponse(
                    content="",
                    tool_calls=[
                        FakeToolCall(
                            "call_1",
                            "query_train_info",
                            '{"departure":"A","arrival":"B"}',
                        )
                    ],
                    prompt_tokens=20,
                    completion_tokens=7,
                    finish_reason="tool_calls",
                ),
                FakeResponse(
                    content="<think>reasoning</think><plan>Day 1 itinerary</plan>",
                    prompt_tokens=11,
                    completion_tokens=5,
                ),
            ]
        ),
    )
    monkeypatch.setattr(
        travel_module.TravelAgentRunner,
        "_exec_tool",
        lambda self, name, arguments: '{"status":"ok"}',
    )

    output_root = tmp_path / "travel_output"
    results = travel_module.run_agent_inference(
        model="qwen-plus",
        language="en",
        test_data_path=test_data_path,
        database_dir=TRAVEL_DATABASE_DIR,
        tool_schema_path=TRAVEL_SCHEMA_PATH,
        output_dir=output_root,
        workers=1,
        max_llm_calls=2,
        runs=1,
        system="A",
    )

    assert results["success"] == 1
    trajectory = json.loads(
        (output_root / "trajectories" / "id_0.json").read_text(encoding="utf-8")
    )
    assistant_with_tool_call = trajectory["messages"][2]
    assert assistant_with_tool_call["tool_calls"][0]["id"] == "call_1"
    assert (
        assistant_with_tool_call["tool_calls"][0]["function"]["name"]
        == "query_train_info"
    )
    assert (
        assistant_with_tool_call["tool_calls"][0]["function"]["arguments"]
        == '{"departure":"A","arrival":"B"}'
    )


def test_travel_run_agent_inference_scopes_trace_per_case(monkeypatch, tmp_path):
    test_data_path = tmp_path / "travel_samples.json"
    test_data_path.write_text(
        json.dumps(
            [
                {
                    "id": 0,
                    "query": "test travel query 0",
                    "meta_info": {"days": 2},
                },
                {
                    "id": 1,
                    "query": "test travel query 1",
                    "meta_info": {"days": 3},
                },
            ]
        ),
        encoding="utf-8",
    )

    captured_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        travel_module,
        "build_langfuse_trace_id",
        lambda *parts: "|".join(str(part) for part in parts),
    )
    monkeypatch.setattr(
        travel_module,
        "call_chat_completion",
        _fake_completion_factory(
            [
                FakeResponse(
                    content="",
                    tool_calls=[
                        FakeToolCall(
                            "call_1",
                            "query_train_info",
                            '{"departure":"A","arrival":"B"}',
                        )
                    ],
                    prompt_tokens=20,
                    completion_tokens=7,
                    finish_reason="tool_calls",
                ),
                FakeResponse(
                    content="<think>reasoning</think><plan>Day 1 itinerary</plan>",
                    prompt_tokens=11,
                    completion_tokens=5,
                ),
                FakeResponse(
                    content="",
                    tool_calls=[
                        FakeToolCall(
                            "call_2",
                            "query_train_info",
                            '{"departure":"C","arrival":"D"}',
                        )
                    ],
                    prompt_tokens=21,
                    completion_tokens=7,
                    finish_reason="tool_calls",
                ),
                FakeResponse(
                    content="<think>reasoning</think><plan>Day 2 itinerary</plan>",
                    prompt_tokens=12,
                    completion_tokens=5,
                ),
            ],
            captured_calls,
        ),
    )
    monkeypatch.setattr(
        travel_module.TravelAgentRunner,
        "_exec_tool",
        lambda self, name, arguments: '{"status":"ok"}',
    )

    results = travel_module.run_agent_inference(
        model="qwen-plus",
        language="en",
        test_data_path=test_data_path,
        database_dir=TRAVEL_DATABASE_DIR,
        tool_schema_path=TRAVEL_SCHEMA_PATH,
        output_dir=tmp_path / "travel_output",
        workers=1,
        max_llm_calls=2,
        runs=1,
        system="A",
        session_id="bench-session",
    )

    assert results["success"] == 2
    assert [call["trace_id"] for call in captured_calls] == [
        "bench-session|travel|qwen-plus|0|0",
        "bench-session|travel|qwen-plus|0|0",
        "bench-session|travel|qwen-plus|0|1",
        "bench-session|travel|qwen-plus|0|1",
    ]
    assert {call["session_id"] for call in captured_calls} == {"bench-session"}


def test_run_benchmark_from_cfg_launches_selected_domains_and_aggregates(monkeypatch):
    shopping_calls: list[dict[str, object]] = []
    travel_calls: list[dict[str, object]] = []
    aggregate_roots: list[Path | None] = []
    monkeypatch.setattr(
        orchestration_module,
        "run_shopping",
        lambda **kwargs: shopping_calls.append(
            {
                "output_root": kwargs["output_root"],
                "langfuse_session_id": kwargs["langfuse_session_id"],
            }
        ),
    )
    monkeypatch.setattr(
        orchestration_module,
        "run_travel",
        lambda **kwargs: travel_calls.append(
            {
                "output_root": kwargs["output_root"],
                "langfuse_session_id": kwargs["langfuse_session_id"],
            }
        ),
    )
    monkeypatch.setattr(
        orchestration_module,
        "aggregate_results",
        lambda model, benchmark_output_root=None: aggregate_roots.append(
            benchmark_output_root
        ),
    )
    cfg = OmegaConf.create(
        {
            "domains": ["travel", "shopping"],
            "models": {"executor": "qwen3-14b", "overseer": "deepseek-v3.2"},
            "system": {"name": "A"},
            "runtime": {"workers": 1, "max_llm_calls": 20, "runs": 4},
            "shopping": {"levels": [1], "sample_ids": ["0"]},
            "travel": {
                "language": "en",
                "start_from": "inference",
                "evaluation_mode": "auto",
                "sample_ids": ["0"],
                "verbose": False,
                "debug": False,
            },
        }
    )

    orchestration_module.run_benchmark_from_cfg(cfg, Path("/tmp") / "bench-session")

    assert shopping_calls == [
        {
            "output_root": Path("/tmp") / "bench-session" / "shopping",
            "langfuse_session_id": "bench-session",
        }
    ]
    assert travel_calls == [
        {
            "output_root": Path("/tmp") / "bench-session" / "travel",
            "langfuse_session_id": "bench-session",
        }
    ]
    assert aggregate_roots == [Path("/tmp") / "bench-session"]


def test_run_experiment_loads_named_config_and_writes_session_metadata(
    monkeypatch, tmp_path
):
    from deepplanning import orchestration as orchestration_module

    config_root = tmp_path / "configs"
    experiments_dir = config_root / "experiments"
    experiments_dir.mkdir(parents=True)
    (experiments_dir / "named.yaml").write_text(
        "name: system-a-proof\n", encoding="utf-8"
    )

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        orchestration_module, "_session_timestamp", lambda: "2026-04-11_12-00-00"
    )
    monkeypatch.setattr(
        orchestration_module,
        "compose_config",
        lambda config_name, overrides: OmegaConf.create(
            {
                "name": "system-a-proof",
                "domains": ["travel", "shopping"],
                "models": {"executor": "qwen3-14b", "overseer": "deepseek-v3.2"},
                "system": {"name": "A"},
                "runtime": {"workers": 2, "max_llm_calls": 15, "runs": 3},
                "shopping": {"levels": [1], "sample_ids": ["1"]},
                "travel": {
                    "language": "en",
                    "start_from": "inference",
                    "evaluation_mode": "auto",
                    "sample_ids": ["0"],
                    "verbose": False,
                    "debug": False,
                },
                "output_root": str(tmp_path / "sessions"),
                "session_root": "",
                "metadata_filename": "experiment_session.json",
            }
        ),
    )
    monkeypatch.setattr(
        orchestration_module,
        "named_experiment_path",
        lambda experiment_key: (
            experiments_dir / f"{experiment_key}.yaml" if experiment_key else None
        ),
    )
    monkeypatch.setattr(
        orchestration_module,
        "run_benchmark_from_cfg",
        lambda cfg, benchmark_output_root=None: captured.update(
            {
                "cfg": cfg,
                "benchmark_output_root": benchmark_output_root,
            }
        ),
    )

    experiment_script.run("experiment=named", f"output_root={tmp_path / 'sessions'}")

    session_root = tmp_path / "sessions" / "system-a-proof" / "2026-04-11_12-00-00"
    metadata = json.loads(
        (session_root / "experiment_session.json").read_text(encoding="utf-8")
    )

    assert captured["benchmark_output_root"] == session_root
    assert metadata["experiment"]["key"] == "named"
    assert metadata["experiment"]["name"] == "system-a-proof"
    assert metadata["experiment"]["config_path"] == str(experiments_dir / "named.yaml")
    assert metadata["parameters"]["travel"]["sample_ids"] == ["0"]
    assert metadata["parameters"]["shopping"]["sample_ids"] == ["1"]
    assert metadata["parameters"]["session_root"] == str(session_root)
    assert metadata["launched_command"][:4] == [
        "pixi",
        "run",
        "deepplanning-experiment",
        "--",
    ]
    assert (session_root / "config.yaml").exists()
    assert (session_root / "overrides.txt").read_text(
        encoding="utf-8"
    ).splitlines() == [
        "experiment=named",
        f"output_root={tmp_path / 'sessions'}",
    ]


def test_real_hydra_composes_named_experiment_config():
    from deepplanning.config import compose_config

    cfg = compose_config("experiment", ["experiment=system_a_smoke"])

    assert cfg.name == "system-a-smoke"
    assert list(cfg.domains) == ["travel", "shopping"]
    assert list(cfg.shopping.levels) == [1]
    assert list(cfg.shopping.sample_ids) == ["1"]
    assert str(cfg.travel.language) == "en"
    assert list(cfg.travel.sample_ids) == ["0"]
    assert str(cfg.system.name) == "A"


def test_run_experiment_script_reaches_argument_validation_without_import_failure():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_experiment.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    combined_output = result.stdout + result.stderr

    assert result.returncode != 0
    assert "ModuleNotFoundError" not in combined_output
    assert "Experiment runs require an explicit 'name'." in combined_output


def test_checked_in_system_a_smoke_config_uses_existing_benchmark_sample_ids():
    config_path = REPO_ROOT / "configs" / "experiments" / "system_a_smoke.yaml"
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

    shopping_sample_ids = parse_id_list(config["shopping"]["sample_ids"])
    shopping_samples = load_json_file(
        shopping_runtime_module.SHOPPING_ROOT
        / "data"
        / f"level_{config['shopping']['levels'][0]}_query_meta.json"
    )
    travel_sample_ids = parse_id_list(config["travel"]["sample_ids"])
    travel_samples = load_json_file(
        travel_runtime_module.TRAVEL_ROOT
        / "data"
        / f"travelplanning_query_{config['travel']['language']}.json"
    )

    assert filter_samples_by_ids(shopping_samples, shopping_sample_ids)
    assert filter_samples_by_ids(travel_samples, travel_sample_ids)


def test_travel_wrapper_converts_and_evaluates_each_run(monkeypatch, tmp_path):
    travel_db_root = tmp_path / "travel_data"
    (travel_db_root / "database_en").mkdir(parents=True)
    test_data_path = tmp_path / "travel_samples.json"
    test_data_path.write_text(
        json.dumps([{"id": 0, "query": "test"}]), encoding="utf-8"
    )

    monkeypatch.setattr(travel_runtime_module, "TRAVEL_DATA_ROOT", travel_db_root)
    monkeypatch.setattr(
        travel_runtime_module,
        "prepare_test_data",
        lambda language, output_dir, sample_ids: test_data_path,
    )

    inference_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        travel_runtime_module.travel_agent_runner,
        "run_agent_inference",
        lambda **kwargs: inference_calls.append(kwargs) or {"success": 2, "total": 2},
    )

    conversion_dirs: list[Path] = []
    evaluation_dirs: list[Path] = []

    class FakeConvert:
        @staticmethod
        def convert_reports(**kwargs):
            conversion_dirs.append(kwargs["result_dir"])
            converted_dir = kwargs["result_dir"] / "converted_plans"
            converted_dir.mkdir(parents=True, exist_ok=True)
            (converted_dir / "id_0_converted.json").write_text(
                json.dumps({"daily_plans": []}),
                encoding="utf-8",
            )
            return {
                "total": 1,
                "converted": 1,
                "skipped": 0,
                "success": 1,
                "failed": 0,
                "results": [{"success": True, "sample_id": "0"}],
            }

    class FakeEval:
        @staticmethod
        def evaluate_plans(**kwargs):
            evaluation_dirs.append(kwargs["result_dir"])
            evaluation_dir = kwargs["result_dir"] / "evaluation"
            evaluation_dir.mkdir(parents=True, exist_ok=True)
            (evaluation_dir / "id_0_score.json").write_text(
                json.dumps({"sample_id": "0"}),
                encoding="utf-8",
            )
            (evaluation_dir / "evaluation_summary.json").write_text(
                json.dumps(
                    {
                        "total_test_samples": 1,
                        "evaluation_success_count": 1,
                        "metrics": {
                            "composite_score": 1.0,
                            "case_acc": 1.0,
                            "commonsense_score": 1.0,
                            "personalized_score": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            return {
                "total": 1,
                "success": 1,
                "failed": 0,
                "results": [{"success": True, "sample_id": "0"}],
                "metrics": {
                    "composite_score": 1.0,
                    "case_acc": 1.0,
                    "commonsense_score": 1.0,
                    "personalized_score": 1.0,
                },
            }

    cfg = SimpleNamespace(
        output_root=str(tmp_path / "travel_output"),
        start_from="inference",
        workers=1,
        max_llm_calls=2,
        verbose=False,
        evaluation_mode="auto",
    )

    travel_runtime_module.run_language(
        model="qwen3-14b",
        language="en",
        sample_ids=["0"],
        system="A",
        runs=2,
        cfg=cfg,
        convert_report=FakeConvert,
        eval_converted=FakeEval,
    )

    assert inference_calls[0]["runs"] == 2
    assert sorted(inference_calls[0]["output_dir_by_run"]) == [0, 1]
    assert conversion_dirs == [
        tmp_path / "travel_output" / "qwen3-14b_en" / "run_0",
        tmp_path / "travel_output" / "qwen3-14b_en" / "run_1",
    ]
    assert evaluation_dirs == conversion_dirs
    status = json.loads(
        (
            tmp_path
            / "travel_output"
            / "qwen3-14b_en"
            / "run_0"
            / "travel_run_status.json"
        ).read_text(encoding="utf-8")
    )
    assert status["conversion_complete"] is True
    assert status["full_eval_complete"] is True
    assert status["fallback_eval_only"] is False


def test_travel_wrapper_writes_generated_data_only_summary_on_conversion_failure(
    monkeypatch, tmp_path
):
    travel_db_root = tmp_path / "travel_data"
    (travel_db_root / "database_en").mkdir(parents=True)
    test_data_path = tmp_path / "travel_samples.json"
    test_data_path.write_text(
        json.dumps([{"id": 0, "query": "test"}]),
        encoding="utf-8",
    )

    monkeypatch.setattr(travel_runtime_module, "TRAVEL_DATA_ROOT", travel_db_root)
    monkeypatch.setattr(
        travel_runtime_module,
        "prepare_test_data",
        lambda language, output_dir, sample_ids: test_data_path,
    )

    def fake_inference(**kwargs):
        run_output_dir = kwargs["output_dir_by_run"][0]
        (run_output_dir / "reports" / "id_0.txt").write_text(
            "Day 1 itinerary",
            encoding="utf-8",
        )
        (run_output_dir / "trajectories" / "id_0.json").write_text(
            json.dumps({"id": "id_0"}),
            encoding="utf-8",
        )
        (run_output_dir / "task_results.jsonl").write_text(
            json.dumps(
                {
                    "task_id": "id_0",
                    "run_id": 0,
                    "executor_calls": 1,
                    "tool_call_count": 0,
                    "final_stop_reason": "no_tool_calls",
                    "final_output_present": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {"success": 1, "total": 1}

    monkeypatch.setattr(
        travel_runtime_module.travel_agent_runner, "run_agent_inference", fake_inference
    )

    class FakeConvert:
        @staticmethod
        def convert_reports(**kwargs):
            return {
                "total": 1,
                "converted": 0,
                "skipped": 0,
                "success": 0,
                "failed": 1,
                "results": [
                    {"success": False, "sample_id": "0", "error": "parse failure"}
                ],
            }

    class FakeEval:
        calls = 0

        @staticmethod
        def evaluate_plans(**kwargs):
            FakeEval.calls += 1
            return {"total": 0, "success": 0, "failed": 0, "results": []}

    cfg = SimpleNamespace(
        output_root=str(tmp_path / "travel_output"),
        start_from="inference",
        workers=1,
        max_llm_calls=2,
        verbose=False,
        evaluation_mode="auto",
    )

    travel_runtime_module.run_language(
        model="qwen3-14b",
        language="en",
        sample_ids=["0"],
        system="A",
        runs=1,
        cfg=cfg,
        convert_report=FakeConvert,
        eval_converted=FakeEval,
    )

    run_output_dir = tmp_path / "travel_output" / "qwen3-14b_en" / "run_0"
    status = json.loads(
        (run_output_dir / "travel_run_status.json").read_text(encoding="utf-8")
    )
    fallback = json.loads(
        (
            run_output_dir
            / "generated_data_only_evaluation"
            / "generated_data_only_summary.json"
        ).read_text(encoding="utf-8")
    )

    assert FakeEval.calls == 1
    assert status["conversion_complete"] is False
    assert status["full_eval_complete"] is False
    assert status["fallback_eval_only"] is True
    assert fallback["mode"] == "generated_data_only"
    assert fallback["sample_statuses"][0]["conversion_status"] == "failed"
    assert fallback["sample_statuses"][0]["evaluation_status"] == "generated_data_only"


def test_travel_sample_statuses_include_observation_valid(tmp_path):
    test_data_path = tmp_path / "travel_samples.json"
    test_data_path.write_text(
        json.dumps([{"id": 0, "query": "test travel query"}]),
        encoding="utf-8",
    )

    run_output_dir = tmp_path / "travel_run"
    run_output_dir.mkdir()
    (run_output_dir / "task_results.jsonl").write_text(
        json.dumps(
            {
                "task_id": "id_0",
                "run_id": 0,
                "success": False,
                "failure_subtype": "infra_transient",
                "observation_valid": False,
                "final_output_present": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    statuses = travel_runtime_module._build_sample_statuses(
        test_data_path=test_data_path,
        run_output_dir=run_output_dir,
        conversion_results=None,
        evaluation_results=None,
        fallback_active=False,
    )

    assert statuses[0]["failure_subtype"] == "infra_transient"
    assert statuses[0]["observation_valid"] is False
    assert statuses[0]["task_metrics"]["observation_valid"] is False


def test_shopping_wrapper_creates_isolated_run_layouts(monkeypatch, tmp_path):
    from deepplanning import shopping_runner as shopping_runtime_module

    (tmp_path / "database_level1").mkdir()
    test_data_path = tmp_path / "shopping_samples.json"
    test_data_path.write_text(
        json.dumps([{"id": 1, "query": "test shopping query", "level": 1}]),
        encoding="utf-8",
    )

    monkeypatch.setattr(shopping_runtime_module, "SHOPPING_DATA_ROOT", tmp_path)
    monkeypatch.setattr(shopping_runtime_module, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        shopping_runtime_module,
        "import_modules",
        lambda: (object(), object()),
    )
    monkeypatch.setattr(shopping_runtime_module, "load_model_config", lambda model: {})

    prepared_databases: list[Path] = []

    def fake_prepare_run_inputs(
        level, source_database_dir, run_database_dir, sample_ids
    ):
        prepared_databases.append(run_database_dir)
        run_database_dir.mkdir(parents=True, exist_ok=True)
        (run_database_dir / "case_1").mkdir()
        return test_data_path

    monkeypatch.setattr(
        shopping_runtime_module, "prepare_run_inputs", fake_prepare_run_inputs
    )

    inference_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        shopping_runtime_module.shopping_agent_runner,
        "run_agent_inference",
        lambda **kwargs: inference_calls.append(kwargs) or {"success": 2, "total": 2},
    )

    evaluation_calls: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        shopping_runtime_module,
        "evaluate_database",
        lambda database_dir, report_dir, evaluation_pipeline: evaluation_calls.append(
            (database_dir, report_dir)
        ),
    )

    statistics_roots: list[Path] = []
    monkeypatch.setattr(
        shopping_runtime_module,
        "write_statistics",
        lambda model, result_report_root, score_statistics: statistics_roots.append(
            result_report_root
        ),
    )

    shopping_runtime_module.run(
        levels=[1], runs=2, output_root=tmp_path / "shopping_output"
    )

    assert inference_calls[0]["runs"] == 2
    assert sorted(inference_calls[0]["database_dir_by_run"]) == [0, 1]
    assert sorted(inference_calls[0]["output_dir_by_run"]) == [0, 1]
    assert prepared_databases[0] != prepared_databases[1]
    assert all(
        f"run_{run_id}" in str(path) for run_id, path in enumerate(prepared_databases)
    )
    assert len(evaluation_calls) == 2
    assert len(statistics_roots) == 2


def test_aggregate_results_keeps_travel_fallback_artifacts_under_session_root(tmp_path):
    session_root = tmp_path / "session"
    travel_run_dir = session_root / "travel" / "qwen3-14b_en" / "run_0"
    travel_run_dir.mkdir(parents=True)
    (travel_run_dir / "travel_run_status.json").write_text(
        json.dumps(
            {
                "inference_complete": True,
                "conversion_complete": False,
                "full_eval_complete": False,
                "fallback_eval_only": True,
                "official_evaluation_present": False,
                "official_evaluation_summary_path": None,
                "generated_data_only_summary_path": str(
                    travel_run_dir
                    / "generated_data_only_evaluation"
                    / "generated_data_only_summary.json"
                ),
            }
        ),
        encoding="utf-8",
    )

    aggregation_module.aggregate_results(
        "qwen3-14b", benchmark_output_root=session_root
    )

    aggregated = json.loads(
        (
            session_root / "aggregated_results" / "qwen3-14b_run_0_aggregated.json"
        ).read_text(encoding="utf-8")
    )
    assert (
        aggregated["domains"]["travel_artifacts"]["languages"]["en"][
            "fallback_eval_only"
        ]
        is True
    )
    assert aggregated["overall"]["travel_official_metrics_available"] is False
    assert aggregated["overall"]["travel_generated_data_only_available"] is True
