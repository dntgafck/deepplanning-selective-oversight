from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from agent import shopping as shopping_module
from agent import travel as travel_module
from experiment import build_system_config
from oversight import ConversationState

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


class FakeToolFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.function = FakeToolFunction(name, arguments)


class FakeMessage:
    def __init__(
        self, content: str, tool_calls: list[FakeToolCall] | None = None
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls


class FakeChoice:
    def __init__(self, message: FakeMessage) -> None:
        self.message = message


class FakeResponse:
    def __init__(
        self,
        content: str,
        tool_calls: list[FakeToolCall] | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        self.choices = [FakeChoice(FakeMessage(content=content, tool_calls=tool_calls))]
        self.usage = FakeUsage(prompt_tokens, completion_tokens)


def _fake_completion_factory(responses: list[FakeResponse]):
    queue = list(responses)

    async def _fake_call(*args, **kwargs):
        return queue.pop(0)

    return _fake_call


def test_shopping_runner_preserves_tool_message_shape_and_second_phase(
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
                    content="",
                    tool_calls=[FakeToolCall("call_1", "get_cart_info", "{}")],
                    prompt_tokens=10,
                    completion_tokens=4,
                ),
                FakeResponse(
                    content="Phase one complete.",
                    prompt_tokens=11,
                    completion_tokens=5,
                ),
                FakeResponse(
                    content="Cart verified.",
                    prompt_tokens=12,
                    completion_tokens=6,
                ),
            ]
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
    system_config = build_system_config("A", executor_model="qwen-plus", max_steps=5)

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
    assert state.executor_calls == 3
    assert any(
        isinstance(message, dict)
        and message.get("role") == "tool"
        and "tool_call_id" in message
        and "name" not in message
        for message in result.messages
    )
    assert any(
        isinstance(message, dict)
        and message.get("role") == "user"
        and "Check whether the items in the shopping cart meet the requirements."
        in message.get("content", "")
        for message in result.messages
    )
    assert (run_database_dir / "case_1" / "messages.json").exists()


def test_travel_runner_extracts_plan_content_from_benchmark_message_format(monkeypatch):
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
    assert result.messages[0]["role"] == "system"
    assert (
        getattr(result.messages[-1], "content", "")
        == "<think>reasoning</think><plan>Day 1 itinerary</plan>"
    )
