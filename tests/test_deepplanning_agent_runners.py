from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from agent import shopping as shopping_module
from agent import travel as travel_module
from experiment import StructuredLogger, build_system_config
from oversight import ConversationState
from scripts import run_deepplanning_benchmark as benchmark_script
from scripts import run_deepplanning_shopping as shopping_script
from scripts import run_deepplanning_travel as travel_script
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
    assert captured_calls[0]["reasoning_enabled"] is False
    assert captured_calls[1]["reasoning_enabled"] is False
    assert isinstance(captured_calls[1]["messages"][2], FakeMessage)
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
    assert getattr(result.messages[-1], "content", "") == (
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
    assert captured_calls[0]["reasoning_enabled"] is False
    assert captured_calls[1]["reasoning_enabled"] is False
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
    assert captured_calls[0]["reasoning_enabled"] is False
    assert captured_calls[1]["reasoning_enabled"] is False
    assert any(
        isinstance(message, dict)
        and message.get("role") == "tool"
        and message.get("tool_call_id") == "call_1"
        and "name" not in message
        for message in result.messages
    )


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


def test_benchmark_wrapper_forwards_runs_to_domain_scripts(monkeypatch):
    from deepplanning import orchestration as orchestration_module

    shopping_calls: list[Path] = []
    travel_calls: list[Path] = []
    aggregate_roots: list[Path | None] = []
    monkeypatch.setattr(
        orchestration_module,
        "run_shopping",
        lambda **kwargs: shopping_calls.append(kwargs["output_root"]),
    )
    monkeypatch.setattr(
        orchestration_module,
        "run_travel",
        lambda **kwargs: travel_calls.append(kwargs["output_root"]),
    )
    monkeypatch.setattr(
        orchestration_module,
        "aggregate_results",
        lambda model, benchmark_output_root=None: aggregate_roots.append(
            benchmark_output_root
        ),
    )
    monkeypatch.setattr(
        orchestration_module,
        "compose_config",
        lambda config_name, overrides: OmegaConf.create(
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
                "output_root": str(Path("/tmp") / "bench-session"),
                "session_root": "",
            }
        ),
    )

    benchmark_script.run(runs=4, output_root=str(Path("/tmp") / "bench-session"))

    assert shopping_calls == [Path("/tmp") / "bench-session" / "shopping"]
    assert travel_calls == [Path("/tmp") / "bench-session" / "travel"]
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


def test_checked_in_system_a_smoke_config_uses_existing_benchmark_sample_ids():
    config_path = REPO_ROOT / "configs" / "experiments" / "system_a_smoke.yaml"
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

    shopping_sample_ids = parse_id_list(config["shopping"]["sample_ids"])
    shopping_samples = load_json_file(
        shopping_script.SHOPPING_ROOT
        / "data"
        / f"level_{config['shopping']['levels'][0]}_query_meta.json"
    )
    travel_sample_ids = parse_id_list(config["travel"]["sample_ids"])
    travel_samples = load_json_file(
        travel_script.TRAVEL_ROOT
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

    monkeypatch.setattr(travel_script, "TRAVEL_DATA_ROOT", travel_db_root)
    monkeypatch.setattr(
        travel_script,
        "prepare_test_data",
        lambda language, output_dir, sample_ids: test_data_path,
    )

    inference_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        travel_script.travel_runner,
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

    travel_script.run_language(
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

    monkeypatch.setattr(travel_script, "TRAVEL_DATA_ROOT", travel_db_root)
    monkeypatch.setattr(
        travel_script,
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
        travel_script.travel_runner, "run_agent_inference", fake_inference
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

    travel_script.run_language(
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


def test_shopping_wrapper_creates_isolated_run_layouts(monkeypatch, tmp_path):
    from deepplanning import orchestration as orchestration_module
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
    monkeypatch.setattr(
        orchestration_module,
        "compose_config",
        lambda config_name, overrides: OmegaConf.create(
            {
                "domains": ["shopping"],
                "models": {"executor": "qwen3-14b", "overseer": "deepseek-v3.2"},
                "system": {"name": "A"},
                "runtime": {"workers": 1, "max_llm_calls": 2, "runs": 2},
                "shopping": {"levels": [1], "sample_ids": []},
                "travel": {
                    "language": "en",
                    "start_from": "inference",
                    "evaluation_mode": "auto",
                    "sample_ids": [],
                    "verbose": False,
                    "debug": False,
                },
                "output_root": str(tmp_path / "shopping_output"),
                "session_root": "",
            }
        ),
    )

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

    shopping_script.run(runs=2)

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

    benchmark_script.aggregate_results("qwen3-14b", benchmark_output_root=session_root)

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
