from __future__ import annotations

import asyncio
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from experiment import StructuredLogger, build_system_config, run_experiment
from llm import call_chat_completion, extract_usage_tokens
from oversight import ConversationState, apply_intervention, evaluate_oversight

from .base import TaskResult
from .vendor import load_shopping_agent_class, load_shopping_prompt

VendorShoppingFnAgent = load_shopping_agent_class()


def get_system_prompt(level: int) -> str:
    return load_shopping_prompt(level)


def _executor_reasoning_enabled(system_config: Any) -> bool | None:
    return False if getattr(system_config, "name", None) == "A" else None


def _collect_tool_call_parse_warnings(assistant_message: Any) -> list[str]:
    warnings: list[str] = []
    for index, tool_call in enumerate(
        getattr(assistant_message, "tool_calls", None) or []
    ):
        try:
            function = tool_call.function
            _ = function.name
            _ = function.arguments
        except Exception as exc:
            warnings.append(f"tool_calls[{index}] skipped: {exc}")
    return warnings


def _assistant_message_to_dict(msg: Any, calls: list[dict[str, Any]]) -> dict[str, Any]:
    message = {
        "role": "assistant",
        "content": getattr(msg, "content", None) or "",
    }
    reasoning_content = getattr(msg, "reasoning_content", None)
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    if calls:
        message["tool_calls"] = [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": call["arguments"],
                },
            }
            for call in calls
        ]
    return message


def _extract_final_output(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def _resolve_run_database_dir(
    base_database_dir: Path,
    run_id: int,
    database_dir_by_run: dict[int, Path] | None,
    runs: int,
) -> Path:
    if database_dir_by_run is not None:
        return database_dir_by_run[run_id]
    if runs > 1:
        raise ValueError(
            "Shopping multi-run execution requires per-run database directories."
        )
    return base_database_dir


def _resolve_run_output_dir(
    base_output_dir: Path | None,
    run_id: int,
    output_dir_by_run: dict[int, Path] | None,
    runs: int,
) -> Path | None:
    if output_dir_by_run is not None:
        return output_dir_by_run[run_id]
    if base_output_dir is None:
        return None
    if runs > 1:
        return base_output_dir / f"run_{run_id}"
    return base_output_dir


class ShoppingAgentRunner(VendorShoppingFnAgent):
    async def run_task(
        self,
        user_query: str,
        system_prompt: str | None,
        state: ConversationState,
        system_config: Any,
        logger: StructuredLogger | None = None,
        run_id: int = 0,
        save_messages: bool = True,
        sample_id: str | None = None,
        messages_output_dir: str | None = None,
    ) -> TaskResult:
        state.begin()

        messages_file: Path | None = None
        if save_messages:
            if sample_id:
                db_case_dir = self.database_base_path / f"case_{sample_id}"
                db_case_dir.mkdir(parents=True, exist_ok=True)
                messages_file = db_case_dir / "messages.json"
            else:
                msg_dir = Path(
                    messages_output_dir
                    or (Path(__file__).resolve().parent / "result" / "messages")
                )
                msg_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                messages_file = msg_dir / f"messages_{timestamp}.json"

        messages: list[Any] = (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        ) + [{"role": "user", "content": user_query}]
        if save_messages and messages_file is not None:
            self._save_messages(messages, messages_file, 0, "Initial messages")

        messages, _ = await self._run_phase(
            messages=messages,
            state=state,
            system_config=system_config,
            logger=logger,
            run_id=run_id,
            phase_name="initial",
            stop_on_no_calls=False,
            save_messages=save_messages,
            messages_file=messages_file,
        )
        messages = self._add_to_cart(messages)
        messages, phase_stop_reason = await self._run_phase(
            messages=messages,
            state=state,
            system_config=system_config,
            logger=logger,
            run_id=run_id,
            phase_name="cart_check",
            stop_on_no_calls=True,
            save_messages=save_messages,
            messages_file=messages_file,
        )

        final_output = _extract_final_output(messages)
        final_stop_reason = (
            "no_tool_calls"
            if phase_stop_reason == "no_tool_calls"
            else "max_steps_exhausted"
        )
        state.record_final_outcome(
            stop_reason=final_stop_reason,
            output=final_output,
            max_steps_hit=phase_stop_reason != "no_tool_calls",
        )
        state.finish()
        return TaskResult(
            task_id=state.task_id,
            run_id=run_id,
            output=final_output,
            messages=messages,
            state=state,
        )

    async def _run_phase(
        self,
        messages: list[Any],
        state: ConversationState,
        system_config: Any,
        logger: StructuredLogger | None,
        run_id: int,
        phase_name: str,
        stop_on_no_calls: bool,
        save_messages: bool,
        messages_file: Path | None,
    ) -> tuple[list[Any], str]:
        for step_count in range(1, system_config.max_steps + 1):
            request_messages = list(messages)
            started_at = time.time()
            response = await call_chat_completion(
                provider=system_config.executor_provider,
                messages=request_messages,
                tools=self.openai_tools,
                reasoning_enabled=_executor_reasoning_enabled(system_config),
            )
            ended_at = time.time()
            state.record_executor_call(response)

            prompt_tokens, completion_tokens = extract_usage_tokens(response)
            assistant_message = response.choices[0].message
            calls = self._detect_tool_calls(assistant_message)
            parse_warnings = _collect_tool_call_parse_warnings(assistant_message)

            if calls and system_config.oversight_enabled:
                action = evaluate_oversight(response, messages, state, system_config)
                state.record_oversight_decision(step_count, action)
                if logger is not None:
                    await logger.log_event(
                        "oversight_decision",
                        {
                            "domain": "shopping",
                            "task_id": state.task_id,
                            "run_id": run_id,
                            "step": step_count,
                            "trigger_reason": action.trigger_reason,
                            "intervention_type": action.intervention_type,
                            "should_intervene": action.should_intervene,
                        },
                    )
                if action.should_intervene:
                    response = await apply_intervention(
                        action=action,
                        original_response=response,
                        messages=messages,
                        state=state,
                        config=system_config,
                    )
                    assistant_message = response.choices[0].message
                    calls = self._detect_tool_calls(assistant_message)
                    parse_warnings = _collect_tool_call_parse_warnings(
                        assistant_message
                    )

            assistant_payload = _assistant_message_to_dict(assistant_message, calls)
            messages.append(assistant_payload)
            if save_messages and messages_file is not None:
                self._save_messages(
                    messages,
                    messages_file,
                    step_count,
                    f"LLM response ({phase_name}) - {len(calls)} tool calls",
                )

            tool_results: list[dict[str, Any]] = []
            if not calls:
                if logger is not None:
                    await logger.log_turn(
                        domain="shopping",
                        task_id=state.task_id,
                        run_id=run_id,
                        phase=phase_name,
                        turn_index=step_count,
                        started_at=started_at,
                        ended_at=ended_at,
                        request_messages=request_messages,
                        raw_response=response,
                        parsed_tool_calls=calls,
                        parse_warnings=parse_warnings,
                        tool_results=tool_results,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        stop_reason="no_tool_calls",
                        model_alias=system_config.executor_provider.alias,
                    )
                return messages, "no_tool_calls" if stop_on_no_calls else "phase_break"

            for call in calls:
                tool_result = self._exec_tool(call["name"], call["arguments"])
                state.record_tool_call(call, tool_result)
                tool_results.append(
                    {
                        "tool_name": call["name"],
                        "tool_call_id": call["id"],
                        "content": tool_result,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": tool_result,
                    }
                )
            if save_messages and messages_file is not None:
                self._save_messages(
                    messages,
                    messages_file,
                    step_count,
                    f"Tool execution ({phase_name}) - {len(calls)} tools",
                )
            if logger is not None:
                await logger.log_turn(
                    domain="shopping",
                    task_id=state.task_id,
                    run_id=run_id,
                    phase=phase_name,
                    turn_index=step_count,
                    started_at=started_at,
                    ended_at=ended_at,
                    request_messages=request_messages,
                    raw_response=response,
                    parsed_tool_calls=calls,
                    parse_warnings=parse_warnings,
                    tool_results=tool_results,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    stop_reason="tool_calls",
                    model_alias=system_config.executor_provider.alias,
                )

        return messages, "max_steps_exhausted"


async def run_agent_inference_async(
    model: str,
    test_data_path: Path,
    database_dir: Path,
    tool_schema_path: Path,
    system_prompt: str,
    output_dir: Path | None = None,
    workers: int = 10,
    max_llm_calls: int = 100,
    runs: int = 1,
    rerun_ids: list[int] | None = None,
    system: str = "A",
    database_dir_by_run: dict[int, Path] | None = None,
    output_dir_by_run: dict[int, Path] | None = None,
) -> dict[str, Any]:
    with test_data_path.open("r", encoding="utf-8") as fh:
        test_data = json.load(fh)

    if rerun_ids is not None:
        rerun_ids_set = {str(sample_id) for sample_id in rerun_ids}
        original_count = len(test_data)
        test_data = [
            sample for sample in test_data if str(sample.get("id")) in rerun_ids_set
        ]
        print(
            f"  🔄 Filtered {original_count} samples to {len(test_data)} samples for rerun"
        )
        if not test_data:
            return {
                "total": 0,
                "success": 0,
                "failed": 0,
                "elapsed_time": 0,
                "results": [],
            }

    print(f"\n{'=' * 80}")
    print("Agent Inference")
    print(f"{'=' * 80}")
    print(f"Model: {model}")
    print(f"Samples: {len(test_data)}")
    print(f"Workers: {workers}")
    print(f"{'=' * 80}\n")

    system_config = build_system_config(
        system_name=system,
        executor_model=model,
        max_steps=max_llm_calls,
        num_runs=runs,
    )
    started_at = time.time()
    loggers: dict[int, StructuredLogger] = {}

    def resolve_run_database_dir(run_id: int) -> Path:
        return _resolve_run_database_dir(
            database_dir, run_id, database_dir_by_run, runs
        )

    def resolve_run_output_dir(run_id: int) -> Path | None:
        run_output_dir = _resolve_run_output_dir(
            output_dir,
            run_id,
            output_dir_by_run,
            runs,
        )
        if run_output_dir is not None:
            run_output_dir.mkdir(parents=True, exist_ok=True)
        return run_output_dir

    def get_logger(run_id: int) -> StructuredLogger:
        if run_id not in loggers:
            loggers[run_id] = StructuredLogger(resolve_run_output_dir(run_id))
        return loggers[run_id]

    async def process_sample(sample: object, run_id: int) -> dict[str, Any]:
        sample = dict(sample)
        sample_id = str(sample.get("id", "unknown"))
        query = str(sample.get("query", ""))
        state = ConversationState(
            task_id=sample_id,
            domain="shopping",
            complexity=int(sample.get("level", 0) or 0) or None,
            system_config_name=system,
        )

        try:
            run_database_dir = resolve_run_database_dir(run_id)
            logger = get_logger(run_id)
            runner = ShoppingAgentRunner(
                model=system_config.executor_provider.alias,
                sample_id=sample_id,
                database_base_path=str(run_database_dir),
                tool_schema_path=str(tool_schema_path),
            )
            result = await runner.run_task(
                user_query=query,
                system_prompt=system_prompt,
                state=state,
                system_config=system_config,
                logger=logger,
                run_id=run_id,
                save_messages=True,
                sample_id=sample_id,
            )
            await logger.log_result(
                {
                    **result.state.to_metrics(),
                    "task_id": sample_id,
                    "run_id": run_id,
                    "system": system,
                    "domain": "shopping",
                    "final_output": result.output,
                }
            )
            print(f"✅ Sample {sample_id} completed in {state.wall_time_seconds:.2f}s")
            return {
                "id": sample_id,
                "query": query,
                "model": model,
                "messages": result.messages,
                "elapsed_time": state.wall_time_seconds,
                "success": True,
            }
        except Exception as exc:
            await logger.log_event(
                "task_error",
                {
                    "domain": "shopping",
                    "task_id": sample_id,
                    "run_id": run_id,
                    "error": str(exc),
                },
            )
            print(f"❌ Sample {sample_id} failed: {exc}")
            traceback.print_exc()
            return {
                "id": sample_id,
                "query": query,
                "success": False,
                "error": str(exc),
            }

    raw_results = await run_experiment(
        test_data, process_sample, parallel=workers, runs=runs
    )
    results: list[dict[str, Any]] = []
    for item in raw_results:
        if isinstance(item, Exception):
            results.append({"success": False, "error": str(item)})
        else:
            results.append(item)

    success_count = sum(1 for result in results if result.get("success"))
    return {
        "total": len(results),
        "success": success_count,
        "failed": len(results) - success_count,
        "elapsed_time": time.time() - started_at,
        "runs": runs,
        "results": results,
    }


def run_agent_inference(
    model: str,
    test_data_path: Path,
    database_dir: Path,
    tool_schema_path: Path,
    system_prompt: str,
    output_dir: Path | None = None,
    workers: int = 10,
    max_llm_calls: int = 100,
    runs: int = 1,
    rerun_ids: list[int] | None = None,
    system: str = "A",
    database_dir_by_run: dict[int, Path] | None = None,
    output_dir_by_run: dict[int, Path] | None = None,
) -> dict[str, Any]:
    return asyncio.run(
        run_agent_inference_async(
            model=model,
            test_data_path=test_data_path,
            database_dir=database_dir,
            tool_schema_path=tool_schema_path,
            system_prompt=system_prompt,
            output_dir=output_dir,
            workers=workers,
            max_llm_calls=max_llm_calls,
            runs=runs,
            rerun_ids=rerun_ids,
            system=system,
            database_dir_by_run=database_dir_by_run,
            output_dir_by_run=output_dir_by_run,
        )
    )
