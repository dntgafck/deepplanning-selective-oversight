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

        messages = await self._run_phase(
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
        messages = await self._run_phase(
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

        state.finish()
        return TaskResult(
            task_id=state.task_id,
            run_id=run_id,
            output=_extract_final_output(messages),
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
    ) -> list[Any]:
        for step_count in range(1, system_config.max_steps + 1):
            response = await call_chat_completion(
                provider=system_config.executor_provider,
                messages=messages,
                tools=self.openai_tools,
            )
            state.record_executor_call(response)

            prompt_tokens, completion_tokens = extract_usage_tokens(response)
            if logger is not None:
                await logger.log_event(
                    "llm_call",
                    {
                        "domain": "shopping",
                        "phase": phase_name,
                        "task_id": state.task_id,
                        "run_id": run_id,
                        "step": step_count,
                        "model": system_config.executor_provider.alias,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    },
                )

            assistant_message = response.choices[0].message
            calls = self._detect_tool_calls(assistant_message)

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

            assistant_payload = _assistant_message_to_dict(assistant_message, calls)
            messages.append(assistant_payload)
            if save_messages and messages_file is not None:
                self._save_messages(
                    messages,
                    messages_file,
                    step_count,
                    f"LLM response ({phase_name}) - {len(calls)} tool calls",
                )

            if not calls:
                if stop_on_no_calls:
                    return messages
                break

            for call in calls:
                tool_result = self._exec_tool(call["name"], call["arguments"])
                state.record_tool_call(call, tool_result)
                if logger is not None:
                    await logger.log_event(
                        "tool_call",
                        {
                            "domain": "shopping",
                            "task_id": state.task_id,
                            "run_id": run_id,
                            "step": step_count,
                            "tool_name": call["name"],
                            "tool_call_id": call["id"],
                            "result_preview": tool_result[:160],
                        },
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

        return messages


async def run_agent_inference_async(
    model: str,
    test_data_path: Path,
    database_dir: Path,
    tool_schema_path: Path,
    system_prompt: str,
    workers: int = 10,
    max_llm_calls: int = 100,
    rerun_ids: list[int] | None = None,
    system: str = "A",
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

    logger = StructuredLogger(database_dir)
    system_config = build_system_config(
        system_name=system,
        executor_model=model,
        max_steps=max_llm_calls,
    )
    started_at = time.time()

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
            runner = ShoppingAgentRunner(
                model=system_config.executor_provider.alias,
                sample_id=sample_id,
                database_base_path=str(database_dir),
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
        test_data, process_sample, parallel=workers, runs=1
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
        "results": results,
    }


def run_agent_inference(
    model: str,
    test_data_path: Path,
    database_dir: Path,
    tool_schema_path: Path,
    system_prompt: str,
    workers: int = 10,
    max_llm_calls: int = 100,
    rerun_ids: list[int] | None = None,
    system: str = "A",
) -> dict[str, Any]:
    return asyncio.run(
        run_agent_inference_async(
            model=model,
            test_data_path=test_data_path,
            database_dir=database_dir,
            tool_schema_path=tool_schema_path,
            system_prompt=system_prompt,
            workers=workers,
            max_llm_calls=max_llm_calls,
            rerun_ids=rerun_ids,
            system=system,
        )
    )
