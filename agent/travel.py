from __future__ import annotations

import asyncio
import json
import time
import traceback
from pathlib import Path
from typing import Any

from experiment import StructuredLogger, build_system_config, run_experiment
from experiment.logging import serialize_messages
from llm import call_chat_completion, extract_usage_tokens
from oversight import ConversationState, apply_intervention, evaluate_oversight

from .base import TaskResult
from .vendor import load_travel_agent_class, load_travel_prompt

VendorTravelFnAgent = load_travel_agent_class()


def get_system_prompt(language: str) -> str:
    return load_travel_prompt(language)


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


def _resolve_run_output_dir(
    base_output_dir: Path,
    run_id: int,
    output_dir_by_run: dict[int, Path] | None,
    runs: int,
) -> Path:
    if output_dir_by_run is not None:
        return output_dir_by_run[run_id]
    if runs > 1:
        return base_output_dir / f"run_{run_id}"
    return base_output_dir


class TravelAgentRunner(VendorTravelFnAgent):
    async def run_task(
        self,
        user_query: str,
        system_prompt: str | None,
        state: ConversationState,
        system_config: Any,
        logger: StructuredLogger | None = None,
        run_id: int = 0,
    ) -> TaskResult:
        state.begin()

        messages: list[Any] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_query})

        llm_budget = system_config.max_steps
        while llm_budget > 0:
            llm_budget -= 1
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

            step_count = system_config.max_steps - llm_budget
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
                            "domain": "travel",
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

            messages.append(assistant_message)
            tool_results: list[dict[str, Any]] = []
            if calls:
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
                            "name": call["name"],
                            "content": tool_result,
                        }
                    )
                if logger is not None:
                    await logger.log_turn(
                        domain="travel",
                        task_id=state.task_id,
                        run_id=run_id,
                        phase="travel",
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
                continue

            final_plan = self._extract_plan_content(
                getattr(assistant_message, "content", None) or ""
            )
            state.record_final_outcome(
                stop_reason="no_tool_calls",
                output=final_plan,
                max_steps_hit=False,
            )
            if logger is not None:
                await logger.log_turn(
                    domain="travel",
                    task_id=state.task_id,
                    run_id=run_id,
                    phase="travel",
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
            state.finish()
            return TaskResult(
                task_id=state.task_id,
                run_id=run_id,
                output=final_plan,
                messages=messages,
                state=state,
            )

        state.record_final_outcome(
            stop_reason="max_steps_exhausted",
            output=None,
            max_steps_hit=True,
        )
        state.finish()
        return TaskResult(
            task_id=state.task_id,
            run_id=run_id,
            output="Reached max LLM calls without final answer.",
            messages=messages,
            state=state,
        )


async def run_agent_inference_async(
    model: str,
    language: str,
    test_data_path: Path,
    database_dir: Path,
    tool_schema_path: Path,
    output_dir: Path,
    workers: int = 10,
    max_llm_calls: int = 100,
    runs: int = 1,
    rerun_ids: list[int] | None = None,
    system: str = "A",
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
    print(f"Language: {language}")
    print(f"Samples: {len(test_data)}")
    print(f"Workers: {workers}")
    print(f"{'=' * 80}\n")

    system_config = build_system_config(
        system_name=system,
        executor_model=model,
        max_steps=max_llm_calls,
        num_runs=runs,
    )
    system_prompt = get_system_prompt(language)
    started_at = time.time()
    loggers: dict[int, StructuredLogger] = {}

    def resolve_run_output_dir(run_id: int) -> Path:
        run_output_dir = _resolve_run_output_dir(
            output_dir,
            run_id,
            output_dir_by_run,
            runs,
        )
        run_output_dir.mkdir(parents=True, exist_ok=True)
        (run_output_dir / "trajectories").mkdir(exist_ok=True)
        (run_output_dir / "reports").mkdir(exist_ok=True)
        return run_output_dir

    def get_logger(run_id: int) -> StructuredLogger:
        if run_id not in loggers:
            loggers[run_id] = StructuredLogger(resolve_run_output_dir(run_id))
        return loggers[run_id]

    async def process_sample(sample: object, run_id: int) -> dict[str, Any]:
        sample = dict(sample)
        sample_id_raw = sample.get("id", "unknown")
        sample_id = (
            f"id_{sample_id_raw}"
            if str(sample_id_raw).isdigit()
            else str(sample_id_raw)
        )
        query = str(sample.get("query", ""))
        complexity = int(sample.get("meta_info", {}).get("days", 0) or 0) or None
        state = ConversationState(
            task_id=sample_id,
            domain="travel",
            complexity=complexity,
            system_config_name=system,
        )

        try:
            run_output_dir = resolve_run_output_dir(run_id)
            logger = get_logger(run_id)
            runner = TravelAgentRunner(
                model=system_config.executor_provider.alias,
                sample_id=str(sample_id_raw),
                database_base_path=str(database_dir),
                tool_schema_path=str(tool_schema_path),
                language=language,
            )
            result = await runner.run_task(
                user_query=query,
                system_prompt=system_prompt,
                state=state,
                system_config=system_config,
                logger=logger,
                run_id=run_id,
            )

            serialized_messages = serialize_messages(result.messages)
            payload = {
                "id": sample_id,
                "query": query,
                "model": model,
                "language": language,
                "final_plan": result.output,
                "messages": serialized_messages,
                "elapsed_time": state.wall_time_seconds,
                "success": True,
            }
            trajectory_file = run_output_dir / "trajectories" / f"{sample_id}.json"
            trajectory_file.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

            if result.output:
                plan_file = run_output_dir / "reports" / f"{sample_id}.txt"
                plan_file.write_text(result.output, encoding="utf-8")
            else:
                print(f"⚠️  Sample {sample_id}: No plan extracted")

            await logger.log_result(
                {
                    **result.state.to_metrics(),
                    "task_id": sample_id,
                    "run_id": run_id,
                    "system": system,
                    "domain": "travel",
                    "final_output": result.output,
                }
            )
            print(f"✅ Sample {sample_id} completed in {state.wall_time_seconds:.2f}s")
            return payload
        except Exception as exc:
            await logger.log_event(
                "task_error",
                {
                    "domain": "travel",
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
    language: str,
    test_data_path: Path,
    database_dir: Path,
    tool_schema_path: Path,
    output_dir: Path,
    workers: int = 10,
    max_llm_calls: int = 100,
    runs: int = 1,
    rerun_ids: list[int] | None = None,
    system: str = "A",
    output_dir_by_run: dict[int, Path] | None = None,
) -> dict[str, Any]:
    return asyncio.run(
        run_agent_inference_async(
            model=model,
            language=language,
            test_data_path=test_data_path,
            database_dir=database_dir,
            tool_schema_path=tool_schema_path,
            output_dir=output_dir,
            workers=workers,
            max_llm_calls=max_llm_calls,
            runs=runs,
            rerun_ids=rerun_ids,
            system=system,
            output_dir_by_run=output_dir_by_run,
        )
    )
