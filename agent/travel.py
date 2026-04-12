from __future__ import annotations

import asyncio
import json
import time
import traceback
from pathlib import Path
from typing import Any

from experiment import StructuredLogger, build_system_config, run_experiment
from experiment.logging import serialize_exception, serialize_messages
from experiment.progress import InferenceProgressReporter
from failure_subtypes import (
    FAILURE_SUBTYPE_INFRA_TRANSIENT,
    classify_exception_failure_subtype,
    failure_subtype_from_stop_reason,
    observation_valid_for_failure_subtype,
)
from llm import (
    build_langfuse_trace_id,
    call_chat_completion,
    extract_usage_tokens,
    flush_langfuse,
)
from oversight import ConversationState, apply_intervention, evaluate_oversight

from .base import TaskResult
from .vendor import (
    clear_vendored_tool_module_cache,
    load_travel_agent_class,
    load_travel_prompt,
)

VendorTravelFnAgent = load_travel_agent_class()
DEFAULT_INFRA_RETRY_LIMIT = 2


def get_system_prompt(language: str) -> str:
    return load_travel_prompt(language)


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


def _error_summary(error_payload: dict[str, Any]) -> str:
    error_type = error_payload.get("type", "Error")
    message = str(error_payload.get("message", "")).strip()
    return f"{error_type}: {message}" if message else str(error_type)


class TravelAgentRunner(VendorTravelFnAgent):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        clear_vendored_tool_module_cache()
        super().__init__(*args, **kwargs)
        self._install_tool_aliases()

    def _install_tool_aliases(self) -> None:
        canonical_name = "recommend_restaurants"
        alias_name = "recommend_around_restaurants"

        canonical_tool = self.tool_instances.get(canonical_name)
        if canonical_tool is not None and alias_name not in self.tool_instances:
            self.tool_instances[alias_name] = canonical_tool

        if any(
            tool.get("function", {}).get("name") == alias_name
            for tool in self.openai_tools
        ):
            return

        for tool in self.openai_tools:
            function = tool.get("function", {})
            if function.get("name") != canonical_name:
                continue
            aliased_function = dict(function)
            aliased_function["name"] = alias_name
            self.openai_tools.append({"type": "function", "function": aliased_function})
            break

    async def run_task(
        self,
        user_query: str,
        system_prompt: str | None,
        state: ConversationState,
        system_config: Any,
        logger: StructuredLogger | None = None,
        run_id: int = 0,
        trace_id: str | None = None,
        session_id: str | None = None,
    ) -> TaskResult:
        state.begin()

        messages: list[Any] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_query})

        llm_budget = system_config.max_steps
        while llm_budget > 0:
            llm_budget -= 1
            step_count = system_config.max_steps - llm_budget
            request_messages = list(messages)
            started_at = time.time()

            async def log_attempt_error(
                payload: dict[str, Any],
                *,
                step: int = step_count,
                task_id: str = state.task_id,
                model_alias: str = system_config.executor_provider.alias,
            ) -> None:
                error_payload = payload.get("error", {})
                retry_note = (
                    "retrying" if payload.get("will_retry") else "no retries left"
                )
                print(
                    "⚠️  Travel LLM call failed "
                    f"(sample={task_id}, run={run_id}, step={step}, "
                    f"attempt={payload.get('attempt')}/{payload.get('max_attempts')}, "
                    f"{retry_note}): {_error_summary(error_payload)}"
                )
                if logger is None:
                    return
                await logger.log_event(
                    "executor_llm_error",
                    {
                        "domain": "travel",
                        "task_id": task_id,
                        "run_id": run_id,
                        "phase": "travel",
                        "step": step,
                        "model_alias": model_alias,
                        **payload,
                    },
                )

            response = await call_chat_completion(
                provider=system_config.executor_provider,
                messages=request_messages,
                tools=self.openai_tools,
                on_attempt_error=log_attempt_error,
                trace_id=trace_id,
                session_id=session_id,
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
    infra_retry_limit: int = DEFAULT_INFRA_RETRY_LIMIT,
    rerun_ids: list[int] | None = None,
    system: str = "A",
    output_dir_by_run: dict[int, Path] | None = None,
    trace_id: str | None = None,
    session_id: str | None = None,
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
    print(f"Runs: {runs}")
    print(f"Workers: {workers}")
    print(f"Infra retry limit: {infra_retry_limit}")
    progress = InferenceProgressReporter(
        domain="travel",
        samples_per_run=len(test_data),
        runs=runs,
    )
    print(f"Execution mode: {progress.execution_mode(workers)}")
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
        logger = get_logger(run_id)
        sample_trace_id = trace_id or build_langfuse_trace_id(
            session_id,
            "travel",
            model,
            run_id,
            sample_id_raw,
        )

        max_attempts = max(int(infra_retry_limit), 0) + 1
        for attempt_number in range(1, max_attempts + 1):
            state = ConversationState(
                task_id=sample_id,
                domain="travel",
                complexity=complexity,
                system_config_name=system,
            )

            try:
                run_output_dir = resolve_run_output_dir(run_id)
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
                    trace_id=sample_trace_id,
                    session_id=session_id,
                )

                serialized_messages = serialize_messages(result.messages)
                failure_subtype = failure_subtype_from_stop_reason(
                    result.state.final_stop_reason
                )
                observation_valid = observation_valid_for_failure_subtype(
                    failure_subtype
                )
                payload = {
                    "id": sample_id,
                    "query": query,
                    "model": model,
                    "language": language,
                    "final_plan": result.output,
                    "messages": serialized_messages,
                    "elapsed_time": state.wall_time_seconds,
                    "success": True,
                    "failure_subtype": failure_subtype,
                    "observation_valid": observation_valid,
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
                        "success": True,
                        "failure_subtype": failure_subtype,
                        "observation_valid": observation_valid,
                        "final_output": result.output,
                    }
                )
                await progress.record_completion(
                    run_id=run_id,
                    sample_id=sample_id,
                    success=True,
                    elapsed_seconds=state.wall_time_seconds,
                )
                return payload
            except Exception as exc:
                if not state.end_time:
                    state.finish()
                error_payload = serialize_exception(exc)
                failure_subtype = classify_exception_failure_subtype(exc)
                observation_valid = observation_valid_for_failure_subtype(
                    failure_subtype
                )
                should_retry = (
                    failure_subtype == FAILURE_SUBTYPE_INFRA_TRANSIENT
                    and attempt_number < max_attempts
                )
                if should_retry:
                    print(
                        "⚠️  Travel sample invalid due to transient infrastructure error "
                        f"(sample={sample_id}, run={run_id}, attempt={attempt_number}/{max_attempts}); "
                        "retrying same seed/config."
                    )
                    await logger.log_event(
                        "task_invalid_attempt",
                        {
                            "domain": "travel",
                            "task_id": sample_id,
                            "run_id": run_id,
                            "attempt": attempt_number,
                            "max_attempts": max_attempts,
                            "failure_subtype": failure_subtype,
                            "observation_valid": False,
                            "error": error_payload,
                        },
                    )
                    continue

                await logger.log_event(
                    "task_error",
                    {
                        "domain": "travel",
                        "task_id": sample_id,
                        "run_id": run_id,
                        "failure_subtype": failure_subtype,
                        "observation_valid": observation_valid,
                        "error": error_payload,
                    },
                )
                await logger.log_result(
                    {
                        **state.to_metrics(),
                        "task_id": sample_id,
                        "run_id": run_id,
                        "system": system,
                        "domain": "travel",
                        "success": False,
                        "failure_subtype": failure_subtype,
                        "observation_valid": observation_valid,
                        "final_output": None,
                        "error": error_payload,
                    }
                )
                await progress.record_completion(
                    run_id=run_id,
                    sample_id=sample_id,
                    success=False,
                    error_summary=_error_summary(error_payload),
                )
                traceback.print_exc()
                return {
                    "id": sample_id,
                    "query": query,
                    "success": False,
                    "failure_subtype": failure_subtype,
                    "observation_valid": observation_valid,
                    "error": error_payload.get("message", str(exc)),
                }

    raw_results = await run_experiment(
        test_data, process_sample, parallel=workers, runs=runs
    )
    results: list[dict[str, Any]] = []
    for item in raw_results:
        if isinstance(item, Exception):
            results.append(
                {
                    "success": False,
                    "failure_subtype": "none",
                    "observation_valid": True,
                    "error": str(item),
                }
            )
        else:
            results.append(item)

    observed_results = [
        result for result in results if result.get("observation_valid", True)
    ]
    success_count = sum(1 for result in observed_results if result.get("success"))
    return {
        "total": len(observed_results),
        "success": success_count,
        "failed": len(observed_results) - success_count,
        "invalid": len(results) - len(observed_results),
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
    infra_retry_limit: int = DEFAULT_INFRA_RETRY_LIMIT,
    rerun_ids: list[int] | None = None,
    system: str = "A",
    output_dir_by_run: dict[int, Path] | None = None,
    trace_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    async def _run_with_cleanup() -> dict[str, Any]:
        try:
            return await run_agent_inference_async(
                model=model,
                language=language,
                test_data_path=test_data_path,
                database_dir=database_dir,
                tool_schema_path=tool_schema_path,
                output_dir=output_dir,
                workers=workers,
                max_llm_calls=max_llm_calls,
                runs=runs,
                infra_retry_limit=infra_retry_limit,
                rerun_ids=rerun_ids,
                system=system,
                output_dir_by_run=output_dir_by_run,
                trace_id=trace_id,
                session_id=session_id,
            )
        finally:
            flush_langfuse()

    return asyncio.run(_run_with_cleanup())
