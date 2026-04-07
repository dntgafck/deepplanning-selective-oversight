# DeepPlanning Agent Runner Design

Date: 2026-04-07
Status: Approved for implementation planning

## Summary

Design a new root-owned DeepPlanning runner that replaces only the benchmark's agent execution loop while keeping the benchmark's prompts, tool schemas, tool implementations, task data, and evaluation code as the source of truth.

Phase 1 is executor-only. The new runner must support both travel and shopping, preserve each domain's current loop behavior closely enough for evaluator-compatible outputs, and expose a no-op interception seam for future oversight work without introducing a full oversight config system yet.

## Project Context

The current repo already wraps the vendored benchmark with Hydra config under `conf/deepplanning/` and Fire CLIs under `scripts/`.

The vendored DeepPlanning benchmark has materially different runtime behavior across domains.

- Travel builds `system` plus `user` messages, repeatedly calls the LLM with tools, executes tool calls, and extracts the final answer by parsing `<plan>...</plan>` from the terminal assistant response.
- Shopping builds `system` plus `user` messages, runs a tool-using loop, then appends a second hardcoded cart-check user prompt and runs a second loop before evaluation reads `messages.json` and cart state from each case directory.

Because these loops differ, the new runtime should not force both domains into one fake uniform domain model. It should share only the orchestration that is truly common.

## Goals

- Reimplement the agent execution loop in root-owned code.
- Preserve benchmark-owned prompts, tool schemas, task inputs, tool behavior, and evaluation behavior.
- Support both travel and shopping in the first implementation milestone.
- Introduce a shared LLM calling layer with retry handling and structured runtime logging.
- Keep outputs evaluator-compatible for both domains.
- Center the workflow around a new experiment-first CLI.
- Add a no-op interception seam for future oversight logic.

## Non-Goals

- Do not implement real oversight logic in this phase.
- Do not add a `systems/` config family yet.
- Do not create a separate `domains/` config family yet.
- Do not modify vendored benchmark evaluation logic to fit a new artifact format.
- Do not attempt byte-for-byte parity of trajectories; parity is judged by evaluator compatibility and comparable benchmark behavior.

## Architecture

The new runtime is split into four root-owned packages.

- `agent/`
  - shared runner types and minimal common execution skeleton
  - `travel.py` for the travel runner
  - `shopping.py` for the shopping runner
- `llm/`
  - LiteLLM-backed executor client
  - model config resolution
  - retry handling
  - usage and request metadata capture
- `oversight/`
  - no-op future seam only
  - passthrough interception API that returns the original response unchanged in phase 1
- `experiment/`
  - async experiment orchestration
  - concurrency limits
  - task scheduling
  - JSONL event and result logging
  - experiment-first CLI entrypoint

Ownership boundary:

- Root-owned code controls orchestration, LLM calls, retries, logging, and the future interception seam.
- Vendored benchmark code remains the source of truth for prompts, tool definitions, tool implementations, task data, and evaluation.

This avoids subclassing or wrapping the vendored agent classes directly. Those classes are useful references, but they are not treated as extension points.

## Config Design

Keep config minimal for phase 1.

```text
conf/deepplanning/
  models/
    qwen3_14b_dashscope.yaml
  experiments/
    shopping.yaml
    travel.yaml
    benchmark.yaml
  experiment.yaml
```

Config responsibilities:

- `models/`
  - one file per callable model endpoint
  - provider and model stay coupled here
  - includes only LLM runtime details such as LiteLLM model string, `api_base`, `api_key_env`, retry policy, and request defaults
- `experiments/`
  - the glue layer for actual runs
  - includes domain, runner selection, task/data inputs, output locations, `max_steps`, `executor_model`, `parallel`, `runs`, and `sample_ids`

Phase 1 keeps oversight disabled by construction. There is no separate system config surface yet. When real oversight is implemented later, a dedicated config family can be introduced if it is still justified.

## Runner Design

### Shared Execution Skeleton

The shared runner logic is intentionally small.

Responsibilities:

- construct task execution context from experiment config
- call the executor through `llm/`
- record retries, elapsed time, usage metadata, and runtime events
- invoke a no-op interception hook after the executor response and before tool execution
- return a normalized `TaskResult` plus domain-specific artifact paths

The shared layer must not own domain semantics such as final-answer extraction, prompt choice, or special shopping second-phase behavior.

### Travel Runner

`agent.travel` reproduces the benchmark travel loop shape.

Responsibilities:

- load tasks from the existing travel query JSON
- use benchmark `travelplanning.agent.prompts.get_system_prompt(language)` unchanged
- load the benchmark travel tool schema JSON unchanged
- execute the loop as:
  - `system` message if configured
  - `user` query
  - repeated executor call with full history and tools
  - assistant message appended to history
  - tool call detection and execution
  - tool results appended to history
  - stop on no tool calls or max steps
- extract the final plan exactly the same way the current benchmark runner does by parsing `<plan>...</plan>` from the terminal assistant response
- emit evaluator-compatible travel artifacts:
  - `trajectories/id_X.json`
  - `reports/id_X.txt`

Conversion and evaluation remain unchanged and continue to use the vendored benchmark code after inference completes.

### Shopping Runner

`agent.shopping` reproduces the benchmark shopping loop shape.

Responsibilities:

- load tasks from the existing shopping query metadata JSON
- use the benchmark shopping prompt for the requested level unchanged
- load the benchmark shopping tool schema JSON unchanged
- execute the existing two-phase interaction pattern:
  - phase 1 tool-using loop
  - append the same hardcoded cart-check user prompt the benchmark currently injects
  - phase 2 tool-using finalization loop
- execute tools against copied case directories so the benchmark shopping tools continue mutating cart files in the expected place
- emit evaluator-compatible shopping artifacts:
  - `case_X/messages.json` inside the copied run database directory
  - cart state produced by the existing tool implementations

The runner must write evaluator-compatible artifacts directly during execution instead of inventing a new abstract trace that later has to be translated back into benchmark format.

## Benchmark Reuse Points

The following benchmark touchpoints should be reused directly rather than reimplemented.

- Travel prompts: `travelplanning.agent.prompts.get_system_prompt`
- Travel conversion: `travelplanning.evaluation.convert_report.convert_reports`
- Travel evaluation: `travelplanning.evaluation.eval_converted.evaluate_plans`
- Shopping prompts: `shoppingplanning.agent.prompts.prompt_lib.SYSTEM_PROMPT_level{level}`
- Shopping evaluation: `shoppingplanning.evaluation.evaluation_pipeline`
- Shopping statistics: `shoppingplanning.evaluation.score_statistics`
- Travel tool schema JSON files under `travelplanning/tools/`
- Shopping tool schema JSON under `shoppingplanning/tools/`
- Travel tool implementations loaded from benchmark travel tool modules via `BaseTravelTool` subclasses
- Shopping tool implementations loaded from benchmark shopping tool modules via `base_shopping_tool.TOOL_REGISTRY`

The vendored agent loop implementations are reference material, not reuse targets. The new code should reproduce their behavior where necessary without depending on those classes as the runtime core.

## LLM Layer

The `llm/` package provides one executor-facing interface around LiteLLM.

Responsibilities:

- load model configuration from `conf/deepplanning/models/`
- issue chat-completion requests with message history and tools
- apply exponential backoff retries
- normalize response access for the runners
- capture request and response metadata needed for structured logs

Phase 1 only needs executor support. Overseer-specific behavior is deferred.

## Future Oversight Seam

Phase 1 includes a minimal seam only.

- After an executor response is received and before any tool call is executed, the runner calls a passthrough interception function.
- In phase 1, the interception function always returns the original response unchanged.
- There is no oversight config, no intervention policy, and no secondary model call in this phase.

This seam exists only to avoid rewriting the runner again when oversight is introduced later.

## Failure Handling

Keep failure handling operational and minimal.

- LLM calls retry with exponential backoff.
- If retries are exhausted, the current task is marked failed and the rest of the experiment continues.
- Tool execution preserves benchmark behavior. If benchmark tools surface errors as tool outputs, the new runner preserves that behavior instead of inventing a new error protocol.
- Max-step exhaustion is treated as a terminal task outcome, not a process crash.
- Unexpected runner exceptions are captured per task with enough context to debug, including task id, domain, sample id, phase, and last completed step.

## Observability

Observability is local-first and structured.

Write two JSONL streams.

- task summary stream
  - one record per completed or failed task
- event stream
  - one record per notable runtime event such as LLM request started, LLM response received, retry occurred, tool executed, task finished, or task failed

Every event should include stable identifiers:

- experiment id
- run id
- task id
- domain
- step number when applicable

Langfuse support is optional and best-effort.

- enable it only if the required environment variables are present
- failure to initialize or send traces must not block local runs

Benchmark-compatible artifacts remain separate from observability artifacts.

## CLI Direction

Phase 1 centers on a new experiment-first CLI. Existing wrapper scripts may be kept temporarily as thin delegates if useful, but they are not the primary interface for the new system.

The new CLI should select named experiment configs and allow targeted overrides such as subset ids, parallelism, or output roots.

## Validation And Testing

Validation is parity-first.

Pre-validation gate:

- Before running tests, subset runs, or benchmark comparisons, stop and ask the user to configure:
  - model entries under `conf/deepplanning/models/`
  - required API keys in `.env` or environment variables
- Do not start validation until the user confirms configuration is ready.

Required validation for phase 1:

- run both domains with the new runner in executor-only mode
- use small fixed subsets first for quick comparison
- compare results based on evaluator compatibility and comparable metrics, not byte-for-byte trace identity
- reuse existing benchmark evaluation code unchanged

Concrete validation flow:

- Shopping
  - run a small subset with the current path
  - run the same subset with the new runner
  - verify evaluation completes for both
  - compare summary metrics and inspect a sample of `messages.json` and cart outcomes
- Travel
  - run a small subset with the current path
  - run the same subset with the new runner
  - verify conversion and evaluation complete for both
  - compare evaluation summaries and inspect a sample of generated report files

Repo test coverage should stay focused on root-owned logic.

- unit tests for LLM request normalization and retry behavior
- unit tests for event and result serialization
- unit tests for small helper functions introduced in the new runner packages
- no attempt to unit test vendored benchmark tools or evaluation code

Phase 1 success criteria:

- the new CLI runs shopping and travel end-to-end
- both domains emit evaluator-compatible artifacts
- existing evaluation stages run unchanged on those artifacts
- executor-only behavior is comparable enough on fixed subsets to proceed
- the no-op interception seam exists and does nothing

## Deferred Work

The following items are intentionally deferred until after the executor-only runner lands.

- real oversight policies and interventions
- overseer model support
- oversight-specific configuration families
- richer per-conversation research metrics beyond the structured logs required for phase 1
