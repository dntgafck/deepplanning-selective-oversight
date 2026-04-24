# Loop Scenarios

This document summarizes the shopping-wrapper loop and retry scenarios that the
oversight layer must handle, plus the configuration fields that cap or shape
those behaviors.

Relevant implementation points:
- S1 loop detection is triggered in `oversight/triggers.py`.
- S2 and S3 pre-tool veto handling is finalized in `compute_h1_outcome(...)`
  and enforced in `agent/shopping.py`.
- S4 final-verifier retries are handled in the Shopping runner final-checkpoint
  path.
- S5 overseer-call ceilings are configured through `SystemConfig` and the
  system YAMLs.

## Scenario Catalog

| Scenario | Description | Primary guard |
| --- | --- | --- |
| S1: Executor tool-call loop | The executor repeats the same tool pattern after successful executions. | `detect_loop(...)` |
| S2: Exact-repeat veto loop | The overseer blocks the same `(tool, args)` repeatedly. | `max_hard_blocks_per_args` |
| S3: Distinct-args veto loop | The overseer blocks a series of different `(tool, args)` proposals and no tool executes. | `max_consecutive_pre_tool_blocks` |
| S4: Final-verifier repair loop | Final verification keeps rejecting executor drafts. | `final_repair_retry_cap` |
| S5: Runaway overseer invocations | Oversight is triggered too often for one task. | `overseer_call_budget_per_task` |

## Per-System Behavior

| System | S1 | S2 | S3 | S4 | S5 |
| --- | --- | --- | --- | --- | --- |
| A | No overseer; fallback is `max_steps`. | N/A | N/A | N/A | N/A |
| B | Always-on pre/post/final oversight can still nudge on repeated patterns. | Exact-repeat cap still applies. | Distinct-args hard-block streak is force-approved after `max_consecutive_pre_tool_blocks`. | Final-verifier retry cap still applies. | High-ceiling always-on budget from config. |
| C1 | No pre-tool oversight, so executor-only loop handling is still `max_steps`. | N/A | N/A | Final-verifier retry cap still applies. | Bounded by configured `overseer_call_budget_per_task`. |
| C2 | Adaptive loop detection can nudge repeated executed-tool patterns. | Exact-repeat cap still applies. | Distinct-args hard-block streak is force-approved after `max_consecutive_pre_tool_blocks`. | Final-verifier retry cap still applies. | Bounded by configured `overseer_call_budget_per_task`. |
| C2-nt | Same as C2. | Same as C2. | Same as C2. | Same as C2. | Same as C2. |

## Safety-Cap Fields

| Field | Default | Location | Scenario |
| --- | --- | --- | --- |
| `max_hard_blocks_per_args` | `2` | `experiment/config.py`, system YAMLs | S2 |
| `max_consecutive_pre_tool_blocks` | `5` | `experiment/config.py`, `configs/system/B.yaml` | S3 |
| `final_repair_retry_cap` | `2` | `experiment/config.py`, system YAMLs | S4 |
| `overseer_call_budget_per_task` | `8` default, with per-system YAML overrides | `experiment/config.py`, `configs/system/*.yaml` | S5 |
| `loop_similarity_threshold` | `0.92` | `experiment/config.py` | S1 tuning |
| `loop_window` | `5` | `experiment/config.py` | S1 tuning |
| `loop_repeat_count` | `3` | `experiment/config.py` | S1 tuning |
| `max_steps` | `400` | `experiment/config.py` | Last-resort global cap |

## Scenario Notes

S1:
Executed-tool loop detection only sees tool calls that actually ran. It cannot
break a veto loop where nothing executes.

S2:
The exact-repeat cap is keyed by normalized `(tool, args)` identity. It protects
against the same blocked mutation being retried over and over.

S3:
The consecutive-block cap is global across pre-tool hard blocks. Once the
executor has been blocked for the configured streak length, the next blocked
proposal is force-approved and logged as `overseer_override_streak_cap`.

S4:
Final-verifier retries are separate from pre-tool blocking. Exhausting the
repair cap ends the task instead of forcing approval.

S5:
Oversight budgets are orthogonal to the streak cap. The budget limits how often
the overseer can run; the streak cap limits how long repeated hard blocks can
prevent tool execution once oversight is already firing.
