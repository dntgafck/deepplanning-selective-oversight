# deepplanning-selective-oversight

Thesis experiments for the research question:

> **Can selective multi-LLM oversight help smaller agents match or exceed
> stronger single-agent baselines on DeepPlanning at lower or comparable total
> system cost?**

---

## What this is

This repo contains the implementation and experimental results for a master's
thesis investigating **selective oversight** as an architectural pattern for LLM
planning agents.

The core idea: instead of running a powerful (expensive) model as the sole
agent, or having it review every single step, a lightweight executor handles the
task autonomously and a stronger overseer is invoked only when specific risk
signals are detected. The hypothesis is that this selective architecture can
recover most of the performance gains of always-on oversight while spending a
fraction of the cost.

---

## Benchmark

All experiments run on
**[DeepPlanning](https://qwenlm.github.io/Qwen-Agent/en/benchmarks/deepplanning/)**
(Zhang et al., 2026), a long-horizon agentic planning benchmark with two
domains:

- **Travel Planning** — 120 tasks requiring multi-day itineraries with tight
  time, location, and budget constraints, accessed via 9 APIs
- **Shopping Planning** — 120 tasks requiring optimal cart construction under
  product, coupon, and shipping constraints, accessed via 15 APIs

Both domains use code-based automated evaluation with a single binary
correctness metric (Case Accuracy) and partial-credit continuous metrics
(Composite Score for Travel, Match Score for Shopping).

---

## System configurations

Six configurations are available in the active root-wrapper protocol. Current
repo defaults use `Qwen3.5-9B` as the executor, `deepseek-v4-flash` as the
overseer, and treat Shopping-first runs as the primary selective-oversight
evaluation surface:

| System     | Executor   | Overseer                         | Trigger policy                                                 |
| ---------- | ---------- | -------------------------------- | -------------------------------------------------------------- |
| A          | Qwen3.5-9B | —                                | None — executor-only baseline                                  |
| B          | Qwen3.5-9B | DeepSeek-V4-Flash (thinking)     | Every step — always-on ceiling                                 |
| C1         | Qwen3.5-9B | DeepSeek-V4-Flash (thinking)     | Checkpoints only                                               |
| **C2**     | Qwen3.5-9B | DeepSeek-V4-Flash (thinking)     | Adaptive filter — primary system                               |
| C2-noretry | Qwen3.5-9B | DeepSeek-V4-Flash (thinking)     | Adaptive filter, verifier runs once with no final repair retry |
| C2-nt      | Qwen3.5-9B | DeepSeek-V4-Flash (non-thinking) | Adaptive filter — reasoning ablation                           |

Strong monolithic baselines (GPT-5.2, Claude-4.5-Opus, DeepSeek-V3.2, Qwen3-Max)
are cited directly from the DeepPlanning paper and not re-run.

The overseer migration from DeepSeek-V3.2 to DeepSeek-V4-Flash landed in the
root-wrapper protocol on 2026-04-24. The repo now targets the direct DeepSeek
API for overseer calls, while the cited DeepPlanning paper baselines still
reference DeepSeek-V3.2.

The executor and overseer are intentionally from different model families
(Alibaba / DeepSeek) to ensure performance differences are attributable to the
oversight architecture rather than model-family scaling.

Implementation note: the wrapper now resolves each system through
`oversight_profile` into a concrete `OversightController` class. The historical
`oversight_mode` label remains only as a derived compatibility alias on
`SystemConfig`; it is no longer the documented config surface or the runtime
dispatch mechanism.

---

## What the adaptive filter does

The C2 trigger policy monitors the executor's trajectory in real time and
invokes the overseer only when one of five signals fires:

- a tool call returns an error
- a state-changing action (booking, cart modification) is about to be committed
- the same tool is called repeatedly with near-identical arguments (loop
  detection)
- halfway through the trajectory, too few task constraints have been queried
  (coverage deficit)
- the agent is about to submit its final plan

When triggered, the overseer applies a graduated correction — from a light
redirect up to directly re-querying a tool and injecting the authoritative
result — then execution resumes.

Loop and retry failure-mode semantics are documented in
[`docs/loop_scenarios.md`](docs/loop_scenarios.md).

`C2-noretry` is a methodological control for the Shopping headline comparison:
the final verifier still runs once, but it cannot grant the executor a final
do-over. This separates gains from runtime triggers from gains caused by final
verifier repair retries. It is distinct from the later `C2-final` ablation,
which removes the final checkpoint verification component entirely.

---

## Research questions

|     | Question                                                                                               |
| --- | ------------------------------------------------------------------------------------------------------ |
| RQ1 | Does selective oversight improve planning accuracy over an unassisted executor?                        |
| RQ2 | Is selective oversight more cost-efficient than always-on oversight at equal or better accuracy?       |
| RQ3 | Can the architecture compete with stronger monolithic single-agent baselines on a cost-adjusted basis? |
| RQ4 | Does oversight help more on harder tasks?                                                              |
| RQ5 | Which trigger components drive the performance gains?                                                  |

---

## Models and infrastructure

Configured wrapper models use provider-specific OpenAI-compatible endpoints:
Together.ai for `qwen3.5-9b` and the DeepSeek direct API for
`deepseek-v4-flash`.

The `qwen-plus` alias remains in `configs/models.yaml` only as a compatibility
shim for vendored travel conversion code that still requests that name. In the
root wrapper, that alias is routed to the DeepSeek API rather than DashScope.

### DeepSeek overseer configuration

The active wrapper config does **not** use a single `overseer:` block. The
transport config lives in `configs/models.yaml`, the default model selection
lives in `configs/experiment.yaml`, and thinking vs. non-thinking mode is chosen
per system in `configs/system/*.yaml`.

Actual wrapper config shape:

```yaml
# configs/experiment.yaml
models:
  executor: qwen3.5-9b
  overseer: deepseek-v4-flash

# configs/system/C2.yaml
overseer_thinking: true

# configs/system/C2-nt.yaml
overseer_thinking: false

# configs/models.yaml
deepseek-v4-flash:
  model_name: deepseek-v4-flash
  base_url: https://api.deepseek.com
  api_key_env: DEEPSEEK_API_KEY
  request_params:
    reasoning_effort: high
  extra_body:
    thinking:
      type: enabled
```

In other words, `C2-nt` does not switch to a separate overseer model block. It
uses the same `models.overseer=deepseek-v4-flash` alias and disables thinking at
call time via `overseer_thinking: false`.

DeepSeek pricing in `configs/models.yaml` uses cached-input accounting
(`cached_input_output_v1`) with:

- `$0.028 / M` input tokens on cache hit
- `$0.14 / M` input tokens on cache miss
- `$0.28 / M` output tokens

When the provider returns cache-hit and cache-miss token counts, the wrapper
preserves and logs both values.

### Sampling configuration

All configured models are pinned to `temperature=0.0` and `top_p=1.0`. The base
`seed` is set per model in `configs/models.yaml` and the runtime wrappers offset
it by `run_id`, so internal runs `0-3` map to seeds `42-45`.

Seed is best-effort across providers. Even with a fixed seed and
`temperature=0.0`, batching, kernel non-determinism, and provider routing can
still change outputs. The four-run protocol absorbs that residual variance.

Provider seed support status checked against provider docs on April 22, 2026:

- Together.ai: the chat-completions docs list `seed` as a supported request
  parameter.
- DeepSeek direct API: the current chat-completions docs do not document `seed`,
  so this repo sends it as a best-effort top-level parameter and logs the
  requested value for every run.

---

## Benchmark setup

This repo keeps the vendored benchmark submodule read-only during normal use.

- Benchmark data is materialized by DVC under `data/deepplanning/`
- Runtime artifacts are written under `outputs/deepplanning/`
- `.env` lives at the repo root
- Executor credentials are read from `TOGETHER_API_KEY`; overseer credentials
  are read from `DEEPSEEK_API_KEY`
- The wrapper config source of truth lives under `configs/`
- Model transport aliases are owned by the wrapper layer in
  `configs/models.yaml`

Public benchmark runner:

```bash
pixi run dvc repro deepplanning_data
pixi run deepplanning-experiment -- experiment=system_a_smoke
```

### Langfuse session usage

With `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in the root `.env` or
environment, summarize token usage for a Langfuse session by model:

```bash
pixi run langfuse-session-usage -- <session-id>
pixi run langfuse-session-usage -- <session-id> --output outputs/langfuse/session-usage.csv
```

The command defaults to `https://cloud.langfuse.com`; set `LANGFUSE_HOST` to use
another Langfuse host.

To export one CSV that joins timestamp-session Langfuse token usage with local
aggregate metrics for benchmark result sessions:

```bash
pixi run langfuse-results-usage
```

This writes `outputs/deepplanning/langfuse-results-usage.csv` by default. It
scans `shopping-*` and `system-*` result directories, skips `shopping-b` and
metadata `system=B`, and uses each timestamp directory name as the Langfuse
session id.

Shopping is the primary and default v1 oversight domain. The frozen Shopping
split is selected with `shopping.split=tune|test|all`: calibration and tuning
use `shopping.split=tune`, held-out reporting uses `shopping.split=test`, and
`shopping.split=all` remains available for compatibility and smoke/debug runs.
Do not treat the same 120 Shopping tasks as both the tuning and headline
reporting surface.

One-shot model smoke test:

```bash
pixi run model-chat -- qwen3.5-9b
pixi run model-chat -- qwen3.5-9b "Explain recursion in one sentence."
```

The first argument to `model-chat` must match a model alias declared in
`configs/models.yaml`. The command sends a single chat completion request using
that provider config and prints the full response JSON returned by the client.
If no prompt is passed, a built-in smoke prompt is used.

Override examples:

```bash
pixi run deepplanning-experiment -- experiment=system_a_smoke name=my-smoke
pixi run deepplanning-experiment -- name=shop-tune domains=[shopping] shopping.split=tune system=C2
pixi run deepplanning-experiment -- name=shop-heldout domains=[shopping] shopping.split=test system=C2
pixi run deepplanning-experiment -- name=shop-ablation domains=[shopping] shopping.levels=[1,2] models.executor=qwen3.5-9b
```

Travel support remains in the wrapper for optional extension work, but Travel is
not part of the default v1 headline path and does not currently have active
Travel-specific oversight calibration.

Each experiment session writes a timestamped directory under
`outputs/deepplanning/experiments/<name>/<timestamp>/` containing:

- `config.yaml`
- `overrides.txt`
- `experiment_session.json`
- domain outputs under `travel/` and `shopping/`

The documented interface is `scripts/run_experiment.py` via
`pixi run deepplanning-experiment`.
