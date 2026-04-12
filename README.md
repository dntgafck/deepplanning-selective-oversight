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

Five configurations are compared:

| System | Executor  | Overseer                    | Trigger policy                       |
| ------ | --------- | --------------------------- | ------------------------------------ |
| A      | Qwen3-14B | —                           | None — executor-only baseline        |
| B      | Qwen3-14B | DeepSeek-V3.2 (thinking)    | Every step — always-on ceiling       |
| C1     | Qwen3-14B | DeepSeek-V3.2 (thinking)    | Checkpoints only                     |
| **C2** | Qwen3-14B | DeepSeek-V3.2 (thinking)    | Adaptive filter — primary system     |
| C2-nt  | Qwen3-14B | DeepSeek-V3.2 (no thinking) | Adaptive filter — reasoning ablation |

Strong monolithic baselines (GPT-5.2, Claude-4.5-Opus, DeepSeek-V3.2, Qwen3-Max)
are cited directly from the DeepPlanning paper and not re-run.

The executor and overseer are intentionally from different model families
(Alibaba / DeepSeek) to ensure performance differences are attributable to the
oversight architecture rather than model-family scaling.

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

All models are accessed via [OpenRouter](https://openrouter.ai). Each task is
run four times for robustness (seeds 42–45, temperature 0). Total experimental
runs: ~4 800 across all systems.

---

## Benchmark setup

This repo keeps the vendored benchmark submodule read-only during normal use.

- Benchmark data is materialized by DVC under `data/deepplanning/`
- Runtime artifacts are written under `outputs/deepplanning/`
- `.env` lives at the repo root
- The wrapper config source of truth lives under `configs/`
- Model transport aliases are owned by the wrapper layer in
  `configs/models.yaml`

Public benchmark runner:

```bash
pixi run dvc repro deepplanning_data
pixi run deepplanning-experiment -- experiment=system_a_smoke
```

Override examples:

```bash
pixi run deepplanning-experiment -- experiment=system_a_smoke name=my-smoke
pixi run deepplanning-experiment -- name=travel-c2 domains=[travel] system=C2 travel.language=en
pixi run deepplanning-experiment -- name=shop-ablation domains=[shopping] shopping.levels=[1,2] models.executor=qwen-plus
```

Each experiment session writes a timestamped directory under
`outputs/deepplanning/experiments/<name>/<timestamp>/` containing:

- `config.yaml`
- `overrides.txt`
- `experiment_session.json`
- domain outputs under `travel/` and `shopping/`

The documented interface is `scripts/run_experiment.py` via
`pixi run deepplanning-experiment`.
