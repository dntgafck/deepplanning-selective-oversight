# AGENTS.md

- Use the root Pixi environment for repo commands, including DVC:
  `pixi run ...`.
- Prefer Pixi tasks over ad hoc invocation. Public benchmark entrypoint should
  be run as `pixi run deepplanning-experiment -- ...`.
- This repo is mostly a wrapper around the benchmark submodule. The code that
  actually runs DeepPlanning lives in
  `external/qwen-agent/benchmark/deepplanning/`; root currently only adds
  Pixi/DVC/data-bootstrap glue.
- `external/qwen-agent` is a git submodule. Treat submodule updates as
  deliberate; do not rewrite or replace vendored benchmark code unless the task
  explicitly requires it.
- Root wrapper CLIs use the canonical Hydra config tree under `configs/`. Prefer
  Hydra-style overrides such as `experiment=system_a_smoke`,
  `models.executor=qwen-plus`, or `travel.language=en`. The older direct
  benchmark/domain scripts are compatibility wrappers, not the documented public
  interface.

- Data bootstrap is wired through the root DVC stage `deepplanning_data` in
  `dvc.yaml`. It runs `scripts/download_deepplanning_data.py` and stores
  benchmark databases under `data/deepplanning/` so the submodule stays clean.
- The public benchmark entrypoint for this repo is `scripts/run_experiment.py`.
  Run it from the repo root with Pixi, e.g.
  `pixi run deepplanning-experiment -- experiment=system_a_smoke`.
- Keep `.env` at the repo root, not inside `external/qwen-agent/`. Root wrapper
  model definitions live in Hydra config under `configs/models.yaml` and are
  patched into vendored benchmark imports at runtime.

- Focused benchmark runs use the root wrappers so data and outputs stay outside
  the submodule:
- Shopping only:
  `pixi run deepplanning-experiment -- name=shopping-only domains=[shopping] models.executor=qwen-plus`
- Travel only:
  `pixi run deepplanning-experiment -- name=travel-only domains=[travel] models.executor=qwen-plus`
- Unified runner:
  `pixi run deepplanning-experiment -- name=travel-shopping domains=[travel,shopping] models.executor=qwen-plus`
- Travel control knobs map to Hydra overrides such as `travel.language`,
  `travel.start_from`, and `session_root`.

- Travel results now live under `outputs/deepplanning/travel/`; if you need a
  fresh rerun, clear or change that output directory first.
- Shopping input databases remain immutable under `data/deepplanning/shopping/`.
  Each run copies them into `outputs/deepplanning/shopping/database_infered/`
  before inference.

- `pixi run lint-all` runs root pre-commit hooks and will rewrite
  Markdown/TOML/YAML/JSON via Prettier plus EOF fixes. Expect doc-only
  formatting changes if you touch prose files.
- After making code or config changes, run `pixi run lint-all` from the repo
  root before considering the task complete. If hooks modify files, review the
  result and rerun until lint passes cleanly.
- `pixi run test` is currently not a useful repo smoke test on a fresh checkout:
  it runs bare `pytest` at repo root, recurses into `external/qwen-agent/tests`,
  and fails during collection because the Pixi env does not include submodule
  test/runtime deps such as `qwen_agent`, `openai`, and `json5`.

<!-- repo-task-proof-loop:start -->

## Repo task proof loop

For substantial features, refactors, and bug fixes, use the repo-task-proof-loop
workflow.

Required artifact path:

- Keep all task artifacts in `.agent/tasks/<TASK_ID>/` inside this repository.

Required sequence:

1. Freeze `.agent/tasks/<TASK_ID>/spec.md` before implementation.
2. Implement against explicit acceptance criteria (`AC1`, `AC2`, ...).
3. Create `evidence.md`, `evidence.json`, and raw artifacts.
4. Run a fresh verification pass against the current codebase and rerun checks.
5. If verification is not `PASS`, write `problems.md`, apply the smallest safe
   fix, and reverify.

Hard rules:

- Do not claim completion unless every acceptance criterion is `PASS`.
- Verifiers judge current code and current command results, not prior chat
  claims.
- Fixers should make the smallest defensible diff.
- For broad Codex tasks, bounded fan-out is allowed only after `init`, only when
  the user has explicitly asked for delegation or parallel agent work, and only
  when task shape warrants it: use bounded `explorer` children before or after
  spec freeze, use bounded `worker` children only after the spec is frozen, keep
  the task tree shallow, keep evidence ownership with one builder, and keep
  verdict ownership with one fresh verifier.
- This root `AGENTS.md` block is the repo-wide Codex baseline. More-specific
  nested `AGENTS.override.md` or `AGENTS.md` files still take precedence for
  their directory trees.
- Keep this block lean. If the workflow needs more Codex guidance, prefer nested
  `AGENTS.md` / `AGENTS.override.md` files or configured fallback guide docs
  instead of expanding this root block indefinitely.

Installed workflow agents:

- `.codex/agents/task-spec-freezer.toml`
- `.codex/agents/task-builder.toml`
- `.codex/agents/task-verifier.toml`
- `.codex/agents/task-fixer.toml`
<!-- repo-task-proof-loop:end -->
