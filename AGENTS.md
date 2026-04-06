# AGENTS.md

- Use the root Pixi environment for repo commands, including DVC:
  `pixi run ...`.
- Prefer Pixi tasks over ad hoc invocation. Benchmark entrypoint should be run
  as `pixi run deepplanning-benchmark -- ...`.
- This repo is mostly a wrapper around the benchmark submodule. The code that
  actually runs DeepPlanning lives in
  `external/qwen-agent/benchmark/deepplanning/`; root currently only adds
  Pixi/DVC/data-bootstrap glue.
- `external/qwen-agent` is a git submodule. Treat submodule updates as
  deliberate; do not rewrite or replace vendored benchmark code unless the task
  explicitly requires it.
- Root wrapper CLIs use Hydra config files under `conf/deepplanning/` plus Fire
  argument overrides. Prefer updating defaults there and use Fire-style flags
  such as `--models="qwen-plus"` or `--travel_language=en` when invoking wrapper
  scripts.

- Data bootstrap is wired through the root DVC stage `deepplanning_data` in
  `dvc.yaml`. It runs `scripts/download_deepplanning_data.py` and stores
  benchmark databases under `data/deepplanning/` so the submodule stays clean.
- The benchmark entrypoint for this repo is
  `scripts/run_deepplanning_benchmark.py`. Run it from the repo root with Pixi,
  e.g. `pixi run deepplanning-benchmark -- --domains="travel shopping"`.
- Keep `.env` at the repo root, not inside `external/qwen-agent/`. Root wrapper
  model definitions live in Hydra config under `conf/deepplanning/models.yaml`
  and are patched into vendored benchmark imports at runtime.

- Focused benchmark runs use the root wrappers so data and outputs stay outside
  the submodule:
- Shopping only:
  `pixi run deepplanning-benchmark -- --domains="shopping" --models="qwen-plus"`
- Travel only:
  `pixi run deepplanning-benchmark -- --domains="travel" --models="qwen-plus"`
- Unified runner:
  `pixi run deepplanning-benchmark -- --domains="travel shopping" --models="qwen-plus"`
- Travel control knobs map to wrapper args: `--language`, `--start_from`, and
  `--output_root`.

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
