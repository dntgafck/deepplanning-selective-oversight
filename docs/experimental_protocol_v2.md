# Experimental Protocol: Selective Oversight for LLM Planning Agents on DeepPlanning

**Document type:** Thesis methodology chapter — experimental protocol
**Version:** 2.0 **Date:** April 2026

---

## 1. Experimental Matrix

### 1.1 System Configurations

The experiment evaluates five system configurations plus cited baselines,
forming a controlled comparison structure where each system isolates a single
architectural variable.

| System | Executor                  | Overseer                    | Trigger Policy  | Purpose                     |
| ------ | ------------------------- | --------------------------- | --------------- | --------------------------- |
| A      | Qwen3.5-9B (non-thinking) | None                        | —               | Executor-only baseline      |
| B      | Qwen3.5-9B (non-thinking) | DeepSeek-V3.2 (thinking)    | Every step      | Always-on oversight ceiling |
| C1     | Qwen3.5-9B (non-thinking) | DeepSeek-V3.2 (thinking)    | Checkpoint-only | Minimal selective oversight |
| C2     | Qwen3.5-9B (non-thinking) | DeepSeek-V3.2 (thinking)    | Adaptive filter | Primary selective oversight |
| C2-nt  | Qwen3.5-9B (non-thinking) | DeepSeek-V3.2 (no thinking) | Adaptive filter | Overseer reasoning ablation |
| D      | (Cited from paper)        | —                           | —               | Strong monolithic baselines |

**System D baselines** are drawn directly from the DeepPlanning paper (Zhang et
al., 2026) and are not re-run. For Shopping Planning specifically: DeepSeek-V3.2
with thinking (42.5% Case Accuracy, 78.8% Match Score), GPT-5.2-high (54.2% Case
Accuracy, 84.8% Match Score), Claude-4.5-Opus with thinking (45.0% Case
Accuracy, 80.0% Match Score), and Qwen3-Max with thinking (43.5% Case Accuracy,
82.6% Match Score).

**Design rationale for cross-family pairing.** The executor (Qwen3.5-9B, Alibaba
Qwen family) and overseer (DeepSeek-V3.2, DeepSeek-AI) belong to different model
families. This guards against the critique that the overseer is merely a larger
version of the same model. It also ensures that the overseer brings genuinely
different capabilities and failure modes, making any performance gains
attributable to the oversight architecture rather than model-family scaling.

**Executor choice rationale.** Qwen3.5-9B is a 9B-parameter dense model run in
non-thinking mode. It is not present in the DeepPlanning paper, so System A must
be run to establish an executor baseline. Its small size and low per-token cost
make it a realistic candidate for the "small agent" role in the oversight
architecture. The model supports function calling and the OpenAI tool schema
format required by the benchmark.

### 1.2 Domain Scope

**Primary domain: Shopping Planning (120 tasks).** All system configurations are
evaluated on the full Shopping Planning benchmark. Shopping is the primary
domain for several reasons: baseline Case Accuracy rates are meaningfully above
zero across published models (e.g., DeepSeek-V3.2 non-thinking: 10.6%), making
it the domain where oversight-induced improvements in binary task success can be
measured; the evaluation is fully automated with no LLM-based parsing step,
eliminating a potential confound; and task complexity is structured into three
levels (Level 1–3), enabling stratified analysis.

**Optional extension: Travel Planning (120 tasks).** Travel tasks are included
only if timeline and budget permit after all Shopping experiments are complete.
Travel is deprioritized because even frontier non-reasoning models score 0.0%
Case Accuracy on Travel, making binary success metrics uninformative for a small
executor. If Travel is run, Composite Score (average of Commonsense Score and
Personalized Score) serves as the primary metric rather than Case Accuracy.

### 1.3 Run Count

Following the benchmark protocol, each task is run four times for robustness,
and results are averaged across runs. However, the budget constraints (Section
1.4) necessitate a tiered approach to run allocation.

| Tier      | Systems      | Runs per task | Shopping runs | Priority  |
| --------- | ------------ | ------------- | ------------- | --------- |
| Core      | A, C2, C2-nt | 4             | 480 each      | Must-have |
| Important | C1           | 4             | 480           | High      |
| Ceiling   | B            | 2             | 240           | Medium    |

System B is allocated 2 runs instead of 4 because it serves primarily as an
upper-bound reference for oversight cost, not as a system the thesis claims to
outperform. Two runs are sufficient to establish a cost ceiling and approximate
performance ceiling, even if they provide weaker statistical power for paired
testing.

### 1.4 Budget Allocation

Empirical cost data from a complete System A run (120 Shopping tasks × 1 run)
establishes the executor baseline cost:

**Measured System A cost:** $7.06 per run (120 tasks) **Token profile:**
80,236,815 input → 1,236,712 output (81,473,527 total) **Per-task executor
cost:** ~$0.059

This measured cost is ~15× higher than the pre-experiment estimate, driven by
the large input token context that accumulates over long-horizon tasks (average
~669K input tokens per task). All budget projections below use this empirical
baseline.

**Overseer cost estimation.** The overseer (DeepSeek-V3.2) is priced at $0.26/M
input, $0.38/M output on OpenRouter. Overseer cost per task depends on
invocation frequency and context size. The estimates below assume: System B
invokes the overseer at every interaction turn (~30 turns/task average), C2
invokes at ~20% of steps (~6 invocations/task), and C1 invokes at ~3–5
checkpoints per task. Overseer context includes the task query, constraint
checklist, and recent trajectory excerpt (~3K–8K tokens input per call; ~500–2K
tokens output per call).

| System    | Executor cost/run | Overseer cost/run (est.) | Total cost/run | Runs | Subtotal  |
| --------- | ----------------- | ------------------------ | -------------- | ---- | --------- |
| A         | $7.06             | $0.00                    | $7.06          | 4    | $28.25    |
| C2        | $7.06\*           | ~$2.50                   | ~$9.56         | 4    | ~$38.24   |
| C2-nt     | $7.06\*           | ~$1.80                   | ~$8.86         | 4    | ~$35.44   |
| C1        | $7.06\*           | ~$1.20                   | ~$8.26         | 4    | ~$33.04   |
| B         | $7.06\*           | ~$8.00                   | ~$15.06        | 2    | ~$30.12   |
| Dev/debug | —                 | —                        | —              | —    | ~$15†     |
| **Total** |                   |                          |                |      | **~$180** |

\*Executor cost in overseen systems may differ from System A because corrections
alter the trajectory. Conservatively assumed equal; actual cost may be somewhat
higher (corrections extend trajectories) or lower (early error catching shortens
them).

†Development and debugging budget reduced from $20 because Gate 1 (executor
viability) is already passed — System A has completed a full run.

**Budget problem.** The total significantly exceeds the original $80–100 target.
The protocol addresses this through a prioritized execution plan (Section 6)
with explicit decision points for budget-constrained fallback.

**Minimum viable experiment (MVE).** If budget must be capped at ~$100, the MVE
is:

| System        | Runs | Cost     |
| ------------- | ---- | -------- |
| A             | 4    | $28.25   |
| C2            | 4    | ~$38.24  |
| C2-nt         | 2    | ~$17.72  |
| Dev/debug     | —    | ~$15     |
| **MVE total** |      | **~$99** |

This still answers RQ1 (A vs. C2), the novel ablation (C2 vs. C2-nt), and
supports RQ3 (C2 vs. D-cited). It sacrifices RQ2 (B vs. C2) and the C1
comparison, which can be reported as future work.

---

## 2. Metrics Collected Per Run

Every experimental run logs the following metrics. All metrics are computed
automatically using DeepPlanning's code-based automated evaluation.

### 2.1 Primary Performance Metric

**Case Accuracy** (Shopping Planning). Binary: 1 if all products in the cart
exactly match the ground-truth products, 0 otherwise. This is the strictest
measure and the primary comparison metric.

### 2.2 Secondary Performance Metric

**Match Score** (Shopping Planning, 0–1 continuous). Number of correctly matched
cart items divided by total ground-truth items. Captures partial correctness
when Case Accuracy is zero. This is essential for measuring oversight-driven
improvement on tasks that remain partially incorrect.

### 2.3 Optional Metrics (Travel, if run)

If Travel Planning is executed, the following metrics apply: Commonsense Score
(0–1 continuous, 8 dimensions × 21 checkpoints), Personalized Score (binary),
Composite Score (average of the two, primary Travel metric), and Case Accuracy
(expected near-zero for a small non-reasoning executor).

### 2.4 Efficiency Metrics

| Metric                    | Description                                         | Granularity |
| ------------------------- | --------------------------------------------------- | ----------- |
| Executor input tokens     | Total input tokens consumed by executor             | Per run     |
| Executor output tokens    | Total output tokens produced by executor            | Per run     |
| Overseer input tokens     | Total input tokens consumed by overseer             | Per run     |
| Overseer output tokens    | Total output tokens produced by overseer            | Per run     |
| Total tool calls          | Number of tool invocations by executor              | Per run     |
| Interaction turns         | Number of executor–environment turn cycles          | Per run     |
| Overseer invocation count | Number of times the overseer was called             | Per run     |
| Overseer invocation rate  | (Overseer invocations / total executor steps) × 100 | Per run     |
| Measured API cost         | Actual cost reported by OpenRouter per run          | Per run     |

**Cost computation.** Actual cost is recorded from the OpenRouter API response
headers. This is preferred over formula-based estimation because it accounts for
provider-specific pricing nuances and any overhead. For comparison purposes, the
formula-based estimate is also logged:

```
cost = (exec_input × price_in_exec/M) + (exec_output × price_out_exec/M)
     + (overseer_input × $0.26/M) + (overseer_output × $0.38/M)
```

### 2.5 Oversight-Specific Metrics (Systems B, C1, C2, C2-nt only)

| Metric                        | Description                                                                                                                   |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Trigger type distribution     | Count of each trigger type fired (error, mutating action, loop, coverage deficit, checkpoint)                                 |
| Correction level distribution | Count of each correction level applied (approve, guidance, correct observation, verification)                                 |
| Overseer latency              | Wall-clock time per overseer invocation                                                                                       |
| Intervention success rate     | Whether the correction led to task improvement (measured post hoc by comparing corrected vs. uncorrected trajectory segments) |

### 2.6 Logging Requirements

Every run must produce a complete trajectory log in JSON format containing: the
task ID, system configuration, run index (1–4), the full sequence of executor
messages (including tool calls and responses), all overseer invocations (trigger
type, overseer input, overseer output, correction applied), token counts per
turn, timestamps, and the final plan output. These logs enable post-hoc
analysis, ablation studies, and error classification.

---

## 3. Comparisons and Statistical Tests

### 3.1 Research Question Mapping

| Comparison       | Systems      | Research Question                                              | Primary Metric                            |
| ---------------- | ------------ | -------------------------------------------------------------- | ----------------------------------------- |
| A vs. C2         | A, C2        | **RQ1:** Does selective oversight help?                        | Case Accuracy, Match Score                |
| B vs. C2         | B, C2        | **RQ2:** Is selective better than always-on?                   | Case Accuracy at matched or lower cost    |
| C2 vs. D         | C2, D-cited  | **RQ3:** Can the architecture compete with monolithic?         | Case Accuracy vs. cost                    |
| C2 by complexity | C2 subgroups | **RQ4:** Does oversight help more on harder tasks?             | Case Accuracy by Shopping Level (1–3)     |
| C2 ablations     | C2 variants  | **RQ5:** What drives the gains?                                | Case Accuracy delta per ablated component |
| C2 vs. C2-nt     | C2, C2-nt    | **Novel:** Does the overseer need reasoning?                   | Case Accuracy, cost                       |
| C1 vs. C2        | C1, C2       | **Supplementary:** Does mid-trajectory intervention add value? | Case Accuracy, overseer invocation count  |

Under the minimum viable experiment, RQ2 and the C1 comparison are deferred.

### 3.2 Statistical Testing Strategy

**Structure of the data.** Each system–task combination has 4 independent runs
(2 for System B). With 120 Shopping tasks, this yields 120 paired observations
per system comparison after averaging across runs.

**Primary test: paired comparison at the task level.** For each task, average
the runs to obtain a per-task mean score. This yields 120 paired observations
for each system comparison.

- **McNemar's test** for Case Accuracy. Binarize per-task Case Accuracy by
  majority rule (≥ 3 of 4 runs correct = task success for 4-run systems; ≥ 2 of
  2 for System B). McNemar's test on the 2×2 table of concordant/discordant task
  outcomes is appropriate for binary paired data.
- **Wilcoxon signed-rank test** for Match Score. This continuous metric is
  unlikely to be normally distributed given floor/ceiling effects. Wilcoxon is
  robust to non-normality and appropriate for paired designs.
- **Effect size:** Report Cohen's d (or rank-biserial correlation for Wilcoxon)
  alongside p-values. With n = 120 tasks, the study has adequate power to detect
  medium effects (d ≈ 0.3) at α = 0.05.

**Correction for multiple comparisons.** Apply Holm–Bonferroni correction across
all planned comparisons to maintain family-wise error rate at α = 0.05.

**For RQ3 (C2 vs. D):** System D results are cited means without per-task data,
so paired testing is not possible. Report C2's per-task distribution and compare
its mean against D's published mean using a one-sample t-test or bootstrap
confidence interval. Acknowledge this as a weaker comparison and interpret
conservatively.

**For RQ4 (complexity analysis):** Shopping tasks have three complexity levels
(Level 1: straightforward item matching; Level 2: price-range constraints; Level
3: coupon timing and combinatorial optimization). Report per-level Case Accuracy
and Match Score for Systems A and C2, and test for an interaction effect between
system and complexity using a stratified Wilcoxon test or ordinal logistic
regression.

**Robustness.** Report 95% bootstrap confidence intervals (10,000 resamples over
tasks) for all point estimates. With 4 runs per task, per-task variance is
noisy; the bootstrap over the 120 tasks provides a more reliable interval.

### 3.3 Cost–Performance Analysis

For RQ2 and RQ3, the thesis argument depends on cost efficiency. Report:

- Mean cost per task for each system (with 95% CI), computed from actual API
  cost logs.
- A cost–accuracy scatter plot positioning all systems. For System D baselines,
  estimate cost from the paper's reported tool-call counts and published
  per-token pricing.
- Pareto frontier analysis: identify which systems are Pareto-optimal on the
  (Case Accuracy, cost per task) plane.

---

## 4. Domain Scope and the Travel Question

### 4.1 Shopping as Primary Domain

Shopping Planning is the sole required domain for several reasons. First,
non-reasoning models achieve meaningful baseline Case Accuracy on Shopping
(DeepSeek-V3.2 non-thinking: 10.6%), creating room for oversight to produce
measurable binary improvements. Second, Shopping evaluation is fully automated —
no LLM-based plan parsing step is needed, eliminating a potential confound.
Third, Shopping's three-level complexity structure enables clean stratified
analysis without the continuous complexity gradient of Travel (2–7 day
itineraries).

### 4.2 Travel as Optional Extension

If timeline and budget permit after all Shopping experiments, Travel Planning
can be run for systems A and C2 only (240 runs each × $7.06 + overseer ≈ $38–40
total). The purpose would be to test whether selective oversight improves
Composite Score in a domain where full task success is extremely unlikely. This
would strengthen the thesis by demonstrating generalizability, but the core
thesis argument does not depend on it.

**Decision rule for Travel.** Run Travel only if: (a) all Shopping experiments
are complete, (b) remaining budget exceeds $40, and (c) timeline permits at
least one additional week of experimentation.

---

## 5. Ablation Plan

### 5.1 Existing Ablations in the Matrix

Two ablations are already built into the main experimental matrix:

- **C1 (checkpoint-only):** Tests whether restricting oversight to natural
  checkpoints and final verification is sufficient, versus C2's mid-trajectory
  adaptive triggering. In Shopping, checkpoints occur per-item and at final cart
  submission.
- **C2-nt (no thinking):** Tests whether the overseer requires deep reasoning
  (extended thinking mode) or whether a quick non-thinking review suffices. This
  is a novel comparison not found in the literature.

### 5.2 Proposed Additional Ablations

All additional ablations modify System C2 by removing exactly one trigger
component, holding everything else constant. They are run on Shopping only (120
tasks × 2 runs each to conserve budget).

| Ablation    | Modification                                | Tests What                                                                                   |
| ----------- | ------------------------------------------- | -------------------------------------------------------------------------------------------- |
| C2−final    | Remove final checkpoint verification        | How much does end-of-plan verification contribute? (Expected highest-impact ablation.)       |
| C2−coverage | Remove coverage deficit trigger at midpoint | Does proactive coverage checking add value, or do errors surface at final checkpoint anyway? |
| C2−loop     | Remove loop detection trigger               | Is loop detection catching real failures or noise?                                           |
| C2−mutate   | Remove mutating action trigger              | Does pre-verification of cart modifications matter?                                          |

**Interpretation framework.** If removing a trigger component causes a
statistically significant drop in Case Accuracy relative to full C2, that
component is validated. If removal has no effect, the trigger is contributing
cost without benefit and should be pruned.

### 5.3 Ablation Budget and Priority

At 2 runs per ablation variant (reduced from 4 to conserve budget):

| Priority | Ablation    | Est. cost (2 runs) | Cumulative |
| -------- | ----------- | ------------------ | ---------- |
| 1        | C2−final    | ~$17               | $17        |
| 2        | C2−coverage | ~$18               | $35        |
| 3        | C2−loop     | ~$18               | $53        |
| 4        | C2−mutate   | ~$18               | $71        |

Under tight budget, prioritize C2−final (tests the highest-value component) and
C2−coverage (tests the novel coverage-deficit signal). The remaining two are run
only if budget permits. Reducing to 2 runs weakens statistical power but still
enables directional conclusions about which triggers contribute.

---

## 6. Implementation Order and Dependencies

### 6.1 Dependency Graph

```
Phase 1: System A baseline [PARTIALLY COMPLETE]
    │
    ├─ ✓ GATE 1 PASSED: Qwen3.5-9B produces valid tool calls
    ├─ ✓ Full System A run 1 complete ($7.06 spent)
    │
    ├─ Remaining: 3 more System A runs (runs 2–4)
    ├─ Analyze System A failure patterns across full dataset
    │   → Characterize error types (A1, A2, A3, B1, B2, C)
    │   → Calibrate trigger thresholds for adaptive filter
    │
Phase 2: Oversight layer + C2 pilot
    │
    ├─ Implement adaptive filter (rule-based, no LLM)
    ├─ Implement overseer integration (DeepSeek-V3.2 API)
    ├─ Implement graduated correction protocol
    ├─ Test C2 on 10 Shopping tasks → validate pipeline
    │   → Measure actual overseer cost per task
    │
    ├─ GATE 2: Do triggers fire at appropriate moments?
    │          Are corrections incorporated by executor?
    │          Is overseer cost per task within estimate?
    │          If NO → iterate on design / adjust budget
    │
Phase 3: Core systems (A × 4, C2 × 4, C2-nt × 4)
    │
    ├─ Full C2 on Shopping (120 × 4 runs)
    ├─ Full C2-nt on Shopping (120 × 4 runs)
    ├─ System A runs 2–4
    │
    ├─ GATE 3: Does C2 improve over A?
    │          If NO → diagnose; still a valid negative result
    │          If YES → proceed to expansion
    │
    ├─ ── BUDGET CHECK ──
    │
Phase 4: Expansion (budget-dependent)
    │
    ├─ C1 (checkpoint-only) × 4 runs
    ├─ B (always-on) × 2 runs
    ├─ Ablations: C2−final, C2−coverage (× 2 runs each)
    │
    ├─ Optional: Travel (A + C2 only)
    │
Phase 5: Analysis and writing
    │
    ├─ Compute all metrics
    ├─ Statistical tests
    ├─ Cost–performance plots
    ├─ Error classification on selected trajectories
    ├─ Write results and discussion
```

### 6.2 Key Dependencies and Gates

**Gate 1: Executor tool-calling viability.** ✓ PASSED. System A has completed a
full run of 120 Shopping tasks, confirming that Qwen3.5-9B can execute the
DeepPlanning function-calling protocol.

**Gate 2: Oversight layer viability.** After System A failure pattern analysis,
the oversight layer must be calibrated. The adaptive filter's trigger thresholds
(loop detection window, coverage deficit threshold, mutating action classifier)
are tuned against observed System A trajectories. A critical sub-gate is
measuring actual overseer cost on 10 pilot tasks — if overseer cost per task
exceeds $0.10 (vs. estimated ~$0.02), the budget must be revised or the overseer
context window reduced.

**Gate 3: C2 vs. A improvement.** If C2 does not improve over A on the full
Shopping run, the architecture needs diagnosis. Possible outcomes: (a) triggers
fire but corrections are ineffective → revise correction prompts; (b) triggers
don't fire → thresholds too conservative; (c) oversight genuinely cannot help at
this executor capability level → valid negative result, write up accordingly.

**Budget gate (between Phase 3 and 4).** After the core systems complete, assess
remaining budget and decide which Phase 4 elements to execute:

| Remaining budget | Execute                                   |
| ---------------- | ----------------------------------------- |
| > $80            | C1 + B + top 2 ablations + Travel (A, C2) |
| $50–80           | C1 + B + C2−final ablation                |
| $30–50           | C1 or B (not both)                        |
| < $30            | Skip Phase 4; analyze core results        |

---

## 7. Risk Mitigation

### 7.1 Budget Is the Primary Risk

**Risk level:** High. Empirical executor costs are ~15× higher than
pre-experiment estimates. The full experimental matrix (~$180) exceeds the
$80–100 target.

**Mitigation:** The tiered execution plan (Section 6) ensures the minimum viable
experiment (~$99) answers the core research questions (RQ1, RQ3, and the novel
C2 vs. C2-nt ablation). Track cumulative spend after every batch of 120 runs
using OpenRouter's usage dashboard. The MVE (A + C2 + C2-nt) is prioritized; all
other systems are expansion. If System C2 shows no improvement over A in the
pilot (Phase 2), stop early — the budget is better spent on diagnosis and
iteration than on scaling a non-working architecture.

### 7.2 Oversight Does Not Improve Anything

**Risk level:** Moderate. The trigger policy may be poorly calibrated, or the
overseer's corrections may not translate to measurable improvement.

**Mitigation:** The phased approach (pilot on 10 tasks before full runs) catches
this early. Examine trajectory logs: are triggers firing at the right moments?
Are corrections being incorporated? Is the overseer producing actionable
guidance? If oversight genuinely cannot help because the executor fails too
fundamentally, this is still a valid thesis result — the thesis question is
framed as "can selective oversight help?" and the answer "no, and here is why"
is still a contribution.

### 7.3 Overseer Cost Higher Than Estimated

**Risk level:** Moderate. The overseer cost estimates are projections, not
measured. If the overseer's context window grows larger than assumed, or if
thinking mode produces very long reasoning traces, costs could exceed estimates.

**Mitigation:** Run C2 on 10 tasks first and measure actual overseer cost per
task. If it exceeds $0.10/task (vs. estimated ~$0.02), options include: reducing
overseer context window by summarizing trajectory history instead of passing raw
turns; switching to C2-nt as the primary system (no thinking = fewer output
tokens); reducing to 2 runs for all systems.

### 7.4 API Rate Limits Slow Experiments

**Risk level:** Low-moderate. OpenRouter may impose per-minute or per-day rate
limits.

**Mitigation:** Implement exponential backoff with jitter on all API calls. Log
rate-limit responses. If throughput is limited, run tasks sequentially in
overnight batches. Budget 1–2 weeks wall-clock time for the core experiment.
Each task is independent.

### 7.5 DeepPlanning Sandbox Issues

**Risk level:** Low. Sandbox is already set up and System A has run
successfully.

---

## 8. Reproducibility

### 8.1 Model Configuration

| Parameter          | Executor (Qwen3.5-9B)                              | Overseer (DeepSeek-V3.2)          |
| ------------------ | -------------------------------------------------- | --------------------------------- |
| Provider           | OpenRouter                                         | OpenRouter                        |
| Temperature        | 0.0 (greedy)                                       | 0.0 (greedy)                      |
| Top-p              | 1.0 (disabled)                                     | 1.0 (disabled)                    |
| Max output tokens  | Model default                                      | Model default                     |
| Thinking mode      | Disabled                                           | C2/C1/B: enabled; C2-nt: disabled |
| Tool schema format | OpenAI tool-call format                            | N/A (text-based oversight)        |
| Seed               | Fixed per run (seeds: 42, 43, 44, 45 for runs 1–4) | Same seed sequence                |

**Note on temperature 0.0:** Even at temperature 0, API-served models may
exhibit non-determinism due to batching, floating-point ordering, and
provider-side load balancing. The multi-run protocol mitigates this. Seeds are
set where the API supports them, but exact reproducibility across dates or
provider routing is not guaranteed.

### 8.2 Benchmark Configuration

| Parameter                   | Value                                                        |
| --------------------------- | ------------------------------------------------------------ |
| Domain                      | Shopping Planning (primary); Travel Planning (optional)      |
| Tasks                       | 120 Shopping (all levels)                                    |
| Maximum tool calls per task | 400                                                          |
| Stopping rule               | Task complete (agent signals done) OR 400 tool calls reached |
| Task timeout                | None (rely on tool-call limit)                               |
| Evaluation method           | DeepPlanning automated code-based evaluation                 |
| Task language               | English                                                      |

### 8.3 Oversight Configuration (Systems B, C1, C2, C2-nt)

**Adaptive filter parameters (C2, C2-nt):**

| Trigger          | Threshold / Condition                                                                                                              |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Error occurrence | Tool call returns error status code                                                                                                |
| Mutating action  | Tool call is classified as state-changing (add-to-cart, apply coupon, confirm purchase) vs. read-only (search, query, get details) |
| Loop detection   | Same tool called ≥3 times with cosine similarity >0.9 on arguments within a 5-step window                                          |
| Coverage deficit | At 50% of expected trajectory length, if <50% of extracted constraint entities have been queried                                   |
| Final checkpoint | Always trigger before plan commit (agent signals intent to finalize cart)                                                          |

**Checkpoint-only parameters (C1):** Final checkpoint trigger plus one
checkpoint after each item selection.

**Always-on parameters (B):** Overseer invoked after every executor step. No
filter.

### 8.4 Empirical Baseline Data

System A (run 1) provides the following empirical baseline for Shopping
Planning:

| Metric                  | Value      |
| ----------------------- | ---------- |
| Total cost (120 tasks)  | $7.06      |
| Total input tokens      | 80,236,815 |
| Total output tokens     | 1,236,712  |
| Mean input tokens/task  | ~668,640   |
| Mean output tokens/task | ~10,306    |
| Input:output ratio      | ~65:1      |

The extreme input-heaviness (98.5% of tokens are input) reflects the growing
context window as the agent accumulates tool responses over long-horizon tasks.
This has implications for overseer cost: the overseer's input tokens (trajectory
context) will dominate its cost, not its output.

### 8.5 Logging Specification

Each run produces a JSON log file with the following structure:

```json
{
  "task_id": "shopping_042",
  "system": "C2",
  "run_index": 1,
  "seed": 42,
  "trajectory": [
    {
      "turn": 1,
      "executor_input": "...",
      "executor_output": "...",
      "tool_calls": [...],
      "tool_responses": [...],
      "executor_tokens": {"input": 1234, "output": 567},
      "oversight": {
        "triggered": true,
        "trigger_type": "coverage_deficit",
        "overseer_input": "...",
        "overseer_output": "...",
        "correction_level": "guidance",
        "correction_content": "...",
        "overseer_tokens": {"input": 2345, "output": 890}
      },
      "timestamp": "2026-04-15T14:23:01Z"
    }
  ],
  "final_plan": "...",
  "metrics": {
    "case_accuracy": 0,
    "match_score": 0.75,
    "total_tool_calls": 87,
    "interaction_turns": 34,
    "overseer_invocations": 5,
    "overseer_invocation_rate": 0.147,
    "cost_executor": 0.054,
    "cost_overseer": 0.018,
    "cost_total": 0.072,
    "cost_actual_api": 0.071
  }
}
```

---

## 9. Timeline

### Phase 1: System A Completion + Failure Analysis (Week 1)

**Status:** Partially complete. Run 1 of System A is done.

- Day 1–3: Complete System A runs 2–4 on Shopping (360 runs, ~$21).
- Day 3–5: Analyze System A trajectories. Classify failure patterns (A1:
  insufficient search, A2: tool misuse, A3: fact displacement, B1: explicit
  constraint violation, B2: implicit constraint violation, C: global
  optimization failure). Document frequency by Shopping complexity level.
- Day 5–7: Calibrate trigger thresholds. Define constraint extraction prompt.
  Test adaptive filter logic offline against recorded System A trajectories.

**Deliverable:** Complete System A results. Failure pattern analysis. Trigger
calibration report.

### Phase 2: Oversight Layer + C2 Pilot (Week 2–3)

- Day 8–11: Implement adaptive filter module (rule-based). Implement overseer
  integration: prompt template, context window construction, correction parsing.
- Day 11–13: Implement graduated correction protocol (levels 1–4). Implement
  context pruning.
- Day 13–15: End-to-end C2 test on 10 Shopping tasks. Measure actual overseer
  cost per task. Debug pipeline.
- Day 15–17: **Gate 2.** Review C2 trajectories qualitatively. Iterate if
  needed.

**Deliverable:** Validated C2 pipeline. Measured overseer cost baseline.

### Phase 3: Core Systems (Week 3–5)

- Day 17–21: Full C2 on Shopping (120 × 4 runs, ~$38).
- Day 21–25: Full C2-nt on Shopping (120 × 4 runs, ~$35).
- Day 25–26: **Gate 3** — compare C2 vs. A.
- Day 26: **Budget check.** Assess remaining funds and decide Phase 4 scope.

**Deliverable:** Core results (A, C2, C2-nt). Go/no-go for expansion.

### Phase 4: Expansion (Week 5–7, budget-dependent)

- Day 27–30: C1 on Shopping (120 × 4 runs, ~$33).
- Day 30–33: B on Shopping (120 × 2 runs, ~$30).
- Day 33–37: Priority ablations (C2−final, C2−coverage at 2 runs each, ~$35).
- Day 37–42: Optional Travel runs (A + C2 only).

**Deliverable:** Complete experimental data.

### Phase 5: Analysis and Writing (Week 7–10)

- Day 43–46: Compute all metrics. Summary tables. Per-task distributions.
- Day 46–49: Statistical tests (McNemar, Wilcoxon, bootstrap CIs).
  Cost–performance plots.
- Day 49–52: Complexity-stratified analysis (RQ4). Ablation analysis (RQ5).
  Error classification.
- Day 52–56: Write results chapter. Discussion of findings.
- Day 56–63: Revise methodology. Complete thesis draft.

---

## Appendix A: Overseer Prompt Templates

The overseer receives a structured prompt containing: (1) the original user task
query, (2) the constraint checklist extracted from the query, (3) the relevant
trajectory context (recent tool calls and responses, not the full history, to
control overseer input token cost), (4) the trigger reason, and (5) instructions
for the appropriate correction level. Exact templates will be finalized during
Phase 2 and documented in the thesis appendix.

## Appendix B: Constraint Extraction

For each task, a one-shot prompt extracts a structured constraint checklist from
the user query before execution begins. This checklist is used by the adaptive
filter to track coverage deficit. The extraction prompt will be validated
against 20 manually annotated Shopping tasks to verify accuracy before
deployment.

## Appendix C: Shopping Tool Classification

DeepPlanning Shopping tools (15 APIs) are classified as read-only or mutating:

- **Read-only:** search_products, get_product_details, get_product_reviews,
  get_available_coupons, check_shipping, get_user_profile, get_cart_contents,
  etc.
- **Mutating:** add_to_cart, remove_from_cart, apply_coupon, confirm_purchase,
  etc.

The exact classification will be documented from the benchmark's tool schema and
included in the thesis appendix.
