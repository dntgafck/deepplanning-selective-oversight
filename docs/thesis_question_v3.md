# Thesis Question

## Working thesis question

**Can a selective oversight architecture — pairing a small LLM executor (Qwen3.5-9B) with a selectively triggered stronger overseer (DeepSeek-V3.2) — improve end-to-end planning reliability on the DeepPlanning Shopping benchmark, while achieving a better cost–performance tradeoff than both always-on oversight and stronger monolithic single-agent baselines?**

## Shorter version

**Can selective multi-LLM oversight help a small planning agent match or exceed stronger monolithic baselines on DeepPlanning Shopping at lower or comparable total system cost?**

## Operational definition of "better"

For this thesis, **better** does not mean "the small model is smarter." It means one of the following:

1. **Higher Case Accuracy at equal or lower total system cost**
2. **Equal Case Accuracy at lower total system cost**
3. If neither system dominates, **a better Pareto frontier over Case Accuracy and total cost**

---

# Research Logic

This thesis separates:

1. **Architecture design questions** — how the system works
2. **Research and evaluation questions** — what the thesis is trying to prove

---

# Architecture Design Questions

These define the mechanism being studied, not the thesis claims themselves.

## AQ1 — Monitoring signals

**What observable signals should the system monitor to detect likely executor failure?**

Grounded in DeepPlanning Shopping failure modes (Zhang et al., 2026, §4.5):

- **A1 — Insufficient search:** has the executor queried all constraint-relevant entities (products, coupons, sizing, shipping)?
- **A2 — Tool misuse:** malformed tool arguments, inappropriate tool selection
- **B1 — Explicit constraint violation:** ignoring user-stated requirements (brand, rating, price range)
- **B2 — Implicit constraint violation:** conflicts with environmental reality (stock limits, coupon eligibility rules)
- **C — Global optimization failure:** suboptimal cart composition across items, coupons, and budget
- **A3 — Fact displacement:** retrieving correct info but misquoting it in the final cart

## AQ2 — Trigger policy

**When should oversight be activated?**

The adaptive filter implements five rule-based triggers, each targeting specific failure modes:

| Trigger | Condition | Targets |
|---------|-----------|---------|
| Error occurrence | Tool call returns error | A2 |
| Mutating action | State-changing tool call (add-to-cart, apply coupon, confirm purchase) | B1, B2, C |
| Loop detection | Same tool called ≥3 times with similar arguments in 5-step window | A1, behavioral drift |
| Coverage deficit | At 50% trajectory length, <50% of constraint entities queried | A1 |
| Final checkpoint | Agent signals intent to finalize cart | A3, B1, B2, C |

The filter is LLM-free and adds negligible cost. It escalates to the overseer only on flagged steps, making oversight selective rather than continuous.

## AQ3 — Correction mechanism

**What should the overseer do once triggered?**

A graduated correction protocol, where intervention intensity matches the severity of the detected issue:

| Level | Action | When used | Cost |
|-------|--------|-----------|------|
| 1 — Approve | No intervention needed | Mutating action verified as correct | Minimal (1 overseer call) |
| 2 — Guidance | Hint appended to executor's next prompt | Coverage deficit, soft constraint concern | Low |
| 3 — Correct observation | Replace or fix tool output in context | Tool misuse, malformed response | Moderate |
| 4 — Verification | Full constraint check against cart state | Final checkpoint, suspected global failure | High |

Post-correction context pruning removes stale or erroneous trajectory history to prevent cascading confusion.

---

# Main Research Questions

## RQ1 — Reliability effect (Core)

**Does selective oversight improve the planning reliability of a small LLM agent on DeepPlanning Shopping?**

Tested by: **A vs. C2** on Case Accuracy and Match Score.

This is the core question — whether the architecture improves end-to-end planning quality over the same executor without oversight.

## RQ2 — Selectivity effect (Budget-contingent)

**Is selective oversight more efficient than always-on oversight?**

Tested by: **B vs. C2** on Case Accuracy at matched or lower cost.

This tests whether selective intervention recovers most of the reliability gain without paying the full cost of continuous supervision. This comparison depends on budget availability for System B runs.

## RQ3 — Competitiveness against stronger baselines (Core)

**Can a small-model selective-oversight architecture match or exceed at least one stronger monolithic baseline at lower or comparable total system cost?**

Tested by: **C2 vs. D** (cited from DeepPlanning paper) on Case Accuracy vs. estimated cost.

The relevant D baselines for Shopping are: DeepSeek-V3.2 with thinking (42.5%), Qwen3-Max with thinking (43.5%), Claude-4.5-Opus with thinking (45.0%), and GPT-5.2-high (54.2%). Since D results are cited means without per-task data, this is a weaker comparison (one-sample test against published mean) and is interpreted conservatively.

Note: DeepSeek-V3.2 with thinking is both the overseer model and a System D baseline. Comparing C2 (Qwen3.5-9B + selective DeepSeek oversight) against monolithic DeepSeek directly tests whether selective deployment of the overseer is more cost-efficient than running it as a standalone planner.

## RQ4 — Complexity sensitivity (Core)

**Does selective oversight help more on harder Shopping tasks than on easier ones?**

Tested by: **C2 stratified by Shopping Level (1–3)** on Case Accuracy and Match Score.

Shopping complexity increases from Level 1 (straightforward item matching) through Level 2 (price-range constraints) to Level 3 (coupon timing and combinatorial optimization). Harder tasks create more opportunities for early errors to compound, which should make oversight more valuable.

## RQ5 — Mechanism sensitivity (Budget-contingent)

**Which trigger signals and correction actions contribute most to the final gains?**

Tested by: **C2 ablations** (C2−final, C2−coverage, C2−loop, C2−mutate) on Case Accuracy delta vs. full C2.

This makes the thesis more informative: not only whether selective oversight works, but which components drive the improvement. Ablation runs depend on remaining budget after core systems.

## Novel Ablation — Overseer reasoning mode (Core)

**Does the overseer require extended thinking (reasoning mode) to provide effective oversight, or is a quick non-thinking review sufficient?**

Tested by: **C2 vs. C2-nt** on Case Accuracy and cost.

This ablation has not been tested in the literature. DeepSeek-V3.2 is priced identically in both modes, but thinking mode produces more output tokens (reasoning traces), increasing cost. If C2-nt performs comparably to C2, it implies that lightweight, fast oversight is viable — a finding with practical implications for production deployment.

---

# Hypotheses

## H1 — Oversight effectiveness

A small-model agent with selective oversight (C2) will achieve higher Case Accuracy on Shopping than the same agent without oversight (A).

## H2 — Selectivity matters

Selective oversight (C2) will achieve a better cost–performance tradeoff than always-on oversight (B): comparable Case Accuracy at substantially lower total system cost.

## H3 — Competitiveness with stronger monolithic agents

C2 can match or exceed the Shopping Case Accuracy of at least one System D baseline (DeepSeek-V3.2 with thinking: 42.5%) at lower total system cost.

## H4 — Stronger gains on harder cases

The Case Accuracy improvement from oversight (C2 − A) will be larger on Level 3 Shopping tasks than on Level 1 tasks.

## H5 — Mechanism concentration

A small subset of trigger signals will account for a disproportionate share of successful interventions. Specifically, the final checkpoint and coverage deficit triggers are expected to contribute the most, while loop detection contributes the least.

## H6 — Reasoning mode is not required (Novel)

C2-nt (overseer without thinking) will achieve Case Accuracy within 5 percentage points of C2 (overseer with thinking), demonstrating that lightweight oversight is viable.

---

# Evaluation Matrix

## System A — Small executor baseline

Qwen3.5-9B in non-thinking mode, no oversight. Establishes the executor's standalone capability. Must be run (not in the DeepPlanning paper).

## System B — Small executor + always-on oversight (budget-contingent)

Qwen3.5-9B executor with DeepSeek-V3.2 (thinking) reviewing every step. Establishes the oversight ceiling: maximum accuracy gain at maximum cost. Allocated 2 runs (vs. 4 for core systems) due to its high per-run cost (~$15/run).

## System C1 — Small executor + checkpoint-only oversight (budget-contingent)

Qwen3.5-9B executor with DeepSeek-V3.2 (thinking) reviewing only at natural checkpoints (per-item and final cart). Tests whether mid-trajectory intervention adds value over periodic review.

## System C2 — Small executor + adaptive selective oversight (Primary)

Qwen3.5-9B executor with DeepSeek-V3.2 (thinking) triggered by the adaptive filter. The primary thesis system. All five trigger types active. Graduated correction protocol.

## System C2-nt — C2 with non-thinking overseer (Novel ablation)

Identical to C2 but the overseer runs without thinking mode. Tests whether deep reasoning is required for effective oversight.

## System D — Strong monolithic baselines (Cited)

Results cited directly from Zhang et al. (2026), not re-run. Shopping Case Accuracy: DeepSeek-V3.2 thinking (42.5%), Qwen3-Max thinking (43.5%), Claude-4.5-Opus thinking (45.0%), GPT-5.2-high (54.2%).

---

# What to Keep Fixed

To make the comparison credible, keep constant across all systems:

- Benchmark tasks (120 Shopping, identical per system)
- Tool access (same 15 Shopping APIs)
- Output format (structured JSON shopping cart)
- Stopping rules (task complete or 400 tool calls)
- Prompt style (same system prompt for executor across A, B, C1, C2, C2-nt)
- Evaluation pipeline (DeepPlanning automated code-based evaluation)
- Seeds per run (42, 43, 44, 45 for runs 1–4)
- Temperature (0.0 for all models)

---

# Metrics

## Primary performance metric

**Case Accuracy** (Shopping)

Binary: 1 if all cart items exactly match ground truth, 0 otherwise. The main comparison metric because partial correctness does not guarantee a valid plan.

## Secondary performance metric

**Match Score** (Shopping, 0–1 continuous)

Fraction of correctly matched cart items. Essential for measuring improvement on tasks that remain partially incorrect, especially given expected floor effects with a small executor.

## Primary efficiency metric

**Total system inference cost** (measured per task)

Actual API cost reported by OpenRouter, covering executor tokens + overseer tokens. This is the denominator of the cost–performance tradeoff.

## Secondary efficiency metrics

- Executor and overseer token counts (input and output, separately)
- Total tool calls
- Interaction turns
- Overseer invocation count and rate (% of executor steps triggering oversight)
- Trigger type distribution
- Correction level distribution

## Optional metrics (Travel, if run)

Composite Score (average of Commonsense Score and Personalized Score), Commonsense Score, Personalized Score, Case Accuracy (expected near-zero).

---

# Trigger Design for Selective Oversight

The adaptive filter implements five LLM-free, rule-based triggers grounded in DeepPlanning Shopping failure modes:

1. **Error occurrence** → immediate overseer diagnosis (targets A2: tool misuse)
2. **Mutating action** → overseer verifies before state change (targets B1, B2, C: constraint violations and optimization failures)
3. **Loop detection** → overseer provides strategic redirection (targets behavioral drift, A1: search gaps)
4. **Coverage deficit at midpoint** → overseer injects search guidance (targets A1: insufficient search)
5. **Final checkpoint** → overseer runs full constraint verification before cart commit (targets A3: fact displacement, C: global optimization)

The filter is deliberately lightweight: pure rule-based logic, no LLM calls, negligible compute cost. It escalates to the overseer only on flagged steps, expected to trigger on ~15–25% of executor steps based on analogous findings in the literature (SABER: mutating actions are 14–18% of steps; SupervisorAgent: rule-based filter catches ~20% of interactions).

---

# What Counts as Support for the Thesis

## Strong support

C2 beats at least one System D baseline on Shopping Case Accuracy while using less or comparable total cost. C2 also significantly beats A. C2-nt performs comparably to C2 (validating lightweight oversight).

## Good support

C2 matches a System D baseline on Shopping Case Accuracy with clearly lower total cost. Or: C2 does not reach D's accuracy but lies on a better Pareto frontier (higher accuracy per dollar than D).

## Still valid support

C2 significantly improves over A but does not match D. C2 also outperforms B on the cost–accuracy tradeoff. This positions selective oversight as valuable for small agents without claiming it substitutes for model scale entirely.

## Weak outcome

C2 improves over A only marginally or only on Match Score (not Case Accuracy). No improvement over B's cost–accuracy tradeoff.

This is still a valid thesis result, but the claim narrows from "architecture-level improvement" to "selective oversight provides limited partial-correctness gains for smaller agents."

## Negative outcome

C2 does not improve over A on any metric, or C2 performs worse than A (poorly calibrated triggers actively harm the executor). This is a valid negative finding about the limits of selective oversight for weak executors. The thesis contribution shifts to: characterization of failure modes, analysis of why oversight fails, and design recommendations for when selective oversight is and is not appropriate.

---

# Recommended Experimental Comparisons

Listed in priority order (core comparisons first, budget-contingent second):

## Core Comparisons

**A vs. C2** — Does selective oversight help the same small executor?  
**C2 vs. C2-nt** — Does the overseer need reasoning mode? (Novel)  
**C2 vs. D** — Can the architecture compete with a stronger monolithic agent?  
**C2 by Shopping Level** — Does oversight help more on harder tasks?

## Budget-Contingent Comparisons

**B vs. C2** — Is selective oversight more cost-efficient than always-on?  
**C1 vs. C2** — Does mid-trajectory intervention add value over checkpoints?  
**C2 ablations** — Which triggers drive the gains?

---

# Suggested Thesis Framing

This thesis does **not** claim that a smaller model becomes intrinsically more capable. It claims that a better **agent architecture** — specifically, a selective oversight mechanism pairing a cheap executor with a selectively triggered stronger overseer — can improve long-horizon planning reliability under a constrained compute budget.

The framing is explicitly about **cost-efficient architecture**, not about model capability. The key experimental lever is the trigger policy: when does it pay to invoke the expensive overseer, and when is the executor sufficient on its own?

Three properties make this framing distinctive:

1. **Cross-family pairing.** The executor (Qwen) and overseer (DeepSeek) are from different model families, ruling out the explanation that gains come from an implicit model-family scaling effect.
2. **Selective vs. always-on as an explicit comparison.** The thesis tests not just whether oversight helps, but whether selectivity itself is a design advantage.
3. **Overseer reasoning ablation.** The C2 vs. C2-nt comparison tests a practical deployment question — whether deep reasoning is required for effective oversight — that has not been addressed in the literature.

---

# Compact Version for Proposal Forms

**Main question:**  
Can selective oversight improve the reliability of a small LLM planning agent (Qwen3.5-9B) on DeepPlanning Shopping while preserving a better cost–performance tradeoff than always-on oversight and strong monolithic baselines (cited from DeepPlanning paper)?

**Core mechanism questions:**  
1. What failure signals should be monitored? (Rule-based: errors, mutations, loops, coverage, final check)  
2. When should oversight trigger? (Adaptive filter, ~15–25% of steps)  
3. What corrective action should follow? (Graduated: approve → guidance → correct → verify)

**Main evaluation questions:**  
1. Does oversight improve reliability? (A vs. C2)  
2. Does the overseer need deep reasoning? (C2 vs. C2-nt — novel)  
3. Can the architecture compete with stronger monolithic baselines? (C2 vs. D-cited)  
4. Does it help more on harder tasks? (C2 by Shopping Level 1–3)

**Budget-contingent questions:**  
5. Is selectivity more efficient than always-on oversight? (B vs. C2)  
6. Which triggers drive the gains? (C2 ablations)
