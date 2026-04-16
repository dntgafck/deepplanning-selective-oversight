# C2-lite Selective Oversight Design Memo

**Project:** `deepplanning-selective-oversight`  
**Domain priority:** Shopping Planning first  
**Status:** Integrated architecture memo for the next implementation-planning chat  
**Target system:** C2-lite v1.2

---

## 0. Purpose of this document

This memo consolidates the design discussion into one implementation-facing markdown document.

It is written for the *next* chat that will produce the handoff-grade implementation plan. Its job is to eliminate ambiguity early, preserve the key methodological constraints from the thesis, and make one concrete architectural choice.

This is **not** a production spec and **not** code. It is the pre-spec design memo.

The memo intentionally does four things at once:

1. grounds the design in the **actual repo** rather than generic agent patterns,
2. keeps the architecture aligned with the **active thesis documents** rather than archived drafts,
3. reframes the overseer prompts as a **generic scaffold instantiated from the executor’s task contract**, and
4. ends with a **single recommended C2-lite design** that is realistic to implement and defend in a short master’s thesis.

---

## 1. Active source of truth and source anchors

### 1.1 Thesis-level source of truth

These are the active documents that should govern design choices:

- `thesis_question_v3.md`
- `experimental_protocol_v2.md`
- `cross_reference_synthesis.md`

Archived v1/v2 files are historical only and should not drive new design decisions.

### 1.2 Repo files that matter most

These are the most relevant current code paths for Shopping oversight integration:

- `agent/shopping.py`
- `oversight/__init__.py`
- `oversight/state.py`
- `experiment/logging.py`
- `deepplanning/shopping_runner.py`
- `deepplanning/orchestration.py`
- `experiment/config.py`

### 1.3 Upstream benchmark files that matter most

These should be treated as authoritative for benchmark semantics:

- upstream Shopping executor prompt file: `benchmark/deepplanning/shoppingplanning/agent/prompts.py`
- upstream Shopping tool schema: `benchmark/deepplanning/shoppingplanning/tools/shopping_tool_schema.json`

### 1.4 Why these anchors matter

The core design mistake to avoid is inventing an overseer that is either:

- too **benchmark-handcrafted**, or
- too **generic to know what the executor is actually supposed to do**.

The correct compromise is:

- use the **executor system prompt + tool schema** as the authoritative task contract source,
- compile them into a structured execution contract,
- run a **generic overseer scaffold** over that contract plus compact runtime state.

That keeps the architecture general **without** pretending the benchmark has no task policy.

---

## 2. Thesis framing and design objective

### 2.1 What the thesis is actually about

The thesis is not trying to show that Qwen3.5-9B somehow becomes intrinsically “smart enough” on its own.

The thesis question is architectural:

> can a selective oversight architecture, using a small executor and a stronger selectively triggered overseer, improve end-to-end planning reliability on DeepPlanning Shopping while giving a better cost–performance tradeoff than always-on oversight and at least some stronger monolithic baselines?

That means the design must optimize for:

- **case-level reliability**,
- **cost discipline**,
- **clean ablations**, and
- **methodological defensibility**.

### 2.2 What the design is *not* trying to do

This design is **not** trying to:

- redesign the benchmark,
- redesign the executor into a new planning system,
- introduce training-heavy verifier models,
- solve general long-horizon planning as a research program,
- or maximize architectural sophistication.

This is a **thesis-winning** design target, not a maximal systems-paper target.

### 2.3 Core design question in one line

The practical question is:

> what is the simplest selective oversight architecture that can be inserted into the current Shopping wrapper loop, preserve executor benchmark faithfulness, and still plausibly improve reliability enough to matter?

---

## 3. Hard constraints vs flexible choices

### 3.1 Hard constraints

These should be treated as fixed:

- Executor must be **Qwen3.5-9B**, **non-thinking**, in A, B, C1, C2, C2-nt.
- Overseer must be **DeepSeek-V3.2**.
  - thinking mode for B, C1, C2
  - non-thinking mode for C2-nt
- Benchmark task loading, sandbox/tool execution, output extraction, and evaluation remain unchanged.
- Only wrapper loop, oversight layer, logging, and minimal config/plumbing may change.
- Do **not** change the executor’s benchmark-faithful live conversation history.
- Do **not** introduce executor-side pruning, summarization, compaction, or history rewriting.
- Do **not** add training-heavy components.
- Keep the design ablatable into:
  - `C2-final`
  - `C2-coverage`
  - `C2-loop`
  - `C2-mutate`
- Shopping is the primary domain. Travel must not shape v1.

### 3.2 Flexible choices

These are the real design variables:

- what the overseer sees,
- when the overseer is triggered,
- what exact hook points exist,
- how guidance is injected,
- how loop similarity is defined,
- how coverage is tracked,
- what exact prompt inventory exists,
- and what is fixed versus configurable.

### 3.3 Non-obvious methodological consequence

Because the executor’s benchmark-faithful history cannot be rewritten, the old idea of “append correction into history” is weaker than it first sounds.

For v1, the clean interpretation is:

- persisted benchmark history stays unchanged,
- oversight influence is injected only through **transient request-side notices**,
- those notices are logged for auditability,
- but they do **not** become part of the stored benchmark conversation trajectory.

That is a crucial design decision.

---

## 4. Repository-grounded current state

### 4.1 What the current Shopping run actually does

The current Shopping execution path is effectively:

1. launcher calls orchestration,
2. orchestration dispatches Shopping,
3. Shopping runner prepares per-run benchmark databases and log directories,
4. the Shopping agent executes a two-phase wrapper loop,
5. benchmark evaluation runs afterward.

The two-phase structure matters:

- phase 1: initial search / build trajectory
- phase 2: cart-check / finalize trajectory

This is not an abstract “single long loop.” It already has a structural midpoint.

### 4.2 Key repo facts that shape the design

#### Fact A — the executor system prompt is loaded by the wrapper

`agent/shopping.py` exposes `get_system_prompt(level)` and returns `load_shopping_prompt(level)`. That means the benchmark’s own Shopping prompt is already a clean authoritative source for the task contract. The wrapper is not inventing task rules; it is importing them from the vendored benchmark.  

#### Fact B — the Shopping wrapper is already two-phase

The current loop runs an `initial` phase, then injects the cart-check transition via `self._add_to_cart(messages)`, then runs a `cart_check` phase with `stop_on_no_calls=True`. This provides a natural repo-grounded midpoint and a natural pre-finalization checkpoint.  

#### Fact C — `request_messages` is cloned from `messages`

Inside `_run_phase`, the wrapper constructs `request_messages = list(messages)` before each executor call. That is the cleanest place to inject transient oversight notices without rewriting the persisted benchmark history.  

#### Fact D — oversight exists architecturally but is currently a stub

`oversight/__init__.py` defines `OversightAction`, but `evaluate_oversight(...)` currently returns `OversightAction(should_intervene=False)`, and `apply_intervention(...)` returns the original response unchanged. So the architecture has an insertion point, but not real behavior yet.

#### Fact E — the current oversight call site is only pre-tool

The existing Shopping loop calls oversight only when the executor produced tool calls and before those calls are executed. There is no real post-tool error hook and no mandatory final-verification hook on the `cart_check` no-tool-call exit.

#### Fact F — current runtime state is useful but insufficient

`oversight/state.py` already tracks executor tokens, overseer tokens, trigger history, and tool-call history. But `record_tool_call(...)` currently stores only:

- `tool_name`
- `args_hash`
- `result_summary`

That is enough for exact-repeat logic, but not enough for “highly similar args” loop detection unless the state is extended to store normalized raw arguments too.

#### Fact G — structured logging is already good enough

`experiment/logging.py` uses `StructuredLogger`, initializes `agent_events.jsonl` and `task_results.jsonl`, and `log_turn(...)` already records:

- `request_messages`
- raw response
- parsed tool calls
- tool results
- token counts
- stop reason

So C2-lite does not need a new logging subsystem; it needs additional event types and fields.

#### Fact H — there is still stale executor fallback residue

`deepplanning/orchestration.py` still falls back to `qwen3-14b` in `_executor_models(...)`. That conflicts with the active thesis source of truth, which now fixes the executor to Qwen3.5-9B. This should be removed or forced explicit.

### 4.3 Current hook points and missing hook points

#### Already available

- **Pre-tool hook:** after executor output is parsed into tool calls, before tools execute.

#### Missing and needed

- **Post-tool hook:** after tool results are available.
- **Phase-boundary hook:** after `initial`, before `cart_check`.
- **Final no-tool-call hook:** just before `cart_check` returns with no more tool calls.

These three additions are enough to support the full C2-lite trigger set.

### 4.4 Repo-aware midpoint choice

Because the Shopping wrapper is already two-phase, the best midpoint definition is **not** “50% of turns.”

The right midpoint for Shopping v1 is:

> the boundary between the `initial` phase and the `cart_check` phase.

That is deterministic, easy to explain, and aligned with the repo’s actual control flow.

---

## 5. Why the overseer should be generic but contract-grounded

### 5.1 What was wrong with the earlier Shopping-shaped prompt idea

A Shopping-specialized overseer prompt is too benchmark-fitted.

It risks turning the overseer into a second planner whose competence comes partly from hand-tuned benchmark knowledge rather than from a reusable selective-oversight mechanism.

That is weaker methodologically.

### 5.2 What is also wrong with “fully generic, no task contract” oversight

A totally domain-free overseer that sees only the user query and recent steps throws away important normative information that the executor was explicitly given.

In Shopping, the user query alone does **not** tell the overseer that:

- the cart is the source of truth,
- `get_cart_info` must be checked before finalization,
- Level 2 makes budget primary,
- Level 3 adds coupon scope, threshold, and stacking rules.

Those rules live in the executor system prompt and the tool schema.

### 5.3 The right compromise

The correct abstraction is:

> **generic overseer scaffold** + **compiled execution contract** + **task checklist** + **compact trajectory state**

So the overseer prompt itself should be generic:

- it monitors,
- verifies,
- and intervenes,
- but it does not define domain policy.

Domain policy comes from a compiled artifact derived from the executor’s own prompt and tool schema.

### 5.4 Thesis claim this supports

The thesis should not claim a completely domain-free overseer.

The stronger and more honest claim is:

> a generic selective-oversight architecture can be instantiated from the executor’s own task contract, rather than from benchmark-handcrafted overseer prompts.

That is a better thesis claim.

---

## 6. Candidate C2-lite designs

This section preserves the design-space exploration before convergence.

### 6.1 Candidate A — minimal-risk / simplest

#### Shape

- keep a very small oversight controller,
- add only high-precision triggers,
- minimize overseer context,
- avoid semantic loop detection beyond exact matches.

#### Trigger behavior

- tool/error occurrence: post-tool only on explicit failure payloads
- loop detection: exact same tool + exact args hash repeated 3 times in 5 executed calls
- mutating action: pre-exec on Shopping mutators
- coverage deficit: one-shot at phase boundary using coarse checklist
- final verification: always at final no-tool-call checkpoint

#### Strengths

- lowest integration risk
- lowest cost overhead
- easiest methodology story

#### Weaknesses

- likely misses semantically repetitive loops
- weak midpoint coverage signal
- final repair often arrives late

#### Judgment

Safe, but too conservative if C2 is supposed to visibly outperform A.

---

### 6.2 Candidate B — balanced / recommended

#### Shape

- use compact derived overseer state,
- add the missing three hook points,
- support semantic loop detection via normalized args similarity,
- keep interventions transient and narrow.

#### Trigger behavior

- tool/error occurrence: post-tool on explicit failures or malformed observations
- loop detection: pre-exec if current proposed tool call is highly similar to two of last five executed calls of same tool
- mutating action: pre-exec on Shopping mutators
- coverage deficit: one-shot at the real Shopping midpoint
- final verification: mandatory at final no-tool-call checkpoint

#### Strengths

- best cost/benefit tradeoff
- strongest fit with current repo
- cleanest mapping to planned ablations
- easy to explain

#### Weaknesses

- still depends on checklist quality
- needs modest runtime state extensions

#### Judgment

Best first C2-lite.

---

### 6.3 Candidate C — stronger oversight but still feasible

#### Shape

- everything in Candidate B,
- plus extra checkpoint verification after cart mutations,
- richer cart-diff reasoning.

#### Strengths

- higher ceiling for catching coupon/cart optimization failures early

#### Weaknesses

- higher cost
- more opportunity for overseer overreach
- starts to blur into a more elaborate architecture than the thesis needs

#### Judgment

Potentially better later, but not the right first C2-lite.

---

## 7. Decision matrix

| Criterion | Candidate A | Candidate B | Candidate C |
|---|---|---|---|
| Integration risk | Low | Low–Medium | Medium |
| Implementation effort | Low | Medium | Medium–High |
| Expected accuracy gain | Low–Medium | Medium–High | High |
| Expected cost overhead | Low | Medium | Medium–High |
| Trigger precision | High on mutate/final, weak on loop/coverage | Good overall | Good, but may overfire |
| Correction usefulness | Moderate | High | High |
| Ease of ablation | Good | Excellent | Good |
| Ease of methodology explanation | Excellent | Excellent | Fair |
| Robustness to weak overseer | High | Good | Lower |
| Fit with current repo | Very high | Very high | Moderate |

### Decision

Choose **Candidate B**.

It is the best balance of:

- repo fit,
- cost discipline,
- ablation cleanliness,
- and thesis defensibility.

---

## 8. Recommended architecture: generic scaffold instantiated from compiled contract

### 8.1 Architectural principle

The final recommended architecture is:

> **LLM-free trigger filter** → **generic overseer prompt scaffold** → **task-specific runtime reasoning via compiled contract + checklist + compact state**

### 8.2 Main conceptual components

There are four core artifacts:

1. **ExecutionContract**  
   compiled from executor system prompt + tool schema

2. **TaskChecklist**  
   compiled from task query under the execution contract

3. **CompactOverseerState**  
   trigger-local runtime slice derived from trajectory and current proposal/state

4. **TransientNotice**  
   deterministic renderer that converts overseer JSON into a one-turn request-side intervention

These are the key abstractions the next implementation-planning chat should treat as first-class.

---

## 9. Core artifacts and schemas

## 9.1 ExecutionContract

### Purpose

Represent the executor’s task policy in a compact, explicit, reusable structure.

### Inputs

- executor system prompt text
- tool schema
- domain label
- optional level identifier

### Run frequency

- once per `(domain, executor_prompt_hash, tool_schema_hash)`
- cached to disk

### Why this exists

The executor system prompt defines normative task policy, but it is written for execution, not for monitoring. The overseer needs a normalized representation of that policy.

### Recommended schema

```json
{
  "contract_id": "shopping_level3_<prompt_hash>",
  "domain": "shopping",
  "agent_role": "shopping assistant",
  "primary_objective": "minimize final cart price",
  "objective_priority": [
    "satisfy required items",
    "respect level-specific hard constraints",
    "optimize final total"
  ],
  "hard_rules": [
    {
      "id": "cart_authoritative",
      "text": "Cart state is the single source of truth"
    },
    {
      "id": "must_verify_cart_before_finalize",
      "text": "Verify cart with get_cart_info before finalizing"
    }
  ],
  "state_authority_rules": [
    {
      "state": "cart",
      "tool": "get_cart_info",
      "authoritative": true
    }
  ],
  "level_policy": {
    "budget_priority": "primary|secondary|none",
    "coupon_reasoning_required": true,
    "allow_over_budget_explanation": true
  },
  "tool_semantics": {
    "mutating_tools": [],
    "read_only_tools": [],
    "search_tools": [],
    "verification_tools": []
  },
  "final_output_requirements": [
    "final cart contents",
    "final calculated price",
    "brief justification"
  ]
}
```

### Important rule

`tool_semantics` must be derived from the tool schema / repo mapping logic, not from the prompt text alone.

---

## 9.2 TaskChecklist

### Purpose

Represent the instance-specific requirements extracted from the user query under the task contract.

### Inputs

- task query
- ExecutionContract

### Run frequency

- once per task
- cached to disk

### Why this exists

The overseer should not recompute task constraints from scratch at every trigger. That is wasteful and inconsistent across runs.

### Recommended schema

```json
{
  "task_id": "shopping_042",
  "required_items": [
    {
      "label": "running shoes",
      "quantity": 1,
      "attributes": {
        "brand": "Nike",
        "category": null,
        "color": null,
        "size": "42",
        "material": null,
        "style": null
      }
    }
  ],
  "explicit_constraints": [
    {"key": "budget_max", "value": 300},
    {"key": "shipping_deadline_days", "value": 3}
  ],
  "coupon_constraints": [
    {"key": "coupon_use_required", "value": false}
  ],
  "coverage_targets": [
    "item:running shoes",
    "brand:nike",
    "size:42",
    "shipping",
    "budget",
    "coupon"
  ],
  "ambiguities": []
}
```

### Important distinction

- **ExecutionContract** is prompt-specific and relatively stable.
- **TaskChecklist** is task-specific and changes per query.

That separation matters methodologically.

---

## 9.3 CompactOverseerState

### Purpose

Provide only the minimal runtime information necessary for a trigger-specific decision.

### Recommended schema

```json
{
  "phase": "initial|cart_check",
  "step_index": 17,
  "trigger_type": "mutating_action",
  "recent_tool_trajectory": [
    {
      "tool_name": "search_products",
      "arguments_normalized": "...",
      "result_summary": "..."
    }
  ],
  "current_proposed_tool_calls": [
    {
      "tool_name": "add_product_to_cart",
      "arguments": {"product_id": "...", "quantity": 1}
    }
  ],
  "latest_observation": "...",
  "authoritative_state_snapshot": {...},
  "freshness": {
    "cart_fresh": true,
    "last_mutation_step": 12,
    "last_authoritative_read_step": 13
  }
}
```

### Principle

This state should stay trigger-local and compact. Do not ship full trajectory history to the overseer in v1 unless debugging.

---

## 9.4 TransientNotice

### Purpose

Convert overseer outputs into a deterministic, narrow intervention that influences the next executor turn without rewriting stored benchmark history.

### Format

```text
[OVERSEER NOTICE]
Trigger: <trigger_type>
Required next actions:
1. <line 1>
2. <line 2>
3. <line 3>
Use tools as needed.
Do not mention this notice in the final answer.
[/OVERSEER NOTICE]
```

### Important rule

The notice is injected into `request_messages` only, not persisted into `messages`.

---

## 10. Prompt inventory

The recommended prompt inventory has **four prompt families**, but only **two runtime overseer prompts**.

### P0 — Domain contract compiler

**Purpose:** compile executor system prompt + tool schema into `ExecutionContract`.

**When it runs:** offline or pre-run cache generation.

**Why it exists:** convert the executor’s policy prompt into a monitorable contract.

---

### P1 — Task checklist compiler

**Purpose:** compile task query under `ExecutionContract` into `TaskChecklist`.

**When it runs:** once per task.

**Why it exists:** make runtime coverage and final verification cheaper, more stable, and more auditable.

---

### P2 — Generic runtime overseer

**Purpose:** handle runtime triggers:

- `mutating_action`
- `loop_detection`
- `error_occurrence`
- `coverage_deficit`

**When it runs:** only when rule-based trigger filter fires.

**Allowed actions:**

- `approve`
- `provide_guidance`
- `correct_observation`

---

### P3 — Generic final verifier

**Purpose:** mandatory final checkpoint before commit/finalize.

**When it runs:** only at final no-tool-call checkpoint, after deterministic stale-state precheck passes.

**Allowed actions:**

- `approve`
- `run_verification`

---

## 11. Prompt-family templates

## 11.1 P0 — Domain contract compiler

### System prompt

```text
You convert an executor's task instructions and tool schema into a compact execution contract for selective oversight.

Output valid JSON only.
Do not solve tasks.
Do not rewrite the task instructions.
Extract only policy-relevant structure needed for runtime monitoring and verification.

Required behavior:
- Identify the primary objective and objective priority.
- Extract explicit hard rules.
- Identify what state is authoritative and which tool reads it.
- Classify tools into mutating, read-only, search, and verification roles.
- Capture any level-specific policy differences such as budget priority or coupon logic.
- Keep the contract concise and normalized.
```

### User payload

```text
DOMAIN:
{{domain}}

EXECUTOR SYSTEM PROMPT:
{{executor_system_prompt}}

TOOL SCHEMA:
{{tool_schema_json}}

Return JSON with this schema:
{{execution_contract_schema}}
```

---

## 11.2 P1 — Task checklist compiler

### System prompt

```text
You convert a task query into an instance-specific checklist under the provided execution contract.

Output valid JSON only.
Do not solve the task.
Do not invent preferences or constraints.
If information is ambiguous, record the ambiguity.
Separate coverage targets from final verification constraints.
```

### User payload

```text
EXECUTION CONTRACT:
{{execution_contract_json}}

TASK QUERY:
{{task_query}}

Return JSON with this schema:
{{task_checklist_schema}}
```

---

## 11.3 P2 — Generic runtime overseer

### System prompt

```text
You are a selective execution overseer.

Your job is to evaluate the current executor step against:
1. the execution contract,
2. the task checklist,
3. the trigger-local trajectory state.

You are not the primary planner.
Do not solve the full task from scratch.
Prefer approve when evidence is insufficient.
Use only the allowed actions.
Keep any corrective guidance short, specific, and action-oriented.
Output valid JSON only.
```

### User payload

```json
{
  "mode": "runtime",
  "trigger_type": "mutating_action|loop_detection|error_occurrence|coverage_deficit",
  "allowed_actions": ["approve", "provide_guidance", "correct_observation"],
  "task_query": "...",
  "execution_contract": {...},
  "task_checklist": {...},
  "phase": "initial|cart_check",
  "recent_tool_trajectory": [...],
  "current_proposed_tool_calls": [...],
  "latest_observation": "...",
  "authoritative_state_snapshot": {...},
  "freshness": {...}
}
```

### Output schema

```json
{
  "action": "approve|provide_guidance|correct_observation",
  "decision_summary": "string",
  "block_current_tool": true,
  "guidance_lines": ["string"],
  "corrected_observation": "string|null",
  "violated_contract_ids": ["string"],
  "unmet_checklist_keys": ["string"]
}
```

### Trigger-specific policy notes

#### Mutating action

- allowed actions: `approve`, `provide_guidance`
- default bias: approve unless mutation appears premature, unsafe, or unsupported
- preferred guidance style: tell the executor what to verify before mutating

#### Loop detection

- allowed actions: `provide_guidance`
- if rule fired, do not silently approve
- guidance should redirect the search dimension, not just say “try again”

#### Error occurrence

- allowed actions: `provide_guidance`, `correct_observation`
- use `correct_observation` only when the provided evidence supports a direct correction

#### Coverage deficit

- allowed actions: `provide_guidance`
- guidance should mention missing coverage targets only
- cap guidance to 3 lines

---

## 11.4 P3 — Generic final verifier

### System prompt

```text
You are the final execution verifier.

The authoritative state snapshot is the source of truth.
Approve finalization only if the current state satisfies the execution contract and the task checklist.
Do not solve the task yourself.
If finalization should be delayed, state the specific blockers and the next required executor actions.
Output valid JSON only.
```

### User payload

```json
{
  "mode": "final_verification",
  "task_query": "...",
  "execution_contract": {...},
  "task_checklist": {...},
  "recent_tool_trajectory": [...],
  "authoritative_state_snapshot": {...},
  "draft_final_answer": "...",
  "finalization_retry_count": 0
}
```

### Output schema

```json
{
  "action": "approve|run_verification",
  "pass": true,
  "decision_summary": "string",
  "blockers": [
    {
      "type": "missing_item|constraint_violation|fact_displacement|budget|shipping|coupon",
      "detail": "string"
    }
  ],
  "next_step_notice_lines": ["string"],
  "violated_contract_ids": ["string"],
  "unmet_checklist_keys": ["string"]
}
```

---

## 12. Exact trigger policy and defaults

This section gives the concrete v1 defaults.

## 12.1 Trigger set

C2-lite v1 implements exactly these five triggers:

1. `error_occurrence`
2. `loop_detection`
3. `mutating_action`
4. `coverage_deficit`
5. `final_checkpoint`

---

## 12.2 Trigger evaluation order

The best ordering is **by hook site**, not as one flat list.

### Hook H1 — pre-tool execution

1. `mutating_action`
2. `loop_detection`

### Hook H2 — post-tool execution

3. `error_occurrence`

### Hook H3 — phase boundary (`initial -> cart_check`)

4. `coverage_deficit`

### Hook H4 — final no-tool-call checkpoint

5. `final_checkpoint`

### Precedence rule inside H1

If a proposed action is both mutating and repetitive, treat it primarily as **mutating**. State-changing risk is more important than loop labeling.

---

## 12.3 Exact definition of highly similar args

Use a **normalized JSON string** approach in v1.

### Recommended rule

The current proposed tool call is considered highly similar to a past call if:

- `tool_name` is the same, and
- either:
  - canonical normalized args string is exactly equal, or
  - `SequenceMatcher` ratio on canonical normalized args strings is **>= 0.92**

### Normalization recipe

- parse JSON if possible
- lowercase string values
- strip surrounding whitespace
- sort keys recursively
- serialize canonically

### Why not embeddings in v1

- harder to justify threshold
- extra dependencies
- more opaque methodology
- unnecessary for a first selective architecture in a short thesis

---

## 12.4 Exact loop trigger condition

Trigger `loop_detection` when:

- the current proposed tool call matches the same tool,
- and is highly similar to at least **two** of the last **five** executed tool calls,
- so the current proposal would create the **third** similar call in that 5-call window.

### Numeric defaults

- repeat count: **3**
- window size: **5 executed tool calls**
- similarity threshold: **0.92**

---

## 12.5 Exact midpoint definition

For Shopping v1:

> midpoint = the boundary between `initial` and `cart_check`

Do **not** use half of turns, half of tools, or expected item count for v1.

Reason:

- the repo already has a two-phase structure,
- the phase boundary is deterministic,
- it is easy to defend,
- and it directly matches the finalization semantics of Shopping.

---

## 12.6 Exact mutating-tool set for Shopping

Use the real tool schema names:

- `add_product_to_cart`
- `delete_product_from_cart`
- `add_coupon_to_cart`
- `delete_coupon_from_cart`

Treat `get_cart_info` as read-only but **verification-critical**.

Do not rely on shorthand names from prose drafts such as `confirm_purchase`, which do not exist in the actual Shopping tool schema.

---

## 12.7 Coverage deficit definition

### Principle

Coverage tracking should use only **search-relevant targets**, not every global optimization objective.

### Why

If coverage tries to track all final objective elements at midpoint, it becomes noisy and under-precise.

### v1 rule

At the phase boundary, compute:

```text
coverage_fraction = covered_coverage_targets / total_coverage_targets
```

Trigger `coverage_deficit` if:

```text
coverage_fraction < 0.50
```

### Covered means

A coverage target counts as covered if the recent executed trajectory contains evidence that the executor has queried or inspected that target or an equivalent normalized variant.

### Examples

- `item:running shoes` covered by search/query over running shoes
- `brand:nike` covered by brand filter or product detail inspection involving Nike
- `coupon` covered by coupon-related search or cart coupon reasoning
- `shipping` covered by transport-time calculation or explicit shipping inspection

---

## 12.8 Final verification behavior

Use a **staged final checkpoint**, not a single-pass judge.

### Stage 0 — deterministic stale-state precheck

Before calling the overseer:

- if any mutating action occurred after the last authoritative cart read,
- do **not** call the overseer yet,
- instead inject a fixed transient notice:

```text
Call get_cart_info before finalizing. The cart state is the source of truth.
```

Then continue one more executor turn in `cart_check`.

### Stage 1 — final verifier

When:

- `cart_check` produced no tool calls, and
- the cart is fresh,

call `P3`.

### Final verifier outcomes

- `approve` -> finalize now
- `run_verification` -> do not finalize; inject repair notice; continue one more executor turn

### Retry cap

- max final repair retries: **2**

This prevents endless endgame ping-pong.

---

## 13. Intervention semantics

## 13.1 Important implementation reinterpretation

The current code shape suggests “apply intervention to response,” but for v1 that is the wrong semantics.

The correct runtime interpretation is:

- **approve** -> continue normally
- **provide_guidance** -> optionally block current tool execution, set `pending_executor_notice`, then continue to next executor turn
- **correct_observation** -> set `pending_executor_notice` containing authoritative correction, then continue
- **run_verification** -> final verifier blocked commit; inject repair notice and continue

### Crucial rule

Do **not** rewrite persisted `messages` history and do **not** overwrite previously returned tool observations in-place.

That would violate the benchmark-faithfulness constraint.

---

## 13.2 Deterministic notice renderer

The notice renderer should be deterministic, not another LLM step.

### Renderer rules

- max 3 guidance lines
- max ~120 words total
- no chain-of-thought
- no full task replanning in notice text
- no explanation of overseer internals

### Standard template

```text
[OVERSEER NOTICE]
Trigger: {{trigger_type}}
Required next actions:
1. {{line_1}}
2. {{line_2}}
3. {{line_3}}
Use tools as needed.
Do not mention this notice in the final answer.
[/OVERSEER NOTICE]
```

---

## 13.3 Mutating-action intervention policy

### Pre-execution

When the current proposed action is a mutator:

- default: `approve`
- if risky / premature / inconsistent with current contract or checklist state: `provide_guidance`
- if `provide_guidance`, block current mutation and request a corrective next step

### Guidance examples

- verify cart state first
- inspect coupon applicability first
- verify missing required attribute first
- compare lowest-cost alternative first

Avoid strong item-choice micromanagement in v1.

---

## 13.4 Error-occurrence intervention policy

### Post-execution

When tool output indicates error, invalid state, or malformed observation:

- use `correct_observation` only when the evidence supports a direct correction
- otherwise use `provide_guidance`

### Examples

Use `correct_observation` for:

- executor appears to misread the tool output
- tool explicitly returned an error and the correction is just to recognize that fact

Use `provide_guidance` for:

- retry with corrected parameters
- inspect user info before choosing size
- re-read product details for missing attribute

---

## 13.5 Coverage-deficit intervention policy

Coverage guidance should be narrow and checklist-oriented.

Good:

- “Search for remaining missing item requirement: size 42.”
- “You have not verified shipping yet.”
- “Check applicable coupons before finalizing.”

Bad:

- long strategic lecture
- full re-solve of task
- overwriting executor autonomy

---

## 13.6 Final-verification intervention policy

When final verifier returns `run_verification`, the next-step notice should state:

- specific blockers,
- specific required next actions,
- and nothing more.

Example pattern:

```text
[OVERSEER NOTICE]
Trigger: final_checkpoint
Required next actions:
1. The cart is missing one required item: running shoes size 42.
2. Re-check coupon applicability after adding any missing item.
3. Call get_cart_info again before finalizing.
Use tools as needed.
Do not mention this notice in the final answer.
[/OVERSEER NOTICE]
```

---

## 14. Shopping v1 instantiation details

This section explains how the generic scaffold is instantiated for the first real domain.

## 14.1 Contract compiler expectations for Shopping

The Shopping `ExecutionContract` should capture, at minimum:

- objective hierarchy by level
- cart-as-authoritative-state rule
- mandatory `get_cart_info` before finalization
- budget priority differences between levels
- coupon reasoning requirements at Level 3
- tool semantics split into mutating / read-only / search / verification

### Expected level differences

- **Level 1:** primary objective is absolute lowest final price
- **Level 2:** meeting budget is primary; minimizing price is secondary within budget
- **Level 3:** products **plus coupons** determine optimality; coupon scope/threshold/stacking rules matter

## 14.2 Checklist compiler expectations for Shopping

The `TaskChecklist` for Shopping should at minimum capture:

- required products
- key attributes per product
- explicit budget constraints if present
- shipping/timing constraints if present
- coupon-related requirements if present
- search-relevant coverage targets
- ambiguities

### Important midpoint rule

Coverage should track:

- required product entities
- important product attributes
- coupon requirement/search relevance
- shipping/budget items where relevant

It should **not** try to judge global optimality at midpoint.

## 14.3 Authoritative state snapshot for Shopping

The authoritative state snapshot should be built primarily from the latest `get_cart_info` result.

Suggested structure:

```json
{
  "cart_items": [
    {
      "product_id": "...",
      "name": "...",
      "quantity": 1,
      "price": 99.0
    }
  ],
  "used_coupons": [
    {
      "coupon_name": "Cross-store: ¥30 off every ¥300",
      "quantity": 1
    }
  ],
  "summary": {
    "total_items_count": 2,
    "total_price": 268.0
  }
}
```

### Freshness flags

Also track:

- `last_mutation_step`
- `last_authoritative_read_step`
- `cart_fresh = last_authoritative_read_step >= last_mutation_step`

---

## 15. Required repo changes

This section is intentionally implementation-oriented.

## 15.1 Hook H0 — pre-run artifact preparation

Add cache-aware loading/generation for:

- `ExecutionContract`
- `TaskChecklist`

### Key requirement

These should be artifacts independent of a specific run attempt where possible.

Recommended cache keys:

- contract: `(domain, prompt_hash, tool_schema_hash)`
- checklist: `(task_id, contract_id, task_query_hash)`

---

## 15.2 Hook H1 — pre-tool oversight

Reuse the existing pre-tool oversight hook in `agent/shopping.py`, but change semantics.

### Required behavior

- read `pending_executor_notice` and inject it into `request_messages`
- evaluate mutating and loop triggers before executing proposed tools
- if trigger returns blocking action, skip current tool execution and continue next executor turn

### Important note

Do not mutate stored `messages` for this.

---

## 15.3 Hook H2 — post-tool oversight

Add a new hook immediately after tool results are available.

### Required behavior

- detect tool errors / malformed observations
- construct compact post-tool overseer state
- call P2 only when `error_occurrence` fires
- set `pending_executor_notice` when intervention is needed

---

## 15.4 Hook H3 — midpoint coverage checkpoint

Add one-shot coverage evaluation at:

- after `initial` phase completes
- before `cart_check` begins

### Required behavior

- compute coverage fraction from executed trajectory and `TaskChecklist.coverage_targets`
- if deficit, call P2 with `coverage_deficit`
- set `pending_executor_notice` for first turn of `cart_check`

---

## 15.5 Hook H4 — final no-tool-call verification checkpoint

Add a new hook just before the current `return messages, "no_tool_calls"` branch exits the `cart_check` phase.

### Required behavior

1. run deterministic stale-cart precheck
2. if stale, inject fixed notice and continue
3. else call P3
4. if approved, finalize
5. else inject repair notice and continue
6. enforce retry cap

This is the single most important missing hook.

---

## 15.6 Runtime state extensions

Extend `ConversationState` with fields such as:

- `normalized_tool_args_history`
- `latest_authoritative_cart_snapshot`
- `last_mutation_step`
- `last_authoritative_read_step`
- `pending_executor_notice`
- `execution_contract_id`
- `task_checklist_id`
- `coverage_status`
- `final_verification_retry_count`

### Critical addition

Store normalized raw args for recent tool calls, not only hashes.

---

## 16. Logging additions

Reuse `agent_events.jsonl` and `task_results.jsonl`.

## 16.1 Add per-trigger logging

Log fields such as:

- `trigger_type`
- `phase`
- `step`
- `allowed_actions`
- `similarity_score`
- `coverage_num`
- `coverage_den`
- `coverage_fraction`
- `cart_stale_after_last_mutation`
- `blocked_current_tool`

## 16.2 Add per-overseer-call logging

Log fields such as:

- `overseer_prompt_name`
- `overseer_prompt_version`
- `overseer_model`
- `thinking_mode`
- `input_tokens`
- `output_tokens`
- `latency_ms`
- `compact_state_size_chars`

## 16.3 Add intervention logging

Log fields such as:

- `action`
- `pending_notice_injected`
- `notice_text`
- `notice_length_chars`
- `finalization_delayed`

## 16.4 Add run-summary logging

Log summary metrics such as:

- `overseer_invocation_count_by_trigger`
- `intervention_count_by_action`
- `final_verification_retries`
- `blocked_mutation_count`
- `checklist_version`
- `contract_version`
- executor vs overseer token totals
- total API cost fields

---

## 17. Config changes

Extend `SystemConfig` / config plumbing with fields like:

- `oversight_prompt_version`
- `loop_similarity_threshold`
- `loop_window`
- `coverage_threshold`
- `final_repair_retry_cap`
- `mutating_tools`
- `inject_transient_notice`
- `execution_contract_cache_dir`
- `task_checklist_cache_dir`

### Keep fixed in v1

These should probably remain hard-coded or default-locked initially:

- midpoint = phase boundary
- final checkpoint = final no-tool-call hook
- guidance injection = transient request-side only
- post-correction history pruning = disabled/deferred

### Small but important cleanup

Remove or override the stale `qwen3-14b` fallback in orchestration.

---

## 18. Defaults and thresholds

These are the recommended initial operational defaults.

| Parameter | Default |
|---|---:|
| Loop repeat count | 3 |
| Loop window | 5 executed tool calls |
| Similarity threshold | 0.92 |
| Coverage threshold | 0.50 |
| Coverage checkpoint timing | initial -> cart_check boundary |
| Recent context length for overseer | last 5 executed tool interactions |
| Final repair retry cap | 2 |
| Max notice lines | 3 |
| Max notice length | ~120 words |
| Mandatory cart freshness before final verify | yes |

### What to leave configurable

- similarity threshold
- loop window
- coverage threshold
- recent context length
- final retry cap
- mutating tool set
- prompt version

### What not to parameterize yet

- staged final verification itself
- phase-boundary midpoint choice
- transient request-only intervention semantics

---

## 19. Risks and mitigations

## 19.1 Technical risks

### Risk 1 — final checkpoint never actually triggers correctly

**Cause:** current code returns immediately on no-tool-call cart-check exit.

**Mitigation:** add explicit final no-tool-call hook before return.

---

### Risk 2 — loop detection underperforms

**Cause:** state only stores arg hashes.

**Mitigation:** store normalized raw args string for last N tool calls.

---

### Risk 3 — overseer cost grows too much

**Cause:** oversized overseer context or too many invocations.

**Mitigation:**

- compact trigger-local state only
- cached contract/checklist artifacts
- deterministic stale-state precheck before final verifier
- keep trigger filter LLM-free

---

### Risk 4 — overseer overcorrects correct trajectories

**Cause:** too aggressive interventions or too much free-form guidance.

**Mitigation:**

- default to approve
- keep notices short and evidence-linked
- use high-precision trigger conditions
- reserve strongest intervention for final checkpoint

---

### Risk 5 — endgame ping-pong

**Cause:** final verifier repeatedly refuses approval.

**Mitigation:** cap repair retries at 2 and log exhaustion.

---

## 19.2 Methodological risks

### Risk 6 — criticism that the benchmark interaction was changed

**Mitigation:**

- benchmark loader unchanged
- sandbox/tool execution unchanged
- evaluator unchanged
- executor stored history unchanged
- only wrapper-local transient notices added

---

### Risk 7 — criticism that the overseer is benchmark-tuned

**Mitigation:**

- overseer role prompts are generic
- benchmark/task specifics are compiled from executor prompt + tool schema
- instantiation is contract-based, not prompt-handcrafted

---

### Risk 8 — criticism that coverage extraction is another hidden model dependency

**Mitigation:**

- compute once per task
- cache artifact
- validate on small manually checked subset
- treat as overseer-input preprocessing, not executor behavior

---

### Risk 9 — criticism that context pruning in thesis prose conflicts with faithfulness

**Mitigation:** explicitly defer post-correction context pruning from v1 and state why.

---

## 20. What should *not* be built in v1

Do **not** build any of the following in the first C2-lite implementation:

- embedding-based loop similarity
- learned detectors
- PRMs / reward models
- hidden-state probes
- RL or SFT
- post-correction history pruning
- executor-side summarization or compaction
- dual-loop strategic manager beyond the chosen hooks
- formal SMT/Z3 full solver integration
- Travel-specific logic
- restart trees / MCTS / search-heavy replanning
- extra LLM prompt families beyond the four defined here

These can be discussed as future work or later ablations, but should not enter v1.

---

## 21. Final chosen design

### 21.1 One-sentence summary

The chosen C2-lite v1.2 design is:

> a rule-triggered, generic selective-oversight scaffold whose runtime decisions are grounded in a compiled execution contract derived from the executor’s own system prompt and tool schema, and whose interventions are injected transiently into the next executor request without rewriting stored benchmark history.

### 21.2 Concrete implementation stance

- use **Candidate B**
- Shopping only for first implementation
- generic overseer role prompts
- contract-derived task policy
- cached checklist artifacts
- four prompt families (`P0`..`P3`)
- five triggers only
- four hook sites total
- request-side transient notice injection only
- no pruning or history rewriting

### 21.3 Why this is the best design for the thesis right now

Because it simultaneously achieves:

1. **repo fit**  
   It aligns with the current two-phase Shopping loop and existing oversight stub.

2. **cost discipline**  
   Oversight remains selective and compact-state based.

3. **clean ablations**  
   Each trigger maps cleanly to your planned C2 ablation family.

4. **methodological clarity**  
   The overseer is generic, but still grounded in the same task contract the executor actually received.

5. **implementation realism**  
   It requires wrapper and state changes, not a redesign of the whole benchmark stack.

---

## 22. Compact handoff memo for the next chat

Use this section as the tightest summary.

### Chosen architecture

Implement **C2-lite v1.2** as a selective oversight controller around the existing Shopping wrapper.

### Core principle

Do **not** handcraft Shopping-specific overseer role prompts.

Do **not** reuse the executor prompt verbatim as the overseer prompt.

Instead:

- compile **ExecutionContract** from executor system prompt + tool schema,
- compile **TaskChecklist** from task query under that contract,
- feed those into a **generic runtime overseer** and **generic final verifier**.

### Runtime hooks

Add four effective oversight stages:

- **H0**: pre-run contract/checklist loading
- **H1**: pre-tool oversight for `mutating_action` and `loop_detection`
- **H2**: post-tool oversight for `error_occurrence`
- **H3**: phase-boundary coverage check at `initial -> cart_check`
- **H4**: final no-tool-call checkpoint with staged final verification

### Prompt inventory

Implement exactly four prompt families:

- `P0_domain_contract_compiler`
- `P1_task_checklist_compiler`
- `P2_generic_runtime_overseer`
- `P3_generic_final_verifier`

### Trigger set

Implement only these five triggers in v1:

- `error_occurrence`
- `loop_detection`
- `mutating_action`
- `coverage_deficit`
- `final_checkpoint`

### Exact defaults

- same tool + normalized args similarity `>= 0.92`
- third similar call in last 5 executed calls triggers loop detection
- Shopping midpoint = `initial -> cart_check` boundary
- coverage deficit if coverage fraction `< 0.50`
- final verification staged: stale-cart precheck first, then final verifier
- max final repair retries = 2

### Shopping mutating tools

Use actual schema names:

- `add_product_to_cart`
- `delete_product_from_cart`
- `add_coupon_to_cart`
- `delete_coupon_from_cart`

### Intervention semantics

Do not rewrite benchmark history.

Instead:

- keep stored `messages` benchmark-faithful
- inject notice only into `request_messages`
- render notices deterministically
- log them in structured logs for auditability

### Things to defer

Do not build:

- history pruning
- executor summarization
- learned detectors
- embeddings for loop detection
- Travel-specific logic
- full formal solver verification

### Immediate implementation priority order

1. add missing hook points  
2. add runtime state extensions  
3. implement contract/checklist compilers  
4. implement P2/P3 prompt plumbing  
5. implement deterministic notice injection  
6. extend logging/config  
7. remove stale executor fallback

---

## 23. End state expected from the next chat

The next chat should transform this memo into a handoff-grade implementation plan with:

- concrete file-by-file patch plan,
- exact data class changes,
- exact function signatures,
- config additions,
- logging payload specs,
- and implementation order.

This memo is the architectural foundation for that step.
