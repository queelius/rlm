---
status: hypothesis_with_completed_retrospective_diagnostic
evidence_date: 2026-09-22
question: Does removing redundant action arguments improve execution without doing the planning?
gpu_experiment: not_accepted
novelty: not_established
---

# Public recipe binding: remove redundant arguments, retain planning

CPU-only assessment, 2026-09-22. No implementation or GPU run accepted. This is an
interface-abstraction hypothesis, not a novelty claim or a new task planner.

## Smallest useful intervention

Let the model choose **which item and how many outputs** to craft. The host may
fill the immediate ingredient dictionary using only a recipe previously returned
by an actual `get_info` action in that episode. A strict candidate action is
`{"action":"craft_known","target_item":"product","output_count":4}`.

For the smallest contract, require exactly one previously observed recipe with
concrete ingredient names. Reject unknown recipes, ambiguous alternatives, tag
choices and nondivisible output counts explicitly. Do not query automatically,
round the output quantity, select a recipe, add missing inventory, retry, finish,
or change the model's item/quantity. Multiply each observed per-batch requirement
by `output_count / result_count`, then pass the resulting ordinary craft call to
the unchanged native executor. Log both the model action and bound native action,
the public observation that supplied the recipe, and every rejection.

This provides public recipe memory plus immediate integer arithmetic. It does
**not** expand the dependency graph recursively, total demands across branches,
reserve inventory, choose execution order, decide how much to overproduce, or
decide when the root/child goal is complete. Those remain model decisions. A
recursive agent could use the same narrow primitive inside every frame; success
would motivate testing decomposition above a competent local execution interface,
not claiming a novel hierarchy from argument binding itself.

## Does the native tool already do this?

Partly internally, but not at its public interface. Native `craft` already finds
recipes, checks batch divisibility, computes the required quantities, checks exact
ingredient equality and inventory, and executes a valid craft. It nevertheless
requires the model to repeat the exact ingredient dictionary and rejects missing,
extra or incorrectly scaled entries. The proposal exposes that existing local
computation as an action convenience, constrained to **observed** recipes rather
than consulting the private recipe database. It is not a new planning algorithm.

The known-recipe gate also removes the baseline ability to learn undiscovered
recipe details indirectly from native craft errors. This is a real interface
difference, not merely shorter JSON. Any clean comparison should apply that gate
equally to both arms, while keeping ordinary successful `get_info` observations.

## Retrospective063 evidence—not measured policy improvement

The verified32 TRAIN episodes contain496 craft calls and313 rejected crafts.
I traversed each saved node's `public_history` in order, caching only recipes
returned before each craft. Inventory was reconstructed from the public initial
inventory and actual successful craft actions; rejected calls did not mutate it.
Each hypothetical check used the original target/output quantity and actual state,
without propagating hypothetical successes into later steps or consulting gold.

| Local check on the313 actual craft errors | Calls |
|---|---:|
| Known single concrete recipe; divisible output; corrected ingredients available | 112 |
| Known single concrete recipe; divisible output; still insufficient inventory | 189 |
| Known recipe but output quantity remains nondivisible | 6 |
| Recipe had not previously been returned | 6 |

The112 locally executable substitutions span16 episodes:73 extra-ingredient,
22 missing-ingredient and17 wrong-quantity errors. The189 inventory failures
include all115 original insufficient-inventory errors plus74 argument errors
that would merely expose insufficient inventory next. Thus counting all188
original extra/missing/wrong-amount errors as repaired execution would overstate
the intervention. No multi-recipe or tag-choice case appeared among these errors.

Example: `t03-r2-flat`, history index10, asks for one `c3_i3_13` using `c5_ore`
and `c1_i1_11`. Its already observed recipe instead requires two `c6_i2` and two
`c1_i1_11`; actual inventory contains three and four respectively. Binding would
make this immediate call executable. It does not establish that the whole episode
would succeed, or that consuming those ingredients then is the right plan.

Main reran the exact producing script; report002 is byte-identical to report001.

These112 checks are neither independent opportunities nor an achievable success
ceiling. Earlier substitutions alter later inventories/prompts; some occur on
already successful episodes. They motivate a test, not a claimed rescue count.

## Smallest fixed056 comparison and decision rule

An economical exploratory screen is all eight exposed original VAL tasks × both
existing seeds:16 new recipe-bound episodes with frozen public056 checkpoint23,
original flat goal/history, T0.5, p1, k0,96 calls,8192 total generated tokens,
256 tokens/call and8192 context, versus the completed16 original057 episodes.
Retain every slot and unknown outcome. It is an interface-package comparison,
not a causal isolation of arithmetic, and uses an exposed panel—not confirmation.

For a cleaner matched mechanism test, collect both arms anew on those same32
cells, applying the public known-recipe gate to **both**: explicit ingredient
dictionary versus bound dictionary. Otherwise keep prompts/history/tool feedback
identical except the declared craft schema. Both inherit a single observed-recipe
cache, but only the bound arm uses it to assemble arguments. No hidden recipe
lookup, automatic query, recursive expansion, action fallback or extra training.

Budget roughly one hour for the16-cell screen or two hours for32 fresh cells,
with unknowns retained if capped; completed057 cost35.5 native minutes for32
mixed-profile episodes, not a guaranteed forecast for this changed interface.
Report native terminal success, local argument errors versus inventory shortages,
query/craft/finish counts, premature finishes, all call/input/output costs and
context caps. Shorter action outputs and histories can themselves save context
and compute; that is part of the package, not evidence of better planning.
The056 adapter learned the original full-dictionary schema, so a null may reflect
instruction-transfer failure; do not silently accept old malformed actions.

Promote only if native completion or efficiency improves without hiding missing
outcomes. If only local syntax/argument errors fall while inventory/planning and
terminal success do not improve, stop adding convenience layers and keep the
question on demand composition, sequencing and terminal credit. Do not schedule
this merely to duplicate the currently prepared RL competence test: it is useful
when deciding whether future recursive planning experiments need to carry this
low-level bookkeeping burden at all.

## Exact inspected evidence

Reproducible per-call audit: `R/analysis-textcraft-recipe-binding-001.json`, SHA256
`2c9d59c3f1031a148e285ebe5bc772b798fbcbf09b406988aaa52c2fe961efbb`.
It records all313 rejected craft calls, actual inventory before each call,
the earlier public recipe observation/call ID, derived ingredients and decision,
plus the32 initial states and consumed native node/report hashes. The exact
producing script is preserved at
`R/analysis-source-textcraft-recipe-binding-001/audit_textcraft_recipe_binding.py`,
SHA256 `cf0b6f5474de5b4700ac5410741235272fbaa86c581656b00c9ef8aed783cd94`.
The worktree copy only wraps a101-character string literal after that snapshot;
the calculation is unchanged. Reproduce without overwriting001:

```text
CPUPY R/analysis-source-textcraft-recipe-binding-001/audit_textcraft_recipe_binding.py \
  --report R/analysis-textcraft-recipe-binding-002.json
```

- Completed audit `R/analysis-textcraft-train-readiness-002.json`, SHA256
  `b0f97a7b98ce572e87d583bb738010417006f61d669af84b05724ad74d2d3c9d`.
  It binds all consumed `textcraft-train-readiness-001/nodes/*.json` receipts.
- Sealed063 `textcraft_bridge.py`, SHA256
  `1553db71f5c4dd4738e50d087c0a33d99d9ea7f97a084e136e4360837f2a7d40`:
  `Frame.apply` forwards explicit ingredients unchanged to native `craft`.
- Official repository revision `d9c5857d3a0a056ebc9b047241a2a0c9515aafbe`,
  `plugins/textcraft/platoon/textcraft/env.py`, lines81–226 (`craft`) and228–271
  (`get_info`), SHA256
  `c58ffad58e535d91def7291db148fd8f6433a8b3a915642f634a5b75d6d587d0`.
  Local repository is under `/project/alex_phd/research-cache/repos/platoon-rao-`
  followed by that revision. No external literature search or novelty assessment
  was performed for this bounded note.
