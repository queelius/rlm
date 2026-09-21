# TextCraft-Synth: bounded recursion qualification, not RAO reproduction

## Question and current readiness

Does access to fresh, recursively delegated contexts improve exact crafting success
as recipe depth increases, when **all** root/child model calls and generated tokens
share one budget? This is a concrete recursive stateful task, unlike answering a
short composed question whose whole plan can be written upfront. It still need not
have stochastic answer-conditioned branching: recipes are deterministic and can be
fully inspected and preplanned. The uncertainty is initially undiscovered recipes
and execution state, not an invented restriction on the flat agent.

[RAO already trains recursive agents](https://apga.github.io/RAO/), including local
child rewards on this benchmark. Our strict-JSON, sequential, one-A100 screen would
be a **protocol adaptation**, not a new recursive method or a reproduction of its
Python-REPL/parallel-agent/training results. No environment code was executed, no
package installed, and no GPU job accepted during this inspection.

## Actual cached assets and split limits

Official cache: `/project/alex_phd/research-cache/repos/`
`platoon-rao-d9c5857d3a0a056ebc9b047241a2a0c9515aafbe`.
Git HEAD remains `d9c5857d3a0a056ebc9b047241a2a0c9515aafbe`; clean worktree.
Acquired2026-09-11; code/data re-inspected2026-09-21. Root MIT license; no separate
dataset license was established here. Receipt:
`/project/alex_phd/runs/rlm-research-r4/acquisitions/2026-09-11-platoon-rao.json`.

| Checked-in family | TRAIN | VAL | Root targets shared | VAL targets occurring in TRAIN gold intermediate crafts |
|---|---:|---:|---:|---:|
| Single-target |2,522|632|0|221/256 distinct targets|
| Multi-target |10,000|1,000|0|60/69 distinct targets|

These counts come from JSON parsing, not README defaults. All632 single-target VAL
tasks share at least one crafted item with TRAIN trajectories;1,138 distinct crafted
items appear in both splits. This is a shared-world/recombination/depth benchmark,
**not unseen recipe-family generalization**. Upstream splits target-item pools,
not every ancestor recipe. This is scientifically usable; removing every shared
ancestor would erase the intended compositional question. Future new-world tests
should use separately generated recipe seeds and be labeled separately.

Single-target VAL has147 easy(depth2–3),213 medium(4–6),136 hard(7–9),136 extreme
(10–12) tasks. Target counts:256 request1,256 request2,120 request3. Depth2 tasks
have1–3 saved craft steps; depth3:3–8; depth4:5–17; depth5:9–38; depth6:18–65.
Gold trajectories, difficulty, depth and oracle step counts stay host-only.

**Recipe-cache mismatch matters:** checked-in `synth_recipes` has382 recipes;
189/256 single-target VAL target names are absent there. The official synth factory
instead regenerates recipes with seed42 and `items_per_domain_tier=25` for the
single-target dataset. Do not substitute that stale checked-in recipe directory.
Before any model call, regenerate with the reviewed stdlib generator and validate
every selected gold trace against the native craft/checker methods.

## Exact native semantics and safe interface

The [official environment](https://github.com/ApGa/platoon/blob/d9c5857d3a0a056ebc9b047241a2a0c9515aafbe/plugins/textcraft/platoon/textcraft/env.py)
provides `get_info`, `view_inventory`, and `craft`. `get_info` reveals recipe
ingredients/output counts and item depth, identically to both policies. Crafting
requires exact scaled ingredients, enough inventory, no extraneous ingredients,
and output quantity divisible by recipe output count. Successful crafting consumes
ingredients and adds outputs; invalid actions leave state unchanged.

Root success requires an explicit, nonempty `finish` message and, for every target,
`final_inventory[target] - initial_inventory[target] >= requested_count`.
Child success uses its own start-inventory snapshot and is measured at child finish;
root/children share inventory. A child report is not an oracle answer or root success.

Important replay seam: saved gold `target[1]` is the **number of recipe executions**,
but native `craft(..., target=(item,count))` expects **total output items**. Replay
with saved `result_count`, not `target[1]`. Example VAL0 requests3`t7_i3` and the
gold first step stores `target=[t8_i1,1], result_count=2`: calling native craft with1
would fail divisibility. The final step executes twice and produces4`t7_i3`.

Proposed public actions: exact JSON `get_info(items)`, `view_inventory`,
`craft(ingredients,target_item,output_count)`, `finish(message)`, and recursive-only
`delegate(targets,context)`. Host maps validated fields to **reviewed official methods**;
never evaluate model Python, shell text, imports, or arbitrary attribute names.
Reject booleans as integer counts, unknown fields/actions and nonpositive quantities;
record rejection feedback and charge the call/tokens, with no repair/fallback.
Preserve the public scaling/net-growth instructions for both policies.

For a minimal native environment, generator/recipe logic is stdlib Python; env
imports need IPython/traitlets and the package initializer also traverses the agent
client into OpenAI/LiteLLM. Full `uv sync --extra areal` would add an unnecessary
training backend and notebooks. A focused environment adapter can reuse reviewed
craft/info methods and exact checker without the REPL execution path; any extraction
must retain MIT/source hashes and pass native-method differential fixtures. If native
imports are chosen, inspect/import-test that small dependency path on CPU in an
isolated environment first, not mutate the active training environment. No heavy
AReaL/Tinker/backend install is needed for this frozen-policy screen.

## Smallest informative screen (proposal only)

- Eight outcome-independent VAL tasks: four with recipe depth2–3 and four depth4,
  deterministic hash order within strata, two seeds each. Flat versus max delegation
  depth2(root→child→grandchild):32 complete episodes, not32 independent tasks.
- Same frozen Qwen3-4B base, T.5/top-p1/top-k0,128 output tokens/call,8K input context.
  Both policies may reason in the same bounded optional note field. One96-call and
  8,192-generated-token budget per **whole episode tree**, including delegate,
  child, finish and invalid requests; at most3,072 model calls overall, one-hour cap.
- Sequential child execution avoids shared-inventory races and global-adapter state
  issues. No free helper calls, auto-crafting, gold recipe chain, solver-generated
  subgoals, or hidden-success hints. Flat receives the same recipe access and action
  feedback, can query batches, and may solve the whole task itself.
- CPU feasibility first: exact regenerated recipes plus quantity-correct gold replay
  must reach native success for all eight selected tasks; fixture invalid craft,
  shared inventory, net-growth child/root checker and global budget propagation.
  Native model smoke must return a real valid action promptly; syntax success alone
  is not task success. No outcome-based task replacement.
- Report all32 outcomes: exact native success, malformed/action errors, budget stops,
  actual craft/info/delegation counts, depth reached and parent/child costs. Pair
  policies within task/seed, but any uncertainty resamples only eight tasks. This
  is interface/depth qualification, not a benchmark or latency-parallelism claim.

One resident4B model is approximately8GB BF16 weights plus KV/runtime headroom;
one40GB A100 should fit sequential inference. Duration is a cap/estimate, not measured
throughput: roughly20–60 minutes, depending on tree size and context growth.

Promote only if actions/quantities/checkers are reliable and there are genuine nested
delegations plus correct root outcomes, with a plausible depth-dependent benefit at
the shared budget. If both arms fail basic quantities, qualify the interface/skill
before RL. If flat solves everything cheaply, increase recipe depth prospectively;
if recursion only spends more tokens or achieves local targets without root gains,
do not claim hierarchy benefit. Larger search/training remains unaccepted.

## Provenance hashes

Paths below are relative to `plugins/textcraft/platoon/textcraft/` in the pinned cache:

- `env.py`: `c58ffad58e535d91def7291db148fd8f6433a8b3a915642f634a5b75d6d587d0`
- `synth_tasks.py`: `5cc082f13d6e6671aae2dfcf043f68a0d099fb3ec05d75e9a50885cb35d033c0`
- `synth_recipe_generator.py`: `1c33e0a6f61759eb3a8eb33b35f0b88361155525f5f575885b53acbd011efb0e`
- `synth_recipe_loader.py`: `50b0ec5cc22fa31987e136c5e9806215b3a4cd854f2d7579391eb52e57eb1916`
- `textcraft_synth_train.jsonl`: `685823fbca90aa89e061e61fef58d0122def4221eed14cf2c8c730341363a00f`
- `textcraft_synth_val.jsonl`: `84a123ee46e29e65f4d7b95f4943aa56907fc3fd5992268faa749602579dfc0c`

Existing September11 code notes remain the authority on inspected RAO credit/weighting
implementation; this note only adds task/protocol readiness and does not duplicate it.
