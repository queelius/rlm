# World47: fewer tool errors did not guarantee finishing the job

24 September 2026. Descriptive reading of **all four discordant paired slots**, not a new evaluation or a causal mechanism test. Eight task parents × two sampling seeds, one changed recipe world, one training seed. “Corrected” below means the quantity-corrected original teacher; “public” means the public-discovery teacher.

The completed native audit gives corrected **4/16** versus public **6/16**: public has three paired wins and one loss, on only three distinct task parents. Public-minus-corrected is +12.5 percentage points; parent-bootstrap95% interval **−12.5 to +43.75 points**. This small comparison does not establish a general advantage.

Corrected made43 native action errors in491 calls; public made312 native action errors **plus one schema error** in621 calls. These are not interchangeable “invalid” totals. Corrected also issued358 recipe queries and115 crafts, versus public198 queries and407 crafts. Fewer error messages partly accompany a different allocation of actions, not necessarily more completed work.

## Every discordant pair

| Task / repeat | Corrected trace | Public trace | Native outcome, corrected → public |
|---|---|---|---|
| val.325 /0 (`t00-r0`) | Queries several nonexistent names; eventually finds the target recipe, makes2 when goal requires3, then finishes.7 calls,1 error. | Queries target first; makes4 in valid batches of2, then finishes.6 calls,0 errors. | 0 →1 |
| val.325 /1 (`t00-r1`) | Longer detour through nonexistent names, then again makes2 and finishes below goal3.15 calls,1 error. | Queries target first; makes4, then an unnecessary additional2, and finishes with6.7 calls,0 errors. | 0 →1 |
| val.629 /0 (`t01-r0`) | Initially attempts final craft without prerequisites. Makes both prerequisites successfully, but finishes without assembling the target.8 calls,1 error. | Queries and crafts both prerequisites, then performs final assembly and finishes.7 calls,0 errors. | 0 →1 |
| val.47 /0 (`t02-r0`) | Starts on a different item and makes two rejected crafts, then corrects the quantity, follows the target dependencies and completes.16 calls,2 errors. | Queries the correct dependencies first, but repeatedly adds an extraneous ingredient despite explicit rejection.67 calls,43 errors; finishes without target. | 1 →0 |

All four ended with an explicit `finish`; their score differences are not missing-owner or timeout artifacts. Public's overproduction on325/repeat1 is a successful but wasteful route, not evidence of optimal quantity planning.

## Two slide-friendly examples

History indices below are zero-based and match the `cNNN` native call suffixes in these traces.

**1. “It made the parts, then stopped before assembling the product.”**

On629/repeat0, corrected had already obtained4 units of `a5_i1` and2 of `a8_i1`, enough for the previously observed target recipe. Its next action was `finish`; target inventory remained0. Public executed that last assembly, producing2 target units for a goal of1. This directly illustrates a gap between valid local actions and completing the original goal. It does **not** reveal the model's internal reason for stopping.

Native pointers: corrected [node history](/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/textcraft-world47-quantity-corrected-seed2208-quantity_corrected_original-004/nodes/t01-r0-flat-original-n0.json), indices4,6,7 (parts, parts, finish); public [node history](/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/textcraft-world47-seed2026092208-public-001/nodes/t01-r0-flat-original-n0.json), indices2,4,5,6 (parts, parts, assembly, finish). Call IDs use the same indices, e.g. public `t01-r0-flat-original-c005`.

Important other-repeat check: corrected **does** complete629/repeat1, after49 calls and3 action errors; public completes it in7 calls without errors. This is a stochastic route/completion difference, not proof that corrected cannot do the task.

**2. “Looking up the right recipe is not enough if the next action keeps violating it.”**

On47/repeat0, public reads that `a1_i1` requires only `raw_a3`, yet repeatedly includes `raw_a1`. Across the episode there are36 explicit “extra raw_a1” rejections. It sometimes queries the recipes again but returns to the same bad ingredient combinations. Corrected also errs initially, then changes its ingredient quantity and proceeds through final assembly. This is the counterexample to claiming that public discovery alone solves execution.

Native pointers: public [node history](/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/textcraft-world47-seed2026092208-public-001/nodes/t02-r0-flat-original-n0.json), index4 observes the recipe,6 violates it,28 observes it again, later entries repeat the extra ingredient,66 finishes without target; corrected [node history](/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/textcraft-world47-quantity-corrected-seed2208-quantity_corrected_original-004/nodes/t02-r0-flat-original-n0.json), indices3→4 correct a rejected quantity,14 performs final assembly,15 finishes.

Other-repeat check: **both fail47/repeat1**. Corrected reaches96 calls while repeatedly querying a nonexistent item; public finishes after27 calls with15 action errors. Do not present corrected's one successful route as reliable repair behavior.

## Full-panel premature-finish check

This static check includes **all 16 observed slots per arm**, not just discordant examples. A premature finish is an explicit `finish` while final minus initial target inventory remains below the public goal.

| Recorded outcome or state | Corrected | Public |
|---|---:|---:|
| Native successes | 4/16 | 6/16 |
| Premature finish with unmet goal | 11/16 | 9/16 |
| Non-finish cap with unmet goal | 1/16 | 1/16 |
| Premature finishes: sufficient immediate ingredients for remaining target | 3/11 | 0/9 |
| Premature finishes: observed target recipe, insufficient immediate ingredients | 4/11 | 9/9 |
| Premature finishes: no observed concrete target recipe | 4/11 | 0/9 |
| Overlapping flag: some target produced, but below required quantity | 2/11 | 0/9 |

The three corrected cases with sufficient immediate inputs are both repeats of325 (made2, goal3) and629/repeat0 (made the parts but not the target). All nine public premature finishes still lack at least one immediate ingredient. Corrected's non-finish is47/repeat1 at the96-call cap; public's is `t06-r0` at the context cap.

Immediate-input sufficiency uses only the last target recipe actually returned in public query history, its batch yield, and the saved final inventory: required batches are `ceil(remaining target / batch yield)`. The receipt records each slot, deficit, public recipe-history index, requirements, available quantities and source hashes. There is no hidden-recipe lookup, recursive demand planning, counterfactual propagation or new native rollout.

This distinguishes an observable **goal-completion check** from immediate **recipe binding**. Rejecting a premature finish could provide useful feedback, but would not itself find missing recipes, assemble missing parts or stop execution loops. These counts are **not predicted rescued successes**.

## What this supports—and does not

Plain-language takeaway: **finding the needed information, applying it correctly, and stopping only after the whole goal is met are separate failure points.** This panel contains examples of all three. Lower tool-error counts alone are an insufficient progress metric.

These are observed routes, not interventions on an individual error or proofs of teacher-induced mechanisms. There are no chain-of-thought claims, faithful-reasoning claims, recursion gains or novelty claims here. The useful research question is how a learned procedure maintains an unfinished goal while converting public feedback into subsequent actions—not simply how to reduce invalid JSON or repeat recipe lookups.

Provenance: [completed native comparison](/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/analysis-textcraft-world47-quantity-corrected-seed2208-004.json). The adjacent [compact receipt](WORLD47-CONTRASTS-20260924.json) pins the report, plans, episodes and node histories, plus ordered hashes of every call in the inspected traces. All consumed native artifacts were checked against the completed report's hashes. No model calls, rescoring or raw-artifact edits were performed.
