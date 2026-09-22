# Public stopping is ready; an inventory-only cache is redundant

CPU assessment on completed source044 and source049, before source052 outcomes.
No GPU run, controller change, or new success measurement is accepted here.

## Smallest stopping baseline

The existing public frame exposes target quantities, inventory at task start,
and exact current inventory on **every** request. A host can check
`all(current[item] - initial[item] >= target[item])`, treating absent items as
zero, without consulting recipes, gold actions, task depth, or the native score.
When true, it executes the existing strict
`{"action":"finish","message":"done"}` through `Frame.apply`. It does not
silently mark the episode successful; ordinary native evaluation follows.

Check this at the beginning of each action slot, before tokenization/model
inference, but after checking owner deadline and remaining action budget. Charge
one logical/environment action for host finish against the same 96-action limit;
charge zero generated tokens and zero physical model calls. Keep those three
counts separate. Never emit a finish after exhausting the action limit or owner
deadline. The final permitted craft therefore still needs one remaining action
to finish. An exhausted model-token allowance need not prohibit a host finish,
which consumes no model tokens; declare this rather than quietly counting it as
another model response. Retain a host-action receipt with the public quantities,
predicate value and native feedback. Do not fabricate native model receipts.

This is a controller with an explicit stopping rule, not the same learned-action
policy and not evidence that the model learned termination. It preserves the
benchmark's required production and actual finish operation. The existing copied
state audit already verified agreement with native finish on the examined states;
it is not necessary to repeat that full audit. Source049 reached this public
condition in three of 16 episodes but emitted no finish. These are counterfactual
opportunities, not a newly measured 3/16 policy score.

Implementation seam is small: one public-only predicate and one host-action branch
before `budget.reserve()` in an additive flat collector, with independent logical
action versus physical-call accounting. Four focused fixtures suffice: existing
stock does not count as new production; every target must be met; a finish uses
one remaining action but no model call; a real pinned native finish agrees on a
copied completed state. Preserve source052 unchanged.

## What a state cache can and cannot test

Caching current inventory would add no information: it is already present before
every action, even during repeated exhausted-ingredient errors. Source049 t03/r0
queries both root `m0_i2_20` and intermediate `m0_i1`, successfully crafts the
intermediate at call003, then repeats that craft from call004 with zero `m0_ore`.
The root recipe and the current stock were both available. Source049 t02/r0
queries its root and all three immediate prerequisites, successfully crafts all
three at calls004–006, then crafts an already-sufficient prerequisite again at
call007. These are concrete failures to use available facts; they do not prove a
memory or attention mechanism.

If source052 leaves these errors, the smallest informative cache comparison is:

- **Public-stop:** unchanged selected flat prompt/profile, host stopping above.
- **Public-stop + queried-recipe cache:** identical except a deterministic trailing
  JSON field maps each previously queried item to its returned ingredient maps
  and result counts, sorted by item ID. Preserve recipe alternative order. Include
  no unqueried recipe, depth, inferred next action, computed demand plan, or gold.

The cache contains static recipe facts only. Do not retain an old query's
`in_inventory` as current stock or interpret `can_craft` as current action
feasibility. Both policies retain the complete original action/feedback history,
exact current inventory, strict parser, 96 total action slots, 8192 output tokens,
256 per model call, 8192 context and no truncation. The cache may cause earlier
context exhaustion; record its actual input cost and cap effects. This isolates
an explicit repeated-fact presentation aid, **not** compression or memory access.
Replacing history by a cache would simultaneously change retention, context
budget and recovery evidence, so defer that larger packaged intervention.

Use all eight existing tasks and both original seeds, with the same fixed model
checkpoint in both arms: 32 new episodes, maximum 3072 model calls (usually fewer),
with a prospective 90-minute global cap and missing outcomes left unknown. Choose
the checkpoint/profile explicitly after the accepted source052 readout; do not
mix weights between these two arms. With the base reminder profile, source049's
16 episodes cost 1328 calls and 31.1 native minutes, so two new arms could require
about an hour, not a negligible GPU add-on. Host-stop alone is cheaper; its most
immediate value is controller accounting, because its observed trace opportunity
is already known. Do not launch either merely to reproduce that counterfactual.

Primary comparisons should separate actual native success, first public goal
attainment, model-chosen versus host finish, rejected craft actions and repeated
queries. Report all tokens, physical model calls, logical/environment actions and
time. An improvement in native success only from host finish is termination
engineering; improved goal attainment under the cache would support a useful
presentation intervention, not novel planning or recursive decomposition.

## Decision and provenance

Wait for the fixed source052 result. If production succeeds but finish fails,
public stopping is the cheapest useful harness baseline. If discovery or quantity
planning still fails, the CPU-qualified public-discovery teacher is a stronger
training comparison than another inventory reminder. If recipes were queried but
not used, the two-arm cache comparison above is a limited, interpretable option.
It should not displace a ready more informative learning experiment.

Evidence: `TEXTCRAFT-INSTRUCTION-FINDINGS.md`,
`analysis-textcraft-instruction-control-001.json` SHA256
`615740e685636c2f870cb4f3b2fcd601a4935f91eba9cd6d57f0efbd81e41cd6`,
and the already completed `analysis-textcraft-finish-opportunity-001.json` SHA256
`32b6944d3707057b90c10b88f786d52431a9ee820b28437372314ac96bd92108`.
The two illustrations were read from immutable source049 nodes
`t03-r0-flat-n0.json` and `t02-r0-flat-n0.json`, not selected from source052.
