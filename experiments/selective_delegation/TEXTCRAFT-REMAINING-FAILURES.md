---
status: descriptive_trace_analysis
question: Which execution errors remain after information gathering improves?
failed_episodes: 6
task_parents: 3
causal_status: mechanisms_to_test_not_established_causes
gpu_acceptance: none
---

# Remaining public-teacher failures: public facts versus execution

This audit covers all six original-prompt failures in the completed public-information-teacher
readout: the two seeds of each of `textcraft_synth.val.567`, `.207`, and `.352`.  It reads only
their saved public action/feedback histories.  Recipe-world inspection is used only to audit the
outcome, never proposed as a policy input.

## What failed first

| Parent / seed | First consequential error (call ID) | Were relevant public facts already available? | Interpretation |
| --- | --- | --- | --- |
| `.567` / r0 | Tried root `c3_i4` before holding `c2_i3_12`, `c0_i3`, or `c9_i3` (`c013`). | Root recipe was returned; the latter two prerequisite recipes had not yet been queried. | Discovery remains needed, but a public inventory check would reject this craft. Later quantity and resource errors compound it. |
| `.567` / r1 | Asked `c1_i1` to consume itself as an extra ingredient (`c014`). | Yes: its exact recipe had been returned at `c006`. | Exact recipe validation would reject this immediately; later it exhausts `c6_ore` and repeatedly retries an infeasible craft. |
| `.207` / r0 | Tried `m1_i3_11` before obtaining `m0_i2_10` (`c006`). | The missing subrecipe had not yet been queried. | Initial discovery is genuinely incomplete. By `c019`, the public root recipe and public inventory make one root craft feasible, but it finished at `c020`. |
| `.207` / r1 | Supplied 4 rather than 1 `m1_i1` for `m0_i2_10` (`c005`). | Yes: both recipe facts were returned at `c003–c004`; inventory was public. | A quantity-feasibility display or ledger could flag the mismatch. The run then repeats wrong scales and attempts branches without `m8_i1`; it does not discover/craft that branch before finishing. |
| `.352` / r0 | Tried root `c0_i4_20` with only half the needed `c7_i3_17` (`c010`). | Yes: the root and all queried branch recipes were already public at `c000–c009`. | Strongest demand-accounting case: it then doubles a malformed ingredient list rather than calculating required branch quantities. |
| `.352` / r1 | Tried `c4_i3` before holding `c2_i1`/`c2_i2` (`c010`). | Yes: the exact recipe had been returned at `c003`; inventory was public. | It later makes several exact-ingredient and quantity errors, then repeatedly retries the same infeasible `c4_i3` craft (`c028–c059`). |

The IDs above are suffixes of the public run's saved calls, e.g.
`textcraft-public-discovery-readout-001/calls/t07-r1-flat-original-c028.json`.
The root and the initially attempted recipes are public facts, so this is not a claim that an
unobserved world oracle is necessary.  For `.207` / r0, the public root recipe requires one
`m1_i3_11` and one `m3_i2_13`; the saved public final inventory has one and two respectively.
Thus “one root craft feasible” is an offline state-transition counterfactual, not an observed
rescue.  Conversely, an audit finding that a recipe was feasible does not establish that the model
had correctly represented every needed dependency.

## Important interface semantics

`get_info.can_craft` means **a recipe exists**, not that the current inventory can execute one.
The pinned TextCraft environment constructs it as `recipe_db.can_craft(item)` in
`plugins/textcraft/platoon/textcraft/env.py:243` (file SHA-256
`c58ffad58e535d91def7291db148fd8f6433a8b3a915642f634a5b75d6d587d0`); the recipe database defines
that predicate as `item in self.recipes` in `synth_recipe_generator.py:406–408` (SHA-256
`1c33e0a6f61759eb3a8eb33b35f0b88361155525f5f575885b53acbd011efb0e`), repository commit
`d9c5857d3a0a056ebc9b047241a2a0c9515aafbe`.

The saved public fixture `t06-r0-flat-original` demonstrates the distinction without invoking an
oracle: call `c000` reports `m1_i4` as `can_craft: true` while `in_inventory: 0`; its returned
recipe requires two other produced items.  Consequently, the simplest prospective clarification
is a prompt sentence: **“`can_craft` means a recipe exists; use the displayed inventory and recipe
ingredients to determine whether it can be executed now.”**  This changes interpretation, not
information, and should be compared as its own prompt intervention before a derived ledger.  The
present trajectories do not show that this semantic ambiguity caused their errors.

## Contrasts

The public teacher did succeed on other multi-step parents despite some rejected crafts.  For
example, `textcraft_synth.val.435` / r1 queried its target and intermediate recipes, accumulated
the needed items, then produced the root in calls `c028–c029` and finished (`t05-r1-flat-original`).
That says a public recipe-and-inventory interface can support successful execution; it does not
isolate why that trajectory recovered.

The sole paired public loss is `.207` / r1: the earlier privileged teacher succeeded in 40 calls,
while public teacher failed in 65.  The privileged trajectory also made rejected attempts before
recovering to craft the needed `m1_i2_11`, `m0_i2_10`, `m1_i3_11`, `m3_i2_13`, and root.  It is
therefore a caution against attributing the aggregate gain simply to target-first querying or to a
particular teacher ordering.

## What a harness could and could not fix

1. **Smallest targeted test: a public recipe-fact memo.**  Keep the same query/craft budget and
   action model.  Render only recipes returned by earlier `get_info` calls, alongside the already
   explicit current inventory, in a compact structured memo.  This is a salience/retention
   intervention over the existing history, not access to new facts, a new memory mechanism, or a
   feasibility calculation.  It adds prompt tokens; a same-token unstructured-history control
   would be needed to attribute a result to the structure itself.
2. **Separate derived-computation arm: public feasibility or quantity-demand ledger.**  Host code
   could calculate feasibility or an unmet dependency demand using only target demand, current
   inventory, and already-returned recipes.  That is a new derived computation, not a recipe memo.
   It must not traverse unqueried recipes, expose hidden facts, choose an action, or silently repair
   a model action.  A displayed *recommended* next legal craft would be host planning and should
   be labelled and tested as such, not presented as a pure display.  Charge derived fields' tokens
   to every arm.  This targets `.352` and the known-recipe portions of `.567`/`.207`.
3. **Do not add another craft validator as an intervention.**  The native bridge already validates
   and rejects illegal crafts and charges the turn.  A duplicate rejecter with the same feedback
   changes nothing; a different feedback/retry mechanism would be a distinct protocol intervention
   and must be evaluated as one, rather than being described as a reasoning improvement.

The memo arm distinguishes presentation/retention from lack of prerequisite discovery.  If it
removes malformed/repeated actions but not completion failures, a separately declared derived
demand calculation may be worth testing.  If it does not improve over a future same-token
unstructured-history control, retire the structured-memo explanation.  Neither result would
establish a generic memory or RLM novelty claim.

## Limits and provenance

All six failures are from three exposed validation parents with two correlated seeds each; they
are mechanism examples, not prevalence estimates.  The source readout report is
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/analysis-textcraft-public-teacher-001.json`
(`sha256 f80a4000e9a2ec2cbae14f1ccf52f54f69a569d624ed3c4deed6cf8dd260a582`).
Saved node histories under `textcraft-public-discovery-readout-001/nodes/` provide the public
actions and feedback; the paired old trajectory is in `textcraft-trained-readout-001/nodes/`.
