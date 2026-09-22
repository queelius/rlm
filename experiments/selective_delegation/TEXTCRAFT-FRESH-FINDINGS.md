---
status: completed_exploratory_comparison
evidence_cutoff_utc: 2026-09-22T12:54:00Z
question: Does teaching public information gathering help with new crafting goals?
primary_panel: 16_frozen_goal_items_two_samples_each_same_recipe_world
claim_strength: teacher_package_transfer_not_order_only_or_recursion
report_sha256: 574b545bfb6bd7b00c3e83758d7bc4007464afca86ca9a958d4679f96995e480
---

# Revised demonstrations help with new goals, but longer plans remain difficult

The model trained on the revised demonstrations solves **15 of 32 attempts**,
compared with **1 of 32** for the earlier demonstrations. The same 16 goals were
sampled twice under each model. There are 14 paired gains, no losses, and 18 ties;
10 of the 16 goals improve on at least one sample.

The revised teacher shows how to discover the next ingredient from information
the student can actually obtain. For example, it asks for the final product's
recipe before using that reply to decide which intermediate recipe to inspect.
The earlier teacher could select an intermediate item using its hidden plan.
This result supports the revised teaching procedure as a package. It does not
isolate query order from the changed histories and exposure to information.

Both models are the same 4B base model with LoRA training on 32 tasks, 366 action
examples and 23 updates. The final checkpoints were fixed before this evaluation.
Neither model uses helper agents in this comparison: it is an action-controller
experiment, not evidence that recursion or a new RLM architecture helped.

## The improvement is clear on this panel, not universal

The paired improvement is **43.75 percentage points**, with an exploratory 95%
task-cluster bootstrap interval of **25 to 62.5 points**. Resampling keeps the
two samples of each goal together. It does not make the common recipe world
or shared prerequisites independent. This is one training seed and one world.

The predefined depth groups show an important limitation:

| Longest recipe chain | Goals | Earlier demonstrations: successes | Revised demonstrations: successes |
|---|---:|---:|---:|
| Two crafting levels |4|1/8|7/8|
| Three crafting levels |4|0/8|5/8|
| Four crafting levels |8|0/16|3/16|

These groups differ in recipes, quantities and branching as well as depth, so
the table is descriptive, not a controlled estimate of depth's causal effect.
On the deepest group, five goals fail on both samples and the other three
succeed only once each. Improved demonstrations have not made longer plans
reliable.

## What “new goals” means

Selection was frozen before collecting policy outcomes. All 16 are new root
goals under that rule, but three final items had already appeared as intermediate
products in both training datasets. On those three goals the comparison is 0/6
versus 4/6. On the remaining 13, whose exact identifiers are absent from both
datasets' prompts and targets, it is **1/26 versus 11/26**, with 10 gains and no
losses. This additional diagnostic does not replace the full 16-goal primary test.

The [input-only scope audit](TEXTCRAFT-FRESH-SCOPE.md) defined this distinction
while collection was running, before opening the completed paired result.
Fourteen of the full 16 goals still share some prerequisites with the 32 training
tasks. This is transfer within a familiar type of crafting world, not transfer
to an unrelated domain or a claim about the base model's pretraining exposure.

## Better completion is not cheaper or cleaner execution

| Recorded quantity | Earlier demonstrations | Revised demonstrations |
|---|---:|---:|
| Model calls |1,340|1,206|
| Recipe queries |1,037|378|
| Crafting attempts |259|797|
| Rejected crafting attempts |119|550|
| Generated tokens |29,700|37,486|
| Summed model-call time |42.7 minutes|51.8 minutes|

The revised model queries less and attempts more crafting, but many commands
are rejected. It uses 10% fewer calls while taking about 22% more summed
model-call time. The higher raw error count partly reflects attempting much
more crafting; the rejected fraction also rises, from 119/259 to 550/797.
Neither quantity establishes the cause of the success improvement.

All 64 attempts have verified outcomes and zero transport failures. The earlier
model finishes 27 attempts, reaches the call limit three times and the context
limit twice. The revised model finishes 29 and reaches the context limit three
times. A finish action alone does not count as success: the environment checks
whether the requested inventory was actually made. No missing outcome was
silently counted as a failure.

## What this changes

Together with the [changed-recipe-world result](TEXTCRAFT-CHANGED-WORLD-FINDINGS.md),
this strengthens the case for studying **how demonstrations teach information
gathering and execution**, rather than focusing only on final-answer examples.
It remains a promising empirical lead, not an established novel method; related
imitation-learning work already studies teacher/student information mismatch.

The residual errors motivate the already accepted terminal-reward RL pilot and
its extra-SFT control. That comparison asks whether learning from the model's
own outcomes improves execution beyond more demonstrations. It is now running;
no RL improvement is claimed here. The separate [recipe-binding proposal](TEXTCRAFT-PUBLIC-RECIPE-BINDING-IDEA.md)
would test whether the harness should fill in deterministic ingredient arguments
while leaving the choice of subgoals and quantities to the model. It is still a
proposal, not a measured architecture gain.

## Evidence and reproduction

R=`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.
Immutable native report: `R/analysis-textcraft-fresh-001.json`, SHA256
`574b545bfb6bd7b00c3e83758d7bc4007464afca86ca9a958d4679f96995e480`.
Both 32-episode arms were replayed against unchanged native environment rules;
the automatic analysis exited 0 with no unresolved starts or orphaned calls.

The input tasks, seeds, checkpoints, caps and collection sources are pinned in
`R/INDEPENDENT-TRAINING-QUEUE-007.json`. Original outputs are
`R/textcraft-fresh-privileged-001` and `R/textcraft-fresh-public-001`.
Owner wall times were 2,590.94 and 3,138.24 seconds, respectively; both terminated
normally within their one-hour caps. The source is
`R/source-textcraft-fresh-readout-001/analyze_textcraft_fresh.py`, with CLI
arguments `--privileged-output`, `--public-output` and a new `--report` path.
The report includes task/repeat pairs, depth groups, exact receipt hashes,
token counts, stopping reasons and the 20,000-draw bootstrap seed 2026092206.
