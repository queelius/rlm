---
status: completed
prepared_date: 2026-09-22
question: Does revised SFT improve over the untrained base on the same fresh goals?
collection_cap_minutes: 120
per_episode_limits_changed: false
---

# Next control: compare with the base model on the same new goals

**Completion update, September 22 at 14:45 UTC:** all 32 base attempts completed
and all 96 three-arm outcomes were verified. Base/old/revised training solve
6/32, 1/32 and 15/32. Revised versus base has ten paired gains and one loss.
See the [completed findings and limitations](TEXTCRAFT-FRESH-BASE-FINDINGS.md).
The prospective rationale and launch record below are preserved as history.

The revised demonstrations outperform the earlier demonstrations on the fresh
panel, 15/32 versus 1/32. We still need to know whether they improve over the
base model itself. Otherwise, the result could mainly show that the earlier
training damaged a useful existing ability.

The prepared comparison evaluates the unadapted base on exactly the same 16
goals, two sampling seeds per goal, prompts and per-attempt budgets. No new tasks
or favorable seeds are selected. The earlier historical base comparison used
different, previously examined goals and cannot fill this gap.

The total collection cap is two hours, rather than the one hour given to each
teacher-trained model. This is **not extra reasoning per task**: each attempt
still has the same call, generated-token, context and per-request time limits.
Both teacher arms already completed all their attempts within one hour. The
larger batch-level cap allows time to collect all the base model's attempts,
which may be slower; it was declared before collecting any base outcomes.

The analyzer reports actual per-request time limits, any deadline reductions,
and possible time-truncated outputs. Missing or unavailable base outcomes remain
unknown, with full-panel bounds and no complete-panel effect or uncertainty
interval. Any actual timing asymmetry must be considered before claiming a
matched policy comparison. No rerun or checkpoint is selected by its score.

Interpretation:

- If revised SFT beats the base, the evidence for a useful learned procedure is stronger.
- If they are similar, avoiding harmful demonstrations may explain much of the gain.
- If the base is better, the revised training still leaves a regression to explain.

All three are useful research outcomes. This remains one base model, one SFT
seed and one recipe world; the overlap caveats in the
[fresh-goal scope audit](TEXTCRAFT-FRESH-SCOPE.md) still apply.

Source067 is additive: the old source066 proposal and running RL/control sources
are untouched. Main reviewed the changes and ran all three sealed focused tests
(passed in 5.32 seconds). At that preparation checkpoint, GPU acceptance and
launch were pending; the subsequent launch is recorded below.

R=`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.
Commands, pins and analysis arguments are in
`R/TEXTCRAFT-FRESH-BASE-120MIN-PROPOSED-001.json`, SHA256
`1c46af8461a7555a913748ef785ba46f0d21c9e1d315da8290a499d3a52028c7`.
Source manifest: `R/source-067-textcraft-fresh-base-120min/SOURCE.json`, SHA256
`ce4ffd5243203417223d06490caf002664283bc3cb0aa86734efe4c4a62b774d`.
Prepared output: `R/textcraft-fresh-base-002/PLAN.json`, SHA256
`f3d451e3f271d3a52cbd600e74e9493f8192da9bcb9adc77697b531f13e5f178`.

## Launch update, September 22 at 13:35 UTC

The control is now running. The preceding RL job stopped at a numerical
consistency check, so its dependent comparisons could not proceed. This
independent comparison uses no RL checkpoint and can answer its original
question while the numerical issue is investigated. Its first actual model
response returned in 1.54 seconds. No base-model score is reported until the
planned collection and native analysis finish.

Accepted queue receipt: `R/INDEPENDENT-TRAINING-QUEUE-009.json`, SHA256
`5e5a67259ba8eb7eae24b0d93ce930da2f04e6263cb677dbb80a4db7df76ad73`.
The queue also runs the prepared analyzer after collection.
