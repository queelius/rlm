---
status: completed_cpu_diagnostic
evidence_date: 2026-09-22
question: Which actions receive the first batch's terminal-reward credit?
claim_strength: objective_census_not_measured_learning
---

# First063 batch: terminal credit also lands on failed actions

## A concrete example in plain language

One successful attempt contains this sequence. To make it readable, the table
renames the two intermediate items **A** and **B** and omits unrelated steps;
the quantities and outcomes are from the saved experiment.

| Selected event | What happened |
|---|---|
| The model asks for the final item's recipe. | The reply says that one item needs two A and two B. |
| The model requests one item using three A and three B. | The environment rejects the request because the quantities are wrong. |
| The model immediately retries using two A and two B. | The environment successfully makes one item. |
| The model continues and finishes. | It ultimately has three final items, meeting the goal of at least two. |

For this attempt, **both the rejected command and the corrected command get
the same positive weight in the training objective**. The attempt succeeded,
while one of the other three attempts on this task failed; the resulting weight
is +1/3 for every emitted action token in this attempt. The objective does not
separately label the first command as a mistake and the second as its correction.
That is not a proof that the failed command becomes more likely after training:
the combined gradient from all examples determines the actual update.

This also illustrates the [recipe-binding idea](TEXTCRAFT-PUBLIC-RECIPE-BINDING-IDEA.md):
the correct recipe was already public, and the model retained the right item
and output quantity. Filling the ingredient quantities from that known recipe
could avoid this particular rejected command. It would not decide which item
to make, how many are needed for the whole task, or whether to delegate.

Trace: TRAIN task1680, episode `t03-r0-flat`, consecutive calls `c009` and `c010`.
The original final item is `c3_i3_13`; A is `c6_i2` and B is `c1_i1_11`.
Under `R/textcraft-train-readiness-001`, node
`nodes/t03-r0-flat-n0.json` has SHA256
`6f9aeaf9262b8a1c13c33711f536710c1668312341897f16ab0afd4eabad09fc`;
episode `episodes/t03-r0-flat.json` has SHA256
`e23bd9108534434343b7f7faae296ec04dc118251034ca0698d4e9fe847c1122`.
This is an illustrative recorded recovery, not a new estimate of its frequency.

## Full-batch counts and limitations

CPU census of all32 completed TRAIN episodes/8 tasks/770 calls/24,009 emitted
tokens. Native successes25/32; four all-success groups, one all-failure group,
three mixed groups with3/4 successes each. Existing native audit002 and every
selected call/node/episode hash were checked; no new model calls or reward edits.

Only the three mixed groups receive nonzero RLOO credit:12 episodes,388 calls,
12,074 tokens. Their successful episodes have advantage+1/3 and unsuccessful
episodes−1. The other20 episodes/382 calls/11,935 tokens retain zero advantage
in the complete32-episode denominator.

| Trajectory credit | Episodes | Calls | Emitted tokens | Sum advantage×tokens |
|---|---:|---:|---:|---:|
| Positive |9|300|9,390|+3,130|
| Negative |3|88|2,684|−2,684|
| Zero |20|382|11,935|0|

Positive trajectories average33.3 calls/1,043.3 tokens; negative29.3/894.7.
Successful mixed-group paths range13–72 calls, unsuccessful26–32. Thus success
is not uniformly the shorter action sequence. All32 terminate with a finish;
observed native success, not a finish string alone, defines reward.

| Action feedback | Positive calls/tokens | Negative calls/tokens | Zero calls/tokens |
|---|---:|---:|---:|
| Failed native craft |136 /5,149|42 /1,727|135 /6,640|
| Craft without explicit error |87 /3,052|15 /475|81 /2,803|
| Query without explicit error |68 /1,099|28 /452|146 /2,292|
| Finish without explicit error |9 /90|3 /30|20 /200|

There are no additional schema/rejected-action categories in this completed
batch. “Without explicit error” does **not** establish necessity or correctness
for the root goal. In particular, finish can be syntactically successful while
the native root checker returns0.

Failed crafts account for45.3% of positively credited calls and54.8% of their
tokens, occurring in all nine positively credited episodes. Their signed
weighted-token sum is+1,716.33. Negative credit reaches46 actions/957 tokens
without explicit error (15 crafts,28 queries,3 finishes), as well as42 failed
crafts/1,727 tokens. The positive and negative failed-craft scalar totals are
close (+1,716.33 versus−1,727), **not evidence their gradients cancel**: prompts,
targets and parameter derivatives differ. Total absolute weighted-token mass
is5,814; signed mass+446. Divide these coefficients by32 for the stated loss.

Concrete native traces under `R/textcraft-train-readiness-001/`:

- `calls/t03-r0-flat-c007.json`, matched node `nodes/t03-r0-flat-n0.json`:
  crafting `c3_i3_13` requested two `c6_i2` when feedback required four. This
 41-token failed craft still receives+1/3 because the full episode later succeeds.
- `calls/t07-r0-flat-c008.json`, node `nodes/t07-r0-flat-n0.json`: successfully
  crafted four `a1_i1_21`; this30-token action receives−1 because the episode's
  eventual root outcome is failure.

This is a mechanism diagnostic of whole-trajectory credit, not evidence that
065 RL worsened behavior or that an alternative reward would improve it.
Positive advantage does not guarantee a probability increase after shared
gradients, Adam and global clipping. Gradient norms/clipping and actual parameter
movement are not measured here. No gates, filters, process rewards, or changes
to accepted008 are introduced.

Main independently reran the sealed analyzer. Report002 is byte-identical to
report001. The [recent literature note](LITERATURE-CREDIT-AND-STATE-20260922.md)
places this diagnostic beside existing local-credit methods; it is not a novelty claim.

Reproducibility: standalone `analyze_textcraft_credit.py`; one focused fixture
passes from `R/analysis-source-textcraft-credit-001` (manifest SHA256
`7560c951558fcbda9b3cca7829bc739669dcebb26ec708f3aca7793e92a1b3c4`).
Immutable report `R/analysis-textcraft-credit-001.json` SHA256
`a8ecf22311fb59491be9b7099e7a51d9ee529334b898826cee0707d4a07696fb`;
sibling Markdown summarizes counts. JSON contains all32 per-task episode
calls/tokens/reward/advantage and every770 classified call, with source pointers.

```text
CUDA_VISIBLE_DEVICES='' TRAINPY R/analysis-source-textcraft-credit-001/analyze_textcraft_credit.py \
  --output R/textcraft-train-readiness-001 \
  --native-report R/analysis-textcraft-train-readiness-002.json \
  --report R/<new-replay-report>.json
```

R is `/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.
