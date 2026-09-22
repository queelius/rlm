---
status: completed_first_update_diagnostic_parent_run_later_failed
evidence_date: 2026-09-22
question: Did the first RL update change the policy, and which saved actions changed likelihood?
claim_strength: fixed_prefix_likelihood_movement_not_task_improvement
---

# First terminal-RL update: measurable movement, not yet better task performance

The first actual training update changed the adapter and improved the saved-batch
advantage-weighted likelihood objective. It did not simply increase every
positively credited action. Some failed crafts became more likely, but the
preidentified wrong-quantity example became **less** likely while its subsequent
correct craft became more likely. There is no held-out RL result here.

**Later status:** the parent run stopped before its second update because a
generation-versus-replay probability check exceeded its declared tolerance.
The saved first-update measurements below remain diagnostic observations,
not a usable final endpoint or a completed RL-versus-SFT comparison. See the
[numerical follow-up](LITERATURE-NUMERICAL-REPLAY-20260922.md).

This check asks: given exactly the same saved context, how likely is the same
recorded command before and after training? It does not yet ask the updated
model to solve a fresh task and measure whether it succeeds more often.

The unchanged first TRAIN batch contains8 tasks×4 episodes,770 calls and24,009
emitted tokens. Only3 mixed-reward tasks contribute gradients:12 episodes,
388 calls,12,074 tokens. Of those388 calls,245 moved in the direction of their
advantage and5 were numerically unchanged. These are dependent observations
from3 tasks, not388 independent examples of learning success.

| Saved action class | Calls | Likelihood increased | Sum sequence Δlogp |
|---|---:|---:|---:|
| Positive-credit failed craft |136|94|+10.9615|
| Positive-credit craft without explicit error |87|66|+6.8261|
| Positive-credit query |68|26|−8.9082|
| Negative-credit finish |3|0|−3.3188|

“Without explicit error” is not a claim that an action was necessary or useful.
The sum of log-probability changes across different calls is not a percentage
change in success probability, or the probability of one common event.

## Keep the preidentified counterexample

Both actions in `t03-r0-flat` received+1/3 trajectory advantage. The pair was
identified before observing the update, not selected afterward to fit a story:

- `c009`: requested three `c6_i2` and three `c1_i1_11` for one `c3_i3_13`.
  Native feedback required two, not three. Sequence Δlogp=−0.03609254.
- `c010`: used two of each and successfully crafted one `c3_i3_13`.
  Sequence Δlogp=+0.13642323.

Thus a positive credit coefficient is not a guaranteed likelihood increase.
Shared parameter updates are consistent here with reducing a local mistake
despite its positive trajectory credit. This does not identify which other
gradients caused the change, establish causal action value, or demonstrate that
the next on-policy episode will recover more reliably.

## The query decrease is mostly repeated lookups, not root lookup suppression

A small public-history-only census classifies all96 credited queries; every
query requests exactly one item. It uses public root targets, initial inventory,
earlier returned recipe ingredients and prior query actions—no hidden recipes.
“First publicly introduced lookup” is a visibility label, **not “needed.”**

| Positive-credit query subtype | Calls | Increased | Sum Δlogp |
|---|---:|---:|---:|
| First root-target lookup |9|9|+0.0345|
| First other publicly introduced item |39|17|−0.8712|
| Repeated lookup |16|0|−6.4365|
| First initial-inventory item |3|0|−1.1143|
| First name not previously introduced by public evidence |1|0|−0.5207|

The16 repeated positive-credit queries account for about72% of the net−8.9082
query logp sum. All12 first-root queries increase slightly, including the3 in
negative-credit episodes. Among negative-credit queries,11 repeats contribute
−5.5510 of the total−6.7476. These observations narrow the aggregate description;
they do not prove repeated queries are useless or that discovery improved.
“First root-target lookup” means the first query about that target in the
episode, not necessarily the first action of the episode.

## Technical checks and limits

Independently read the actual BEFORE/AFTER arrays:388 matched credited calls,
245 moving with advantage, sum A×Δlogp=+23.01497845, divided by the fixed32
episode denominator gives+0.71921808 in the frozen-batch surrogate. Each of the
3 active task aggregates is positive. This is not an on-policy reward curve.

- Actual saved Adam scalar step=1; adapter L2 delta=0.08082993.
- Pre-clipping gradient norm=28.92340088, threshold1. The implied gradient
  multiplier≈0.03457408 is **not the Adam step size**: moment normalization and
  parameterwise scaling follow clipping. No effective learning-rate claim.
- Training/evaluation teacher-forced replay max/mean gap=0 on12,074 credited
  tokens in this run. Separately, the fresh qualification's generation/replay
  max/mean gap was0.18835354/0.01345590. Historical063 generation logps were not
  recorded; their BEFORE values were recomputed under the unchanged warm weights.
- AFTER logps for382 zero-advantage calls are omitted, not measured unchanged.
  This includes both all-success and all-failure groups. No full-policy KL,
  omitted-action movement, new rollout success or held-out gain is measured.
- The comparison teacher-forces the same emitted response IDs on the same saved
  prefixes atT0.5. It does not follow new trajectories under the updated policy.
  Adam/clipping, gradient sharing, state differences and3-task dependence limit
  causal interpretation. Do not change accepted065v2 based on this diagnostic.

Verified report `R/analysis-textcraft-update-001.json`, SHA256
`e19bd51cc230e636899f0690c9523bdd07d146947338eb2685aeef5179b4f151`.
Actual checkpoint `R/textcraft-terminal-rl-002/boundaries/sample-0001/checkpoint-0001`,
COMMIT SHA256
`063785496989e1cc16b46bd701a92b71d6c8c50aac1a5fde8c2ed27b66f518c7`.
Native examples live under `R/textcraft-train-readiness-001/calls/` and
`nodes/t03-r0-flat-n0.json`.

Optional query census: `R/analysis-textcraft-query-movement-001.json`; standalone
CPU driver `R/analysis-source-textcraft-query-movement-001/audit.py` pins the
first-update report and records every native node hash. Reproduce with
`TRAINPY <driver> --report R/<new-query-replay>.json`; existing outputs cannot
be overwritten. This note and the optional census add no GPU/model calls.

Main independently reran the original query driver: report002 is byte-identical
to report001, and all32 node hashes match the native receipts in the committed
training BATCH. A formatted copy is versioned as
[`audit_textcraft_query_movement.py`](audit_textcraft_query_movement.py); its
results were compared to the original with only the source-code hash allowed
to differ. The original sealed source and reports remain unchanged.

R=`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.
