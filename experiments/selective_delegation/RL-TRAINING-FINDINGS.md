# Four real planner-RL updates; no demonstrated reward trend

Completed `rl-planner-001`: four fresh RLOO optimizer updates from SFT checkpoint
48, on the same 16 training parents with four sampled plans per parent/update.
All checkpoints were committed and the run ended normally. This establishes a
working reward-driven planner update, not improved held-out performance.

Crucially, each batch's reward is measured **before its update**: batches 1–4
evaluate SFT, RL checkpoint 1, RL checkpoint 2, and RL checkpoint 3 respectively.
They do not evaluate checkpoint 4. Different sampling seeds across batches and
reused training parents prevent a causal/generalization interpretation of this
short reward sequence. Its first and last values are identical.

## Updates and admission

| Update | Pre-update EM / 64 | Qualifying groups / 16 | Groups with fully scored reward variation | Pre-clip gradient norm | Adapter step L2 change |
|---|---:|---:|---:|---:|---:|
| 1 | 37 / 64 (57.81%) | 3 | 3 | 2.5441 | 0.080834 |
| 2 | 36 / 64 (56.25%) | 5 | 4 | 5.7521 | 0.058285 |
| 3 | 40 / 64 (62.50%) | 4 | 4 | 5.7777 | 0.049812 |
| 4 | 37 / 64 (57.81%) | 6 | 5 | 7.0269 | 0.045029 |

Admission required at least two groups with distinct valid question lists and
both EM reward values among valid plans. All 256 roots parsed successfully, so
admission was never explained merely by malformed-root versus valid-root syntax.
Of 18 qualifying parent-update groups, 16 contained both correct and incorrect
**valid final answers**. Two groups (one each in updates 2 and 4) qualified solely
through downstream protocol failure versus success. Four qualifying groups
included downstream failures in total; two of those also had fully scored reward
variation. These counts overlap as stated, rather than summing to 18.

There were 58/54/58/57 scored finals and 6/10/6/7 invalid-helper-JSON outcomes by
batch. Across all batches: 150 correct, 77 scored-but-incorrect, and 29 helper
protocol failures. No invalid roots, dependency failures, invalid finals, missing
generations, or transport failures were recorded. Regrading every actual final
with official alias-aware `probe.grade` reproduced every saved binary reward.
Fully scored variation is stronger than syntax-only variation, but still does
not isolate semantic plan quality from plan-dependent downstream sampling.

The batch loss always divided by 64. Only 12/20/16/24 trajectories had nonzero
leave-other-three-out advantages; the other 184 of 256 correctly contributed zero
gradient. Four parents never received a positive reward in their 16 attempts;
five were always correct. Those nine parents supplied no within-group learning
signal throughout this run.

## Diversity, not a collapse claim

| Batch | Distinct valid question lists / 64 | Parents with four identical sampled plans / 16 | Mean emitted root tokens |
|---|---:|---:|---:|
| 1 | 56 | 0 | 25.47 |
| 2 | 57 | 1 | 25.03 |
| 3 | 49 | 2 | 24.84 |
| 4 | 56 | 0 | 25.06 |

Exact-list diversity dips in batch 3 and recovers; root length stays stable.
This is no observed short-run collapse, not a guarantee against longer-run
collapse or a semantic-diversity measurement. Two-question plans dominate:
47/48/48/49 per batch. Remaining lengths are 1–4; the per-step helper cap remains
`floor(384 / plan_length)`. Identical plans within each of the four batches had
identical rewards, consistent with frozen helpers and shared within-group seeds.

## Likelihood and numerical checks

Only 16,515,072 FP32 LoRA parameters were trainable; the load receipt records all
non-LoRA parameters frozen. Every root's recorded adapter hash matches its
pre-update checkpoint, every helper/final records its adapter disabled, and all
four checkpoint COMMIT manifests verify against their files. No weights changed
between collection and the single gradient accumulation pass.

For each saved root trajectory, define `delta = sum(after_logps) - sum(before_logps)`
using full-forward scoring at temperature 0.8, including emitted EOS. The last
column below is `sum(advantage * delta) / 64`, not reward gain or KL divergence.

| Update | Mean delta, positive advantage | Mean delta, negative advantage | Advantage-weighted likelihood gain |
|---|---:|---:|---:|
| 1 | +0.4953 | −1.0223 | +0.08068 |
| 2 | +0.5213 | −0.7281 | +0.12392 |
| 3 | +0.2668 | −0.5855 | +0.06858 |
| 4 | +0.1733 | −0.7544 | +0.09808 |

All updates move the matched full-forward objective in the intended direction.
Across the 72 nonzero-advantage trajectories, 27/34 positive-advantage sequences
increase in likelihood and 36/38 negative-advantage sequences decrease. Shared
parameters and Adam momentum do not require every sequence to move independently
in its preferred direction.

Train-mode gradient replay versus eval-mode full-forward scoring has recorded
maximum absolute difference **0.0** in each update. Cached generation versus
full-forward scoring is not identical:

| Batch | Mean absolute token-logp gap | 95th percentile | Maximum |
|---|---:|---:|---:|
| 1 | 0.00929 | 0.05938 | 0.37284 |
| 2 | 0.00880 | 0.06016 | 0.25414 |
| 3 | 0.00780 | 0.04639 | 0.37531 |
| 4 | 0.00865 | 0.05225 | 0.32113 |

Units are natural-log probability. These are the observed BF16 cache/full-forward
path discrepancies; their numerical cause was not separately ablated. Mean
absolute sequence-logp gaps are 0.109/0.094/0.093/0.098 nats, with maxima up to
0.782 nats. Do not claim exact behavior/replay likelihood equality. Matched
full-forward before/after movement is nevertheless directly measured. No KL
penalty, importance ratio, or trust-region constraint was used.

## Actual acquisition cost and duration

| Batch | Calls | Input tokens | Output tokens | Total tokens | Rollout wall span (seconds) |
|---|---:|---:|---:|---:|---:|
| 1 | 267 | 589,748 | 3,458 | 593,206 | 240.7 |
| 2 | 261 | 571,057 | 3,383 | 574,440 | 235.4 |
| 3 | 265 | 582,318 | 3,357 | 585,675 | 237.0 |
| 4 | 260 | 567,526 | 3,326 | 570,852 | 235.0 |
| Total | 1,053 | 2,310,649 | 13,524 | 2,324,173 | — |

Owner elapsed time: **1,026.8 seconds (17.11 minutes)**, including loading,
rollouts, gradient work, checkpoint writes, and likelihood diagnostics. Summed
native-call durations are 940.3 seconds; this is not optimizer-only timing.
All usage is known and there are zero unresolved starts. Scoring/replay forward
passes consume compute but are not additional generated-token calls in this table.

## Evidence and next decision

Run root:
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/rl-planner-001`.

Source evidence is the immutable `PLAN.json`, `LOAD-2a268b0cddf4.json`,
`batch-0001` through `batch-0004` (`BATCH.json`, episodes, calls,
`BEFORE_LOGPS.json`, `AFTER_LOGPS.json`), committed `checkpoint-0001` through
`checkpoint-0004`, and `TERMINAL-2a268b0cddf4.json`. Calculations above join logp
arrays by their saved call IDs, use BATCH group order for advantages, sum native
usage receipts, and define rollout span as last call end minus first call start.
Token-gap percentiles use sorted absolute differences at index
`floor(0.95 * (N - 1))`. No raw training traces are copied into this note.

- PLAN SHA256: `be75909a150c2c88e77f68108c59f19eae84b26ef89c282cb01972eb0c92c5d9`.
- Final adapter SHA256: `970c760f11eb0f906f382a9fc2f9ac3654e3ba25777586b4af2473907b618f7c`.
- Audit cutoff: completed four-update terminal receipt, 2026-09-21 10:44:29 UTC.

The next informative result is the fixed checkpoint-4 versus SFT-parent comparison
on the same untouched held-validation parents and identical execution contract.
Use paired parent analysis, keep protocol failures and acquisition costs visible,
and do not relabel these 256 training attempts as 256 independent test examples.
