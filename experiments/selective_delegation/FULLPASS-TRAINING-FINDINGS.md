# Full-pass root-RL training: process finding, not held-out evidence

The completed receipt audit covers 16 consecutive train-only updates (1,024
episodes; 4,335 native calls; no failed or unresolved calls) in 5,063.19 s.
It is an optimization-process result, not evidence that RL improved planning or
answering on fresh data.

Across the 256 parent groups, 78 had differing rewards and 76 met the stricter
distinct-valid-plan admission condition. In total, 312/1,024 candidate
trajectories had nonzero leave-one-out advantage. Of the 78 mixed-reward groups,
73 contained both correct and incorrect fully scored answers. Five had reward
variation only because of protocol failures; nine contained any protocol failure
(four of those also had fully scored reward variation). Thus most reward
variation was not merely a parse/dependency artifact, but full-source finals may
still bypass or repair helper chains. It is not a clean plan-quality reward.

Every update changed the adapter and increased the reward-weighted training
log probability. Individually, 126/182 positive-advantage trajectories became
more likely and 109/130 negative-advantage trajectories became less likely;
shared parameter updates need not move every individual example as intended.
This is likelihood movement on sampled training trajectories, not a held-out
RL gain. Exact valid-plan list diversity was present in every
batch but is syntactic, not semantic diversity. Consecutive parent blocks change
task composition, so reward by update is explicitly **not** a learning curve.

A [separate structural check](PLAN-STRUCTURE-FINDINGS.md) finds multiple valid
step/dependency graphs in111/256groups. Among73groups with both correct and
incorrect fully scored answers,38have multiple graphs and35the same graph.
The candidates therefore are not uniformly structurally identical, but neither
graph diversity nor different wording establishes useful semantic alternatives.

Receipts record 9,847,733 prompt and 57,060 completion tokens. Summed returned
call latency was 4,718.35 s; it must not be equated with GPU kernel time. The
344.84 s difference from wall time includes optimization, checkpoint saving,
loading and other work outside the call timers; it is not optimizer time alone.
Calls were sequential, not concurrent. The run's new model calls account for
most of its elapsed time, unlike a short supervised pass over already prepared
targets.

| Role | Calls | Input tokens | Output tokens |
|---|---:|---:|---:|
| Planner | 1,024 | 383,540 | 28,094 |
| Helper | 2,301 | 6,463,360 | 19,499 |
| Final answer | 1,010 | 3,000,833 | 9,467 |

Fourteen attempts stopped before a final call: two invalid plans and twelve
invalid dependencies. There were no helper-format failures on these training
trajectories. Training-data performance is not a helper transfer result.

Authoritative audit: `analysis-rl-fullpass-001.json`, SHA-256
`0f6df49ab55d1f2454920d64fd63794090c67cfb3cebbd8a7d17eec8e771201d`.
Final checkpoint 0016 hashes: adapter weights
`1fa77aee09622e8a9491c173c4178930e4e40329aa3cf7b58cae8a44c7c7dd45`,
config `7d4bbcf647bf641b89e42ca67f14d5967fb3008a59c4c16729456b5a0f8384ed`,
state `1fe84c22735e2314bfa80b0b283fded0e0b566598553a04412daaf47806d55eb`,
and commit `3599d3703cb2e57467b9ccecea5928ecf81ff147f4fa553fbf480e025c892840`.
