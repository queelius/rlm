# Paired sufficiency RL: real updates, sparse and mixed TRAIN bottlenecks

RL037 completed cleanly on 2026-09-21 at19:54:15 UTC: eight sampled blocks and
eight optimizer updates, with no skipped all-zero block. The fixed endpoint is
`sufficiency-rl-001/boundaries/sample-0008/checkpoint-0008`.
Extra-SFT038 completed cleanly at20:12:16 UTC with eight matched updates.
The combined sealed training audit passed. This document describes TRAIN evidence;
completed039 results are separate in `PAIRED-RL-HELDOUT-FINDINGS.md`;046 remains pending.

This finite TRAIN competence/objective diagnostic is not a novel abstention method
or a generalization result. **Each block contains different questions: the reward
column is not a learning curve.** The128 TRAIN parents comprise95 two-hop,26
three-hop and7 four-hop questions. Later039 is all two-hop;046 contains16 three-hop
and16 four-hop parents, so046 does not test entirely unseen training depths.

## What actually learned

All1,024 native requests and emitted-token/official paired-reward bindings passed
sealed analyzer002. There were no missing or transport-failed calls, and510/512
paired candidates were protocol-valid. The two invalid outputs supplied numeric
`answer` values80 and1 instead of strings; they were not repaired.

| Changing TRAIN block | Paired EM | Mixed groups | Gradient norm | Actual adapter delta L2 | Advantage-weighted logp movement |
|---|---:|---:|---:|---:|---:|
| 1 | 4/64 | 2/16 | 1.7003 | 0.08083 | +2.914 |
| 2 | 10/64 | 4/16 | 2.6063 | 0.05093 | +0.896 |
| 3 | 13/64 | 6/16 | 3.0728 | 0.04344 | +2.245 |
| 4 | 8/64 | 5/16 | 2.4264 | 0.04047 | +1.866 |
| 5 | 13/64 | 8/16 | 3.4589 | 0.03921 | +2.605 |
| 6 | 12/64 | 8/16 | 2.0376 | 0.03663 | +1.137 |
| 7 | 8/64 | 4/16 | 1.9478 | 0.03343 | +0.766 |
| 8 | 10/64 | 5/16 | 3.2362 | 0.03250 | +0.786 |

Across128 parent groups,42 had nonzero RLOO advantages,85 were uniformly zero and
one was uniformly successful. All42 mixed groups contain valid correct and
incorrect paired candidates; one also contains a malformed candidate. Aggregate
on-policy reward is78/512, not an estimate of improvement over the warmstart.
Only4,165/12,387 emitted tokens receive nonzero advantage (336 responses from168
paired candidates). Correct uniformly successful candidates also give zero
within-group advantage.

All eight saved adapter deltas were independently recomputed on CPU from adjacent
checkpoints and match their update receipts (504 LoRA tensors;16,515,072 scalars).
The native prompt/seed/adapter/T=.8 bindings, actual emitted token lengths including
EOS when emitted, RLOO advantages and denominator64 sequence-sum losses were
checked. All eight aggregate advantage-weighted post-update likelihood movements
are positive. That establishes actual parameter/likelihood movement, not held gain.

## What zero paired reward hides

Among510 fully protocol-valid pairs,227 positive responses abstain,156 attempt an
EM-wrong answer and127 are positive-EM correct;170 negatives overanswer.
Only78 pairs meet both labels and positive EM. Another102 pairs have both labels
right but a wrong positive answer. Those columns overlap except the positive
response partition. Exact mismatch is official alias-aware EM, not an independent
semantic adjudication of annotation completeness.

| Block | All-zero groups / pairs | Positive abstains | Attempts with wrong EM | Correct positive blocked by negative | Both labels right, wrong positive EM |
|---|---:|---:|---:|---:|---:|
| 1 | 14 / 56 | 30 | 22 | 4 | 15 |
| 2 | 11 / 44 | 18 | 23 | 3 | 14 |
| 3 | 10 / 40 | 17 | 22 | 1 | 12 |
| 4 | 11 / 44 | 26 | 14 | 4 | 8 |
| 5 | 8 / 32 | 17 | 11 | 3 | 5 |
| 6 | 8 / 32 | 12 | 18 | 2 | 12 |
| 7 | 12 / 48 | 35 | 13 | 0 | 12 |
| 8 | 11 / 44 | 27 | 17 | 0 | 13 |

The85 all-zero groups contain340 pairs: one malformed pair,182 positive abstentions,
140 attempted wrong answers and17 correct positives blocked by the negative
response. Of these,91 have both labels correct but miss the positive answer.
Fifty-six of85 all-zero groups still vary in both-label correctness across their
four candidates. A partial/label reward would therefore expose more variation,
but it changes the objective and could reinforce wrong-content positives; it is
not automatically a better solution. Mixed groups and uniformly successful groups
remain in the full external decomposition, not omitted as inconvenient cases.

## Numeric replay and cost

BF16 cached generation versus full replay is approximate, not numerically exact.
Across4,165 credited tokens, absolute gap median/p90/p95/max is
0.000000596/0.000130/0.007568/0.204433 log-probability units.
Across336 credited responses, sequence-sum signed gaps range−0.150432 to+0.181913;
absolute median/p90/p95 is0.003637/0.088565/0.119297.
The saved generation and replay use the same T=.8 sampling distribution, but these
finite numerical differences preclude claiming exact numeric on-policy equivalence.
No automatic stop or owner/objective change was made on that basis.

RL physical rollout cost:1,024 calls,2,896,748 native input tokens and12,387 output
tokens;1,160.53 native-service seconds and1,627.75 owner wall seconds (27.13min,
including training/replay/checkpoint overhead). Native token totals are not lower
bounds. Teacher-forced SFT training cost is a separate category; matched examples
and optimizer steps do not imply matched tokens, information or FLOPs.

## Matched extra-SFT control completed

Both runs start from the identical joint32 adapter with a fresh optimizer and
LR2e−5. The control used the exact eight admitted RL blocks and eight updates:
1,024 teacher-forced responses,12,580 masked gold target tokens, and1,074.04s
owner wall time (17.90min). Its gradient norms range0.47982–0.88813 and recorded
adapter deltas L2 range0.03100–0.08075. The source/checkpoint audit verifies the
actual matched schedule, warmstart and final committed step8; no endpoint was
substituted. All eight saved control adapter deltas were independently recomputed
on CPU and match the receipts. Zero native generation calls for SFT means teacher forcing, not zero
model compute. Relative to RL's4,165 nonzero-advantage emitted tokens, the control
receives many more supervised target tokens; this is not an information/FLOP match.

## Held decision boundary

The held question is whether these eight actual updates improve the fixed paired
objective relative to joint32 and matched additional SFT. Continuing
to raise TRAIN dose, changing reward or claiming better calibration is not
justified by this changing-batch readout alone. Completed039 finds both continuations
improve paired score but no established RL advantage over extra SFT; see the separate
held findings for intervals and the supported-answer/overanswer tradeoff. Predeclared046
remains the deeper-combination check, without retuning these endpoints.

Evidence under R (September21 selective-delegation run store):
`analysis-sufficiency-rl-training-001.json/.md` is the completed combined audit
(JSON SHA256 `7864b69eac4fab5b0c1e2c589303487f3859ce8bef71c39475496e4e359138b8`).
`analysis-sufficiency-rl-only-001.json/.md` (sealed analyzer002, all native receipts,
source hashes and boundaries); `analysis-sufficiency-rl-errors-001.json` with
`analysis-sufficiency-rl-error-source-001.py`; and
`analysis-sufficiency-rl-numerics-001.json` with
`analysis-sufficiency-rl-numerics-source-001.py`. The earlier independent first-block
receipt `analysis-sufficiency-rl-first-block-001.json` is retained unchanged.
