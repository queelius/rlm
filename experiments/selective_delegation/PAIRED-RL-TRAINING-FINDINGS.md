# Paired sufficiency RL: sparse first-block learning signal

Interim completed-block evidence only, 2026-09-21 19:32 UTC. RL037 remains active;
extra-SFT038 and held039 results are not inferred here. This is a finite TRAIN
competence/objective diagnostic, not a novel abstention method or generalization result.

Block1 contains 16 parents × four paired candidates, 128 native calls. All 64 pairs
are protocol-valid; 4/64 receive official paired EM reward. Two groups have two
successes each; fourteen have zero. Distinct answers alone therefore do not ensure
useful paired-reward variation.

| Pair subset | Positive abstains | Attempts with wrong positive EM | Correct positive | Negative overanswer | Both labels correct, positive EM wrong |
|---|---:|---:|---:|---:|---:|
| All64 | 33 | 23 | 8 | 23 | 16 |
| Fourteen zero-reward groups /56 | 30 | 22 | 4 | 23 | 15 |
| Two credited groups /8 | 3 | 1 | 4 | 0 | 1 |

The first three columns partition each row; remaining columns overlap. Correct
positive means positive answerable=true and official alias-aware exact match.
All four correct positives in zero-reward groups are blocked by the negative
response. Fifteen zero-reward pairs get both answerability labels right but miss
positive EM. Thus neither pure label calibration nor pure answer training alone
explains all lost credit. Exact mismatch is not proof that annotation is complete.

All128 saved public prompts, native input/output IDs, T=.8 sampling, seeds,
adapter identities and official paired rewards were independently checked.
Step1/sample-cursor1 is committed. Gradient norm before clipping is 1.700338;
actual saved-adapter delta L2 is 0.08082924, recomputed on CPU across504 LoRA
tensors/16,515,072 scalars and matching the receipt. Recomputed emitted-token
RLOO loss with denominator64 is −0.01594458. Advantage-weighted post-update
sequence log-probability movement is +2.91362. Only190 of1,505 generated tokens
receive nonzero credit (16 responses from eight paired candidates).

BF16 cached-generation versus full replay is approximate, not numerically exact.
Absolute token-gap median/p90/p95/max is 0.000000596/0.000961/0.019287/0.136963.
Sequence-sum signed gaps range −0.087233 to+0.087326; absolute median0.069017.
All credited pair advantages are ±2/3. Twelve of16 individual response likelihoods
move with their advantage; identical correct negative responses cancel within
each credited group, so opposite movements for some duplicate negative responses
are not independently a credit-routing defect. These are measured discrepancies,
not grounds by themselves for stopping or claims of exact on-policy numeric replay.

No objective or owner change was made. Later fixed blocks must distinguish
repeated sparse usable signal from all-zero batches; held readout is required to
assess whether updates repair conservatism rather than merely fit these pairs.

Evidence: `R/sufficiency-rl-001/batches/sample-0001/{ROLLOUT,UPDATE}.json`,
`boundaries/sample-0001/BOUNDARY.json`, and immutable
`R/analysis-sufficiency-rl-first-block-001.json` (native/checkpoint source hashes,
per-parent decomposition and sequence-gap measurements). R is the September21
selective-delegation run store. Complete037/038 analysis will use sealed
`R/analysis-source-sufficiency-rl-002/analyze_sufficiency_rl.py`.
