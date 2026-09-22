# Held paired sufficiency: both training methods help, no established RL advantage

Completed039 readout, 2026-09-21 20:19:57 UTC. Independent sealed analyzer002
validated all384 native requests/responses, official grades, source and fixed
adapter endpoints. All calls are available and protocol-valid: no missing data,
schema failures or unresolved starts. No partial held outcome selected a checkpoint.

The panel contains32 two-hop parents, each with positive/negative variants and
two seeds. It has30 frozen atomic-component clusters, not64 independent parents.
The warm joint32 adapter is compared with its fixed eight-update paired-RL and
matched eight-update extra-SFT descendants, under the identical public contract.

| Policy | Paired EM /64 | Paired F1 | Both labels /64 | Supported EM /64 | False abstentions /64 | Negative overanswers /64 |
|---|---:|---:|---:|---:|---:|---:|
| Warm joint32 | 10 | .15625 | 16 | 13 | 42 | 6 |
| Paired RL8 | 18 | .31161 | 29 | 26 | 22 | 14 |
| Matched extra-SFT8 | 19 | .30580 | 30 | 27 | 17 | 17 |

Both continuations reduce the warmstart's over-abstention and increase supported
answer accuracy, while also overanswering more unanswerable cases. Their net
paired scores improve on this fixed panel; this is not merely a higher rate of
returning an answer. **There is no demonstrated paired-RL advantage over matched
additional SFT.** The SFT–RL uncertainty interval is not an equivalence test.

| Contrast | Paired EM change, percentage points [95% CI] | Paired F1 change, percentage points [95% CI] | EM wins / losses |
|---|---:|---:|---:|
| RL − warm | +12.50 [+5.00,+20.31] | +15.54 [+8.33,+23.57] | 11 /3 |
| Extra SFT − warm | +14.06 [+5.88,+23.44] | +14.96 [+6.67,+24.33] | 12 /3 |
| Extra SFT − RL | +1.56 [−3.33,+6.94] | −0.58 [−6.83,+5.71] | 2 /1 |

All changes are between valid scored pairs, not protocol recoveries. Intervals
resample the frozen30 component clusters with parent weighting, retaining both
variants and seeds;20,000 draws, seed2026092200, percentile95. Parent-only
companions are retained in the JSON. These are exploratory, unadjusted intervals
on a small two-hop panel, not a claim of broad task-distribution independence.

## Concrete tradeoff, not a decomposition claim

For parent `2hop__195917_575657`, seed2026092181, the warmstart abstains on the
supported question about William W. Blair's birthplace county; both trained
policies answer Orleans County correctly and still abstain on the negative variant.
Conversely, for `2hop__445963_6095`, seed2026092182, all three correctly answer
1792 on the supported question. Both continuations lose the paired reward by
answering1808 on the officially unanswerable variant, where warm joint32 abstains.

The small SFT–RL difference is also mixed: SFT gains the Slade School of Art answer
where RL abstains (`2hop__501624_181960`, seed2026092181); RL gives Yang Xingmi
where SFT gives Wang Rong (`2hop__668407_683671`, seed2026092182). These examples
illustrate saved predictions, not adjudication beyond the official host labels.
All64 paired attempts per policy remain in the analysis.

## Cost and interpretation

Each policy costs128 calls and346,614 native input tokens. Output tokens are
1,378/1,498/1,522 for warm/RL/SFT; native service time129.31/137.02/138.75s.
The whole readout costs384 calls,1,039,842 input and4,398 output tokens,
405.08s native service and450.52s owner wall time. Training is separate:
RL1,627.75s and extra SFT1,074.04s. Matching eight updates, sampled parents and
1,024 response slots does not match information, credited tokens or FLOPs:
RL credits4,165 emitted tokens; SFT teaches12,580 gold target tokens.

The useful conclusion is narrower than an RL-specific result: this warmstart's
conservatism is repairable with either small continuation, and explicit supported
accuracy plus negative-overanswer measurements expose the tradeoff. Do not use
the changing TRAIN block rewards as an improvement curve. Do not claim a new
abstention method, decomposition benefit, or superiority of paired over independent
per-variant rewards: none is tested here.

Next informative evidence is already-frozen046 on deeper combinations, with the
same endpoints and no tuning from039. Its16 three-hop/16 four-hop parents are not
unseen training depths (the RL TRAIN set includes26 three-hop and7 four-hop parents).
A later joint-pair versus per-variant reward control could isolate the credit
question, but no such training or new GPU run is accepted by this analysis.

Evidence: `R/analysis-sufficiency-heldout-rl-001.json/.md`, JSON SHA256
`a5276f2be83e54271d2582d9ad5df5c65f41b976e47960f422c707e747e3a419`;
collector `R/sufficiency-heldout-readout-001`, cases
`R/sufficiency-heldout-inputs-003/cases.jsonl` SHA256
`52d4c74fedbb89b3b6b56d26538d95d2d512445d0bf7c52883d765c7b1195150`;
analyzer `R/analysis-source-sufficiency-rl-002/analyze_sufficiency_rl.py` SHA256
`42ad523a9aedab9a3d200eabfdb632ac942ecbc56dbe42961d7d8f1ff2f8e428`.
R denotes the September21 selective-delegation run store. Full native receipt,
model/source/checkpoint hashes and frozen clusters are in the immutable report.
