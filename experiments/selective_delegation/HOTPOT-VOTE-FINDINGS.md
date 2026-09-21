---
status: completed_exploratory
evidence_cutoff_utc: 2026-09-21T16:56:00Z
question: Does three-answer voting improve the cheap direct baseline?
decision: retire_this_low_temperature_vote_as_an_improvement_strategy
---

# Repeating the direct answer did not improve it

On the fixed 128-question Hotpot panel, sampled twice, three-answer voting gives
exactly the same chosen answers as the original single direct call: 152 correct
out of 256, with official answer F1 of 69.36%. The decomposition system gets
151 correct with F1 of 71.05%; neither paired difference is established.

This was not three identical random seeds. Each original response was retained
as voter zero and two fresh responses used its actual seed plus 10,000 and
20,000. Nevertheless, at temperature 0.5, 251 of 256 sets contain only one
distinct normalized ballot. Four sets contain two and one contains three. The
predeclared majority rule, breaking ties toward the earliest voter, chooses
voter zero in every case. Whitespace variations do not constitute new answers.

All 512 new calls returned. Seven of the 768 total ballots were malformed;
malformed responses were explicit invalid ballots, not silently discarded.
Two final voted outputs consequently receive protocol-zero scores. No missing
voter, inferred answer, gold-aware choice, or answer fallback was used.

| Policy | Correct / 256 | Calls for deployment | Total tokens |
| --- | ---: | ---: | ---: |
| One direct answer | 152 | 256 | 436,983 |
| Three direct answers, vote | 152 | 768 | 1,310,973 |
| Trained planner and helpers | 151 | 1,071 | 1,501,958 |

The new collection itself costs 512 calls and 873,990 tokens. The vote's full
deployment cost also includes the reused original responses; it is lower than
the decomposition system's cost, but this is not an exactly equal-compute test.
Collection owner wall time is 362.73 seconds, not GPU kernel time.

The vote-minus-direct paired interval is exactly zero because every chosen
answer is unchanged. This does not prove voting is universally ineffective:
it tests this model, prompt, temperature, dataset and three-sample rule. Raising
temperature could produce more variation, but we will not spend the next batch
on a blind sampling sweep while distinct information-access tasks are ready.

## Evidence

External root:
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.
The sealed collector is `source-025-hotpot-vote`; new receipts are in
`hotpot-fresh-direct-vote-001`. The paired report is
`analysis-hotpot-fresh-direct-vote-001/REPORT.json`, with 20,000 parent-bootstrap
draws, seed 2026092179. Native request records retain all actual seeds, sampling
settings, outputs and token counts. The two repeats are clustered by their
128 parents rather than counted as 256 independent questions.
