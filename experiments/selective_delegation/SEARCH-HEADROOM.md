---
question_id: selecting-among-plans
status: exploratory_diagnostic
source_run: rl-planner-001
cutoff_utc: 2026-09-21T10:44:29Z
---

# Would search among plans help?

The four-plan RL groups let us inspect a small amount of search headroom without
generating more answers. This is a retrospective training-data diagnostic, not
an evaluation of tree search or a learned verifier.

| Pre-update batch | Mean success of one sampled plan | At least one correct among four | Answer plurality /16 | First candidate /16 |
|---|---:|---:|---:|---:|
| 1 | 57.8% | 68.8% | 10 | 9 |
| 2 | 56.3% | 75.0% | 9 | 9 |
| 3 | 62.5% | 75.0% | 10 | 10 |
| 4 | 57.8% | 75.0% | 10 | 9 |

The second column averages all64 candidate outcomes. The third uses gold answers
to choose the best of four and is therefore **not deployable**. It is observed
best-of-four headroom, not a bound on every possible search procedure. Plurality
chooses the most frequent normalized valid final answer; ties choose the lowest
candidate index, and no valid final scores zero. Its selection never reads gold.
Normalization is the official MuSiQue answer normalization. First-candidate
counts are included for a concrete fixed-choice comparison; neither column is
an independent test, and all batches reuse the same16 training parents.

There are sometimes better candidates, but ordinary answer agreement does not
reliably identify them. This strengthens the case for testing a useful evaluator
before adding MCTS/PUCT machinery. At deployment, the gold-answer reward used
during RL is unavailable. A search experiment would need an explicit non-gold
value estimate and must compete with simpler sampling at the same call budget.
Using the learned planner's probability as a search prior does not by itself
provide that value estimate.

Reproduction: group `batch-*/episodes/*.json` by `case_id`, sort by `candidate`,
use saved binary `reward` for final scoring, and count normalized
`score.parsed` among rows with `score.valid=true`. All16 parents stay in each
denominator. No additional GPU calls or model updates were performed.

## Exact request reuse is a smaller opportunity than tree search

A secondary CPU audit grouped frozen helper/final calls by their entire saved
request within each update. It did not group roots or reuse samples across
updates. All 126 duplicate groups returned identical text, native token IDs,
availability, and finish/error status in the observed run.

Keeping each group's first successful response would avoid 210 of 797 downstream
calls: 174 helpers and 36 finals. That is 26.3% of downstream calls, 19.9% of all
1,053 calls, and 24.1% of all recorded tokens. The duplicate calls took 117.5
seconds of recorded service time, or 12.5% of call service time. These are
retrospective avoidable costs, **not a measured wall-clock speedup**.

This supports a modest, exact within-update reuse optimization if later runs
scale up. It does not establish better answers, more informative samples, or
the value of MCTS. No cache was implemented. Preserve strict request identity,
seeds, frozen model identity, and separate physical versus logical call costs
if testing reuse. The reproducer is `audit_memoization.py`; the external receipt
is `analysis-rl-memoization-001.json/.md`.
