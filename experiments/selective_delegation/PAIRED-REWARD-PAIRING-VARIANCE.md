# Paired product-reward pairing variance

## Result

For four fixed positive draws `P_i` and four fixed negative draws `N_j`, the
proposed coefficients are the exact conditional expectation of the *current*
uniform-diagonal-pairing RLOO raw score-function gradient, averaged across all
24 one-to-one pairings. This is Rao--Blackwellization of pairing noise, not a
new reward or extra rollout.

Let `R_i=P_i N_{pi(i)}` and
`A_i=R_i-(sum(k!=i) R_k)/3`. The current per-parent mean gradient is
`(1/4) sum_i A_i (s^+_i+s^-_{pi(i)})`. Averaging over a uniform permutation
gives coefficients

```text
c+_i = mean(N) * (P_i - mean(P_-i)) / 4
c-_j = mean(P) * (N_j - mean(N_-j)) / 4.
```

The `/4` is required: `rl_sufficiency.py` has 16 parents × four candidate
pairs and divides each of the eight response losses by 64. Enumerating all 24
permutations for a nontrivial vector-score fixture exactly matches this closed
form (floating error only). The report is at
`R/analysis-paired-reward-pairing-002/REPORT.json`.

## Actual completed 037 rollouts

The read-only audit covers all eight committed `sufficiency-rl-001` sample
blocks (128 parent groups), never the incomplete/noncommitted directory. Of
those groups, 46 have mixed positive marginal success and 87 mixed negative
marginal success. The observed diagonal product has nonzero RLOO credit in 42
groups; the conditional-expectation coefficients are nonzero in 52, including
10 groups whose realized diagonal was flat zero. This is the precise "restored"
signal: some saved marginal successes were mismatched by candidate index.

Normalized absolute coefficient mass across **both** positive and negative
response sides is 48.0 for the observed diagonal and 29.5 after conditional
averaging. That descriptive difference is not itself a
variance estimate or an expected learning gain. A marginally mixed side still
gets zero credit when the other side has zero mean, as required by the same
product objective.

## Scope and decision relevance

The result assumes four exchangeable independent variant draws conditional on a
parent and an arbitrary uniform one-to-one diagonal pairing. It applies to the
raw gradient **before** gradient clipping and Adam; it does not guarantee lower
variance after those nonlinear updates, better convergence, or value for
parent-dependent recursive trajectories. It is a cheap prospective control only
if a later RL comparison is otherwise warranted: retain the same eight calls,
product objective, and root log-prob trajectories, replacing only index pairing
with the predeclared all-pairing conditional expectation.

This is a standard variance-reduction pattern, not a novelty claim. Related
factorized-action Rao--Blackwell baselines appear in [Wu et al., 2018](https://arxiv.org/pdf/1803.07246).
[ConsistRoll](https://arxiv.org/html/2606.29812v1) is related paired-
consistency-reward work, but uses each answer-preserving view's own correctness
plus a correctness-gated joint/agreement bonus and GRPO. It neither establishes
novelty nor validates this unnormalized-RLOO calculation for deliberately
different answerability variants.
