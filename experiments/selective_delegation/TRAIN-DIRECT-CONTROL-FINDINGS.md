# TRAIN direct-control findings

The TRAIN-only execution comparison does not reduce to recovery from a bad
plan-only final. Across the 320 logical slots, direct was 148 correct, plan-only
was 99, and executed helper reports were 198. The component-clustered,
parent-weighted EM contrasts (20,000 draws; seed `2026092191`; 16 parents in 15
atomic-component clusters) were:

- executed helper minus direct: +15.63 pp, 95% CI [+1.76, +32.67]; F1 +17.00 pp
  [+2.33, +34.81];
- direct minus plan-only: +15.31 pp, 95% CI [+1.25, +34.00]; F1 +15.26 pp
  [+1.15, +33.89].

The direct-to-plan-only contrast confirms that final answers conditioned on a
generated plan can be harmful here. Yet the executed-to-direct contrast is also
positive: helpers add more than merely restoring that loss in this fixed TRAIN
trace setting. All 320 paired comparisons were both-valid JSON: executed versus
direct had 61 wins, 11 losses, and 248 ties; direct versus plan-only had 52 wins,
3 losses, and 265 ties. This is not an oracle routing result, a held-out result,
or evidence that a helper policy is deployable.

Physical accounting is deliberately separate from logical outcomes. The new
direct control returned 80/80 calls (each parent/setting response is copied into
four candidate-correlated logical slots), with no unavailable or invalid-JSON
calls: 220,266 tokens and 46.12 summed returned-call seconds. The observed
helper/plan-only outcomes are reused from `training-execution-credit-001`; their
analysis-time calls were 330 calls / 931,283 tokens, while the original
historical acquisition recorded 1,109 calls / 2,957,611 tokens. Neither reused
quantity is new direct-control expenditure, and the 80-call direct construction
is not compute-matched to the candidate-specific historical execution traces.

The immutable receipt is
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/analysis-training-direct-001.json`
(`16daac320d033f769c9453366a91aff271026795bdbd8ac332eb6f9e4664cd3b`),
with the frozen analyzer and fixture in
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/analysis-source-training-direct-001/`.
