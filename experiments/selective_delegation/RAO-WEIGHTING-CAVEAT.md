# RAO weighting caveat: a narrow estimator qualification

Status: `cpu_diagnostic_not_novelty`. This does not claim RAO's leave-one-out
lemma is false, that the method is harmful, or that an update sign reverses.

For the paper's unweighted LOO baseline, conditioning on a node task/history makes
the other independent root-tree rewards a valid action-independent baseline. The
issue begins only after multiplying advantages by a depth weight computed from the
same batch's sampled tree sizes.

The exact two-root enumeration in
`R/analysis-rao-weighting-001/report.json` uses root `A~Bernoulli(p)`, root reward
`A`, and one deterministic child iff `A=1`. With paper Eq. 4, the root weights for
total child counts 0, 1, 2 are 1, .75, 1. The **sum over G=2 roots** of the
unweighted baseline-score term is zero; its weighted expectation is
`.5*p^2*(1-p)` (0.0315 at p=.3; 0.0625 at p=.5). Divide every reported G=2
quantity by two for a per-root mean; do not compare the raw 0.42 reward-score sum
at p=.3 to the derivative of the per-root `J=p` (0.21).

Thus the weighting changes the estimator and the LOO cancellation proof no longer
directly applies. That may be an intentional reweighted surrogate, not an error.
It is also a familiar issue with sample-dependent/self-normalized weighting, not a
novel result.

A distinct exact two-parameter toy supports the narrower semigradient statement.
Let root delegation `A~Bernoulli(p)`, root reward `A`, and, only if delegated,
child success `C~Bernoulli(q)` with child-local reward `C`. For
the explicitly chosen expected *sum of node rewards per root rollout*,
`J=p+p*q`, a root-local score using only root reward has `d/dp=1`; the full
derivative is `1+q`, leaving the policy-induced child-distribution term `q`.
This establishes the caveat for that objective, but not a universal statement:
different parent rewards/delegation bonuses can supply cancellation terms.
This visitation-weighted objective is not automatically the paper's normalized
distribution over tasks at each depth. Normalization and absent-depth handling
must be specified before applying this example to that objective.

The pinned code does not exactly implement paper Eq. 4 in general. Its default
`depth_level_weighting` is false (`platoon/train/tinker/config_defs.py:26`). When
enabled, `tinker/rl.py:118-156,612` counts trajectories, normalizes by action-token
mass, and multiplies **advantages**. Equal lengths recover the toy-style frequency
factor, so the toy applies to that Tinker path. In contrast, `areal/rl.py:351-392`
uses trajectory counts and datum/token mass to pre-weight `batch["rewards"]`
**before** `actor.compute_advantages`; it is not the same estimator. This audit
does not infer the toy's baseline effect for AReaL without tracing its downstream
advantage computation. It also did not enumerate experiment configs, so it does
not assert weighting was enabled in a particular RAO run.

Inputs: RAO [§2.2 and Appendix A.2](https://arxiv.org/html/2605.06639v1) and pinned
commit `d9c5857d3a0a056ebc9b047241a2a0c9515aafbe`; exact hashes are in the external
receipt.
