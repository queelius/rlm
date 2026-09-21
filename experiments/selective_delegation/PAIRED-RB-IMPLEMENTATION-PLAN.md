# Pairing-averaged RL estimator implementation plan

> **For agentic workers:** Use superpowers:executing-plans for this bounded CPU implementation. Main reviews the focused scientific checks before any GPU acceptance. The user's autonomous exploratory-work instruction replaces an interactive approval pause.

**Goal:** Prepare an optional estimator that averages away arbitrary positive/negative sample pairing without changing the expected paired-correctness objective.

**Architecture:** Add an explicit optional estimator to the repository's paired trainer. Preserve the default diagonal estimator and all sealed source037/038 copies. Keep collection, optimizer settings, checkpoints and budgets unchanged; choose credit per response, not just per paired sample.

**Tech stack:** Existing Python, PyTorch, PEFT and native Transformers collector; CPU fixtures only during preparation.

**Spec:** `PAIRED-REWARD-PAIRING-VARIANCE.md`, corrected additive analysis002, and the existing `SUFFICIENCY-RL-IMPLEMENTATION-PLAN.md` contracts. This plan is not GPU acceptance or a novelty claim.

## Global constraints

- Exactly16 parents ×(4 positive draws +4 negative draws) per block means128 calls, not a16-way resampling expansion. Each draw sees only its own public variant.
- Preserve8 fixed blocks,1,024-call cap,45-minute cap, joint32 warmstart, fresh Adam2e-5, zero weight decay, clip1, dropout0 and native/replay T=.8.
- Raw product objective remains `Pr(positive exact answer) * Pr(correct negative refusal)` conditional on each parent. Malformed returned JSON earns zero marginal success; missing/native-failed responses remain unknown and stop the owner.
- Continue to divide summed per-response losses by64; coefficients below intentionally omit the audit's per-parent `/4`, which this batch denominator already supplies.
- No GPU, new data selection, source-seal edits or launch acceptance during this implementation. No unchanged default owner is rerun.

## Review focus

1. A mispaired group with positive successes `[1,0,0,0]` and negative successes `[0,1,0,0]` must receive nonzero averaged credit even though all four diagonal rewards are zero.
2. If either marginal has no successes, the corresponding product derivative must follow the formula; if every selected coefficient is zero, Adam must not move from hidden momentum.
3. Wrong positive content with a correct answerability label is still positive success0; use the official alias-aware grader, not label-only reward.
4. A negative side that is uniformly correct has zero negative-side credit, while the positive side retains ordinary leave-one-out credit.
5. Saved likelihood audits, zero-block cursors and source/PLAN identity must use the selected coefficients consistently. EOS is included only when actually emitted; native exceptions are not converted to reward0.

## Task1: Explicit response coefficients and focused tests

Files: modify `rl_sufficiency.py`; extend `test_rl_sufficiency.py`. Use a pure helper such as `response_advantages(groups, estimator)` returning one `(positive, negative)` coefficient pair per candidate in each parent group.

- [ ] Add a failing fixture for the mispaired-success case and a direct24-permutation expectation fixture with nonconstant scalar score features.
- [ ] Implement the following coefficients, with four successes per side and no standard-deviation normalization:

```python
def marginal_credit(positive, negative):
    pmean = sum(positive) / 4
    nmean = sum(negative) / 4
    return [
        (nmean * (p - (sum(positive) - p) / 3),
         pmean * (n - (sum(negative) - n) / 3))
        for p, n in zip(positive, negative, strict=True)
    ]
```

- [ ] In collection, compute each positive marginal with the official score against a synthetic known-correct negative; compute the negative marginal from its parsed official answerability requirement. Save both marginals and assert their product equals the existing paired reward. Never put these host scores in a model prompt.
- [ ] For default `diagonal`, return `(a,a)` for each current RLOO advantage. For explicit `pairing_mean`, return the formula above. Reject unknown estimator names.
- [ ] Cover all-zero marginals, all-one negative marginals, malformed numeric answers and native unavailability using the existing parse/reward fixtures.

## Task2: Integrate only the selected credit into the existing trainer

- [ ] Add CLI `--estimator` with choices `diagonal,pairing_mean`, default `diagonal`; use `getattr(args, "estimator", "diagonal")` for existing programmatic callers. Reject `pairing_mean` in SFT mode, where it has no meaning.
- [ ] Record estimator and its exact loss definition in PLAN. For each sampled block, compute coefficients once and pass them to effective-group counting, state advancement and optimization. Preserve diagonal reward summaries separately; they remain valid observations, not the averaged estimator.
- [ ] Apply response-specific loss `-advantage * emitted_token_logps.sum() / 64`; preserve original sampling, replay, clipping, optimizer and checkpoint code. Save each actual coefficient with each likelihood audit. Record selected effective groups and comparable two-response absolute coefficient mass.
- [ ] Extend a tiny real-PEFT CPU fixture to show nonzero adapter movement in a mispaired group, then an all-zero block after populated Adam state leaves weights and optimizer state unchanged. Pin the default estimator's coefficient/loss behavior to the existing diagonal fixture.
- [ ] Run only relevant paired-training tests and Ruff from the worktree. Report source diff, fixture results and any necessary analyzer changes; do not claim old analyzer002 qualifies the new estimator.

## Decision before a GPU run

Main reviews the pending039/046 evidence and the corrected CPU audit. If accepted, use a new immutable source/output, start from the same joint32 warmstart (not the037 endpoint), and collect fresh on-policy samples with the same declared seeds and fixed parent blocks. Later trajectories can diverge as policies change. Compare clipping, actual credit and held behavior, not only TRAIN reward. This is one exploratory training seed, not a replicated algorithm result. An additional readout must explicitly authenticate the new endpoint; do not reuse an old decision file or silently replace a checkpoint.

The raw-gradient conditional-expectation identity does not guarantee improvement after clipping/Adam. Saving roughly ten otherwise inactive parent groups is a plausible small intervention, not a cure for pervasive wrong answers or missing task competence. If the eventual result is uninformative, prioritize the action-learning/recursive-task experiments rather than extending this estimator indefinitely.
