---
question_id: SD-EXECUTION-CREDIT
status: conditional_diagnostic_not_accepted
updated_utc: 2026-09-21T15:48:00Z
depends_on:
  - PLAN-ONLY-FINDINGS.md
  - FROZEN-EXECUTION-FINDINGS.md
  - TRAIN-FIT-FINDINGS.md
---

# Does the planner's reward identify useful helper work?

Our final model can read the original documents. Therefore, a correct final
answer does not establish that executing the proposed subquestions helped.
The completed development comparison already shows that supplying the plan
without helper answers is competitive. This suggests a more specific diagnostic
before changing the training objective: **does the reward distinguish plans
whose execution helps from plans that merely give the final model useful wording?**

## Smallest useful comparison

Reuse the first TRAIN batch's 64 frozen supervised plans: 16 parents, four
candidate plans each. We already have their original executed outcomes and
four additional executions with different downstream seeds. For every plan
and each of those five seed settings, generate a final answer with the helper
report emptied, retaining the same question, documents, plan, final seed,
temperature, output cap, and base model. No new root/helper generation or
optimizer update is needed. This is 320 additional finals. Ten fixed factual
replays (first two sorted parents, candidate zero, all five settings) would
check the native prompt/token path. Maximum330calls and20minutes.

Keep all selected parents and candidates, including any invalid execution
outcomes; protocol failures remain distinct from unavailable generations. The
first batch was selected before this question, not for a favorable effect.
These are exposed TRAIN cases and a diagnostic, not generalization evidence.

For candidate i, let E_i be the executed answer's correctness and P_i the
plan-only answer's correctness. The measured benefit is D_i = E_i - P_i.
Within each four-candidate group, compare the existing terminal-reward
advantages with the advantages formed from D. Report how often the two agree,
disagree, or provide no within-group signal, separately for both-valid outcomes.
Repeat this comparison across the five saved seed settings. Seed repetitions
are not independent parents; neither are the four candidate plans.

Also report absolute accuracy, useful/harmful helper reports, strict output
validity, and actual new-call cost. A report improving E but not D can still be
valuable for answering questions; this diagnostic is about what its reward
identifies, not redefining a good answer after seeing it.

## What it would change

If ordinary reward and execution benefit largely agree, an elaborate new credit
scheme has little motivation here. If they often disagree, that would justify
a bounded training comparison, not establish that a new objective will work.
If almost all outcomes are insensitive to execution, prioritize tasks where
requesting help changes available information instead of adding more RL steps
on fully visible short contexts.

An action-dependent counterfactual is **not an ordinary unbiased baseline**:
optimizing E-P changes the objective. It could favor a modest answerer with a
large relative helper gain over a better absolute answerer. Any later experiment
must explicitly specify that tradeoff and retain absolute accuracy and direct
answering controls. Counterfactual credit assignment itself is established
prior art; the proposed diagnostic is not a novelty claim.

This document records a conditional question. No collector or GPU run is
accepted by its existence. Do not modify ongoing RL training, sealed sources,
or existing reports to implement it.
