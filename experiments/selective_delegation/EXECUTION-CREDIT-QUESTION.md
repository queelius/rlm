---
question_id: SD-EXECUTION-CREDIT
status: accepted_queued
updated_utc: 2026-09-21T16:06:00Z
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

There is a cleaner possible use of D: learning whether to execute an already
written plan. For that binary decision, D estimates the difference between the
two available actions; extra execution cost can be charged explicitly. This is
distinct from rewarding the planner for producing a large D. A later learned
gate would need fresh training examples, input features available before helper
execution, and comparison with always-cheap, always-execute, and a simple public
plan-length rule. An oracle choosing with saved gold-scored outcomes is only
headroom, never a deployable policy or evidence that such a gate can learn.

Accepted at16:06UTC after the stopped-batch audit reinforced this question.
`EXECUTION-CREDIT-DECISION-001.json` binds sealed `source-023-execution-credit`
and the CPU-prepared output PLAN. Supervisor47179 waits for the ALFWorld screen,
then collects at most330calls/20minutes and performs the native paired analysis.
Eight focused fixtures passed, including actual saved RL/frozen request formats
and unavailable-control accounting. All320 source finals and ten replay controls
are available. No new diagnostic outcomes existed at acceptance.

There are16parents and15atomic components, not16verified independent examples.
Source acquisition contains1,109deduplicated historical calls, accounted
separately from new controls/finals. No optimizer/root/helper generation is
accepted here; existing training, sealed sources and reports remain unchanged.
