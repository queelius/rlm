---
title: "What the literature changes about the next experiment"
date: 2026-09-29
status: targeted_primary_source_review_not_a_novelty_proof
retrieved_utc_date: 2026-09-29
scope: "Teaching, sparse feedback, recursive credit, and harness composition"
---

# Use the literature to sharpen comparisons, not to start another large framework

This is a targeted update to the existing
[teaching prior-art review](../../experiments/selective_delegation/teaching_prior_art_20260929/README.md)
and [learning-decision review](../../experiments/selective_delegation/literature_decisions_20260929/README.md).
No paper result below was reproduced here. No new package, model, dataset or
GPU experiment was installed or launched for this review.

## 1. Teacher information mismatch is an established problem

[Guided Policy Optimization, v2, March 13, 2026](https://arxiv.org/html/2505.15418v2)
co-trains a privileged guide with a partially observing learner and constrains
their separation. Its didactic examples and method already address why an
informed teacher can be difficult or inappropriate to imitate. Read sections
2.2, 3 and 5; the general motivation is not our novelty. Our narrower candidate
is an executable repair of existing histories while retaining the taught
answers, tested at fixed update and answer-token budgets. That distinction
still needs stronger transfer evidence and a fuller closest-method comparison.

Decision: prioritize the already accepted second-model repair comparison,
then an untouched goal/world split and the previously proposed consistent
item-name permutation control. Do not call the current result proof that
hidden information alone caused the gain.

## 2. A new credit-assignment paper does not remove the need for useful experience

[Reinforcement Learning with Decomposed Subtasks, v1, September 22, 2026](https://arxiv.org/html/2609.27035v1)
uses fixed subtask categories, self-grading, reflect-and-retry rollouts and
subtask-specific token credit. Sections 3.2–3.4 distinguish that package from
merely changing a scalar reward. Section 6 notes fixed taxonomies and weights;
Figure 2's intervals describe evaluation sampling, not training-seed variation.

Our inference from its reward formula: redistributing a terminal reward of
zero still produces zero subtask rewards. If every original and retry in a
group fails with zero reward, this redistribution alone cannot fix our flat
groups. Successful retries or a different valid signal would be needed. Thus
the next useful comparison is recovery experience versus ordinary experience,
not immediate transplantation of the full estimator. Reconsider structured
credit if we first obtain contrasting successful and unsuccessful continuations.

## 3. Recursive RL is already published work; our question must be narrower

[Recursive Agent Optimization, v1, May 7, 2026](https://arxiv.org/html/2605.06639v1)
already trains recursive delegation with local task success and depth-aware
training. Its TextCraft-Synth setting uses no delegation bonus. The reward
and trajectory-weighting ablations in section 5 test ingredients absent from
our flat terminal-only pilots. Our null or small flat-policy results therefore
do not contradict its recursive-training results.

Decision: first complete the queued whole-goal flat/fixed/adaptive comparison.
If a helper has measurable utility, investigate when *local* completion fails
to help the parent because it consumes shared materials or budget. A helper's
own success and its contribution to the original task are different targets.
That is a possible focused follow-up, not a demonstrated novel result. A
one-goal feasibility screen cannot itself support a general architecture claim.

## 4. Harness evolution has become a more crowded research direction

[PluginRSI, v1, September 26, 2026](https://arxiv.org/html/2609.32423v1)
separates a harness into reusable mechanisms, changes one mechanism with the
others fixed, and also recombines the surrounding workflow. Read sections
3.1–3.2 and Appendix D. The authors flag weaker transfer across dissimilar
domains and changes that require coordinated modifications as limitations.

Decision: do not pitch “evolve the harness” as a contribution by itself. Our
smaller opportunity is to measure which responsibilities should belong to
code versus learned decisions, and how that division changes learning. Recipe
binding already improves execution; a reliable learning benefit remains
unproven. The queued argument-mask and execution controls are closer to this
question than launching an unrestricted harness search.

## A search result that should not redirect the project

The September 26 revision of
[Break Step, arXiv:2609.11149v2](https://arxiv.org/abs/2609.11149v2)
replaces the older title “A Fragility Spectrum for Recursive Language-Model
Training” and traces a rapid synthetic-data diversity loss to replayed sampling
noise. This is repeated training on generated data, not recursive subagent
delegation. Search vocabulary can conflate those meanings. It is a useful
sampling-audit reminder, not evidence against the RLM architecture or an
identified cause of our current failures.

## Ranked next decisions

| Priority | Question and smallest informative comparison | Decision rule and expected shape |
| --- | --- | --- |
| 1 | Does fixed-answer history repair help a second model family? Finish the accepted Phi comparison before adding another variant. | A replicated benefit warrants a fresh, untouched goal/world test. Failure narrows the claim and raises copying/model-specific explanations. Use the existing job caps, not a duplicate launch. |
| 2 | Can examples from learner-reached error states teach useful recovery? Compare the CPU-qualified A-only recovery examples with matched ordinary A examples, from the same checkpoint. | Promote only with better native completion from ordinary A starts, including previously all-failing goals. The frozen proposal's gate is at least 3/16 net over the ordinary control, gains on at least two previously flat goals, and at most one loss on the previously mixed group. About 2.3–3 hours estimated, 310-minute cap; no B selection. |
| 3 | Does a helper improve the original task under the same total budget? Finish the accepted complete-task pilot. | Check parent success and total cost, not child success alone. Only a useful feasibility signal justifies a larger fresh-goal study or training delegation. |
| Conditional | If recovery teaching fails, do fixed public prefixes create useful RL reward contrast? Compare prefix-start and ordinary-start rollouts, then one fixed update. | Charge prefix work and train only the generated suffix; test from ordinary starts. Roughly 3–4.5 hours, 6-hour cap. This is an established curriculum idea, not a novelty claim. |

These are research priorities, not claims that new GPU jobs have been admitted.
The live owners retain their sealed sources and accepted sequence. The current
session cannot write the external queue; do not mistake this local plan for a
successful scheduling change. Avoid adding temperature, learning-rate, search,
or new-estimator sweeps until one of these comparisons resolves the bottleneck.
