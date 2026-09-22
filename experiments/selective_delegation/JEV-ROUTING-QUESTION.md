---
title: Can a fast decision model predict when delegation is worth its cost?
date: 2026-09-21
status: prospective_not_accepted_for_gpu
origin: user_food_for_thought
priority: conditional_on_demonstrated_action_headroom
---

# A possible follow-up, not a change of direction

The user suggested Jev and universal classifiers as a fast (System 1) decision
layer, then emphasized that task-specific fine-tuning may be preferable. This
is an assessment only. No Jev API calls, data uploads, or new GPU jobs were made.

The useful distinction is between choosing a kind of work and inventing the work
itself. A fast model could choose answer, inspect, delegate, or stop; a generative
model would still formulate a subproblem or write Python. A fine-tuned language
model can also serve as that fast decision layer. These are not mutually exclusive
architectural categories.

## What the primary sources establish

- [Jev documentation](https://docs.typesafe.ai/introduction) describes typed
  decisions rather than free-text generation. Its
  [model documentation](https://docs.typesafe.ai/models) describes hosted models
  with shared weights, not customer-specific fine-tuning or LoRA. This makes it
  a possible comparison, not our default trainable research component.
- [Jev confidence](https://docs.typesafe.ai/confidence) describes concentration
  of an output distribution. This does not establish calibrated probabilities
  of successful recursion on our tasks. Its
  [documented limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13)
  include multi-hop indirection, arithmetic, and irrelevant long states.
- [Universal classifiers via natural-language inference](https://arxiv.org/abs/2312.17543)
  provide an open, efficient classification approach; they do not establish
  equivalence to Jev or competence at our recursion decisions.
- [RouteLLM](https://arxiv.org/abs/2406.18665) already learns economical choices
  between language models. Generic learned routing is therefore not a novelty
  claim. Recursive, state-dependent decisions need their own prior-art review.

## Falsifiable question and smallest useful comparison

Can a small learned decision model predict when an additional helper or recursive
step improves the final result enough to justify its cost, including when the
solver's abilities or the task distribution change?

First check whether available actions actually have complementary strengths.
Routing cannot create useful delegation if helpers seldom improve the result.
Existing exposed panels may support exploration, not a fresh confirmation claim.

If there is headroom, use frozen public decision states with paired downstream
action outcomes. Keep gold answers and future outcomes out of router inputs.
Compare the best fixed action, a cheap rule, an open classifier, and a small
fine-tuned language model; Jev is an optional external comparison. Separate
training, calibration, and held evaluation by problem/source groups. Repeat
stochastic continuations where needed: one lucky outcome is not a reliable label.

Measure final task success and total cost, including feature extraction,
summarization, and routing. Classification accuracy or confident outputs alone
are insufficient. Begin with CPU analysis of existing paired outcomes; only
accept a bounded GPU training/readout proposal after useful headroom is visible.

Promote the idea if held-out routing beats strong fixed and simple-rule controls
at comparable total cost. Revise or retire it if gains vanish after overhead,
depend on leaked future information, or fail when the executor changes.

## Current decision

Continue the accepted training and evaluation queue. Prefer controllable,
task-trained components for any first learned-routing experiment. Preserve Jev
as a possible baseline, not a new dependency or an assumed architectural advance.
