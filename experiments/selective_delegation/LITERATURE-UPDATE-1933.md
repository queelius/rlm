---
status: primary_literature_input_not_experimental_result
checked_utc: 2026-09-21T19:33:00Z
question: Can experience identify which part of a delegated task needs improvement?
gpu_acceptance: none
---

# Separate explanations from tested causes

[CHIME, version1](https://arxiv.org/html/2609.02074v1), submitted September2,
maintains separate planning and execution memories. A frozen model reviews a
trajectory and attributes its outcome before updating those memories. Its
reported checks include repeated attribution and rerunning tasks with the
resulting advice. This is direct prior work for learning reusable guidance from
stage-specific failures, not a new idea for us. Its backbone models and task
interfaces differ from our small local-model experiments.

Our interpretation: repeatable diagnoses are not automatically correct causal
diagnoses. Helpful advice also does not prove that its accompanying explanation
identified the original cause. We should distinguish model-written explanations
from interventions that keep one component fixed and change another.

## Consequence for our next decisions

Our household traces already contain goals that name a relevant action while
the worker repeatedly does something else. That motivates a worker-competence
test; it does not establish that planning was otherwise sufficient. The accepted
action-training comparison changes the worker while keeping the manager model
unchanged. Retaining a trained flat policy tests whether any improvement needs
the manager at all.

If an additional memory experiment becomes worthwhile, the smallest informative
comparison would hold the model and task panel fixed and compare an equal-sized
set of TRAIN-derived action advice with planning advice, plus no advice. All
generation and retrieval costs count. Success-filtered or hindsight-generated
advice must be labeled, and held-task outcomes must not generate the advice.
This is a prospective comparison, not a current GPU job or novelty claim.

The more immediate new context-isolation paper is discussed in
[the context-sufficiency feasibility memo](CONTEXT-SUFFICIENCY-FEASIBILITY.md).
Its title should not be read as a universal guarantee that less context helps:
we must first identify which information a subtask genuinely requires.
