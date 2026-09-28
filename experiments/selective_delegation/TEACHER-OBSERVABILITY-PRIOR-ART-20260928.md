---
date: 2026-09-28
question_id: TC-TEACHING-MECHANISM
status: primary_source_screen
related_evidence: finding_textcraft_breadth_20260928.md
assets_downloaded: none
---

# Why usable demonstrations are a research question, not a novelty claim

Our broad result now has unusually clean answer-side matching: both teaching
packages supervise the same action multiset and exactly 8,820 answer tokens.
The teacher with advance recipe knowledge frequently names an intermediate
before the learner has seen that name. The discovery teacher first obtains
public information. The resulting model behaviors differ sharply on new goals.
Conditioning history and input-token exposure still differ, so a controlled
reordering experiment is needed before assigning a cause.

## Closest prior work checked today

- [Student-Informed Teacher Training](https://arxiv.org/abs/2412.09149) directly
  addresses teachers whose privileged observations lead to behavior a partially
  informed learner cannot imitate. It adapts teacher training to the learner.
  Thus the general teacher/student information mismatch is established prior art.
- [Guided Policy Optimization under Partial Observability](https://arxiv.org/abs/2505.15418)
  co-trains a privileged guide and a learner, constraining their alignment. Its
  existence also rules out claiming that making a teacher imitable is new.
- [Privileged Information Distillation for Language Models](https://arxiv.org/abs/2602.04942)
  studies this problem in multi-turn language agents, including action-only
  demonstrations. Its proposed joint teacher/student and on-policy objectives
  are closer alternatives than generic SFT followed by an unrelated RL sweep.
- [Causal Confusion in Imitation Learning](https://arxiv.org/abs/1905.11979)
  shows that observational imitation can learn misleading dependencies and uses
  interventions to distinguish them. This is related, but it is not the same
  as our missing-information-before-action diagnostic.

These are primary-source abstracts checked on September 28, not full replications
or exhaustive literature coverage. Consult the full methods before implementing
their objectives. The separate [current methods review](LITERATURE-OPPORTUNITIES-20260928.md)
covers recursive agents and action/argument learning.

## The smaller, falsifiable contribution we can pursue

Can an offline transformation make an expert trace easier to learn by changing
**when an action is demonstrated**, while preserving the task, actions and
per-minibatch supervised targets? Compare the original sequence, a minimally
reordered information-available sequence, and a different legal query order.
Regenerate every public observation through native execution.

This tests conditioning histories, not merely additional examples or bigger
updates. It remains an offline schedule derived from an expert trace, not proof
that the scheduling program uses only public information to choose all actions.
More training for the original teacher is an important competing explanation.

An existing [paired-world CPU experiment](TEXTCRAFT-OBSERVATIONAL-AMBIGUITY.md)
already showed identical starting observations with different privileged first
labels, while public querying solved both worlds. Do not rerun that diagnostic
as a new discovery. The useful next step is a model-training intervention and,
if positive, replication in another partially observed tool task.

The experiment can fail constructively: no benefit from repaired histories would
weaken the observability explanation; benefit from several legal orders would
support it more than one specially chosen demonstration sequence. A paper needs
that mechanism and transfer evidence, not just the existing large package gap.
