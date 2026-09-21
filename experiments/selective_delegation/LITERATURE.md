---
date: 2026-09-21
status: experiment-design-notes
---

# Relevant literature and what it changes

These are primary-source claims, not results we have independently replicated.
The current pilot is a diagnostic step toward learning useful delegation, not a
claim that learning to delegate is itself new.

## Learned delegation is already an established research direction

[General learned delegation by clones](https://arxiv.org/html/2602.13262v1)
trains shared-weight helpers with global rewards and studies cost-aware delegation,
including MuSiQue. Its protocol-based gradient gating is explicitly biased; a
substring-based estimate of helper usefulness performed poorly. For us, this
supports testing actual interventions at a common decision state instead of
treating mention of a helper's answer as evidence that the helper caused success.
Its reported model is also Qwen3-4B-Instruct-2507. A simple same-model delegation
demonstration would therefore not establish novelty.

[Recursive Agent Optimization](https://arxiv.org/abs/2605.06639) also trains agents
to recursively delegate and reports extrapolation to harder problems. We need a
more specific contribution than "RL can learn recursion": for example, a clear
measurement and training method for deciding when an additional decomposition is
worthwhile, with evidence that this transfers beyond the training distribution.

## A compact decision policy is a reasonable diagnostic

[Learning When to Think](https://arxiv.org/html/2608.20256v2) learns among three
reasoning modes using reward shaping and distinct generation limits. This motivates
our initial small action set. It does not establish that our four interventions
have useful differences, or that a title-index observation contains enough
information to predict those differences. We will test both assumptions before
building a more complicated search algorithm.

## Context isolation may help, but it is a separate intervention

[Recursive Language Models Generalize Out of Domain](https://arxiv.org/pdf/2609.20831)
argues that isolated subtask contexts can exclude shortcuts available to a full-
trace learner. This motivates a later controlled comparison of helper visibility:
full source versus a selected subset, while keeping the final answerer's access
fixed. The current pilot deliberately retains full source everywhere to separate
an additional reasoning step from information loss. It does not test recursive
context isolation or prove compositional generalization.

The retrieved abstract's submission date and identifier month are inconsistent;
do not describe this as a newly submitted September21 paper without verification.

## Experiments enabled, rather than a new infrastructure queue

1. Current: compare finish, reconsider, targeted evidence and proposed subquestions
   from one fixed initial attempt;32 parents,3 continuation seeds,one A100,2h cap.
2. Immediate diagnostic: change only the final answer-format instruction if
   verbose answers contaminate the exact-match reward; same saved reports,1 A100,
   at most384 new final calls,2h cap.
3. Conditional follow-up: train a small observation-only action policy and compare
   with fixed actions on new parents. Promote only if benefits survive real cost
   accounting and fresh continuation samples; retire or revise if one fixed
   action suffices or action-value differences do not repeat.
4. Later: vary helper visibility or add a second decision step. These should
   answer an observed limitation, not merely make the harness more elaborate.
