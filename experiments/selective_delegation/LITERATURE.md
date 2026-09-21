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

## Follow-up literature check: September 21, 10:23 UTC

These primary sources were read while the GPU ran validation. They constrain
our novelty claims; they are not evidence that our own experiments work.

### Planning and execution are already studied separately

[GlobalRAG](https://arxiv.org/html/2510.20548v1) trains with plan-structure,
plan-meaning, subgoal-completion, format and final-answer rewards. Its training
uses teacher trajectories and a changing balance between process and outcome
rewards. Our finding that plans can be ignored is therefore not a new failure
category. The useful local question is narrower: does our particular frozen
executor make alternative plans produce sufficiently different outcomes for
terminal-reward learning? The accepted16-parent, four-plan RL pilot tests this
on one A100, with a three-hour overall cap. If rewards are flat, intermediate
supervision is a candidate comparison, not an automatic success claim.

### Learning to stop searching is not new by itself

[FrugalRAG, ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ede6f43d254731152970009c172d5561-Abstract-Conference.html)
starts with supervised exploration and uses RL to reduce retrieval steps while
balancing answer quality and cost. Thus “use SFT, then learn decomposition depth
with RL” is not a sufficient novelty claim. Our first RL test deliberately uses
only answer reward to diagnose learning before adding a cost trade-off. A later
cost-aware comparison would need actual token/call accounting, competitive fixed
budgets, and an advantage beyond simply shortening every plan. Do not launch a
large depth sweep before the present variance and held-out checks.

### Executable plans and role-by-role training are also prior art

[PyRAG](https://arxiv.org/html/2605.12975v1) represents retrieval and answering as
Python programs with explicit intermediate variables. Its appendix describes
training the answer role before planning and decomposition, with other roles
frozen, because later decisions depend on the executor's quality. It also reports
cases where final aggregation misuses correct intermediate values. Our explicit
question binding and the Hudson example therefore do not establish novelty.

The practical implication is to test which component limits improvement. If the
planner's rewards stay flat, a matched stronger-helper or helper-training test
may be more informative than increasing planner RL dose. A small comparison
could reuse fixed plans on32 parents and replace only the helper model, recording
fresh final calls and changed compute (one A100, roughly20–60minutes). Promote
that direction only if stronger helpers expose a repeatable planning advantage;
otherwise reconsider the task and information-access constraints. The published
[implementation](https://github.com/GasolSun36/PyRAG) was subsequently cloned for
read-only inspection at commit `5d8ab2ea10b9bf3da1ab2581c8ad5aa93d5df263` into
`/project/alex_phd/research-cache/repos/PyRAG-inspect-20260921`. No project-level
license declaration was found; no code was executed, installed, or redistributed.
Its final synthesis can omit original documents. Our queued frozen-trace
comparison tests that information-access difference locally; it does not claim
the no-document interface is new. See [AGGREGATION-PLAN.md](AGGREGATION-PLAN.md).

## Counterfactual credit check: September 21, 10:51 UTC

[C3](https://arxiv.org/html/2603.06859v1) already compares alternative messages
under identical saved contexts and frozen downstream policies, then uses
leave-one-out return differences. Its method also discusses shared decoding
seeds and repeated continuations. Therefore our common-seed root-plan RLOO is
not a novel credit-assignment algorithm, and frozen-context replay itself is
not a publication claim.

The actionable question is whether a candidate plan's apparent advantage
survives new downstream samples. Our queued aggregation probe supplies two
new finals per frozen trace, but cannot separate plan quality from the already
sampled helper answers. If final-repeat noise is small but ranking remains
uncertain, rerunning the same plans with fresh helper seeds is the next targeted
comparison (about256 helper/final calls on16 parents, one A100). Promote repeated
reward estimation only if it improves ranking reliability enough to justify
those calls; otherwise spend the budget on more distinct training questions.
