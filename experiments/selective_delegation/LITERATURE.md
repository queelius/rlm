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

## Structured planning and revision: September 21, 11:22 UTC

[DecomposeR](https://arxiv.org/html/2605.30824v1) already separates planner and
answerer adapters, trains them in stages, represents plans as dependency graphs,
and lets search observations inform a revised plan. Its planner reward combines
coverage, search, and structural signals; this is not the same objective as our
terminal exact-answer reward. Accordingly, separate-role training, graph plans,
and feedback-conditioned plan revision are not novelty claims for us.

Our narrower diagnostic is whether apparent planner gains survive a fixed,
more reliable execution interface. The in-progress four-hop readout also reveals
forward/self references in otherwise parseable supervised plans. If this persists
in the completed report, a later comparison could test one-shot plans against
incremental next-question decisions under matched information and declared costs.
That would test an extrapolation failure, not introduce plan revision itself.
Do not add structural reward simply because it is easy to score: producing more
valid steps need not improve answers. Finish the frozen helper comparison first.

[HiPER](https://arxiv.org/html/2602.16165v1) separately models subgoal selection,
continuation/switching, and execution. Its learned critics estimate advantages
at two time scales; the experiments concern interactive ALFWorld and WebShop,
not our fixed-document question lists. This is relevant prior art for deciding
when to revise a subgoal, but importing a critic framework is not the next
small comparison. Its reported learning curves also span many more updates
than our four-step probe. Our weak result cannot distinguish an inadequate
training dose from a fundamental limit of terminal-reward planner learning.
If the helper comparison exposes usable answer variation, a predeclared
larger-data/dose test is warranted before rejecting that learning approach.

## Executor training and credit allocation: September21,12:22UTC

[MetaAgent-X](https://arxiv.org/html/2605.14212v1) is especially close prior art
for the current decision. It trains both workflow design and execution, using
four sampled designs and four executions per design, and alternates role-specific
training stages. Its main setup shares weights, with separate-policy ablations.
Thus neither an executor bottleneck nor planner–executor co-adaptation is a new
claim for us. Our current separate, frozen helper makes the incremental root
effect easier to isolate, but differs from its jointly evolving system.

Local implication, not a reported paper result: if the new root-RL readout is
weak, replay the same frozen plans with several fresh helper seeds before adding
tree search or repeated reward estimation. This tests whether plan rankings
survive execution noise. A bounded64-plan ×four-continuation pilot would require
roughly800–1,100 new helper/final calls for typical plans, with a one-A100-hour
cap. Promote repeated execution only if ranking stability improves enough to
justify its cost; otherwise prefer more distinct training questions. This is
conditional preparation, not an accepted additional GPU job.

[SkillGate](https://arxiv.org/html/2608.18852v1) separates skill-selection tokens
from execution tokens and supplies selection-specific credit using a known
correct skill. Its diagnosis includes diluted and wrong-signed outcome credit.
That does not directly diagnose our current runner: only root-plan tokens enter
our loss, so long helper outputs cannot dilute them. We also do not have a unique
known-correct generated plan. Copying its auxiliary objective would add privileged
supervision, not simply repair our implementation. The useful local question is
whether varying downstream execution changes a plan's reward despite unchanged
planning; our fixed-helper and conditional repeated-execution comparisons address
that without inventing a gold planner action.
