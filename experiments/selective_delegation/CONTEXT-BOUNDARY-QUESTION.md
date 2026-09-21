---
question_id: rq:delegation_context_boundary
created_utc: 2026-09-21T18:40:00Z
status: prospective_not_gpu_accepted
depends_on: source044_textcraft_interface_qualification
novelty: mechanism_question_not_claimed_new_framework
---

# What should a child agent know about its parent's task?

Delegation changes more than who writes the next action. It also changes which
information the model sees. A fresh child context may remove distracting history,
but it may also omit a constraint needed for the overall task. Our current
TextCraft pilot qualifies the interface; it will not separate these explanations.

For example, a child asked to make two gears can complete that local request yet
use material that the parent still needs for another component. This is an
illustrative scenario, not an observed failure in our pending experiment. A useful
child contract might include both the requested output and a resource reservation.
Whether this matters in the selected recipe world must be measured, not assumed.

## The literature gives competing reasons to share or hide context

[ReCAP, version1](https://arxiv.org/html/2510.23822v1) keeps an evolving shared
context and restores parent plans after returning from subtasks. It argues that
isolated contexts can lose cross-level continuity. Its reported evaluations use
larger commercial models, different demonstrations and interfaces; its success
rates are not comparable to our small-model ALFWorld screen. It is direct prior
art for parent-plan reminders and shared-context recursion, not a new idea for us.

[Recursive Models for Long-Horizon Reasoning, version1](https://arxiv.org/html/2603.02112v1)
instead studies isolated recursive contexts, passing results back while discarding
child reasoning. Its formal constructions and trained SAT experiments motivate
bounded local contexts. They do not establish that arbitrary learned
decompositions or our crafting policy obtain those benefits. In particular,
expressive power does not by itself establish practical sample efficiency.

[Routed Graph Handoff, version1](https://arxiv.org/html/2608.25277v1) investigates
structured versus prose delegation. The authors explicitly qualify their routing
as largely task-type selection, not fine-grained instance adaptation. Its graph
condition also requires an executor instruction explaining the schema. This
reinforces our existing lesson: distinguish an interface-understanding improvement
from better decomposition. We have not reproduced its experiments or audited its
artifacts. Its routing appendix contains apparently inconsistent counts for
AppWorld graph assignments and misclassifications, so do not reuse those counts
as established evidence without checking the underlying artifacts.

These primary versions were inspected on September21. They constrain novelty;
none proves which context boundary will work in our setting.

## Smallest useful follow-up

Only after the pending pilot produces interpretable behavior, compare two recursive
policies with the same model, delegation action, tool interface and global budget:

1. A child receives its local target, current inventory and the parent's chosen note.
2. It additionally receives the parent's goal and public action/observation history.

This tests the value of exposing parent context, not the isolated value of
recursion. Keep the flat baseline. Count extra input tokens and generation time;
equal output limits do not imply equal compute. Use the full preselected panel,
not only cases on which one context variant succeeded. If long histories exceed
the declared context limit, record that policy limitation; do not silently give
one arm a privileged summary or a larger window.

The first small comparison would use one A100,16 new episodes on the existing
eight-task/two-seed pilot panel, with a one-hour cap and complete per-call receipts.
That is an exposed-panel mechanism probe. A positive signal requires new tasks,
another seed, and explicit assessment of whether the local subtask was necessary
for the root outcome before any broader claim.

Promote the question if parent information systematically rescues lost constraints
or systematically distracts otherwise competent children. Revise it if merely
understanding the JSON contract explains the change. Retire it on this task if
delegation is almost never used, all variants fail basic crafting, or the shared
state already exposes every relevant constraint. Do not build a learned context
router until there is a predictable difference worth routing between.
