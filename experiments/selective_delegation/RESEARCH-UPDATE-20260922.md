---
status: exploratory
evidence_cutoff_utc: 2026-09-22T06:22:00Z
question: What prevents useful learning and reliable use of additional computation?
publication_status: mechanisms_to_test_not_established_architecture_advantage
---

# September 22: separate learning to answer from learning when to act

The latest evidence suggests two different problems. Training can make the model
less reluctant to answer, without reliably teaching it when the evidence is
enough. In an interactive task, the model can make real progress and then fail
to recognize that it should stop. Neither problem is solved merely by allowing
more helper calls.

## Training helped on easier questions, but the joint gain did not clearly transfer

Each test asks the model to handle two versions of a question: one with enough
supporting evidence, and one labeled unanswerable. A correct pair requires both
a correct supported answer and a refusal on the unsupported version.

| Fixed model | Two-step question panel: correct pairs /64 | Three/four-step panel: correct pairs /64 |
|---|---:|---:|
| Supervised starting point | 10 | 6 |
| Eight additional RL updates | 18 | 7 |
| Eight additional supervised updates | 19 | 6 |

Both continuations improve the first panel. RL does not establish an advantage
over additional supervised training. On the deeper panel, neither continuation
clearly improves joint correctness. Both answer more supported questions
correctly but also answer more questions labeled unanswerable. The two panels
differ in more than the number of reasoning steps, so this is not a controlled
measurement of the effect of depth alone.

Each panel contains 32 questions sampled twice, not 64 independent questions.
Some document content and earlier evaluation components recur. These are small,
exploratory comparisons using one training seed. Full uncertainty estimates,
costs and examples are in the [first held comparison](PAIRED-RL-HELDOUT-FINDINGS.md)
and [deeper comparison](PAIRED-RL-COMPOSITIONAL-FINDINGS.md).

The next accepted RL run changes one thing: instead of rewarding a candidate
only when both sides are correct, it rewards each side's success equally.
Saved samples suggest this would provide a learning signal on 98 of 128 training
questions rather than 42. However, 46 of the 56 newly active questions contain no
correct supported answer among their samples. More feedback could mainly teach
refusal, not better reasoning. A new 32-question panel was frozen before any new
updates to test this possibility. See the [objective control](PAIRED-ADDITIVE-RL-PLAN.md).

## Interactive failures reveal a concrete harness question

In the crafting environment, a model must query recipes, make ingredients, and
explicitly finish. In one example it needs one target item and already has three,
but continues working until it runs out of context. Across the observed pilot,
four failed episodes had already reached sufficient inventory but never finished.
These are still failures under the unchanged task rules, not corrected successes.

The pilot also produces many multi-action replies where the interface requires
one action. No recursive helper was called at all. Its score differences therefore
cannot demonstrate a benefit from recursion. The run reached its time cap; seven
of 32 planned outcomes are missing or unknown. See the [trace-based analysis](TEXTCRAFT-PILOT-FINDINGS.md).

The running comparison gives the same model a clearer final reminder to use the
current inventory, return one action, and finish when the goal is met. It keeps
the strict parser, tools and budgets unchanged. A fixed small supervised run is
queued separately: 366 public query/craft/finish examples from 32 training tasks,
one epoch, 23 updates. The follow-up will distinguish a prompt improvement from
a weight improvement. These demonstrations teach basic task execution, not
recursive planning.

On the household-task benchmark, the completed action-training run did not help:
both flat and manager/worker policies solve 2 of 24 attempts after training versus
4 of 24 before it. We are not extending that recipe without a new explanation.
See [the completed action-training comparison](ALFWORLD-ACTION-SFT-FINDINGS.md).

## What would make this a stronger research contribution?

A useful result would show that a particular change helps a model use additional
work effectively, survives a strong cheap baseline, and transfers beyond the
examples that motivated it. Our immediate priorities are identifying informative
RL feedback and making reliable task-state decisions. A fast classifier, trained
LLM, or recursive planner is a possible component—not an assumed answer.

Replacing raw history with explicit state is already prior art; for example,
[SKILL.state](https://arxiv.org/html/2608.26263v3) studies that design. Our own
harness already supplies current inventory, so a future state-representation
experiment must distinguish useful retained evidence from distracting action
history, rather than claim to invent state tracking.

These experiments isolate parts of the RLM problem. They do not yet evaluate a
general Python-based recursive problem solver or establish a new architecture.

## Operational checkpoint

The GPU was idle for approximately eight hours after the capped crafting run
blocked an unrelated queued evaluation. That was lost research time. The
evaluation has now completed, and a tested replacement queue distinguishes GPU
release from scientific success. Training checkpoints and native records remain
in the external research store; source and findings are periodically pushed to
the research branch. GitHub is not a backup of the model checkpoints.
