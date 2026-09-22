---
status: exploratory
evidence_cutoff_utc: 2026-09-22T10:45:00Z
question: What prevents useful learning and reliable use of additional computation?
publication_status: mechanisms_to_test_not_established_architecture_advantage
---

# September 22: separate learning to answer from learning when to act

## New lead: demonstrate how to discover the information needed to act

Changing the crafting demonstrations raised success from **3 to 10 out of 16
attempts** with the original prompt. The old demonstrations often began by
querying an intermediate ingredient selected using the teacher's hidden plan.
The new demonstrations start from the visible goal and show how recipe replies
reveal the ingredients to investigate next. Both training runs use the same
32 tasks, 366 action examples and 23 updates; the histories and token counts differ.

For a simplified illustration, suppose the task is “make a desk.” The old teacher
may begin by asking how to make a plank because it already knows the desk recipe.
The new teacher first asks for the desk recipe, learns that planks and legs are
needed, and then investigates those items. The student sees how the next subproblem
was found, not just which subproblem an informed expert chose. This illustration
uses familiar names; the actual benchmark uses generated item identifiers.

The comparison gains eight previously failed attempts and loses one previously
successful attempt. Its task-cluster uncertainty interval is wide: approximately
6 to 81 percentage points for the net improvement of 44 points. These are eight
previously examined tasks sampled twice, not 16 independent tasks, and only one
training seed. With the separate reminder prompt, success rises from 2 to 10
out of 16. All outcomes were replayed against the environment's unchanged rules.

This is evidence for the revised **teaching procedure as a package**, not proof
that query order alone caused the gain or that recursion helped. No helpers ran.
The short procedural-instruction control is now complete: the old trained model
solves **0/16**, versus3/16 without those added instructions. It uses850 model
calls,770 of them recipe queries, with no transport failures. This one fixed
instruction package does not recover the demonstration gain; it is not a test
of every possible prompt. The [control report](TEXTCRAFT-PROCEDURE-CONTROL-FINDINGS.md)
records all three paired losses and the uncertainty interval.
The changed-recipe comparison is now complete. In a newly generated recipe
world, the revised-teaching model again solves **10/16**, versus **3/16** for the
old teaching procedure. Here the paired comparison has seven wins and no losses;
the task-cluster interval for the improvement is **12.5 to 75 percentage points**.
Both models receive the same new recipes through tool replies and the same
starting supplies. This strengthens the evidence beyond the original recipe
world, but it still uses the same eight goal names and one new world. We changed
dependencies and starting supplies, so scores across the two worlds are not a
controlled comparison of equally difficult tasks.

The revised model uses fewer calls in the new world (547 versus700), but more
summed model-call time (about24 versus20 minutes). Better task performance is
therefore not an automatic speed improvement. No helpers run in either arm.
See the [changed-world result](TEXTCRAFT-CHANGED-WORLD-FINDINGS.md).

A paired comparison on16 newly selected goal names in the original world is
now accepted and queued. Its tasks were frozen before collecting model outcomes;
most still share intermediate recipes with training. Separately, the GPU is
sampling four attempts on each of eight training tasks to check whether terminal
success gives RL useful within-task reward variation. That is a readiness test,
not an RL improvement. A small two-update pilot is being prepared conditionally.
See the [detailed finding](TEXTCRAFT-PUBLIC-DISCOVERY-FINDINGS.md).

The [remaining-failure audit](TEXTCRAFT-REMAINING-FAILURES.md) separates learning
what to investigate from using the discovered recipes correctly. Several failed
attempts already have the relevant recipe but use the wrong quantities or act
before making prerequisites. The environment also labels recipe existence as
`can_craft`; that field does not check current inventory. A plain-language
clarification is therefore a cheaper prospective control than adding a planner.
The [imitation-gap literature review](LITERATURE-IMITATION-GAP-20260922.md) explains
why the broad teacher-information mismatch is prior art and what stronger
transfer evidence would be needed for a research contribution.

## Earlier findings and the motivation for that comparison

The latest evidence suggests two different problems. Training can make the model
less reluctant to answer, without reliably teaching it when the evidence is
enough. In an interactive task, the model can make real progress and then fail
to recognize that it should stop. After action training, its main failures instead
involve guessed recipe names, wrong intermediate products and stopping too early.
Neither problem is solved merely by allowing more helper calls.

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

The reward-control experiment is now complete. Instead of rewarding a candidate
only when both sides are correct, the new run rewards each side's success equally.
On a separately frozen mixed-depth panel, the starting model gets 6 correct pairs
out of 64, the original RL model gets 10, additional supervised training gets 10,
and the new RL model gets 4. The new reward loses six pairs and gains none against
the original RL model. Its estimated difference is −9.4 percentage points, with
a paired component-cluster bootstrap interval of −19.0 to −1.7 points.

This is not simply a failed software run: all eight updates and all 512 evaluation
responses were verified. The new model answers fewer unsupported questions
(2 versus 8), but refuses more supported questions (52 versus 40). Thus more
frequent reward does not automatically teach the desired balance. This remains
one training seed and a small exploratory panel, not a general verdict on RL.
The original RL model still does not outperform additional supervised training.
See the [completed comparison](PAIRED-REWARD-CONTROL-FINDINGS.md).

The [training audit](PAIRED-ADDITIVE-TRAINING-FINDINGS.md) distinguishes an earlier
offline projection of 98 active groups from the 54 actually observed under the
new policy. It also verifies that identical first samples produced materially
different parameter-update directions. See the [objective control](PAIRED-ADDITIVE-RL-PLAN.md)
for the prospectively fixed comparison.

The [credit-assignment control](PAIRED-PAIRING-MEAN-FINDINGS.md) is also complete.
It keeps the original joint reward but averages over ways of pairing the sampled
responses. The first update uses identical samples and starting weights, so we
can verify that it changes learning from the same information. Across training,
more question groups receive nonzero credit, but fewer individual responses do.

That change does not help the evaluation: the new model scores 6 correct pairs
out of 64 versus 10 for the original RL model, with four losses and no wins.
It again refuses more supported questions. The estimated difference is −6.25
percentage points, with a component-cluster interval of −12.90 to −1.47 points.
This is one training seed on an already exposed panel, not a general test of
variance reduction. We are retiring this variant rather than extending it
without a new hypothesis. Neither reward frequency nor additional credited
groups has proved sufficient to improve this task.

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

The completed reminder comparison tells the same model to use current inventory,
return one action, and finish when the goal is met. It solves none of 16 trials.
On the 13 trials with observed original-baseline outcomes, malformed replies
fall from 118 to 4, but native action errors rise from 212 to 692. Three reminder
trials reach sufficient target inventory; none chooses to finish. Better format
does not establish better task performance. See the [qualified comparison](TEXTCRAFT-INSTRUCTION-FINDINGS.md).

A separate supervised run completed one epoch and 23 updates on 366
query/craft/finish examples from 32 training tasks. Its completed evaluation
solves 3/16 attempts with the original prompt and 2/16 with the reminder. Against
the original baseline, there is one observed win, no loss and three unknown
comparisons. Against the complete reminder baseline, there are two wins and no
losses; the estimated gain is 12.5 percentage points, with an eight-task bootstrap
interval from zero to 31.25. These are modest, uncertain gains, not general planning.

The failures now tell a different story. All trained replies satisfy the JSON
schema, but none of the 27 failed attempts ever achieves the public item goal.
The original-prompt model finishes unsuccessfully 11 times; the reminder model
often loops through recipe queries instead. A rule that finishes once enough
items exist would therefore rescue none of these trained failures. In one
trace, the model uses all available ore making the wrong intermediate, then
discovers the actual goal recipe too late. See the [complete paired analysis
and concrete examples](TEXTCRAFT-ACTION-SFT-FINDINGS.md).

A comparison of the saved actions reveals a sharper result: training improved
formatting while damaging information gathering. On the 13 original-prompt
trials where both models have recorded outcomes, the base model begins by asking
for the goal's recipe in all 13; the trained model does so in only two. With the
reminder, that falls from 16 of 16 to three of 16. The base model makes no queries
for nonexistent items; after training, 344 of 522 original-prompt queries and
593 of 1,078 reminder queries ask about nonexistent items. These are repeated
actions, not independent examples. This links the regression to the trained
adapter, but does not yet prove which feature of the demonstrations caused it.

A [training-data audit](TEXTCRAFT-TRAINING-COVERAGE.md) sharpens that limitation:
30 of 32 demonstrations begin with an intermediate recipe chosen by the
teacher's hidden plan, before the model has seen how that ingredient connects
to the goal. These are legitimate supervised targets, but they do not teach
a complete procedure for discovering the prerequisites. None of the eight
evaluation goals appears as a training recipe, although 21 of their 47 required
recipes do. Better performance would therefore be useful evidence of learning,
not by itself proof of general decomposition or unfamiliar-world transfer.

A CPU-only [public-discovery teacher](PUBLIC-DISCOVERY-READINESS.md) now completes
all 32 training tasks by querying the goal recipe and discovering prerequisites
from actual replies. It produces the same number of action examples naturally,
but longer input histories. The observed discovery failures now motivate an
follow-up training comparison using these demonstrations, with the same
model, training tasks, 23 updates and fixed final-checkpoint rule. That comparison
is now complete, with the improvement reported above. The changed teacher
histories are not perfectly token-matched. The original-prompt comparison was
primary, not whichever prompt won.

The all-query audit makes the distinction concrete: 135 of the original 167
recipe queries ask about an identifier absent from the preceding public input.
The revised demonstrations have zero such queries. Arbitrary queries are legal;
the problem is not an invalid target or leaked model input. The revised teacher
shows how to discover each needed name, whereas the original often supplies
the name without demonstrating how to find it. The revised teaching package now
improves the small exposed panel; isolating the mechanism and testing transfer
remain experimental questions.

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
