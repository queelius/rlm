---
status: exploratory
evidence_cutoff_utc: 2026-09-22T08:04:00Z
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

One bounded [credit-assignment control](PAIRING-MEAN-RL-DECISION.md) is queued:
keep the original joint reward, but average over the arbitrary ways of pairing
the sampled positive and negative responses. This asks whether less noisy
credit actually improves the trained policy after the optimizer acts. The CPU
calculation alone cannot answer that question. It is a known estimator idea,
not a new reward or a claimed new algorithm; its readout reuses an exposed
development panel and is explicitly exploratory.

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

A separate supervised run has completed one epoch and 23 updates on 366
query/craft/finish examples from 32 training tasks. Its evaluations are running;
we do not yet know whether it improves task success. They will compare original
and reminder prompts with the fixed trained weights. These demonstrations
teach basic task execution, not recursive planning.

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
but longer input histories. This makes a follow-up teaching comparison feasible;
it is not yet a trained-model improvement. We will use the current crafting
evaluation to decide whether that comparison addresses an observed weakness.

The all-query audit makes the distinction concrete: 135 of the original 167
recipe queries ask about an identifier absent from the preceding public input.
The revised demonstrations have zero such queries. Arbitrary queries are legal;
the problem is not an invalid target or leaked model input. The revised teacher
shows how to discover each needed name, whereas the original often supplies
the name without demonstrating how to find it. Whether this distinction improves
learned behavior remains an experimental question.

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
