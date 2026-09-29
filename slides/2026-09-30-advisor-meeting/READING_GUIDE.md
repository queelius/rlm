---
title: "Teaching usable next steps: presenter reading guide"
meeting_date: 2026-09-30
prepared_date: 2026-09-29
evidence_cutoff_utc: "2026-09-29T05:30:50Z"
teaching_and_helper_evidence_cutoff_utc: "2026-09-29T01:14:27Z"
supplementary_reward_diagnosis_date: "2026-09-29"
status: exploratory
---

# Read this first: the whole story

This is a guide to understanding the discussion, not a script to memorize. The
five concepts below explain the eight-slide meeting story. The questions afterward
are optional preparation. Earlier decks are historical records; you do not need
to learn them to explain this update.

If you only read one paragraph: we are studying how a small model can learn
useful steps for an unfamiliar task. The strongest result comes from improving
what the learner sees before a demonstrated action, while keeping that action
the same. That helped in repeated small tests. Reward training helped familiar
goals, but barely helped different goals so far. Next we need stronger transfer
tests and a fair test of whether a helper improves the entire task. The evidence
is promising but preliminary, and it does not yet show a general recursion benefit.

## 1. Why study small steps before recursive models?

Our long-term interest is a recursive language model: a model that can ask
another model to handle a useful part of a larger task. For example, a model
making a tool might ask a helper to produce one missing component. Before that
can help, the model must recognize what it needs, obtain missing information,
and choose a valid next action. We study those ingredients in TextCraft, a
text-based crafting game. The learner receives a goal and starting inventory,
looks up recipes, and makes items. A recipe lookup reveals information; a
crafting action changes the inventory or returns an error. Success means the
game's own rules confirm the requested quantities. Persuasive writing earns
no credit. These experiments study useful actions, but do not yet demonstrate
that recursion improves task completion.

## 2. What changed in the teaching examples?

Supervised fine-tuning means training a model to produce an example's next
action from the information shown before it. Our original scripted teacher
already knew all the recipes. It sometimes asked about a component before the
learner had seen that component's name. Consider an illustrative example,
not an actual recorded task: the goal is “make a lantern,” but the next action
to learn is “look up the frame recipe.” That action becomes easier to understand if the
history first includes “a lantern needs a frame.” We repaired the lookup order
and rebuilt the histories. The correct next-action answers stayed the same.
The idea is to give the learner a visible reason for the action it is asked
to copy. The repair uses an already known solution; it does not independently
discover one.

## 3. What is the strongest result?

Repaired examples helped in three small comparisons: successes rose from
1 to 14, from 2 to 13 after repeating training, and from 3 to 17 on eight
additional goals, each out of 32 attempts. This is encouraging because the
effect was not confined to the first training run or first group of goals.
There are 16 distinct goal identities across this chart, with repeated
attempts and recipe worlds; there are not 96 independent problems. The
comparison fixes the taught answers and number of training updates, although
the repaired input histories are longer. More training also partly rescues
the original examples. The supported claim is therefore better learning at
the tested training budgets, with the precise explanation still unresolved.

## 4. What did reward training add?

Reinforcement learning means updating a model from rewards for its attempts,
rather than supplying the correct next action at every step. Here the reward
comes from completing the crafting task. With one update, success on familiar
training goals rose from 9 to 13 out of 16 when the model supplied ingredient
arguments itself, and from 14 to 16 when code filled ingredients from recipes
already seen. On eight different target goals, the corresponding changes
were only 4 to 5 and 5 to 5 out of 16. Those goals share the same recipe world;
this is not a comparison between unrelated datasets. The familiar improvement
mostly failed to transfer in this small test. It does not establish that
reinforcement learning cannot generalize.

## 5. What decision follows?

The most promising result is the teaching repair, so the next question is
whether it works in another model family. That result is pending. The direct
recursion question is whether one helper improves completion of the whole
task under the same total allowance for model calls and generated text. A
helper has completed a local subtask, but the short screening run ended
before the whole task could be assessed. The advisor discussion can therefore
focus on what would most change confidence: broader repair transfer, a
clearer explanation of why repair helps, or a complete-task benefit from
delegation. Those are distinct claims and need distinct evidence.

# Concrete explanations for the discussion

## Where each idea appears in the deck

| Slide | What you should take away |
| --- | --- |
| 1 | RLMs motivate the long-term question; choosing useful actions is the immediate test. |
| 2 | The crafting goal, available materials, and recipe lookups make success concrete. |
| 3 | Worked examples teach a next action; reward training learns from attempt outcomes. |
| 4 | The repair changes what the model has seen before an unchanged taught action. |
| 5 | That narrower teaching change helped in three small comparisons. |
| 6 | Familiar-goal RL gains mostly did not transfer to different target goals. |
| 7 | A helper must earn its cost by improving the original task, not just its own part. |
| 8 | A focused teaching-repair paper needs broader tests and a clearer explanation. |

## The lantern example, step by step

This example is invented for explanation. Start with two sticks and one glowing
stone. Looking up the lantern reveals that it needs a frame and a glowing stone.
Looking up the frame reveals that it needs two sticks. Make the frame, consuming
the sticks, then make the lantern, consuming the frame and stone. The game can
check that one lantern now exists. The arrows on slide 2 mean “needs,” not
the order in which the model makes the objects.

Our actual tasks use unfamiliar item names, often codes, and longer dependency
chains. Familiar words in the illustration make it readable; they are not the
literal training data. Looking up a recipe supplies information. It does not
give the model free materials or change its trained weights.

## What exactly does the learner see and produce?

At each decision, the model receives the task goal, inventory information,
and public interaction history: its earlier actions and the game's replies.
Its output is a next tool action, such as requesting a recipe or crafting an
item. It is not simply writing a final answer to a static question. The
consequences of one action become part of the input for the next decision.
“Public” means available through that interaction; it does not mean public
on the internet.

Here is a deliberately simplified illustration:

| Before history repair | After history repair |
|---|---|
| Goal: make a lantern. No observation has mentioned a frame. | Goal: make a lantern. An earlier lookup says a lantern needs a frame. |
| Correct next action to learn: look up the frame recipe. | The same correct next action to learn: look up the frame recipe. |

The problem in the first column is not necessarily that the action would
fail. The teacher may know a perfectly useful component. The question is
whether a learner can reliably learn when to choose that action from what
it has actually seen. A separate counterfactual check found that an
unpredictable teacher query could still lead to success. That is why we
should not equate a hard-to-predict label with a harmful action.

## How are example training and reward training different here?

In supervised fine-tuning, a crafting demonstration supplies the next action:
given this goal and history, look up the frame recipe. Training increases the chance
that the model produces that demonstrated action. In reinforcement learning,
the model attempts the task and receives a task-completion reward. The
update can favor rewarded behavior without a teacher labeling every next
step. Both approaches change model weights; simply showing a recipe during
an evaluation attempt is neither kind of training.

The reward-trained models started from an example-trained model. There was
one separate collection and update for each tool interface, not a single
updated model that achieved both rows in the slide. The interface with code
assistance supplies ingredient arguments only from an already observed
recipe. The model still chooses what to make and how much. Code does not
reveal hidden recipes, replenish missing stock, or guarantee success.

## Why keep the answers fixed while changing the histories?

Changing an entire teacher would change many things at once: its chosen
actions, ordering, wording, and possibly how much training it provides.
Keeping the literal next-action answers fixed gives a narrower comparison.
For the repair experiments, the matched fits preserve starting weights,
366 answer strings, training-row order, 23 updates, and the answer-token
counts within each training batch. Crafting order also stays fixed.
The changed ingredient is the history attached to each answer, produced
by rescheduling recipe lookups so queried names have appeared publicly.

This still does not isolate one psychological explanation. Clearer context,
easier copying of item names, and different input lengths could all
contribute. A second valid randomized lookup schedule also helped, which
makes one uniquely lucky order less compelling as the entire explanation.
It does not eliminate those other possibilities.

## Does “same number of updates” mean a fair compute comparison?

It means the model took the same number of optimization steps. It does not
mean it processed the same amount of text or used exactly the same computing
work. Original inputs contain 414,682 tokens across training; stable repaired
inputs contain 431,274. The taught answer-token total stays at 8,820,
including end-of-answer tokens. We can say “same answers and training steps,”
but should not say “equal total tokens” or “equal compute.” This distinction
also matters when comparing model families, whose tokenizers and trainable
weight adapters differ.

## What do the repeated results establish—and leave open?

Repeating training checks whether the first result depended on one random
training realization. Adding eight goals checks whether it depended only on
the first group. Each 32-attempt row contains eight goals in two recipe
worlds, with two attempts per world. The first two rows reuse the same goals.
Repeated attempts are useful evidence about those goals, but do not create
new task identities. These panels were already examined during the wider
research campaign; they are exploratory evidence, not a final untouched
benchmark test.

With extra training, the original examples rise from 1 to 8 successes out
of 32 at both the 46-update and 69-update endpoints. The repaired pilot
reaches 14 at 23 updates. This makes limited-budget learnability the useful
interpretation. It would be too strong to call the original examples
unlearnable or claim the repair must remain better with unlimited training.
Repair has also not clearly beaten the alternative teacher that discovers
recipes through interaction. It improves the original teacher package;
it is not established as the best possible teaching method.

## What does the reward-training transfer result actually compare?

“Familiar” means the target goals appeared in the reward-training batch,
although evaluation uses fresh sampling seeds. “Different” means eight
other target-item goals were excluded from that update. Each column uses
two attempts per goal. Both groups live in one shared recipe world and may
share component recipes. Compare the before-and-after change within each
column: the final score of one column cannot be fairly compared with the
other as though their difficulties were identical.

One update recipe and one realized training batch per interface leave
substantial uncertainty. Familiar-goal gains also used more generated text;
success alone is not evidence of improved efficiency. The useful decision
is to finish the bounded comparisons with training on different goals and
with additional example-based training. These help separate a narrow
familiarity effect from learning that carries to other goals. No completed
result from those pending comparisons is included here.

## Why is the helper result still pending?

A subtask can succeed while the original task fails, consumes too many
resources, or runs out of budget. A helper also consumes model calls and
generated text and changes the shared inventory. The screen showed that a
real helper could be invoked and return useful local work. All original
tasks hit the screen's 32-response limit before a complete-task outcome
was established, so their outcomes are unknown, not recorded failures.

The actual first pilot compares no helper with two programmed rules for using
one helper: a fixed rule and a rule that responds to current needs. It uses one
previously examined goal, with two attempts per setting. It is a small feasibility
test, not a general result or a newly invented ability to delegate.

The planned comparison gives working alone and working with one helper the
same total budgets: 96 calls and 8,192 generated tokens, with the same
context limit. Helper actions spend that shared allowance. Only a complete
whole-task comparison can establish a benefit. Routing is currently a rule
in the surrounding program, not a learned decision to recurse.

## Optional: what did the newest training collection reveal?

A supplementary September 29 diagnosis examined attempts collected for the
next reward-training update. It found 5 successes in 32 attempts: eight new
training goals, four attempts each. Only three goals had a mixture of success
and failure. For the other five, all four attempts failed, giving this
particular within-goal reward-comparison rule no direct signal about which
attempt to favor. Learning from the other goals can still affect them.
This is a diagnosis of available training signal, not a before-and-after
learning result. Scripted recovery from an early saved error state succeeded
for all eight goals using public information. That motivates a possible
comparison of recovery examples with ordinary worked examples, after the
already planned controls; it does not show the model can recover itself.

# Likely advisor questions

**Does the model write Python in these experiments?** The broader RLM can use
code, but these crafting comparisons isolate a simpler structured tool interface:
look up a recipe, craft an item, or finish. They are tests of learning useful
tool actions, not evidence that we have trained general Python-based recursion.

**Is the helper a different model or a person?** It is another model invocation,
not a person. A surrounding program passes the subtask and information, runs
actions, and returns the result. Using a helper is an execution choice, not
another training update. The current pilot does not train the delegation rule.

**Why does the research take much longer than a training update?** A weight
update can be short. Collecting and checking many full crafting attempts requires
many model calls, and we test controls and repeat outcomes. Training time and
total experiment time are different quantities; fast training alone does not
describe the whole campaign.

**What is new here?** The candidate contribution is a controlled repair of
demonstration histories while preserving taught answers, with replicated
improvement at a limited training budget. Prior work already studies teachers
that know more than learners, imitation under partial information, and
demonstration repair. Neither the general problem nor changing observation
conditioning is new. The exact novelty remains to be established.

**Have you shown the mechanism is missing information?** No. The repair is
motivated by the mismatch between teacher knowledge and learner history,
but input length and easier use of item-name cues also change. The observed
performance effect is clearer than its causal explanation.

**Does this generalize to another model?** The specific repair has only been
shown in the Qwen3-4B family here. Phi-4-mini has earlier teacher-package
results, but its repaired-model comparison is pending. Those earlier results
cannot substitute for testing the repair itself.

**Is the score a language-model judge's opinion?** No. Saved attempts are
checked against the game's native rules and requested quantities. Invalid
actions remain part of the attempt and its cost.

**Have you shown recursive reasoning helps?** No. We have a teaching result
and real helper execution. A whole-task helper advantage under a common
resource budget remains unmeasured in this evidence snapshot.

**What would change the next decision?** Repair failing in a second family
would narrow its scope. Repair succeeding would motivate a carefully fixed
new-goal/new-world test. A helper matching solo performance at higher cost
would argue against adding more delegation complexity. Reward-training gains
that remain confined to familiar goals would weaken the case for expanding
that training recipe.

# Evidence and cutoff

This guide uses teaching and helper evidence available by **29 September
2026, 01:14:27 UTC**, and the completed reward-training transfer evidence
available by **05:30:50 UTC**. Pending means pending at those cutoffs.
The separately labeled training-collection diagnosis is an additional
September 29 source; its counts do not establish a training improvement.
If later results change the story, update the slides and guide together.

- [Teaching results, controls, exact counts, and limits](../../experiments/selective_delegation/teaching_synthesis_20260929/FINDING.md).
- [Reward-training familiar/different-goal comparison and costs](../../experiments/selective_delegation/rl_transfer_results_20260929/README.md).
- [Helper screens, shared budgets, and other harness findings](../../experiments/selective_delegation/harness_synthesis_20260929/README.md).
- [Prior work, novelty boundary, and unresolved mechanisms](../../experiments/selective_delegation/teaching_prior_art_20260929/README.md).
- [Supplementary training-collection reward diagnosis](../../experiments/selective_delegation/fresh_reward_diagnosis_20260929/README.md).
- [Previous deck's detailed speaker guide](../2026-09-29-research-update/speaker-guide.md), for historical continuity only.
