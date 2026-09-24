# A short conversation, with details available if asked

## 1. Why are we doing this?

Suggested opening: “Last time, I wanted to explore learning from rewards and
better ways to divide a task. We have now tried several forms of reward-based
training. Some helped, but the results also pointed us toward how we teach the
model and what we ask its tools to handle.”

The crafting example is deliberately simple. The interesting question is not
whether a computer can make a virtual pickaxe. It is whether a model can find
missing information and take a sequence of useful actions. The game supplies a
check independent of how confident or convincing the model sounds.

An RLM can use code and delegate to helper models, which can in principle
delegate again. The recent crafting comparisons use one model and tools. They
study prerequisites for reliable action, not an established benefit of recursion.

## 2. Did RL work?

Short answer: “We saw some useful changes, but not yet reliable improvement in
planning or a clear general advantage over more training from examples.”

RL changes model weights using feedback on the model's own sampled answers or
actions. Supervised fine-tuning (SFT) instead teaches it to imitate supplied
answers or actions. Neither is simply putting examples into the prompt.

Keep these numbers in reserve; do not read the whole table aloud.

| Study | Before and after RL | Important qualification |
|---|---|---|
| News categorization | 422 to 437/512; another training seed reached 436 | On fresh articles, 422 to 427/429; additional SFT reached 426. |
| Exact answer delivery from conversations | 24 to 27/32 on new conversations | Both scored 29/32 after ignoring edge whitespace; retrieval did not improve in aggregate. |
| Selecting qualifying records | 6 to 8/24; new sampling seeds gave 5 to 7/24 | Small panel; no robust generalization claim. |
| Answer when supported; refuse otherwise | 10 to 18/64 successful pairs | Additional SFT reached 19. Harder questions gave 6 to 7, with SFT at 6. |
| Planning helper questions | 53 to 56/128; five more updates gave 54 | The improvement is uncertain. Direct answering got 54 with much lower inference cost. |
| Crafting | 15 to 15/32 after one update, then 16 after another | Larger updates and positive-only credit did not yield a convincing gain. |

A successful answer/refusal pair requires a correct answer when evidence is
provided AND refusal when necessary evidence is removed. It is not just a
measurement of how often the model answers.

The successful crafting training attempts contained 177 rejected actions.
Thus a reward for the whole attempt does not cleanly identify good individual
actions. This is an observed problem and a hypothesis about weak learning,
not proof that changing intermediate rewards will solve it. Giving partial
credit in another study encouraged too many refusals and made the main score
worse. More feedback is not automatically better feedback.

Sources: [news helper](../../docs/research-checkpoints/2026-09-12-broader-rl-learning-signal.md),
[answer delivery](../../docs/research-checkpoints/2026-09-13-from-answer-delivery-to-information-selection.md),
[record selection](../../docs/research-checkpoints/2026-09-13-decision-interfaces-and-record-grouping.md),
[answer/refusal](../../experiments/selective_delegation/PAIRED-RL-HELDOUT-FINDINGS.md),
[harder questions](../../experiments/selective_delegation/PAIRED-RL-COMPOSITIONAL-FINDINGS.md),
[planner](../../experiments/selective_delegation/FRESH-CONTRACT-FINDINGS.md),
[crafting credit](../../experiments/selective_delegation/RL-CREDIT-ASSIGNMENT-20260923.md),
[completed positive-only comparison](../../experiments/selective_delegation/MORNING-UPDATE-20260924.md).

## 3. What is more promising now?

**How we teach:** Both teachers are programs, not people or larger models.
One can plan using recipes it already knows and then demonstrate the actions.
The other discovers the required recipes through the same lookup interface
available to the learner. Both show lookups. We train copies of the same small
model to imitate their next actions.

A simplified training example is: “Goal: make a pickaxe; recipe not yet known”
paired with “Ask for the pickaxe recipe.” The literal records also include
inventory and history, and the target is a structured tool command.

Discovery examples gave 15 versus 5 successes out of 32; a second training seed
gave 10 versus 3. The first teacher's known quantity error was corrected before
these comparisons. The packages also differ in action order and text, so we have
not isolated one cause. These are related attempts, not independent datasets.

**How we divide the work:** The model chooses the item and amount. Code fills in
ingredient arguments using an already observed recipe. It does not secretly
look up recipes, choose the plan, or supply missing materials.

For a simplified illustration, suppose a previously read recipe says one item
needs two planks and a stick. The model chooses to make one item; code supplies
that ingredient list instead of relying on the model to copy it correctly.
This is an illustration, not a literal saved recipe or an extra experiment.

The four complete comparisons were 6 to 12, 8 to 10, 9 to 14, and 9 to 13
successes out of 16. Same model weights within each comparison; no extra
training. If one number helps the conversation, use “6 to 12 in one small test,”
not “we doubled performance in general.” The smallest gain is uncertain.

Sources: [corrected teaching result](../../experiments/selective_delegation/RESEARCH-UPDATE-20260924.md),
[first three tool results](../../experiments/selective_delegation/RECIPE-BINDING-RESULTS-20260924.md),
[fourth tool result](../2026-09-25-advisor-meeting/fourth-tool-result.md).

## 4. What is the research contribution we are pursuing?

“Use code for exact details” is not itself a new idea. A possible contribution
is evidence about WHEN this division of work helps, how it interacts with
training examples, and whether it makes useful decisions easier to learn.
Novelty still needs a careful comparison with prior work.

The queued study crosses both teaching methods with and without ingredient
assistance on additional goals and recipe settings. It evaluates existing
checkpoints; it is not new RL training. Later we can test whether the improved
interface changes how well RL learns. Learned helper use and deeper decomposition
remain future research directions.

Ask: “What other task would make this convincing?” A useful candidate has
dependent steps, information the model must discover, and a checkable outcome.
Keep this as an invitation to discuss, not a claim that transfer has been shown.
