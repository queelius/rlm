---
title: How to understand and present this research
meeting_date: 2026-09-11
status: discussion_draft
evidence_cutoff_utc: 2026-09-11T00:15:00Z
---

# Start here

The question behind this project is simple: can a small language model do more
useful work if it learns how to inspect a problem, ask smaller questions, and
combine the answers?

We have not solved that general problem. We have found two encouraging pieces.
Examples can teach the main model a substantially better routine for selecting
and carrying out a calculation. Separately, small changes to how inputs and
answers are matched can substantially improve a helper's answers in a batch.
We also have evidence that neither improvement automatically makes the whole
system reliable.

Those are the three ideas to carry into the meeting. You do not need to memorize
training-run names or the history of each debugging attempt.

## A two-minute explanation

“A language model normally receives some text and writes a response. We are
studying a setup where it can instead keep a large input in a Python workspace,
inspect parts of it, and ask helper model calls smaller questions. Python can
then combine the answers.

“We started by teaching a small model examples of how to use this setup. It became
much better at choosing and executing the calculation a question required.
That improvement still appeared when we changed names and numbers. Training from
the same starting point with another set of demonstrations produced a similar large gain.

“We also discovered that the way answers are attached to input records matters
much more than we expected. Matching row numbers on inputs and outputs largely
repaired one substantial failure in batched text classification.
The benefit also survived when we replaced row numbers with arbitrary matching tags,
and it appeared in a second model from the same family.

“But better individual answers did not reliably produce the right combined
answer. This reward-training recipe also failed to improve the model.
The next step is to identify which improvements carry through to a useful
whole-task result, then test whether they transfer to another model or task.
I would like your advice on which narrow question would make the strongest
next study.”

## A few terms, translated

| Term you may see in the research notes | What it means here |
|---|---|
| RLM: recursive language model | A model working through a program that lets it inspect stored input and ask smaller model calls. |
| Harness or scaffold | The surrounding code, instructions, tools, and rules for passing information. |
| Root or controller | The main model call that decides what to do next. |
| Child or leaf | A helper call that answers a smaller question. It need not use a different base model. |
| SFT: supervised fine-tuning | Training on examples of the actions we want the model to produce. |
| LoRA adapter | A relatively small collection of extra trainable weights attached to a larger, fixed model. |
| RLVR | Learning from attempted solutions whose outcomes can be checked automatically. |
| Label | A category assigned to a record, such as “place question.” |
| Readout or evaluation panel | A fixed set of questions used to measure a model. It is not another training run. |
| Grounded or faithful calculation | The recorded program actually carried out the requested calculation using the helper results it received. |
| NULL or unavailable | We do not have a usable authenticated final outcome. This is different from an observed wrong answer. |
| Context cluster | One underlying input that generates several related questions or repeated measurements. Those measurements are not independent. |

# Slide-by-slide guide

## 1. Can a small model solve harder problems by dividing the work?

**Point to make.** This is a research question, not a claim that we already have
a general-purpose problem solver.

The small model in the highlighted experiments is Qwen3-4B-Instruct-2507, a
released model with roughly four billion parameters. We did not train it from
scratch. We added relatively small weight updates for particular roles.

**Say:** “We are studying both what the model learns and how the surrounding
program helps or hinders it. The interaction between those two is the interesting
part.”

**If asked whether this is new:** Recursive model calls and Python tools are not
our inventions. Our current possible contribution is a careful account of a
specific failure and a useful, reproducible improvement.

## 2. An RLM gives the model a workspace and a way to ask smaller questions.

Follow the diagram from the main model to the stored input, then to the helpers,
then to the returned results. The arrows show a loop, not a single fixed recipe.

The main model can request a small sample, inspect lengths or counts, run code,
or hand a piece to another model call. The full input and intermediate answers
can stay in Python variables. Only information selected for observation needs
to enter the main model's text context.

This matters beyond fitting a long document in memory. A complicated unfamiliar
task might be reducible to a familiar procedure: divide records, ask a question
about each part, and combine the answers. That is the larger idea emphasized in
[Alex Zhang's blog](https://alexzhang13.github.io/blog/2026/harness/). It is a
motivating hypothesis for our work, not something our small experiments have
already established generally.

**If asked why “recursive”:** A helper can itself use the same setup and divide
its problem again. Our present highlighted tests mostly exercise one layer.
We should not describe them as demonstrations of deep recursive planning.

**If asked whether Python solves everything:** No. Python can count correctly
once it has the right categories and code. A model still has to interpret text
and choose the right operation.

## 3. The helper reads the text; Python does the counting.

Walk through the four rows slowly. Oslo, Peru's capital, and Kyoto are places.
The Hamlet question asks about a person, so it does not contribute to place totals.
Ada's place-question weights add to seven. Ben's add to two. Only Ada exceeds five.

The weights are artificial numerical fields, not confidence scores. They make
the required calculation easy to specify and check.

**Two different ways to fail:**

- If the helper wrongly calls the Hamlet question a place question, Python might
  correctly add six and two for Ben, giving eight. The arithmetic is right but
  the evidence is wrong.
- If the helper labels every row correctly but the main model counts all place
  records instead of users whose totals exceed five, it answers a different
  question. Better classification alone does not fix that.

**Important:** The displayed table is a teaching illustration, not a copied
training row. The actual source was the public TREC question-classification
dataset, with six broad categories. We added record IDs, users, weights, and
questions requiring calculations across the records.

## 4. We trained the model on examples of actions, not just final answers.

A training example was a short interaction, not simply an input paired with “1.”
The model was shown what it would observe, followed by demonstrated actions
that used the workspace and the helper.

For the slide's simplified example, an observation might look like this:

    Question: How many users have a place-question weight total above 5?
    Records: /workspace/records.jsonl
    Number of records: 4
    Record fields: id, user, question, weight

This is an illustrative observation, not a verbatim exported training message.
The actual observation included task and environment information; the point is
that the question and a way to access the input were available without pasting
every record into the main model's first message.

The demonstrated sequence was:

1. Load the records and ask the helper for their categories.
2. Use the helper's actual returned categories to perform the requested calculation.
3. Return the result that Python produced.

Only the main model's demonstrated actions were training targets. Tool results
and helper answers were context, not text the main model was being trained to
invent. Wrong helper labels were preserved. We did not replace them with correct
reference labels and pretend the model had observed those.

The highlighted experiment used 72 interactions, with three main-model actions
per interaction: 216 action examples. The 72 questions were generated from eight
16-record inputs, not from 72 independent corpora. We made six complete passes,
each accumulating a weight update over all 72 interactions. “Six updates” does
not mean that the model saw only six examples.

**What was the starting point?** A model already trained on an earlier workspace
routine. This experiment asks whether more varied demonstrations improve that
routine. It is not a comparison with an untouched base model.

**What did the model choose at test time?** Its own code and helper calls.
We did not supply the demonstration's algorithm during this root evaluation.

## 5. Training gains appeared again with a different set of demonstrations.

The bars show successes among 72 planned questions. A success means both that the
answer was correct and that the recorded program performed the requested
calculation using the helper results it actually received.

The starting model had 12 such successes. Training with one set of 72 demonstrations
raised that to 50. We then restarted from the same starting model and trained with
another separately prepared set of 72 demonstrations. That model achieved 55.

**How to read the three bars:** They are three model versions on the same evaluation.
The third bar is not the result of giving the second model more training. Both
trained versions received the same amount of training, using different examples.

The evaluation reused eight source inputs but changed user names, weights, and
thresholds. It is a shared evaluation, not a new test set for each training run.
The original evaluation also improved from 15 to 52 successes. A fresh sampling-seed
evaluation improved too. Those original, fresh-sampling, and changed-metadata
checks all evaluated **one trained model**. The new third bar comes from a separate
training run. We now have two training realizations, but still only this one
shared, research-exposed evaluation panel for their comparison.

**Could missing results explain the improvement?** Seventeen baseline questions
lacked a usable final result. If all seventeen had been successes, the baseline
would reach 29, still below either trained model. The new training run has two
missing results of its own: both questions were attempted but timed out. Its
55 verified successes could therefore become at most 57. We retain their missing
status rather than calling them known mistakes.

**Could it just be answering zero?** No, not entirely. Among the 44 questions
with nonzero correct answers, successes improved from 7 to 25 for the first trained
model. The newly trained model achieved 29, with one outcome unknown.
This is useful supporting evidence, not a claim that all easy-answer effects
have been eliminated.

**Is the second set of examples better?** We should not conclude that from 55
versus 50. There are only two training realizations, and the input set is small
and familiar to the research process. The important finding is that the large gain
appeared again. Testing genuinely new inputs is more important than celebrating
the five-answer difference.

**Have we shown generalization to any new problem?** No. The operation families
and source records were familiar within the research program. We have shown
useful transfer across a bounded change in names and numerical fields.

## 6. An answer can be right for the wrong record.

In this separate study, the model reads short passages and statements, then
chooses whether a passage supports a statement, contradicts it, or leaves it
unsettled. This uses MultiNLI, not the TREC counting task.

Imagine that the answer slot meant for one record is stamped with another
visible record's ID. The model often answers for the record named by that ID.
The output can have the required format and a sensible label, yet be wrong for
the intended record.

The experiment compared identifiers pointing to another visible record with
unrelated identifiers that did not point to a different record. Accuracy was
298/768, or 38.8%, for the misleading identifiers, and 618/768, or 80.5%, for
the unrelated ones. Matching identifiers scored 611/768, or 79.6%.

There was a more specific diagnostic too. Where the two records had different
correct labels, the misleading-ID output matched the named record 361 times
and the intended record 110 times, out of 530 positions. That supports the
wrong-record explanation rather than merely showing a random accuracy drop.

**Did the model invent the wrong IDs?** No. The experiment deliberately supplied
them, and constrained output enforced those IDs. The model still chose the
category. Perfect formatting therefore does not establish correct correspondence.

**Is this an internal-mechanism result?** It is a behavioral result. We did not
measure attention or intervene in internal model representations.

## 7. Matching row numbers on both sides helped keep answers aligned.

Explain the horizontal axis before discussing the heights. The four bars say
where we added row numbers: nowhere, only in the answers, only in the input,
or in both places. In the last condition, each answer carries the row number
shown beside its input.

The vertical axis is accuracy on the later 32 records in a 48-record batch.
That is where the preceding experiments showed a substantial problem.

The bars are 37.6%, 41.3%, 40.8%, and 87.2%. Numbering just one side had a small
effect; matching the two sides had a much larger effect. The combined advantage
was positive in all sixteen newly selected batches.

**What exactly is the 42.77-point result in the detailed notes?**
Adding output numbers helped by 3.71 percentage points without input numbers,
but by 46.48 points with input numbers. The difference is 42.77 points.
That was the comparison specified before this replication. The simple
87.2% minus 37.6% difference is also a valid description of two bars, but it is
not that pre-specified combined-effect estimate.

**Why are there 1,536 labels per bar?** Sixteen batches, times 32 later records,
times three identifier conditions. There are sixteen main paired input units,
not 1,536 independent experiments. The three identifier conditions test whether
the pattern depends on misleading, unrelated, or matching identifiers.

**Is adding row numbers a new invention?** No. Our possible contribution is
explaining the failure, separating competing explanations, and demonstrating
whether fixing it helps a complete task. The next slide tests arbitrary keys
against ordinary sequence numbers.

## 8. Matching tags helped both tested models, and the tags did not have to be numbers.

The previous result left a question: did row numbers help because they connected
each answer to its input, or because the model could count through 0, 1, 2, and so on?

A **key** is just a distinct tag. Imagine writing `k7ab` beside Maya's bike
statement and requiring its answer to appear beside `k7ab` too. Another statement
gets a different tag. We chose these as identifiers, not as a counting sequence.
These short tags are illustrative; the experiment used longer fixed tags.

For the smaller model, we tested four versions on sixteen more batches: no added
matching tags, ordinary row numbers, the same numbers shuffled, and arbitrary text
tags. Later-record accuracy was 34.2%, 85.4%, 86.1%, and 84.9%, respectively.

We then tested the same inputs with Qwen3-8B, a larger model from the same family.
The figure shows the three conditions shared by both models. Each pair of bars
compares the smaller and larger model under the same answer format:

| Format | Smaller model | Larger model |
|---|---:|---:|
| No matching tags | 34.2% | 31.5% |
| Row numbers | 85.4% | 85.5% |
| Arbitrary text tags | 84.9% | 81.1% |

Both tag types improved accuracy in all sixteen batches for both models.
The shuffled-number control is still in the evidence notes; we did not test it
with the larger model, so it is not included in this paired figure.

**Say:** “It did not need to count in order to get this benefit. Distinct matching
tags helped both models. Ordinary row numbers performed better than arbitrary
tags for the larger model in this particular test.”

**Did the model learn to copy these tags?** This test does not show that. The
output rules supplied each tag and fixed the answer order. The model chose the
reading-comprehension label that followed it. We changed the interface around
the model, not its weights.

**Why is the starting bar different from slide 7?** Slide 8 uses another set of
inputs. The second model reuses slide 8's inputs. Compare versions within each
figure, not the small differences between figures.

**Did making the model bigger solve the problem?** Not here. The untagged format
was still weak. But this is not a clean experiment on size alone: these are two
different released models, with differences beyond their number of learned weights.
The result supports testing the interface further; it does not establish a general
rule about model size. The smaller model is Qwen3-4B-Instruct-2507; the larger is
Qwen3-8B. “4B” and “8B” refer to roughly four and eight billion learned weights.

**Does this prove the cause inside the model?** No. It shows that ordinary
counting order is not required in this setup, but several explanations remain.
The arbitrary tags also use more tokens than the numbers; this was not an
equal-length test. We still need new inputs, a different model family, and another
task before claiming a general rule.

## 9. Better individual labels did not reliably produce the right combined answer.

This slide changes which model we trained. Slide 5 trained the main model.
Here we separately trained the **helper** on two response formats.

We supplied the complete plan and the Python calculation. That removes learned
planning as an explanation for success or failure. We then asked whether better
helper labels improve the final result.

The helper improved from 1,129 to 1,148 correct labels out of 1,280 label slots.
But exact final answers improved only from zero of eight to one of eight.
The total size of the numerical errors actually increased from 117 to 130.
Three of four input groups improved, while one deteriorated enough to offset
those gains. That is why we say “not reliably,” rather than “no effect.”

The reference-label bar answers an important diagnostic question: does the fixed
Python calculation work when given correct categories? Yes, all eight answers
then match. Those reference labels were never given to the experimental model.

The real aggregate question is slightly different from the slide 3 illustration.
It asks for the total weight of category-B records belonging to users who also
have a category-A record. A single wrong A label can change whether many B weights
are included. Mistakes therefore have unequal consequences.

**A concrete example of that problem.** Suppose we want the total weight of place
questions from users who also asked a person question:

| User | Question | Weight | Correct category |
|---|---|---:|---|
| Ada | Who wrote Hamlet? | 1 | Person |
| Ada | Where is Oslo? | 11 | Place |
| Ada | Where is Kyoto? | 17 | Place |
| Ben | Where is Rome? | 4 | Place |

Ada qualifies because she asked a person question. Her place questions contribute
11 + 17 = 28. Ben does not qualify. If the helper misses Ada's single person
question, the answer becomes zero instead of 28. Correcting several other,
less consequential labels would not necessarily undo that one mistake. This is
an illustration of the calculation, not an additional measured result.

**Why not ask the helper again?** We tried selective rechecking. One study corrected
a net 40 additional labels but produced only one exact answer out of eight.
Two helper samples could also agree on the same wrong label. Repetition is not
automatically verification.

**Why not ask for a short summary instead of all the labels?** We tried having
the helper return per-user flags and sums. All outputs had the required format,
but the numerical errors were much larger. Shorter output is not useful if
the model cannot calculate it reliably. These are informative unsuccessful
experiments in the evidence document, not omitted successful results.

## 10. This reward-training recipe did not improve the trained model.

Unlike supervised learning from a demonstration, reward training lets the model
attempt a solution and uses a checked outcome to adjust its behavior.

Six real weight updates were completed in this series. The final model produced
47 verified correct answers out of 72 planned questions, compared with 55 for the
unchanged starting model on the same question coordinates. Missing outcomes
cannot reverse that decline: the possible totals are 47–50 after training and
55–56 before.

Do not compare these numbers directly with slide 5. Slide 5 requires the correct
calculation as well as the answer and uses changed names and numbers. Slide 10
uses final-answer correctness on a different fixed evaluation.

Reading the programs was informative. Of 45 available questions requiring
combined operations, 42 acquired complete helper labels, 32 actually carried out
the requested calculation, and 26 did that and returned the right answer.
Four additional “correct” final zeros came from the wrong calculation.

**Did the reward cause those shortcuts?** We do not know. We can say that a reward
checking only the final number cannot distinguish those paths from genuine
successes. That is a reason to test a better measurement or training method,
not proof of the cause of this decline.

**Does this mean reinforcement learning is a dead end?** No. This is one small
model, one update size, one reward, one curriculum, and a previously used panel.
A smaller update size is a useful controlled follow-up.

## 11. The next experiments should test whether these gains survive a harder setting.

There are two separate uncertainties. For training, does the routine work on new
material and unfamiliar combinations of operations? We have now repeated training
with another demonstration set, but not evaluated both trained models on new inputs.
For input–answer matching, do better labels lead to a better combined answer?

The matching-key test has now passed its planned checks. The practical next step
is to compare the old interface with matching arbitrary tags inside a whole task.
We should first supply the same plan to both versions, then let the main model
choose its own steps. That separates errors in the labels from errors in using them.

The matching benefit has now appeared in two models from the same family.
After a local improvement survives the whole-task test, a different model family
or task is more informative than repeating many variants on the same familiar input.

**What could become a paper soon?** A narrow, replicated explanation of a
substantial failure, with a controlled fix and evidence of downstream value.
We should not promise a venue or acceptance date. The publication-options
document gives conditional work estimates, not guarantees.

## 12. The longer-term goal is to learn how to investigate a new problem.

Today we supply demonstration routines. Eventually we want the model to decide
what to inspect and how to divide the work.

For example, it might first sample ten records to discover the input's structure,
check which fields are present, ask a helper about ambiguous categories, and
then choose a grouping or a more targeted query. It might revise its plan when
the returned information is insufficient.

The related rlm-bootstrap project contains ideas for a learning loop: collect
attempts, evaluate them, retain useful examples, train, and repeat. It is an
inspiration and potential foundation, not a separate completed result being
pooled into these slides.

We can also change the tools and instructions while training the model. To know
what helped, compare the old and new model with both the old and new tools.
Otherwise an improvement in the surrounding program can be mistaken for learning.

## 13. Which result should we turn into the next focused study?

Pause here. Do not turn the questions into another lecture.

The first question asks what evidence colleagues would find convincing. The
second invites an application where the problem is genuinely consequential.
It is easier to give useful advice about a concrete task than about “general
compositional generalization.”

Our current recommendation is the input–answer matching route: use the completed
matching-key control to design a test of final-task usefulness. The
training direction remains promising. Two separately prepared training sets now
support its large gain; a broader evaluation and more training repetitions are
still needed before supporting a general claim.

# Questions that span the whole talk

## Why did the work take hours if a training pass took minutes?

Weight updates were only part of the work. We also generated real helper outputs
for training examples, loaded models, ran multi-step evaluations, saved checkpoints,
and checked traces and calculations. Evaluation itself uses the GPU: one final
answer may require several model calls and Python interactions.

For the original question-sensitive supervised run, the measured training and
checkpoint interval was about 739 seconds, or 12.3 minutes. The full workflow was
about 3,899 seconds, or 65 minutes. Those are measurements for that workflow,
not an accounting of every reserved hour in this multi-day project.

For the later reward-training continuation, four additional optimization passes
took about 127 seconds; the continuation including solution generation took about
1,780 seconds, followed by about 728 seconds of evaluation. CPU analysis and
earlier infrastructure failures add further elapsed time. We should not call
all of it “training,” nor assume every reserved second was useful GPU work.
The same original study's operations record also reports about four hours and
23 minutes of idle reserved GPU time during a session interruption. That was
lost allocation, not scientific training time, and is accounted for separately.

## How are weights loaded and updated?

The base model stays fixed during LoRA training. A training process updates the
adapter and saves checkpoints, including optimizer state needed for resumption.
An inference service then loads a selected saved adapter to generate solutions
or evaluate it. We do not silently change weights underneath an evaluation.
Each measured run records which checkpoint it used.

## What is promising enough to emphasize?

The large improvement in requested calculations after supervised training, and
the replicated input–answer matching effect. The composition failures tell us
what is still missing and make the next study more focused. They should qualify
the positive findings rather than become a catalogue of debugging stories.

## What should I avoid saying?

- “We solved general decomposition.”
- “The model has never seen these public texts.”
- “Three evaluations are three independent training replications.”
- “Correct formatting proves correct reasoning.”
- “A correct final zero proves the right calculation happened.”
- “The helper became more accurate, so the full RLM must be better.”
- “Our result proves how attention works.”
- “Six unsuccessful reward updates show that reinforcement learning cannot work.”

# Evidence key

| Slide reference | Meaning | Where to read more |
|---|---|---|
| S1 | Main-model training, changed-names-and-numbers evaluation | Finding 1 in evidence-and-methods; portable claims file, sft section. |
| S2 | Repeat training from the same start with another demonstration set | Finding 1's new-corpus follow-up; claims file, sft section. |
| H1 | Misleading identifiers redirect answers to another record | Finding 2's identifier discussion; claims file, wrong_reference section. |
| H2 | Matching input and output row numbers | Finding 2; claims file, row_matching section. |
| H3 | Arbitrary matching tags versus row numbers | Finding 2's follow-up; claims file, stable_keys section. |
| H4 | Matching tags tested with a larger model in the same family | Finding 2's second-model follow-up; claims file, cross_model_keys section. |
| C1 | Extra helper training and supplied-plan final answers | Child-training and composition findings; claims file, composition section. |
| R1 | Six-update reward-training result | Final reward-training paragraph and source report; claims file, reward_training section. |

The [evidence document](evidence-and-methods.md) contains exact local source paths
and hashes. The portable [numerical file](data/claims.json) keeps the figures
rebuildable on a machine without the full experiment store.
