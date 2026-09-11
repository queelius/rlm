---
title: How to understand and present this research
meeting_date: 2026-09-11
status: discussion_draft
evidence_cutoff_utc: 2026-09-11T03:45:00Z
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
Two separately trained versions now show that improvement on newly selected
records, though the question types are still familiar.

“We also discovered that the way answers are attached to input records matters
much more than we expected. As we increased the number of questions in one call, accuracy without matching
tags fell from about 83% to 44%. With tags, it stayed near 85%.
The benefit also survived when we replaced row numbers with arbitrary matching tags,
and it appeared in two more models, including one from another family. The size
of the benefit varied; it did not make every model equally accurate.
In another control, using different arbitrary tags beside inputs and answers
worked much worse than reusing the same tags. The task and record order stayed
the same, and every output had valid formatting.

“But better individual answers did not reliably produce the right combined
answer. Reward training has not yet improved the model, even after we tried smaller updates.
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

## 2. An RLM lets a language model use Python and ask smaller questions.

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

## 4. We taught the main model what to do next, not just what answer to give.

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

## 5. Training improved performance on newly selected inputs.

The bars count successes among the same 72 planned questions. A success requires
both a correct final answer and recorded code that performs the requested
calculation using the helper results it actually received. A lucky correct
number from the wrong calculation does not qualify.

The starting model achieved 12 successes. Two copies trained on different sets
of 72 demonstrations achieved 55 and 53. Each training set was repeated six times.
The helper stayed fixed. These are three versions on one shared test; the third
bar is not extra training on the second model.

**What is new about this test?** We selected eight new sets of 16 records,
excluded from the named inventory of our earlier main-model studies. Each set
supports nine familiar questions, giving 72 questions per model. This is a
stronger test than merely changing the names and numbers in our old inputs.
It does not show that the public source text was absent from model pretraining
or every earlier helper experiment.

**Could missing results explain the improvement?** No. Seven starting-model
questions lacked a usable final outcome. Even if all seven were successes, the
starting total would be 19. The first trained model had no missing outcomes;
the second had two, allowing 53–55 successes. These are missing-outcome bounds,
not statistical confidence intervals. Both trained models improved in observed
successes on all eight input sets.

**Did it improve on questions requiring several steps?** Yes. On the 48 questions
combining familiar operations, the starting model had no verified successes
(with five outcomes missing), versus 35 and 34 after training. For example, the
model must identify users with one question type, then sum those users' weights
for a different type. We tested familiar combinations on new records, not new
combinations invented for this evaluation.

**Could it just be answering zero?** No. On 43 questions with nonzero correct
answers, verified successes were 9 before training and 29 and 28 afterward.
There were four, zero, and two missing outcomes, respectively.

**Is one demonstration set better?** We cannot tell from 55 versus 53. The useful
result is the large improvement in both versions, not a ranking based on two
training realizations and eight underlying inputs.

**What happened to the earlier chart?** It remains in the evidence document and
the numerical file's `sft_reused_input` section. It showed 12 → 50 and 55 on
previously used records with changed names and numbers. The current chart adds
new-input evidence; it does not correct or erase the old experiment.

**Who checked the calculations?** Agents reviewed the recorded programs and
linked tool results for all 207 available episodes. The main agent checked
the numerical accounting and selected execution paths. These were unblinded,
agent-authored judgments, not an independent human annotation study.

**What should I say?** “Examples taught a useful routine, and that routine worked
better on newly selected records. We have not yet shown open-ended planning.”

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

## 7. Matching tags kept answers accurate as the helper handled more records at once.

Start with the axes. The horizontal axis is how many reading questions the helper
answers in one call: 8, 16, 32, 48, or 64. The vertical axis is the percentage of
correct reading labels across **all** answers in that batch.

A tag is a name repeated next to an input and its answer. For example:

- Input: `row 7: Maya bought a vehicle.`
- Answer: `row 7: supported.`

A row number works, but an arbitrary name such as `k7ab` can play the same role.
Software supplies the names and answer order. The model still has to choose
“supported,” “contradicted,” or “unclear.” The tags do not contain those answers.

| Questions in one call | No matching tags | Matching row numbers | Matching arbitrary tags |
|---:|---:|---:|---:|
| 8 | 82.8% | 87.5% | 84.4% |
| 16 | 82.0% | 86.7% | 87.1% |
| 32 | 56.4% | 86.9% | 85.7% |
| 48 | 49.2% | 85.4% | 85.5% |
| 64 | 43.8% | 84.6% | 83.4% |

**What is the important pattern?** Without matching tags, accuracy falls sharply
as more questions share one call. With either kind of tag, it stays near 85%.
At 32, 48, and 64 questions, both tagged versions beat the untagged version on
every one of the 16 input sets.

**Was this a training run?** No. We kept the same released Qwen3-4B model and
changed the input and output format. All 240 outputs were available and followed
their required format. The difference is not a count of malformed responses.

**How was the comparison kept fair?** Each of 16 newly selected record sets
contains 64 questions. The smaller batches use prefixes of the same set, so
formats face the same questions at each size. The order is fixed by a hash,
not chosen after seeing the answers. Each point pools 16 sets, not hundreds of
independent experiments. The nested sizes are related measurements.

**Is 32 a universal model limit?** No. It is the first tested size showing a large,
consistent benefit in this experiment. Another model, task, order, or prompt
could behave differently. The curve matters more than claiming a sharp threshold.

**Does this make large batches faster?** We have not established that. Larger
batches can reduce the number of calls, but tags add output tokens. We must
measure accuracy, elapsed time, and token use together before making a speed claim.

**Is numbering a new invention?** No. An earlier four-condition test, retained
as supporting evidence [H2], found that numbering only inputs or only outputs
helped little, while numbering both helped much more. The present curve extends
that evidence by showing when a substantial failure emerges as batches grow.
The possible contribution is a carefully characterized failure and a useful fix,
not the invention of numbered prompts.

## 8. Matching tags helped three models, including one from another family.

The curve on slide 7 used one model. Here we ask whether matching tags also help
other models, and whether arbitrary names work as well as row numbers.

A **key** is just a distinct tag. Imagine writing `k7ab` beside Maya's bike
statement and requiring its answer to appear beside `k7ab` too. Another statement
gets a different tag. We chose these as identifiers, not as a counting sequence.
These short tags are illustrative; the experiment used longer fixed tags.

For the smaller model, we tested four versions on sixteen more batches: no added
matching tags, ordinary row numbers, the same numbers shuffled, and arbitrary text
tags. Later-record accuracy was 34.2%, 85.4%, 86.1%, and 84.9%, respectively.

We then tested the same inputs with Qwen3-8B, a larger model from the same family,
and Mistral-7B-Instruct-v0.3, a model from another family. A model family is a
related line of models developed together; testing another family helps check
whether our finding depends on that particular line. Each group of three bars
compares the models under one answer format:

| Format | Qwen (4B) | Qwen (8B) | Mistral (7B) |
|---|---:|---:|---:|
| No matching tags | 34.2% | 31.5% | 31.9% |
| Row numbers | 85.4% | 85.5% | 55.8% |
| Arbitrary text tags | 84.9% | 81.1% | 52.5% |

Arbitrary tags improved accuracy in all sixteen batches for every model. Row
numbers helped every batch for the two Qwen models and fifteen of sixteen for
Mistral. The shuffled-number control is still in the evidence notes; it was
tested only with the smaller Qwen model, so it is not in this shared comparison.

**Say:** “It did not need to count in order to get this benefit. Distinct matching
tags helped all three models. The direction repeated, but the new model still
made substantially more errors. This is a useful improvement, not a complete fix.”

**Did the model learn to copy these tags?** This test does not show that. The
output rules supplied each tag and fixed the answer order. The model chose the
reading-comprehension label that followed it. We changed the interface around
the model, not its weights.

**Why are these scores different from slide 7?** These are different inputs,
and this figure scores only the last 32 answers in each 48-question batch.
Slide 7 scores every answer at each size. All three models here share the same
inputs. Compare formats within a figure, not absolute scores across figures.

**Did making the model bigger solve the problem?** Not here. The untagged format
was still weak. But this is not a clean experiment on size or model family alone:
weights, tokenizers, instructions, and input limits differ. The result supports
testing the interface further, not a general ranking of model families. “4B,”
“8B,” and “7B” refer to roughly four, eight, and seven billion learned weights.

**Does this prove the cause inside the model?** No. It shows that ordinary
counting order is not required in this setup, but several explanations remain.
The arbitrary tags also use more tokens than the numbers; this was not an
equal-length test. We have now checked another model family and new inputs in a separate size study,
but still need another task and a whole-system test before claiming a general rule.

**Haven't other researchers already studied output rules?** Yes. Earlier work
showed that rules controlling the allowed output can change task accuracy, not
just whether an answer has the right format. They can also hurt performance;
there is no general rule that tighter restrictions make a model better.
See [grammar-constrained decoding](https://aclanthology.org/2023.emnlp-main.674/),
[CRANE](https://arxiv.org/abs/2502.09061v4), and
[The Hidden Cost of Structure](https://aclanthology.org/2025.ranlp-1.124/).

Our narrower finding is that these batches already produced valid outputs, yet
many labels were wrong until we added matching tags. Arbitrary tags worked too.
The possible contribution is explaining this failure and testing whether the
fix helps the complete task. We have not yet separated the effects of the
visible tags, the enforced output rules, and the extra text generated before
each label. That is a useful open question, not an established explanation.

## 9. Repeating the same tag beside an input and its answer improved accuracy.

**Question.** Is it enough to put a tag on each side, or does using the same tag
beside an input and its answer matter?

We returned to the smaller Qwen model and the same 16 batches used on slide 8.
Each batch contained 48 passage–statement pairs. The model still had to say
whether each passage supported, contradicted, or left its statement unclear.
The first answer always belonged to the first displayed record, the second to
the second, and so on. We did not change that instruction between conditions.

The slide's two tag pairs are illustrative:

| Condition | Tag beside the input | Tag beside its answer |
|---|---|---|
| Different tags | `k7ab` | `z2pm` |
| Matching tags | `k7ab` | `k7ab` |

The tag says nothing about the correct reading label. It is just a name.
Actual tags were longer random-looking strings. We balanced two sets of tags:
each set appeared equally often on the input side and the answer side. This
reduces the chance that one unusually helpful collection of strings explains
the comparison. Software supplied the output tags and their order; the model
chose the reading labels.

The bars show 1,000/3,072 correct later labels with different tag sets and
2,420/3,072 with matching sets: **32.6% versus 78.8%**, a difference of 46.2
percentage points. Matching helped in all 16 batches. All 192 model responses
arrived and followed the required format. The difference therefore is not
explained by one condition producing more malformed answers.

**Does this conflict with slide 6, where unrelated names worked well?** No.
Slide 6 asks whether an output name misleadingly points to another visible
record. Slide 9 asks whether an *additional arbitrary tag* repeats beside the
input and its answer. These are different changes, in different prompts and
evaluation panels. Here the benefit of matching was similar whether the other
record names were misleading, unrelated, or aligned. Do not compare the 80%
on slide 6 with the 33% here as though only tag matching had changed.

**Did the different-tag condition ask the model to answer another question?**
No. The instruction explicitly said that output position determined the
intended record and that these tags were bookkeeping. It did not ask the model
to copy the input tag when the output format required a different one.

**What can we conclude?** Repeating the same tag improved the measured reading
accuracy in this setup. It is consistent with better input–answer matching,
but does not reveal the process inside the model. It also does not yet show
that the full RLM produces better final answers. That is the practical next test.

**If asked about additional checks:** [Supporting findings](later-findings.md) separates
the benefit of enforcing a usable output format from this all-valid matching-tag
comparison. It also explains a small fixed-calculation follow-up, without
presenting it as successful use by the main model. These details stay outside
the main slides to keep the story focused.

**Say:** “The tags do not contain the answer. But using the same tag on both
sides made the reading answers much more accurate, even though the format was
already correct in both cases.”

## 10. Small helper improvements were not enough for reliable final answers.

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

## 11. Reward training has not yet improved this trained model.

Unlike supervised learning from a demonstration, reward training lets the model
attempt a solution and uses a checked outcome to adjust its behavior.

Six real weight updates were completed in the larger-update run. Its final model produced
47 verified correct answers out of 72 planned questions, compared with 55 for the
unchanged starting model on the same question coordinates. Missing outcomes
cannot reverse that decline: the possible totals are 47–50 after training and
55–56 before.

We then restarted from the same trained model and tried an update size five times
smaller. That run also made six actual updates. It produced 52 verified correct
answers, with four unavailable outcomes: its possible total is 52–56. This is
not a clear decline or improvement. Among the 48 questions combining operations,
the start and the new model each answered 36 correctly. On the 46 questions
with both outcomes available, there were three wins and three losses.

**How to read the three boxes:** They show ranges, not three exact accuracy scores.
The low end counts verified correct answers; the high end also allows every
missing outcome to have been correct. Missing does not mean wrong. These are
two separate reward-training runs from one common starting model. The smaller-update run did not continue training the
larger-update model. “Larger” and “smaller” describe how strongly each update
changes the weights, not a larger language model or more training examples.

Do not compare these numbers directly with slide 5. Slide 5 requires the correct
calculation as well as the answer and now uses newly selected records. Slide 11
uses final-answer correctness on an earlier, different fixed evaluation.

Reading the programs was informative. Of 45 available questions requiring
combined operations, 42 acquired complete helper labels, 32 actually carried out
the requested calculation, and 26 did that and returned the right answer.
Four additional “correct” final zeros came from the wrong calculation. Those
counts describe the larger-update run. The smaller-update run's separate review
found 60 faithful calculations among 68 observed paths and 49 that were also
correct. On the combined-operation questions, faithful-and-correct totals tied
the starting model at 33/48. These complete path reviews were agent-authored,
not blinded human judgments.

**Did the reward cause those shortcuts?** We do not know. We can say that a reward
checking only the final number cannot distinguish those paths from genuine
successes. That is a reason to test a better measurement or training method,
not proof of the cause of this decline.

**Does this mean reinforcement learning is a dead end?** No. We tried one small
model, two update sizes, few updates, sparse final-answer rewards, and a previously
used evaluation. The next useful question is whether checking intermediate work
gives a better learning signal. That is a proposal, not a demonstrated remedy.

## 12. The next test is whether these gains help solve a complete problem.

We have made progress on one earlier uncertainty: both trained models now work
better on newly selected records. We still need to test whether the model can
combine familiar calculations in ways not demonstrated during training.

The matching-tag result raises a different question. More accurate helper labels
are useful, but does the main model use them to produce more correct final answers?
We should compare the same model and task with and without matching tags.

A small follow-up already applied fixed Python calculations to saved helper
labels: final successes increased from 2/16 without tags to 4/16 with row numbers
and 5/16 with arbitrary tags. This is encouraging, but it is a post-hoc calculation,
not successful use by the trained main model. The attempted main-model test
had only eight usable outcomes out of 96 because of tool-interface and input-limit
problems. It does not answer the whole-system question. The
[supporting note](later-findings.md) explains that distinction.

The next experiment should first make the main model's tool interface work
reliably, then compare matching formats with the task, model, and budgets fixed.
This avoids mistaking an interface failure for evidence against the reading result.

**What could become a paper soon?** A narrow, replicated account of this failure,
a controlled improvement, and evidence that the improvement matters in a complete
task. Numbered prompts already exist; we need more than a report that numbering helps.
We should not promise a venue or acceptance date.

## 13. The longer-term goal is to learn how to investigate a new problem.

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

## 14. Which result should we turn into the next focused study?

Pause here. Do not turn the questions into another lecture.

The first question asks what evidence colleagues would find convincing. The
second invites an application where the problem is genuinely consequential.
It is easier to give useful advice about a concrete task than about “general
compositional generalization.”

Our current recommendation is the input–answer matching route: use the completed
matching-key control to design a test of final-task usefulness. The
training direction remains promising. Two separately prepared training sets now
support its large gain on newly selected records; unfamiliar combinations of
operations and more training repetitions are still needed for a broader claim.

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

The smaller-update reward run gives another concrete breakdown: about 28.2 minutes
generating training attempts, 6.7 minutes updating weights, and 10.0 minutes
evaluating the final model. Its full workflow took 57.4 minutes; the remainder
included loading models and coordinating the stages. It made 1,307 model requests,
not merely six calls. Six was the number of weight updates.

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
| S1 | Main-model training, changed-names-and-numbers evaluation | Finding 1 in evidence-and-methods; portable claims file, sft_reused_input section. |
| S2 | Repeat training from the same start with another demonstration set | Finding 1's new-corpus follow-up; claims file, sft_reused_input section. |
| S3 | Three model versions evaluated on eight newly selected root input sets | Fresh-input follow-up; claims file, sft section. |
| H1 | Misleading identifiers redirect answers to another record | Finding 3's identifier discussion; claims file, wrong_reference section. |
| H2 | Matching input and output row numbers | Finding 2; claims file, row_matching section. |
| H3 | Arbitrary matching tags versus row numbers | Finding 2's follow-up; claims file, stable_keys section. |
| H4 | Matching tags tested with a larger model in the same family | Finding 2's second-model follow-up; claims file, cross_model_keys section. |
| H5 | Matching tags tested with Mistral, from another family | Cross-family matching-tag follow-up; claims file, cross_model_keys section. |
| H6 | Matching versus different arbitrary tags, with both sides tagged | Literal-tag-reuse follow-up; claims file, literal_tag_matching section. |
| H8 | Batch-size curves with and without matching tags | Nested-size follow-up; claims file, nested_batch section. |
| C1 | Extra helper training and supplied-plan final answers | Child-training and composition findings; claims file, composition section. |
| R1 | Six-update reward-training result | Final reward-training paragraph and source report; claims file, reward_training section. |
| R2 | Smaller-update reward-training follow-up | Same section's follow-up; claims file, reward_training_lower_rate section. |

The [evidence document](evidence-and-methods.md) contains exact local source paths
and hashes. The portable [numerical file](data/claims.json) keeps the figures
rebuildable on a machine without the full experiment store.
