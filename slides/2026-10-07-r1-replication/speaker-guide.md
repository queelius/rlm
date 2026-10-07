# Learning from correct answers: a small reproduction

Evidence cutoff: 7 October 2026, 12:35 UTC. Aim for about five minutes.
The main lesson is about the starting prompt and the format used in training.
Chat-style training raised scores from 154 to 308 out of 500 with chat input,
but from 305 to 317 with the question alone. A second run trained on questions
alone instead. It scored 314 with question-only input and 168 in chat style.
A fresh repeat of question-only training scored 306, against the same starting
score of 305. We have verified the learning pipeline and a strong input-format
effect. We have not established a reliable improvement beyond the stronger
starting setup. The repeated question-only run's near-zero change is an important result, not
one to hide behind the first run's better score.

## 1. Why start with this experiment? (45 seconds)

Say: “Before proposing how to train recursive language models, I wanted to run a
known language-model reinforcement-learning method correctly. I chose a math task
because a program can check the final answer. The immediate question is whether
this small run improves answers to questions left out of training.”

A recursive language model uses code to inspect or split its input and can ask
helper models to answer smaller questions. This experiment does not use that
machinery. It establishes a simpler reference for learning from final outcomes.
Dr. GRPO is the authors' method, not ours. The paper examines how the starting
model, prompt, and learning objective affect apparent reasoning improvements.
Our task uses its official implementation at a greatly reduced scale.

## 2. What does the learning step do? (60 seconds)

Say: “The model writes several possible solutions. A checker gives each final
answer a score of one or zero. We compare each score with the average for that
question, and use the differences to change the model's weights.”

In the invented slide example, the four rewards are 1, 1, 0, 0. Their mean is
0.5, so the relative signals are +0.5, +0.5, -0.5, -0.5. A positive signal
pushes the model toward the sampled successful attempt; a negative signal pushes
it away from the unsuccessful attempt. This describes the direction of the
learning objective, not a guarantee that every answer becomes more likely after
one shared parameter update. The actual optimizer uses probability ratios and
clipping, so it is not simply “add 0.5 to a weight.”

The model is already pretrained. “No supervised fine-tuning” means we add no
intermediate stage that teaches worked solutions by imitation. It does not mean
the model starts without knowledge. Nor does a correct final answer certify every
reasoning step. If all eight attempts receive the same reward, the mean-subtracted
signal for that question is zero.

Dr. GRPO modifies how the training objective is normalized to remove biases
identified in GRPO. We did not compare those algorithms, so do not attribute our
observed change specifically to that modification.

## 3. What exactly did we run? (60 seconds)

Say: “We ran three short training experiments on one A100. Each started with the
same model, used the same 512 math questions, and completed 32 weight updates.
One trained on chat-style inputs. Two received only the questions.
We tested the starting model and each trained model in both formats.”

The run started from Qwen2.5-Math-1.5B, not an instruction-tuned model. Each training
question yields eight sampled answers. Sixteen questions produce 128 attempts
for one accumulated optimizer update. The main run restarted from the base;
it did not inherit the pilot's four updates. It took about 50 minutes, including
generation, progress checks, and checkpoint saving. Setup and separate evaluations
take additional time.

There are three distinct roles for questions: training questions affect weights;
64 monitoring questions show progress; 128 final questions are excluded from
both training and monitoring. “Held out” means held out from this experiment,
not proven absent from the model's original pretraining. Exact normalized
question-overlap checks found no overlap between the official training collection
and MATH500; they cannot rule out paraphrases or pretraining exposure.

The expanded 500-question set includes those 64 monitoring questions, the
original 128 final questions, and 308 additional questions. All were excluded
from weight training, but the 64 were repeatedly inspected. Do not call the
whole set an untouched test.

What is a prompt? It is the text supplied to the model. In chat style, the
authors' code adds conversation markers identifying system, user, and assistant,
plus an instruction to explain the steps and put the final answer in a box.
For example, a user question might be “Write 3/20 as a decimal.” In the
question-only condition, that question is the entire input. Both are existing
options in the authors' evaluation code, not newly invented methods.

The first training run used chat style; the other two used questions alone.
Every run started from the original weights, not from another trained model.
Within a model's row on slide 4, testing changes the input format, not the weights.
For each format, the before/after comparison uses the same
questions, final-answer checker, greedy generation, and 3,000-token response cap.
Greedy generation selects the most likely token at each step. Training instead
samples alternatives. This base model is not the instruction-tuned variant;
chat formatting is not automatically its best starting interface.

## 4. What do the results show so far? (60 seconds)

Say: “With a chat-style prompt, training doubled the number of correct answers,
from 154 to 308. But the original model could already answer 305 correctly if
we simply gave it the question. With that stronger starting prompt, the same
training improved the score to 317. Training on questions alone instead gave
314. A repeat of that training gave 306. We cannot yet claim a reliable
improvement beyond the stronger starting setup.”

Read down a column to see the effect of training while keeping the prompt fixed.
Read across a row to see the effect of changing the prompt while keeping the
model weights fixed. The three trained rows are three separate training runs,
not different checkpoints chosen for different tests. All eight cells use the
same 500 questions and the prescribed final models.

Chat-style training improved question-only answers by 12: 18 became correct
and six became incorrect. Question-only training improved them by nine: 24
became correct and 15 became incorrect. The latter is a net increase of 1.8
percentage points, not 24 successes with no cost. The fresh question-only
repeat had 18 gains and 17 losses, leaving just one extra correct answer,
or 0.2 percentage points. Both runs improved slightly, but two small and
different outcomes do not establish a dependable or substantial improvement.
This does not show that more training would fail; that requires another test.

The other direction is also informative. Question-only training scored only
168 in chat style, compared with 154 before training. It did not produce the
large chat-style gain of the chat-trained model. That supports examining what
format the model learned to handle, but does not isolate every causal mechanism.
The repeated question-only run scored 166 in chat style, also far below the
chat-trained model's 308. The similar cross-format scores do not turn the small
question-only gains into reliable evidence of broadly improved reasoning.

The score counts final answers accepted by the authors' mathematical answer
checker. It recognizes mathematically equivalent answers, not just identical
strings. Both sides use the same full checker. We independently rechecked every
saved reward and matched all prompts and references to the fixed dataset.
The faster checker inside training is separate.

This does not prove that all improvement is prompt repair, and it does not
prove that no useful learning occurred. It shows why a weakly prompted baseline
can exaggerate the apparent training gain. It qualitatively reproduces a
warning in the paper, not a new discovery or its full benchmark result.
Questions are test cases, not 500 independent training runs. We have only two
training realizations for question-only inputs, with the same data and settings.

The earlier separate 128-question test scored 41 before and 80 after, with
44 gains and five losses. We keep that result unchanged. In a new full500
generation batch, its same 128 questions scored 41 and 82 under chat style.
That is repeat-generation variation, not a second training run or a reason
to replace the earlier result with a more favorable number.

The 64-question monitoring set improved from 20 to 36 at the final repeat.
An earlier evaluation after the same number of updates scored 39. Greedy GPU
generation was not exactly repeatable in our observations; we have not
established its cause. We did not use this variation to select a model.

## 5. What actually improved, and what follows? (60 seconds)

Say: “The result is useful, but we need to understand what changed. One question
asked for three twentieths as a decimal. Before training, the model repeated
the question until it ran out of space. Afterward, it explained the division
and answered 0.15. We should not assume that means it learned division from scratch.”

This is actual test row 5, with responses summarized rather than quoted.
The arithmetic is 3/20 = 15/100 = 0.15. Of the 44 newly correct answers, 36 had
originally reached the response limit. Thirty-seven originally lacked a
parseable final answer; these categories overlap. The total number of capped
responses fell from 56 to 12. A response can reach the cap after giving an
answer, so reaching the cap is not automatically the same as being wrong or
unfinished. These counts describe behavior, not the causal mechanism of every
improvement. Some model responses contain Python-looking code and claimed
output. No tools executed that code in this experiment.

The prompt control is now complete and appears on slide 4. Total responses
reaching the limit fell from 205 to 29 in the full500 chat-style comparison.
With questions alone, they changed from 15 to 17. Thus the large reduction
in runaway responses is not an equally large effect in both input formats.

The question-only training follow-up is now complete. It scored 314/500 with
questions alone, versus 305 before training. Capped responses increased from
15 to 19; this is not another large reduction in runaway answers. Under chat
input, it scored 168 and still had 201 capped responses, compared with 205
before training. We report both outcomes, not just its better input format.

Repeating evaluation of the same first-run weights produced exactly
the same saved answers and scores, 305 and 314. A fresh training repeat, with
the same settings and a different learner seed, then scored 306. These answer
different questions: rerunning
evaluation checks scoring/generation repeatability; rerunning training checks
whether another set of learning updates produces a similar result. Exact
repeatability in these identical batches does not invalidate the variation
observed earlier when the evaluation batch composition changed.

The fresh repeat used the same 512 questions and 32 updates, not additional
training of the earlier model. The learner seed changed, while the authors'
time-based generation seeding was retained. It is another training realization,
not a perfectly controlled attribution to one random-number setting. On the
308 questions outside the monitoring and original test subsets, its score was
188 versus 187 for the starting model. Thus its near-zero overall gain is not
explained solely by the 64 monitored questions.

In chat style, this repeat scored 166, with 25 gains and 13 losses against the
starting model. It still had 198 responses reach the length limit, compared
with 205 before training. The two input formats therefore tell a consistent
practical story: training on questions alone did not produce the large
chat-response recovery seen after chat-style training.

These are exploratory follow-ups chosen after inspecting the earlier results,
not independent confirmation on a new test set. The first raw-training attempt
at 10:35 used a stale transformed-data cache and was rejected. The valid run
used an isolated cache, started at 10:40, and finished at 11:24. We inspected
its actual training inputs rather than trusting only the configuration. This
is an important practical lesson for future RLM experiments too.

Successful optimization, useful answer improvement, and useful recursive
behavior are three different claims. We verified actual learning updates and
observed a large improvement under chat-style testing. But the stronger
question-only baseline makes the general learning-gain claim much smaller,
and repeating training did not establish a reliable substantial gain.
We have not tested recursive behavior here. Our workload and update
schedule are much smaller than the paper's large experiments. This is a small
reproduction of a recipe, not a new algorithm or its headline results.

For recursive training, final success may depend on several helper calls. The
open question is how to learn which choices contributed. A focused future test
would hold the task, tools, and evaluation budget fixed while comparing trained
and starting models. This is a proposed next question, not a result of today's
math experiment.

Before that, a longer run starting from the strong question-only setup could
test whether our small training budget is the limiting factor. This is a
proposed comparison, not a promise that more updates will help. Keep the final
evaluation fixed, report every accepted run, and separate new exploration from
confirmation on questions not used to make those decisions.

## Sources and evidence

- [Paper: Liu et al., *Understanding R1-Zero-Like Training: A Critical Perspective*](https://arxiv.org/abs/2503.20783).
- [Official source at the pinned commit](https://github.com/sail-sg/understand-r1-zero/tree/dfca49dd460ee7cc8e4a5a162c876a7fd6993b87).
- [Local protocol](../../docs/r1-replication-2026-10-07/README.md).
- [Fixed test scoring receipt](../../docs/r1-replication-2026-10-07/fixed128-scoring-receipt.json).
- [Prompt comparison receipt](../../docs/r1-replication-2026-10-07/prompt-control-scoring-receipt.json).
- [Evidence and limits](evidence.md).
