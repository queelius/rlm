# Learning from correct answers: a small reproduction

Evidence cutoff: 7 October 2026, 10:17 UTC. Aim for about five minutes.
The fixed test is complete: 41/128 correct before training and 80/128 afterward.
The broader 500-question comparison and prompt controls are running.

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

Say: “We updated the full weights of a 1.5-billion-parameter math model on one
40-gigabyte A100. The main run used 512 training questions and completed 32
updates. We tested it on 128 questions excluded from training and progress checks.”

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

Before and after evaluation uses the same questions, Qwen-Math prompt, final
answer checker, greedy generation, and 3,000-token response cap. Greedy generation
selects the most likely token at each step. Training instead samples alternatives.

## 4. What do the results show so far? (60 seconds)

Say: “On the same 128 test questions, the starting model answered 41 correctly
and the trained model answered 80 correctly. Forty-four answers became correct
and five became incorrect. We used the planned final model, not whichever
checkpoint looked best.”

The score counts final answers accepted by the authors' mathematical answer
checker. It recognizes mathematically equivalent answers, not just identical
strings. Both sides use the same full checker. We independently rechecked every
saved reward and matched all prompts and references to the fixed dataset.
The faster checker inside training is separate.

The net increase is 39 correct answers, about 30 percentage points. It is large
in this test, but comes from one training run. Questions are test cases, not
128 independent training runs. Repeating training remains important.

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

The paper motivates a useful control: give the original model just the question,
without surrounding chat-style text. Perhaps it already answers well under
that format. We will compare both original and trained weights under both
formats, rather than choose a favorable prompt for one side. Those results
are pending. The full500 comparison includes the 64 monitoring questions,
the original 128 final questions, and 308 additional questions. More evaluation
questions improve coverage but cannot replace an independent training repeat.

Successful optimization, useful answer improvement, and useful recursive
behavior are three different claims. We verified the first and observed the
second in one run. We have not tested the third here. Our workload and update
schedule are much smaller than the paper's large experiments. This is a small
reproduction of a recipe, not a new algorithm or its headline results.

For recursive training, final success may depend on several helper calls. The
open question is how to learn which choices contributed. A focused future test
would hold the task, tools, and evaluation budget fixed while comparing trained
and starting models. This is a proposed next question, not a result of today's
math experiment.

## Sources and evidence

- [Paper: Liu et al., *Understanding R1-Zero-Like Training: A Critical Perspective*](https://arxiv.org/abs/2503.20783).
- [Official source at the pinned commit](https://github.com/sail-sg/understand-r1-zero/tree/dfca49dd460ee7cc8e4a5a162c876a7fd6993b87).
- [Local protocol](../../docs/r1-replication-2026-10-07/README.md).
- [Fixed test scoring receipt](../../docs/r1-replication-2026-10-07/fixed128-scoring-receipt.json).
- [Evidence and limits](evidence.md).
