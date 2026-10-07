# Learning from correct answers: a small reproduction

Draft evidence cutoff: 7 October 2026, pilot final rescoring. Aim for about five minutes.
The main 128-question comparison is pending. Update the visible slide, this guide,
`results.tex`, `speaker-notes.json`, evidence, and cutoff together when it arrives.

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
pilot change specifically to that modification.

## 3. What exactly did we run? (60 seconds)

Say: “We updated the full weights of a 1.5-billion-parameter math model on one
40-gigabyte A100. The pilot used 64 training questions and four updates. A fresh
run uses 512 training questions and plans 32 updates.”

Both start from Qwen2.5-Math-1.5B, not an instruction-tuned model. Each training
question yields eight sampled answers. Sixteen questions produce 128 attempts
for one accumulated optimizer update. The larger run restarts from the base;
it does not inherit the pilot's four updates.

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

Say: “After four updates, the pilot went from 20 to 22 correct answers out of
64 on the final monitoring check. Three became correct, and one became incorrect.
An earlier evaluation of the same trained weights scored 23. This small change
and the variation between evaluations are reasons to be cautious.”

Those numbers use the authors' final evaluation checker on the saved answers.
The faster checker inside training reports 19 and 21 on the final repeat. Report one checker
consistently; do not mix a baseline from one with a final score from the other.
The difference is a reminder that the definition of correctness is part of the
experiment. Review the changed answers before interpreting them. Eight of 64
generated texts differed between the two evaluations of the same trained weights;
the training update count did not change. We have not attributed this variation
to a specific cause. Greedy decoding did not make this implementation perfectly
repeatable in this observation.

The planned main comparison uses 128 distinct final questions after 32 updates.
It is pending at this cutoff. A pending result is not a zero or a null result.
When available, report both newly correct and newly incorrect answers, together
with the before/after counts. Do not choose the checkpoint with the highest
monitoring score and then present it as a preselected final checkpoint.

## 5. What actually improved, and what follows? (60 seconds)

Say: “A higher score can come from finishing an answer, not discovering a new
method. In the fuel example, the model initially repeated the problem without
answering. After training, it completed the calculation. Two of the three gains
were recoveries from this kind of repetition. One earlier success regressed into
repetition. We need to understand these behaviors before claiming better reasoning.”

The SUV uses 12,000 / 15 = 800 gallons. The hybrid uses 12,000 / 48 = 250 gallons.
The difference is 550 gallons. The slide shortens an actual monitoring question
(zero-based row 29); its before/after descriptions are summaries, not quotations.
The other recovery, row 37, had already derived the correct number before
getting stuck in checking it. Row 54 changed a wrong final answer into the right
one with valid formatting in both responses. The model generated code-looking
text there, but no tools executed that code. The one regression is row 2.

The larger separate test decides whether a repeat is worthwhile. After we
understand this reference, we can test whether similar feedback teaches useful
subquestions rather than just reliable formatting and completion.

Successful optimization, useful answer improvement, and useful recursive behavior
are three different claims. This pilot does not establish the latter two
reliably. A small null result would not disprove the paper either: model size,
training duration, and update schedule differ from its large experiments.

For recursive training, final success may depend on several helper calls. The
open question is how to learn which choices contributed. A focused future test
would hold the task, tools, and evaluation budget fixed while comparing trained
and starting models. This is a proposed next question, not a result of today's
math experiment.

## Sources and evidence

- [Paper: Liu et al., *Understanding R1-Zero-Like Training: A Critical Perspective*](https://arxiv.org/abs/2503.20783).
- [Official source at the pinned commit](https://github.com/sail-sg/understand-r1-zero/tree/dfca49dd460ee7cc8e4a5a162c876a7fd6993b87).
- [Local protocol](../../docs/r1-replication-2026-10-07/README.md).
- [Draft evidence and update instructions](evidence.md).
