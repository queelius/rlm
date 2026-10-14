# RL reproduction status and earlier learning lessons

Eight slides, about nine minutes. Current evidence cutoff: **14 October 2026,
07:06 UTC**. The primary objective is an end-to-end reproduction of the published
SPO training result on our A100. That objective has **not** been achieved.
Slide 1 gives the current status; slides 2–8 preserve the historical 7 October
Dr. GRPO experiment and its RLM teaching discussion. These are separate methods,
models and benchmarks, not one combined learning curve.

Our earlier small math reproduction supplies one example of learning working. A model
with mathematical preparation can generate useful attempts, receive checked
feedback, and improve. Our RLM experience raises a harder teaching question:
how do we help a model combine skills in an unfamiliar environment?

Run `make present` in this slide directory on your laptop for the speaker
and audience views, or `make rehearse` for private preparation. The short pdfpc
notes are cues; the material below is optional background, not a script to read
in full. See the [local setup instructions](README.md).

## 1. SPO reproduction: fixed endpoint still pending

Say: “Our latest own result is 55.88%, below the paper's reported 56.7% and above
the 39.77% SFT baseline. Test accuracy had reached 56.85% at 8,016 updates, then
fell to 56.44% and 55.88%. These are intermediate points on one training run.
We still need the fixed 11,040-update endpoint and final independent check.”

The released SPO model scored **56.69%**, matching the paper's **56.7%** to one
decimal place. Our earlier model scored **22.19% after 7,856 optimizer updates** on
its latest scheduled GSM8K test: 4,684 correct out of 21,104 sampled answers
(1,319 questions, 16 answers each). This is mean sampled-answer correctness,
not greedy accuracy or the chance that at least one answer succeeds.
Independent regrading matched all late test, validation and training-subset
answers. Earlier higher checkpoints are not substituted for the latest result.

The run's last logged optimizer step was 7,995, short of the prescribed 11,040.
Its immediate failure was an optimizer gradient-norm assertion after rejected
microbatches, not an owner timeout. That terminal failure is distinct from the
earlier accuracy decline; do not attribute either to allocation expiry.

The new GPU diagnosis found a training/generation inconsistency: generation
includes a beginning-of-sequence marker (BOS), while saved training inputs omit
it despite the configuration requesting it. For the **same saved answers**,
adding that marker barely changes the original SFT model's probability scores
but changes the late model's scores substantially across 16 fixed questions.
HF and vLLM closely agree when the same token IDs and temperature are used.
This supports investigating the mismatch; it does not establish why accuracy
collapsed or demonstrate that correcting it improves learning.

The 16-question sample uses the first saved response from each of the first
16 distinct question IDs, without reward filtering. One question overlaps the
earlier diagnostic. They are late-policy responses, not newly generated SFT
answers. Exact numbers and input provenance are in [the evidence](evidence.md)
and [portable status receipt](current-status-evidence.json).

The current audit finds **11,792 / 21,104 = 55.875663% after 8,336 updates**,
versus original SFT 8,394 / 21,104 = 39.774450%: **+16.101213 points**.
Validation is 70.492627%; non-held-out training 87.281586%. All 33,024 answers
and 21 native metrics agree with the independent regrade. Since 8,176 updates,
test loses 119 correct responses, validation loses four and training loses 14.
The intermediate curve retains all 53 scheduled points; no checkpoint replaces
the fixed endpoint.

The new run has also passed the old run's 7,995-step failure count.
That is operational progress, not a clean BOS-only ablation or proof of the
earlier collapse's cause. The process-local BOS and finite-zero BF16 adapters
are disclosed deviations; historical author code and selection remain unresolved.
Keep the original 1,000-collection learning-rate horizon and fixed 690-collection /
11,040-update endpoint, followed by final native test and independent regrade.

For RLM work this provides a stronger calibration of reward/update/evaluation
practice, not a proof that SPO is the right algorithm or that gains transfer
to tool use, recursive decisions, costs or credit assignment. Those need
their own tests. The user's purpose is methodological confidence, not a
requirement to reuse every algorithmic detail.

Transition: “The following slides preserve what the earlier, smaller Dr. GRPO
exercise taught us. Those lessons remain useful, but are not the SPO result.”

## 2. Before rewarding success, make success reachable

Say: “We want a model to decide when to use code, when to ask a helper, and how
to combine the results. We tried RL on math partly to understand what makes
learning work. The lesson is to look at what the learner brings to the task,
not only at the algorithm we use to train it.”

Pretraining provides prior knowledge and habits. A prompt can make that
knowledge easier or harder to use. Demonstrations can show unfamiliar actions.
Practice with feedback can improve choices. These are different parts of
teaching, and we can experiment with each.

Do not turn the working explanation into “RLMs cannot learn” or “we never taught
the basics.” Our earlier experiments did teach routines and produced some RL
gains. The harder problem was dependable whole-task performance and transfer.
We suspect unfamiliar workflows and insufficient preparation contribute; our
math/RLM comparison does not isolate a single cause.

Transition: “Math gives us a useful example of a learner with a head start.”

## 3. Math RL builds on a strong head start

Say: “This model did not begin by discovering mathematics from scratch. It had
already learned from substantial mathematical material. We asked it to attempt
problems, checked its answers, and trained from the successes and mistakes.”

The three ingredients are prior competence, exploration that sometimes reaches
success, and feedback that distinguishes useful attempts. The chart is a short
illustration of that loop working, not the centerpiece of the talk.

The original model is Qwen2.5-Math-1.5B. Its authors describe specialized
pretraining using mathematical text, code and generated mathematical examples.
That makes prior competence a well-grounded motivation; we did not independently
measure how much each pretraining ingredient caused our gain.
[Model authors' account](https://qwenlm.github.io/blog/qwen2.5-math/).

Our comparison rose from 20 to 47 correct answers on the same 64-question
development monitor. All seven checks are plotted, including the last dip.
Call this a successful small-scale reproduction of RL-driven improvement, not
a reproduction of the paper's complete benchmark result.

One important lesson for RLMs: do not hide existing competence with a poor
interface. In our separate 500-question comparison, the original model answered
154 correctly with chat instructions and 305 with questions alone. After short
chat-style RL, the same two input formats gave 308 and 317. So the apparent
gain depends strongly on where the model starts. Details are below if asked.

Transition: “An RLM may have useful knowledge but face a new way of using it.”

## 4. An RLM must combine skills in a new setting

Tell the example slowly: “One document tells us Mira works at Aster Lab.
Another tells us Aster Lab is in Oxford. The second helper needs the name
Aster Lab. Just telling it to find the city leaves out the connection.”

The example is invented. Its point is not that this trivial question needs
delegation; a direct answer is a valid baseline. It illustrates passing the
right information between steps. With larger tasks, the model must also decide
what to read, what to delegate, and whether a helper's answer is trustworthy.

A programmer can know Python yet struggle with an unfamiliar software interface.
Likewise, a model can know facts and code without reliably coordinating our
particular workspace and helpers. This is the intuitive meaning of a possible
distribution shift: familiar skills used in a less familiar workflow.

We cannot infer that all base models lack tool experience or that pretraining
never exposed them to related tasks. The practical question is which abilities
our chosen model actually demonstrates in our environment.

The [earlier audit](../../docs/research-audit-2026-09-29/README.md#1-what-the-history-actually-taught-us)
records learned executable routines and some local gains, but weaker transfer.
That supports investigating coordination and transfer, not declaring all the
old RL experiments unsuccessful.

Transition: “If the whole task is out of reach, what kind of practice helps?”

## 5. Useful practice is challenging, but within reach

Say: “A child without algebra is unlikely to learn calculus just by receiving
a series of wrong marks. A worked example and a simpler exercise give them
something to build on. That is an analogy for a possible teaching strategy,
not a claim that models and children learn by identical mechanisms.”

Use the simple equation. For 3x + 2 = 14, x = 4 is correct; x = 5 is not.
Some right and some wrong attempts give our method a contrast to learn from.
If every attempt receives the same reward, this particular comparison supplies
no preference between them.

The contrast must occur among attempts on one question. A batch of perfectly
solved easy questions and impossible hard questions does not solve that problem.
There is no claim that 50% accuracy is a universal target or that all RL methods
require mixed binary outcomes.

Curriculum is one possibility: find reachable challenges, then adjust difficulty.
Another is to show a missing skill. Another is to explore a wider variety of plans.
A task might be reachable already but poorly rewarded; that needs a different fix.

Transition: “One possible bridge is from examples to independent practice.”

## 6. Teach, practise, remove help, then test transfer

Say: “First find what the model can do. Show a missing step if needed. Let it
practise choices. Then remove the help and ask whether it can do something new.”

For the Mira example, initially show how the employer's name is passed to the
next helper. Later omit that hint. A further task might require combining two
helper answers rather than following a single chain. That tests a new combination,
not merely copying the same routine with different names.

Examples can appear in the prompt or in supervised fine-tuning, where training
teaches the model to imitate demonstrations. Neither extra SFT nor this exact
teaching order is mandatory. A capable model or a clear prompt may already suffice.

The teaching order itself is an open question. Compare the same examples in a
skill-building order and a shuffled order at matched cost. A simple mixed set
could work just as well. That would be useful evidence against a more elaborate
curriculum. Check retention of earlier skills as well as the new skill.

Transition: “Not every failure means the learner needs the same help.”

## 7. Different failures need different interventions

Walk through the three cases rather than listing every possible method.

- No observed successes: try a clearer prompt, a verified demonstration, an
  easier subtask or a stronger model. Zero observed successes is a diagnostic,
  not proof the model can never solve the task.
- Occasional successes: more attempts or more varied plans may make those
  successes available for training. Temperature changes randomness, not
  competence; it can also produce invalid calls.
- Mixed outcomes without improvement: inspect the checker, actual tool execution,
  weight updates and how the training score affects earlier choices.

More attempts on each problem mean fewer problems at the same budget. Keep the
evaluation attempt count fixed so extra inference is not mistaken for learning.
Trying different decomposition plans may be more useful than sampling many
near-identical answers, but that is another testable idea.

A final correct answer can conceal a bad plan, and a wrong answer can follow
several good decisions. This motivates exploring better feedback and learning
from error-and-repair examples. Do not reward more calls or longer text simply
because they look like effort.

Transition: “These explanations suggest a portfolio, not one final recipe.”

## 8. An open agenda of small experiments

Say: “We are in an exploratory phase. These are examples of questions we can
test cheaply. Each result should sharpen, expand or retire an idea.”

The visible comparisons cover hidden ability, a missing skill, teaching order
and exploration. Other branches include demonstration quality, repair examples,
stronger starting models, helper competence, context limits and feedback.

For example, first change only the helper-call example in the prompt. Does
valid execution improve? Does the right employer name reach the second call?
Do final answers improve on new cases? A positive result points toward an
interface issue, without requiring a long training run.

A training comparison should measure three stages: original model, after
preparation, and after RL. If demonstrations improve performance but RL adds
nothing, we learned something about teaching the model, not about RL gains.
If improvement disappears when hints are removed, we learned about dependence
on help rather than independent competence.

Use small panels to learn quickly, retain failures, and repeat promising
effects. Keep helpers and overall budgets comparable. Retain a direct-answer
baseline: a useful RLM should learn when delegation helps, not merely do it more.

Close: “The earlier math exercise suggested several RLM teaching experiments.
Our immediate priority is now the full SPO training reproduction. The fresh
run has reached the paper's numerical score range, but the fixed endpoint
and final comparison remain unfinished.”

## Optional details if someone asks

These details support questions; they need not enter the main presentation.

- **Method:** authors' Dr. GRPO implementation, Qwen2.5-Math-1.5B and one A100.
  We added no SFT before the math RL run. Each training question received eight
  attempts, scored using the mathematical answer checker.
- **Chart:** 20, 42, 42, 43, 43, 48, 47 correct out of 64 at updates
  0, 32, 64, 96, 128, 160, 192. One greedy answer per evaluation question;
  same chat input and 3,000-token limit. The questions were excluded from weight
  training but repeatedly inspected, so they are a development monitor.
- **Stop:** the user stopped training after 219 updates. Model and optimizer
  checkpoints through 216 remain. The chart's last point is a measurement at
  192, not a score for the last saved weights. Final evaluations were canceled.
  This historical Dr. GRPO result retains its 7 October 2026, 20:56 UTC cutoff.
- **Prompt comparison:** all values below count accepted answers out of the
  same 500 questions. Each row uses the same weights in both input formats.
  The trained row is the earlier short run, not the stopped longer run.

| Model | Chat instructions | Question alone |
| --- | ---: | ---: |
| Original model | 154 | 305 |
| After short chat-style RL | 308 | 317 |

Separate question-only training runs reached 314 and 306, versus a starting 305.
Keep these modest results in the record. This is why we should not attribute the
whole chat-format gain to newly acquired mathematical reasoning.

The paper's 1.5B chat comparison reports 33.0% to 74.2%. Our evaluation of its
released trained weights gives 73.2% on all 500 questions. Those are author
reference scores, not our training gain. See the [full evidence](evidence.md)
and [learning guide](../../docs/r1-replication-2026-10-07/learning-guide.pdf).

The paper studies starting competence, exploration and prompt compatibility:
[Understanding R1-Zero-Like Training](https://arxiv.org/html/2503.20783v2).
Related methods and additional questions are retained in the
[research portfolio](../../docs/r1-replication-2026-10-07/RESEARCH-QUESTIONS.md).
We do not claim novelty for combining demonstrations, curriculum and RL.
