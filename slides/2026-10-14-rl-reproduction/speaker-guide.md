# Speaker guide for the RL reproduction result

The central message is: **we trained a model to the paper's performance level,
verified the improvement, and now have a credible reference for developing RL
for recursive language models.** The result supports confidence in our
experimental practice; it does not settle which algorithm or feedback will
work for RLM.

Evidence cutoff: 14 October 2026, 18:10 UTC. Allow roughly eight to nine minutes
for the five main slides; use the appendix only for questions.

## 1 Why the advisor asked for a reproduction

Start with the research motivation, not implementation details. We want to
learn how feedback can improve a model's use of tools and delegated subtasks.
That setting has several moving parts. A known math benchmark gives us a
simpler place to establish that we can run and measure a real RL learning loop.

Explain RL as learning from feedback on the model's own attempts. Explain an
RLM as a model that uses code and can call models on smaller parts of a problem.
The advisor's exercise asks whether our understanding survives contact with a
substantial published experiment. That is what this result addresses.

Transition: “We therefore reproduced the training stage, not just a score from
someone else's trained weights.”

## 2 What we actually reproduced

The experiment starts with the authors' already math-trained RhoMath model,
with about 1.1 billion parameters. We did not reproduce its pretraining or
earlier supervised training. We ran Segment Policy Optimization on GSM8K,
a collection of arithmetic word problems, using one A100 GPU.

For the algorithm, the slide's short explanation is sufficient: SPO samples
possible continuations to estimate which portions of an answer help. Do not
describe this as a human supplying a correct label for every reasoning step.

Distinguish the two checks. Scoring the authors' released trained model tests
our evaluation procedure. Training our own weights from the starting model
tests whether our RL workflow actually produces learning.

Transition: “The important comparison is our own starting model against our own
final model, under the same scoring rule.”

## 3 The result and the learning curve

Our matched start scores 39.77%; our fixed final model scores 58.19%. The gain
is 18.42 percentage points. The paper reports 56.7%, so our run recovered its
performance level. Do not turn the 1.49-point difference into a claim that our
method is superior.

The curve contains all 70 scheduled test evaluations and the matched starting
score. It shows the development of the run rather than only its most favorable
point. We report the predetermined final checkpoint at 11,040 updates; some
intermediate scores are slightly higher.

Explain the denominator: 1,319 questions, each with 16 sampled answers.
We grade every answer and average correctness: 12,281 correct out of 21,104.
This is not the fraction of questions with at least one correct answer, nor
majority voting. Multiple answers to the same question are not independent
training runs.

Independent regrading checked the saved answers against the official scoring
procedure. The result is one seed with disclosed runtime adaptations, not an
exact reconstruction of every historical implementation detail.

Transition: “The score is useful because we can connect it to a verified
learning process.”

## 4 What this taught us about doing RL research

The concrete accomplishment is end-to-end experimental competence. We can
start from known weights, generate attempts, assign checked feedback, update
the model, and measure the resulting gain. A matched baseline prevents us
from mistaking an already capable model for a training success.

The exercise also gives us a reference task for interpreting future RLM
results. If a more complex experiment does not improve, we can ask separately
whether the starting workflow is usable, whether feedback distinguishes useful
behavior, whether training changes that behavior, and whether the evaluation
captures the intended benefit. Those are questions to test, not diagnoses
already established by the math result.

Keep the advisor conclusion direct: “We have demonstrated that we can carry out
and validate a published LLM RL learning result.” Avoid claiming that one
reproduction certifies every algorithm or establishes reliability across seeds.

## 5 How we carry this into RLM

The RLM task adds decisions about tool calls, decomposition, combining answers
and resource use. Better final accuracy can be less useful if it requires an
unacceptable increase in calls or tokens. That is why matched budgets and cost
belong beside correctness in the proposed comparisons.

First establish what the starting model can do with the RLM workflow, compared
with answering directly. Then compare that same RLM before and after RL on
untouched evaluation tasks with matched budgets. Once there is a basic learning
signal, test whether feedback on intermediate decisions helps more than a final
outcome reward alone.

SPO offers a useful question about local credit assignment, not a commitment
to its exact algorithm. We can change the method while retaining matched
baselines, meaningful rewards, clear evaluation rules and inspectable evidence.

Close with: “We now have a tested foundation for RL experimentation. The next
step is to find out what feedback improves recursive, tool-using decisions.”

## Optional appendix

Use the appendix to answer protocol questions. Training used seed 42 and the
fixed 690-collection endpoint; the original learning-rate horizon was retained.
The authors' released model scored 56.69% in our evaluator, supporting the
comparison to their 56.7% reference. Their reported starting reference of 40.3%
is separate from our matched 39.77% starting score.

Runtime token alignment and numerical adapters are disclosed here and in the
linked research report. They limit claims of exact historical fidelity; they
are not the narrative of the talk. No causal ablation or RLM transfer result is
claimed. Consult [evidence.md](evidence.md) for the complete source map.
