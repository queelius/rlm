# Lessons from math RL for training RLMs

Evidence cutoff: 7 October 2026, 19:40 UTC. Six slides, about six minutes.
The reproduction supplies evidence that our learning loop can improve answers.
The main discussion is what makes that learning possible, and how we could
create those conditions for recursive language models. The RLM interventions
below are proposals, not findings from the math experiment.

## 1. A small reproduction gives us a learning signal

Say: “We eventually want an RLM to learn when to run code, when to ask a helper,
and how to divide work. First we practiced with a published RL method on math,
where we can check the final answer. On a fixed set of 64 questions, the score
rose from 20 to 48. That gives us a small-scale reproduction of RL-driven
improvement. Now, what made that possible?”

We used the authors' Dr. GRPO implementation, Qwen2.5-Math-1.5B, and one A100.
We added no supervised fine-tuning (SFT): the starting model already had math
skills. The ongoing run uses 4,080 training questions, eight responses per
question, and a fixed 255-update budget. All six scheduled monitoring points
are shown: 20, 42, 42, 43, 43, 48 at updates 0, 32, 64, 96, 128, 160.
The latest comparison contains six newly correct answers and one regression.

Each score counts final answers accepted by the full mathematical checker.
Each question gets one greedy answer, the same chat input and a 3,000-token cap.
The 64 questions were excluded from weight training but repeatedly inspected.
48/64 is 75% on this monitor, not a measured 75% on MATH500.
The final checkpoint, not the best intermediate one, will receive the planned
full500 and broader math evaluations. One run does not establish repeatability.

If asked about the paper: its 1.5B chat-template result is 33.0% to 74.2%.
Our evaluator scores the authors' released weights at 73.2% on all500.
That is an evaluator/reference check, not our own training gain. The full
published training history and benchmark result are not reproduced here.
[Paper](https://arxiv.org/html/2503.20783v2).

## 2. Useful practice is neither all wrong nor all right

Say: “For each question, we compare the model's attempts with one another.
If all eight fail, their scores are identical. If all eight succeed, they are
also identical. Some success and some failure give the method a useful contrast.”

For scores 1,1,0,0 the mean is 0.5. Subtracting it gives +0.5,+0.5,-0.5,-0.5.
The objective favors the successful sampled answers relative to the failures.
It does not guarantee that every individual output's probability moves in that
direction after a shared parameter update.

This is specifically the binary, mean-subtracted reward signal in this method.
It is not a theorem that all RL requires mixed binary outcomes. Other objectives,
graded rewards, or auxiliary losses can supply different signals.

Curriculum is a plausible way to find problems within reach and increase their
difficulty as the model improves. Crucially, averaging an always-solved easy
question and an always-failed hard question to 50% does not create within-question
contrast. Nor does this argument establish a universal optimal success rate.
A mixed group can still teach the wrong thing if the checker is flawed or the
successful answer is a lucky guess.

## 3. Existing skill needs a usable interface

Say: “The same starting weights solved 154 of500 with chat instructions, but305
when we supplied the question alone. So before asking whether RL works, ask
whether the starting interface lets the model use what it already knows.”

Both table columns use the same two weight sets: original and short chat-trained.
Only the evaluation input format changes between rows. The before/after gains
are154 with chat instructions and12 with questions alone. Separate question-only
training runs added9 and1 correct answer. Keep those less favorable outcomes in
the record; the full table is in the learning guide.

The paper also studies base-model and template effects. Our application to RLMs
is a hypothesis: a model may know Python and reasoning but not our tool protocol.
A clear prompt or demonstrations might make existing ability accessible.
SFT could teach valid calls, context passing, real-result reading and recovery.
Our earlier audit already records successful SFT of executable routines with
weaker flexible transfer. The new hypothesis concerns connecting those skills
into whole-task competence, not pretending we never taught basic tool use.
It is an option, not a mandatory prerequisite for RL. More capable models or
teacher-generated demonstrations are alternatives.

## 4. Teach the interface, then practice useful choices

Say: “Imagine two documents. One names a scientist's employer; the other gives
the employer's city. If every helper call crashes, teach the interface. If calls
work but omit the employer's name, teach passing context. If a few attempts solve
it correctly, extra exploration may provide the successes RL needs.”

This is an invented example, not a measured RLM result. It distinguishes tool
execution from useful decomposition. A helper cannot identify the city if it
is not told which employer to look for. A good strategy could extract the name,
pass it to a second lookup, and combine the evidence. A direct solution remains
a valid choice when delegation adds no value.

More attempts can expose rare successes, but may duplicate the same mistake.
For illustration, independent attempts each with a 1% success chance give about
7.7% chance of any success in8 attempts, versus27.5% in32. Actual attempts need
not be independent, and their success rate changes with sampling choices.
Higher temperature may diversify approaches but may also break tool calls.
More rollouts on one problem leave less budget for other problems.

Other options include easier tasks, hints gradually removed, verified teacher
demonstrations, a stronger starting model, or exploring different decomposition
plans. Tune these using training/development tasks, not a final benchmark.

Judge real execution and verified final answers. Generated code-looking text is
not evidence of a tool call. Do not reward more helper calls or longer answers
as substitutes for success. If we use partial rewards, check that they cannot be
earned while consistently failing the real task.

## 5. Prior work supports the ingredients, not our sequence

Say: “These ingredients have precedents. DAPO selects informative reward groups;
RLM work trains from demonstrations; DeepSeek-R1 uses a supervised cold start.
The open question for us is which competencies to teach and whether their order
changes subsequent learning and transfer.”

Adaptive sampling and a fixed easy-to-hard sequence are different interventions.
Demonstration training does not establish that a particular curriculum is optimal.
R1-Zero is a counterexample to requiring SFT in every setting. The cited work
motivates our comparisons, not a novelty claim for simply combining SFT and RL.

## 6. A portfolio of small experiments, not one settled story

Say: “We have several plausible explanations, not one answer. We will use small
comparisons to distinguish missing skills, teaching order, exploration, and
learning from mistakes. Curriculum is one branch of this program.”

The slide names four branches. Keep models, task manifests and budgets shared
where possible, and change one factor in each comparison. A prompt/teaching
screen comes before expensive RL when no valid successful attempts are reachable.
An exploration screen compares attempts per question and plan diversity at equal
total cost. A repair screen asks whether examples of mistakes and corrections
improve recovery on new errors. These are proposed experiments, not launched runs.

The following teaching-order comparison is one concrete example, not our sole
research question. Further questions are in the
[research portfolio](../../docs/r1-replication-2026-10-07/RESEARCH-QUESTIONS.md).

Say: “The next question is whether a little instruction or demonstration can
unlock RL of useful decomposition. We should first measure whether the model
produces any successful attempts, then whether RL improves beyond that preparation.”

Compare prompt examples alone, staged SFT on verified RLM demonstrations, and
SFT on exactly the same examples shuffled. Match exposure and training budget
between the two SFT arms. Keep the backbone, helper weights, tasks and RL budget
fixed. This separates whether demonstrations help from whether their order helps.
Before training, record all-fail, mixed-success and all-success groups, with
invalid calls identified separately. Then report each setup's own before/after
gain on separate questions, using the same generation and tool budget.
Record preparation cost as well as RL cost.

If preparation raises success but RL adds nothing, it is an interface/SFT result,
not evidence that RL learned decomposition. If it creates contrast and RL adds
transferable gains, that supports the hypothesis. If mixed groups already exist
but learning fails, investigate rewards, credit assignment and optimizer updates
instead of adding demonstrations indiscriminately.

The high-level sequence is: use Python and read real results; make one valid
helper call; connect dependent calls; then choose a useful decomposition or solve
directly. Check skills before teaching them. Gradually remove assistance while
retaining some earlier practice. Test new combinations of skills, not just longer
versions of one retrieval task. Adaptive RL task sampling is a separate later
comparison, so we do not change teaching order and every RL setting together.

Related work makes the broad premise plausible but not novel by itself:
[DAPO](https://arxiv.org/html/2503.14476v1) filters all-correct and all-wrong
groups; [the RLM paper](https://arxiv.org/html/2512.24601v3) trains on filtered
demonstrations of the environment; [DeepSeek-R1](https://arxiv.org/html/2501.12948v1)
uses a supervised cold start before RL, unlike R1-Zero. The proposed contribution
would have to identify which preparation or ordering matters, and why it transfers.

The current math run continues unchanged. Its fixed final evaluations and ready
small follow-ups remain under their existing resource owners. These brainstorming
ideas do not silently replace a live experiment.

## Evidence and further reading

- [Self-contained overview and complete comparisons](../../docs/r1-replication-2026-10-07/START-HERE.md)
- [Learning guide PDF](../../docs/r1-replication-2026-10-07/learning-guide.pdf)
- [All six monitoring points and verification receipt](../../docs/r1-replication-2026-10-07/interim-monitor-1940.json)
- [Source and score evidence](evidence.md)

The brainstorming structure separates observed math results, the mathematical
property of group-relative rewards, and proposed RLM interventions. It should
not turn a motivating example into an empirical claim.
