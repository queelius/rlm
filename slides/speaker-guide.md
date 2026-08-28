# Speaker guide and likely questions

This guide accompanies `rlm-preliminary-results.tex`. It is written for the presenter, not for
the audience. The slides deliberately omit most implementation details. Use this guide when you
want a concrete example, a short explanation, or a careful answer to a follow-up question.

The detailed research record remains in `preliminary-experiment-analysis.md`.

## The whole talk in one minute

Language models usually do best when a problem resembles their training experience. An RLM gives
a model a Python workspace and lets it send selected pieces to focused model calls. The larger
hypothesis is that the system may solve an unfamiliar whole by turning it into familiar pieces.

Our preliminary supervised-training studies show that an 8-billion-parameter model can learn
basic RLM procedures and several kinds of step-by-step work inside the workspace. The strongest
broader result was 24 correct answers in a 36-task exploratory test. All 24 local tasks were
solved, but none of the 12 tasks requiring model-to-model handoffs were solved. Therefore, the
central recursive capability remains an open research problem.

The next studies will compare ordinary practice with practice inside the RLM and, later, test
whether reward-based training can improve the model's decisions.

## Slide 1: The motivating question

### What the audience should remember

We are asking whether a model can solve a new kind of problem by constructing the solution from
smaller problems that look more familiar.

### Suggested explanation

“Language models are often impressive on familiar-looking tasks but less reliable when the whole
problem has a new structure. We are studying whether an RLM can help by keeping the full problem
in a workspace and turning it into smaller, more familiar pieces. We have completed preliminary
training studies, but the larger claim is still an open question.”

### Helpful analogy

A researcher may not solve a large project in one mental step. They divide it into finding
sources, extracting evidence, checking calculations, and writing a synthesis. Each smaller job is
more familiar than the project as a whole.

### Likely questions

**What is an RLM?**

An RLM is a language model placed inside a scaffold that provides a Python workspace and permits
additional, focused model calls. The model decides what to inspect, what work to perform, and when
to make another call.

**Are you claiming that the trained model now solves genuinely new problems?**

No. The completed studies are controlled mechanism tests. They show that training can change how
the model uses the RLM. They do not yet establish broad compositional generalization.

### Important boundary

Do not describe the larger hypothesis as a completed result. Say “we are testing whether,” not
“we have shown that.”

## Slide 2: What the RLM supplies

### What the audience should remember

The full problem can remain outside the model's immediate prompt. The model can inspect selected
parts, work with code, and ask focused model calls to handle pieces.

### Suggested explanation

“The RLM keeps the full input and intermediate work in a Python workspace. The main model can
inspect only what it needs. It may ask another model call to work on one selected piece, and that
call can repeat the process. The main model eventually combines the returned evidence into one
answer.”

### Concrete example

Imagine a 500-page document collection. The main model could inspect the table of contents, select
three relevant sections, ask focused calls to summarize those sections, and combine the summaries.
The model need not place all 500 pages into every prompt.

This is an illustration of the RLM idea, not a task used in the completed training studies.

### Likely questions

**Is a “focused model call” a smaller neural network?**

Not necessarily. “Focused” describes the narrower piece of work and smaller context. It may call
the same underlying model again.

**Does the main model ever see the entire input?**

It may, but it does not have to. The input can remain in the workspace while the model inspects
selected parts or receives summaries returned by other calls.

**Why use Python?**

Python provides persistent state, exact calculations, and controlled access to portions of a
large input. It also leaves a record of what the model did.

### Important boundary

The workspace supports decomposition; it does not guarantee that the model will choose a useful
decomposition.

## Slide 3: The larger hypothesis

### What the audience should remember

The scaffold may help a model generalize by composing familiar local solutions into a new whole.

### Suggested explanation

“The model may not have seen the original problem structure during training. It may still know how
to divide that kind of object, solve each resulting piece, and combine results. At each stage, the
local problem can look more like something the model has encountered before. Researchers call
this compositional generalization.”

### Concrete example

Suppose the unfamiliar task is to compare a large set of policies across several criteria. The
system might divide the collection by topic, extract the same facts from each topic, compare the
facts, and then synthesize a conclusion. Each extraction and comparison step is familiar even if
the complete policy question is new.

### Likely questions

**Why should smaller pieces be easier?**

The model's training data contain many common local patterns: classification, extraction,
calculation, comparison, and summarization. A useful decomposition may turn a novel whole into
instances of those familiar patterns.

**Is this just asking the model to think step by step?**

It is more structured than a single step-by-step response. The RLM retains external state, can
inspect selected data with code, and can make additional model calls with different local
contexts.

**Does decomposition always help?**

No. A poor division can lose information, duplicate work, or make the final combination harder.
Learning when and how to divide a problem is part of the research question.

### Important boundary

This slide presents the motivating theory. Slides 5 and 6 present the much narrower evidence we
have so far.

## Slide 4: The supervised-training foundation

### What the audience should remember

We built a repeatable process that creates checked examples, trains a model, loads an exact saved
version into the RLM, and evaluates it on new test tasks.

### Suggested explanation

“Supervised fine-tuning means showing the model examples of the behavior we want. We created
examples of useful RLM work, trained the model to imitate those decisions, saved exact versions,
and tested selected versions inside the real RLM. This turns the idea into a repeatable experiment
rather than a one-off demonstration.”

### Concrete example

A worked example can show the model: inspect the task stored in the workspace, calculate the
requested statistic with Python, and then return the result in the required form. Training exposes
the model to many checked examples of that pattern.

### Likely questions

**Which model did you train?**

We used a pinned revision of Qwen3-8B. It was capable enough for mechanism experiments and small
enough to train on one 40 GB A100.

**Was this full-model training?**

No. We used LoRA. The 8-billion-parameter base model remained frozen while approximately 43.6
million adapter parameters were updated. The saved adapter was about 175 MB.

**Where did the dataset come from?**

We generated it locally. Programs created controlled list, string, record, and nested-data tasks;
scripted expert procedures demonstrated the RLM actions; and exact programs checked the answers.
It was not a downloaded Hugging Face dataset.

**Why use synthetic data?**

Synthetic tasks provide exact answers and let us isolate one mechanism at a time. They are useful
for determining whether the model learned a procedure. Success on them is not evidence of broad
real-world reasoning.

**How is the trained model loaded and updated?**

During training, one process owns the frozen base model and the mutable LoRA adapter. After a
checkpoint is saved, a separate vLLM process loads the base plus that exact adapter for evaluation.
The serving model is not silently updated while evaluation is running.

**Why did the GPUs remain occupied for hours if one run took about 11 minutes?**

The 11-minute number described only one LoRA job. The campaign included several training jobs,
four later jobs lasting about an hour each, model loading, server startup, and many serial RLM
evaluations. Identifiable training exceeded five GPU-hours, and principal evaluation records
contained about seven hours of elapsed episode time. Reserved GPU memory also remains visible
while a model server waits between requests.

### Important boundary

A functioning experimental pipeline is a prerequisite, not evidence that the scientific
hypothesis is true.

## Slide 5: Learning a basic RLM routine

### What the audience should remember

Supervised training changed the model from failing to use two narrow RLM procedures to performing
them almost perfectly on new controlled tasks.

### Suggested explanation

“Here is a simplified version of the task. The data and question are available in the workspace.
The desired behavior is to read the task, use Python to count the matching values, and return 3.
Before supervised training, the base model did not reliably complete either tested RLM procedure.
After training, it solved 29 of 30 tasks in the basic-routine test and all 192 tasks in a second
fixed-program test. These results show procedure learning, not broad reasoning.”

### Walk through the example

The simplified list is `[5, 8, 10, 21, 25]`. The numbers divisible by 5 are 5, 10, and 25, so the
answer is 3.

The first experiment required two model turns:

1. Inspect the hidden task in the workspace.
2. Use Python and return the answer.

The second experiment used one fixed Python program containing six possible operations. The task
specified which operation and values to use. New test values and renderings were held back, but
the program itself did not change.

### Likely questions

**What does “before training” mean?**

It means the same Qwen3-8B base model inside the RLM without the trained LoRA adapter. It does not
mean the ordinary direct-answer condition.

**How did the model perform when answering directly?**

It solved 11 of 30 tasks in the first test and 9 of 192 in the fixed-program test. Those are not
equal-compute comparisons: the RLM prompts, outputs, Python work, and sometimes number of model
calls differed. That is why the slide focuses on whether SFT installed the intended RLM behavior.

**How much training data was used?**

The first study used 80 training tasks exported as 160 model-turn examples. The fixed-program
study used 576 examples.

**Does 192/192 mean the model learned general reasoning?**

No. Every target used the same fixed six-branch Python program. The result shows that SFT can
install a dependable mechanism within a defined task format.

**Were the test tasks new?**

Yes, within the scope of each controlled test. They used held-back values and surface forms, but
they did not introduce a broadly new natural-language problem family.

**Why not show the earlier 43/192 result?**

That experiment revealed a data-design flaw: some target actions depended on task information the
model could not see. The model learned shortcuts rather than a task-conditioned procedure. We
preserve it as useful negative evidence in the technical analysis, not as a positive headline.

### Important boundary

Say “the model learned these procedures.” Do not say “SFT made the model a broadly better
reasoner.”

## Slide 6: Local work versus handoffs

### What the audience should remember

In a broader exploratory test, the model learned four kinds of local work inside the workspace but
did not learn the two kinds of task that required handing pieces to other model calls.

### Suggested explanation

“We next asked whether training could teach more than one fixed procedure. A simplified local task
is to total records by group: A has 4 and 2, so its total is 6; B has 3. The trained model reliably
performed local patterns such as inspection, calculation, remembering intermediate work, checking,
and repairing a visible error. It solved all 24 local tasks. It solved none of the 12 tasks that
required model-to-model handoffs. That handoff is the bridge to the larger recursive idea.”

### Walk through the two sides

For local work, the model could keep the relevant information inside one RLM workspace:

1. Inspect the task and data.
2. Calculate with Python.
3. Preserve or check intermediate state when required.
4. Return the answer, or repair a controlled visible error.

For a handoff task, the expected pattern was different:

1. Inspect the task.
2. Send one or more selected pieces to focused model calls.
3. Read the returned answers.
4. Combine them into the final answer.

### Where 24/24 and 0/12 came from

The development test contained six workflow types with six tasks per type:

- Four local workflow types: 6/6 each, or 24/24 altogether.
- Two model-handoff workflow types: 0/6 each, or 0/12 altogether.

The total was therefore 24/36.

### Likely questions

**What were the four successful workflow types?**

They were: choose and run an appropriate program; inspect and then calculate; preserve and verify
intermediate state; and repair a controlled visible fault.

**What were the two unsuccessful workflow types?**

One required delegating a selected piece to a model call. The other required sending multiple
pieces in a batch and combining the returned results.

**Did the model refuse to make the calls?**

Not simply. It attempted the expected interfaces in some cases, but none of the 12 handoff tasks
produced a fully correct, qualified result. The experiment therefore does not establish reliable
delegation.

**How large was the broader training set?**

It contained 1,656 training episodes exported as 2,592 model-turn examples, covering 24 operations
and six workflow types.

**How did the untrained and direct models perform?**

Both scored 0/36 on this development confirmation. The trained model scored 24/36.

**Was this a sealed or replicated result?**

No. It was a small development-only study using one training seed. It is promising evidence that
motivates a stronger experiment, not a final estimate of performance.

**Does this disprove the RLM idea?**

No. It identifies the current bottleneck. The local procedures were learnable under this
curriculum; the handoff procedures need better data, training, task design, or model capability.

### Important boundary

The defensible conclusion is “local RLM work transferred before recursive handoffs.” Do not say
that recursive decomposition was learned.

## Slide 7: Planned self-training

### What the audience should remember

The next planned study asks whether practicing successful work inside the RLM teaches more than
practicing direct answers.

### Suggested explanation

“Instead of relying only on scripted expert examples, the model will attempt new tasks inside the
RLM. An automatic checker will identify successful attempts. We will train on the successful RLM
steps and compare that with training on direct solutions from the same starting model. This is a
designed experiment, not a completed result.”

### Helpful analogy

This is like comparing two kinds of practice. One student practices only final answers. Another
practices how to organize the workspace, divide the problem, check intermediate work, and produce
the answer. We want to know whether the second kind of practice transfers better.

### Likely questions

**Has this experiment run?**

No. The adjacent `rlm-bootstrap` project contains the design and reproducibility workflow, but no
self-training rollout, training run, or evaluation result has occurred yet.

**How is this different from the completed SFT studies?**

The completed studies mainly used scripted expert behavior. The planned experiment will collect
successful behavior produced by the model itself and compare RLM-practice with direct-practice
training.

**Could keeping only successes reinforce shortcuts?**

Yes. The experiment needs task splits, exact checks, trace audits, and matched controls so that a
shortcut cannot be mistaken for learning to decompose.

### Important boundary

Always introduce this slide as planned work.

## Slide 8: Research questions and requested input

### What the audience should remember

The immediate goal is not simply a higher score. It is to determine whether improvement transfers,
whether decomposition caused it, and whether the training signal is informative.

### Suggested explanation

“Our next studies will test whether any learning carries to new examples, formats, and kinds of
problem. We will compare decomposition with ordinary practice and with simply giving the model
more calls. Reward-based training comes later, after we can reliably produce both good and bad
valid attempts. We would value advice about test domains, fair comparisons, and risks in this
design.”

### Concrete examples of useful input

Colleagues could help identify:

- tasks with a natural but nontrivial decomposition;
- tasks where the quality of the decomposition can be measured separately from the final answer;
- fair direct-answer and extra-call controls;
- realistic changes in input format or problem structure; and
- failure modes that an automatic verifier might miss.

The sibling `structured-decomposition-benchmark` project may provide candidate structures and
measurement ideas. It should be described as a source of future task designs, not as completed
training evidence for this deck.

### Likely questions

**What will you do immediately next?**

First, complete the held-out behavioral comparison for the four already-trained input-format
adapters. Then run the matched direct-practice versus RLM-practice self-training experiment. Resume
reward-based training only after a frozen probe produces valid attempts with meaningful reward
variation.

**Has reward-based training improved the model yet?**

No. Diagnostic probes exercised rollout and scoring machinery, but no justified optimizer update
occurred. Some valid groups were all correct and therefore provided no learning signal; other
attempts were invalid.

**Why not immediately move to a larger model?**

The first question is whether the scaffold and training method cause improvement. Qwen3-8B fits on
one A100, which makes independent training and evaluation copies easier to control. A larger model
can be tested after the comparison is scientifically sound.

**How will RLVR update the model safely?**

A rollout server should use an immutable policy version while a separate trainer owns the mutable
adapter. A complete batch records the generated tokens, old-policy probabilities, policy identity,
RLM trace, and verifier score before the trainer publishes a new version.

### Important boundary

The purpose of the final slide is to invite criticism and collaboration. Do not imply that the
future experiments have already succeeded.

## Quick factual reference

- **Base model:** a pinned revision of Qwen3-8B.
- **Training method:** rank-16 LoRA, with the base model frozen.
- **Dataset source:** locally generated synthetic tasks, not a downloaded Hugging Face dataset.
- **Completed positive evidence:** narrow procedures and four local workflow types.
- **Central missing capability:** reliable model-to-model handoffs and recursive decomposition.
- **Completed RLVR updates:** none.
- **Completed self-training runs in `rlm-bootstrap`:** none.
- **Latest matched input-format training:** four adapters trained; held-out behavioral evaluation
  remains pending.
- **GPU use:** both A100s were used, sometimes for separate concurrent training jobs and sometimes
  for training versus serving or evaluation.

## If time is very short

The essential sequence is:

1. Slide 1: State the question.
2. Slides 2--3: Explain the RLM and the compositional-generalization hypothesis.
3. Slide 4: Say that the repeatable training loop works.
4. Slide 5: Show that SFT taught basic RLM procedures.
5. Slide 6: State the most important finding and limitation: local work transferred, handoffs did
   not.
6. Slides 7--8: Mark future work clearly and ask for input.

