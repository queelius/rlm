---
title: "Teaching useful steps: presenter reading guide"
meeting_date: 2026-09-30
prepared_date: 2026-09-29
evidence_cutoff_utc: "2026-09-29T18:57:05.251012Z"
audience_pdf_cutoff_utc: "2026-09-29T18:57:05.251012Z"
teaching_evidence_cutoff_utc: "2026-09-29T01:14:27.611452Z"
status: exploratory
---

# Read this first

We want a small model to solve a larger task by finding useful information,
choosing actions, and sometimes asking a helper. The experiments have helped
separate these abilities. Better-looking answers, better tool commands and
better whole-task decisions are not the same achievement.

Our most promising lead is narrower than the original ambition: changing what
the model sees before a demonstrated action can make the same training answers
more useful. The proposed next study asks whether this survives stronger
comparisons and genuinely new problems. It is a possible paper about improving
worked examples, not a claim that we have solved recursive reasoning.

RL produced real changes and some local gains, but many gains weakened on new
examples or against more ordinary supervised training. The latest crafting
sequence is **4, 4, 5, then 4 successes out of 16**. The temporary extra success
did not persist. Three updates are not a verdict on RL.

For quick preparation, read this introduction, slides 2, 4–5 and 7–10 below, then
the [advisor Q&A](ADVISOR_QA.md). The
[paper assessment](../../docs/research-audit-2026-09-29/publication-assessment.md)
explains the proposed comparisons and stopping decisions.
The [retrospective](../../docs/research-audit-2026-09-29/README.md) preserves the
longer history, contrary findings and source records.

## Slide 1: What are we trying to learn?

An RLM is a language model inside a program that lets it inspect information,
run code and ask another model to solve a smaller task. A helper may ask its
own helper. A large unfamiliar problem might become manageable through smaller,
more familiar decisions.

That is the long-term motivation, not a result we have established. Our immediate
question is whether better worked examples teach useful next steps. Most current
crafting studies use one model choosing successive tool actions. They do not
train it to write arbitrary Python or build a deep tree of helpers.

Some other tests used splits that our code specified, such as assigning one
helper per candidate being checked against requirements. A successful test
of that arrangement does not show that the model learned how to choose a split.
The crafting work studies another ingredient: choosing the next tool action.
Not every experiment ran the full recursive Python-based RLM system.

A possible opening is: “We have been testing the ingredients needed for useful
decomposition. The clearest lead concerns how we teach the individual steps.”

## Slide 2: Did we teach the model how to decompose a question?

Yes, in a specific, limited way. MuSiQue asks questions that require combining
information from several documents. We trained a model to turn an original
question into a list of smaller questions. This is different from our code
cutting a long document into equal-sized chunks.

Here is the slide's invented example:

> **Original question:** In which country was the author of *The Glass Orchard* born?
>
> **Smaller question 1:** Who wrote *The Glass Orchard*?
>
> **Smaller question 2:** In which country was the person identified in answer 1 born?

The first answer is needed to ask the second question precisely. If a helper
answers “Mira Vale” to question 1, the code can turn question 2 into “In which
country was Mira Vale born?” Both the book and person are invented examples,
not a recorded evaluation item.

**What did the training example contain?** The input was the original question
and a list of document titles. The desired output was an annotated list of
smaller questions from the MuSiQue training data. We used 256 examples containing
two- or three-question reference plans. The model learned to produce such
lists through supervised fine-tuning. Later RL rewarded generated plans that
led to correct final answers.

**What did the model choose at evaluation time?** It wrote its own smaller
questions, using its existing language understanding and the additional
training. It did not receive the reference decomposition. Our interface allowed
one to eight questions and references to earlier answers.

**What did the code choose?** The code provided the format and execution
procedure. It ran helpers and substituted earlier answers into later questions.
In the main trained-planner experiments, the model produced the question list
before those answers arrived. It was not learning an unrestricted recursive
tree or repeatedly deciding whether each new subproblem needed another helper.

**What happened?** The complete system did not reliably beat answering the
original question directly, and it used more text. This does not mean the
model failed to learn question-list generation, or that decomposition can never
help. It means that the generated plans and helper execution did not establish
a reliable advantage on our broader evaluations.

There is another important limitation: helpers and the final answering model
could read the original documents. The final model could sometimes answer
correctly despite an unhelpful plan. A correct final answer was therefore not
proof that the smaller questions had done useful work.

For the meeting, say: “We tried training the model to propose smaller questions,
not just splitting documents by length. It could produce plans, but we have
not yet shown that those plans reliably improve the final answer.”

## Slide 3: How does the crafting example work?

The lantern is an invented, simplified example. Start with two sticks and one
glowing stone. A lantern recipe needs one frame and one glowing stone. A frame
recipe needs two sticks. Look up those recipes, make the frame, then make the
lantern. The arrows mean “needs,” not execution order.

Actual TextCraft-Synth experiments use unfamiliar coded names and longer chains.
The model receives its goal, inventory, earlier actions and the game's replies.
It outputs a structured action to look up a recipe, craft something or finish.
Looking up a recipe reveals information; crafting changes inventory or reports
an error. A lookup does not change the model's weights.

**How is the score computed?** Each attempt contributes one success or zero.
The game's own checker must confirm the requested increase in the target item
over starting inventory when the model finishes. There is no partial credit for just a component,
issuing a valid command or describing a good plan. A started attempt that reaches
its task budget without completion does not count as a success. A missing outcome
is different: we report it separately, not as a measured failure. All displayed
crafting panels have their planned outcomes recorded.

Our wrapper uses selected native rules from Platoon. These selected panels,
limits and changed recipe worlds are not the official paper's full benchmark.

## Slide 4: What exactly is SFT learning, and what did we repair?

Supervised fine-tuning, or SFT, trains the model to produce the next action shown
in a worked example. A simplified pair is:

> **Input:** Make one lantern. You have two sticks and one glowing stone.
> An earlier lookup says a lantern needs one frame and one glowing stone.
>
> **Training answer:** Look up the frame recipe.

The actual answer is a tool action, for example:

    {"action":"get_info","items":["frame"]}

The readable name is invented; the action format is representative. Training
makes that answer more likely given that input. It does not require the model
to write an explanation or receive a whole-task reward.

The original expert already knew the plan. Some examples asked the learner
to look up a part before its history had introduced the part's name. The repair
reorders existing lookups and rebuilds each input history. “Look up the frame”
can now follow “a lantern needs a frame.” The taught actions stay fixed.

This is an offline repair of an existing solution, not a planner discovering
a solution. An unseen-name lookup is not necessarily illegal or useless. We have
not proved the original examples unlearnable.

## Slide 5: How should I read the teaching graph?

Each bar is the number of whole-task successes out of 32 attempts. Gray uses
original examples; teal uses repaired examples. The counts are **1→14, 2→13,
and 3→17**. Each row tests eight goals in two recipe worlds, twice per world.
The second row repeats training on the same goals; the third tests eight
additional goals. There are **sixteen distinct goals total**, not ninety-six
independent tasks. These exploratory panels have now been examined.

We hold the 366 taught action strings, 8,820 answer tokens, training-row order,
answer-token totals within each batch, paired initial weights and 23 updates
fixed. Repaired inputs contain 431,274 tokens versus 414,682 original input
tokens. This is not exactly equal computing work.

**What could weaken the claim?** More training partly rescues the originals:
1→8→8/32 at 23/46/69 updates on the initial panel. A randomized valid lookup
order also helps. A teacher that discovers recipes in order performs similarly
to repair; repair has not clearly beaten it.

An earlier, uncorrected demonstration package produced a policy scoring 1/32
on an additional panel, versus base 6/32 and discovery-based teaching 15/32.
Correcting quantity handling raised that earlier baseline to 5/32. The graph's
original examples already include that quantity correction; do not confuse
the two baselines. All expert demonstrations finish their tasks, but their
trained policies can still perform poorly. Stronger comparisons are essential.

The mechanism also remains open. Showing a useful fact, making a name easier
to copy, changing recent context and making inputs longer all differ. The repair's
second-model test is pending. Earlier Phi results for another teaching package
do not substitute for this particular test.

## Slide 6: What does the code do?

The model chooses the item and quantity. When exactly one applicable recipe
has already been observed and its batch size permits the request, code writes
the ingredient arguments. It does not choose a goal, search hidden recipes,
create materials or finish automatically.

In the lantern illustration, the model chooses “make one lantern.” Code uses
the observed recipe to write “one frame and one glowing stone” into the command.
The game still rejects it if those materials are missing.

This is an execution comparison, not an additional learned decomposition method.
The percentages are **323/768 = 42.1%** versus **381/768 = 49.6%**, rounded on the
slide. This is 48 goals in four recipe worlds, two training seeds and repeated
attempts. Each paired comparison uses identical model weights. There are 74
gains and 16 losses; assistance does not help every attempt.

This is an interface result, not extra training or proof of improved planning.
It suggests leaving choices to the model while code handles exact details
already determined by observed facts. That general design idea is not new.
Recipe notebooks and calculated remaining-work displays did not provide a
useful rescue in the completed controls.

## Slide 7: What did the broader dataset campaign teach us?

This campaign compared different ways to use two released models, with no
additional training. It covered MuSiQue, FinQA, BoolQ, AG News counting and
a length-limited subset of LongBench v2. We tried direct answering, splitting
the input among helpers that wrote summaries, and helpers asked to preserve
specific facts such as names, dates and quantities.

**The split was programmed, not learned.** Code divided text into chunks.
The model did not choose task-specific boundaries. In these tests, the final
reader saw the helpers' reports instead of the full original documents.
Information omitted by a helper could therefore be lost. This differs from
the later MuSiQue trained-planner system, whose final reader saw the originals.

**The main negative result:** splitting and summarizing did not consistently
improve answers. Results varied by dataset and model; asking for detailed
factual notes was not a dependable remedy. Giving four helpers instead of two
did not provide a broad rescue. We cannot turn this into “decomposition never
works,” because the particular splitting and reporting methods were limited.

**The positive result:** FinQA financial questions benefited from code-executed
arithmetic. Here is the slide's invented, simplified example:

> A report says revenue rose from 120 to 150. What was the percentage increase?
>
> The model chooses: (150 − 120) / 120 × 100.
>
> Code computes: 25%.

The model still has to understand the question, find the right numbers and
choose the operations. Code does not supply those choices. It just executes
the arithmetic. There are no helper agents in this condition.

On the same 508 numeric questions, the smaller model produced 51 matching
answers when calculating directly and 108 when code executed its chosen
expression. The second model went from 52 to 102 matching answers.
Both conditions used one model call per question, with less than 1% more
total tokens for the arithmetic version. These are comparisons of unchanged
models with different interfaces, not SFT/RL improvements.

**What exactly was scored?** The output had to numerically match the stored
answer within a strict tolerance. That is not the same as an independent human
judgment of correct financial reasoning. Rounding, percentage conventions and
some faulty reference answers affect the counts. A looser post-hoc tolerance
preserved the direction, but did not fix all those problems. The examples were
previously exposed development data, and many questions shared source documents.
Do not describe this as untouched confirmation or official FinQA accuracy.

**Why include it?** It gives a concrete alternative to “add more helpers.”
Sometimes the useful division is between interpreting a problem and carrying
out an exact calculation. This is an established idea, not a novelty claim.
It supports testing which responsibilities belong to the model and which to code.

**What would a follow-up require?** First manually audit a bounded set of gains
and losses, including units and rounding. Then, if the contrast survives, freeze
new examples and fair direct/arithmetic comparisons. That follow-up has not
been completed and is not a prerequisite for the teaching-history paper.

## Slide 8: What did we learn from RL?

Reinforcement learning, or RL, updates the model using rewards for its own
attempts. The crafting model starts from SFT; the game rewards actual completion.

The graph shows training on eight goals and testing eight other goals twice:
**4/16 before RL, 4/16 after one update, 5/16 after two, and 4/16 after three**.
Every fixed endpoint is shown. The vertical axis runs from zero to sixteen,
the full possible score, rather than magnifying the one-attempt change.

The second-update success was real: the model made a missing component and
then the target. At the third update, that attempt failed again; the other
fifteen outcomes were unchanged. We do not yet know why it was lost.

These evaluation goals come from the official TRAIN pool, are different from
the eight current RL-training goals, and share a recipe world and some component
recipes. This repeatedly examined panel is diagnostic data, not a fresh official
test set. We do not use it to choose training examples, rewards or a favorable
checkpoint.

One extra SFT update gives 4/16. It matches the initial model, learning rate and
one update, not the examples, answer-token dose or computing work. It is not
a matched control for three RL updates. With ingredient assistance, the sequence
is 5→4→4→4/16; one extra SFT step gives 5/16. Both third readouts completed
without transport failures.

**Did RL ever help?** Yes, on some local measures. Earlier familiar-goal crafting
results were 9→13/16 without code assistance and 14→16/16 with it; different-goal
changes were only 4→5 and 5→5. Root-controller and news-helper studies also had
positive pilots that weakened on broader tests. Neither “RL never worked” nor
“we demonstrated a general RL improvement” is an accurate summary.

An informative control came from **MuSiQue-Full**. The task was to answer when
evidence was sufficient and decline otherwise. Starting at 10 correct
answer/decline pairs out of 64 attempts, eight RL updates reached 18; eight
extra SFT updates reached 19. RL helped the starting model, but did not establish
an advantage over continuing SFT.

**What next for RL?** Test a specific explanation. For example, do recovery
examples create useful successful attempts on goals where all current attempts
fail? That tests a shortage of useful experience rather than assuming a larger
update will fix it. It is separate from the proposed teaching-history paper.

## Slide 9: What did the different approaches teach us?

**Programmed candidate-level splitting** improved complete selection from 10 to
22 successes out of 24 attempts on twelve fresh cases. Our code assigned one
helper per candidate; the model did not invent that division of work. It used
roughly 3.7 times as many input tokens. This is a useful result about a particular
split, not a learned general strategy.

The slide's other two examples, news classification and conversation retrieval,
show why we also ask whether a gain survives new examples and which ability
actually improved. The fuller dataset notes below include the question-answering
results introduced on slide 2.

**MuSiQue and HotpotQA** require combining information from documents. We trained
a model to write smaller questions, ran helpers, and asked a final model to answer.
On 64 fresh MuSiQue questions attempted twice, direct answering scored 54/128,
an SFT planner 53, and an RL planner 56. After five more RL updates, the planner
was back at 54. On larger HotpotQA transfer, direct answering scored 152/256 and
the decomposition package 151/256, using about 3.44 times as many tokens.
This system did not clearly earn its cost. It does not prove decomposition
never works. New questions can still share source documents.

**AG News and DBpedia** ask for news topics and encyclopedia-entry categories.
News-helper RL looked better on an earlier panel. Fresh news examples reduced
its advantage over extra SFT to one or three correct articles out of 512,
depending on the training repeat; uncertainty intervals included no advantage.
Transfer to DBpedia was similarly unresolved. Agreement across training repeats
on old examples is not the same as transfer to new examples.

**MRCR** asks for a particular earlier assistant reply in a repeated-request
conversation. Procedural SFT taught Python retrieval. Later selected RL improved
exact answers from 24 to 27 on 32 fresh conversations. Both models still found
the right target 29 times, and both scored 29 if edge whitespace was ignored.
Most improvement concerned exact delivery of an already found answer. These
runs made no helper calls.

The lesson is to identify what changed, not just whether a score rose. We need
a final-task metric and traces showing which ability changed. The
[dataset map](ADVISOR_QA.md#which-datasets-did-we-use) separates actual training,
transfer tests and comparisons of unchanged models with different interfaces.

## Slide 10: What is the proposed paper?

The question is: **Can an executable repair of a demonstration's history improve
learning while keeping its taught actions fixed?** Test that narrow claim before
turning it into a general claim about RLMs.

Three comparisons would most change our confidence:

1. **New problems:** freeze new recipe structures and goals before inspecting
   results. More random attempts at familiar problems are not enough.
2. **Strong alternatives:** compare repair with original examples, more training
   on the originals, and a teacher that discovers recipes in order.
3. **Explanation and limits:** test another model and renamed items; distinguish
   useful observations from merely making names easier to copy.

Holding the answer strings and training-row order fixed makes this a useful
controlled comparison. It does not fix input length, recent context or compute.
A native-valid reordered-history control without the visibility constraint
would help separate those explanations. Our existing randomized control still
enforces visibility.

Related work already studies teacher–learner information differences and
rescheduling demonstrations. The possible contribution is the specific verified
repair, its effects under strong controls and its measured limits. We have not
established priority or usefulness across tool environments. A second task
would be needed for a broad claim, not merely renamed crafting items.

If the gain disappears against a good teacher or on new problems, narrow or
retire the general-method claim. A careful small study is preferable to a broad
story assembled from unrelated wins. Ask the advisors: “What evidence would
make this useful beyond this game?”

The [paper assessment](../../docs/research-audit-2026-09-29/publication-assessment.md)
gives a bounded plan. Those are proposed next experiments, not newly launched jobs.

### Where do helpers and recovery fit now?

They remain separate research directions. A helper made a part in a short
screen, but every parent reached its cap before its whole-task outcome was
known. The planned whole-task pilot compares working alone with two programmed
one-helper rules, on one previously examined goal with two attempts per setting
and a shared call/token budget. This is not yet evidence of useful delegation.

Recovery examples are prepared, not trained. Across the first three current
training collections, four difficult goals failed all 96 attempts across two
interfaces and three rounds. That means four goal identities, not 96 independent
problems. When all four same-goal samples fail, this relative-reward rule cannot
prefer one to another. Other-goal learning can still change the policy. Recovery
teaching is a possible remedy, not an established one. Do not use diagnostic
evaluation failures to select recovery training examples.

## Practical questions

**Why hours of GPU time for short fits?** Collecting and testing multi-step
attempts consumes most of the work. The first four RL collections in the current
campaign made 6,826 calls, about 4.99 hours of summed model-call time; their four
fits took 79.2 minutes including replay checks. Evaluations add more calls.
This is not a complete accounting of the allocation or an excuse for avoidable
idle time.

**Do fewer errors mean a better policy?** Not necessarily. Stopping early can
avoid errors without achieving the goal. We report completion, errors and cost
separately.

**What else is in the supporting reports?** They include tests of recipe
notebooks and calculated remaining-work displays that did not provide a useful
rescue, as well as recovery examples that are prepared but have not trained a
model. The reports also give the detailed controls behind the candidate-selection
and arithmetic examples now included in the deck. We do not need every
individual run on a slide to explain the main lessons.

## Evidence and cutoff

The audience PDF and guide include completed results through **September 29,
18:57 UTC**: the raw third RL readout finished at 18:20, the assisted one at 18:57.
The teaching comparison retains its earlier 01:14 cutoff; ingredient assistance
uses fixed completed panels 00–05. Statements about pending work refer to this
snapshot.

- [Advisor Q&A and dataset map](ADVISOR_QA.md).
- [Historical synthesis](../../docs/research-audit-2026-09-29/README.md).
- [Teaching evidence](../../experiments/selective_delegation/teaching_synthesis_20260929/FINDING.md).
- [Ingredient assistance](../../experiments/selective_delegation/finding_textcraft_breadth_20260928.md).
- [Current RL and third-update addendum](../../docs/research-audit-2026-09-29/current-rl.md).
- [Earlier RL transfer](../../experiments/selective_delegation/rl_transfer_results_20260929/README.md).
- [Focused paper proposal and prior work](../../docs/research-audit-2026-09-29/publication-assessment.md).
- [Recovery preparation, not a trained result](../../experiments/selective_delegation/recovery_teaching_prepare_20260929/README.md).
