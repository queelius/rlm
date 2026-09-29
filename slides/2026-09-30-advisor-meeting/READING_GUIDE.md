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

For quick preparation, read this introduction, slides 3–4 and 6–8 below, then
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
crafting studies are flat tool-action policies, not models writing arbitrary
Python or learning a deep recursive tree.

A possible opening is: “We have been testing the ingredients needed for useful
decomposition. The clearest lead concerns how we teach the individual steps.”

## Slide 2: How does the crafting example work?

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
The native game must confirm the requested increase in the target item over
starting inventory when the model finishes. There is no partial credit for just a component,
issuing a valid command or describing a good plan. A started attempt that reaches
its task budget without completion does not count as a success. A missing outcome
is different: we report it separately, not as a measured failure. All displayed
crafting panels have their planned outcomes recorded.

Our wrapper uses selected native rules from Platoon. These selected panels,
limits and changed recipe worlds are not the official paper's full benchmark.

## Slide 3: What exactly is SFT learning, and what did we repair?

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

## Slide 4: How should I read the teaching graph?

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

## Slide 5: What does the code do?

The model chooses the item and quantity. When exactly one applicable recipe
has already been observed and its batch size permits the request, code writes
the ingredient arguments. It does not choose a goal, search hidden recipes,
create materials or finish automatically.

In the lantern illustration, the model chooses “make one lantern.” Code uses
the observed recipe to write “one frame and one glowing stone” into the command.
The game still rejects it if those materials are missing.

The percentages are **323/768 = 42.1%** versus **381/768 = 49.6%**, rounded on the
slide. This is 48 goals in four recipe worlds, two training seeds and repeated
attempts. Each paired comparison uses identical model weights. There are 74
gains and 16 losses; assistance does not help every attempt.

This is an interface result, not extra training or proof of improved planning.
It suggests leaving choices to the model while code handles exact details
already determined by observed facts. That general design idea is not new.
Recipe notebooks and calculated remaining-work displays did not provide a
useful rescue in the completed controls.

## Slide 6: What did we learn from RL?

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

## Slide 7: What did the other datasets teach us?

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

## Slide 8: What is the proposed paper?

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

**Is there useful evidence outside the deck?** Yes. A fixed helper-per-candidate
method improved complete selection 10→22/24 on twelve fresh cases, at roughly
3.7 times the input tokens. Code-executed arithmetic improved strict numerical
matches on a FinQA-derived panel, with important target and exposure limits.
Neither is evidence that our learned general decomposition policy works.
See the retrospective and Q&A rather than adding every result to this talk.

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
