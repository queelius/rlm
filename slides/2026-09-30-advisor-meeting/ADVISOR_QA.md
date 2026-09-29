---
title: "Advisor questions: datasets, RL lessons and the possible paper"
meeting_date: 2026-09-30
updated_date: 2026-09-29
latest_crafting_results_utc: "2026-09-29T18:57:05.251012Z"
status: exploratory_findings_and_proposed_followup
---

# A short answer to “What did you learn?”

“We found that how we present a worked example can matter greatly, even when
the actions we teach stay the same. We also found that some improvements come
from making tool commands more reliable, rather than from better planning.
RL produced local gains, but these often weakened on new examples or compared
with more supervised training. I propose concentrating the next study on the
teaching examples and testing that result against stronger alternatives.”

This is a summary, not a script you need to memorize. The
[reading guide](READING_GUIDE.md) explains each slide and its example.

## Did the model learn to decompose problems, or did our code do it?

Both kinds of experiment exist, and they need different descriptions.

- **Programmed splits:** Our code divided documents into groups or assigned
  one helper per candidate. The model did not learn that division of work.
- **Learned question lists:** In MuSiQue, we trained the model on annotated
  smaller questions, then used RL based on the final answer. It generated
  its own questions at evaluation time, but usually wrote the whole list
  before helpers answered. That was not an unrestricted recursive tree.
- **Learned crafting actions:** We trained one model to choose recipe lookups
  and crafting actions. The strongest history-repair result concerns learning
  those steps, not learning to create helpers.

The MuSiQue example is: “In which country was the author of *The Glass Orchard*
born?” becomes “Who wrote the book?” and then “In which country was that author
born?” The book is invented. This requires recognizing a useful dependency;
it is not merely cutting a document in half. Nevertheless, our trained
question-planning system did not reliably outperform direct answering.

For the full teaching and execution example, read
[slide 2 in the guide](READING_GUIDE.md#slide-2-did-we-teach-the-model-how-to-decompose-a-question).

## Which datasets did we use?

It is important to separate three things: data used to **train** a model, data
used to **test transfer**, and tests that changed **only the surrounding code**.
We did not fine-tune or apply RL separately on every dataset listed below.

| Dataset | What the model must do | What we trained or compared |
|---|---|---|
| **TextCraft-Synth** | Look up recipes and make requested items. | SFT and RL of a tool-action policy. Separate comparisons change the code that writes ingredient arguments. This is the deck's main experimental setting. |
| **MuSiQue** | Answer a question using evidence from several documents. | SFT and RL of a model that proposes smaller questions; separate helper SFT. Compare the resulting system with direct answering. |
| **MuSiQue-Full** | Answer when evidence is sufficient and decline otherwise. | SFT and RL of a direct answering model, using paired supported/unsupported variants. This is a version of MuSiQue, not a separate new benchmark or a helper system. |
| **HotpotQA** | Answer multi-document questions. | Transfer tests of MuSiQue-trained planners/helpers. These results do not establish a separate Hotpot training run. |
| **TREC and tasks built from TREC** | Classify question types, then count or combine those labels. | Actual helper SFT and root-controller SFT/RL, as well as code/interface comparisons. The compound tasks are our constructed tasks, not official TREC classification scores. |
| **AG News → DBpedia** | Classify news topics, then test on encyclopedia-entry categories. | News-helper SFT/RL; DBpedia tests transfer rather than DBpedia training. |
| **OpenAI MRCR** | Find and reproduce a particular earlier reply in a conversation. | Python-retrieval controller SFT and later RL on final-answer delivery. No helper calls in these runs. |
| **ALFWorld** | Perform household tasks through text actions. | Action-policy SFT and comparisons of flat versus manager/worker interfaces. We do not have an established ALFWorld RL gain. |

Other breadth studies used **MultiNLI** (reasoning about two statements),
**FinQA** (financial arithmetic), **QAMPARI** (questions with many answers),
**BoolQ** (yes/no questions), and **LongBench v2** (longer-context questions).
The relevant comparisons mainly tested unchanged models with different
surrounding code or prompts. They should not be described as dataset-specific
RL results. The LongBench comparison admitted only 63 of 503 examples under
its input-length bound, so it was not a full long-context benchmark evaluation.

For a short meeting, mention TextCraft-Synth, MuSiQue/HotpotQA and one example
such as AG News or MRCR. The purpose is to explain the research, not recite a
list of benchmark names. The table is available if someone asks about breadth.

## What counts as success in the graphs?

In crafting, an attempt succeeds only when the game confirms the requested
increase in target items over starting inventory at the finish. A correct-looking explanation,
valid command or completed component does not receive partial credit.

- **Teaching graph:** each bar is successes out of 32 attempts: eight goals,
  two recipe worlds, two attempts per world. The first two rows reuse goals;
  the whole graph contains sixteen distinct goals.
- **Code-assistance percentages:** 323/768 and 381/768, rounded to 42% and 50%.
  These cover 48 goals, four worlds and repeated training/attempt seeds.
- **RL graph:** each point tests the same eight other goals twice, for sixteen
  attempts. Those goals are separate from the current RL-training goals, but
  come from the official TRAIN pool and share world/components. We report all
  fixed endpoints, not a selected best score.

These are selected experimental panels, not scores on the full official
TextCraft benchmark. Attempts are not all independent problems. The graphs
show descriptive counts; they do not supply a precise population-level
confidence interval. A new task/world can be more informative than more
attempts at the same task.

Metrics differ elsewhere. News uses correct labels; the sufficiency task
requires both answering a supported question and declining its paired
unsupported version; MRCR requires exact answer reproduction. We should not
compare their percentages as if they measure the same ability.

## Did RL work at all?

Yes, in bounded senses. Here are useful examples, including their controls:

| Study | What improved? | Why we are cautious |
|---|---|---|
| Earlier crafting RL | Familiar-goal success rose 9→13/16 without assistance and 14→16/16 with assistance. | Different-goal changes were only 4→5 and 5→5. Familiar-task improvement did not imply broad transfer. |
| Current crafting RL | Other-goal scores were 4→4→5→4/16 through three updates. | The extra success did not persist. One extra SFT step scored 4/16, but is not a matched three-step control. |
| MuSiQue-Full sufficiency | Correct pairs rose 10→18/64 after eight RL updates. | Eight extra SFT updates reached 19/64. Deeper questions gave 6/7/6 for starting/RL/SFT models. The 64 attempts reuse 32 parent questions. |
| News-helper RL | Earlier scores rose from 422 to 437 or 436 correct articles out of 512. | On fresh examples, extra SFT scored 426 and the RL models 427/429. The RL-specific advantage was unresolved; DBpedia transfer was also unresolved. |
| MRCR delivery | Selected RL improved exact answers 24→27/32 on fresh conversations. | Correct retrieval and whitespace-normalized answers both stayed 29/32. Most gain was exact delivery, not improved search or decomposition. |

Earlier TREC-derived root RL also had a 5→16/24 positive pilot. It covered just
six underlying contexts, and a broader follow-up left its main composition
subset unchanged at 5/24. We keep that positive pilot in the record without
using it as the headline for general reasoning.

## What are the main RL lessons?

1. **Continuing training is a real alternative explanation.** Compare RL with
   more SFT, not just an earlier checkpoint. Equal update counts still do not
   guarantee equal data, answer tokens or total computing work.
2. **A reward must distinguish useful attempts.** Four current training goals
   failed all 96 sampled attempts across three rounds and two interfaces.
   Within each four-attempt group, this relative-reward method has no success
   difference to learn from. That suggests testing better experience or recovery
   teaching; it does not prove that every RL method would fail.
3. **A local skill need not help the whole system.** Better helper labels,
   cleaner commands and fewer errors do not guarantee task completion.
4. **Check fresh problems and fixed checkpoints.** An early gain can disappear
   with more training or on new examples. Random sampling repeats are not new
   tasks. Do not select training or favorable checkpoints using diagnostic B.
5. **Inspect what the score rewards.** Exact output delivery can improve without
   better retrieval. Easier copying can help without better planning. That
   distinction matters for choosing the next experiment and describing a paper.

## Could the optimizer simply have been broken?

Audited completed runs have saved parameter updates, optimizer continuity and
likelihood changes. The current updates passed replay checks. That establishes
working optimization, not an informative reward or useful transfer. Some older
attempts did fail before updating; those are separate operational failures and
are not counted as completed null learning results.

## What model are we discussing? Did we update the RLM itself?

Most current crafting results use **Qwen3-4B-Instruct-2507**, with lightweight
adapter fine-tuning. Phi-4-mini comparisons probe another model family; the
specific history-repair test on Phi is still pending at this cutoff. Earlier
campaigns used other models and interfaces, so the retrospective identifies
models per experiment rather than treating everything as one model.

SFT and RL change model weights through adapters. Ingredient assistance changes
the interface with fixed weights. The history repair changes the training data,
not the runtime planner. These are three different interventions. The current
crafting actor chooses structured tool actions; it is not learning arbitrary
recursive Python programs in these experiments.

## Why is the teaching result a possible paper?

It asks a controlled question: can the same taught actions become more useful
when their input histories are repaired? We preserve literal action answers
and training-row order, then test actual task completion. The effect repeats
in the small studies. That gives us a focused result to investigate.

It is not yet a general method claim. The policy trained on the original
demonstrations performs poorly, a
discovery-based teacher already performs well, longer original training partly
closes the gap, and the datasets are small. Related work already studies
teacher–learner information mismatch and demonstration rescheduling. The
proposed contribution is a particular executable repair, strong comparisons
and a measured boundary of usefulness, not the general idea that examples matter.

## What would convince us, or make us change direction?

A frozen comparison on genuinely new task structures, repeated training runs,
another model, and strong teaching alternatives would raise confidence. Renaming
items and changing which facts are visible could help explain the effect.
A broader tool-agent claim would need another task environment where the repair
applies naturally.

If the gain appears only against our weakest teacher or on familiar worlds,
narrow the claim. If it survives neither fresh tasks nor strong comparisons,
do not keep adding favorable panels. Preserve the useful diagnosis and pursue
another direction. See the [paper assessment](../../docs/research-audit-2026-09-29/publication-assessment.md).

## Why did this take hours when SFT fits were short?

A fit learns from saved examples. A rollout must repeatedly call the model,
execute actions and wait for replies until a task ends. Training collection,
evaluation, repeated fits, controls and debugging all add time. Report measured
collection, fit and evaluation time separately; do not call the whole GPU
allocation “training time.” This distinction does not excuse idle GPU time.

## Source map

- [Early SFT/RL, TREC, news and MRCR](../../docs/research-audit-2026-09-29/early-history.md).
- [Question answering, sufficiency, ALFWorld and breadth tests](../../docs/research-audit-2026-09-29/decomposition-and-transfer.md).
- [Teaching and execution controls](../../docs/research-audit-2026-09-29/teaching-and-execution.md).
- [Current RL and dated third-update addendum](../../docs/research-audit-2026-09-29/current-rl.md).
- [Slide-specific numerical evidence](evidence.json).
