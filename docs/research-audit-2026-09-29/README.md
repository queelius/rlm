---
title: "Where the RLM research stands: a retrospective and a decision"
date: 2026-09-29
status: exploratory_research_synthesis
coverage: "Recovered August 24 experiments through September 29"
latest_completed_result_cutoff_utc: "2026-09-29T14:02:51.479831Z"
purpose: "Assess the research trajectory, strongest claims, and next informative experiments"
claim_boundary: "No established general recursive-architecture or broadly transferable RL advantage"
---

# The short answer

We are making research progress, but the strongest result is not yet the one
we originally set out to obtain. We started by asking whether a small model
could learn to use an RLM and become better at difficult problems through
decomposition. We have taught useful routines, found important weaknesses in
model–tool interfaces, and measured some real RL improvements. We have **not**
yet established a generally useful learned decomposition policy or a broad
advantage from recursive execution.

The strongest current paper candidate is more specific:

> Can we make correct worked examples easier to learn from by changing what
> the model has seen before each demonstrated action, without changing the
> action we teach?

The crafting experiments now give a repeated positive result for that
intervention. They also leave concrete rival explanations and transfer tests.
That is a better basis for a focused paper than collecting more isolated
improvements under the broad label “RLM.”

The longer-term architecture question is still worth pursuing:

> Which decisions should a model learn, which exact operations should code
> handle, and when does asking a helper actually help the original task?

Our old record-identity experiments, recent recipe binding, and remaining
inventory failures make that question concrete. They do not yet answer it
generally.

## How to read this audit

This page gives the research argument and decisions. The supporting reports
preserve the details and contrary evidence:

- [Early history](early-history.md): original SFT, root/helper interfaces,
  correspondence, early RL, retrieval and fixed candidate-level decomposition.
- [Decomposition and transfer](decomposition-and-transfer.md): QA controls,
  answer/refusal training, the recovered September 14–15 breadth campaign,
  household tasks and actual helper execution.
- [Teaching and execution](teaching-and-execution.md): TextCraft discovery,
  controlled history repair, extra-training and second-model controls, recipe
  binding, inventory and failed notebook/demand interventions.
- [Current RL results](current-rl.md): the complete first RL/SFT comparison,
  both second updates, and the training-side reason to change the next probe.
- [Literature and next tests](literature-and-next-tests.md): close precedents,
  two recent papers, discriminating comparisons, costs and decision rules.
- [Machine-readable latest evidence](fresh-results.json) and
  [claim register](claims.json): source-linked facts, limitations and decisions.

This is a targeted audit of the major experimental lines, not a replay of
every raw trace or an exhaustive catalog of every failed launch. Each report
states its verification scope. Historical snapshots retain their original
cutoffs; a then-pending statement is not evidence that the experiment remains
unfinished today. Related goals, recipe worlds and repeated seeds are not
counted as independent problems.

## 1. What the history actually taught us

| Phase | What looked promising | What survived the stronger checks |
| --- | --- | --- |
| Getting the RLM routine to work | SFT could turn an initially unsuccessful controller into one that reliably used Python and returned answers. | Useful interface competence. Perfect performance on one dispatcher did not transfer to delegation workflows. |
| Better helpers and record handling | Accurate local classification and explicit record identifiers produced large gains. | Correspondence mattered across model families, but small independent calls were strong, sometimes faster baselines. Better helper labels did not always improve the root's answer. |
| Teaching and rewarding the root | Root SFT improved faithful execution; several RL pilots improved small panels. | Positive learning is real. Broader tasks, fresh examples and extra-SFT controls repeatedly reduced the apparent RL advantage. |
| Learned QA decomposition | Trained plans and helpers improved selected conditions. | Fresh direct and plan-only controls often matched them at about one-third the tokens. More planner training did not resolve the gap. |
| Broader datasets and exact calculation | A completed 18-hour campaign compared direct answers, summaries, factual notes and arithmetic across two model sizes. | Generic decomposition did not consistently help. Code-executed arithmetic roughly doubled strict numeric matches on FinQA, but the cases were previously exposed and scoring has known limitations. |
| Interactive crafting | Teaching public recipe discovery strongly outperformed the earlier teacher package. | A tighter history-only repair retained every target answer and repeated its benefit. Exact ingredient binding separately helped execution. |
| Today's RL and helper tests | Real weight updates, some familiar-task gains, and real child execution. | Transfer is small so far. Many training goals still provide no relative reward signal; whole-task helper benefit remains unmeasured. |

One historical connection is especially useful. In August, the model was
sometimes trained to choose a task-specific program before its prompt revealed
which task it had been given. It learned favorite operations instead. A generic
program that inspected the actual input fixed that narrow problem. In today's
crafting examples, an expert sometimes asks about a component before any
observation has introduced it. Reordering the history makes the same next
action easier to justify from the learner's view.

These episodes motivate a common research question about the information
available at a decision. They are **not proof of one common causal mechanism**.
In the crafting counterfactual test, different expert queries could all lead
to success. Copying a visible item name, changing recent history and providing
longer inputs are still possible contributors to the new training gain.

## 2. The results most worth retaining

### A. Teaching-history repair is the strongest recent learning result

The controlled repair preserves all 366 taught action strings, all 8,820 answer
tokens and each optimizer batch's answer-token total. At the same 23 training
updates, successes change from **1→14, 2→13 and 3→17 out of 32** across the
pilot, another training seed and eight additional goals. The added comparisons
together give 5→30/64. There are sixteen distinct goal identities in these
comparisons, not ninety-six independent tasks.

The practical example is simple: before teaching “look up the frame recipe,”
show the earlier observation “a lantern needs a frame.” The action label stays
the same. This illustration uses familiar words rather than the actual coded
item names.

Important limits change the claim. Repaired inputs are longer. Additional
training partly rescues the original examples: 1→8→8/32 at the fixed 23/46/69
update endpoints. Randomized valid query ordering also helps. The repair does
not clearly beat the existing discovery teacher. The specific repair has not
yet been tested successfully in the second model family. Thus the result is
better learning at the tested budgets, not proof of an unlearnable baseline,
an optimal ordering, or a general planning algorithm.

### B. Execution assistance is a useful architecture result, but not learned planning

On the broader completed crafting slice, the discovery-trained model improves
from **323 to 381 successes in 768 attempts** when code fills ingredient
arguments from a recipe the model already looked up. This covers 48 goal
identities, multiple worlds, two training seeds and repeated attempts.
There are 74 paired gains and 16 losses, not universal improvement. The
model still chooses the item, requested quantity and when to finish.

This is stronger than saying “better prompting helps.” It is a concrete change
to the division of work between the model and code. But giving the model a
recipe notebook or a calculated remaining-work table did not produce a useful
rescue in the completed controls. More information and more bookkeeping are
not automatically better interfaces.

The older correspondence work supports the same *design direction*, not the
same mechanism. Matching record IDs improved large-batch label assignment
across Qwen and Mistral; on one equal-work comparison, however, singleton
calls scored 665/768 versus 651/768 for large keyed calls and ran faster.
Any new interface must beat a simple baseline on the metric we actually care
about, not merely reduce the number of model calls.

The audit also recovered a positive from the September 14–15 breadth campaign
that had not received a clear follow-up. On 508 numeric financial questions,
letting code evaluate an expression chosen by the model improves strict
matches from **51→108 for the 4B model and 52→102 for the 8B model**. Each
setting still makes one model call, and total tokens increase by less than 1%.
The model chooses the operands and operations; code only performs arithmetic.
Direct outputs were almost always valid numbers, so malformed output is not
the main explanation. A looser, post-hoc numeric tolerance preserves the
direction, but does not resolve known unit, rounding and reference-answer
problems. Cases were previously exposed and many share source documents.
This is supporting evidence for a useful division of work, not official FinQA
accuracy, a novel method, or independent-document confirmation. The
[breadth audit](decomposition-and-transfer.md)
preserves paired counts, source pins and the bounded semantic check needed
before promoting the result.

### C. Some fixed decomposition really helped; learned adaptive decomposition is still open

An earlier selection task is worth recovering from the archive. After Python
organized each candidate's public record, one helper per candidate improved
complete-answer accuracy from **10 to 22 out of 24 attempts** on twelve fresh
cases. It used roughly 3.7 times the input tokens. Confidence-selected partial
refinement did not recover that gain reliably.

That is a useful example of decomposition helping when local decisions are
separable. It was a programmed split, not a learned plan or recursive tree.
In contrast, larger QA comparisons did not establish an advantage: Hotpot's
direct/planned counts were 152/151 of 256 with 3.44 times as many planner
tokens; fresh MuSiQue direct/SFT-planned/RL-planned counts were 54/53/56 of 128.

The right conclusion is not “decomposition fails.” It is that the benefit
depends on what is being separated, what information each part needs, and
whether the final answer actually depends on the helper. Full-context QA,
independent record decisions and shared-inventory crafting test different
conditions. We should stop treating them as interchangeable RLM evaluations.

### D. RL worked locally, but useful transfer remains the weak link

Keep the positive history. An early root continuation improved 5→16/24 on
six new compositions of familiar helper-training questions. An independent
fit improved 9→15/24 on the same exposed contexts. Those studies included
checkpoint selection and narrow data support. A broader fixed-endpoint run
later improved only 13→16/48; its main composition subset stayed 5→5/24.

News-helper RL also produced 422→437 and 422→436 correct answers of 512 in
two training seeds. On fresh examples the models scored 427/429, compared
with 426 for extra SFT. The larger claim weakened. Evidence-sufficiency RL
improved paired answers 10→18/64, but extra SFT reached 19; deeper questions
gave 6/7/6 for start/RL/SFT. These are real results, not failed optimizers.

Today's different-goal crafting diagnostic follows that pattern. The first
raw and assisted RL updates give 4→4/16 and 5→4/16; extra SFT gives 4 and 5.
The second raw update now gives 5/16, one genuine gain without a paired loss.
The assisted second update stays at 4/16; fewer formatting errors do not
restore the lost success, and total cost rises. These repeated
eight-goal diagnostics cannot establish broad transfer or justify choosing
the best-looking checkpoint.

## 3. What most likely limits the current progress?

The evidence supports several distinct bottlenecks, not one universal defect:

1. **An example can ask for an action without making its reason visible.**
   History repair directly tests one remedy. Its exact learning mechanism
   remains unresolved.
2. **A correct decision can be lost in its representation or execution.**
   Record correspondence, ingredient arguments and retrieved-answer delivery
   have each produced measurable failures and specific successful repairs.
3. **Successful components do not guarantee a successful whole task.**
   A helper can classify correctly while the root mishandles its answer.
   A child can make a part while the original crafting goal remains unfinished.
4. **The current RL rule often has little contrast to learn from.** Four
   crafting training goals fail in all 64 attempts across two interfaces and
   two rounds. Their relative advantages are zero. Across the four collections,
   only about a quarter of generated tokens receive nonzero relative credit.
5. **Some comparisons offer little reason for decomposition to help.** A
   final reader that already has all the original evidence can bypass the
   planner. In that setting, good final answers do not validate useful plans.

These explain why another learning-rate sweep or more recursive calls is not
automatically the most informative next experiment. They do not rule out
insufficient training dose, a better optimizer, more capable models or genuinely
useful recursion. Most recent runs are small adapter updates, not large-scale
RL training comparable to published recursive-agent systems.

## 4. The course correction I recommend

**Make teaching repair the primary publication candidate.** Finish the queued
Phi repair test, then freeze a genuinely new goal/world evaluation and the
bounded item-name control. A second-family failure is useful: narrow the
claim or pivot to the responsible representation, rather than add variants
until a favorable number appears. Prior work already covers privileged
teachers and information mismatch; the executable repair and its controls
must carry the contribution, not the broad motivation.

**Keep RL, but change the question from “more updates?” to “better experience?”**
Finish the accepted four-step sequence and objective controls. Prioritize the
already prepared A-only learner-state recovery examples against ordinary
examples. First test whether they improve complete tasks from normal starting
states. Only then ask whether RL can build on that competence. No diagnostic
B score or error should choose the training examples, objective or checkpoint.

**Give recursion one fair, bounded chance to demonstrate usefulness.** The
queued whole-task comparison is the correct next probe. Judge the original
goal under a shared compute budget, not just a child's local success. If it
ties a cheaper flat policy or only solves its own subtask, do not escalate to
a deeper tree. If it helps, freeze a multi-goal replication before learning
the router. The existing one-goal pilot is feasibility, not publication evidence.

**Preserve one longer-term question about selective views.** The successful
candidate-local and correspondence studies suggest learning which records or
subproblem to expose, rather than assuming more decomposition everywhere.
This is a reserve direction, not another concurrent training campaign. It
needs a decision-relevant budget-matched baseline and a failure condition.

The [ranked comparison memo](literature-and-next-tests.md) gives exact readiness,
budgets and promotion rules. The current session has not changed sealed live
sources, admitted duplicate jobs or secretly tuned on B.

## 5. What I would tell the advisors

“We have learned that getting an agent to emit valid steps is not enough. The
steps must make sense from the information it has actually seen, and the
surrounding code must preserve what those steps mean. Our clearest result is
that repairing the context of existing training examples can substantially
improve a small model's ability to finish crafting tasks, without changing
the taught answers. Reward training has produced some local improvements,
but transfer is still limited. Next we are testing whether the teaching repair
works in another model, whether recovery examples improve difficult tasks,
and whether a helper improves the whole task enough to justify its cost.”

This is a coherent preliminary story. It does not need to claim that we have
already solved learned decomposition, invented recursive RL, or produced a
publication-ready general method. The current eight-slide deck appropriately
centers the repair; the historical and latest controls belong in the reading
guide unless they materially change that conclusion.

## Operational assessment

The latest review verified actual saved inference returns, not just occupied
GPU memory. Four fresh-A collections generated 6,826 calls and about 4.99 hours
of summed model-service time. Their four optimizer stages took about 79.2
minutes including replay checks. The eight completed diagnostic cells add
about 4.26 hours of summed model-service time, including reused baselines.
These figures explain why a short weight update can be part of a long study;
they are not an estimate of total allocated GPU hours or productive utilization.

There is also an operations weakness: the most important independent controls
can wait behind long sequences of closely related updates. Keeping the GPU
occupied is necessary, but useful decisions per GPU-hour matter more than
descriptor counts. At the next authorized scheduling boundary, prioritize the
short second-family and whole-task controls over extending unchanged RL.

The externally owned queue continues. This managed session can read its
results and save this audit, but cannot write the external run store or Git
metadata. Therefore it cannot acknowledge the pending automated review,
reorder that queue, or publish these new documents to GitHub. The prior
automatic reasoning loop is awaiting that acknowledgement; it is not restored
merely because GPU jobs are running. The
[restricted handoff](../../experiments/selective_delegation/research_review_20260929/RESTRICTED_HANDOFF_20260929.md)
records the exact outstanding receipt and last verified remote commit.
