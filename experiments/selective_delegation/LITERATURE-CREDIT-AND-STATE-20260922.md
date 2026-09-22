---
status: research_inputs_not_implemented
reviewed_utc: 2026-09-22
question: Which local learning signals could address the observed execution failures?
decision: finish_terminal_RL_control_before_changing_objective
---

# Recent credit-assignment work: what it changes for our next experiments

The completed TRAIN check has usable success/failure variation, so it is worth
finishing the accepted terminal-reward pilot. It does not establish that a single
success/failure signal gives good guidance to every action. The papers below make
that distinction especially relevant. This is a targeted reading of primary
papers, not an exhaustive novelty review or an independent reproduction.

## Four relevant approaches

**TRACE (July 15, 2026)** estimates progress at tool-call boundaries using a frozen
model's probability of the known answer. It combines changes in that estimate
with the final outcome signal. Its exact telescoping argument applies to the
one-step value differences, not automatically to the complete multi-step,
normalized training objective. [Paper, version 1](https://arxiv.org/html/2607.13988v1).

Our inference: this is more directly suited to our question-answering experiments
than to crafting. Crafting has a verifiable final inventory, not one natural
gold-answer string whose likelihood measures progress. Using the probability of
`finish` would not establish that the inventory is correct. Before any adoption,
test whether a proposed progress score actually separates useful and redundant
transitions on saved TRAIN examples. Do not silently add an answer oracle to the
policy's inputs or call the resulting new objective unchanged terminal RL.

**BEACON (May 7, 2026)** uses task milestones to separate portions of a long
trajectory and estimate credit at multiple scales. Its detectors use environment
feedback, but are task-specific. The paper explicitly identifies automatic
milestone discovery as an open direction and discusses assumptions about how
much earlier history matters after a milestone.
[Paper, version 1](https://arxiv.org/html/2605.06078v1).

Our inference: a successful intermediate craft is an observable event, but not
necessarily useful progress. It may consume resources needed by another branch.
That makes “reward every valid craft” a poor automatic translation. A small
future comparison could test final-outcome credit against credit anchored to
explicit, checkable subgoals, retaining terminal success as the evaluation.
It must declare who chose those subgoals and what information they used.

**SkillGate (August 19, 2026)** separates credit for selecting a skill from credit
for executing the task. Its selector signal is directed to the skill-name tokens
and uses an oracle identity within a candidate slate; this is not unlabeled
discovery of an arbitrary decomposition. The paper includes controls that vary
where the selection reward lands.
[Paper, version 1](https://arxiv.org/html/2608.18852v1).

Our inference: it is close prior art for training an RLM's decision to delegate
separately from the helper's eventual success. It argues for measuring whether
the delegation decision receives useful credit, rather than merely adding a
classifier or a larger router. An analogous experiment here needs a defensible
training target for delegation. A hand-selected “correct” decomposition would
change the research question and must be exposed as supervision.

**ReBel (May 19, 2026)** trains explicit beliefs about the environment, checks
them against observable feedback, and groups steps by related belief states.
Unobservable predicates are masked rather than treated as false. Its benchmark
representations and consistency checks are designed for their environments.
[Paper, version 1](https://arxiv.org/html/2605.20061v1).

Our inference: this is relevant to quantities and discovered-recipe tracking.
However, our prompt already supplies current inventory. Adding another copy of
inventory is not automatically a new belief capability. The useful distinction
would be between observed facts, derived requirements, and unverified guesses.
A prospective training target must avoid supplying undiscovered recipes under
the label of state tracking.

## Ranked decisions for this campaign

1. Finish the accepted terminal-RL / extra-SFT / unchanged-model comparison.
   This tests whether the now-competent model benefits from final-outcome learning
   before introducing a second reward mechanism.
2. Analyze which actions receive positive and negative credit in the actual
   sampled batch. Successful recovery can give positive credit to earlier errors;
   the incidence is measurable, but does not by itself prove a biased gradient
   or predict the optimizer's effect.
3. Keep the [public recipe-binding test](TEXTCRAFT-PUBLIC-RECIPE-BINDING-IDEA.md)
   separate from reward design. It changes the action interface while leaving
   choices of subgoals, quantities and order to the model. This can distinguish
   an execution-interface burden from a planning limitation.
4. If execution remains the obstacle, compare a narrowly defined local signal
   against the same terminal baseline. Use at most two updates and the existing
   eight-task/four-sample TRAIN shape initially, with one A100 and a three-hour
   cap; add matched readouts. The existing readiness run took about 32 minutes,
   but new credit computation can add cost. Promote only after a measured benefit
   beyond extra SFT, then test a fresh training seed and new evaluation tasks.

These readings narrow the novelty claim: local credit, state summaries, and
separate selector training already have close precedents. A stronger contribution
would establish **when an interface or decomposition decision becomes learnable,
why a controlled change helps, and where that improvement transfers**. Our current
results motivate that program; they do not establish it.
