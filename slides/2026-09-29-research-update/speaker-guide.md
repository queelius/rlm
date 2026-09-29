---
title: "Teaching usable next steps: speaker guide"
date: 2026-09-29
evidence_cutoff_utc: "2026-09-29T01:14:27Z"
status: exploratory
---

# Suggested opening

“Our strongest new result is about making existing worked examples easier to
learn from. We changed the information shown before an action, while keeping
the actions taught to the model fixed. That helped again after retraining and
on another group of goals. We also have a smaller reward-training result, but
that still needs a transfer test.”

The talk is about five minutes, with time for discussion. The short pdfpc
notes are prompts, not a script; essential qualifications are visible in the PDF.

## 1. Why usable steps?

TextCraft supplies a goal, initial stock and tools for looking up recipes and
crafting. A tool action either changes the native inventory or returns an error.
Success checks the requested quantity, relative to the original task, when the
attempt ends. Fluent explanations do not earn credit.

A recursive language model can delegate a part of a task. Here we first isolate
the building blocks: selecting a useful next action, finding missing facts and
executing a command. The teaching and RL slides use Qwen3-4B-Instruct-2507 with
small LoRA weight adapters. None of those results establishes a recursion benefit.

## 2. What exactly changed in the examples?

Supervised fine-tuning teaches a model to output the demonstrated next action
given the goal, inventory and interaction history. The original scripted teacher
already knew the recipe graph. It sometimes queried a part whose name had not
yet appeared in the learner's public observations.

The stable-visible repair reschedules recipe queries so each queried name has
first become available through the public interaction. It then reconstructs the
training histories. The pickaxe and part Z are a deliberately simplified
illustration, not a quoted dataset task.

This is an offline repair using a known correct action list, not an independent
planner discovering a solution. Both fit seeds preserve their matched recorded
starting weights, all 366 literal answer strings, original training-row order,
all 23 minibatch target-token denominators and 8,820 answer tokens including EOS.
Crafting order stays fixed. Input histories and input-token dose differ
(414,682 originally; 431,274 after stable repair). This is a much tighter
comparison than changing the entire teacher, but it does not isolate every
possible explanation or equalize all computation.

## 3. What does the replication establish?

The chart compares the quantity-corrected known-recipe teacher with its stable
history repair, without ingredient assistance. The pilot gives 1→14 successes,
a second training seed gives 2→13, and eight additional goals give 3→17, all out
of 32. Each row crosses eight goals, two recipe worlds and two attempt seeds.
The first two rows share goals. There are 16 distinct goal identities in the
whole chart, not 96 independent problems.

The two added comparisons give 5→30/64: 26 paired improvements, one regression
and 37 ties. Their exploratory goal-grouped 95% interval is +21.88 to +57.81
percentage points. It conditions on this fixed exposed design; shared recipes
can still correlate different goals. All listed teaching outcomes are known and
natively audited.

A randomized valid repair also helped (27/64 on the added comparisons).
The discovery-teacher reference achieved 28/64; stable repair has not clearly
outperformed it. More training partly rescues the original examples:
1→8→8/32 at 23, 46 and 69 updates. The repair's 23-update pilot gives 14/32.
Thus the claim is improved learning at the tested budgets, not unlearnability
of the original examples or an unlimited-training advantage.

These panels had been examined in the wider research campaign. The added
measurements were fixed after the pilot, but this is not a final untouched test.
The general problem of teacher–learner information mismatch already has prior
work; this deck makes no novelty claim.

## 4. What does the first RL update show?

Reinforcement learning (RL) learns from task-completion rewards. Here, one
training step uses a collected batch of attempts, rather than new worked examples.

Starting from the same discovery-trained model, we collected a batch of attempts
and made one terminal-reward weight update separately for each tool interface.
Each row compares the unchanged warm actor with the actor trained for that row.
It is not one common updated model achieving both reported improvements.

Without assistance, successes rose 9→13/16 (four paired wins, no losses).
With observed-recipe assistance, they rose 14→16/16 (two wins, no losses).
Code fills ingredients from an already observed recipe; the model still chooses
the item and output quantity. It does not receive hidden recipes or stock repair.

All eight evaluation goals were used in RL training; the evaluation sampling
seeds are new. There is one realized collection/update per interface. The
eight-goal intervals are +6.25 to +50 percentage points without assistance and
0 to +31.25 with it, excluding training-seed uncertainty.

Cost rose: output tokens 9,610→10,224 (+6.4%) and 8,462→10,325 (+22.0%).
Calls rose 326→340 and 277→303; service time rose 7.4% and 24.0%.
One long duplicate-field rejection loop accounts for much of the assisted cost,
and remains included. Other wins involve quantity completion, recipe correction
or longer recovery. This is not a clean demonstration of efficient planning,
held-out transfer, or superiority to additional supervised training.
Different-goal and extra-SFT controls were pending at the fixed cutoff.

## 5. Why these next tests?

We will use Phi-4-mini to test whether the specific history repair works in a
second model family. Its completed original-package comparison favored discovery over known
recipes (4/16 versus 1/16 without assistance), but there is no repaired-Phi result
at the cutoff. Equal update counts across model families do not mean equal
tokens, adapter sizes or computation.

The helper screens created real child calls. A fixed helper made its requested
two units; an adaptive helper made only two of four. Every original task was
censored by the 32-response screening cap, so those root outcomes are unknown,
not failures. A successful local subtask alone cannot establish whole-task value.

The selected next comparison is no helper versus fixed and adaptive rules,
one exposed goal and two paired new sampling seeds. Every arm gets the same
global 96 calls, 8,192 generated tokens, 8,192-token context and at most one
child; child actions spend the parent's budget and shared inventory. This is
deterministic harness routing, not trained recursion. It is a small mechanism
probe, not evidence of generalization.

If asked about other harness work: a remaining-work table solved 0/8, exactly
like its control, while increasing output cost. A household-game command result
was only 4/24 and concentrated in one scene. Neither justifies expanding a grid.

## Evidence and exact provenance

- [Fixed-cutoff plain-language findings](../../experiments/selective_delegation/FINDINGS-20260929.md).
- [Teaching synthesis and all controls](../../experiments/selective_delegation/teaching_synthesis_20260929/FINDING.md)
  and [machine-readable native-audit lineage](../../experiments/selective_delegation/teaching_synthesis_20260929/FINDING.json).
- [Full crossed RL matrix, behavior and cost](../../experiments/selective_delegation/rl_outcomes_20260928/README.md).
- [Harness results and censored helper screens](../../experiments/selective_delegation/harness_synthesis_20260929/README.md).
- [Portable chart data and source hashes](evidence.json).

Do not replace pending statements with later results without updating this
deck's visible claims, chart data, notes and cutoff together. Preserve the
September 25 decks as historical versions.
