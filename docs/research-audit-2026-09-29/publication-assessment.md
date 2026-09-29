---
id: teaching-history-publication-assessment-20260929
title: "A focused paper candidate: repairing histories in tool-use examples"
date: 2026-09-29
status: proposal_not_launched
primary_question: "Can rebuilding the history attached to fixed demonstrated actions improve learning on new tasks?"
evidence_level: repeated_exploratory_comparisons
current_scope: TextCraft_Synth_Qwen3_4B
decision: prioritize_bounded_teaching_study_over_new_unrelated_branches
---

# Recommendation

Pursue a focused empirical paper about **repairing the histories in tool-use
training examples**, rather than an umbrella paper claiming successful RLM
learning or general RL improvement. A provisional title is *Same Actions,
Different Histories: Repairing Examples for Tool-Using Language Models*.

There is enough signal to justify a concentrated follow-up, but not yet enough
to call the method novel, broadly useful or publication-ready. Start the paper's
question, method description and evidence ledger now. Let prospective tests
determine its eventual claim and venue, not the wish to salvage past compute.

## What the existing evidence supports

The repair reschedules existing recipe lookups, replays them through the native
game and rebuilds the observations attached to each taught action. It preserves
literal action strings, answer token IDs, crafting order, original training-row
order and paired batch target-token totals. It does not generate a new solution
or train from a different list of correct actions.

Original→repaired comparisons are 1→14, 2→13 and 3→17 successes out of 32.
These involve sixteen distinct goals overall, repeated attempts and two recipe
worlds. They are exploratory panels, not ninety-six independent unseen tasks.
The [teaching synthesis](../../experiments/selective_delegation/teaching_synthesis_20260929/FINDING.md)
and [audit](teaching-and-execution.md) document the evidence and controls.

Important contrary evidence is already available:

- The policy trained on original demonstrations performs poorly, although the
  demonstrations themselves succeed. An earlier uncorrected teaching package
  scored below the base model; the present matched comparison already includes
  the quantity correction. These are not the same baseline.
- A teacher that discovers recipes in order performs similarly to repair.
  The repaired method has not established an advantage over this alternative.
- More training partly rescues original examples: 1→8→8/32 at fixed
  23/46/69-update checkpoints on the initial panel.
- Input length, recent history, visible names and total compute change.
  Fixed answers do not isolate a single causal mechanism.
- The specific second-model repair is pending. Other Phi teaching results
  cannot stand in for it.

The current defensible conclusion is that **the same actions can teach very
differently when paired with different executable histories**, at these budgets
and in this environment. Better planning is not established.

## Closest prior work changes the novelty claim

The broad idea that teacher information or input histories can make imitation
harder is established. [Causal Confusion in Imitation Learning](https://arxiv.org/html/1905.11979v2)
shows how conditioning can change closed-loop behavior despite good imitation
fit. [Student-Informed Teacher Training](https://arxiv.org/html/2412.09149v2)
addresses mismatches between a privileged teacher and the student's observations.
[Privileged Information Distillation for Language Models](https://arxiv.org/html/2602.04942v3)
studies related teacher/student information differences for language-model training.

A particularly close method-level comparison is DATS in
[Tri-Manual Visuomotor Imitation Learning of Robot Policies](https://arxiv.org/html/2607.25731v4).
It reschedules fixed robot-motion segments under dependency constraints to
construct training demonstrations. It therefore rules out a broad claim that
offline demonstration rescheduling is new. Its composite actions and observation
construction differ from our fixed literal tool-action targets and native replay.

A bounded search has not identified the exact conjunction we test: fixed
literal complete tool-action labels, fixed optimizer presentation of those
labels, and regenerated native execution histories. That is a possible distinction,
not proof of priority. The paper must establish why the distinction is useful,
not merely describe a different implementation.

## A bounded evidence plan

The following is a proposal, not a newly launched campaign. Existing accepted
GPU work continues under its owners. The current managed session cannot modify
the external queue or its output store.

### 1. Run cheap scope checks first

Finish the already queued Phi repair screen. Prepare a small development set
of fresh recipe structures, plus a native-valid item-renaming check. Use a
separate development split for deciding whether the method runs sensibly; do
not inspect the later confirmation outcomes and then redesign the test.

Check whether repair applies to an existing second tool environment on CPUs.
Report eligible traces, ineligible traces and replay failures. ToolHop and
ToolSandbox are candidates to inspect, not datasets already used here. Their
dependency/state constraints may make this particular repair inapplicable.
Do not construct a contrived new benchmark just to guarantee an effect.

### 2. Freeze one primary fresh-problem comparison

Proposed design: 48 independently generated goal/dependency graphs, three
predeclared Qwen training seeds, and four conditions:

| Condition | What it tests |
|---|---|
| Original examples, 23 updates | The matched-label, matched-update baseline. |
| Repaired examples, 23 updates | The specific intervention. |
| Discovery-based teacher, 23 updates | A stronger available teaching alternative. |
| Original examples, 69 updates | Whether simply training longer closes the gap. |

Use one predeclared rollout seed per graph/fit/condition initially: 576 trained
policy episodes. Reuse the exact frozen existing checkpoints where appropriate;
do not choose a fit because its old test score was favorable. Include the base
model as a descriptive reference, not a substitute for the trained controls.

Define genuinely new before evaluation: unused world seeds, no previous goal
identities or training-query products, and normalized dependency-graph checks.
Report shared components and motifs rather than claiming perfect independence
from a different numeric world seed. Choose depth/quantity strata in advance.

Primary outcome: native whole-task completion. Also record native action errors,
invalid outputs, calls, generated/input tokens, wall time and fit/collection
cost separately. Keep every planned slot and every failure category.
Uncertainty should respect graph/world units and paired conditions. Show each
training seed separately; three fits do not precisely characterize all fits.

At recent task latencies, this is roughly 10–20 A100-hours of evaluation,
plus preparation/fits and any base-model reference. This is a planning range,
not a reservation or a measured total. Stop and re-estimate after a timed pilot.

### 3. Separate information from copying and generic reordering

The existing randomized repair still enforces name visibility. It cannot
distinguish helpful evidence from generic rescheduling. Add a native-feasible
reordering control with similar displacement but without the visibility rule,
if such traces can be constructed without changing actions or success.

Test consistent unfamiliar aliases for item identifiers in both the model's
view and legal native actions. This tests reliance on familiar names but alone
does not distinguish copying from understanding a recipe relation.

A small supporting label-likelihood probe can preserve names while masking
recipe relationships, with a matched irrelevant-mask control. This is an
input diagnostic, not an executable whole-task evaluation or causal proof.
Keep it separate from the headline completion metric.

### 4. Replicate scope only if the lead survives

Use a second model family with predeclared conditions and more than one fit,
then one naturally applicable second task if claiming a general tool-agent
method. A second collection of crafting names is not a second domain.
Do not keep trying model families until one is positive.

The optional alias and second-model comparisons could add roughly 7–14 GPU-hours,
depending on task length. A second environment is not promised within this
allocation. Do not let porting infrastructure displace the decisive fresh-task
comparison. Queue admission still depends on actual remaining time and owners.

## Decide in advance what the results mean

Before inspecting confirmation outcomes, freeze a practically meaningful
matched-dose threshold, for example ten percentage points, and an analysis
that reports paired task uncertainty and seed consistency. Do not extend the
sample until a desired significance level appears.

- **Repair helps on new problems and multiple fits:** pursue the controlled
  empirical paper; establish scope and the strongest competing explanation.
- **Repair matches a good discovery teacher:** possible usefulness for reusing
  existing traces, but not superiority to the best teacher. Measure any claimed
  data-generation or computing advantage; it is not currently established.
- **More original training catches up:** claim finite-budget behavior only if
  actual compute accounting supports an advantage. Do not claim a higher ceiling.
- **Only old worlds or the weakest baseline show a gain:** narrow to a teaching
  pathology/case study or retire the general-method paper.
- **Only names/copying explain the gain:** report that mechanism honestly. It
  could still inform data construction, but is not learned task decomposition.

## What belongs in this paper, and what does not?

Core figures: a worked original/repaired example; fresh-task completion with
strong baselines and seed-specific results; training-budget curves; a compact
mechanism/scope figure if justified. Publish the transformation, manifests,
replay checks, eligibility/failure accounting and licensed evaluation assets.

The code-assistance and RL findings motivate careful separation of interface,
learning and planning. They need not become another large experimental section.
Finish bounded accepted RL comparisons, but do not make an RL breakthrough or
a new recursive architecture a prerequisite for this paper. Recovery teaching
and complete-task helper value remain separate possible projects.

The next discussion with advisors should be about the missing evidence and
appropriate scope, not a promise of novelty or acceptance. A narrow result
that survives serious alternatives is more useful than a collection of
unrelated favorable scores.
