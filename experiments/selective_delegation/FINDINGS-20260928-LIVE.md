---
date: 2026-09-28
cutoff_utc: "2026-09-28T12:52:00Z"
status: exploratory
hardware: one_A100_40GB
primary_question: "Which decisions should the model learn, and which details should the harness calculate?"
---

# September 28: findings and next decisions

## The main result remains the teaching comparison

Across the completed six-panel study, demonstrations that discover recipes through
the available tools teach a much more effective model than demonstrations planned
with advance recipe knowledge. Both contain exactly the same per-task answer-token
multisets. Their order and the information available before each answer differ.

This is a promising mechanism to investigate, not a new general law. Prior work
already describes failures caused by teaching with information the learner lacks.
Our next question is narrower: **can we repair the order of demonstrations while
keeping their supervised answers fixed?** Two new training orders, longer-training
controls, and a second model family are being prepared or queued to challenge that
explanation. See the [full study](finding_textcraft_breadth_20260928.md) and
[primary-paper comparison](TEACHER-OBSERVABILITY-METHODS-REVIEW-20260928.md).

## Two additional groups give modest support for execution assistance

These are panel06 additions, not silently pooled into the earlier six-panel cutoff.
Each comparison has eight goal identities and two repeats, with all16 outcomes
observed and independently replayed in the native environment.

| Recipe world | Original actions | Code fills known ingredients | Model calls | Native execution errors |
|---|---:|---:|---:|---:|
| Original world42 | 7/16 success | 7/16 success | 724 → 573 | 358 → 205 |
| Changed world50 | 7/16 success | 8/16 success | 689 → 637 | 329 → 201 |

The first comparison has no changed successes. The second has one win and no losses.
This is useful cost evidence, not a large new accuracy result. There are no transport
failures or missing task outcomes. Context-limit endings decline from3→1 and2→0.
External evidence: `R/analysis-textcraft-breadth-p06-w{42,50}-soriginal-001.json`.

## Many deeper failures leave usable information and materials

A separate analysis checks all384 completed, discovery-trained, execution-assisted
depth4/5 attempts in the fixed six-panel study. Of308 failures,147 still have a
completion path using only recipes already returned and remaining stock. In222
failures the actor repeats an identical stock-deficient craft without changing
inventory. The dataset also has156 failures with incomplete recipe discovery;
these are overlapping diagnostics, not mutually exclusive causes.

For example, the model repeatedly asks to make an item that needs two units of an
ingredient when it has only one. It has already seen a recipe that makes three
units of that ingredient, and has the material to use it, but finishes without
doing so. Another attempt finishes while the final requested craft is already
possible.

This motivates a small test of a public, computed remaining-work table versus a
matched fact table with those computed columns hidden. This is arithmetic help
from code, not a claim that another memory notebook works. Earlier notebook
experiments did not improve success. The [failure analysis](inventory_bottleneck_20260928/FINDING.md)
reports selection, dependence between repeated tasks, examples and limitations.

## The first delegation screen did not reach delegation

All three tiny screening arms stopped after three actual model responses, before
any helper was created. The model queried the goal, then queried two real
prerequisites. Our screening harness rejected those useful queries because they
were not in its imposed alphabetical order.

That is a limitation of this experiment, not evidence against recursive solving.
The task outcomes remain **unknown**, not failures. No transport call failed.
The original summaries inherit an old16-slot group count, although each PLAN and
admission audit correctly describes one task; use the latter for this screen.

A separate, source-pinned repair makes discovery order optional while retaining
actual model-emitted delegation, shared inventory, global budgets and the short
screening cap. Its real saved-request CPU replay passes, including child actions
and charged errors. It has not yet run on the GPU. Reusing this task is explicitly
exploratory, not a new held-out confirmation.

## Two RL updates work numerically; task gains remain unmeasured

The original-action collection completed all 32 attempts, with 24 successes,
790 model responses and no transport failures. There were 348 native execution
errors and one malformed action along the way; these are not hidden by successful
final outcomes. The unchanged model generated this batch on familiar TRAIN goals.
It is a training input, not a before/after improvement or a transfer result.

Four of the eight tasks have both successful and unsuccessful attempts. Their
16 trajectories supply nonzero training credit. All four initial samples per
task choose the same recipe lookup, but later identical observations sometimes
produce different choices. So a fixed first step is not evidence that the model
never explores. See the [completed signal analysis](rl_signal_20260928/RAW-RESULT.md)
for exact-prefix comparisons and token-level accounting.

About 44% of valid crafting-command tokens lie wholly inside the ingredient
field. This measures output cost, not harmful gradients. A separate proposed
test would remove loss on ingredient values only when execution code actually
overwrites them. That is a deliberately biased learning intervention, not a
correction to the standard full-trajectory policy-gradient objective.

The assisted batch is also complete: 28/32 successes, six paired wins and two
losses against the original-action batch, 556 calls and 15,761 generated tokens.
It still has three mixed-reward task groups. Execution assistance makes this
collection cheaper and more successful, but also reduces the amount of nonzero
training credit. The [paired batch report](rl_signal_20260928/BINDER-RESULT.md)
keeps these exposure differences explicit.

The first original-action optimizer completed in 940.6 seconds and committed
adapter, optimizer and random state. Its training/evaluation likelihood replay
agrees exactly. Average sampled-token log probabilities moved by +0.00148 on
positive-credit trajectories and −0.00192 on negative-credit trajectories. The
adapter changed measurably. This is a working numerical update, **not yet evidence
of better task success**. Its fixed before/after evaluations have not completed.

The matched assisted optimizer also completed, in 340.7 seconds. Its original-token
training/evaluation replay agrees exactly, and sampled-token log probabilities
moved by +0.00220 on positive-credit trajectories and −0.00659 on negative-credit
trajectories. It committed its own independently warm-started adapter and optimizer.
Neither update's numerical progress establishes better task performance.

The unchanged-model original-action evaluation completed with 9/16 successes,
326 calls and no missing outcomes or transport errors. These are the predeclared
new rollout seeds, not the four collection seeds. The trained-model evaluation
is now returning real actions. The accepted study evaluates each trained
model through both interfaces. Later studies use new
optimization tasks, separate diagnostic tasks, an extra-supervised-update control,
a changed reward, and a shorter action format.

Do not call queued updates completed, or call an interface improvement learned
improvement. A convincing result needs better task success relative to that
interface's unchanged-weight baseline, with failed and unavailable attempts kept.

## What “new problems” means here

The [generalization audit](GENERALIZATION-AUDIT-20260928.md) confirms 48 distinct
evaluation goal roots, none used as a goal in the 32 supervised demonstrations.
Four did appear as prerequisite products, and most original-world tasks reuse
some familiar recipes. This is new-goal composition, not wholly unfamiliar
vocabulary or domains.

The three changed recipe worlds provide a stronger check: almost none of their
evaluation tasks share an exact named recipe with the supervised examples. They
still share a generator and item vocabulary. A sensitivity analysis grouping
goals that share recipe dependencies retains the positive execution-assistance
effect, but has only five groups and is explicitly exploratory.

## Conflicting teacher suggestions do not automatically make a task hard

The [CPU continuation study](teacher_counterfactual_20260928/README.md) constructs
eight pairs with exactly the same visible prompt and input tokens, but different
hidden recipes. An all-knowing teacher suggests a different next query in each
world. We force either suggestion, or a public-information-based query, and then
let the same deterministic public planner continue. All 48 continuations finish
successfully. The alternative query costs at most one extra lookup.

This is a useful limit on our explanation: inability to predict the teacher's
exact preferred query does not imply inability to solve the task. It does not
explain away the measured SFT gap either. The continuation is a capable scripted
planner, stock is generously provisioned, and no model was trained or evaluated.
None of the oracle query names was already visible; this is not a test of
conflicting choices among visible alternatives. The diagnostic B goals remain
excluded from optimization, and these synthetic worlds do not replace their
fixed queued model evaluations.

[Error-triggered subgoals](REACTIVE-SUBGOAL-PROPOSAL-20260928.md) remain a proposal,
not an implemented or accepted GPU test. It would compare a helper with giving
the parent the same missing-ingredient hint, conditional on the earlier screens.

## Artifact pointers

`R` is
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.
See [the portfolio](EXPERIMENT-PORTFOLIO-20260928.md) for prospective comparisons.
Immutable ACCEPTED/PLAN files, native traces, optimizer checkpoints and completion
receipts live externally. GitHub contains source and compact evidence documents,
not a backup of model weights. The September25 emailed slides retain their
historical cutoff.
