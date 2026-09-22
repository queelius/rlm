---
status: exploratory
evidence_cutoff_utc: 2026-09-22T10:45:00Z
question: Does the teaching-package advantage survive changed recipes?
training_seeds: 1
evaluation_task_parents: 8
evaluation_worlds: 1
exposure: retained_goal_names_in_a_new_recipe_world
publication_status: positive_transfer_signal_pending_fresh_goals_and_training_replication
---

# Teaching public discovery still helps after the recipes change

## Result

On the eight previously examined root goals, evaluated in the reconstructed world-43 recipe
environment with new constructed inventories, the public-information-teacher adapter completed
**10/16** episodes versus **3/16** for the privileged-teacher adapter. The paired comparison has
7 public wins, 0 losses, and 9 ties: **+43.75 percentage points**, with a 20,000-draw task-parent
bootstrap interval of **[+12.5, +75.0] points**. All 16 matched slots were observed; there were no
unknown outcomes or transport failures.

This is stronger evidence than the earlier same-world exposed readout that the public-teacher
package can survive a recipe/inventory change. It is not a fresh-goal test, a matched-difficulty
comparison with world 42, or evidence of stability across many new worlds: the root goals and
two correlated seeds are retained, while dependency graphs, constructed inventories, and task
difficulty changed together.

## Cost and behavior

Public teacher used fewer calls (547 versus 700) and fewer prompt tokens (1,699,797 versus
2,048,376), but more summed native service time (1,428.6 versus 1,206.8 seconds) and more output
tokens (17,410 versus 13,962). It also had more native action errors (248 versus 55) and
invalid-schema outputs (20 versus 1). Both arms had zero child nodes. These are actual physical
costs for the two new world-43 runs, not an equal-compute claim.

Public teacher recorded 10 root calls after the requested quantity had been met, versus 3 for
privileged teacher. Inspection of the report-bound audits shows all ten public instances were
successful episodes; none is a failed root that continued after goal completion. Thus this count
alone is not evidence of a termination regression.

## Representative trajectories

* **Gain — `textcraft_synth.val.325`, repeat 0.** Public teacher queried target `c3_i2_23`,
  queried its returned prerequisite, crafted it, produced the target, and finished in seven calls.
  Privileged teacher first queried/crafted nonexistent items, only later reached the target recipe,
  and finished unsuccessfully after 30 calls. These are saved calls
  `t00-r0-flat-original-c000` onward in the respective world-43 outputs.
* **Important retained failure — `textcraft_synth.val.567`, repeat 0.** Public teacher did query
  the depth-four target and multiple returned branch recipes, but its first craft failed and it
  ultimately hit context cap after 82 calls with 58 native action errors. Privileged teacher also
  failed. The gain therefore does not establish robust multi-step quantity execution.

These examples describe completed trajectories; they do not identify why either teacher succeeded
or failed.

## Provenance and decision implication

Authoritative comparison:
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/analysis-textcraft-world43-001.json`
(SHA-256 `057ef2810a2da7132b699f19e107a58d993780d25f085d497b92a77f872701df`). The report binds both
native arm receipts and applies the parent-cluster uncertainty calculation.

The result supports transfer to this changed recipe world, but not a generic
generalization claim. The most informative next comparison is the already accepted fresh paired
panel: it can separate retained-goal/world sensitivity from transfer to unexamined goals. If that
panel does not preserve a directional benefit, retain world-43 as a useful changed-environment
diagnostic rather than evidence for broad public-discovery transfer.
