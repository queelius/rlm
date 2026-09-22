---
status: exploratory
evidence_cutoff_utc: 2026-09-22T09:35:00Z
question: Does demonstrating public information gathering improve learned task execution?
primary_comparison: fixed_original_prompt_public_vs_privileged_teacher
training_seeds: 1
evaluation_task_parents: 8
exposure: previously_examined_validation_tasks_in_shared_recipe_world
publication_status: promising_teaching_package_signal_pending_transfer_and_controls
---

# Public-information discovery teacher: exposed TextCraft readout

## Result

On the eight already-exposed TextCraft validation task parents (two fixed seeds each), the
public-information teacher's action-SFT readout completed **10/16** original-prompt episodes,
versus **3/16** for the earlier privileged-teacher readout.  The paired difference is **+7/16
(+43.75 percentage points)**: eight public wins, one loss, and seven ties.  A 20,000-draw
task-parent bootstrap retaining both seeds gives a 95% interval of **[+6.25, +81.25] points**.
All 16 pairs were observed; no result was imputed.

The reminder prompt is a secondary consistency check, not a separate selection result: public
teacher **10/16** versus privileged teacher **2/16**, with eight wins, no losses, and a
task-cluster interval of **[+18.75, +81.25] points**.

This is a meaningful exposed-panel improvement in executable behavior, not a claim of fresh-task
generalization or a causal proof that teacher ordering alone caused the change.  The two one-epoch
SFT datasets have the same 366-row dose and 23 updates, but their teacher histories and therefore
their token distributions differ.  The eight parents share one recipe world and the two seeds per
parent are correlated.  The original prompt is primary; the reminder is secondary.

Both training PLAN files fix seed2026092208 and the same optimizer, learning
rate, batch sizes, LoRA settings and package versions. Their step-zero adapter
files are byte-identical (SHA256
`6c1543bbf6a235890f3bb9f73f545c9b845cb8d93ee45a8bca11850aeaf2897a`).
Thus this is not a comparison between different random initial adapters.
It still needs another training seed before claiming stability across training.

A target-multiset audit further narrows the dose difference.  All 32 tasks have identical
step-zero prompt strings.  Canonical parsed actions, raw target strings, and unmasked label-token
tuples all agree for `get_info` and `finish` on all 32 tasks and for crafts on 31.  The sole
difference is `textcraft_synth.train.1029`: privileged rows teach
`raw_t8:6 -> output_count:12`, public-teacher rows teach `raw_t8:4 -> output_count:8` for
`t4_i1`.  Thus the supervised target multisets are near-identical at both raw-string and target-
token levels.  This is still not order-only causality or identical loss conditioning: row/action
order and later prompt histories differ.  The audit also reads both actual checkpoint-zero adapter
files and verifies their bytes against their COMMIT hashes before comparing them.  Receipt:
`analysis-textcraft-target-multiset-004.json` (SHA-256
`2ec8717a2eb769a1ef42b2471e8acfde7e6c7573ca9bc349a80cc92c7094c84c`).

## Work and behavior

For the original prompt, public-teacher execution used 370 native calls, 835,133 prompt tokens,
and 11,089 completion tokens (882.5 summed service seconds), compared with 646 calls, 1,859,461
prompt tokens, and 12,729 completion tokens (1,080.4 summed service seconds) for privileged
teacher.  These are observed rollout costs, not matched-compute claims: the policies took different
trajectories.  Neither run had transport failures.

The sealed behavior audit adds a plausible behavioral correlate, while remaining descriptive:

| Original-prompt behavior (16 episodes) | Privileged teacher | Public teacher |
| --- | ---: | ---: |
| First physical query was the root target | 3 | 16 |
| Ever queried the root target | 11 | 16 |
| Queries of nonexistent items | 344 | 0 |
| Repeated nonexistent-item mentions | 264 | 0 |
| Queries of a recipe already returned in earlier public feedback | 70 | 11 |

The last row counts repeated static facts, not automatically wasted actions: a query also returns
the current quantity of that item, although the full inventory was already public at every turn.
Importantly, the upstream `can_craft` field means that a recipe exists, **not** that current
inventory can execute it. It is not a dynamic feasibility signal. The public-teacher trajectories
were much less dominated by repeated static-recipe queries.

## Completed examples

* **Gain — `textcraft_synth.val.325`, seed/repeat 0.**  The public-teacher policy first queried
  its actual target `c3_i2_23`, queried the two discovered prerequisites, crafted them, and
  finished successfully in seven calls.  The privileged-teacher policy first queried an unrelated
  nonexistent item, repeatedly returned to nonexistent names, eventually finished after 73 calls,
  and still lacked all three target items.  This is an illustrative trajectory, not evidence that
  every gain arose from the first query.
* **Failure retained — `textcraft_synth.val.567`, seed/repeat 0.**  The public-teacher policy did
  start at its depth-four target and discovered several prerequisite recipes, but attempted crafts
  before enough prerequisites were available, made quantity/ingredient errors, and finished after
  51 calls without either required target item.  The privileged-teacher policy also failed.  Thus
  root-first public discovery is not sufficient for multi-step quantity accounting or termination.
* **Public loss retained — `textcraft_synth.val.207`, seed/repeat 1.**  Privileged teacher
  succeeded in 40 calls, while public teacher queried its target first but failed after 65 calls.
  The paired result contains a real loss, rather than an all-win narrative.

## Provenance and next question

The primary comparator is
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/analysis-textcraft-public-teacher-001.json`
(`sha256 f80a4000e9a2ec2cbae14f1ccf52f54f69a569d624ed3c4deed6cf8dd260a582`).  The behavior audit is
`analysis-textcraft-public-teacher-behavior-002.json`
(`sha256 fb69e285d0cc39786d050d8886a227ff96e1cd83a45ea582ea3656168b2c901f`), paired against the
earlier privileged report `analysis-textcraft-trained-readout-001.json`
(`sha256 f84ea2aa2a86c1190fe4db0cb069242f0c1e88e35849e6a1a63eca819f8e9887`).

The smallest belief-changing follow-up is an independently selected changed-world or held-task
readout with public-only prompts, preserving the action budget and scoring.  It would test whether
the learned root-first, query-then-craft protocol transfers beyond this shared recipe world.  If
that panel does not retain a directional benefit, retire the broad discovery/generalization story
and keep the present result as an exposed competence/protocol finding.
