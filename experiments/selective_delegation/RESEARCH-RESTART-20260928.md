---
date: 2026-09-28
status: active_exploration
question_ids:
  - TC-TEACHING-MECHANISM
  - TC-EXECUTION-AND-RL
  - TC-ADAPTIVE-DECOMPOSITION
allocation_end_utc: "2026-10-01T09:50:35Z"
evidence_cutoff: "Six complete breadth panels, 00 through 05"
---

# Research restart and decisions

The user requests autonomous research until tomorrow, encourages parallel
literature and experiment preparation, and prioritizes useful use of the GPU.
The account endpoint reported 99% remaining at 10:01 UTC. The current allocation
started at 09:50 UTC and supplies one A100 40 GB until October 1 at 09:50 UTC.

## What the completed campaign changes

The September 24 campaign stopped near its fixed September 27 deadline. There
are 192 completed, natively audited runs covering six groups of eight goals,
four recipe worlds, two training seeds, two teaching methods, and two execution
interfaces. Each condition attempts each goal twice. Repeated attempts, worlds,
and model seeds are correlated; these are not 768 independent tasks per condition.

The first reconciliation gives:

| Teaching examples | Model supplies ingredients | Code supplies ingredients |
|---|---:|---:|
| Discover recipes through public queries | 323/768 | 381/768 |
| Plan from known recipes, with quantity bug corrected | 39/768 | 48/768 |

The discovery model gains 58 successful attempts with 2,417 fewer model calls
(24,759 to 22,342). The much larger teaching-method difference remains. These
packages share tasks, row counts, update counts, and exactly 8,820 supervised
answer tokens. The full reconciliation additionally verifies identical per-task
action multisets. They differ in action order, conditioning histories and input
token exposure (5.64% more for discovery). A causal explanation of the teaching
difference still requires further controls. See the
[complete analysis](finding_textcraft_breadth_20260928.md).

The word `corrected` in old directory names means a quantity-corrected
known-recipe teacher. It does **not** mean a corrected discovery-trained model.
Earlier short status reports used this label without enough explanation.

## Initial GPU work

The old campaign also completed the unassisted arm for panel 06/world 42. The
restart finishes its already chosen assisted counterpart, then compares both
interfaces on the same new panel with recipe world 50. These are three new GPU
collectors, each capped at 45 minutes, followed by native audits and paired
analyses. Episode receipts preserve partial progress. Completed trajectories are
not rerun. Expected useful collection coverage is about one hour to 90 minutes;
the queue has a 3.5-hour outer bound, not 3.5 hours of promised GPU work.

The first collector returned 47 real model responses and zero failed calls at
the automatic first-response check. Its independent episode data remain outside
the six-panel synthesis above.

### First completed restart comparison

Panel 06/world 42 completes at 10:23 UTC: original and assisted execution both
solve 7/16 attempts, with zero paired wins or losses. Assistance uses 573 rather
than 724 model calls (20.86% fewer), with 205 rather than 358 native action
errors. This is a cost benefit, not a new accuracy gain. All 16 pairs are observed
and independently natively audited. The result stays outside the fixed six-panel
cutoff, including its null success difference.

Evidence: `analysis-textcraft-breadth-p06-w42-soriginal-001.json` under the
external root below. The next changed-recipe comparison is running.

External operation:
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/resume-20260928-001/`

`ACCEPTED.json` holds the exact source/input pins, commands, deadline, and
allocation variables. `launch_source.py` preserves the executed launcher bytes
before working-copy formatting. `campaign/` holds response checks and supervisor
state; `queue/` holds per-job logs and exit records. GPU ownership remains in
each scientific run's OWNER/TERMINAL receipts.

Two further queues are accepted and waiting for their predecessors:

- `rl-collect-queue-20260928-001/`: paired original/assisted TRAIN collections,
  each with eight goals and four fresh sampling seeds.
- `rl-update-queue-20260928-001/`: one terminal-reward update per interface,
  unchanged-checkpoint controls, and four evaluations crossing the two trained
  checkpoints with the two execution interfaces. Readouts have two attempts per
  goal. Failed or flat-reward training is skipped explicitly, not called an RL
  improvement. Expected useful work is two to four hours after collection;
  the 12-hour queue allowance includes waiting and is not promised GPU coverage.

These use exposed TRAIN goals for a mechanism pilot. Three further queues are
now accepted, in order:

- `teacher-controls-queue-20260928-001/`: two matched visible-query teaching
  orders, then paired world42/50 readouts.
- `interface-dose-queue-20260928-001/`: compact-action SFT/readouts, then two
  additional epochs for both original teachers with fixed checkpoint46/69
  readouts in both worlds. The existing optimizer and random state are restored.
- `fresh-rl-queue-20260928-002/`: up to four fresh collection/update/readout
  cycles per execution interface on new TRAIN goals, with a disjoint diagnostic
  group and an additional-SFT control. This is not official held-out confirmation.

The last two queues wait for complete predecessor-queue release, so a scientific
job skipped before acquiring the GPU cannot strand independent following work.
Older accepted sources and receipts remain unchanged. At10:52 the six accepted
queues contain61 scientific stages, many conditional, and25 CPU stages. Their
expected useful work is about a day if both learning arms stay informative.

`health-watch-20260928-001/` records first model responses, failures, skips and
completions without Codex tokens. Current source/input checks and small runtime
fixtures passed. The main quota was94% at10:45; no reset is assumed.
The [experiment portfolio](EXPERIMENT-PORTFOLIO-20260928.md) distinguishes accepted
work from proposed comparisons; do not mistake a proposal for an active run.

## Ranked next experiments

1. **Does execution assistance make useful decisions easier to learn with RL?**
   Compare fresh TRAIN rollouts from the same discovery checkpoint with and
   without ingredient assistance. Use native task success, original sampled
   action-token probabilities, matched task/seed schedules, and a small update.
   Evaluate improvement against each arm's own initial score. If promising,
   compare several fresh updates and an extra supervised-training control.
   Frozen evaluation gains alone do not establish better learning.
2. **Does asking for fewer action fields help?** Separate choosing an item and
   amount from generating ingredient arguments that the assisted environment
   subsequently replaces. Compare a simpler command format with the existing
   format, with explicit adaptation training where needed. This tests whether
   unused output decisions add learning noise or merely generation cost.
3. **What makes discovery demonstrations easier to learn?** Distinguish action
   order and observable information from training dose and presentation. Match
   tasks and objective; treat package comparisons as motivation for controlled
   ablations. Check completed experiments before accepting additional training.
4. **When is further decomposition useful?** Prospectively vary shared
   prerequisites and inventory constraints at matched depth. Compare direct
   action, fixed delegation and a learned delegation decision. The current
   constructed inventories are generous, so current depth failures do not
   establish a resource-scarcity mechanism.

Primary-literature review and CPU analysis proceed alongside GPU collection.
Candidates must specify the decision they resolve, a short runnable comparison,
and a retirement condition. Additional evaluation grids require a new question;
the six-panel result already answers whether the first small effect repeats.

## Reporting

Preserve the emailed September 25 deck as a historical cutoff while preparing a
new research briefing. Any later slide update must show the broader evidence,
the remaining teaching confounds and whether actual RL improved. Save clear
positive, negative, incomplete and invalid results, and source links for the next
session. Publish source and research notes at completed checkpoints; raw traces
and weights remain in the external research store.
