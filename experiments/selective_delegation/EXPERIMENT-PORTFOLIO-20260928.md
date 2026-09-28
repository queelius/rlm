---
date: 2026-09-28
status: active_exploration
allocation_end_utc: "2026-10-01T09:50:35Z"
hardware: one_A100_40GB
source_branch: research/selective-delegation-20260921
main_question: "Which choices should the model learn, and which details should code handle?"
evidence: finding_textcraft_breadth_20260928.json
---

# Several informative comparisons, one evolving research program

Latest queue amendment: [16:01 UTC follow-up admission](FOLLOWUP-ADMISSION-20260928.md).
The two new evidence-driven studies are accepted, not yet completed.

The user requests many experiments across different ideas. Breadth means testing
different explanations and capabilities, not repeatedly measuring the same small
effect. GPU jobs run sequentially on one card; data preparation, native replay,
literature review and analysis run concurrently on CPUs. No repeated approval is
needed. Check the external live queue for execution status; this document records
the research design, not a real-time dashboard.

## What we know

With the same supervised answer tokens, teaching recipe discovery is much more
effective than teaching actions chosen with advance knowledge of recipes. In
the completed six-panel study, discovery succeeds in 323 of 768 attempts; the
known-recipe method succeeds in 39. Code that fills in already-observed recipe
ingredients raises these to 381 and 48. These are 48 goal identities repeated
across recipe worlds and model seeds, not 768 independent problems per condition.

This suggests studying the **information and decisions we ask the model to
learn**, not only changing the RL optimizer. It does not establish a new general
principle or a successful learned recursive policy. Prior work already studies
privileged teachers, recursive RL and action/argument separation.

## Current comparisons

**September 28, 15:58 UTC evidence update.** The first teaching-order pilot and
six-cell RL evaluation are complete. Stable/random query-order repairs produce
14/32 and 12/32 successes versus the original teacher's 1/32, with the same
supervised minibatch targets. One raw-interface RL update improves 9/16 to 13/16
on familiar goals; assisted RL improves its own 14/16 baseline to 16/16.
These are small, dependent exploratory panels, not broad generalization.
Compact actions reduce output tokens by about 39%, with mixed success changes.
Read the [current findings](FINDINGS-20260928-LIVE.md) and
[RL behavior/cost analysis](rl_outcomes_20260928/README.md).

Both fixed-dose training continuations have also finished; their task readouts
are running. The following table records each comparison's **original acceptance
status**, not its live execution status. Two additional prepared studies respond
to the positive signals: [teaching replication](teaching_replication_20260928/README.md)
and [fixed-RL-checkpoint transfer](rl_transfer_20260928/README.md). They do not
replace the independent quantity, helper, second-model or ALFWorld comparisons.

| Question | Small comparison | Status at acceptance | What changes the next decision? |
|---|---|---|---|
| Does execution assistance repeat on another group? | Finish the already-started panel 06 pair, then its changed-recipe pair. | Three GPU collectors accepted. | Complete the pair; do not expand the old grid without a new question. |
| Does assistance help RL learn? | Same starting checkpoint, tasks and seeds; original versus assisted execution during one signed reward update. | Collections and two updates accepted. | Check reward variation, actual weight/probability changes and before/after success. |
| Is RL spending loss terms on fields that execution code replaces? | Independently warm-start one masked update on the exact assisted batch; evaluate through the unchanged assisted interface. | One training and one 16-attempt readout accepted after the existing tail. | A masked-minus-full gain needs random-mask or gradient-scale controls before a specific mechanism claim. |
| Is improvement in the model or only the tool? | Evaluate both trained checkpoints through both execution interfaces, with unchanged-checkpoint controls. | Six GPU readouts accepted, conditional on usable endpoints. | A learned gain must exceed each interface's own baseline. |
| Does RL improve unfamiliar goals? | Train on new group A; evaluate on disjoint new group B. Start both execution arms from the same checkpoint. | Four fresh collection/update/readout cycles per arm accepted, plus warm and additional-SFT controls. | Continue fresh updates only with informative rewards and usable numerical diagnostics; report every fixed checkpoint. |
| Does teaching information in a usable order explain the gap? | Reorder the known-recipe demonstrations so names are visible before they are queried. Preserve task/action multiset and minibatch answer targets. | Two trainings and four paired-world readouts accepted. | Improvement over matched known-recipe training supports a conditioning-history mechanism; a second legal order tests dependence on one ordering. |
| Is the weaker teacher merely undertrained? | Fixed additional training doses for both teaching methods, without selecting the best evaluation checkpoint. | Both continuations and all eight fixed checkpoint/world readouts accepted. | If longer training closes the gap, revise the teaching-mechanism claim. |
| Do unnecessary action fields impair learning? | Ask for item and quantity only; code supplies known ingredients. Compare with the current assisted full-action format. | Matched-epoch compact SFT and two paired-world readouts accepted. | Compare success and output costs first. Compact also rejects unobserved recipes, so this is not pure token removal. |
| Can more informative feedback help RL? | Reuse the first fresh batch for one independent update with a small error penalty, while keeping native success unchanged. | Two one-step trainings and two fixed diagnostic readouts accepted. | Success must improve or be preserved; merely learning to stop early and make fewer errors is not success. |
| When should a task be divided? | First check actual helper uptake with the existing globally budgeted interface; then compare fixed and public-information-based delegation. | First three screens stopped on an imposed query-order constraint before helpers; separate relaxed-order screens accepted. | The first screen is not evidence against recursion. Establish actual use before interpreting an efficacy grid. |
| Does the simpler action format make reward learning easier? | One compact-interface RL cycle on the same fresh optimization/diagnostic groups, with its own unchanged-weight control. | Four scientific stages accepted with a deferred real SFT checkpoint binding. | Compare learning gains within each interface; raw post-training scores mix model and tool effects. |
| Does the teaching result extend beyond one model? | Train Phi-4-mini on the same two teaching packages, then compare original and assisted actions in two recipe worlds. | Two fixed-endpoint trainings, two tiny base-reference cells and eight trained readouts accepted. | Replication supports two-family transfer of the original comparison, not transfer of the new teaching repair. |
| Does explicit quantity bookkeeping help deeper tasks? | Same public fact table with versus without computed remaining-demand columns; matched state-conditional prompt length. | Two eight-attempt readouts accepted, following the completed failure audit. | Better root completion, not merely fewer repeated errors; replicate on new tasks if promising. |
| Does an action's representation change what SFT learns in another environment? | In ALFWorld, teach action indices versus exact available command text, using the same demonstrations and public action lists. | One new command training and four paired readouts accepted; the existing index-trained checkpoint is reused. | Compare each trained actor with its own base-interface control; unequal target-token doses limit interpretation. |

The fresh group A/B study is still within TextCraft. Group B is an official
TRAIN subset kept out of the new optimizer, not an untouched benchmark test.
It also supplies goals for an explicitly synthetic CPU diagnostic; that is not
training data or a replacement for the predeclared model evaluation.
Recipe overlap and shared generator remain. A promising mechanism should next
be tested in a second environment with the same information/decision structure.

## Controls and stopping decisions

- Keep seeds, starting weights and caps paired. Preserve native errors and missing
  outcomes separately. Do not train on validation trajectories.
- Compare RL with an additional supervised update before claiming RL superiority.
  Matching optimizer steps does not match supervised tokens or FLOPs; record both.
- Use success at the original goal, calls, generated tokens, native errors and
  actual elapsed GPU work. A successful helper is not necessarily a useful helper.
- Measure semantic action diversity separately from alternative spellings or
  ingredient fields that execution code ignores.
- Stop an uninformative branch after its planned comparison. All-flat rewards,
  repeated numerical failure or no useful policy movement call for a changed
  experiment, not endless repetitions.
- First results are exploratory. Replicate an effect across seeds/new inputs,
  challenge the strongest alternative explanation, then reserve a fixed test.

## Resumption and artifacts

External root:
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.

Each accepted operation has immutable commands/source pins, task and model
identity, seed, cap, scientific owner, per-episode results and terminal receipts.
Every optimizer update commits its adapter, optimizer and random state. Queued
work is resumable by inspecting receipts; never blindly rerun an attempted output.
Source and concise analyses go to GitHub, weights and full traces stay external.

At 10:52 UTC six accepted queues contain 61 scientific GPU stages and 25 CPU
audits/comparisons. They cover different questions, not 61 independent hypotheses.
Some training/readout stages skip if their prerequisites fail. Expected useful
coverage is roughly a day if the learning arms remain viable, not a guarantee
that the entire three-day allocation is filled. A token-free response/completion
journal runs at `health-watch-20260928-001/`; its alerts are observations, not an
automatic scientific decision maker. Review results at each completed boundary.

At11:17 the reward-cost and six fixed training-fit measurements are also accepted.
The tiny delegation probe was promoted ahead of the original paired RL collections:
`rl-collect-with-delegation-20260928-001/` supersedes only the idle
`rl-collect-queue-20260928-001/` supervisor. No scientific owner was interrupted,
and the original RL commands, caps and downstream paths are unchanged. Inspect
`SUPERSEDED.json` before counting queues; duplicate references are not extra runs.
There are now74 unique accepted scientific output stages, many conditional.

A CPU-only crossed-results observer will write
`crossed-rl-analysis-20260928-001/FINDINGS.md` when the familiar-goal RL comparison
settles. It distinguishes changing weights from changing tools and preserves
unavailable endpoints. The main source/findings checkpoint is Git`3bbd182` on the
research branch; full model artifacts remain external.

At11:36, `compact-delegation-queue-20260928-001` is accepted behind the reward
diagnostics queue. Its seven scientific stages bring the unique accepted total
to81, not81 research questions. The new helper screen is a narrow repair with
unchanged32-response/15-minute caps; the four compact-RL stages compare learning
against the compact actor's own baseline. Independent review and actual saved
request/native replay passed. Estimated additional useful work2–4hours plus a
short screen; deadlines still respect the allocation. The active GPU is doing
the first original-action RL collection, not the queued training yet.

Read [today's running findings](FINDINGS-20260928-LIVE.md) for the completed
panel06 pairs, remaining-failure analysis and the unsuccessful first screen.

At 11:49, `transfer-queue-20260928-001` was accepted after the compact/delegation
queue. Its two quantity-table readouts and twelve Phi stages bring the total to
95 unique scientific output stages. Expected useful work is approximately three
to five hours; the sum of individual limits is not a runtime prediction. The
table comparison has four previously exposed goals, not sixteen independent
problems. Phi has eight goal identities; its four base-model attempts are only
a small descriptive reference. All training-dependent readouts require actual,
successfully committed checkpoints.

At 11:56, the original-action RL collection completed: all 32 native outcomes
were observed, including 24 successes, with 790 model responses and no transport
failures. The assisted collection started automatically and is returning real
responses. These are samples for training, not a measured RL improvement.

## Current execution order, 12:09 UTC

The [scheduling review](SCHEDULING-REVIEW-20260928.md) found that independent
comparisons would otherwise wait roughly 18–32 hours behind repeated RL cycles.
Four authenticated idle supervisors were superseded, without interrupting any
scientific owner or changing any inherited experiment command, output, cap or
original input pin. Their original receipts remain available.

The active paired collection, crossed RL readouts, teaching-order controls and
interface/dose work keep their order. Then the new
`information-first-tail-20260928-001` runs:

1. Three short helper-uptake screens.
2. Public quantity-table and Phi comparisons.
3. The ALFWorld representation-learning pilot.
4. The unchanged fresh-goal RL cycles, reward diagnostics and compact RL.

The tail preserves 84 inherited descriptors and adds six ALFWorld descriptors:
90 jobs, including 65 scientific output paths and 25 audits/comparisons. Across
the current campaign, 100 distinct scientific output stages are accepted, many
conditional. Do not double-count references in superseded receipts. About four
to seven useful hours of distinct comparisons now precede repeated fresh RL;
these are forecasts, not completed work or guaranteed durations.

Read the [publication decision memo](PUBLICATION-DECISIONS-20260928.md) for claim
boundaries and the experiments that would promote or retire each explanation.

## Numerical updates and a targeted diagnostic, 12:43 UTC

Both first RL updates completed and committed usable checkpoints. The original
action update took 940.6 seconds; the assisted-action update took 340.7 seconds.
Their sampled-token probabilities changed in the expected reward direction.
Task evaluations are now running; no learned success improvement is established.

`payload-mask-queue-20260928-001` adds only two scientific stages after the existing
tail: one independent same-batch update and one paired readout. It removes direct
loss on strictly interior ingredient-value tokens only when the public-recipe
binder replaces that field. All original generated context, rewards and replay
checks remain. The mask covers 27.27% of absolute advantage-weighted token mass,
not 27.27% of a measured gradient. This is a biased loss experiment with no
inference-token savings. It is not a correction to valid full-token RL.

The campaign now has 102 distinct accepted scientific output stages, many
conditional, not 102 questions or results. The new queue has 105 minutes of
scientific caps; expected work is approximately 30–65 minutes. Its original
full-token and unchanged-weight controls are reused, not rerun. See the
[protocol](rl_payload_mask_20260928/README.md) and
[current findings](FINDINGS-20260928-LIVE.md).
