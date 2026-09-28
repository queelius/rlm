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

| Question | Small comparison | Status at acceptance | What changes the next decision? |
|---|---|---|---|
| Does execution assistance repeat on another group? | Finish the already-started panel 06 pair, then its changed-recipe pair. | Three GPU collectors accepted. | Complete the pair; do not expand the old grid without a new question. |
| Does assistance help RL learn? | Same starting checkpoint, tasks and seeds; original versus assisted execution during one signed reward update. | Collections and two updates accepted. | Check reward variation, actual weight/probability changes and before/after success. |
| Is improvement in the model or only the tool? | Evaluate both trained checkpoints through both execution interfaces, with unchanged-checkpoint controls. | Six GPU readouts accepted, conditional on usable endpoints. | A learned gain must exceed each interface's own baseline. |
| Does RL improve unfamiliar goals? | Train on new group A; evaluate on disjoint new group B. Start both execution arms from the same checkpoint. | Four fresh collection/update/readout cycles per arm accepted, plus warm and additional-SFT controls. | Continue fresh updates only with informative rewards and usable numerical diagnostics; report every fixed checkpoint. |
| Does teaching information in a usable order explain the gap? | Reorder the known-recipe demonstrations so names are visible before they are queried. Preserve task/action multiset and minibatch answer targets. | Two trainings and four paired-world readouts accepted. | Improvement over matched known-recipe training supports a conditioning-history mechanism; a second legal order tests dependence on one ordering. |
| Is the weaker teacher merely undertrained? | Fixed additional training doses for both teaching methods, without selecting the best evaluation checkpoint. | Both continuations and all eight fixed checkpoint/world readouts accepted. | If longer training closes the gap, revise the teaching-mechanism claim. |
| Do unnecessary action fields impair learning? | Ask for item and quantity only; code supplies known ingredients. Compare with the current assisted full-action format. | Matched-epoch compact SFT and two paired-world readouts accepted. | Compare success and output costs first. Compact also rejects unobserved recipes, so this is not pure token removal. |
| Can more informative feedback help RL? | Reuse the first fresh batch for one independent update with a small error penalty, while keeping native success unchanged. | Two one-step trainings and two fixed diagnostic readouts accepted. | Success must improve or be preserved; merely learning to stop early and make fewer errors is not success. |
| When should a task be divided? | First check actual helper uptake with the existing globally budgeted interface; then compare fixed and public-information-based delegation. | First three screens stopped on an imposed query-order constraint before helpers; separate relaxed-order screens accepted. | The first screen is not evidence against recursion. Establish actual use before interpreting an efficacy grid. |
| Does the simpler action format make reward learning easier? | One compact-interface RL cycle on the same fresh optimization/diagnostic groups, with its own unchanged-weight control. | Four scientific stages accepted with a deferred real SFT checkpoint binding. | Compare learning gains within each interface; raw post-training scores mix model and tool effects. |
| Does the teaching result extend beyond one model? | Qualify another public3–4B model family for the same paired teacher study. | CPU model/template qualification, not yet a GPU run. | Stop on genuine incompatibility; choose a fixed endpoint and preserve all outcomes. |
| Does explicit quantity bookkeeping help deeper tasks? | Same public fact table with versus without computed remaining-demand columns; matched state-conditional prompt length. | CPU preparation, following a complete failure audit. | Better root completion, not merely fewer repeated errors; replicate on new tasks if promising. |
| Does an action's representation change what SFT learns in another environment? | In ALFWorld, teach action indices versus exact available command text, using the same demonstrations and public action lists. | CPU qualification/design. | Compare each trained actor with its own base-interface control; no automatic repair or hierarchy claim. |

The fresh group A/B study is still within TextCraft. Group B is an unused official
TRAIN subset held out from the new optimizer, not an untouched benchmark test.
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
