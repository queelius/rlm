---
question_id: textcraft-terminal-reward-one-step
status: exploratory_complete
evidence_cutoff_utc: 2026-09-22T17:04:38Z
claim: no_demonstrated_one_step_rl_advantage
next_decision: repair_continuation_interface_and_complete_teacher_seed_replication
---

# One RL update changes behavior but does not improve total success

The completed comparison does **not** show a useful RL gain. The starting model
and the model after one reward-based update each solve 15 of 32 attempts.
An additional supervised update solves 17. The differences are uncertain; this
is not evidence that supervised learning reliably beats RL either.

| Model | Successful attempts | Model calls | Rejected environment actions | Malformed actions |
|---|---:|---:|---:|---:|
| Starting model, already trained on revised demonstrations | 15/32 | 1,206 | 550 | 2 |
| After one RL update | 15/32 | 1,318 | 711 | 26 |
| After one matched supervised update | 17/32 | 1,155 | 563 | 1 |

There are 16 goals, each attempted with two fixed sampling seeds—not 32
independent tasks. Every outcome was observed and checked against the native
environment rules. There were no transport failures, missing outcomes,
unresolved call records or calls unassigned to an episode. Rejected environment
actions and malformed actions are behavior errors, not infrastructure failures.
Their counts are descriptive and also depend on how long trajectories run.

## What changed, and how uncertain is it?

Relative to the starting model, RL gains two successful attempts and loses two.
The supervised update gains four and loses two. Comparing RL directly with the
supervised update gives one win and three losses. The 95% intervals, obtained
by resampling goals while keeping both attempts together, are:

- RL minus starting model: **0 percentage points**, interval **−12.5 to +12.5**.
- Supervised update minus starting model: **+6.25 points**, **−6.25 to +18.75**.
- RL minus supervised update: **−6.25 points**, **−18.75 to +6.25**.

All include zero. RL also uses more calls and output tokens in this comparison:
42,901 output tokens versus 37,486 before the update and 37,203 for the supervised
control. There is no measured accuracy or efficiency gain here.

## What was matched?

Both continuations start from the same demonstration-trained checkpoint and
make one actual optimizer update, at learning rate 0.00002. The supervised
control processes 501 whole demonstration rows, cycling through the frozen
86-row subset of the same eight training tasks. Its 12,074 target tokens exactly
match the RL update's number of nonzero-credit output tokens, with no overshoot.
This matches update count and this token dose—not histories, information,
objective, sampling temperature or total computation.

RL assigns credit based on whether complete training attempts succeeded. It
does not separately label each intermediate action as helpful. Thus, an action
inside an eventually successful attempt may receive credit even when that action
itself was rejected. Earlier saved-gradient analysis identifies this as a
possible weakness, not an established cause of the new-goal result.

## Limitations and next decisions

This is an explicitly amended evaluation of the sole committed update from a
run that subsequently failed a numerical check. The original run remains failed;
we did not select a best checkpoint or relabel it as completed two-step RL.
The evaluation panel is reused, shares the training recipe world, and uses no
helper agents. It is not evidence about the benefit of recursion or broad
benchmark transfer. One small update cannot settle whether RL can work.

The result rules out a large observed benefit from this particular update on
this panel. It does not distinguish insufficient update size from unhelpful
credit assignment. A numerical-repair continuation remains the next accepted
check. Its first attempt passed the finite-gradient probe but then failed on a
missing validation function, before any new rollouts or update. That failure
requires an interface repair, not a new learning hypothesis or looser tolerance.
The independent second-seed teacher replication proceeds while this is repaired.

If the repaired continuation completes without a gain, a same-batch update-size
comparison would be more informative than collecting the same kind of rollouts
again. If it improves, replication and a supervised control starting from that
same intermediate checkpoint are still needed before claiming an RL advantage.
These are conditional follow-ups, not additional accepted GPU jobs.

## Evidence and reporting

The compact [machine-readable summary](textcraft-one-step-results.json) is
extracted from the native three-arm report. Full report:
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/analysis-textcraft-stopped-step1-001.json`.
SHA256: `dfe3c3db7d64be4accd5c5b70479ba78b1fb52d639e324a06832bf3143c20ad9`.

See the [prospective one-step amendment](TEXTCRAFT-STOPPED-STEP1-READOUT.md)
and [numerical continuation plan](TEXTCRAFT-FP16-CONTINUATION-PLAN.md).
The historical advisor deck retains its original cutoff. This null comparison
belongs in the current research synthesis and would qualify any future RL claim;
it does not replace the stronger demonstration-training finding as the main story.
