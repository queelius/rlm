# Varied-goal RL: fixed diagnostic results

## First raw-interface update

Evidence cutoff: **September 29, 08:02:52 UTC**. Both cells are complete and
natively checked. The model received one RL update on eight A training goals,
then used the unchanged raw action interface on eight different B goals, with
two fixed attempts per goal. B is diagnostic only, not optimization data.

**Successful attempts stayed at 4/16, with exactly the same four successes.**
There are zero paired gains, zero losses, four shared successes and twelve
shared failures. The model used more calls and tokens after training.

| Measurement | Starting model | After one RL update |
| --- | ---: | ---: |
| Successful attempts | 4 / 16 | 4 / 16 |
| Model calls | 778 | 825 |
| Generated tokens | 23,516 | 25,919 |
| Native action errors | 348 | 359 |
| Invalid action schemas | 1 | 3 |
| Attempts stopped by the context limit | 2 | 3 |
| Transport failures / unknown outcomes | 0 / 0 | 0 / 0 |

Calls rose 6.0%, generated tokens 10.2%, and summed model-service time 10.4%.
The last measure is a request-time sum, not allocated GPU wall time. The
observed pair offers no task-success or inference-cost benefit. It does not
establish a general harmful effect of RL or statistical equivalence.

The two consistently successful goals, TRAIN256 and TRAIN1847, still succeed
on both seeds and use the same call/token counts. All six other goals still
fail twice, although their trajectories and costs change. For example, one
TRAIN672 attempt increases from 23 to 71 calls without reaching its goal.
Equal final scores therefore do not mean an unchanged policy. A model update
and intended average training-likelihood changes were already verified in the
[optimizer review](../research_review_20260929/reviews/2026-09-29-0711.md).

## What this changes, and what it does not

The first update has **not shown transfer on this fixed panel**. One update,
eight roots and two seeds do not settle whether longer training, different
experience or another objective would work. There is no independent training
replication here. No confidence interval from resampling sixteen identical
paired outcomes should be read as proof of no effect.

These are the **same B roots and seeds, with the reused warm baseline**, as the
[earlier familiar-RL transfer study](../rl_transfer_results_20260929/README.md).
The new checkpoint was trained on different A roots. This is a new training
comparison, not a second independent evaluation dataset. B is drawn from the
official TRAIN split, excludes A/old-SFT target roots, and shares the same
recipe world; it is not official held-out or broad out-of-domain confirmation.

The assisted first-update readout and extra-SFT controls are pending at this
cutoff. Do not label the two-interface study complete. Continue the already
fixed sequence and report every numbered endpoint. B outcomes must not choose
training data, learning rates, continuation gates or favorable checkpoints.
Future recovery-teaching/prefix-start proposals remain grounded in the earlier
A-side reward-contrast diagnosis, not selected from this B result.

## Evidence and advisor decision

[step-0001-raw.json](step-0001-raw.json) records all sixteen paired outcomes,
costs, endpoint identities, source hashes and limits. Verified the common
task/seed list, model/dataset/runtime/cap contracts, diagnostic manifest, and
the exact committed RL endpoint. Checked 96 small bindings: the episode,
root-node and first returned call for each of the 32 attempts. Native replay
scores agree with saved episode/node scores. Reused completed native audits;
no additional GPU replay or model-tensor hashing.

The advisor guide gets a clearly labeled later-result paragraph. Keep the
audience PDF's existing evidence cutoff and story until the pending controls
form a useful compact comparison. This result reinforces the existing limited-
transfer qualification; it is not a new positive headline or another slide.
