# Varied-goal RL: fixed diagnostic results

## Both first-update task comparisons

Evidence cutoff: **September 29, 08:37:19 UTC**. Both RL task comparisons are
complete. The extra-SFT comparison remains pending. Neither first update
improved success on this fixed different-goal panel.

| Execution setting | Success before → after | Paired gains / losses | Calls before → after | Output tokens before → after |
| --- | ---: | ---: | ---: | ---: |
| Model writes ingredient arguments | 4 → 4 / 16 | 0 / 0 | 778 → 825 | 23,516 → 25,919 |
| Code fills observed ingredient arguments | 5 → 4 / 16 | 0 / 1 | 673 → 716 | 21,759 → 24,983 |

The assisted result adds 32 checked before/after outcomes, of which sixteen
are the reused warm baseline, not new runs. Its calls rise 6.4% and output
tokens 14.8%; summed service time rises 15.1%. The only lost success is
TRAIN1796, seed 202609280900: 17 calls and 448 tokens before training versus
23 calls and 664 tokens afterward. It explicitly finishes without the requested goal.
The other fifteen paired task scores are unchanged. Both updated models now
solve exactly the four attempts on TRAIN256 and TRAIN1847.

For the assisted model, native action errors rise from 241 to 261, invalid
action schemas from 3 to 22, and context-limit stops from 1 to 3. Transport failures and unknown task
outcomes remain zero. Error totals alone do not explain the lost success.
A separate CPU review checked the actual rejected returns and native histories:

- TRAIN964, repeat 1, accounts for 21 new invalid actions, versus zero before.
  Fifteen are EOS-terminated objects with duplicate ingredient keys; six are
  repetitive, unterminated objects reaching the 256-token response cap. They
  consume 2,511 of that attempt's 3,941 output tokens. It fails in both conditions,
  now at the context limit. This is not the lost successful attempt.
- One extra-bracket error on TRAIN672, repeat 0, remains unchanged. Two duplicate-
  key errors on TRAIN1796, repeat 1, disappear, giving the net increase from 3 to 22.
- The sole lost success, TRAIN1796, repeat 0, has **zero schema errors in either
  condition**. The updated model crafts two of the three requested items and
  explicitly finishes. Warm crafts four and finishes successfully. Quantity
  completion/termination, not a formatting rejection, explains this failure.

The pinned strict parser reproduces all 25 rejections across both readouts; each
has a distinct saved call, with no observed accounting/parser discrepancy.
An independent check verified the lost attempt's target/final inventory and final
actions, plus representative duplicate-key and 256-token capped returns. The
formatting concentration plausibly adds cost and context pressure, but cannot
be credited with the lost success. No training change follows from this B-only
diagnosis. Details and exact paths are in the
[research review](../research_review_20260929/reviews/2026-09-29-0838.md).

This is a narrow result from one realized update per interface. It does not
establish general harm from RL or assistance. Both training likelihood updates
were real, but neither supplied a task-success benefit on this panel. The
same exposed B roots/seeds and shared recipe world limit both comparisons.
Keep all prospectively fixed endpoints and pending controls; no B-driven tuning
or checkpoint selection. The four-step sequence is not complete.

[step-0001-binder.json](step-0001-binder.json) preserves its paired outcomes,
costs, endpoint identity and source hashes. The same 96 small receipt/contract
checks used for the raw report were performed. The advisor guide receives this
later result; the audience PDF still waits for the extra-SFT control before a
possible compact update to its RL comparison.

## Earlier raw-only snapshot, 08:02 UTC

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
