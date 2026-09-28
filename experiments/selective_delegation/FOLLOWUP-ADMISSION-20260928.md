---
date: 2026-09-28
accepted_utc: "2026-09-28T16:01:27Z"
status: accepted_waiting_for_active_dose_readouts
claim_level: exploratory
queue: information-first-tail-20260928-002
receipt_sha256: f3944b75c55aab0611cdd861d6940384c30ca265db0b9ca3807a7abe13bf2f83
new_scientific_stages: 14
scientific_owners_interrupted: 0
---

# Follow the two positive signals without dropping independent tests

Two completed comparisons changed the next decision. Repairing teaching histories
raised success from 1/32 to 14/32 or 12/32, with exact supervised minibatch targets
preserved. One RL update raised ordinary-interface success from 9/16 to 13/16 on
familiar goals. Neither result yet establishes reliable transfer. See the
[findings](FINDINGS-20260928-LIVE.md) and
[publication boundaries](PUBLICATION-DECISIONS-20260928.md).

## What is newly queued

1. **Does the teaching repair repeat?** Two second-seed trainings and eight
   readouts test both repaired orders across another fit seed and eight additional
   goal roots. Both recipes and endpoints are fixed; there is no best-arm selection.
   Success supports reproducibility of this repair package. A mixed result narrows
   the claim. The active longer-training controls separately test undertraining.
2. **Do the existing RL gains carry to new goals?** Four readouts cross the two
   already-trained actors with ordinary and assisted execution on frozen diagnostic
   B. The two unchanged-weight controls are moved earlier unchanged, not rerun.
   No new optimizer is involved, so this is distinct from the existing fresh-A RL
   campaign. B has been exposed diagnostically; it is not an untouched final test.

Expected additional useful work is roughly four to seven GPU hours, not the sum
of all worst-case caps. Each stage retains its checkpoint/episode artifacts and
deadline. The one-card allocation ends October 1 at 09:50:35 UTC. The full tail's
106.43-hour sum of individual limits exceeds remaining allocation and is **not**
an ETA or a promise that every stage will execute. The executor enforces the
remaining-allocation boundary and skips unavailable scientific prerequisites.

## Actual queue change

The active interface/dose queue and its scientific owner are untouched. Its
successor is now `information-first-tail-20260928-002`, supervisor PID1147467.
The order is:

1. Existing helper-uptake, quantity-table, Phi and ALFWorld comparisons.
2. New teaching replication.
3. Existing independent warm-B controls, then new fixed-checkpoint RL transfer.
4. Existing fresh-RL cycles, reward/fit diagnostics, compact RL and payload mask.

All 94 inherited job descriptors retain their original commands, output paths,
caps and original pins. Extra queue-provenance pins were added. The first 41 jobs
remain in exactly the old order; only the two independent warm controls move.
There are 120 tail descriptors: 81 scientific outputs and 39 CPU audits/reports.
Across this campaign, 116 distinct scientific output stages have been accepted,
including completed and conditional work. These are not 116 hypotheses or results.

Only authenticated idle waiters PID1105638 and PID1115763 were superseded. Their
original receipts remain unchanged and have additive `SUPERSEDED.json` records.
No training, evaluation, checkpoint or source of a running owner was interrupted
or edited. Do not restart those waiters.

## Verification and resume pointers

`R=/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.

- New queue: `R/information-first-tail-20260928-002/{ACCEPTED,LAUNCH}.json`.
- Teaching: `R/textcraft-teaching-replication-20260928-001/PREPARED-JOBS.json`,
  SHA256 `eb0214d13a25304919f1f5a9c3adfba59c79af38e87af1df58c3f0de8ee403ab`.
- Transfer: `R/textcraft-rl-transfer-20260928-001/PREPARED-JOBS.json`,
  SHA256 `a484f4695be7b3c03f130db2e55e3e3703852fcc81ace8793bb022352ecaf923`.
- Both preparations have real saved-request/native-replay fixtures. A bounded
  independent review checked actual endpoints, baseline/task pairing, warm-control
  independence and exact per-minibatch teaching targets. Twenty-three focused
  tests, scoped Ruff/format and diff checks passed; no broad-suite claim is made.
- Live admission reauthenticated both idle processes before and after suspension,
  checked the predecessor was still active, and verified 296 unique input pins.
  The new waiting process and invocation receipt were observed alive after launch.
- The active dose readout returned 97 real model calls with zero transport failures
  at 15:59 UTC. Both training continuations completed fixed steps46/69 beforehand.

The health journal checks progress every15 seconds; the existing completed-controls
reader checks every60 seconds. They keep writing local evidence without a Codex
turn. They are not an open-ended scientific decision maker or a scheduled Codex
wakeup. This queue executes accepted comparisons automatically; further adaptive
choices require reviewing their results.

The previously delivered advisor deck retains its historical evidence cutoff.
These two new signals should inform the next deck, with familiar-goal RL versus
new-goal transfer explicit. No completed replication or new-domain claim is added
to the old presentation. GitHub source checkpoints do not back up model weights
or full external run artifacts.
