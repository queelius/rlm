# Unattended TextCraft breadth campaign

## Research question

Does letting code fill in already-known recipe ingredients help on additional
goals, and does it change which teaching method works best?

This is a fixed, exploratory evaluation campaign using existing trained models,
not 60 hours of new model training. There are 256 GPU arms (4,096 planned attempts)
available; only the ordered prefix that fits the allocation will run. Each arm
saves individual attempts. Missing or interrupted outcomes must stay separate
from failures. CPU audits and model loading mean GPU utilization will not be 100%.
No Codex/API calls are required to keep this script running.

The order is fixed, not an autonomous scientific decision-maker. Later analysis
must decide which effects merit replication or a different experiment. A short
comparison may straddle the final cutoff; report it as incomplete rather than
silently substituting another attempt. Dependent audits can fail after a failed
collector; independent later collectors may still proceed. An unresolved GPU
owner stops the queue rather than allowing overlapping jobs.

## Running and resuming

Status below was checked on September 24 at 18:17 UTC. PIDs are historical
identifiers, not a claim that they will still be alive when this file is read.
`R` below means `/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.
Read `R/textcraft-breadth-campaign-002/CAMPAIGN-INVOCATION.json` and
`R/independent-training-queue-029/` for campaign/individual job records.
Do not relaunch this queue over existing outputs. To resume after interruption,
authenticate that its owners are gone and build a new receipt for unfinished
comparisons, preserving the original attempts and counting their compute.

The checked-in Python files are source backups; their frozen dependencies and
absolute input paths remain in the external research store. A GitHub push does
not back up model weights or raw experiment results.

Verification: all 150 unique queued pins matched; all 640 job names were unique.
Two focused native/checkpoint tests passed under each of the raw and assisted
import packages (four checks total). Ruff passed for the added source files.
The current older batch was returning real responses without transport failures;
the new breadth campaign has not started its first GPU arm yet.

The live fixed-UTC supervisor is PID `735056`, with generic queue child PID `735058`.
It waits for the World49 queue’s authenticated final public owner, then stops admitting work at
2026-09-27 06:00 UTC (or the earlier actual allocation margin).

The immutable merged receipt is
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/TEXTCRAFT-BREADTH-CAMPAIGN-002.json`
(SHA-256 `342dc7041263bd96bab0c32d4fe46bf5a8be50d5d76e2be1c05e11999e9d43f1`).
It interleaves complete five-job raw/binder/audit/comparison blocks for the public and
quantity-corrected teacher packages.  Its source matrices are the frozen public and corrected
breadth receipts named in `source_receipts`; every job retains its own source, input, model and
checkpoint pins.

Panels are label-blind constructed TextCraft goals: eight disjoint eight-goal panels over worlds
42/50/51/52, fixed actor seeds 2026092208 and 2026092291, and depth mix 2/2/1/3 for depths 2/3/4/5.
Each collector arm has 16 episodes, a 45-minute owner cap, and native audit plus paired comparison.
This tests execution binding across the two teacher packages; it is not a claim of generic memory,
recursion, or fresh natural-language generalization.

The supervisor source is
`R/source-fixed-deadline-campaign-001/run_fixed_deadline_campaign.py`.  It delegates pin and owner
lifecycle enforcement to the existing generic runner; it bounds the UTC deadline, records a first
native-response check, and only sends SIGTERM after verifying an owner PID/create-time identity
when a first request has had sustained all-failed native calls.
