---
date: 2026-10-05
status: admitted_owner_waiting_for_supplemental_batch
question: Does the fixed100k walking effect repeat across new training seeds?
training_seeds: [234, 567, 890]
primary_endpoint: 100000_training_simulator_steps
external_campaign: /project/alex_phd/runs/curl-walker-additional-seeds-20261005
---

# Does the result change when we train a different set of models?

The original walking study produced two wins for CURL and one large win for
the control. Their average scores were almost tied. Three pairs cannot tell
us whether there is a reliable benefit or how frequently each method wins.

We will train three more matched pairs, using new random seeds 234, 567 and
890. The learner, task, random crops, training budget and evaluation settings
are unchanged. Each model starts from fresh weights. Both methods are run for
every seed; we will finish the whole cohort rather than stop after a favorable
or unfavorable pair.

## What is measured?

At exactly 100,000 training simulator steps, evaluate the final policy on the
same ten starts, 10000--10009, as the original study. A score is the mean sum of
rewards over those ten episodes. Report each matched training-seed difference,
then the new cohort's average. Show the original cohort separately. A combined
six-pair summary can be descriptive, but this follow-up was adaptively motivated
by earlier results, not a new untouched confirmatory study.

The [separate fresh-start test](WALKER_FRESH_STARTS.md) does something different:
it tests the six existing models from more starting states without training.
It does not create additional independent training runs. Its outcomes are not
used to select the seeds, settings, checkpoints or schedule for this follow-up.

## What would change our view?

A repeated advantage across the new pairs would strengthen the case for an
effect in this setting and motivate testing its persistence or mechanism.
Continued mixed outcomes would weaken a claim of dependable benefit and make
variation across training runs central to the report. Consistent control wins
would revise the earlier positive impression. None alone establishes a universal
rule across tasks or reproduces the paper's full benchmark.

## Budget, reproducibility and current status

The six jobs run sequentially on one A100. Each uses 50,000 decisions,
100,000 training simulator steps, 49,000 updates and 260,000 separate evaluation
steps. The expected total is 7--9 hours, with a two-hour cap per job. Full-state
checkpoints target 15 minutes at natural episode boundaries and final completion.
Failed or interrupted attempts remain visible; no favorable checkpoint selection.

Executable code is byte-for-byte the original verified walking snapshot. Only
campaign metadata and planned training seeds changed. The external root stores
QUEUE.json, SOURCE.json, ADMISSION.md and LAUNCH.json. The last file distinguishes
a waiting owner from actual training. At the October 5, 08:07 UTC cutoff, its
owner is waiting on the shared GPU lock held by the entire supplemental batch.
It will begin after that lock is released. No new training result exists yet.

The decision follows the complete original cohort, before any complete
supplemental policy comparison existed. Original evidence, checkpoints and
all live source snapshots are preserved. This remains a learning reproduction
and exploratory extension, not a novelty claim.
