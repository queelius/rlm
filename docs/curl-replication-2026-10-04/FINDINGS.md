---
date: 2026-10-04
cutoff_utc: 2026-10-04T17:28:00Z
stage: exploratory_compatibility_reproduction
question: Does contrastive learning improve pixel-based control beyond random-crop augmentation?
primary_endpoint: 100000_training_simulator_steps
external_runs: /project/alex_phd/runs/curl-replication-20261004
---

# What we have learned so far

The first full reference learned better control. The matched control and
repetitions are still running, so the comparison is not complete.

## First 100k reference: meaningful learning in one seed

Fresh run `curl-seed123-100k-v1` completed exactly 12,500 decisions and 100,000
training simulator steps, with 11,500 update calls. Its fixed ten-episode mean
return was 8.4446 before updates and 678.0208 at the endpoint. This is reward
on a roughly 0–1,000 scale, not a success percentage. Native records and config
are published in [data/first-reference](data/first-reference).

The trajectory fluctuates substantially: the 96k score was 423.07, followed by
678.02 at 100k. The endpoint was prespecified, not picked after looking at scores,
but this last-point jump makes repeatability and later behavior important. Do
not infer a stable 678-level policy or an advantage over the control yet.

The paper's 582 ± 146 is a mean and SD over ten trained seeds, not a threshold
that one local result can pass. Our runtime differs from the historical stack.
The result supports continuing the reference/control comparison; it is not a
claim that we reproduced the entire paper or discovered a new method.

The loop took 866.44 seconds, including evaluation; the final 2,540,402,075-byte
checkpoint took 11.43 seconds more. Setup and startup are excluded. Evaluation
consumed 260,000 separate simulator steps, not training data. The control
started automatically within about one second of reference termination and
produced scientific records within 20 seconds of launch.

Next decision: finish the already-queued three-seed pair before choosing another
arm. The [follow-up memo](NEXT_COMPARISONS.md) explains why a matched extension
to 500k and a second task are more informative than declaring victory from one
score. A shuffled-positive control is conditional, not launched.

## First pilot: execution works

Run `pilot-123-v1` used 1,200 decisions, with the first 1,000 collecting random
actions. It performed 200 subsequent learning updates. The 9,600 training
simulator steps and 6,000 evaluation steps are counted separately. The timed
training/evaluation loop took 29.90 seconds; final checkpoint writing took
another 1.37 seconds. Startup and environment installation are not included.

The deterministic two-episode mean return went from 8.40 before training to
22.58 at the endpoint. This short check is **not** evidence of a reliable
learning improvement, and is excluded from the main comparison. The contrastive
loss was near log(128) at the first updates, consistent with initially weak
image matching. Neither that observation nor a finite loss proves good control.

The final checkpoint was 366,475,547 bytes and includes optimizers, temperature,
replay data and random states. A separate CPU test using the actual upstream
pixel agent reproduced its next fixed-batch update exactly after restoration,
including tied parameter identities. That is narrower than exact GPU trajectory
reproduction. The pilot ended partway through an episode; its checkpoint labels
that truncation, and a later resume would start a fresh episode.

## What could still explain a poor full run?

The modern simulator/renderer differs from the historical stack. The task may
also need more experience than this endpoint, and one random training seed can
be misleading. We preserve the authors' learning code, including its two
overlapping encoder optimizers, rather than silently changing the method.

The matched control retains random crops and all SAC updates, but omits CURL's
extra update. Therefore a difference would measure the contribution of that
whole extra update, including extra optimization and computing time. It would
not isolate the mathematical objective from the number of encoder updates.

## Next decisions

1. Finish fresh CURL and no-contrastive runs for seed 123 at 100k steps.
2. Complete both arms for seeds 456 and 789 regardless of the first pair's sign.
3. If both are valid, consider a 500k extension of both arms. Do not select only
   a favorable seed or checkpoint. If learning fails, inspect actual observations
   and learning signals before expanding the budget.
4. Independent contrastive calculation check passed: NumPy's direct calculation
   agreed with the authors' Torch logits, loss, feature/weight gradients and one
   SGD matrix update to within 5.56e-17 absolute error. Correct matches gave loss
   0.29334; deliberately mismatched pairs gave 2.06667. These are invented small
   inputs, not learned features or RL evidence. The actual training uses Adam.

No novelty or publication claim is made. The immediate output is a reproducible,
understandable learning example that improves our experimental practice.

## Operations and honest cost accounting

Execution access was restored around 16:30 UTC. The first real pilot started
around 17:11 UTC after environment installation, adapter implementation and
focused checks. This preparation interval was not GPU training. An initial
background shell launch disappeared before producing records; a persistent
execution session then ran successfully. That brief failed launch consumed no
scientific budget and is retained in this account. Future owners must verify
actual episode/update records, not assume a returned PID means work is running.

The advisor deck is not changed by this first reproduction run. The user changed the priority to
paper reproduction and requested a learning document; that document is the
appropriate place for these execution findings.
