---
date: 2026-10-04
cutoff_utc: 2026-10-04T18:37:00Z
stage: exploratory_compatibility_reproduction
question: Does contrastive learning improve pixel-based control beyond random-crop augmentation?
primary_endpoint: 100000_training_simulator_steps
external_runs: /project/alex_phd/runs/curl-replication-20261004
---

# What we have learned so far

Both versions learned in all three completed pairs. CURL finished higher in
each at the prespecified endpoint. All six saved models are now queued for
continued training to 500k steps; the first continuation is running.

## All three original pairs are complete

| Training seed | CURL | Same crops, no image matching | Paired difference |
|---|---:|---:|---:|
| 123 | 678.02 | 454.47 | +223.55 |
| 456 | 446.17 | 240.54 | +205.63 |
| 789 | 587.99 | 463.26 | +124.74 |
| Mean across three training seeds | 570.73 | 386.09 | +184.64 |

The standard deviations across trained seeds are 116.89 for CURL and 126.13
for the control. They describe variation between training runs, not uncertainty
from treating the ten evaluation episodes as independent trained models. The
[summary](data/three-pairs-summary.json) and all six native run directories
provide the numbers behind the figure. No completed run was excluded.

**What changed:** the third, prespecified comparison also favors CURL, though
its advantage is smaller. The direction repeated across all three training
seeds. This supports a local benefit from the additional image-matching update
at 100k interactions. It does not demonstrate a consistently higher curve,
general usefulness across tasks, or a novel method. Our mean is near the paper's
published 582, but agreement on one task with three seeds and a modern simulator
is not a reproduction of the paper's ten-seed benchmark.

**Competing explanations and limits:** the extra update changes both the
learning objective and optimization work. We have not isolated correct image
correspondence from all other effects of that update. Learning curves fluctuate,
and longer training may let the control catch up. The same ten evaluation
starts were reused throughout: this makes comparisons stable, but does not
establish performance on a broad range of new starting conditions.

**Native checks:** all six runs have exact terminal counters of 12,500 decisions,
100,000 training simulator steps and 11,500 update calls; all ten terminal
evaluation seeds are present. Logged updates are finite, no failure events
appear, and final checkpoints match their byte-count receipts at natural episode
boundaries. Each run also used 260,000 evaluation simulator steps, kept out of
training. The third control's loop took 792.07 seconds and its checkpoint write
took 11.48 seconds; the third CURL loop took 853.63 seconds plus 11.62 seconds
for its checkpoint.

**Decision:** continue all six models to the previously proposed 500k endpoint.
The owner launched at 18:36:58 UTC after every original result was checked.
Each continuation retains its original seed, replay, optimizers and counters;
each parent/child chain is one replicate. Original checkpoints remain intact.
The question is whether the early advantage lasts, shrinks or reverses. These
are documented resumed runs, not a claim of bit-for-bit identity to uninterrupted
GPU training. There are no completed 500k scores at this cutoff.

**Publication/teaching impact:** the eight-page guide now shows all three pairs,
the precise score definition and the longer-training question. This is useful
replication evidence and a worked learning example, not a publication novelty
claim. Finish the horizon comparison before choosing the next mechanism probe
or a second task.

## Earlier checkpoint at 18:15: two matched pairs

The second matched control completed with a score of **240.54**, versus
**446.17** for CURL, a difference of **205.63** reward points. The first pair's
difference was **223.55**. All four runs completed the same 100,000 training
simulator steps and 11,500 update calls. Each score averages ten fixed evaluation
episodes. There are two independently trained pairs, not twenty repetitions.

| Training seed | CURL | Same crops, no image matching | Paired difference |
|---|---:|---:|---:|
| 123 | 678.02 | 454.47 | +223.55 |
| 456 | 446.17 | 240.54 | +205.63 |

This strengthens the initial signal: the higher CURL endpoint was not confined
to the first training seed. It remains a small exploratory comparison on one
task and one training budget. We have not selected favorable checkpoints or
dropped unfavorable runs. The difference measures the whole additional update,
not just correct image correspondence independently of extra optimization.

The curves give a more qualified picture than the endpoints alone. The first
pair often traded places. CURL led for more of the second run. As a post-hoc
descriptive check, the second pair's interpolated curve averages are 272.08
for CURL and 187.46 for the control, compared with nearly equal averages in
the first pair. These are not additional primary outcomes or independent
repetitions. We retain the prespecified endpoint as the main comparison.

The second control had no failure records or nonfinite logged update metrics.
Its training/evaluation loop took 785.78 seconds and its final 2.506 GB checkpoint
took another 12.16 seconds. Evaluation consumed 260,000 separate simulator steps,
not training data. Its [native records](data/second-control) are published.

**Decision:** finish the third pair, then extend all three pairs to 500,000
steps if their final records and checkpoints are valid. This tests whether the
early difference lasts, shrinks or reverses. Prepare the queue changes while
the last pair runs, preserving all original checkpoints and analysis. The PDF
now shows both completed pairs and explicitly says that the evidence is limited.

## Earlier checkpoint: the second reference before its control finished

With a new training seed, CURL improved from **18.62** to **446.17** at the
same fixed 100,000-step endpoint. Its first training seed finished at 678.02.
This is direct evidence that the exact result depends on the training run,
even when the task and settings are unchanged. It is not a failure of the
second run: it completed 11,500 updates, all recorded update metrics were finite,
and it saved its full checkpoint. The last three scores were 399.57, 492.46
and 446.17, so this run did not repeat the first run's final upward jump.

The matched control for this new seed is still training. **We cannot yet say
whether CURL's advantage repeated.** Comparing two completed CURL runs against
just one control would change the comparison as results arrive. We will keep
the matching by training seed and report unfinished runs separately.

Native evidence is in [data/second-reference](data/second-reference). There are
ten evaluation episodes per score, not ten independent training runs. The loop
took 851.57 seconds, excluding startup and the final checkpoint write.

**Decision:** leave the queued comparisons unchanged. Prepare the longer-training
comparison on CPUs while these finish. The PDF retains its explicitly dated
first-pair snapshot; this individual, as-yet-unpaired result reinforces its
existing caution rather than changing the main conclusion. Update the PDF when
the next complete pair changes what can be concluded.

## First matched pair: a higher endpoint, not a consistent lead

At 100,000 training simulator steps, CURL scored **678.02** and the control
without image matching scored **454.47**. Both started at **8.44**. These are
means of ten fixed evaluation episodes from **one training seed per version**,
not ten independently trained models. The paired endpoint difference is +223.55.
Both runs completed 11,500 update calls and saved their full checkpoints.
The control's native records are in [data/first-control](data/first-control).

The learning curves trade places. At 76k steps the control scored 354.84 while
CURL scored 222.24. CURL's final jump matters to its endpoint advantage. Because
evaluation uses the same starting seeds and deterministic actions, these curves
are not merely noisy because we drew different evaluation starts each time.
The policies themselves change during training.

As a **post-hoc descriptive check**, linearly interpolating and averaging each
recorded learning curve over 0–100k steps gives 249.29 for CURL and 250.30 for
the control. This was not a prespecified primary metric and does not replace
the endpoint. It cautions against claiming a consistent sample-efficiency gain
from the last point alone. Checkpoints are not independent replicates.

CURL's training/evaluation loop took 866.44 seconds; the control took 782.83.
Their evaluation portions were 371.88 and 362.31 seconds. This is an equal
training-interaction comparison, not equal computing time. Startup and final
checkpoint writes, about 11 seconds each, are excluded from those loop times.

**Decision:** finish both arms for the two remaining prespecified training seeds.
If those runs are valid, a paired extension to 500k can test whether the apparent
late advantage persists. Do not launch a new loss variant merely because one
endpoint is favorable. A fixed additional evaluation panel for all final
policies is another possible check, not a substitute for training repetitions.

## Earlier checkpoint: first 100k reference

Fresh run `curl-seed123-100k-v1` completed exactly 12,500 decisions and 100,000
training simulator steps, with 11,500 update calls. Its fixed ten-episode mean
return was 8.4446 before updates and 678.0208 at the endpoint. This is reward
on a roughly 0–1,000 scale, not a success percentage. Native records and config
are published in [data/first-reference](data/first-reference).

The trajectory fluctuates substantially: the 96k score was 423.07, followed by
678.02 at 100k. The endpoint was prespecified, not picked after looking at scores,
but this last-point jump makes repeatability and later behavior important. Do
not infer a stable 678-level policy or a repeatable advantage over the control.

The paper's 582 ± 146 is a mean and SD over ten trained seeds, not a threshold
that one local result can pass. Our runtime differs from the historical stack.
That result supported continuing the reference/control comparison; it was not a
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

1. The fresh CURL and no-contrastive runs for seed 123 are complete at 100k steps.
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
