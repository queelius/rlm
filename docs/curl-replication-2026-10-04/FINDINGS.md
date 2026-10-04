---
date: 2026-10-04
cutoff_utc: 2026-10-04T22:48:00Z
original_100k_cohort_cutoff_utc: 2026-10-04T18:37:00Z
execution_update_utc: 2026-10-04T22:48:00Z
stage: exploratory_compatibility_reproduction
question: Does contrastive learning improve pixel-based control beyond random-crop augmentation?
primary_endpoint: 100000_training_simulator_steps
continuation_endpoint: 500000_retained_training_simulator_steps
external_runs: /project/alex_phd/runs/curl-replication-20261004
---

# What we have learned so far

Both versions learned in all three completed 100k pairs, where CURL finished
higher. **The two completed 500k pairs have different winners.** The first
favors the control; the second favors CURL. The early advantage is more consistent
so far than the later advantage. The third pair remains incomplete. We cannot
conclude general late harm, superiority or equivalence from these few seeds.

## Two longer pairs at 22:48 UTC: the first reversal is not universal

| Training seed | CURL at 100k | Control at 100k | CURL at 500k | Control at 500k | Final paired difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| 123 | 678.02 | 454.47 | 842.74 | 866.94 | -24.20 |
| 456 | 446.17 | 240.54 | 866.14 | 810.96 | +55.18 |

All are the declared endpoint scores, not best checkpoints. Each score averages
ten fixed evaluation starts; each row is one training seed. The corresponding
early differences were +223.55 and +205.63. Both late differences are smaller,
but their signs differ. The two-pair average difference is +15.49, a descriptive
number that must not hide the opposing outcomes or be treated as precise evidence.

**What changed:** the second pair contradicts a general account in which the
control necessarily catches up and wins. Extra image matching may help early
learning more reliably than final performance, and the later result can depend
on training randomness. This is a hypothesis supported by a small local pattern,
not a general causal mechanism or evidence that the methods are equivalent.

**New endpoint checks:** control seed 456 ended naturally at 62,500 decisions /
500,000 training steps / 61,500 update calls, exit 0 and complete=true. The
mean 810.9574102627266 was independently recalculated from the ten expected
evaluation seeds. All 3,500 logged training values in this invocation are finite;
no failure occurred. Its last five means were 801.17, 781.36, 804.81, 815.77 and
810.96. The final checkpoint is 11,354,767,581 bytes, matching the receipt;
write time was 29.44 seconds. Both second-pair chains pass configuration,
ancestry and fixed-endpoint checks. Each used 500k physical training interactions
and 1.26m evaluation interactions. CURL still performs extra optimizer work,
so matching interactions does not match wall-clock cost. First-pair CURL's extra
102k recovery interactions remain separately reported.

**Decision:** finish the third pair, then prioritize testing scope over explaining
the first seed's reversal as general late harm. The owner already started CURL
seed 789; real scientific updates arrived within 11 seconds, followed by finite
learning and a checked evaluation of 565.99 at 116k. Its matched control is
queued. A CPU protocol check is preparing a second-task comparison, not changing
the live cartpole runs. The shared-checkpoint matching-on/off proposal remains
available to test continued value, but no mechanism study was launched.

The guide now shows both completed pairs side by side and keeps the three early
results. Evidence: [second control records](extension-data/no-curl-seed456-500k),
[two-pair snapshot](extension-data/two-500k-pairs-summary.json), and
[completed-pair curves](figures/completed-500k-pairs.pdf). Earlier sections below
preserve interpretations at their original cutoffs.

## Second longer CURL result at 21:53 UTC: more learning, comparison still pending

CURL seed 456 finished at **866.1353**, up from **446.1711** at 100k. It used
500k physical training interactions, without an abandoned branch. This is the
second completed longer CURL model, but its control is still training. Do not
compare the mean of two completed CURL seeds with one completed control seed as
though that were a matched comparison. The only completed 500k pair remains
seed 123, where the control finished higher.

The new endpoint is the planned final policy, averaged over the ten expected
evaluation starts. The last five measured means were 862.47, 863.91, 860.56,
860.65 and 866.14. The final mean was independently recalculated from episode
returns. Native completion is 62,500 decisions / 500,000 training steps / 61,500
update calls; exit 0, complete=true, natural final episode, no failure, and all
4,000 logged metric values in this invocation are finite. The final checkpoint
is 11,386,654,987 bytes, matching its receipt; the write took 29.82 seconds.
Evaluation used 1.26m interactions across the original run and continuation,
separate from training. Continuing a saved model is not a new independent seed.

**Interpretation:** both completed CURL models learned substantially more with
longer training. That does not establish that continuing the extra objective
helps compared with omitting it. The fresh seed's successful large saves also
show that the repaired save path works beyond the original recovery case.
This is operational confirmation, not an extra scientific replicate.

**Decision:** keep the fixed-endpoint queue unchanged. The owner started control
seed 456 about three seconds after CURL exited. Real learning returned within
11 seconds; the control subsequently reached 127k with finite updates and a
checked mean of 455.1383 at 124k. Both seed-789 extensions remain queued.
Conditional mechanism and second-task proposals in NEXT_COMPARISONS.md remain
unlaunched. The guide records the second CURL endpoint without adding an
unmatched curve or claiming a second paired result.

Evidence: [completed second CURL records](extension-data/curl-seed456-500k),
[dated continuation snapshot](extension-data/second-curl-500k-summary.json).
Original records are under the repaired external campaign root. The older
sections below preserve what was known at their stated cutoffs.

## First longer pair at 20:50 UTC: an early lead does not guarantee a later lead

| Retained training steps | CURL | Same crops, no image matching | CURL minus control |
| --- | ---: | ---: | ---: |
| 100,000 | 678.02 | 454.47 | +223.55 |
| 500,000 | 842.74 | 866.94 | -24.20 |

Both rows follow the same training seed, not independent repetitions. Each score
averages ten fixed evaluation starts. The 500k scores are the planned terminal
policies, not the best points along the curves. Neither failed attempts nor
the diagnostic pilot contribute scores. See the [paired summary](extension-data/first-500k-pair-summary.json),
[completed control records](extension-data/recovered-control-seed123), and
[paired curve](figures/first-500k-pair.pdf).

**What changed:** the early difference in this pair did not persist. That is
consistent with an early-learning benefit, followed by the control catching up.
It is not proof that the extra objective caused late harm, nor that the methods
are equivalent or that the control usually wins. A single training seed cannot
establish the size or consistency of this late difference. The other two seeds
and another task could change the interpretation.

**Checks:** the control ended naturally at 62,500 decisions / 500,000 retained
training steps / 61,500 update calls, with all ten expected evaluation seeds and
mean return 866.9394162304535. Its 2,639 logged update values in this invocation
are finite; no failure appears. The final checkpoint is 11,354,767,581 bytes,
matching its receipt, with a 30.45-second write. Exit was 0 and complete=true.
The last five evaluation means were 854.99, 853.15, 874.71, 860.42 and 866.94.
Both chains pass the same configuration/ancestry and endpoint checks.

**Costs and scope:** CURL used 602k physical training interactions after repeating
102k lost-state work; the control used 500k. Both retain a 500k-step history.
Their physical evaluation totals are 1.53m and 1.27m interactions, respectively,
separate from learning. This is not an equal-physical-cost efficiency comparison.
The chart excludes the abandoned branch and any unfinished training seed.

**Decision:** finish the remaining two prespecified pairs. CURL seed 456 has
already resumed from 100k under the same owner and produced real updates within
12 seconds, then finite learning and checked evaluation through 108k. Its control
and both seed-789 extensions remain queued. No live source was changed.

The learning guide gains one page showing both curves and the limits above.
The original 100k result is preserved, now clearly labeled as early performance.
This is a useful reproduction lesson, not a novel general result. Any experiment
about *when* to stop the extra matching objective must be declared as a new
comparison, not inferred from this changing ordering alone.

## Earlier 500k result at 20:08 UTC: a higher score, not yet a comparison

The first CURL model scored **842.7436** at the fixed final checkpoint, compared
with **678.0208** at 100k: a gain of **164.7227** reward points for the same
training seed. The score averages ten fixed evaluation episodes on cartpole
swing-up. Those episodes are not ten separately trained models. The final
reward was not chosen from the best checkpoint; the last five measured means
were 843.62, 832.22, 847.59, 847.46 and 842.74.

**Interpretation:** this model learned a stronger controller with longer
training. It does not yet show that image matching remains beneficial compared
with the same-crops control. That control is now running and had already learned
substantially before its planned technical stop. Nor does one score close to a
published average reproduce the paper's multi-seed result on its original stack.
Continue the prespecified paired endpoints rather than declaring success early.

**Native completion checks:** 62,500 decisions, 500,000 retained training steps,
61,500 update calls, all ten expected evaluation seeds, a natural final episode,
no failure and all 1,928 logged update values finite in the repaired invocation.
The final checkpoint is 11,386,654,987 bytes, matching its receipt; its write took
29.68 seconds. The owner reports exit 0 and complete=true. Full original ancestry
and recovery metadata pass the continuation analysis.

**Cost:** this one chain used 602,000 physical training interactions because
102,000 had to be repeated after the saving failure. Its three invocations also
performed 1,530,000 actual evaluation interactions (260k + 770k + 500k), none used
for learning. The restored counter ends at 1,270,000 evaluation interactions;
that counter omits the abandoned branch, so it is not the physical total.
No equal-physical-budget claim is justified by the repaired comparison.

**Next:** finish its matched control, then the remaining two pairs, all already
queued. The control resumed at 198k, produced real update metrics within
18 seconds of launch and reached 203k with a checked ten-episode evaluation.
No extra GPU owner or new training variant was started. See the
[completed recovery records](extension-data/recovered-curl-seed123) and the
[dated continuation summary](extension-data/first-500k-summary.json).
The guide now shows this single completed endpoint explicitly while retaining
the three-pair 100k figure. A paired 500k plot will be more informative once both
versions have finished; no unpaired line is presented as comparative evidence.

## Recovery check at 19:54 UTC

The repaired run successfully saved its first large live checkpoint at
**408,000 retained training steps**: 9,316,517,003 bytes, written in 23.24 seconds.
The file size matches the native receipt, and finite updates and natural episodes
continued beyond the old 409k failure point. This verifies the repaired save path
on the real workload; it is not a new final performance score.

The repeated portion of training provides a useful consistency check. All 1,024
overlapping logged update values, 260 evaluation episode returns and 102 training
episode returns after the restored 307k boundary exactly match the old branch.
This is observed agreement for these records on this machine. It does not prove
that all unlogged state matches or that every future restart will reproduce an
uninterrupted CUDA run. Repeated interactions still cost real compute and are
not additional independent evidence. Five other extensions remain queued.

The new review alert itself was the already documented clean control stop,
not an additional completed scientific result. Continue to the fixed endpoints.
The learning PDF's completed-results cutoff and figure remain unchanged.

## Earlier recovery decision at 19:42: a save-format limit, not a learning result

The first CURL extension reached **409,000 training steps**, then failed while
saving its replay memory. Our checkpoint adapter relied on a serialization
format that cannot encode a single NumPy array larger than 4 GiB. The learning
metrics remained finite; the last measured evaluation was 823.41 at 408k.
That intermediate reward is not a substitute for the missing 500k endpoint.
This is a defect in our adapter, not evidence of a failure in the paper's method.

The previous checkpoint at **307,000 steps** survived because saves replace
the old file only after the new one is complete. We repaired the format and
verified a real save-and-reload of an array larger than 4 GiB. The independent
root check took 26.82 seconds, about 16.5 GiB peak RAM and 4.30 GB temporary
disk; all three focused checkpoint tests passed. The small pilot and earlier
checkpoint checks did not cover this size boundary.

The fixed recovery rule is to use the latest intact natural-boundary checkpoint,
regardless of its reward. We preserved the failed run and asked its owner to stop
the control before the same limit. The control saved cleanly at **198,000 steps**.
The repaired owner then acquired the same GPU lock and resumed CURL at **307,000**.
Its first real evaluation returned within 37 seconds of launch; finite learning
updates and natural episodes followed. At this check it had reached 328,000
retained steps. The control and four unstarted extensions remain queued.
See the [recovery plan](RECOVERY_PLAN.md).
The learning algorithm, evaluation starts and retained training target stay the
same. Startup and resumed learning are verified; completion and a large live
checkpoint under the repaired format still need observation.

**An important accounting distinction:** the failed 307k-to-409k branch consumed
102,000 real training interactions, about 15 minutes since the previous save,
but its learning state was not retained. A recovered CURL path that finishes at
500k will therefore have used **602k physical training interactions**. We must
report this extra cost, not describe the repaired comparison as equal physical
compute. Its saved model will still contain a 500k-step training history. The
abandoned curve stays available as diagnostic evidence but must not be spliced
into the recovered curve or counted as another training seed.

Native [failed-attempt records](extension-data/failed-seed123) include the error,
successful earlier checkpoint receipts and interrupted learning curve. They are
stored outside the original `data/` cohort so the unchanged 100k report cannot
accidentally combine duplicate training seeds. The original six 100k results
and figure remain valid. The [cleanly stopped control](extension-data/stopped-control-seed123)
also retains its native records. There are no completed 500k scores at this update.

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
