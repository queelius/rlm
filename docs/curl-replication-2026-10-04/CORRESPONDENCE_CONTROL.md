---
date: 2026-10-05
status: first_training_run_active_two_following_seeds_queued
question: Does matching the right observations matter for the early cartpole benefit?
task: cartpole/swingup
new_arm: shuffled_curl
training_seeds: [123, 456, 789]
primary_endpoint: 100000_training_simulator_steps
external_campaign: /project/alex_phd/runs/curl-cartpole-correspondence-20261005
---

# Does the model need to match the right pictures?

Our three original cartpole comparisons favored CURL at 100,000 training steps.
That comparison removed the whole extra image-matching update, not just its
information. It therefore cannot tell us which part of the update helped.
This small follow-up keeps the update but changes which images it asks the model
to match. It is an exploratory explanation check, not a new algorithm or a test
of the cause of the mixed walking results.

## A concrete example

Normally, the learner sees two crops made from the same recorded observation.
It is taught to pick one as the match for the other. For example, two slightly
different views of the same moment when the pole leans right belong together.

In the new condition, the second view comes from a different batch position.
The exercise now teaches the model to match unrelated recorded moments. The real
match remains among the candidates but is treated as a wrong choice. We keep
the network, image crops, reward learning, number of updates and optimizer steps
unchanged. This isolates the pairing rule more closely than deleting the extra
update altogether. These verbal examples are simplified, not saved simulator frames.

**Important limit:** wrong matches can actively damage learning. If the new
condition performs poorly, that alone does not show why correct matching helps.
We must compare all three conditions, including the original no-matching control.

## Fixed comparison

| Condition | What happens in the extra learning exercise? |
|---|---|
| Original CURL | Two crops from the same recorded observation are the intended match. |
| Original same-crops control | The learner still uses random crops, but skips the extra matching update. |
| New wrong-matching condition | Matching labels are retained while the candidate views are reassigned. |

Run three new models from fresh weights, with training seeds 123, 456 and 789.
Reuse the original six fixed-100k comparison records only after verifying that
the reference path and scientific settings are unchanged. These are three new
models in a third condition, not three additional CURL/control replication pairs.
The same initial seed does not make later actions and training experience identical.

Preserve cartpole/swingup, action repeat 8, 12,500 decisions including 1,000
warm-up decisions, batch size 128, and 11,500 updates. Evaluate at decision 0
and every 500 decisions through the fixed endpoint, with the original starts
10000--10009. The primary score is average total episode reward at exactly
100,000 training simulator steps. Higher is better. No best-checkpoint selection,
early stopping on reward, or missing-as-zero scores. Each run also spends
260,000 simulator steps on evaluation, excluded from its training experience.

## Implementation and checks

Change only the target-key ordering immediately before the pinned upstream
similarity calculation. Keep the diagonal labels and both upstream auxiliary
encoder optimizer steps. At each update, draw a permutation uniformly until
no position remains unchanged. Use a private NumPy PCG64 generator seeded with
the training seed plus 1,000,000; checkpoint and restore that generator's state.
Its draws must not advance the replay, crop or action random streams.

Reassigned positions can still contain the same replay record because batches
sample with replacement. Record the rate of duplicate replay indices and the
remaining exact-record matches. Different replay records may also depict similar
physical states; these diagnostics do not establish semantic dissimilarity.
The diagnostics are sampled every 100 updates alongside the ordinary loss logs;
they are not a complete count of every matching target used during training.

The implementation passed focused checks with the real upstream learner on
CPUs: unchanged reference updates, altered pairings with retained optimizer
behavior, unchanged shared random state, duplicate-index accounting, and
restoration of the next update after checkpoint save/load. Resume rejects a
missing permutation state or a checkpoint from another condition. Root reran
12 tests and six subtests successfully; one existing opt-in test for files
larger than 4 GiB was not repeated. Ruff checks passed. Independent review
found no material implementation problems. This verifies the adapter, not a
GPU learning result or bit-identical full simulator reruns.

Implementation: [the small pairing controller](../../experiments/curl_replication_20261004/contrastive_control.py),
[trainer integration](../../experiments/curl_replication_20261004/train_reference.py),
and [focused tests](../../experiments/curl_replication_20261004/test_contrastive_control.py).
A separate source snapshot includes the new module; all copied executable
files match the verified implementation. The live walking snapshot is unchanged.
At 14:33 UTC, the three-job batch was prepared without an owner or GPU outputs.
At 14:56 UTC, its owner started waiting behind the final walking control.
At 16:03 UTC the walking cohort finished, and the first new child launched
0.081 seconds after its queue ended. Real test returns arrived after 19.01
seconds; finite learning updates followed. Logged pairings have no unchanged
positions, with occasional residual same-record matches from duplicate replay
samples, as anticipated. These are sampled diagnostics, not an outcome or a
claim that every incorrectly paired picture is semantically unrelated.

## Budget, ordering and interpretation

Expected total time is 45--60 minutes; each new run has a 2,700-second cap.
Save the complete state periodically at natural episode boundaries and at the
endpoint. Retain all failed attempts. Allow approximately 10.2 GB for three
final checkpoints and one temporary save, in addition to unfinished walking jobs.
The allocation deadline remains epoch 1791387366. Use the existing GPU lock and
start training only after all six additional walking jobs have finished or been
accounted for by their owner. A waiting owner was admitted only after verifying
that the sixth/final walking job was running and holding that lock. Waiting is
not training. The handoff has now occurred, but no completed correspondence
outcome is available at this cutoff.

Report all three conditions at the fixed endpoint, with individual training-seed
scores and unsmoothed curves. Correct matches above both alternatives would be
consistent with useful correspondence information in this setting. Correct and
wrong matches performing similarly would weaken that explanation. Wrong matches
below the no-matching control would chiefly demonstrate harm from incorrect
targets. Three seeds and one task cannot establish a general mechanism.

Keep this follow-up separate from the longer-cartpole and walking results in
[the findings](FINDINGS.md). The [next-comparison record](NEXT_COMPARISONS.md)
explains why it was ranked before a more expensive longer walking study.
