---
date: 2026-10-05
status: declared_before_supplemental_results_now_running
question: How sensitive are the walking comparisons to the starting states used for testing?
training_endpoint: 100000_simulator_steps
training_seeds: [123, 456, 789]
evaluation_seeds: 20000_to_20049_inclusive
---

# Test the same learned walkers from more starting positions

When this panel was declared, the first two completed comparisons favored CURL by very different
amounts: about 242 reward points for one training pair and 24 for the other.
Each score averages ten test episodes. In the second control, nine returns
range from 291 to 503, while one is 47. We keep that difficult episode in the
original result. Instead of deciding which episodes look representative, we
will test every final model on the same larger, new set of starting states.

For example, a walker may recover well from one initial posture but poorly from
another. More test starts help measure that sensitivity. They do not tell us
how well a newly trained model would do: we still have only three training
pairs, however many times we test each model.

## The comparison is fixed before these new tests

- Finish all six original walking runs first. Use their final 100,000-step
  checkpoints, never a checkpoint chosen for its score.
- Test both versions for training seeds 123, 456 and 789 on starts
  **20000 through 20049**, in ascending order. The original ten starts,
  10000 through 10009, are not part of this new panel.
- Keep the same simulator, image processing and deterministic policy as in
  the original evaluations. Do not train, sample training experience, or
  update the model during these tests.
- Run one policy at a time, with the existing shared GPU lock. Each policy
  gets all 50 starts. A failed or unfinished panel has no final score; it
  does not count as a zero, and we do not substitute other starts.
- Preserve the original primary results. Report this panel separately as an
  exploratory check motivated by the observed variability, not as a newly
  independent confirmation of the training effect.

The machine-readable declaration next to this document fixes membership and
budgets. Checkpoint checksums will be recorded when the completed checkpoints
are admitted; the last pair is still training at declaration time.

## What we will report

For each of the six policies, report the average total episode reward over
the 50 new starts. Higher reward means better walking behavior, not a higher
percentage correct. For each training seed, subtract the control's average
from CURL's average. Show all three differences, including any negative ones,
alongside the unchanged original ten-start results.

Keep every per-start return, its evaluation seed and actual simulator-step
count. Descriptive distributions and paired per-start differences can show
whether poor starting positions drive a gap. Do not pool the 300 episodes as
300 independently trained models or claim tight training-effect uncertainty
from them. We will not select a preferred training seed, checkpoint or later
training schedule using this panel.

If the small second-pair gap changes sign or size, the interpretation should
emphasize sensitivity to test starts. If it remains small while the first
gap remains large, training-run variability remains an important explanation.
If all three gaps remain positive, that supports the early pattern on these
additional starts, but it still does not isolate why contrastive learning
helps. None of these outcomes establishes how long an advantage lasts.

## Execution and cost

The evaluation-only tool reuses the existing evaluation function and restores
the saved actor. It does not construct a training replay buffer or restore
optimizers. The full checkpoint format still requires reading and temporarily
deserializing roughly 9 GB, including stored replay arrays; this is memory and
I/O cost, not new training. No new model checkpoint is written.

Evaluate one complete episode at a time and save its return immediately. That
provides a real scientific progress record within approximately 90 seconds
of the first evaluation request, rather than waiting for all 50 episodes.
Native walking evaluation timings suggest about 43 minutes for the six
panels, or roughly 45–60 minutes including loading. Budget 20 minutes per
policy and two hours for the serial batch, subject to the allocation deadline
1791387366. These are caps and estimates, not measured supplemental timings.

The expected additional cost is 300 evaluation episodes, or 300,000 simulator
steps, and **zero training steps**. Save small JSON records with configuration,
parent and source identities, raw returns, counters, timing, and completion or
failure receipts. Keep them outside the original training directories, under
`/project/alex_phd/runs/curl-walker-fresh-starts-20261005`.

This document declares the experiment; it is not evidence that it has run.
Before launch, verify all six parent endpoints, the current owner has exited,
and the shared GPU lock is available. The live training source, original queue,
primary evaluation and checkpoints remain unchanged.

Execution update, October 5 at 08:03 UTC: all original parents completed, the
original queue ended, and the prepared owner acquired the shared lock and
started this panel. A real first episode return arrived about 35 seconds after
child launch. The above conditions describe the admission checks, not a request
to launch it again. No supplemental scores are included in the original-cohort
PDF cutoff. The third original pair favored the control; this does not change
the declared panel membership, settings or reporting separation.

The per-policy entrypoint is
[`evaluate_frozen.py`](../../experiments/curl_replication_20261004/evaluate_frozen.py).
Its invocation for one completed parent is shown below. This is not a command
to run alongside the current training owner: a serial batch owner must first
hold the shared GPU lock and enforce the batch/allocation caps.

```sh
/project/alex_phd/envs/curl-8416d6e/bin/python \
  experiments/curl_replication_20261004/evaluate_frozen.py \
  --source /project/alex_phd/research-cache/repos/curl-8416d6e \
  --parent-run /project/alex_phd/runs/curl-walker-replication-20261004/curl-seed123-100k-v1/run \
  --output /project/alex_phd/runs/curl-walker-fresh-starts-20261005/curl-seed123 \
  --seed-start 20000 --episodes 50 --device cuda --max-seconds 1200
```

The entrypoint refuses an existing output directory and records failed
attempts without a final score. Its time cap is checked between complete
episodes; the future serial owner must provide the outer hard timeout. The
six-policy owner and GPU launch are a separate admission step, not implied by
the existence of this tool.

## Preparation checks

Four focused CPU tests exercise a small real serialized checkpoint through the
actual actor-construction and evaluation functions, with a controlled toy
environment. They check expected actions and rewards, center crops, immediate
episode records, a single large-file checksum pass, provenance, rejected parent
or panel mismatches, incomplete caps, retained failures and overwrite refusal.
Root and an independent reviewer reran them successfully; Ruff checks passed.
These are software checks, not walking scores. Full-size checkpoint loading,
real rendering and GPU throughput still need verification at actual launch.

## Serial launch and unattended handoff

[`run_frozen_queue.py`](../../experiments/curl_replication_20261004/run_frozen_queue.py)
can wait while the final original control is training. During that wait it
holds only its own batch-owner lock, not the GPU lock. It requires the original
queue's end record and all six successful native training endpoints before
acquiring the shared GPU lock. It then runs the fixed six panels sequentially.

The owner checks actual episode returns, not GPU memory or a live process.
It validates each final mean against all 50 declared episodes. Failed or
unfinished attempts remain visible and unscored; valid remaining policies can
still be evaluated. Existing attempts are not silently overwritten or retried.
The 20-minute policy cap and two-hour active-batch cap reserve two minutes
for stopping an unresponsive child. Waiting for training does not consume
the active-batch budget; the allocation deadline still bounds everything.

Eight focused CPU tests use real short-lived subprocesses to check waiting,
shared ownership, exact panel membership, return-based completion, incorrect
means, stalled children, STOP, deadline and duplicate-owner handling. Review
found an exception path that released the shared lock too early. A failing-first
test reproduced it; the corrected owner keeps the lock until its own child
has exited. Independent re-review passed. These checks do not replace checking
real simulator returns once the supplemental evaluation starts.

The prepared external configuration is
`/project/alex_phd/runs/curl-walker-fresh-starts-20261005/QUEUE.json`.
Its `HANDOFF.md`, `SOURCE.json`, `LAUNCH.json` and `queue.jsonl` record actual
deployment and progress. The reusable invocation is:

```sh
python experiments/curl_replication_20261004/run_frozen_queue.py \
  /project/alex_phd/runs/curl-walker-fresh-starts-20261005/QUEUE.json
```

Do not launch a duplicate owner. The deployed process uses a separately sealed
source copy; the command above describes the interface, not a request to start
a second process. Each policy's native outputs are under its job directory's
`run/`, with the owner's completion receipt in the job directory itself.
