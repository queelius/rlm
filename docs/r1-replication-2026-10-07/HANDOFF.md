# Completed short LLM RL reproduction

**Progress check, 16:36 UTC:** The longer run has completed 22 learning calls
and saved model and optimizer files at `step_00008` and `step_00016`.
Twenty-one reported gradient norms are finite and positive. An audit of 1,920
new responses found no prompt/reference or token/reward alignment errors;
eight position-selected answers were independently regraded, with agreement.
See the [bounded audit](longer-attempt4-progress-1636.json). No final score
exists yet. Continue owner exec30612 unchanged through training and its final
evaluations; do not admit another job at trainer exit. Checkpoints are saved,
but exact training-resume semantics remain unvalidated.

**Latest operational update, 16:03 UTC:** Attempt 4 started at 16:02 under
exclusive sequence owner exec30612, with its own training and all six final
evaluation conditions. The old observer was replaced by `reviews-longer-v4`;
read external `SESSION_CHECKPOINT.md` for authenticated process identities.
The small A100 memory check passed; it does not yet establish full-run stability.
All four broader reference conditions are complete and independently checked;
see [results](README.md) and [receipt](broader-reference-receipt.json). No
larger-run trained-model endpoint exists yet. The PDFs remain dated 15:55.

**Earlier phase, 7 October at 15:55 UTC:** The user requested continued work toward
reproducing the paper's training result. A fresh-base run of roughly 4,000
questions and 250 weight updates is prepared, not yet launched, with fixed final
evaluations planned after completion. Attempt 1 failed with GPU memory exhaustion;
the allocator retry failed at model initialization. Attempt 3 completed 12 updates
and sampled 13 collections before another memory failure; it saved no checkpoint.
The [failure receipt](longer-attempt3-failure.json) is retained, not a benchmark result.
Prepared attempt 4 uses a private memory adapter with three passing focused CPU
tests, including matching actual learning-step updates. The shared environment
and official source remain unchanged; GPU memory stability is not yet established.
Its loader-derived budget stays at 4,080 questions and 255 updates, with saving
every eight updates. The owner will handle all six final tests: MATH500, AMC,
and Minerva in both chat and question-only formats. Its caps are ten hours for
training and thirteen hours for the whole sequence. Separate broader checks of
the base and authors' released models are underway, not our training results.
The authors' MATH500 score is 366/500 (73.2%), versus 74.2% reported; those are
their released weights, not a model we trained. Read the
[longer-run plan](longer-training-plan.md) and current
external `SESSION_CHECKPOINT.md` before touching GPU processes. The completion
notice below describes only the earlier presentation batch. Actual allocation
ends 9 October at 20:39 UTC; the 14:00 window below is historical.

Historical batch status: 7 October 2026, 13:51 UTC. The requested five-slide presentation was
published before 13:00 UTC. The optional experiment batch is complete before
14:00 UTC. No training or evaluation job remains active from this batch.

## What to take away

We ran the authors' LLM RL code, verified sampled answers, reward calculations,
nonzero updates and saved weight changes, and evaluated prescribed final
checkpoints. This validates a working implementation path, not the paper's
published numerical results or RL for recursive language models.

The strongest lesson is to check the starting prompt. Without any training,
the base model scored 154/500 in chat style and 305/500 with the question alone.
Chat-style Dr. GRPO training raised the chat score to 308, but that large gain
does not by itself show newly acquired mathematical ability.

Starting from the original model and training on questions alone gave 314 and 306
in two Dr. GRPO runs. Standard GRPO gave 318 in one further run. Each used 512
questions and 32 updates. These are small exploratory gains over 305, with
insufficient training replications to establish dependable gains or rank the
algorithms. Standard GRPO also scored 169 in chat style, so prompt sensitivity
persists. All results, including regressions, are retained.

Six fixed-weight repeated evaluations were byte-identical to their originals.
That checks evaluation repeatability under the same settings, not training
replication. Earlier monitoring outputs did vary at unchanged weights, so we
do not claim that all greedy GPU generation is deterministic.

## What to try next, in order

1. Run a longer fresh-base comparison against the strong question-only
   baseline, using the paper's schedule as the reference. Fix the training
   budget and final-checkpoint rule before inspecting results.
2. Repeat a meaningful gain with fresh training realizations. More test
   questions and repeated tests of the same weights cannot replace this.
3. If an algorithm difference persists, separate the two normalization
   changes and effective update scale. The present GRPO switch changes them
   together, so it cannot identify the cause of a score difference.
4. Return to RLM training with these checks in place: inspect actual inputs,
   rewards and learned behavior; distinguish a stronger starting setup from
   a training gain; grade tool execution separately from plausible code text.

The longer retry is prepared and broader reference checks are underway, as
recorded above. The remaining items are ranked future experiments, not claims
of success. The earlier presentation window is no longer the research deadline.

## Preserve and resume

- Read the [learning PDF](learning-guide.pdf), [research record](README.md),
  and [five-slide presentation](../../slides/2026-10-07-r1-replication/research-update.pdf).
- Operational checkpoint and queue:
  `/project/alex_phd/runs/r1-zero-replication-20261007/SESSION_CHECKPOINT.md`
  and `NEXT_JOBS.md` in that directory.
- Training outputs and full final model/optimizer checkpoints:
  `/home/atowell/research-runs/r1-zero-replication-20261007/`.
- Native evaluations, immutable protocols, and review receipts:
  `/project/alex_phd/runs/r1-zero-replication-20261007/`.
- Official source and environment are pinned in the research record. Do not
  modify sealed source or reuse transformed dataset caches between conditions.
- Saved optimizer state exists, but native resume has unverified continuation
  semantics, including actor synchronization and data position. Prefer a fresh
  base run for the next scientific comparison; do not call resume equivalent
  to uninterrupted training without testing it.

GitHub contains the PDFs, source and lightweight evidence, not the heavy
checkpoint files. Keep both external stores when changing allocations.
