# Completed short LLM RL reproduction

**New phase, 7 October at 15:08 UTC:** The user requested continued work toward
reproducing the paper's training result. A fresh-base run of roughly 4,000
questions and 250 weight updates is now active, with fixed final evaluations
planned after completion. Attempt 1 failed with GPU memory exhaustion; the
allocator retry failed at model initialization. Current attempt 3 began at
15:24 using the smaller collection size of our successful short runs. Its
loader-derived budget is 4,080 questions and 255 updates; its own sequence
handles both final tests. Read the
[longer-run plan](longer-training-plan.md) and current
external `SESSION_CHECKPOINT.md` before touching GPU processes. The completion
notice below describes only the earlier presentation batch. Actual allocation
ends 9 October at 20:39 UTC; the 14:00 window below is historical.

Status: 7 October 2026, 13:51 UTC. The requested five-slide presentation was
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

These are ranked future experiments, not running jobs or claims of success.
No extra repeat or rushed new direction is needed to fill the remaining
minutes of the presentation window.

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
