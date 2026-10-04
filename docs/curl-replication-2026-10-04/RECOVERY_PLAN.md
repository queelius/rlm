# Recovering from the large-checkpoint failure

Date: 2026-10-04, approximately 19:29 UTC. This bounded repair follows the
user's standing instruction to resolve experimental failures autonomously.

## Purpose and decision

Recover useful learning state without changing the scientific comparison.
At 409k, the first CURL extension failed in `torch.save`: the default pickle
protocol cannot serialize a single NumPy array larger than 4 GiB. The latest
intact checkpoint is at 307k. Aggregate checkpoint size was not the relevant
limit. The control is still learning under the old owner; do not edit that
owner's sealed source or let it repeat the known failure.

Restarting all six extensions from their original 100k states is simpler but
discards recoverable learning. Prefer resuming the latest successful natural-
boundary checkpoints for the two started runs, and the original 100k parents
for the other four. This choice is based on recoverable state, not reward.

## Bounded implementation checklist

- [x] Add a real, opt-in greater-than-4-GiB save/load regression, observe failure,
  then explicitly use a suitable pickle protocol in `checkpoint.py`. Preserve
  atomic replacement, trusted loading, arrays, optimizers, counters and RNG.
- [x] Add optional `recovery: true` queue admission without weakening ordinary
  100k-parent admission. A recovery requires a terminated parent receipt, its
  latest successful checkpoint and matching natural-boundary episode, scientific
  configuration and source. Store `kind: recovery`, `restore_step`,
  `restore_env_steps`, `origin_job_path`, `origin_job_sha256` alongside existing
  parent checksum/config/identity fields. Hash large weights only before locking.
- [x] Extend continuation analysis for this one additional recovery link. Verify
  the origin job receipt and preserved original 100k ancestry. Join the ancestor
  curve only through the restored checkpoint. Retain later abandoned-branch
  measurements separately; never splice them into the resumed trajectory.
  Require the child's actual resume step and fixed 500k counters. One complete
  chain remains one training seed, including its earlier failed attempt.
- [x] Test admission/rejection and branch truncation with small CPU fixtures;
  independently review the focused diff. No broad unrelated test campaign.
- [x] Seal a new owner under a separate repair campaign root. Stop the current
  owner via its existing STOP mechanism once replacements are ready, before
  the replay array exceeds the known limit. Preserve the final stopped control
  checkpoint and all original artifacts. Launch under the same GPU lock and
  verify actual resumed learning promptly. Keep all six prespecified seeds/arms.
- [x] Add the repair root to a new observer configuration without editing the
  live observer source. Record failure accounting, recovery ancestry, lost
  training interval, source changes and tests. Update the learning guide's
  execution status and public evidence without inventing a 500k score.

## Scope and checks

Queue implementation: `experiments/curl_replication_20261004/run_queue.py` and
its focused tests. Analysis: `summarize_continuations.py` and its focused tests.
No policy, optimizer, reward, evaluation-seed or learning-budget changes.
The driver already restores arbitrary natural-boundary checkpoint counters;
verify its actual resume event against the declared saved boundary.

The failure is an execution failure, not a score of zero. The discarded
307k-to-409k branch consumed real compute and remains documented. Do not call
the resumed path bit-identical to uninterrupted CUDA training. The existing
100k cohort and its published figures remain valid.

## Verified outcome at 19:48 UTC

The repaired queue is active, with CURL resumed at 307k and the cleanly stopped
198k control next. Five jobs remain queued. The original owner has exited.
Native learning returned within 37 seconds of repaired child launch. The
observer uses CONFIG-v3 to watch all three roots; its prior acknowledgment
history is intact. The eight-page guide was rebuilt and visually inspected.
The next scientific milestone is still the fixed 500k comparison. A large live
checkpoint under the repaired format has not yet been observed at this cutoff.
