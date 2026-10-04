# Focused CURL implementation review

Reviewed 2026-10-04 against `experiments/curl_replication_20261004/IMPLEMENTATION_PLAN.md`
and the official CURL source at `8416d6e3869e38ca0e46fcbc54a2f784dc09d7fc`.
The reviewed additions are `train_reference.py`, `env_adapter.py`, `checkpoint.py`,
their focused tests, and the numerical manifest. The worktree base is `6acbb87`.
This is source and CPU verification, not a review of completed scientific results.

## Findings

**Critical:** None found for the selected cartpole/swingup experiment.

**Important:** None requiring changes to the fresh reference/control queue.

**Minor / explicit limits:**

- `--max-seconds` is a per-invocation budget. Checkpoints report cumulative
  `elapsed_seconds`, but resuming starts a new invocation clock. This does not
  affect the current fresh runs. Any future continuation must receive an explicit
  budget within the remaining allocation. Stopping occurs after the current
  episode, followed by evaluation and checkpoint I/O; it is not an exact hard
  wall-clock cutoff.
- A decision-budget truncation is recorded as `boundary_kind=budget_truncation`
  and `truncated_by_budget=true`. Its checkpoint can resume by resetting the
  episode, without simulator/frame-history restoration. Do not claim exact
  trajectory continuation for such a checkpoint. The 1,200-decision pilot has
  this limit and is excluded from seed means; the 12,500-decision reference ends
  at a natural 125-decision cartpole boundary.

## Scientific contracts checked

- Rewards are summed over actual simulator repeats; the adapter stops at the
  simulator ending or 1,000 underlying steps and reports actual repeat counts.
  Time-limit endings bootstrap; training and evaluation counts remain separate.
- Evaluation uses a separate simulator, deterministic center-cropped policy
  actions, raw episode reward sums, actual task reseeding, and restoration of
  Python, NumPy, Torch and CUDA training RNG states.
- Collection chooses the action before the update, as upstream does. Warm-up
  counts decisions; subsequent decisions call the unchanged upstream update once.
  Critic, actor/temperature, EMA and contrastive ordering remain upstream.
- Both overlapping query-encoder Adam optimizers remain intact. The reference
  does not silently replace the authors' two contrastive optimizer steps.
- The no-CURL arm replaces only `update_cpc`. It retains upstream positive-crop
  sampling, SAC updates and target updates. It is a matched auxiliary-training
  intervention, not the paper's Pixel SAC baseline or an equal-compute comparison.
- Checkpoints retain modules, all five optimizers, temperature, filled replay
  contents and ring indices, RNG states, counters and config. Loading copies into
  existing parameters, preserving actor/critic convolution ties and CURL aliases.

## Verification

CPU-only, with GPUs hidden, the focused checkpoint and evaluation tests passed:

```sh
env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  CURL_SOURCE=/project/alex_phd/research-cache/repos/curl-8416d6e \
  /project/alex_phd/envs/curl-8416d6e/bin/python -B -m pytest \
  -p no:cacheprovider \
  experiments/curl_replication_20261004/test_checkpoint.py \
  experiments/curl_replication_20261004/test_train_reference.py -q
```

Result: **3 passed**. This includes a bit-exact next fixed-batch update with the
authors' pixel agent after checkpoint restore, plus shared-parameter identity
checks, wrapped replay restoration and evaluation RNG/reward accounting.
Real-simulator test code was inspected; this reviewer did not run rendering or
GPU work and did not inspect the live sealed pilot as a completed result.

## Behavior set aside

Modern simulator/renderer differences remain an explicitly documented
compatibility adaptation. Generalization to other domains, arbitrary image
shapes, cross-device reproducibility, full physics continuation, broad test
coverage, style cleanup and production hardening are outside this review.
Historical blocked-status metadata and incomplete independent-objective work
should be reconciled at their own milestones; neither warrants stopping the
independent pilot. No source changes or production fixes are requested.
