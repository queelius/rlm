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

## Follow-up: metadata for a resumed run in a new directory

At the first paired-results review, a focused test reproduced one metadata
omission: resuming into a new output directory did not write `config.json`.
The driver now writes the configuration when that file is absent, after
validating the checkpoint's scientific settings and episode boundary. Existing
configuration files remain unchanged; the resume event still records the parent
checkpoint and current extended configuration.

The two added tests cover new-directory metadata, preservation of an existing
configuration, and rejection of a changed training seed. Together with the
original tests, **five CPU tests passed**. An independent diff review found no
blocking issue. This does not alter the learning algorithm or any live sealed
run. Analysis of a future 500k continuation still needs an explicit endpoint and
resume provenance; the current 100k summary deliberately excludes resumed runs.

## Follow-up: continuation queue and multi-root review

The next source snapshot adds optional resume, explicit shared-lock and
environment-step endpoint fields to the existing queue. Native parent records
must show matching completed 100k training and a natural episode boundary.
Checkpoint hashing occurs before the GPU lock; the child receives the resolved
resume path, and the original parent is not overwritten. The observer can
watch sibling output roots while retaining its existing delivery and quota rules.

Twenty focused CPU tests passed across the queue and observer. New fixtures
exercise real subprocess argument/lock behavior, one pre-lock checkpoint hash,
wrong or truncated parent rejection, a wrong simulator-step endpoint, and
terminal events from two roots without duplication. Independent source review
found no blocking scientific or process-ownership defect. Payload restoration
remains the driver's responsibility; this is not a new claim of bit-exact
uninterrupted GPU training. The 500k analysis is separate work, not supplied by
these queue changes.

## Follow-up: separate continuation analysis and live resumption

`summarize_continuations.py` is separate from the unchanged 100k summarizer.
It authenticates the original configuration and native terminal records against
the queue's parent receipt, checks the small configuration checksum and saved
checkpoint identity, and joins the parent curve to the child exactly once.
It does not repeatedly read or hash multi-gigabyte weights. One parent/child
chain counts as one training seed. The 500k endpoint requires exact cumulative
decision/interaction/update counters, ten distinct evaluation starts and a
successful queue receipt. Duplicate child attempts are retained but cannot
contribute a selected endpoint. Missing and failed runs never become zero scores.

Thirteen focused tests passed, along with Ruff checks and formatting. Independent
source review found no blocking issue and checked the real first continuation's
receipt, configuration and resume schema. The final small additions preserve
the verified parent curve before child startup and preserve measured points
from partially written child logs without admitting a terminal score. Their
focused tests also passed; the root reviewer inspected the final implementation.
JSON and Markdown reporting require only the standard library; plots are optional.

A real read of the active 500k directory correctly reports one incomplete chain
and zero scored chains. GPU resumption also worked: the first child restored
decision 12,500 and produced finite update metrics, natural training episodes
and a ten-episode evaluation at 104,000 steps (mean reward 679.7581). This is a
live execution check, not evidence of a completed 500k comparison or bit-exact
identity to uninterrupted training. No sealed training source was edited.

## Follow-up: a large checkpoint failure and bounded recovery

The first longer CURL run failed while saving at 409k training steps. Default
pickle protocol 2 cannot serialize a single NumPy array larger than 4 GiB.
This defect was in our added checkpoint code. It was not covered by the smaller
CPU fixtures or the completed 100k runs. Atomic replacement preserved the
307k checkpoint. The production repair explicitly requests protocol 4.

A real 4 GiB + 1 byte array reproduced the old failure, then saved and reloaded
with matching content under the repair. The root independently reran all three
checkpoint tests, including this opt-in test: 26.82 seconds for the large case,
about 16.5 GiB peak RAM, and a 4,294,984,149-byte temporary checkpoint. No live
source or checkpoint was overwritten. Six focused queue tests also passed.

Recovery admission accepts a closed failed or stopped origin, its latest natural
checkpoint and authenticated original 100k ancestry. The new analysis separates
the abandoned branch from the retained history. It still requires a fixed 500k
endpoint and counts a complete chain as one seed. Twenty-four focused analysis
tests and Ruff checks passed in the root session. Independent review identified
admission mismatches; failing rejection fixtures reproduced them before repair.
The reviewer found no remaining important issue in the final focused diff.

The old control stopped and saved at 198k under its own owner. The replacement
owner acquired the same lock and CURL resumed at 307k, returning a real evaluation
within 37 seconds, followed by finite updates. A read of those actual records
correctly reports one incomplete recovery, zero terminal scores and 102k abandoned
training interactions. This verifies the handoff, not a completed experiment.
We still need to observe a large live checkpoint save with the repaired format.
