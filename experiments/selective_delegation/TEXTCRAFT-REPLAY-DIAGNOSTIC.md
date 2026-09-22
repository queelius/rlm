# Checkpoint-1 replay discrepancy: diagnosis, not a tolerance change

## Additive diagnostic 002: full scan has priority

Version 001 source/output/CPU receipts remain unchanged. Version 002 first replays
**all 712 saved emitted sequences** through checkpoint-1 full forward, persisting
each result and then `FULL-FORWARD-SUMMARY.json`: exact token maximum/mean,
quantiles, offending IDs and both original threshold branches, without asserting
or changing the thresholds. A timeout retains per-call results and a clearly
partial aggregate. This directly attempts to reconstruct the failed batch check;
the prior subset-only 001 design below is retained as history.

Only after the complete full-scan aggregate is saved does 002 run forced cached
same-token comparisons on the unchanged fixed44 plus up to three highest-gap
calls not already selected (ties by call ID). These additions are explicitly
post-hoc numerical diagnosis, not outcome-selected performance evaluation. The
two same-seed native regenerations run last and are optional if time is short.
The same 600-second cap applies; no optimizer, tolerance or dtype changes.
Four focused tests cover the original fixtures plus aggregate thresholds and
deterministic post-hoc selection. `PLAN.json` now pins all 712 saved calls and
generation-score receipts. Main alone accepts and launches a new 002 output.

2026-09-22, CPU preparation only. The second RL002 rollout completed 712 calls and
32 native-verified episodes, then stopped **before optimizer step 2**. Checkpoint 1
remains intact; the run correctly declares its endpoint unusable. Neither another
optimizer update nor a relaxed numerical threshold is proposed here.

## Current design: reconstruct the full discrepancy first (version 002)

Review identified that a 44-call subset might miss the actual offending token
and cannot determine the full-batch mean. The original full scan took only a
few minutes, so version002 first recomputes **all 712 saved calls**. It saves
each result and the complete maximum, mean, quantiles and both threshold
conditions before doing the more detailed cached comparison. No new trajectory
is sampled for that full scan.

It then checks the same 44 fixed requests described below, plus up to three
largest-gap requests outside that set. These additions are explicitly selected
after seeing numerical differences to diagnose them; they are not an unbiased
sample or a task-performance comparison. Two native regeneration checks come
last and are optional if time is short. The total cap remains ten minutes.
An interrupted full scan retains partial results rather than a complete-batch
claim. The original version001 and its preparation receipts remain unchanged.

Main read the original implementation and the complete version002 change, and
ran all four sealed focused tests (passed in5.05 seconds). The main CPU rerun
also exactly reproduced the overlap result below and checked the matched calls
against the second batch's native-audit hashes. No GPU diagnostic has run yet.

At13:56 UTC, version002 was accepted into queue010 behind the active base-model
comparison and its CPU analysis. The queued diagnostic does not interrupt them.
Receipt: `R/INDEPENDENT-TRAINING-QUEUE-010.json`, SHA256
`34ef1fa66bdfbf04a7bb317247ff3636496cac839b5d9f50506424eb4da60b50`.
Its first launcher preflight exposed an output-inventory assumption for CPU jobs;
the additive generic002 launcher now ignores outputless CPU jobs only for GPU
ownership checks, while still waiting for the authenticated whole supervisor.
Four focused launcher tests passed on the main rerun, and the actual preflight
confirmed it waits. No live run or original receipt was edited.

Version002 source manifest: `R/source-textcraft-replay-diagnostic-002/SOURCE.json`,
SHA256 `553c4b6fcd76e9718395edd19a8798aa5db1b5f07005a85976023ff119209740`.
Output: `R/textcraft-replay-diagnostic-002`; PLAN SHA256
`526054848a68b30e788c2f39ba57a0e5693117393ecab2adceca9595fe7a488f`.
Proposal: `R/TEXTCRAFT-REPLAY-DIAGNOSTIC-PROPOSED-002.json`, SHA256
`aa3347fb9fb3d37b08b84764763f5111bffa33e258ca3d44c90325915d2d499c`.

## What the failure does and does not establish

In source-065-textcraft-terminal-rl-v2/rl_textcraft_terminal.py:550–581, the collector
recomputes each saved emitted sequence under unchanged sampled weights. It compares
these full-forward log probabilities to sidecar scores captured during generation.
It raises if maximum absolute token discrepancy exceeds .25 **or** mean exceeds
.025, **before saving BEFORE_LOGPS.json**. Consequently the original failed maximum,
mean, offending token and failed branch are unknown; they cannot be recovered from
the exception alone. BATCH.json, native calls and generation-logps remain available.

Generation scores are post-temperature, with T=.5, top-p=1 and top-k=0; replay uses
raw full-forward logits divided by .5. Thus there is no evident double-temperature
mistake. Sample 2 was collected after checkpoint 1 in frozen/eval mode, with no
intervening optimizer update. Replay also precedes the next train-mode switch.
BF16 cached-versus-full arithmetic is plausible, not established. A stale adapter,
unexpected generation processing or mode/configuration discrepancy remains a
possible alternative until the same-token comparison is measured. Earlier exact
train/eval replay agreement is not evidence that cached/full paths agree.

## Existing CPU evidence at identical weights and token sequences

Joining sample-1 credited AFTER_LOGPS to sample-2 calls by **both complete input
and output token-ID arrays** finds 58 sample-2 calls representing 23 unique
prefix/output pairs, 1,103 token observations including repeats. These compare
checkpoint-1 full-forward scores with checkpoint-1 captured generation scores:

| Statistic | Observed value |
|---|---:|
| Maximum absolute token difference | 0.13279235 |
| Mean absolute token difference | 0.00080318 |
| Largest-gap call | t07-r3-flat-c009 |
| Matching earlier call | t07-r2-flat-c009 |
| That call's generation-minus-full sequence sum | −0.07472602 |

This small, non-independent overlap does **not** identify the failed threshold on
all 712 calls. Its reproducible receipt is
`R/textcraft-replay-diagnostic-cpu-001/CPU-EXACT-OVERLAP.json`, SHA256
`875c660d7dd6027ea370c43a06b17d28f79032dad5bfbebcd876b6879c75b0d7`.
The equivalent computation is `cpu_overlap()` in `diagnose_textcraft_replay.py`.

## Original 44-call design (version 001, superseded before GPU launch)

The prepared selection is outcome/gap-blind: all 32 episode-last calls, plus one
fixed SHA-selected call from each of 12 input-length rank bins excluding those
last calls (44 total). No reward filtering or replacement after a discrepancy.
The CPU-only selection PLAN is
`R/textcraft-replay-diagnostic-cpu-001/PLAN.json`, SHA256
`b309850dfa5e81aa9e089c2a989a5418c7ad034260faeda79590d988670f801f`.
This preparation artifact binds current worktree source; a later source seal must
prepare its own new output, without overwriting this CPU receipt.

On exactly committed checkpoint 1, BF16 base/FP32 LoRA and SDPA, compare:

1. Saved generation scores against full forward on saved emitted IDs.
2. Forced cached incremental forward on those **same IDs** against both paths.
3. Same-seed native generation for just the first two selected calls, recording
   actual IDs/scores and refusing to align scores if outputs diverge.

Full-forward results are saved before cached/re-generation work, and same-token
results before native regeneration. Every token ID, signed/absolute difference,
sequence sum, quantiles and top offending positions is retained. No tolerance
assertion erases diagnostic results. Model config, backend, library/device versions,
dtype and frozen/eval state are recorded. There is no optimizer, training, environment
action or repair. Existing exclusive GPU lock and lease margin apply; wall cap is
600 seconds, preserving partial diagnostics on timeout. This subset cannot establish
the failed full-batch mean.

If newly forced cached scores reproduce captured scores while full-forward differs,
that supports numerical-path rather than policy/trajectory change as the immediate
cause. If captured/forced scores differ materially, investigate that mismatch first.
Either result is a numerical diagnostic, not evidence of long-run RL instability,
reward improvement, or permission to adopt FP16 or change the tolerance.

CPU qualification: three focused tests pass (outcome-blind selection, exact gap
arithmetic, and tiny native PEFT full/cached/generation parity with frozen parameters);
Ruff passes. Actual 44-call preparation completed with CUDA hidden. No GPU launched.

```sh
CUDA_VISIBLE_DEVICES='' "$TRAINPY" "$SOURCE/diagnose_textcraft_replay.py" \
  --output "$R/textcraft-replay-diagnostic-001" --prepare-only
# Main alone accepts/assigns the GPU and runs the same CLI without --prepare-only.
```

`SOURCE` must be a new seal containing this script and the unchanged source065v2
dependencies. Worktree `rl_textcraft_terminal.py` exactly matches the immutable
source065v2 SHA256 `8f9b746f91aa9768760d31976de3b71a8c54f6293251360a7c40c11da81dc956`.
