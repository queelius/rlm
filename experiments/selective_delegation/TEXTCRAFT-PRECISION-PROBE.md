# Prospective FP16 numerical probe, not a trainer repair

## Completed result, September 22 at 15:02 UTC

The bounded probe completed in 92.47 seconds. All 47 fixed-prefix comparisons
and both fresh generations were finite. Its first persisted model calculation
arrived 5.82 seconds after the owner started.

| Same 47 calls, 885 emitted tokens | BF16 | FP16 |
|---|---:|---:|
| Largest absolute cached/full log-probability gap | 0.50000 | 0.04370 |
| Mean absolute gap | 0.003104 | 0.000426 |

Both fresh generations produced the same ten-token actions as their BF16
references, and their own generation/full-forward scores matched exactly.
These two short responses are only a native-path check, not strong evidence
that longer generated trajectories will match. The fixed selection includes
three calls chosen after observing the largest BF16 gaps. It is a numerical
stress diagnostic, not a representative performance sample or the full 712-call
batch. FP16 is promising for reducing this particular discrepancy; it has not
yet been tested through a backward pass or another optimizer update.

**Decision:** preserve the failed BF16 run and inspect its sole committed
one-step checkpoint under an explicit stopped-run evaluation amendment, with
a matched one-step SFT control. Separately, an FP16 continuation remains a
candidate requiring a finite-gradient check and fresh on-policy sampling.
Do not change precision, tolerances or the old run's completion status silently.

Evidence under the selective-delegation research store:
`textcraft-precision-probe-001/SUMMARY.json`, SHA256
`7f4c64d5e8923305fc8b5523c623bef1cf54307220e58b741fbe12917eb1bf80`;
`TERMINAL-c089870a4afc.json`, SHA256
`c87338e911dbf3fb6e798ffe1a066beb72b7d27b2be754b38290ba82b730ace8`.
The prospective specification below remains the launch rationale.

## Prospective specification

2026-09-22. Completed BF16 diagnostic002 reproduced the maximum-only failure:
712 calls /22,193 emitted tokens, max gap .5, mean .0024346973, p99 .07418245.
The full scan took 169.94 seconds. All 47 forced-cache comparisons matched the
saved BF16 generation scores exactly, including the three worst-gap calls.
This supports cached-versus-parallel numerical arithmetic, not an observed
adapter/temperature/identity mismatch. It does not establish acceptable RL bias.

`diagnose_textcraft_precision.py` loads exactly checkpoint1 and the same original
base weights into FP16, preserving FP32 LoRA, SDPA, T=.5, frozen/eval mode, input
token IDs, and the same 47-call selection. It reuses measured BF16 own-cache/full
values, but compares **FP16 cached scores against FP16 full-forward scores**;
old BF16 scores are not the FP16 correctness reference. The three added worst-gap
calls remain explicitly post-hoc numerical diagnosis, not performance selection.

Two fresh FP16 native generations retain actual emitted IDs and probabilities;
their full-forward replay uses those new IDs even if they differ from BF16.
Nonfinite values are recorded and halt the diagnostic. There is no optimizer,
environment step, dtype promotion rule or tolerance change. Same exclusive owner,
lock and allocation margin; 600-second cap and partial persistence. One focused
tiny native-model fixture tests the fresh-ID branch; existing numeric helpers are
unchanged. Main alone accepts and launches.

## Explicit continuation choices after evidence, none implemented

1. **Keep checkpoint1 as the only trained endpoint.** Evaluate it under an explicit
   stopped-run amendment if useful; do not call the failed run a complete two-step
   training run. Preserve optimizer/RNG and all collected batch2 evidence.
2. **Continue BF16 from checkpoint1/Adam using saved batch2 only after an explicit
   numerical-policy decision.** No second Adam step occurred, so this is a real
   continuation, not a fresh optimizer. The same BF16 sampling contract and retained
   batch2 actions must be preserved. A justified replay-path change or explicit
   tolerance amendment would be a new scientific contract requiring separate review,
   not something this probe silently authorizes. Reconstruct and retain all scores
   before any check; do not regenerate or choose trajectories based on reward.
3. **Fork FP16 continuation if it is actually numerically better.** Restore checkpoint1
   FP32 adapter and Adam moments/step/RNG, record the base-dtype intervention, and
   collect a fresh batch under the FP16 policy before updating. The existing BF16
   batch2 is valuable evidence but is not automatically on-policy for FP16; simply
   relabeling/reusing it would confound sampling and replay precision. An explicit
   off-policy correction would be a different, unimplemented experiment.

For every branch retain immutable failed RL002 and diagnostic receipts; restore
actual checkpoint1 optimizer state rather than resetting Adam. Numeric improvement
does not predict task success, justify more dose, or prove a long-run stability claim.
