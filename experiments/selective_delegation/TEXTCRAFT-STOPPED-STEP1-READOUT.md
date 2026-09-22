# Explicit stopped-run one-step diagnostic

## Accepted execution, September 22 at 15:12 UTC

The RL checkpoint readout is running with a one-hour cap and unchanged 32
task/seed slots. Its first real response returned in 2.06 seconds. The follow-on
queue is accepted and waiting for its owner to release the GPU: one-step SFT,
the SFT readout, then CPU-only native analysis. Actual prepared SFT dosing is
501 whole teacher rows and **exactly 12,074 target tokens**, with zero overshoot.
No evaluation outcome has been used to choose a checkpoint or change the panel.

The accepted queue receipt is `INDEPENDENT-TRAINING-QUEUE-011.json`, SHA256
`f7fb401cbea4ad6494c30609491b21009ecc1474767d10df3218b9e3928a642c`, under
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.
Its four-hour maximum includes waiting; each scientific job retains its own cap.
Raw requests, replies, checkpoints and terminal records are saved as work proceeds.
The analysis output is `analysis-textcraft-stopped-step1-001.json`; absence of
that report means the complete comparison is not yet available.

Main verification: nine sealed trainer/readout tests passed in 5.13 seconds;
the separate actual stopped-plan analyzer test passed in 2.71 seconds; all 77
follow-on job pins matched. The original queue runner's three focused tests also
passed. The analyzer explicitly reads the saved amendment and checks both arms
share it; omitting or altering the amendment still rejects the failed endpoint.

## Rationale and limitations

RL002 remains failed and its ordinary endpoint remains unusable. The additive
amendment qualifies only its sole committed nonzero checkpoint1, not a selected
best checkpoint. Actual COMMIT/STATE, boundary, first-batch credit, Adam step1,
optimizer hash, and released failed owner are checked; absence of a second
optimizer attempt is required. No old artifact or success flag is rewritten.

Question: did the one actual RL update change fixed fresh-goal success relative
to warm public056 (15/32), and to one actual update of additional public-teacher
SFT? Reuse all frozen 16 goals x2 seeds with identical BF16 original-prompt native
inference, budgets and unknown handling. This is a stopped-update diagnostic on
an already exposed fixed panel, not completed two-step RL or new held-out evidence.

The control retains the pre-RL frozen86-row order and warm056, fresh Adam2e-5,
rank8/alpha16/dropout0, one optimizer step. Its target is the actual first-step
12,074 nonzero-credit emitted tokens (388 calls), rounded upward to whole teacher
rows with overshoot explicitly recorded. It does not match state histories,
prompt compute, information, FLOPs, or T1 SFT versus T.5 RL.

Independent ready job: cp1 BF16 fresh32 readout, one-hour cap. The additional SFT
control has a 30-minute cap; only its completed one-step checkpoint gets the second
one-hour readout. No zero-step/partial control substitution. Fresh evaluation does
not wait for the control to start. All GPU acceptance/launches belong to main.

Implementation adds explicit `--stopped-amendment` hooks to worktree trainer and
endpoint reader; absent that option their success-only rejection remains unchanged.
The new validator is narrow to RL002's exact numerical failure and pinned cp1.
Focused tests retain original endpoint/teacher-loss checks and reject a different
failure, successful-run relabel, or wrong update count. Source copies are separate;
live and earlier sealed modules are untouched.
