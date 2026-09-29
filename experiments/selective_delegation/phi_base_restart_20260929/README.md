---
date: 2026-09-29
question_id: OP29-phi-base-reference
status: accepted_waiting_for_predecessor
scope: two_preflight_only_baseline_failures
scientific_change: none
---

# Restore the two missing Phi base-reference jobs

Version2 is accepted in [the bounded follow-on queue](../followup_admission_20260929/ADMISSION.md).
It is waiting behind existing work; no new model outcome is claimed yet.

Both September28 base jobs failed before creating a scientific owner or loading
the GPU model. The final immutable-plan equality check compared Python tuples in
`native_stop_binding.*.substitutions` with the lists read from JSON. For **both**
worlds, serializing the rebuilt plan to JSON reproduces the entire saved plan
exactly. No task, seed, model, prompt, stop ID, reward or budget differs.

This is an invalid preflight attempt, not a model failure. The eight trained Phi
readouts had no pre-existing PLAN and completed with their dedicated native
audits; this comparison defect does not invalidate those generated responses.
The original two failed scientific commands used about6.3seconds each, followed
by two failed audits of about5seconds each. They produced no model-quality data.

The new wrapper reuses the frozen evaluator and normalizes only the rebuilt
metadata's JSON representation before checking equality against the pinned
original PLAN. It adds explicit restart provenance and its own source pin, then
runs the unchanged native collector. All original files remain unchanged.
Use `restart_v2.py`. New outputs are
`R/textcraft-phi-base-w{42,50}-20260929-002`. The unused first wrapper and its
`20260929-001` preparations remain unchanged as review evidence. Independent
review found that its runtime namespace omitted `prepare_only=False`, which the
unchanged collector requires. It was never launched. Version 2 fixes that field
without changing any scientific input and records the unused preparations.

There are only two base episodes per world (the original panel positions0/7),
with the same seed2026092204 and15-minute science cap. They are descriptive
references, not a reliable estimate of base-model performance across eight tasks.
The dedicated `PHI-AUDIT.json` has the correct denominator; do not use the
inherited generic16-slot display.

## Verification and dispatch

Five focused tests were observed failing before version 2 existed and now pass.
They use actual saved plans, accept the observed tuple/list distinction and reject
changed worlds, sampling seeds or native stop IDs. Both actual `--prepare-only`
commands pass without changing the frozen source or loading a GPU model. Native
generation/decoding/replay remain the already exercised original Phi code path;
no claim of new live model success is made before execution.
The fifth test passes the actual runtime namespace to the unchanged collector
and reaches its existing-attempt guard before model loading. This catches the
runtime-only field that preparation checks missed.

Use the campaign GPU Python with `restart_v2.py --world 42` or `--world 50`.
Run the same command with `--audit` after the owner releases. Parent scheduling
uses20-minute wrapper caps and5-minute CPU audit caps. Inputs, checkpoint policy,
environment and native ownership are inherited; no new dependency installation.

`R=/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.
Original failure receipts are under
`R/information-first-tail-20260928-002/queue/phi-base-w{42,50}.json`.
The startup defect was diagnosed through actual plan reconstruction, rather
than bypassing the immutable-input check or silently modifying old evidence.
