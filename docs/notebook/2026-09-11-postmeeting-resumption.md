---
date: 2026-09-11
cutoff_utc: "2026-09-11T18:12:30Z"
status: diagnostic_launched_no_completed_result_yet
question: rq:rl-effective-feedback
---

# Research resumed: first measure useful RL feedback

The approved direction is to improve reward-based learning, then learn useful
decomposition and stopping decisions. See the [first execution plan](../research-plans/2026-09-11-first-diagnostic-execution.md).

The first new diagnostic is running on one A100 40 GB, on node an22/allocation5801.
At this cutoff the model server and collector have started; the GPU process holds
33,656 MiB. This is not yet a completed model result or a training improvement.

- Twelve tasks from two existing training contexts; four new sampled attempts each.
- The root and helper models are fixed. No optimizer updates occur in this diagnostic.
- Measure correct answers, operational failures, usable training traces, reward variation
  within each task's four attempts, and compute cost separately.
- These training records have been used before. This is a readiness diagnostic, not
  evidence of held-out generalization.
- The job started at18:11:35 UTC with a1,800-second outer cap and per-attempt artifacts.

## Resume pointers

Active store: `/project/alex_phd/runs/rlm-research-r4`.
Experiment: `sidecars/root-qs6-feedback-diagnostic-v1/`.
Output: `outputs/attempt-001/` within that experiment.
Owner: MAIN (`/root`), terminal session12586.
READY identity: `14d270ac990e331eca21ef8ae4e4c8e6a81cfc961307042c94f57c7566d7573e`.
READY SHA-256: `5028ab0b2076e60d4a5814ab7e1f834fe8e8d75b967af4fa9686cf05bc4a3893`.

Read `OWNER_TERMINAL.json`, `DIAGNOSTIC_SUMMARY.json` and the collection/export artifacts
when available. Inspect the live `RESEARCH_QUEUE.md` before launching anything else.
Six focused CPU checks passed; MAIN separately verified the sealed owner inputs.
Do not replay completed coordinates when resuming a partial attempt.

The old handoff pilot is parked after allocation-runtime and expected-prompt mismatches.
Its failed attempts are preserved as operational failures, not model failures. The new
runtime is `sidecars/runtime-an22-5801-v1`; do not remove its allocation-owned private store.

The shared account monitor reported20% remaining at18:10:38 UTC. Optional research is
winding down under the recorded reserve policy; the accepted GPU job keeps its cap.
Allocation environment reports an end at September15,17:30:16 UTC; this is not a fresh
scheduler query.

This Git checkpoint preserves plans and resume pointers, not model weights, credentials,
or the external experiment directory. No slide update is warranted from runtime repair alone.
