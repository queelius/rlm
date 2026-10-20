---
date: 2026-09-12
cutoff_utc: "2026-09-12T04:51:00Z"
question: rq:rl-effective-feedback
status: root_rl_readout_complete_helper_interventions_queued
evidence_level: exploratory
---

# Is the root learning from its decisions, or from the helper's mistakes?

The new evidence gives us a more focused question than “Should we run RL longer?”
Many failed attempts are consistent with a correct calculation on incorrect helper
information. We are testing whether correcting that information improves the
complete solution, and preparing a small experiment that trains the helper itself.

## The first new learning result

The [one-update comparison](2026-09-12-onebatch-learning.md) has completed its
unchanged-model and root-only RL evaluations:

| Model | Correct answers | Wrong answers | Unavailable answers |
|---|---:|---:|---:|
| Unchanged supervised-trained root | 53 | 17 | 2 |
| After one root-only RL update | 52 | 19 | 1 |

Both attempted the same 72 questions with matched seeds, helper and runtime.
Among the 70 questions with an observed answer in both conditions, RL gained two
correct answers and lost four. It also answered one question whose baseline
attempt timed out. This small update did not improve the observed score; it does
not settle whether RL can help with another objective, component or training dose.
The self-SFT evaluation is running and is not included in this result.

These are already-inspected evaluation contexts, separate from this update's
training batch. The older 57/72 baseline used different sampling seeds and must
not replace today's matched baseline.

## Why we are examining the helper

In the training diagnostic, the helper classified records and the root used those
categories to filter, count or sum. The 48 attempts repeatedly used two contexts
of 16 records, with different questions and sampling seeds. They are not 48
independent source documents.

Of the 22 wrong answers, 18 equal the result of applying the requested calculation
to the helper's incorrect labels. Four disagree with that calculation. We checked
saved model replies against the trusted calculation; we did not execute generated
code during this analysis. The root and helper traces contain 102 wrong labels
among 768 repeated label decisions.

The sharper observation is that three of the five mixed-success training groups
have map-consistent answers on all four attempts. In those groups, changing helper
replies explains the observed success/failure contrast. Mixed rewards alone are
therefore not evidence that the root explored better and worse plans.

This is descriptive evidence, not proof of causal error propagation or faithful
program execution. Terminal-reward RL remains legitimate in stochastic settings;
the concern is how informative this small batch is about the root's decisions.

## The tests that will change our next decision

- **Perfect local helper:** repeat the 48 attempts, supplying correct labels only
  for records the root requests. The root must still select scope and calculate.
- **Replay control:** return the exact original helper replies through the same
  callback mechanism. This helps separate better information from delivery changes.
- **Helper-only learning candidate:** keep the root unchanged and reward the helper
  for the fraction of labels it gets right. No sampled complete map was perfect,
  so whole-map binary reward would provide no contrast in this batch.

The first two jobs are queued automatically after the self-SFT evaluation. The
helper-training candidate is CPU preparation, not a launched or qualified update.
Its likelihood calculation must respect the grammar used to generate helper
replies. The current root trainer cannot simply be reused unchanged for that loss.

Shared-prefix continuation training is also relevant prior art, not a new idea
we can claim as ours. Our possible follow-up would hold a real helper reply fixed
and compare how the root uses it. [Credit Assignment with Resets, v2](https://arxiv.org/html/2605.25507v2).

## Evidence and resumption

Under `/project/alex_phd/runs/rlm-research-r4`:

- `analyses/qs6-real-helper48-error-attribution-2026-09-12/REPORT_V2.md`,
  `ROWS_V2.json`, `SUMMARY_V2.json`, and `reproduce_v2.py` preserve the descriptive
  analysis. MAIN independently reproduced all 48 rows and the 18/4/102 counts.
- `sidecars/root-qs6-matched-eval-v1/outputs/` contains the evaluation exports.
  Physical audits record 311 baseline attempts and 318 RL attempts; saved-episode
  aggregation alone misses three returned baseline calls from a timed-out attempt.
- `questions/rl-effective-feedback.md` and
  `ideas/2026-09-12-credit-after-helper-errors.md` record the changing decisions.
- Read `RESEARCH_QUEUE.md` before launching anything. The automatic evaluation,
  oracle and replay owners remain under MAIN's exclusive GPU coordination.

These reports and resume pointers are Git-backed. External weights and raw run
artifacts are not backed up by a source-repository push. The advisor deck remains
at its meeting cutoff while the explanatory interventions are pending.
