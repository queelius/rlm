---
date: 2026-09-12
cutoff_utc: "2026-09-12T05:51:00Z"
question: rq:rl-effective-feedback
status: fresh_sampling_repeat_complete
evidence_level: exploratory_same_checkpoints_and_exposed_contexts
---

# Learning from successful attempts has a small, mixed signal

We repeated the comparison with new sampling seeds. The model trained on its own
successful attempts again answered 57 of 72 questions correctly. The unchanged
model answered 55 correctly this time, compared with 53 in the first comparison.
This keeps self-SFT useful as a baseline, but does not establish a reliable gain.

Here, **self-SFT means supervised training on the model's successful attempts**:
we saved their root-model actions and trained the root to reproduce them. The
helper's weights stayed fixed. This repeat used the same saved trained checkpoint;
we did not collect a new training batch or perform another training update.

| Evaluation draw | Unchanged model: correct / unavailable | Self-SFT: correct / unavailable |
|---|---:|---:|
| First paired draw | 53 / 2 | 57 / 0 |
| New paired draw | 55 / 2 | 57 / 2 |

Each row covers the same 72 questions. The eight source contexts have already been
used in our research. The new seeds give another sample of model behavior, not
72 new source problems or an independent replication of training.

## What changed in the repeat?

Of the 68 questions with observed answers in both conditions, self-SFT changed
four wrong answers to correct answers, but also changed three correct answers
to wrong answers. The remaining four questions had an unavailable answer in one
condition. Self-SFT recovered two baseline-unavailable answers as correct, while
one baseline-correct and one baseline-wrong answer became unavailable.

The net gain of two therefore combines one additional correct answer among jointly
observed pairs and one net availability-related addition. Missing answers are not
silently counted as observed reasoning mistakes. The effects also vary across
contexts: one context lost two correct answers, while four gained one each.

## The apparent compute saving did not repeat

The first comparison suggested fewer input tokens after self-SFT, but almost all
of that saving came from avoiding one long retry chain. In the new draw, the
direction reversed:

| Physical work in the new draw | Unchanged model | Self-SFT |
|---|---:|---:|
| Actual model-call attempts | 339 | 404 |
| Input tokens in returned calls | 557,729 | 863,518 |
| Output tokens in returned calls | 43,023 | 53,710 |
| Evaluation owner time | 860 seconds | 882 seconds |

Self-SFT used about 55% more returned-call input tokens and 25% more output tokens.
It made 92 helper calls rather than 69. One additional self-SFT request was rejected
for excessive input length; its 8,300 attempted input tokens are recorded separately
from returned-call usage. The unchanged model's physical total includes three
returned calls omitted from the exported summary of a timed-out attempt.

We should not claim that self-SFT generally reduces computation. Concurrent calls,
prefix caching and long unsuccessful paths also mean that token totals and elapsed
time measure different things. Both belong in the report.

## What this changes

The much larger [helper-information diagnostic](2026-09-12-helper-intervention.md)
remains the strongest lead: correct local categories produced 44/48 correct final
answers, compared with 27/48 when replaying the helper's original replies. We are
prioritizing better helper information and smaller helper requests, while retaining
self-SFT as a comparison for future RL experiments.

The proposed helper-RL continuation is paused for a material probability check,
not because RL has been shown to fail. Identical saved helper prompts and outputs
have substantially different logged token probabilities. Those values would change
the importance weights used in learning. We need to understand their meaning before
using them for that update. A serving probe is being repaired after a startup import
error; the failed attempt reached no GPU inference and remains preserved.

## Reproducibility and presentation cutoff

The external research store is `/project/alex_phd/runs/rlm-research-r4`.

- Run: `sidecars/root-qs6-selfsft-seed-repeat-v1/outputs/{baseline,sft}`.
  Both owners completed and released the GPU; both exports cover all 72 coordinates
  with no integrity failures.
- Analysis: `analyses/qs6-selfsft-seed-repeat-2026-09-12/RESULT.json`.
  MAIN generated it with the existing paired endpoint and physical-cost auditor.
  SHA-256: `23704a8517ff3434902f2a15dfdefc1d28d5b966b2401bb1a740ac2b32420253`.
- Frozen repeat plan: new paired seeds `202609120300` through `202609120371`;
  semantic questions and model checkpoints are unchanged.

The delivered advisor deck retains its September 11 cutoff. For the next update,
present the small mixed learning signal alongside the stronger helper diagnostic;
do not reuse the earlier apparent compute saving as a general result. Git preserves
this report, not the external raw runs or model checkpoints.
