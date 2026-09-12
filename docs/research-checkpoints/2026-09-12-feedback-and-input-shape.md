---
title: More training signal did not yet produce better answers
date_utc: "2026-09-12"
evidence_cutoff_utc: "2026-09-12T12:04:00Z"
status: exploratory_completed_comparisons
question: Which explanations for weak reinforcement-learning results still look plausible?
---

# More training signal did not yet produce better answers

We have completed two more checks of why reinforcement learning (RL) has added
little to our supervised helper. Neither produced an improvement. This is useful
for choosing the next experiments, but it is not a general limit on RL.

## Changing how feedback is calculated

We trained two helper models once from identical weights on the same 128 sampled
answers to 32 questions. The usual method compares each answer with other attempts
at the same question. If every attempt is equally wrong, it supplies no relative
signal. The alternative compares with the average reward on the other questions.
It supplies no correct answer, but gives more questions a nonzero training signal.

| Observation | Same-question feedback | Other-question feedback |
|---|---:|---:|
| Questions with nonzero training signal | 5/32 | 32/32 |
| Correct question categories | 120/128 | 120/128 |
| Correct news categories | 112/128 | 112/128 |
| Missing evaluation answers | 0 | 0 |

All 256 evaluated labels were identical across the two updated models. Their
weights did change, and the two updates were different. More nonzero feedback
therefore did not yield better answers in this one-update comparison. Both
models still correct only the same one additional question compared with the
supervised starting helper; neither improves the news result.

## Matching the training input size did not reveal a gain

Our earlier helper RL trained on individual questions but was evaluated on
requests containing four records. Could that mismatch conceal an improvement?
We evaluated the unchanged helper and the four-update reference again, this time
with one record per request and identical inputs across models.

| Correct answers | Supervised starting helper | After four RL updates |
|---|---:|---:|
| Question categories | 120/128 | 120/128 |
| News categories | 109/128 | 108/128 |

Every answer was available. There were zero improvements and one regression:
a news article changed from the correct World category to Business. The fresh
starting-model run reproduced all 256 earlier singleton answers, including their
completion tokens. This comparison does not support input-size mismatch as the
explanation for our weak RL result.

## What we will try next

The evidence now lowers the priority of more small variations on the same 32
training questions. The next accepted training experiment uses 128 new news
examples and a separate 256-example evaluation panel. If that provides useful
learning signal, a larger fixed-dose experiment is more informative than an
endless sequence of single-update checks.

We are also testing the root's procedure rather than only helper classification.
An earlier optional-delegation pilot was dominated by incorrect Python-interface
use and repeated errors; only half its planned episodes ran. A small explicit-
interface check must establish a functioning comparison before we interpret
delegation quality. A separate long-text retrieval calibration has a cleaner
verifier, but its small dataset contains target variants of one underlying text:
it cannot establish generalization to new contexts.

## Limits, costs and evidence

These evaluations reuse a small, now research-exposed panel. They do not show
equivalence of the models' output probabilities, genuine transfer from repeated
evaluation, or that larger and more varied RL training cannot help.

The paired update reused 128 saved actions after a software failure before
either update; recovery sampled zero new actions. The failed source attempt
took 288 seconds, recovery training took 1,252 seconds, and each evaluation
took about 112 seconds. The single-item evaluations required 512 actual requests
in total. Model loading,
collection, training, verification and evaluation are separate costs; a short
optimizer step is not the duration of the whole experiment.

The external research store is `/project/alex_phd/runs/rlm-research-r4/`:

- `analyses/paired-feedback-recovery-2026-09-12/REPORT.json`, SHA256
  `d6d2d8d42ab158aeaf990d6632391f760eab5b6c63e5e0c1890fbf469222c308`.
  Both source-action replays, initial states, optimizer steps and served bindings
  were requalified; saved evaluation tokens were decoded again. The legacy
  evaluator did not retain full HTTP envelopes. Its per-arm `fresh_actions` field
  counts retained rows; use the corrected accounting in the final report.
- `analyses/helper-hf-singleton-policy-comparison-2026-09-12/REPORT.json`, SHA256
  `59e5dcc725cb2aeb60d712786ed4e1b513a9b32d6d0a83bb8fbfdb652cd9da22`.
  All 512 saved responses were decoded and matched to fixed requests and model
  bindings. Canonicalized request JSON is not evidence of original wire ordering.
- `analyses/root-recursion-headroom-partial-2026-09-12/PARTIAL.md` documents the
  incomplete delegation pilot and its interface failures; unattempted episodes
  are not counted as wrong answers.

Source/report pushes preserve readable evidence and resume pointers, not the
external model weights or raw run artifacts.
