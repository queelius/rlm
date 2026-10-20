---
date: 2026-09-12
cutoff_utc: "2026-09-12T05:08:00Z"
question: rq:rl-effective-feedback
status: three_arm_evaluation_complete
evidence_level: exploratory_previously_exposed_panel
---

# Learning from successful examples looks more promising than this small RL update

We completed the first matched comparison in the new experiment. Training the
root model on its own successful attempts gave the best observed result. One
root-only reinforcement-learning update did not improve the overall score.
This is a lead to investigate, not a general conclusion that supervised training
is better than reinforcement learning.

## What did we change?

The root model writes Python, asks a helper model to classify records, and uses
the replies to answer questions about those records. We made two copies of the
same previously trained root. One learned from whether its answers succeeded or
failed (RL). The other copied its own successful attempts (self-SFT). Both took
one optimizer step. The helper model's weights stayed unchanged.

Both updates used a collection of 48 attempts on 12 training questions from two
source contexts. RL used 20 attempts from five groups containing both successes
and failures. Self-SFT used 26 successful attempts from ten questions. They share
the collection pool, but **not the same selected examples or training-token dose**.
Success means a correct final answer; we have not established that every copied
attempt used a fully correct method.

## What happened on the same 72-question evaluation?

| Root model | Correct answers | Wrong answers | Unavailable answers | Physical model requests |
|---|---:|---:|---:|---:|
| Unchanged | 53 | 17 | 2 | 311 |
| After one RL update | 52 | 19 | 1 | 318 |
| After one self-SFT update | 57 | 15 | 0 | 295 |

The evaluation uses eight source contexts with nine questions each. Questions
from one context are related, so these are not 72 independent documents. All
three conditions use matched questions, sampling seeds, helper weights and
runtime settings. These evaluation contexts have been inspected in earlier
research; they are not a pristine confirmation set.
All four additional correct/available outcomes under self-SFT are concentrated
in two of the eight contexts, rather than a broad improvement across sources.

Among questions with an observed answer in both conditions:

- Self-SFT corrected two baseline mistakes and lost no correct answers. It also
  answered both questions whose baseline answers were unavailable.
- RL corrected two baseline mistakes but lost four correct answers. It also
  answered one question whose baseline attempt was unavailable.

The baseline had one timeout and one rejected overlength prompt. RL had one
overlength failure. Self-SFT had neither. Missing answers remain a separate
category; they are not silently counted as wrong answers. No export-integrity
failures were found in these three evaluations.

Self-SFT also used fewer physical model requests and fewer returned prompt tokens
(409,218 versus 501,646 for the baseline). However, avoiding one long retry chain
accounts for 88.7% of that token difference. Excluding that attempt, prompt usage
fell only 2.52%, and both conditions made exactly 220 returned root calls. This is
not evidence of a general 18% efficiency improvement. Summing only saved episodes
would also miss three baseline requests from a timed-out attempt; the table
includes them.

## Why does this change the next experiment?

The [helper-error analysis](2026-09-12-helper-feedback.md) suggests that many
wrong answers are consistent with the root calculating correctly from wrong
helper labels. We therefore should not assume that more root-only training will
solve the main problem.

We are now testing the unchanged root with correct local helper replies, followed
by a control that replays the original replies. The correct-label intervention
is deliberately nondeployable: it diagnoses the bottleneck, not a new working
system. A helper-only RL update is prepared separately, using local label
correctness as its reward. The root will stay unchanged in that comparison.

The trace audit also found two baseline/RL pairs where identical recorded helper
requests and sampling settings produced different replies. We checked the full
request bodies, seeds and fixed helper binding. This is a reproducibility caveat,
not a diagnosis of the backend or evidence that RL is invalid. The replay control
holds returned helper content fixed and will make the interpretation clearer.

The strongest follow-up to the self-SFT result is a repeat with a new seed schedule
and then new source contexts. A small gain on an already-inspected panel is not
enough to claim generalization, reliable RL improvement, or a publishable method.

## Evidence and reproducibility

All external paths below are relative to
`/project/alex_phd/runs/rlm-research-r4`:

- `analyses/qs6-onebatch-learning-2026-09-12/REPORT.md`, `RESULTS.json`, and
  `analyze.py` contain the complete three-arm paired analysis. MAIN executed the
  analyzer after all three owners had completed and released the GPU.
- `sidecars/root-qs6-matched-eval-v1/outputs/{baseline-001,rl-001,sft-001}/`
  preserve the exports, physical request audits and terminal receipts.
- Exported episode SHA-256 values are
  `dd77ffbc4d33c685c2d190aaaf91085a0d03d10338b260d53e2eacbd3e568358`
  (baseline),
  `53749fbd9da9409f06be8ff34f1a1208cf91aa434f1e68d46517fa53907a1fdb`
  (RL), and
  `7df9a987a0d195f0ab91fa6f2a12fee5339e44e6afa40d37180b724527dd13ad`
  (self-SFT).
- `analyses/qs6-onebatch-learning-2026-09-12/discordance-audit/` contains the
  root-program examples and the matched-input/different-helper-output audit.
- `RESEARCH_QUEUE.md` identifies live owners and automatic successors. Read it
  before launching work; this document is a completed-result snapshot.

Editorial decision: preserve the September 11 meeting deck at its delivered
cutoff for now. This report adds the new result without silently changing what
was presented. The helper intervention and a repeat evaluation will determine
the clearest next presentation update. Git backs up this report, not the external
model checkpoints or complete raw run store.
