---
date: 2026-09-12
cutoff_utc: "09:15"
status: exploratory_completed_results
questions:
  - Does grouping change a helper's answers even when group size stays fixed?
  - Can faster generated samples support a controlled RL update?
research_store: /project/alex_phd/runs/rlm-research-r4
---

# Grouping changes answers; the faster RL path passes its first check

The new grouping experiment makes the decomposition question more concrete.
The same model can classify the same item differently depending on which other
items share its request. Simply choosing a smaller group is therefore not the
whole problem: the contents and order of a group can matter too.

Separately, a fresh batch from the faster generation system passed our original
probability checks. This opens a path toward more efficient RL experiments.
It is not yet evidence of a useful weight update or better answers.

## The grouping experiment

We kept the supervised-trained 4B helper, category definitions and group size
fixed. Every request contained sixteen records. Each of the same 128 question
records and 128 news records appeared in four arrangements:

1. Keep the original groups and order.
2. Reverse the order within each original group.
3. Change the other records in each group, keeping each item's numbered slot.
4. Repeat that regrouping with a second fixed random arrangement.

The regroupings did not use the correct labels. All 64 requests returned valid
answers. These are 256 unique records, not 1,024 independent test cases.

| Correct records | Original groups | Reversed order | Regrouping A | Regrouping B |
|---|---:|---:|---:|---:|
| Question categories | 117/128 | 118/128 | 118/128 | 113/128 |
| News categories | 115/128 | 111/128 | 113/128 | 109/128 |

Across the four arrangements, **22 question records and 29 news records changed
their predicted category at least once**. Changes included both corrections and
new mistakes. For example, the question asking who invented the telephone was
classified as asking for a description in the original arrangement, but as
asking for a person in the other three arrangements. Its benchmark label is person.

The two regroupings preserved each record's numbered slot, but not its absolute
token position or all preceding text. Reversal changed order and position together.
These controls establish sensitivity to presentation on this panel; they do not
identify an attention mechanism or show that random regrouping improves accuracy.

All four arrangements used the same total input-token count within each dataset.
The experiment consumed 117,225 input-plus-output tokens and about 10.9 minutes,
including startup. The selected serving configuration was checked in the running
engine, and the GPU was released normally afterward.

## What this changes about the next experiment

The [earlier size comparison](2026-09-12-decomposition-transfer-and-four-step-rl.md)
found that individual requests helped question classification slightly but hurt
news classification, while using about seven times as many tokens as groups of
sixteen. The new result adds a second dimension: **which items are grouped together**.

A useful next question is whether disagreement between two inexpensive groupings
can identify items that deserve more work. That requires testing both the detector
and the proposed repair: agreement can be consistently wrong, and an individual
request can make a previously correct answer worse. An offline replay is being
examined first; no adaptive-harness improvement is claimed here.

## Faster RL: what passed and what has not happened yet

We generated 48 new helper answers from the original supervised checkpoint:
24 attempts on each of two familiar training contexts, with sixteen category
decisions in each answer. We then compared their saved generation probabilities
with the probabilities assigned by the training implementation.

All 48 answers passed the original finite-probability and support checks. The
importance-weight effective sample size was **45.45 out of 48**, above the
predeclared minimum of 38.4. No answer dominated the weights: the largest
normalized share was 2.78%, below the 10% limit. Raw sequence ratios ranged from
0.1904 to 1.3013; the two implementations are not numerically identical.

The samples were collected successfully before a missing-package error stopped
the scoring stage. We recovered by scoring the saved samples with precomputed
grammar masks; we did not generate a replacement batch or modify the shared
training environment. This recovery took 22.7 seconds and made **zero optimizer
updates**. A separately specified one-update experiment is being prepared, with
another check on the actual gradient-computation path before changing weights.

This full-map batch differs from the ongoing single-question RL comparisons in
input grouping, temperature and reward. Their results must not be pooled as if
they were repeated runs of the same experiment.

## Boundaries and evidence

The evaluation records are absent from the verified supervised helper inputs and
the current 32-question RL set. They have now been examined repeatedly in research;
they are not a pristine confirmation set, and base-model pretraining exposure is
unknown. No new root-planning, recursive-stopping or cross-model result is implied.
The delivered September 11 slides retain their historical meeting cutoff.

Authoritative artifacts are in the external research store named above:

| Artifact | Relative path | SHA-256 |
|---|---|---|
| Grouping result | `sidecars/helper-companion-position-v1/outputs/attempt-001/RESULT.json` | `3747f55b59ee93ae0b87535ff71b163ec362e1e6b29a46ebc86219adbb06621b` |
| Independent raw-response audit | `analyses/helper-companion-position-local-review-2026-09-12/REPORT.json` | `9bc764d70b772432bc4b20a3a77b7f6eee74e558e6326ef084c5f90321a0a7d8` |
| Independent fresh-sample probability audit | `analyses/fresh-batch-invariant-hf-recovery-2026-09-12/REPORT.json` | `4b8e6f19f9a6bc0deb741447af0cd0dc04f7a554737472210d8e0e5ccce36df1` |

The grouping audit checks every saved native response against its request, ordered
schema, decoded labels, usage record and benchmark answer. MAIN independently
recomputed the accuracy and instability totals, and separately recalculated the
48 importance ratios and effective sample size. This Git checkpoint preserves
the report, not the model weights or complete external run directory.
