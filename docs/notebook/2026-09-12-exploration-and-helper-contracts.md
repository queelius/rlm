---
date: 2026-09-12
cutoff_utc: "10:08"
status: exploratory_completed_results
questions:
  - Is weak RL improvement mainly caused by insufficient answer variety?
  - Does asking a helper a simpler yes/no question improve its usefulness?
research_store: /project/alex_phd/runs/rlm-research-r4
---

# More exploration did not improve answers; simpler helper requests were worse

Two completed comparisons narrow our next decisions. Increasing answer variety
during RL gave the optimizer more contrasting examples, but did not change the
final evaluation answers. Separately, asking our trained helper for yes/no answers
was much less reliable than asking for its familiar category labels.

Neither result means RL or task decomposition cannot work. They tell us which
simple changes are not enough in the current setup.

## 1. More varied training answers did not produce better evaluation answers

We started both runs from the same supervised-trained 4B helper. Each made four
RL updates, using four newly sampled answers to each of the same 32 training
questions per update. We held the learning rate and seed schedule fixed and
changed sampling temperature from 1 to 2. Higher temperature encourages less
predictable answers.

Our reward comparison is useful when attempts at the same question include both
right and wrong answers. When all four have the same reward, this comparison
provides no learning signal for that question.

| Measurement | Reference sampling | Higher exploration |
|---|---:|---:|
| Groups with both right and wrong answers, across 128 question/update groups | 16 | 40 |
| Correct question categories after training | 120/128 | 120/128 |
| Correct news categories after training | 112/128 | 112/128 |
| Training time | 42.0 minutes | 51.0 minutes |

The models did not merely tie in accuracy: **all 256 predicted labels were
identical**. Before these updates, the supervised helper scored 119/128 question
categories and 112/128 news categories. We still do not have a meaningful RL gain.

This weakens the explanation that answer variety alone is the bottleneck at
this learning rate and training scope. It does not show that the weights or
probability distributions were unchanged, or rule out benefits from exploration
with different training material or larger updates. All four updates completed
and their checkpoints were saved; evaluation used the preselected fourth step.

The next comparisons test a larger learning rate and a reward baseline that can
give negative feedback even when every attempt at a question is wrong. A broader
training-data proposal is also being prepared: the current 32 questions all
appeared in the helper's earlier supervised training.

## 2. Asking only for yes/no was not an easier task for this helper

Suppose we want to count questions asking about a person. We can ask the helper
to classify every question into its familiar categories, then count the person
labels in Python. Alternatively, we can ask only whether each question belongs
to the person category, returning yes or no, then count the yes answers.

We compared these approaches on 128 question records in eight groups. Each
group had six category-count questions, giving 48 related count tasks. The same
fixed helper answered all requests; no training occurred in this experiment.

| Measurement | Familiar full categories | Targeted yes/no |
|---|---:|---:|
| Exactly correct counts | 31/48 | 20/48 |
| Sum of absolute count errors | 18 | 182 |
| Input plus output tokens for all six targets | 12,260 | 75,811 |

Yes/no corrected three counts but broke fourteen that full-category outputs got
right. All 56 physical requests returned valid answers, so this is not a formatting
or missing-result explanation. The largest problems were systematic overcounting,
not occasional arithmetic slips: Python performed the same exact counting step.

The cost comparison depends on what is requested. A full-category map can answer
all six count questions without another model call. Asking six separate yes/no
questions used 6.18 times its tokens. For just one uniformly chosen target, the
average yes/no request used only 1.03 times its tokens—but was still less reliable.
Neither calculation charges the reusable full map six times.

An earlier literal-category/other version also performed poorly. Replacing those
output words with yes/no did not rescue the overall result. We will retain the
familiar full-category interface for this helper and deprioritize further simple
binary rewrites. Training a helper specifically for a different contract would
be a separate experiment; this one does not establish that compact outputs are
intrinsically bad.

## What now looks worth pursuing

- **RL:** distinguish update size, reward comparison, and training coverage.
  Do not respond to the temperature result with more temperature-only runs.
- **Adaptive decomposition:** the queued fresh-data test asks whether disagreement
  between two groupings identifies worthwhile individual checks. It must compete
  with ordinary majority voting, not just a single weak grouping.
- **RLM decisions:** prepare a matched comparison with and without optional helper
  calls. Measure whether there is a useful task-dependent choice before training
  a delegation policy. Helper classification alone is not learned recursion.

The strongest publication direction remains the interaction between a model and
the information passed through its harness. These new results help identify
limits; they are not standalone claims of a new RL algorithm or generalization.
The current evaluation records have been repeatedly examined during research.
Fresh examples, independent repetitions and additional task families are needed
before stronger claims.

## Evidence and resume pointers

All paths below are relative to the external research store named in the header.
GitHub preserves this account, not the external model weights or raw run files.

- T2 raw-token analysis:
  `analyses/helper-hf-fixed256-perturbations-2026-09-12/arms/t2_lr1e5_4step.json`,
  SHA256 `54d66b8e6df95eccc09c24d03a76f243a466df09979e5be22ea5f2478b3162f9`.
- Yes/no independent analysis:
  `analyses/helper-trec-yesno-local-review-2026-09-12/REPORT.json`,
  SHA256 `09144b71897046bb535ab8bb42e04c92cd0aa419e1556f4d39228309286309e0`.
- Live GPU owners, accepted successors, caps and resume paths: `RESEARCH_QUEUE.md`.
- Running interpretation and next decisions: `analyses/CURRENT_SUMMARY.md` and
  `questions/rl-effective-feedback.md`.

The historical advisor deck is unchanged. This report follows the
[grouping results](2026-09-12-grouping-and-fast-rollout-qualification.md) and the
[reference four-update result](2026-09-12-decomposition-transfer-and-four-step-rl.md).
