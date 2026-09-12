---
date: 2026-09-12
cutoff_utc: "2026-09-12T06:27:00Z"
question: rq:rl-effective-feedback
status: exploratory_helper_size_comparison_complete
evidence_level: repeated_inputs_seen_in_prior_helper_sft
---

# Smaller helper requests can help, but we must test on new material

We changed one simple part of the RLM: how many records the helper processes in
one request. The model weights stayed fixed. Smaller requests improved the total
number of correct local categories on two panels, but sometimes made individual
answers worse and required substantially more input text. We have not yet shown
that this improves the final answers produced by the whole RLM.

There is an important limitation: **all 160 questions in these panels had already
been used to train this helper.** These experiments test behavior on familiar
material. They do not establish generalization to unseen questions.

## What did we change?

The helper assigns categories to short questions. For example, a question asking
how many bones are in a human hand should receive the category for a numerical
answer. The root model then uses these categories in a larger calculation.

We sent the same records in groups of 16, groups of four, or individually. Each
condition was repeated twice, with the same inputs and paired condition-level
seeds. The schema, category definitions, helper weights and temperature were fixed.
Requests were issued serially, with the condition order rotated across contexts.

| Panel | Sixteen records per request | Four per request | One per request |
|---|---:|---:|---:|
| 32 questions in two development contexts, each repeated twice | 54/64 correct | 61/64 | 58/64 |
| 128 questions in eight other contexts, each repeated twice | 239/256 correct | 243/256 | 245/256 |

These are repeated predictions, not 64 or 256 independent source questions.
The broader panel was chosen as an exploratory follow-up after seeing the first
result. It was already research-exposed; its local name, “protected,” must not be
read as “unseen by the helper during training.”

In the broader panel, groups of four corrected 11 predictions but spoiled seven
that groups of 16 had answered correctly. Individual requests corrected 13 and
spoiled seven. The effect varied across contexts. Groups of four were best on the
first panel; individual requests were best on the broader one.

## What did it cost?

| Broader panel, two repeats | Sixteen per request | Four per request | One per request |
|---|---:|---:|---:|
| Helper calls | 16 | 64 | 256 |
| Input tokens | 19,820 | 57,404 | 207,740 |
| Output tokens | 4,057 | 4,164 | 4,631 |
| Sum of measured request times | 46.9 seconds | 48.5 seconds | 59.3 seconds |

Compared with groups of 16, groups of four used 2.58 times as many total tokens;
individual requests used 8.89 times as many. Prefix caching reused much of the
repeated input. Consequently, token counts and measured time tell different
parts of the story. These times exclude server startup and are not whole-RLM
latencies or estimates of monetary cost.

## The training-exposure audit changes what we can claim

We checked the actual saved helper-training history, not just dataset split names.
The checkpoint records match two full passes through 5,065 unique training IDs.
Every one of the 32 development questions and 128 broader-panel questions occurs
twice in that optimizer history. The 128 evaluation questions are disjoint from
the 32 questions used by the new RL pilot, but that does not undo the helper's
earlier supervised exposure to both sets.

This does not erase the measured comparisons. It changes their interpretation:
even on previously trained examples, the way we divide the input can change
the helper's answers. A publication-quality generalization claim needs genuinely
new examples and another task or dataset. We are preparing those checks next.

## What follows from this?

The strongest research question is not “Should every request be smaller?” It is:
**Can the system choose when a smaller request or an additional check is worth
its cost?** First we need to establish that better local answers improve the final
task outcome. A prepared comparison will feed the root the actual saved helper
answers from each size condition, without substituting correct labels. This
isolates the consequence of those answers; it is not a deployable runtime test.

A fresh helper-RL pilot is also running at this cutoff. It samples new answers
and computes the learning probabilities in the same model implementation, using
local correctness as the reward. It has not yet produced a completed update or
an answer-quality result. The older helper-RL continuation remains on hold because
its logged probabilities require further investigation; we have not relaxed that
problem away.

## Evidence and resume pointers

All external paths below are relative to
`/project/alex_phd/runs/rlm-research-r4`.

- Two-context run: `sidecars/root-c32-helper-batchsize-v1/outputs/attempt-001`.
  All 84 requests returned valid outputs. Independent analysis:
  `analyses/c32-helper-batchsize-2026-09-12/REPORT.json`, SHA-256
  `5cebbe09f1b338b2bc10d38e0854c3c1f16879b40537f7efbfeb21fc1a903ea1`.
- Eight-context run: `sidecars/root-c32-helper-batchsize-protected-v1/outputs/attempt-001`.
  All 336 requests returned valid outputs. Independent analysis:
  `analyses/c32-helper-batchsize-protected-2026-09-12/REPORT.json`, SHA-256
  `0e03a39acada3f730aba118e9bde40aa1a3c71375026aed92aa2df57dc5f96f3`.
- Actual training exposure: `sidecars/helper-hf-onpolicy-v1/PRIOR_SFT_EXPOSURE.json`,
  SHA-256 `116e83cbed96717ba98e060365d859679c5346463e694ddab27dd9c6119751a8`.
  This pins the training inputs, checkpoint, adapter, and per-step group history.
- Active fresh-RL pilot: `sidecars/helper-hf-onpolicy-v1/outputs/attempt-001`;
  READY SHA-256 `ed56090984c257e4c555be4cabd40388beae7ea3b1f2dd04d2b33033159c7661`.
- Prepared downstream comparison:
  `sidecars/root-qs6-batchsize-map-downstream-v1/READY.json`, SHA-256
  `45da8b2e4fccd931b4ed4086ceb0e996ab365b34196b31f28f4f026737d7a955`.

The delivered September 11 slides retain their historical cutoff. Before reusing
any earlier claim of generalization, check which model saw which examples. This
report is preserved in Git; the external model weights and raw runs are not.
