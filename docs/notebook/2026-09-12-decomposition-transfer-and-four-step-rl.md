---
date: 2026-09-12
cutoff_utc: "2026-09-12T08:32:00Z"
questions: ["rq:rl-effective-feedback", "rq:adaptive-decomposition"]
status: completed_exploratory_comparisons
evidence_level: one_model_two_related_classification_datasets
---

# More decomposition is not always better; more small RL updates did not help yet

Two completed experiments sharpen our next research decisions. First, splitting
the same work into smaller helper requests helped slightly on question categories,
but hurt on news categories. Second, four reinforcement-learning updates produced
the same evaluation answers as our earlier one-update pilot. These are useful
boundaries, not evidence that we have already learned a general decomposition policy.

## The best request size changed with the task

We asked the same supervised-trained 4-billion-parameter helper to categorize
128 questions and 128 news items. Each record was presented in a group of sixteen,
a group of four, and individually. All three conditions used the same model,
category definitions, deterministic sampling setting, and serving process.

| Records in each helper request | Questions correct / 128 | News items correct / 128 | Total reported tokens, both datasets |
|---|---:|---:|---:|
| 16 | 117 | 115 | 29,302 |
| 4 | 119 | 112 | 64,959 |
| 1 | 120 | 109 | 207,621 |

Individual requests corrected four question answers and spoiled one relative to
groups of sixteen. On news, they corrected two and spoiled eight. The news decline
was concentrated in science/technology, though other categories also changed.
Every request returned a valid answer: this is a difference in category decisions,
not missing output or broken formatting.

The smallest requests used about **7.1 times as many reported tokens** overall.
Repeated instructions contribute to that cost. Some input tokens were cached,
so token counts are not equivalent to a bill or GPU time. The complete 336-call
experiment took 527 seconds, including service startup and cleanup. Its 768
prediction slots represent three treatments of the same 256 records, not 768
independent examples.

This revises the interpretation of our earlier, favorable smaller-request results
on familiar question data. A fixed rule to split more is not supported across
these two tasks. The more interesting question is **how to recognize when a
smaller request is worth its cost**, and whether useful neighboring examples
sometimes disappear when we split. That last explanation is a hypothesis, not
a demonstrated mechanism. A regrouping experiment can distinguish request-size
effects from dependence on which other records appear alongside an item.

## Four RL updates gave no additional evaluation improvement

Starting again from the supervised-trained helper, we ran four updates with
freshly sampled answers at every step. This was a new run, not a continuation
of the earlier one-step pilot. One optimizer was carried across all four steps,
and the fourth checkpoint was chosen for evaluation before training began.

| Model condition | Questions correct / 128 | News items correct / 128 |
|---|---:|---:|
| Supervised-trained helper | 119 | 112 |
| Earlier one-update RL pilot | 120 | 112 |
| New four-update RL run | 120 | 112 |

The two RL models produced **identical predicted categories on all 256 records**.
Both corrected the same single question relative to the supervised helper.
Their training random seeds and samples differed, so this is not an isolated
comparison in which only the number of updates changed. It is evidence that
this small additional training effort did not improve this reused evaluation.
There were no missing or invalid evaluation replies.

Training used 32 question-category examples and four sampled answers per question
per step: 512 sampled answers across four steps. A correct category earned one
point; an incorrect category earned zero. Each answer was compared with the
other three answers to the same question. Only 4, 5, 4 and 3 of the 32 questions
respectively produced both correct and incorrect samples. When every sampled
answer has the same reward, this relative-reward objective has no learning
contrast for that question—even if every answer is wrong.

For example, one training question asks what is commonly considered the fifth
sense. Its fixed dataset category is “description and abstract concept,” but
all sixteen answers sampled across the four collections chose “entity.” That
question consequently supplied no direct relative-reward gradient in any step.
We have not changed its dataset label; the example also suggests examining
ambiguous category boundaries separately from learning mechanics.

All four updates passed their sampling-versus-replay probability checks, changed
the adapter weights, and saved checkpoint, optimizer and random-generator state.
Training took 2,523 seconds, about 42 minutes; the subsequent 256-example evaluation
took another 120 seconds. The training sample scores are not final-model accuracy.

The next learning experiment should improve the amount of informative feedback,
rather than merely repeat easy questions for longer. Candidates include broader
training questions and a controlled increase in answer diversity. Any selection
must use training data, not evaluation errors. RL is operational here, but a useful
quality improvement is still unestablished.

## A concrete harness modification is running

We are testing whether the helper should answer only the distinction needed by
the final task. For example, to count sports articles, must it classify every
article into every possible category, or is “sports” versus “other” enough?
The experiment asks every predeclared category-count question and compares both
methods against the same correct counts. It also measures the cost of one target
question versus answering many questions from one reusable full-category map.
It has launched, but has no result at this report's cutoff.

## Boundaries and evidence

The question and news panels are absent from the helper's verified fine-tuning
inputs and these 32 RL training questions. They may have appeared in base-model
pretraining, and we have now reused them in exploratory research. They are not
a pristine confirmatory holdout. Both tasks are classification tasks; neither
establishes general recursive planning or improved final RLM task performance.

The size experiment used one newly attested batch-invariant serving process.
Older default-serving results are not pooled into it. The RL table uses its
own matched four-record evaluation comparisons, not the size experiment as a
replacement control.

External store: `/project/alex_phd/runs/rlm-research-r4`.

- Size run: `sidecars/helper-unseen-size-comparison-v1/outputs/attempt-001`.
  Independent audit: `analyses/helper-unseen-size-transfer-2026-09-12/REPORT.json`,
  SHA-256 `792f7d393d43db617e13836bc6f670e9649143434ed67a04b809cea0f1271413`.
  It checks all 336 raw responses, record identities, matched effects, token
  inventory and actual batch-invariant service evidence.
- Four-update run: `sidecars/helper-hf-onpolicy-fourstep-v1/outputs/attempt-001`.
  Final state SHA-256:
  `4286d0a1143ea916f1bec8f43606ebecc5b4006709a9246f9b614435f9f160fe`.
- Four-update evaluation audit:
  `analyses/helper-hf-fourstep-generalization-2026-09-12/REPORT.json`, SHA-256
  `67d6d7e6533eabe72e0fc58f78bc0980c8c3d4f77ddfacffb437f46f67b47fc6`.
  MAIN also independently decoded the new evaluation and compared all 256
  predictions with the saved one-update pilot.
- Background, original-model control and panel provenance:
  [first helper-RL readout](2026-09-12-helper-rl-first-readout.md).

The September 11 meeting slides retain their historical cutoff. This report is
a post-meeting update. GitHub preserves this document, not the external weights
or complete run store.
