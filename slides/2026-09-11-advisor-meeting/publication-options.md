---
title: Two realistic publication paths
meeting_date: 2026-09-11
evidence_cutoff: 2026-09-11T03:45:00Z
status: discussion_draft
---

# Option 1 — Keeping each answer attached to the right input

**A possible paper.** When a model answers many small questions together, it can
give a sensible answer for the wrong record. We would study when that happens,
which changes to the input and output reduce it, and whether those changes
improve the final combined answer.

## What is established

- A forced wrong visible-record ID redirected labels toward that other record: displayed-record
  accuracy was 36.5%, while the named record's label was chosen 71.5% on unequal-label positions.
- A 16-context replication found unrelated aliases at 80.47% versus shifted visible IDs at 38.80%,
  a +41.67-point paired difference, positive in all 16 context clusters.
- Matched input/output row numbers repaired late-batch errors on another 16-context panel: the
  prospectively primary interaction was +42.77 points, positive in 16/16 contexts.
- A further 16-batch test found arbitrary matching tags at 84.9%, sequential row
  numbers at 85.4%, and no matching keys at 34.2%. Both arbitrary tags and shuffled
  numbers passed their pre-specified checks; ordinary counting order was not needed.
- On the same exposed 16-context panel, Qwen3-8B scored 31.5% with labels only, 85.5% with
  sequential keys, and 81.1% with opaque keys on late positions. The corresponding 4B values were
  34.2%, 85.4%, and 84.9%. This is a bounded same-family check, not a pure capacity comparison.
- The subsequent Mistral-7B check scored 31.9%, 55.8%, and 52.5% on the same
  panel. All 144 outputs were valid. Arbitrary tags helped every batch and row
  numbers helped fifteen of sixteen. The direction extends to another family,
  but it does not reproduce the Qwen models' absolute accuracy.
- A balanced follow-up put arbitrary tags on both sides in every condition.
  Matching tag sets scored 78.8%, versus 32.6% for disjoint sets, on later labels.
  The task and record order were unchanged; all 192 responses were valid and the
  gain appeared in all 16 batches. This supports literal reuse within the tested
  interface, not just the presence of output tags.
- On 16 newly selected context clusters with nested batches, labels-only accuracy fell from 82.8%
  at 8 records to 43.8% at 64, while sequential and opaque matching keys remained at 84.6% and
  83.4% at 64. The pre-specified onset rule first passed at 32 records, and both keyed formats improved all 16
  contexts at sizes 32, 48, and 64. This is one encoding package, not a universal size limit.
- These are native, strict-contract component results. They do not establish an internal attention
  mechanism, free identifier generation, or a better complete RLM.

## Closest prior art and the narrow contribution

[Batch Prompting](https://aclanthology.org/2023.emnlp-industry.74/) already uses position identifiers,
and [BatchPrompt](https://openreview.net/forum?id=Agyicd577r) studies position/permutation effects.
[Cascaded Batch Prompting](https://arxiv.org/abs/2608.27038) separates semantic class generation from
symbol mapping. [Entity-binding work](https://arxiv.org/abs/2310.17191) studies internal binding with
causal interventions, which our behavioral experiments do not reproduce.

The defensible contribution is therefore not “indexes help.” It is a controlled behavioral account
that separates wrong-record semantic capture, unrelated output diversity, output-only position,
matched input/output position, and downstream usability under strict structured generation.

A further prior-art check reinforces that boundary. Input-dependent output rules
already improved task scores in
[Geng et al.](https://aclanthology.org/2023.emnlp-main.674/), while
[CRANE](https://arxiv.org/abs/2502.09061v4) and
[The Hidden Cost of Structure](https://aclanthology.org/2025.ranlp-1.124/) show why
restrictions can also hurt. We should not claim that output rules improving
accuracy is itself new. Our strongest observation is the large, position-specific
label difference among outputs that all satisfy their required structure.
Separating visible input tags from the rules and generated output history remains
future work. This literature check does not change the experimental evidence cutoff.

## What is missing and the decisive next step

The matching-key comparison is complete on two Qwen models and one Mistral model,
supporting arbitrary tags, not just sequence numbering, on the shared exposed panel. The next step is a whole-task
comparison: does that improvement survive when a main model uses the labels to calculate an answer?
Compare identical inputs with and without matching tags, first with a supplied
plan and then with the main model choosing its steps. Score label accuracy,
whether the model uses the labels, the requested calculation, the final answer,
and cost separately. The new batch-size study adds fresh input sets for one model;
fresh sets for the other models and a different task would extend that evidence.
Tag length and forced answer order remain unresolved controls.

The new balanced comparison makes literal reuse a stronger candidate explanation
than the mere presence of tags. It still changes the input/output text package.
The same-prompt output-constraint comparison is now complete: enforcing the format
reduced malformed responses, but it does not explain the matching-tag comparison
where every response was already well formed. The [supporting note](later-findings.md)
separates these effects. A further comparison should separate visible input tags
from the forced output tags and extra generated text. Conditioning only on valid
responses could hide the practical cost of unconstrained generation.

**Path:** 1–3 days for a first whole-task comparison and interpretation; 2–3 weeks for a workshop
paper with another model or task family; 4–6 weeks for a stronger submission with independent
contexts, model/task replication, clustered uncertainty, and end-to-end utility.

# Option 2 — Teaching tool use that improves the final answer

**A possible paper.** Examples can teach a small main model or helper to use its
tools and response formats better. We would ask which improvements actually
survive when smaller answers are combined, and develop tests that distinguish
genuine success from a correct number produced by the wrong calculation.

## What is established

- Mixed-contract child training improved a compact A/B/other interface without sacrificing the
  original six-class interface, unlike single-contract specialization.
- One question-sensitive root SFT checkpoint showed large gains across related readouts, including
  nonzero composed answers.
- A separately captured 72-trajectory corpus, trained from the same already-trained fixed24 adapter
  with fresh Adam and six updates, produced 55/72 correct-and-performed answers on the shared
  metadata panel. Two attempted endpoints were unavailable, giving bounds of 55–57/72; the original
  corpus produced 50/72 and the fixed24 start 12/72.
- On eight newly selected root input groups, faithful-and-correct answers were 12/72 for fixed24
  (bounds 12–19), 55/72 for original-corpus SFT6, and 53/72 for new-corpus SFT6 (bounds 53–55).
  Both trained policies improved every context cluster and achieved 35/48 and 34/48 composed
  successes versus zero for fixed24. The task families and child remain research-exposed.
- Supplying the entire plan and a deterministic reducer still gave 0/8 exact answers with the
  original child despite 88.2% record-label accuracy; oracle labels gave 8/8. A mixed child improved
  only one of eight exact answers. Component improvement is not monotonic system improvement.
- Terminal-only high-LR RL is a clear negative for this exact recipe: after four further genuine
  updates, checkpoint 6 scored 47/72 versus 55/72 at the unchanged start. On composed questions,
  only 26/45 available endpoints were both semantically faithful and strict-correct; four additional
  strict answers were zero-answer coincidences from wrong computations.
- A restart with an update size five times smaller also made six updates and
  showed no clear gain: 52/72 correct overall, possible totals 52–56, versus
  starting totals 55–56. The primary combined-operation comparison tied at
  36/48 correct and 33/48 faithful-and-correct. This discourages another
  learning-rate-only repeat at the same small dose, not reward learning generally.

## Closest prior art and the narrow contribution

[Recursive Language Models](https://arxiv.org/abs/2512.24601) establishes recursive inference and
released transition training; [λ-RLM](https://arxiv.org/abs/2603.20105) motivates typed functional
decomposition; [SRLM](https://arxiv.org/abs/2603.15653) emphasizes program selection. Structured-
output compliance itself is covered by [JSONSchemaBench](https://arxiv.org/abs/2501.10868).

[Steno's cost-aware RLM thesis](https://essay.utwente.nl/essays/110199) is particularly
close: it trains a Qwen3-4B root with Prime-RL and LoRA and reports improved
long-context performance. Our [existing source review](/project/alex_phd/runs/rlm-research-r4/analyses/steno-cost-aware-rlm-question-card-2026-09-10/QUESTION_CARD.md)
documents its much larger training setup and different objective. Training a
small RLM with reward feedback is therefore not, by itself, our novel contribution.
Our negative reward runs do not contradict success with that different recipe.
This added literature context does not change the deck's numerical cutoff.

Our plausible contribution is an empirical training-and-measurement result: internal communication
contracts are learned behaviors; local interface gains can fail at downstream composition; and
terminal accuracy alone overstates progress unless acquisition, operator/scope faithfulness,
stopping, NULLs, and physical cost are audited separately.

## What is missing and the decisive next step

The separate-corpus training run and one newly selected root-input evaluation are complete. The
largest gaps are genuinely new task combinations or a second task, more training realizations, and
evaluation beyond named-inventory newness. The new groups are not guaranteed absent from pretraining
or child training.
Promotion should require performed, nonzero composed answers, not lower training loss or correct
numbers from the wrong computation. The smaller-update reward run now shows no
clear gain either. We should next test whether judging intermediate work
helps. Such a test should reward useful work, not merely making more helper calls.

**Path:** the first newly selected evaluation is complete; allow several days to design and
test a new combination-of-calculations evaluation, then 1–2 weeks for another training
realization plus a second task; 4–8 weeks for a conference-ready comparison of interface SFT, terminal reward, and an
execution-grounded alternative with matched held-out evaluation and cost accounting.

## Evidence added at this cutoff

All paths are under `/project/alex_phd/runs/rlm-research-r4/`.

- S2 report: `analyses/root-question-sensitive-sft-new-corpus-live-2026-09-10/REPORT_RECOVERY_V2.md`,
  SHA-256 `c7e41fadecdb6a46c71ef7b40878058de2cd4f2f2e999babf518c79baefe5a43`; its
  starting-policy/claim erratum is SHA-256
  `7974e7c0ff50ae3ad4dc357b05765445404f9144ff8112e335d8721cb91ba27d`.
- H4 report: `analyses/leaf-mnli-stable-anchor-qwen8b-live-2026-09-10/REPORT.md`, SHA-256
  `e7ba139e12ef0aa0f490380085f1d31a2488172e6599a78536dbd0ccec5caa68`; its final
  audit seal is SHA-256 `b619cb1d4aa79bfa7aa0d292a9e909d640c1efa421b0c0708bd2386bfed0278e`.
- H5 report: `analyses/leaf-mnli-stable-anchor-mistral7b-live-2026-09-11/REPORT.md`,
  SHA-256 `d5c48a217eb3f292397fa2d337fb2cc97f005d0243b0109dec0ccbcf8f437470`.
- R2 report: `analyses/root-question-sensitive-terminal-rlvr-lr1e5-live-2026-09-10/REPORT.md`,
  SHA-256 `d6901670d6889b9469ba17ab51f39de19166bf013e96a620415a0e0ba9a17df4`.
- H6 report: `analyses/leaf-mnli-balanced-tag-match-live-2026-09-11/REPORT_ATTEMPT002.md`,
  SHA-256 `c245e4f0f14b194889837c3cc8d5ccfd313d40d2d6e48f94362c635858cbdabf`.
- S3 report: `analyses/root-question-sensitive-fresh-input-three-policy-live-2026-09-11/REPORT.md`,
  SHA-256 `df5db6c9fe798bebe9e4606a3d6ee7ba273d622078e7b8f1351febeb501547e8`;
  its final seal is `9512db664dbe6fdf578ab477aebf8f33ab1808f8bfec1692d8704ffe71c2ae78`.
- H8 report: `analyses/leaf-mnli-nested-batch-matching-live-2026-09-11/REPORT.md`, SHA-256
  `300f66b7ceac1ed3e89a32c2f44008eb201ca958f5b8b2ba00bd9927068c64ef`;
  its final seal is `87e7552988bb16ddb6588b6a814e1f0b3c8c84061f39db50ef12f2f9d7c8ccbb`.

# Recommendation for the meeting

These are work estimates conditional on useful results and continued compute
access, not promises of publication or acceptance at a particular venue.

Lead with Option 1 because the matching effect appears across several controlled interfaces and
three models from two families, and it is visually explainable. The Mistral check repeats the
direction with lower absolute accuracy. Its largest remaining gap is whether it
helps the final task on fresh evaluation inputs. Present Option 2 as the broader systems story and
publication trajectory, with the explicit status: a supervised result observed after training on
two separately captured corpora and then on one newly selected evaluation panel, informative composition limits,
and no benefit from two small reward-training recipes—not yet a general training theorem or new RL algorithm.
