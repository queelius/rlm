---
title: Two realistic publication paths
meeting_date: 2026-09-11
evidence_cutoff: 2026-09-11T00:15:00Z
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

## What is missing and the decisive next step

The matching-key comparison is complete on two Qwen3 sizes and supports arbitrary tags, not just
sequence numbering, on the shared exposed panel. The immediate next step is a whole-task
comparison: does that improvement survive when a main model uses the labels to calculate an answer?
Compare identical inputs with and without matching tags, first with a supplied
plan and then with the main model choosing its steps. Score label accuracy,
whether the model uses the labels, the requested calculation, the final answer,
and cost separately. Fresh evaluation contexts and a different task or model family remain
necessary to test the scope of the result. Tag length and forced answer order remain unresolved
controls.

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
- Supplying the entire plan and a deterministic reducer still gave 0/8 exact answers with the
  original child despite 88.2% record-label accuracy; oracle labels gave 8/8. A mixed child improved
  only one of eight exact answers. Component improvement is not monotonic system improvement.
- Terminal-only high-LR RL is a clear negative for this exact recipe: after four further genuine
  updates, checkpoint 6 scored 47/72 versus 55/72 at the unchanged start. On composed questions,
  only 26/45 available endpoints were both semantically faithful and strict-correct; four additional
  strict answers were zero-answer coincidences from wrong computations.

## Closest prior art and the narrow contribution

[Recursive Language Models](https://arxiv.org/abs/2512.24601) establishes recursive inference and
released transition training; [λ-RLM](https://arxiv.org/abs/2603.20105) motivates typed functional
decomposition; [SRLM](https://arxiv.org/abs/2603.15653) emphasizes program selection. Structured-
output compliance itself is covered by [JSONSchemaBench](https://arxiv.org/abs/2501.10868).

Our plausible contribution is an empirical training-and-measurement result: internal communication
contracts are learned behaviors; local interface gains can fail at downstream composition; and
terminal accuracy alone overstates progress unless acquisition, operator/scope faithfulness,
stopping, NULLs, and physical cost are audited separately.

## What is missing and the decisive next step

The separate-corpus training run is complete, but both trained checkpoints were evaluated on the
same eight research-exposed source contexts. The largest gap is now evaluation on newly selected
source inputs, followed by additional training realizations or a second task if the effect persists.
Promotion should require performed, nonzero composed answers, not lower training loss or correct
numbers from the wrong computation. Alongside the smaller-update-size
reward-training comparison, we should test whether judging intermediate work
helps. Such a test should reward useful work, not merely making more helper calls.

**Path:** several days for a newly selected evaluation panel; 1–2 weeks for another training
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

# Recommendation for the meeting

These are work estimates conditional on useful results and continued compute
access, not promises of publication or acceptance at a particular venue.

Lead with Option 1 because the matching effect appears across several controlled interfaces and two
sizes in one model family, and it is visually explainable. Its largest remaining gap is whether it
helps the final task on fresh evaluation inputs. Present Option 2 as the broader systems story and
publication trajectory, with the explicit status: a supervised result observed after training on
two separately captured corpora but evaluated on one shared panel, informative composition limits,
and a negative result for one RL recipe—not yet a general training theorem or new RL algorithm.
