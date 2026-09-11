---
title: Supporting findings for questions and discussion
updated_utc: 2026-09-11T03:45:00Z
supporting_results_reviewed_utc: 2026-09-11T03:05:00Z
status: exploratory_supporting_notes
main_deck_evidence_cutoff_utc: 2026-09-11T03:45:00Z
---

# Why these notes are separate

These checks clarify the 14-slide deck's claims without adding more slides.
They were first documented after the earlier draft; the current deck now has a
later cutoff and includes the fresh-input and batch-size results. This file keeps
its original name so existing links remain valid. Read it after the slide-by-slide
guide; it is not additional material you need to present.

## For slide 9: does enforcing the answer format explain the improvement?

There are two different questions:

- Does requiring the right format prevent incomplete or unusable answers?
- When answers already have the right format, does repeating the input's tag
  beside its answer help the model choose the right reading label?

Slide 9 addresses the second question. Every output was well formed in both
conditions, yet matching tags improved reading accuracy from 32.6% to 78.8%.
That difference cannot be explained by more malformed outputs in one condition.

A later check addresses the first question. It used exactly the same prompts
and paired seeds, with software enforcement of the output format switched on or
off. All 48 responses followed the format with enforcement; only 37 of 48 did
without it. Under the strict scoring rule, an incomplete or malformed response
receives no correct labels. Accuracy on the later records was 85.0% with
enforcement and 65.2% without it.

Almost all of that difference was associated with avoiding invalid responses.
Among the 37 pairs that had valid responses on both sides, the net difference
was only one correct label out of 1,184. That smaller comparison is descriptive:
whether a response is valid can itself depend on the condition, so selecting
only those pairs is not a clean test of reasoning ability.

**A short answer for the meeting:** “The output rules help the model finish an
answer in a usable form. Separately, matching tags helped it put the right
reading answer beside the right input, even when formatting was already correct.”

Both studies reuse a small panel of 16 inputs. Neither reveals the process
inside the model. We have not yet shown the matching-tag benefit without output
enforcement; that needs a comparison with both matched and unmatched tags in
both enforcement conditions.

## For slides 10 and 12: do better helper labels improve the final answer?

A later exploratory check provides a small encouraging signal, but not a
successful test of the complete RLM.

We saved helper judgments for eight inputs and applied the same trusted Python
calculation to each set. Each input had a counting question and a weighted-sum
question: 16 answers per condition. The calculation returned the exact answer
on 2 of 16 questions with labels alone, 4 of 16 with row-number tags, and
5 of 16 with arbitrary matching tags.

This calculation was performed after the experiment as a diagnostic. It was
not the trained main model choosing or writing the calculation. The complete
RLM test was inconclusive: only 8 of 96 planned episodes had an observed final
answer; the rest remained unavailable. The main model repeatedly misused the
new helper interface, and some conversations grew beyond the context limit.
That must be separated from whether the saved helper labels were useful.

**A short answer for the meeting:** “Better labels helped a fixed calculation
on a few small examples. We still need to show that the main model can use
those labels reliably in a complete task.”

Do not compare these 2/16 and 5/16 totals directly with slide 10's 0/8 and 1/8.
They use different tasks, inputs, and interventions. Eight shared inputs are
also not sixteen independent tests. This is a promising follow-up, not a
confirmed general benefit.

## Evidence and review

These source paths are relative to the research store
`/project/alex_phd/runs/rlm-research-r4/`.

| Evidence | Source | SHA-256 |
|---|---|---|
| Format-control report | `analyses/leaf-mnli-stable-anchor-grammar-control-live-2026-09-11/REPORT.md` | `19a51e291ca13147648714fe92d45620ea9a64bc383264f2b649a6e586579c25` |
| Format-control native audit | `analyses/leaf-mnli-stable-anchor-grammar-control-live-2026-09-11/AUDIT.json` | `e8d255746e65294a039d810c65732247193f9e1a3a05e7c480e2494de031868b` |
| Format-control seal | `analyses/leaf-mnli-stable-anchor-grammar-control-live-2026-09-11/FINAL_SEAL.json` | `b0577dda8935604de733ba43cda013f435bfbd5334d6835c0bd0e9f0789d027c` |
| Partial whole-task report | `analyses/root-stable-anchor-downstream-bridge-live-2026-09-11/REPORT_ATTEMPT003.md` | `973ba23f60b08d0207131ba5c219cb2d5c76f545c027d29c96f65dfb73ee8f29` |
| Partial whole-task audit | `analyses/root-stable-anchor-downstream-bridge-live-2026-09-11/PARTIAL_NATIVE_ATTEMPT003.json` | `411d80e1a8a5f3d81ae964b498ede05fdc6d0b0ddd27a0857ad1fb0732525574` |
| Partial whole-task seal | `analyses/root-stable-anchor-downstream-bridge-live-2026-09-11/FINAL_ATTEMPT003.json` | `57fafb8fc24d1b32bfd60cd2b85d21ee9a943b5f593e41126a354319119c448e` |

MAIN read both complete audit implementations and reports, checked their sealed
source and terminal references, independently recounted all 96 format-control
responses and 24 helper arrays, and independently recomputed all 48 fixed-Python
answers. The full-RLM availability classification comes from the frozen native
audit; MAIN did not separately read every root program. No sampled model code
was executed during this review.
