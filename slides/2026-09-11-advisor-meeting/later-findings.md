---
title: Supporting findings for questions and discussion
updated_utc: 2026-09-11T13:08:00Z
supporting_results_reviewed_utc: 2026-09-11T13:08:00Z
status: exploratory_supporting_notes
main_deck_evidence_cutoff_utc: 2026-09-11T12:55:00Z
---

# Why these notes are separate

These checks clarify the eight-main-slide deck and its six optional backups.
They were first documented after the earlier draft; the current deck now has a
later cutoff and includes the fresh-input and batch-size results. This file keeps
its original name so existing links remain valid. Read it after the slide-by-slide
guide; it is not additional material you need to present.

## For backup 2: does enforcing the answer format explain the improvement?

There are two different questions:

- Does requiring the right format prevent incomplete or unusable answers?
- When answers already have the right format, does repeating the input's tag
  beside its answer help the model choose the right reading label?

Backup 2 addresses the second question. Every output was well formed in both
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

## For main slides 7–8: do better helper labels improve the final answer?

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

Do not compare these 2/16 and 5/16 totals directly with the older C1 study's 0/8 and 1/8.
They use different tasks, inputs, and interventions. Eight shared inputs are
also not sixteen independent tests. This is a promising follow-up, not a
confirmed general benefit.

## Did copying the trained helper API make the handoff easier to use?

A repaired 16-episode diagnostic gave the same saved helper map and public
records through two interfaces. One was a plain synchronous dictionary. The
other imitated the asynchronous `.answer` interface seen during training. The
plain interface produced 8/8 available finals and 7/8 faithful calculations;
the trained-looking interface produced 7/8 available finals and 7/8 faithful
calculations. Both gave only 1/8 host-correct answers because the saved helper
maps themselves contained label errors.

The trained-looking interface therefore did not establish improved faithful use
on this small, research-exposed panel. Under the prewritten decision rule, the simpler
dictionary is the preferred local boundary. This is useful interface evidence,
not a live-helper or complete-RLM comparison: both arms read frozen maps, and
the prompt wording and API shape changed together.

**A short answer for the meeting:** “Once both interfaces were given the same
records, the model could use either one. Copying the API shape from training did
not help on these eight examples, so the simpler dictionary is the better next
building block.”

## E2: shorter named replies reduced output work and time in both models

After E1 showed that large named calls were slower locally, we asked whether
the reply could be shorter without losing the benefit of matching. For example,
`[{"key":"aa","label":"neutral"}]` can be written as `{"aa":"neutral"}`.
Both carry the same record name and reading judgment; the second repeats fewer
field names and punctuation.

For each model, both formats processed the same 768 records in 16 calls. The
prompt and seed were paired; only the required output schema changed. Each
policy's time sums four nonoverlapping blocks with four simultaneous calls.
Startup is excluded, but failed-generation work is included. There were no
missing responses or unknown token counts, and no prefix-cache reuse.

| Model and format | Correct / 768 | Valid replies / 16 | Seconds | Output tokens |
|---|---:|---:|---:|---:|
| Qwen, longer reply | 660 | 16 | 113.48 | 18,210 |
| Qwen, compact reply | 654 | 16 | 86.79 | 14,104 |
| Mistral, longer reply | 318 | 12 | 250.66 | 34,988 |
| Mistral, compact reply | 365 | 14 | 130.95 | 21,721 |

Qwen retained nearly all accuracy: 85.9% versus 85.2%, while compact replies used
23% less output and 24% less time. Mistral used 38% less output and 48% less time,
but its accuracy gain mainly reflected fewer malformed replies. All six malformed
Mistral replies reached the output limit with long whitespace tails. They count
as wholly wrong under the prewritten metric, not as missing observations.

Among the 11 Mistral contexts valid in both formats, correct labels were 289/528
and 291/528. That outcome-selected comparison is descriptive, not a clean estimate
of better reading. Compact Mistral replies still failed twice. Both models passed
the prewritten exploratory two-percentage-point quality-loss screen; that is not
a confidence interval or a formal equivalence test.

**A short answer for the meeting:** “We can make the named replies more concise.
That reduced output work and local time in both tested models. It is a useful
implementation choice, not proof of better complete RLM solutions.”

This stays in supporting material: it refines the proposed component without
lengthening the main talk. A future control could hold whitespace formatting
fixed to distinguish ordinary brevity from avoidance of runaway whitespace.
Do not compare these fresh-control times directly with E1 as a paired experiment;
the neutral prompt changed between studies.

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
| Saved-map API V2 report | `analyses/root-bridge-native-api-transfer-v2-live-2026-09-11/REPORT.md` | `38d43ce8c507873afb15328f558a21e0a6a344f7c467a3912b097bd8a46b65c0` |
| Saved-map API V2 audit | `analyses/root-bridge-native-api-transfer-v2-live-2026-09-11/AUDIT.json` | `79757f6b9d200e09ee31a0bc9d111f10e27aa180ab9d6e80db87b8ba0ec97600` |
| Compact Qwen report | `analyses/leaf-mnli-compact-keyed-reply-live-2026-09-11/REPORT.md` | `b910bf8ed9f059f42d4e9878003466b2ad8a71308349c55e64d501539afa37a2` |
| Compact Qwen native audit | `analyses/leaf-mnli-compact-keyed-reply-live-2026-09-11/NATIVE_AUDIT.json` | `e5124b7e69547f1ae05325e65a5c72930235c807493a317ed06073832f87947b` |
| Compact Mistral report | `analyses/leaf-mnli-compact-keyed-reply-mistral-live-2026-09-11/REPORT.md` | `4d4c141d0aec9171802d7b310795e7f119bb1bf6a188f0b983b4930571690503` |
| Compact Mistral native audit | `analyses/leaf-mnli-compact-keyed-reply-mistral-live-2026-09-11/NATIVE_AUDIT.json` | `cb15dcbc1e9ed1bd05701affc2313973c612eda9d3c1c2edaad9fc115791f655` |

MAIN read both complete audit implementations and reports, checked their sealed
source and terminal references, independently recounted all 96 format-control
responses and 24 helper arrays, and independently recomputed all 48 fixed-Python
answers. The full-RLM availability classification comes from the frozen native
audit; MAIN did not separately read every root program. No sampled model code
was executed during this review.

That independent replay statement concerns the format control and earlier bridge.
For the API V2 probe, MAIN read the completed report and checked the report/audit
hashes. Its all-path semantic review was performed by the package author after
collection, was unblinded, and was not independently repeated by MAIN.

For both compact-output studies, MAIN read the independent native audit code
and reports, reran all 64 responses, and obtained byte-identical audit outputs.
The review includes actual native prompts, strict key/label decoding, invalid
reply accounting, model identity, timing blocks and complete physical costs.
