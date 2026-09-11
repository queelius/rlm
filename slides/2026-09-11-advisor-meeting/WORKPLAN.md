---
title: Advisor discussion preparation plan
meeting_date: 2026-09-11
status: reviewable_draft_verified
audience: presenter, advisors, and research colleagues
---

# Purpose

Explain what we are trying to learn, show our strongest preliminary evidence,
and invite useful advice about a realistic next research contribution. This
package must also teach the presenter the research; it must not assume that he
remembers implementation details from the automated experiments.

## September 11 presenter-console update

The user requested pdfpc speaker notes, a one-screen laptop launch path, and
periodic GitHub pushes. Keep the14 audience slides and their03:45 numerical
cutoff unless newly reviewed evidence warrants a content change. Put concise
speaking cues and likely questions in a separate pdfpc sidecar, with long
explanations in the existing guide. Make note/slide ordering checkable. Test
the real GUI at laptop resolution, not only PDF text bounds, and document the
need to share only the audience window. Push the verified package and durable
operating instructions without including raw runs or model weights.

# Agreed design

Use approximately twelve slides, with a complete-sentence message on each slide.
The reviewed version has thirteen: the completed arbitrary-tag control earned
one additional slide, rather than crowding the existing comparison.
The two later replication checks update figures on existing slides 5 and 8;
they do not add slides. Keep the shared evaluation inputs and model-family limits explicit.
Explain recursive language models with a concrete example before showing results.
Use native vector figures, clear denominators, and ordinary language. Keep names
of internal training runs and most implementation details out of the slides.
Put methods, limitations, full evidence references, and likely questions in
companion documents. Preserve the previous slide deck as a historical artifact.

The user delegated design decisions and approved examples and informative figures;
no further approval pause is needed. Independent research continues on the A100.

# Work and verification

1. Select claims from authenticated completed experiments. Separate observed
   results from interpretation, pending experiments, and future ambitions.
2. Write the story and worked examples. Use the same example to explain the
   task, training, and the difference between a wrong intermediate answer and
   a wrong calculation.
3. Save a small, portable numerical evidence file. Generate figures from it,
   checking all numerators, denominators, and comparisons against source audits.
4. Produce the Beamer deck, slide-indexed speaker guide, evidence and methods,
   and publication options. Assign independent documents to avoid shared edits.
5. Compile the PDF in a separate CPU-only tool environment. Render and inspect
   every slide. Fix clipping, overlap, small type, and unclear examples rather
   than reducing the font until content fits.
6. Review the package from the perspective of an unfamiliar colleague and the
   presenter. Check that each chart can be explained without reading the research
   log. Verify the final PDF and record its evidence cutoff.

# Decisions that keep the story honest

- The question-sensitive training readouts are related evaluations of one
  checkpoint, not three independent training replications.
- A later separate training corpus provides a second training realization,
  not a second independent evaluation set or proof that its examples are better.
- The model comparison now includes three models from two families, not size alone.
- Later records within one batch are not independent experimental units.
- Structured output forces identifiers and syntax, not correct labels.
- Improved intermediate labels do not automatically mean a correct final answer.
- A small unsuccessful reward-training run is not evidence that reward training
  generally cannot work.
- Introduce future adaptive planning and model--harness co-training as plans,
  not accomplishments. The related repositories provide ideas, not pooled data.

# Presentation design reference

The first reviewable draft is complete. See VERIFICATION.md for the actual
compile, numerical-source, and visual checks. Further experimental results must
be integrated with a new cutoff and another PDF review, not silently substituted.

The sentence-headline and visual-evidence approach follows
[Michael Alley's tutorial](https://www.assertion-evidence.org/tutorial.html).
Secondary details belong in the speaker guide, not in a crowded slide body.

## Living update rule

The user explicitly requested that salient new findings update this package.
Promote a result when it materially changes the story, strengthens an important
check, or changes the next research decision. Update the data, figure, plain-language
interpretation, presenter questions, limitations, evidence links, and cutoff together.
Then compile and inspect the PDF. Prefer revising an existing slide when its
message remains clear; do not add every experiment to the main deck.

## Standalone-reader revision

The user asked that advisors be able to understand the slides without narration.
This began as a bounded revision of the existing 13-slide story.
Put the task, what changed, and the metric in the visible slide body or chart.
Explain the switches between main-model training, helper-interface tests, and
helper training. Define unfamiliar terms where used. Replace ambiguous chart
labels such as "Reference labels" and "Verified successes" with their meaning.
Keep essential limitations visible, with detailed counts and methods in the guide.
Use examples and short complete sentences, not more dense footnotes. Compile,
render, inspect all changed pages, and obtain a standalone-reader review.

The revision also incorporates two reviewed results available at 01:30 UTC:
Mistral repeats the direction of the tag benefit with lower absolute accuracy;
smaller reward updates show no clear improvement. Both revise existing slides,
not the slide budget. All three model formats remain paired on the same inputs;
the two reward runs restart independently rather than forming a training sequence.

## One additional slide for a salient control

The 02:15 evidence revision adds H6 as slide9, taking the deck to14 slides.
MAIN chose to preserve the future-test and discussion slides instead of merging
away their questions. The new comparison gets its own small diagram and chart:
tags on both sides, matching versus different. It reports reading accuracy, not
an unobserved internal mechanism. The guide explicitly explains why this is not
a contradiction of the earlier misleading-name study. Slide5 also puts the
previously used evaluation inputs in the main body. Recompile and inspect the
revised PDF; do not solve overflow by shrinking body text.

The final wording pass clarifies independent training restarts, previous use of
the evaluation records, the unchanged reading task across model comparisons,
and our supplied Python calculation in the helper-training test. Later controls
belong in later-findings.md for now; they do not increase the slide count.

## New-input and batch-size evidence (03:45 cutoff)

Replace slide 5's figure with the shared newly selected-input evaluation and
slide 7's figure with the nested batch-size curves. Keep the earlier measurements
in the evidence document and portable data. Define a matching tag before showing
the curve; distinguish its all-answer metric from the later-answer cross-model
chart. Update the future-test slide because new-input evaluation is now complete,
while new calculation combinations and full-RLM matching benefits remain open.

This bounded revision keeps 14 slides. The initial PDF had two vertical overflows;
remove redundant text and padding, not body-font size. Compile, check all headings
and bounds, visually inspect every final page, and keep exact hashes in the
verification record. The main story should stand on its own; the guide supplies
examples and questions without becoming required narration.
