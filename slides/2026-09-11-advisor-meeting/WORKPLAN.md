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
- The second-model result compares two released models in one family, not size alone.
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
