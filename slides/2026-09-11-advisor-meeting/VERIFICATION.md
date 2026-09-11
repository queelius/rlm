---
title: Verification of the reviewable advisor-meeting draft
verified_date: 2026-09-11
status: compiled_and_visually_reviewed
evidence_cutoff_utc: 2026-09-11T00:15:00Z
---

# What was checked

The deck compiles to thirteen pages with Tectonic 0.17.0. The final compiler log
contains no overfull or underfull boxes, undefined controls, errors, or warnings.
The PDF checker finds all thirteen expected slide headings and page numbers, with
no text extending outside the page.

Every page was rendered and visually inspected by the main agent. The initial
draft had crowded footnotes and several vertical overflows. Content and spacing
were revised rather than shrinking the body font. A later visual pass found
two clipped vertical chart labels and crowded chart annotations that the LaTeX
log did not report; those figures were corrected, rebuilt, and re-inspected.

Five vector figures are included; four are displayed in the deck, while the
original four-condition tag control remains available as a supporting figure.
All thirteen original-source
hashes declared there were available and matched on the research machine.
Their displayed values were checked against the source reports and audits.
All twenty-seven exact-source hashes listed in the evidence document also matched.
This establishes traceability, not independent replication of the experiments.

The original display-number denominator error is not carried forward:
16 batches times 32 later records gives 512 labels per reference-specific cell,
and three reference conditions give 1,536 labels per plotted bar.

# Review changes

An independent clarity review is retained in REVIEW_CLARITY.md as a record of an
earlier draft, not as the current status. Its principal concerns were addressed:

- The overflow warnings were removed and all pages inspected.
- The final reward-training audit was added to the portable evidence.
- The main-model and helper-training comparisons were distinguished.
- The guide explains why the training chart is a related changed-metadata
  evaluation, and distinguishes the newly completed second training corpus
  from the original three readouts of one checkpoint.
- The real dataset and the illustrative status of the toy example are explicit.
- Evidence tags are defined in the guide.
- The unsuccessful rechecking and direct-summary experiments are explained in
  the guide and evidence document without crowding the slide deck.
- Plot labels are derived from data rather than hard-coded numbers.

The shared-passage toy example remains intentionally simple. It illustrates two
complete passage–statement judgments with different correct labels; it is not
a reproduction of the experiment's full batch. The guide explains the actual
record-level manipulation.

# Build checks

Executed: figure regeneration, actual Tectonic compilation, thirteen-page PDF
text/bounds checks, and git diff whitespace checks. All original twelve slides
were visually inspected. After adding the matching-key result, the new slide 8
and revised next-steps slide 11 were rendered and inspected again; other page
content was unchanged apart from numbering. The parent Makefile's meeting target
was dry-run successfully.

The added H3 result was checked against the complete independent audit: all eight
sealed file hashes matched, and the main agent recounted all 192 authenticated
rows into the four plotted totals. The review included the full audit code,
the model/service identity, and its stated limits. The evidence document, guide,
publication options, slide numbering, and cutoff were updated together.

The latexmk route was not executed on this machine because latexmk is not
installed. It uses the standard Beamer source and included vector PDF figures.
Tectonic is the compiler actually used for this verification.

Verified PDF SHA-256:
22c8a2f3e1619db01b0020b1c8aa66455c898963557ade07fa0700717351f656

Verified Beamer source SHA-256:
a9a4928a1305d802f9b99a336f522b3229e413c109fe9c36a6b902dda64d3019

Portable numerical evidence SHA-256:
55036610895067feb05a660ecb6d8368d56795904273ddec07e0e378237639b3

Recheck and update this record after any subsequent slide or figure change.
Rendered previews and compiler intermediates are ignored; the PDF and vector
figures are explicitly allowed into Git.

## September 11 update: two replication checks

Slide 5 now shows 12, 50, and 55 verified successes out of 72. Both trained models
restart from the same already-trained adapter; the third bar is not cumulative
training. The two new-run timeouts remain missing, with possible totals 55–57.
MAIN checked the new audit source and all eight sealed artifact hashes,
recounted the 72 semantic rows, and spot-reviewed six recorded program paths.
The complete 70-path review was performed by the experiment-package author;
that overlap is disclosed. The report's mistaken “released start” wording is
corrected by a preserved additive erratum, also checked.

Slide 8 now compares the same three formats in the smaller and larger Qwen models.
MAIN read the full second-model audit, verified seven sealed hashes, and recounted
all 144 responses into the plotted values. This is not a pure model-size test
or a different model family. The former shuffled-number figure is retained as
supporting material rather than presented as an unmeasured 8B condition.

The updated deck was compiled and rendered again. An initial slide 8 vertical
overflow was fixed by shortening the text and changing the chart proportions;
the body font was not reduced. MAIN inspected final slides 5, 8, 9, and 11.
The final log has no overflow/underflow or other warnings, and all thirteen
heading/footer/bounds checks pass. A second reviewer found no content or
numerical blockers. Its initial footer-clipping report was an image-viewer
artifact and was retracted; MAIN's full images and PDF footer coordinates
confirm that both slides 8 and 9 contain complete, visible footers.
