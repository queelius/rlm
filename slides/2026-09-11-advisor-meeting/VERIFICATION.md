---
title: Verification of the advisor-meeting draft
verified_date: 2026-09-11
status: compiled_and_visually_reviewed
evidence_cutoff_utc: 2026-09-11T03:45:00Z
---

# Current PDF

The current deck has 14 pages. MAIN compiled it with Tectonic 0.17.0, rendered
every page, and visually inspected all 14 pages. The final log has no TeX
warnings, overfull or underfull boxes, undefined controls, or errors.
All expected headings and page numbers passed; no text lies outside a page.

| Artifact | SHA-256 |
|---|---|
| research-update.pdf | `76f81f1f09fda840eb947b06880add4b32725e2a0c31e091f0ca2736351fdc93` |
| research-update.tex | `30a1f0e561d96ea7dae8effaf34700a1fd4b2f1674fa86a2aeecbf096c1f285c` |
| data/claims.json | `5d78977ebf844ced079ab7b21c579c3b801b9d64e4f597b915b4cd9e6ff9b5a6` |
| speaker-notes.json | `ceab2d81e05e43653af74e5d8f15f8866f1eef68ae919ba6aa2913da7f888c7d` |
| research-update.pdfpc | `c4b0382378050550e18ecaf78b126f0f28128ef7dea80eccde634c3229e01fe1` |

## Presenter-console check, approximately 06:33 UTC

Audience source and numerical cutoff are unchanged. A fresh Tectonic build
produced the PDF above; MAIN re-rendered and inspected all14 audience pages.
All28 numerical source entries and all41 evidence-table pins matched again.
No audience page contains a PDF annotation; notes remain in a separate file.

Eight focused launcher tests pass, including reordered/missing notes, explicit
line wrapping and note-height limits. All14 note titles match the compiled PDF.
Notes are63–78 words,13–16 displayed lines, at most42 characters per line.
Focused Ruff and `git diff --check` pass. An independent read-only code review
found no critical or important blocker.

Stock Ubuntu pdfpc4.6 rendered the actual GUI under Xvfb with software rendering.
MAIN inspected all14 presenter captures at1280×720 and additional slides5/9
at1366×768. No clipping or overlap was observed after expanding the presenter
to the full tested area. The actual launcher produced separate presenter and
audience windows; a final rehearsal launch opened only the fullscreen presenter.

The real GUI exposed two issues: plain notes do not wrap, so the generator now
inserts line breaks; the two-window presenter starts at half screen, so the
instructions and launcher say to click it and press `w`. The seemingly useful
`--windowed=presentation` suppressed the audience window on one monitor and is
not used. Configuration also avoids options absent from pdfpc4.6.

These are X11 virtual-display tests, with test-driven resizing, not a physical
laptop/window-manager, Zoom/Meet, macOS, projector, or HiDPI test. Rehearse on
the actual laptop. Some SVG toolbar icons were missing in the isolated setup;
slides, notes, navigation and timer rendered. All owned test processes stopped.

Receipts: `/project/alex_phd/research-cache/tools/pdfpc-laptop-20260911/`, especially
`smoke-006` through `smoke-009`, `smoke-011-launcher-max1280`, and
`smoke-013-final-rehearsal`. [One presenter capture](presenter-preview.png) is in
Git. See TOOLS.md for package provenance.

Seven vector figures are included; five appear in the deck. The earlier
four-condition row-number and stable-key figures remain as supporting artifacts.
All 28 source entries in the numerical file were available and matched their
recorded hashes. All 41 path/hash pairs in the evidence document's source table
also matched. These checks establish traceability, not independent replication.

# What changed in this revision

- Slide 5 replaces the reused-input comparison with the newly selected-input
  evaluation: 12, 55, and 53 successes out of 72. Missing-outcome bounds are
  12–19, 55, and 53–55. Both training runs started from the same model; the helper
  and familiar task types stayed fixed.
- Slide 7 replaces the four-condition chart with batch-size curves. It defines
  tags and states that accuracy includes all answers, not only later positions.
- Slide 8 identifies its different record set and retains its later-position
  metric. Its three models still face the same records as one another.
- Slide 12 no longer describes new-input evaluation as undone. It separates
  new combinations of calculations from the open whole-task matching question.
- The guide, portable data, methods, publication options, README, and cutoff were
  updated together. Earlier S1/S2/H2 evidence was retained, not overwritten.
- Supporting format-control and partial whole-task findings remain in
  later-findings.md. Its introduction now explains that the filename comes from
  an earlier draft; it does not claim a later cutoff than the current deck.

The first compilation of this revision found vertical overflow on slides 7 and 12.
A redundant sentence and a padded takeaway box were removed. The body font was
not reduced. The final figures have readable axes, visible endpoints and counts,
and no observed clipping or overlap.

# Evidence review scope

For S3, MAIN read the audit/report and semantic merge logic, checked the final
evidence pins and complete 216-row arithmetic, and earlier recounted strict
outcomes with 208 saved episode hashes. Agents manually reviewed all 207 available
execution paths. MAIN inspected selected paths, not all 207 independently.
Those judgments are unblinded and agent-authored, not a human annotation study.
The test changes input records, not the families of tasks; public text may have
appeared in pretraining or helper training.

For H8, MAIN read the complete 300-line independent reader and report and reran
that reader into a temporary output. Its result exactly matched the sealed audit
for all 240 request/response records and all 15 format-by-size cells. Six final
artifact pins matched. There are 16 underlying input sets, not 8,064 independent
trials. The result concerns the combined input/output format and does not identify
an internal model mechanism, a universal batch-size limit, or a speedup.

The previous three-model and matching-versus-different-tag evidence retains its
earlier source reviews and qualifications. No new reward-training result is
included: new integration failures and repairs belong in operations records,
not the scientific result slides.

# Executed checks

From this directory, using the isolated figure environment:

```text
python figures.py
tectonic --keep-logs research-update.tex
python verify_deck.py research-update.pdf
```

Also checked: all 14 guide sections in order; current document cutoffs; numerical
source hashes; source-table hashes; and `git diff --check`.
Focused Ruff checks passed for figures.py and verify_deck.py after four
line-length-only corrections; those corrections do not change the generated content.
`make -n -C slides meeting` correctly resolves the latexmk build route.
latexmk itself was not run on this machine; the actual compiler was Tectonic.

The bounds checker cannot detect every overlap, so the visual review remains
necessary. No human comprehension study was performed. The independent reader
review is documented in REVIEW_STANDALONE.md.

# Earlier reviewed snapshot

The immediately preceding wording-only PDF had SHA-256
`85280e610b625f7f671441c4657255f6e0ae501ae54674e993610032438313ed`
and a 02:15 UTC numerical cutoff. It used the older slide 5 and slide 7 figures.
Those findings remain in the portable data and evidence notes; the current PDF
is a coherent later revision, not a silent numerical replacement.

The03:45 revision initially remained local; it is included in the subsequent
presenter-console publication checkpoint. Repository history records the actual
commit. Recheck this record after changing any slide, figure, or displayed value.
