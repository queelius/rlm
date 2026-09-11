---
title: Verification of the advisor-meeting draft
verified_date: 2026-09-11
status: compiled_and_visually_reviewed
evidence_cutoff_utc: 2026-09-11T12:55:00Z
---

# Current revision — whole-deck clarity pass, approximately 14:51 UTC

This revision responds to the presenter's difficulty understanding page 2 and
request to review every slide for similar missing context. All **eight main pages
and six backups** were reviewed, revised where needed, compiled, and visually
inspected by MAIN. The task, example classifications, and final counts are now
explicit on page 2. The rest of the deck distinguishes the two studies, explains
its measures and comparisons, and keeps proposed complete-RLM benefits separate
from observed helper results. [The clarity review](REVIEW_CLARITY.md) records
the page-by-page changes. The talk remains eight main pages, about ten minutes.

| Artifact | SHA-256 |
|---|---|
| research-update.pdf | `8afc3f183dd65c56cb15501afd5f0cb0b4276695165307695f1d9d2174ae9db5` |
| research-update.tex | `48021501e486a79b51b09474893264a9da44ef9f309cbf842b6eb4d93885125b` |
| data/claims.json | `853dcfb4a2d94bfdb06c0b53909b782ee40d454f546ddef1d0232bebd8e27e4f` |
| speaker-notes.json | `3a72825bb59f7e672056a85cdbfe60c252ee541c00ebddb28808fa08a5f25a00` |
| research-update.pdfpc | `3439113992b13503447efee2b684e1ecebbd10c9a9ff16c85e39752f230d6244` |
| speaker-guide.md | `638f19639f59a4b9b1ba81102c52c4932aa79998aa8a2351d50dcf1611eb3cd0` |

Fresh checks passed: 14 focused slide/launcher tests; all PDF titles, guide
headings, note titles, and main/backup footers; no audience annotations or
out-of-page text; focused Ruff and formatting; and `git diff --check`.
The final TeX log contains no overfull/underfull boxes, LaTeX warnings, undefined
controls, or TeX errors. MAIN inspected all 14 audience renders, including footer
spacing, table labels, and the enlarged backup-plot text. No clipping or overlap
was observed. The body font was not reduced to fit the explanations.

All **50 source pins** in the figure data matched. A structured comparison with
commit `30afe9b` confirmed that only two presentation-label fields changed in
`data/claims.json`; all measurements, counts, limits, and provenance are identical.
The numerical evidence cutoff remains 12:55 UTC. This was a communication revision,
not another experiment or a new independent evidence audit.

MAIN also inspected every note page in actual stock pdfpc 4.6, under an Xvfb
1280×720 virtual display with software rendering. All notes fit, and separate
audience/presenter windows and the ten-minute timer appeared. The receipt and
14 captures are in
`/project/alex_phd/research-cache/tools/pdfpc-laptop-20260911/smoke-017-clear-examples/`.
The [presenter preview](presenter-preview.png) now shows the concrete page 2 example.
The check stopped its owned GUI processes. This is not a physical laptop,
projector, or video-meeting test; share only the audience window. Some isolated
toolbar icons remain cosmetic omissions, as in the earlier checks.

Reproduction from this directory:

    make figures tectonic check PYTHON=/project/alex_phd/envs/rlm-advisor-figures-20260911/bin/python TECTONIC=/project/alex_phd/research-cache/tools/tectonic-0.17.0-musl/tectonic

# Historical eight-main-slide revision around 13:41 UTC (superseded)

The audience PDF has **eight main slides and six optional backups**, with a
ten-minute presenter timer. MAIN compiled with Tectonic 0.17.0, rendered and
visually inspected every audience page, then inspected all 14 sets of private
notes in actual stock pdfpc 4.6 at 1280×720 under Xvfb/software rendering.
No clipping or overlap was observed. A chart-axis wording change affected only
page 5; MAIN inspected that page again in both views. The latest clarification
of the Python workspace affected only page 2. MAIN inspected its actual presenter
view again; the other 13 audience pages were pixel-identical to the preceding
GUI-checked PDF. Private notes remained byte-identical across these revisions.

| Artifact | SHA-256 |
|---|---|
| research-update.pdf | `3b23bb31ebea4aff2f39c93ff2844e66646c12fdb7b27a9497556da52bd9c6dd` |
| research-update.tex | `9f4abc307d239c7e503892becf8a2e359763dae1f0c6949244963bdf94ad8ff9` |
| data/claims.json | `8dac2ab8c5972b4bf3ab6d223fcddc8fb050feed3bb76a5fbf85dbb00ae0250d` |
| speaker-notes.json | `0fd08055ab9245465ebfc55a344d6d781fe19218e0e6f28d6289be9a4d1ad57d` |
| research-update.pdfpc | `56e48a99c118d61d869c005b3514ad7ce5b7b3383ec715c7ca577e76f852e13a` |
| speaker-guide.md | `2da7ccac72f4e42637d0ed7f9afbdaef48d5624be1560ff89be19ed1f4331b0f` |

Fresh checks passed: 14 focused slide/launcher tests; all note titles and
main/backup footers; no audience annotations or out-of-page text; focused Ruff
and formatting; `git diff --check`; and a log scan with no warnings, overfull or
underfull boxes, undefined controls, or errors. The numerical file's **50 source
entries** all matched. The evidence document's **66** table pins and supporting
notes' **14** pins matched. The two-model test fixture intentionally omits three
source entries, so its internal printout says 47 rather than the real file's 50.
Fourteen guide headings match the current source. Independent focused code and
lay-audience review found no critical or important blocker; a timing ambiguity
was corrected to specify the sum of four workload blocks per policy.
The guide's subsequent illustrative cross-domain example is explicitly marked
as motivation, not an observed transfer result; it changes neither slides nor notes.

The new evidence includes S4 compound execution, H9 same-panel Mistral, H10
output-order crossing, E1 equal-record workload costs, and R3's secondary
final-answer result. MAIN read and reran the full native readers for H9's 480
responses, H10's 96 responses, and E1's 848 responses, obtaining exact replay
matches. S4's 144-endpoint native extraction replay matched byte-for-byte; its
132 available execution paths were reviewed by an unblinded agent, with selected
paths independently inspected by MAIN. R3's final-answer endpoints were replayed;
the full faithful-calculation population review remains unfinished. Do not infer
that a native arithmetic replay independently repeats every semantic judgment.

Supporting documents also include the two compact-reply comparisons. MAIN read
their full native audit implementations and independently replayed all 64 replies,
obtaining byte-identical audit files. These additions do not change the audience
PDF's numerical cutoff or imply complete-RLM gains.

The final fixed-deadline handoff pilot remains inconclusive: four native-admitted
outcomes and 28 unknown outcomes. MAIN read the incremental native audit and
replayed the complete extraction byte-for-byte, including all 32 original helper
responses, all 32 planned root slots, and the 227-request physical union. MAIN
inspected selected programs and observations from each admitted root; all-path
semantic annotation is an unblinded agent review. This is not a 0/32 observed
failure rate or evidence of a complete-system benefit. The speaker guide,
supporting notes, portable data, and delivery checkpoint record the distinction;
no late pilot number was added to an audience slide.

The GUI receipts are in
`/project/alex_phd/research-cache/tools/pdfpc-laptop-20260911/`, runs
`smoke-014-ten-minute-update`, `smoke-015-final-eight-main`, and
`smoke-016-workspace-final`. The latest run
opened separate audience and presenter windows, displayed the ten-minute timer,
advanced through all pages, and stopped its owned processes. The presenter
window was expanded programmatically to model pressing `w`; the user must do
that on the laptop. [The checked presenter preview](presenter-preview.png) is
from the unchanged page 5. No physical laptop, video-meeting app, macOS or projector test
is claimed. Some isolated-environment toolbar icons are missing; slides, notes,
navigation and timer render. Share only the audience window, not the desktop.

Reproduction from this directory:

    make figures tectonic check PYTHON=/project/alex_phd/envs/rlm-advisor-figures-20260911/bin/python TECTONIC=/project/alex_phd/research-cache/tools/tectonic-0.17.0-musl/tectonic

The smaller experiments launched after this content cutoff are kept in the
research store and supporting notes as they are audited. They do not silently
change the claims or figures in this verified PDF.

# Historical 03:45-cutoff PDF and 06:33 presenter check (superseded)

The following record describes the earlier expanded deck, not the current pages.

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
