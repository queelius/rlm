---
title: "Teaching usable next steps"
date: 2026-09-29
evidence_cutoff_utc: "2026-09-29T05:30:50Z"
unchanged_slides_evidence_cutoff_utc: "2026-09-29T01:14:27Z"
slides: 5
status: exploratory_update
---

# September 29 research update

Open [research-update.pdf](research-update.pdf). This is a new five-slide,
plain-language update for the user and an advisor. The two September 25 decks
are historical records and are not changed. Slide 4 now uses the completed
transfer snapshot at 05:30:50 UTC (displayed 05:30). Slides 1–3 and 5 retain
their original 01:14:27 UTC evidence and displayed 01:14 cutoff; their pending
claims are not silently advanced to the new transfer cutoff.

The story is: teach usable steps; show a same-answer history repair; report
three small repair comparisons; contrast familiar-goal RL gains with the small
changes on different target goals; ask whether the repair transfers and whether a helper benefits
the whole task. Limitations are visible in the audience PDF.

## Build and present

The deck reuses the earlier Beamer colors, type and native TikZ style. It needs
no external art or figure downloads.

    make                         # With latexmk installed.
    make tectonic                # Alternative TeX compiler.
    make notes notes-check       # Python standard library; uses the earlier formatter.
    make present                 # One laptop, two windows: share only the audience window.
    make rehearse                # Private single-screen presenter view.

The two pdfpc commands expand to:

    pdfpc --pdfpc-location=research-update.pdfpc --duration=7 \
      --note-format=plain --windowed=both research-update.pdf
    pdfpc --pdfpc-location=research-update.pdfpc --duration=7 \
      --note-format=plain --single-screen --windowed=none research-update.pdf

Do not share the desktop or presenter window. Anyone looking at the same
physical screen can still see notes. These are the previously checked pdfpc 4.6
flags; no live GUI session is claimed for this deck. Notes use 42-character
lines with at most 14 lines per slide and a 16-point requested note font.

Portable notes are in [research-update.pdfpc](research-update.pdfpc), with
editable [speaker-notes.json](speaker-notes.json). The notes formatter is reused
read-only from the September 11 deck; regenerating notes writes only here.
Longer explanations and questions are in [speaker-guide.md](speaker-guide.md).
[evidence.json](evidence.json) records plotted counts, controls and source hashes.

## Local toolchain and checks

Use the existing CPU-only tools; no installation or training-environment changes
are needed:

    make tectonic notes-check \
      TECTONIC=/project/alex_phd/research-cache/tools/tectonic-0.17.0-musl/tectonic \
      PYTHON=/project/alex_phd/envs/rlm/bin/python

Compiler and renderer provenance is recorded in the
[existing tooling note](../2026-09-11-advisor-meeting/TOOLS.md).
The new deck uses Tectonic 0.17.0 and PyMuPDF 1.26.4 for CPU page inspection.
Compilation and notes use the Python environment above; PDF rendering reuses
`/project/alex_phd/envs/rlm-advisor-figures-20260911/bin/python`, because the
general `rlm` environment has no PyMuPDF. No packages were installed.
The verification receipt records the final PDF, note bounds, source hashes and
preservation of both earlier decks. Compilation and PDF inspection do not
establish GUI behavior on a particular laptop.

## Scope

This is a fixed-cutoff documentation update, not a new experiment or reanalysis
of model trajectories. It uses completed native-audited findings. No GPU jobs,
queue owners, experimental sources, datasets or earlier decks are changed.
The brainstorming skill kept the update within the agreed five-slide story;
verification checks cover the rendered pages and bounded notes.

## Completed transfer update

The [fixed transfer report](../../experiments/selective_delegation/rl_transfer_results_20260929/README.md)
and [paired native evidence](../../experiments/selective_delegation/rl_transfer_results_20260929/RESULTS.json)
replace only slide 4's former “transfer pending” statement. Familiar own-interface
gains of 9→13 and 14→16 become 4→5 and 5→5 on the fixed different-goal diagnostic.
These are root-disjoint official TRAIN goals in the same recipe world, not a
held-out benchmark. Fresh-A learning and extra-SFT controls remain pending at
the transfer cutoff. Audience claims, notes, guide, data and verification were
updated together. No other scientific findings were refreshed.

See [slide 4](rendered/slide-04.png) and the
[static slide-plus-notes preview](rendered/slide-04-notes-preview.png).
The latter checks the actual wrapped note at 16 points; it does not claim a
live pdfpc window test. Final compilation has no overfull boxes.
