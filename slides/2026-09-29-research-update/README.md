---
title: "Teaching usable next steps"
date: 2026-09-29
evidence_cutoff_utc: "2026-09-29T01:14:27Z"
slides: 5
status: exploratory_update
---

# September 29 research update

Open [research-update.pdf](research-update.pdf). This is a new five-slide,
plain-language update for the user and an advisor. The two September 25 decks
are historical records and are not changed. The displayed cutoff is 01:14 UTC;
the underlying fixed snapshot is 01:14:27 UTC.

The story is: teach usable steps; show a same-answer history repair; report
three small repair comparisons; distinguish the familiar-goal RL gain from
its higher cost; ask whether the repair transfers and whether a helper benefits
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
      PYTHON=/project/alex_phd/envs/rlm-advisor-figures-20260911/bin/python

Compiler and renderer provenance is recorded in the
[existing tooling note](../2026-09-11-advisor-meeting/TOOLS.md).
The new deck uses Tectonic 0.17.0 and PyMuPDF 1.26.4 for CPU page inspection.
The verification receipt records the final PDF, note bounds, source hashes and
preservation of both earlier decks. Compilation and PDF inspection do not
establish GUI behavior on a particular laptop.

## Scope

This is a fixed-cutoff documentation update, not a new experiment or reanalysis
of model trajectories. It uses completed native-audited findings. No GPU jobs,
queue owners, experimental sources, datasets or earlier decks are changed.
The brainstorming skill kept the update within the agreed five-slide story;
verification checks cover the rendered pages and bounded notes.
