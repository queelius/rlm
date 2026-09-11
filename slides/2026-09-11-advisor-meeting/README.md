---
title: Advisor research discussion — 11 September 2026
status: reviewable_draft
evidence_cutoff_utc: 2026-09-11T03:45:00Z
---

# Start here

This package explains the research for both the presenter and colleagues who
are new to it. It separates what we observed from what we hope to achieve next.

- [Read the 14-slide PDF](research-update.pdf).
- [Present with pdfpc and private notes on one laptop screen](PRESENTING.md).
- [Learn the material with the slide-by-slide guide](speaker-guide.md).
- [Read the fuller evidence, methods, and unsuccessful experiments](evidence-and-methods.md).
- [Compare two possible publication paths](publication-options.md).
- [Read supporting checks for questions and discussion](later-findings.md), including format validity and a first whole-task attempt.

The [Beamer source](research-update.tex), [figure data](data/claims.json),
and [figure-building script](figures.py) are included. The older deck in the
parent directory is preserved as a historical update.

## The story in three sentences

Examples taught a small main model to carry out useful calculations more reliably,
and two trained versions retained that improvement on newly selected records.
Separately, matching names beside input records and answers kept reading accuracy
high as batches grew, with related improvements in three models.
The next question is whether better helper answers yield more correct complete
answers when the main model combines them.

Most experiments use Qwen3-4B-Instruct-2507. The matching-tag check also uses
Qwen3-8B and Mistral-7B-Instruct-v0.3. We used small adapter updates for training,
not training from scratch. The counting studies use public TREC question texts
with artificial users and weights; the record-matching studies use MultiNLI
reading-comprehension judgments. These are different experiments, not one
shared benchmark score.

## How to prepare for the meeting

Read the guide alongside the slides first. It includes a two-minute overview,
worked examples, an explanation of each figure, and answers to likely questions.
The evidence document is a reference, not required slide narration.

Allow roughly 10–15 minutes for the slides if all are discussed. For a shorter
update, emphasize slides 2–3, 5, 7–9, and 14, and use the guide to answer questions.
Leave time to discuss which result would benefit most from more model families,
a new task, or a clearer explanation of the failure.

For the one-screen laptop, run `make -C slides present` from the repository root.
Click the presenter window and press `w` so the notes fit the full screen.
Share only the audience slide window, not the whole desktop. `make -C slides rehearse`
opens the presenter console alone. The [presentation instructions](PRESENTING.md)
cover installation, note size, controls, and the limits of a mirrored screen.
The [short per-slide cues](speaker-notes.json) supplement the longer guide; they
are not embedded in the audience PDF.

## Build on another machine

The vector figures are included, so ordinary compilation does not need Python
or the GPU experiment store. From this directory:

    make

This uses latexmk and a standard LaTeX installation with Beamer, TikZ,
Latin Modern, and booktabs. Python 3.10+ regenerates the pdfpc notes using only
its standard library. Alternatively:

    make tectonic

That uses Tectonic. To build from the repository root with the existing
slides Makefile:

    make -C slides meeting

To regenerate the figures, install the versions in requirements-figures.txt
in a separate Python environment, then run:

    make figures

To check the PDF's text bounds and render every page for inspection:

    make check

The checked-in numerical file is portable. On the research machine, the figure
script also verifies the listed original-source hashes. Elsewhere it explicitly
reports which originals are unavailable; it does not pretend that a local figure
rebuild is a fresh audit of the raw experiments.

## Evidence cutoff and ongoing work

The deck includes completed, reviewed results available by the cutoff above.
Slide 5 now shows the new-input training evaluation: 12 verified successes before
training versus 55 and 53 afterward, out of the same 72 questions. Missing outcomes
are disclosed. Slide 7 shows the new batch-size curve: untagged accuracy falls
from about 83% to 44%, while tagged accuracy stays near 85%.

The three-model comparison, matching-versus-different-tag control, helper-training
limitations, and earlier reward-training results remain. The new figures replace
two existing figures; the deck stays at 14 slides. Earlier results remain in the
evidence document and numerical file.

The attempted full-RLM matching test is inconclusive because most final outcomes
were unavailable. A small fixed-Python calculation improved, but that is not
successful use by the trained main model. The supporting note explains this.
New reward-training attempts are ongoing and are not included as findings.

Do not silently replace a figure when a new result arrives. Update its numerical
evidence, interpretation, pdfpc notes, guide, and cutoff together; then rebuild and inspect
the PDF. An interesting late result may deserve an extra slide rather than a
denser existing one.

## Verification and presentation choices

The [verification record](VERIFICATION.md) records compilation and visual review.
Full sentences carry the main messages, and scientific figures use visible
denominators and zero-based scales. Secondary details are in the guide rather
than crowded into slide bodies. This follows
[Michael Alley's presentation guidance](https://www.assertion-evidence.org/tutorial.html).

The [tooling record](TOOLS.md) documents the isolated CPU environment and compiler.
No training environment was changed to build this deck.
