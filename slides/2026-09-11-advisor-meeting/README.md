---
title: Advisor research discussion — 11 September 2026
status: reviewable_draft
evidence_cutoff_utc: 2026-09-11T00:15:00Z
---

# Start here

This package explains the research for both the presenter and colleagues who
are new to it. It separates what we observed from what we hope to achieve next.

- [Read the 13-slide PDF](research-update.pdf).
- [Learn the material with the slide-by-slide guide](speaker-guide.md).
- [Read the fuller evidence, methods, and unsuccessful experiments](evidence-and-methods.md).
- [Compare two possible publication paths](publication-options.md).

The [Beamer source](research-update.tex), [figure data](data/claims.json),
and [figure-building script](figures.py) are included. The older deck in the
parent directory is preserved as a historical update.

## The story in three sentences

Examples taught a small main model to choose and carry out a useful calculation
more reliably. Matching input records to output answers also produced a large,
replicated improvement in a separate batched reading task. But better pieces
did not consistently produce the right whole answer, which gives us a focused
next research question.

Most experiments use Qwen3-4B-Instruct-2507. The matching-tag check also uses
Qwen3-8B. We used small adapter updates for training,
not training from scratch. The counting studies use public TREC question texts
with artificial users and weights; the record-matching studies use MultiNLI
reading-comprehension judgments. These are different experiments, not one
shared benchmark score.

## How to prepare for the meeting

Read the guide alongside the slides first. It includes a two-minute overview,
worked examples, an explanation of each figure, and answers to likely questions.
The evidence document is a reference, not required slide narration.

Allow roughly 10–15 minutes for the slides if all are discussed. For a shorter
update, emphasize slides 2–3, 5–9, and 13, and use the guide to answer questions.
Leave time to discuss which result would benefit most from a different model family,
a new task, or a clearer explanation of the failure.

## Build on another machine

The vector figures are included, so ordinary compilation does not need Python
or the GPU experiment store. From this directory:

    make

This uses latexmk and a standard LaTeX installation with Beamer, TikZ,
Latin Modern, and booktabs. Alternatively:

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

The deck includes completed, reviewed results available by the cutoff above,
including two separately prepared training-example sets and the matching-tag
comparison on two models. The new training run completed all six updates but
has two missing evaluation outcomes; these are disclosed, not filled in.
The smaller-update reward-training comparison and the whole-task matching-tag
test remain separate ongoing work, not findings in this version.

Do not silently replace a figure when a new result arrives. Update its numerical
evidence, interpretation, guide, and cutoff together; then rebuild and inspect
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
