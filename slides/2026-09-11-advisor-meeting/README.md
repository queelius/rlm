---
title: Advisor research discussion — 11 September 2026
status: reviewable_draft
evidence_cutoff_utc: 2026-09-11T12:55:00Z
main_slides: 9
optional_backup_slides: 6
talk_minutes: 10
---

# Start here

[Read the slides](research-update.pdf). Present the first nine pages, then stop
for discussion. The remaining six are optional answers to questions, clearly
marked “Backup.” The earlier deck in the parent directory is unchanged.

- [Learn the current talk, with timing and questions for each slide](speaker-guide.md).
- [Understand SFT through a simplified example and an actual training record](sft-worked-example.md).
- [Open pdfpc with private notes on a one-screen laptop](PRESENTING.md).
- [Read the full evidence and methods](evidence-and-methods.md).
- [Consider the possible publication paths](publication-options.md).
- [Find additional experimental checks](later-findings.md).
- [Consult the longer explanations from the expanded draft](detailed-findings-guide.md).

The [Beamer source](research-update.tex), [portable numerical data](data/claims.json),
and [figure-building script](figures.py) are included.

## The story

Our earlier talk showed that worked examples could teach a basic Python routine.
This update asks whether better handoffs can make a more capable RLM reliable.
A visual example shows the larger idea: divide a long input into focused reading
questions, then use Python to combine the answers. The visible task is to count
each user's questions asking for a location; a three-row Ada/Ben example makes
both the helper's job and the final counts explicit.

Training improved a more demanding routine on newly selected records. Separately,
matching arbitrary names beside records and helper answers improved reading
accuracy in Qwen and Mistral without retraining. That motivates a specific
proposed RLM change: let the surrounding program preserve record–answer links
through splitting and recombination, and make the number of questions per helper
call an explicit choice. A same-work comparison found that small calls were
faster locally, while single-question calls repeated more input text. The next
test must measure that tradeoff rather than assume a more elaborate method wins.

The contribution would not be inventing identifiers. It would be showing when
a handoff procedure fails, how a targeted change helps, and whether it improves
complete answers at a useful cost. That last claim remains open.

## What is in the main talk and the backups?

The main talk contains a brief recap, one RLM picture, a worked SFT example, a training result, a
two-record example, the two-model matching result, the same-work accuracy/time
comparison, a proposed RLM change with its decisive test, and a discussion question.

The backups explain the training task, the matching-versus-different-name control,
longer reward training, new combinations of familiar operations, misleading
record names, and the full three-format batch-size comparison.

Two newly reviewed training findings are preserved without extending the talk:

- New combinations: correct answers **and requested calculations** increased
  from 2/72 to 29/72; missing-result bounds are 2–13 and 29–30. Instructions gave
  the plan, so this is execution transfer, not autonomous planning.
- Longer reward training: final-answer counts were 57/72 before training,
  55–57/72 with answer reward, and 54/72 with an extra calculation check.
  This is a different metric; the full review of the calculations is unfinished.

The counting studies use public TREC question texts with artificial users and
weights. The matching studies use MultiNLI reading judgments. They are different
tasks, not one shared benchmark score. Most work uses Qwen3-4B-Instruct-2507;
the family comparison uses Mistral-7B-Instruct-v0.3. Training uses small adapters,
not training a model from scratch.

## Present on one laptop screen

If you are already in this directory:

    make present

For private rehearsal here, use `make rehearse`.

From the repository root:

    make -C slides present

Click the presenter window and press `w` to expand it. Share only the audience
window, never the whole desktop. The timer starts at ten minutes. For private
rehearsal:

    make -C slides rehearse

The notes are separate from the audience PDF. [Presentation instructions](PRESENTING.md)
explain installation, window sharing, controls, and the limits of a mirrored screen.

## Build on another machine

The vector figures are checked in, so compilation does not need Python plotting
packages or the GPU experiment store. From this directory:

    make

This uses latexmk and ordinary Beamer, TikZ, Latin Modern, and booktabs.
Python 3.10+ regenerates the private pdfpc metadata. Alternatively:

    make tectonic

From the repository root, `make -C slides meeting` builds this deck.
To regenerate figures, use the versions in requirements-figures.txt and run
`make figures`. To check the PDF and render every page, run `make check`.

The data are portable. When original experiment files are available, the figure
script verifies their recorded hashes; elsewhere it reports which source files
could not be checked. Rebuilding a figure is not an independent experimental audit.

## Review and updates

[VERIFICATION.md](VERIFICATION.md) records actual compilation and visual checks.
The design uses complete-sentence headlines and evidence that is visible on the
slide, following [Michael Alley's presentation guidance](https://www.assertion-evidence.org/tutorial.html).
Secondary details belong in notes and the guide.

When a late result changes the interpretation, update the figure, numerical
data, notes, guide, and evidence cutoff together. Recompile and inspect the
audience PDF and the real presenter view before pushing. Prefer replacing
redundant content or using a backup over lengthening the main talk.
