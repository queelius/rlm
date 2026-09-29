# Advisor discussion, 30 September 2026

Start with the [audience PDF](research-update.pdf) and the separate
[reading guide](READING_GUIDE.md). The guide explains the whole story first,
then gives examples and answers to likely questions. It is not a script to rehearse.
The [advisor Q&A](ADVISOR_QA.md) adds the dataset map, scoring definitions,
RL lessons and questions about the proposed paper. If time is short, read the
guide's introduction and the Q&A's first answer before reviewing the PDF.

The draft has **ten slides**, including a new full-page MuSiQue example.
The book-author example explains learned question plans; the lantern example
explains learned tool actions. Slide 1 distinguishes both from programmed splits.
A financial-arithmetic example shows an alternative to adding helpers.
The talk can take roughly 12 minutes, or the deck can be read on its own.
The main findings are:

1. Repairing what the model sees before the same demonstrated actions improved
   learning in three small comparisons.
2. Code that fills already-known ingredient arguments improved success from
   about 42% to 50% in a broader comparison, without changing model weights.
3. The current reward-training sequence gives 4, 4, 5 and 4 successes out of 16
   different-goal attempts. The extra success did not persist. Other studies
   also show why we need fresh examples and an additional-SFT control.

The final slide proposes a bounded teaching-history study, not an established
general method. Slide 2 explains the learned-planning attempt and its limits.
Slide 9 explains lessons from programmed splitting, news classification and
conversation retrieval. The pending helper experiment is in the guide,
not a speculative result slide. Details,
uncertainties, and source links are in the guide. [Evidence](evidence.json) pins
the numbers and cutoffs; [editorial workplan](WORKPLAN.md) records deadlines,
revision passes, and which new results would change the deck.
The [historical research audit](../../docs/research-audit-2026-09-29/README.md)
recovers earlier positive results, failed approaches and the reasons for the
proposed next experiments.
The [focused paper assessment](../../docs/research-audit-2026-09-29/publication-assessment.md)
describes stronger baselines, genuinely new problems, prior work and decisions
that would narrow or retire the claim. Its experiment plan is not yet launched.

Earlier decks remain unchanged. This package is the working draft for tomorrow.
This is the portable advisor package: source, compiled PDF, presenter notes
and reading documents stay together. The publication target is GitHub `main`.
Research runtime work remains on its separate branch; publishing this package
does not move or change active GPU jobs. Raw runs and model checkpoints are not
backed up by publishing these documents.

## Present or read

From the repository root:

```sh
make -C slides/2026-09-30-advisor-meeting present
```

On one screen, pdfpc opens audience and presenter windows. Use the presenter
window for the short notes; share only the audience window. Press `w` in the
presenter window if its notes need fitting. The PDF is self-contained without
these notes. From inside this directory, use `make present`, without `-C slides`.

Compile with `make` if latexmk/LaTeX are installed, or `make tectonic notes`
with Tectonic. `make notes-check` checks note alignment and length.

## Current status

The draft is compiled and visually reviewed on 29 September. All ten
slides fit, with no overfull boxes or out-of-page text. Notes match the slides,
use at most nine wrapped lines, and fit a 16-point static presenter preview.
No live pdfpc GUI test was performed on the GPU cluster. See
[verification and revisions](VERIFICATION.md).

Numerical findings are fixed to the cited September 29 reports. The current
experiment stream can change the story; no pending run is reported as a result.
The user's meeting is tomorrow at 3:30 pm, with a review-ready version requested
by 2 pm. Before that review, keep the PDF, notes, guide and evidence synchronized.
