# Advisor discussion, 30 September 2026

Start with the [audience PDF](research-update.pdf) and the separate
[reading guide](READING_GUIDE.md). The guide explains the whole story first,
then gives examples and answers to likely questions. It is not a script to rehearse.

The draft has **eight slides**, for a brief discussion of roughly 10 minutes.
The extra space explains the crafting task and the two training approaches;
it does not add more experimental headlines. The two main findings are:

1. Repairing what the model sees before the same demonstrated actions improved
   learning in three small comparisons.
2. Reward-training gains on familiar goals mostly did not carry to different
   goals in the completed test.

The helper comparison is a next experiment, not a claimed success. Details,
uncertainties, and source links are in the guide. [Evidence](evidence.json) pins
the numbers and cutoffs; [editorial workplan](WORKPLAN.md) records deadlines,
revision passes, and which new results would change the deck.

Earlier decks remain unchanged. This package is the working draft for tomorrow.

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

First readable draft compiled and visually reviewed on 29 September. All eight
slides fit, with no overfull boxes or out-of-page text. Notes match the slides,
use at most nine wrapped lines, and fit a 16-point static presenter preview.
No live pdfpc GUI test was performed on the GPU cluster. See
[verification and revisions](VERIFICATION.md).

Numerical findings are fixed to the cited September 29 reports. The current
experiment stream can change the story; no pending run is reported as a result.
The user's meeting is tomorrow at 3:30 pm, with a review-ready version requested
by 2 pm. Before that review, keep the PDF, notes, guide and evidence synchronized.
