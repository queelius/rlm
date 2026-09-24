# Teaching a model to find what it needs

For the brief, informal conversation with Fujinoki, use the
[new four-slide discussion deck](../2026-09-25-fujinoki/README.md).
This five-slide version is preserved as the more detailed numerical presentation.

Five-slide, plain-language advisor discussion for September25. Evidence cutoff:
September24, 18:30 UTC. This is an exploratory result, not a finished paper.

Open `research-update.pdf`. Source is `research-update.tex`; no Python or external
figure files are needed to compile it. Run `make` with latexmk, or `make tectonic`.

For a one-screen laptop with pdfpc installed, run `make present` from this
directory. It opens audience and speaker windows; share **only the audience
window**, not the desktop. Other people looking directly at the same physical
screen can still see your notes. `make rehearse` opens the single-screen presenter
view for private practice. These commands reuse the earlier deck's pdfpc4.6 flags.

The short notes are in `research-update.pdfpc` and editable source
`speaker-notes.json`. The longer explanations are in [speaker-guide.md](speaker-guide.md).
The notes are supplementary: the PDF contains the essential claim and limitations.

## Why this result earns a short presentation

The teaching advantage repeats under two training seeds after correcting the
known quantity error. Its direction also repeats in three changed task worlds,
although those changed-world comparisons use the uncorrected teacher. The new
world47 control shows4/16 versus6/16 in the first repeat and1/16 versus8/16 in
the second. The first difference is uncertain; the second is clearer. That is
worth a preliminary discussion, not enough to isolate the cause or claim
general-purpose planning. We do not claim recursion helped.

Both training repeats of this control are complete. Four completed tool-change
comparisons appear on slide4:6 to12,8 to10,9 to14 and9 to13 successes out of16.
The fourth comparison completes both models in both recipe settings; see the
[new result receipt](fourth-tool-result.md). Never show partial scores as final results.

## Evidence

- [Numerical reports and source hashes](../../experiments/selective_delegation/overnight-results-20260924.json).
- [Teaching and memory findings](../../experiments/selective_delegation/RESEARCH-UPDATE-20260924.md).
- [Completed RL and memory update](../../experiments/selective_delegation/MORNING-UPDATE-20260924.md).
- [Completed changed-world control](../../experiments/selective_delegation/WORLD47-CONTROL-20260924.md).
- [Recipe-execution results and limitations](../../experiments/selective_delegation/RECIPE-BINDING-RESULTS-20260924.md).
- [Mechanism ideas and limitations](../../experiments/selective_delegation/RL-CREDIT-ASSIGNMENT-20260923.md).

## Five-slide story

1. Why divide the work between a language model and its tools? Why use a game?
2. What does a training example look like, and how do the two teachers differ?
3. Did the training examples change success, including when recipes changed?
4. Did letting code handle ingredients help without extra training?
5. What have we learned, what is still unproven, and what should we test next?

Slide3 retains the bar chart (5/15 and3/10, scale0–32) and summarizes the changed
recipes comparison. Slide4 includes the complete four-cell tool comparison.
Slide5 gives the takeaway and invites discussion, rather than ending with a table.
There are exactly five pages, with no backup slides.

Compiled with the previously verified Tectonic0.17.0 installation; all five
pages were rendered and visually inspected. An initial slide4 overflow was fixed.
The final log has no overfull boxes. Presenter notes use42-character lines,
at most13 lines per slide. This deck's pdfpc window has not been re-tested live;
its launch flags and note format reuse the previously checked deck.
