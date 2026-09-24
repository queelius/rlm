# Teaching a model to find what it needs

Five-slide, plain-language advisor discussion for September25. Evidence cutoff:
September24, 16:25 UTC. This is an exploratory result, not a finished paper.

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

Both training repeats of this control are now complete. The next pilot tests
automatic ingredient binding from already observed recipes. Update slide5 and
its notes together once its native audit is available; never show a partial
score as a final result. Preserve this cutoff and the historical deck separately.

## Evidence

- [Numerical reports and source hashes](../../experiments/selective_delegation/overnight-results-20260924.json).
- [Teaching and memory findings](../../experiments/selective_delegation/RESEARCH-UPDATE-20260924.md).
- [Completed RL and memory update](../../experiments/selective_delegation/MORNING-UPDATE-20260924.md).
- [Completed changed-world control](../../experiments/selective_delegation/WORLD47-CONTROL-20260924.md).
- [Mechanism ideas and limitations](../../experiments/selective_delegation/RL-CREDIT-ASSIGNMENT-20260923.md).

Slide4 now shows world47, not the earlier A/B/C table. The bar chart on slide3
uses corrected-original/public counts5/15 and3/10; its
horizontal scale runs from0 to32. There are exactly five pages, with no backup slides.

Compiled with the previously verified Tectonic0.17.0 installation; all five
pages were rendered and visually inspected. An initial slide4 overflow was fixed.
The final log has no overfull boxes. Presenter notes use42-character lines,
at most13 lines per slide. This deck's pdfpc window has not been re-tested live;
its launch flags and note format reuse the previously checked deck.
