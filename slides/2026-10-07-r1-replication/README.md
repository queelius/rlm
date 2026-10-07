# Five-slide R1-Zero-style reproduction update

`research-update.tex` is the 16:9 Beamer source. `results.tex` holds the numbers
for the checked final comparison. `speaker-guide.md` explains the ideas
in plain language; `speaker-notes.json` holds concise laptop presenter notes.
The deck reports two short training runs and all six model/input combinations.
Chat-style training scored 308/500 in chat style and 317 with questions alone;
question-only training scored 168 and 314. The starting model scored 154 and 305.
The original separate 128-question result (41 to 80) remains in the guide and
evidence. We are checking whether the small improvement over the stronger
question-only starting point repeats.
It is not a reproduction of the paper's full benchmark scores.

Build on a laptop with LaTeX installed:

```sh
make
make notes-check
```

The existing cluster compiler can be used without installation:

```sh
make tectonic TECTONIC=/project/alex_phd/research-cache/tools/tectonic-0.17.0-musl/tectonic
make notes notes-check
```

`make present` opens pdfpc audience and presenter windows; share only the audience
window. `make rehearse` uses a single-screen presenter view. The five-minute timer
is advisory. Rebuild notes after changing their text or slide titles.

Result updates must follow `evidence.md`; change visible slides, notes, guide,
evidence, and cutoff together. No result should be inferred from a running job.

For CPU rendering and text-bound checks with the existing presentation environment:

```sh
/project/alex_phd/envs/rlm-advisor-figures-20260911/bin/python verify_deck.py
```

Inspect all five PNGs in `rendered/` after rebuilding. This checker does not
establish scientific validity and does not replace visual inspection.
