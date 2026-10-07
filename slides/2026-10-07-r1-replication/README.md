# Five-slide R1-Zero-style reproduction update

`research-update.tex` is the 16:9 Beamer source. `results.tex` holds the numbers
for the checked final comparison. `speaker-guide.md` explains the ideas
in plain language; `speaker-notes.json` holds concise laptop presenter notes.
The deck reports the completed 128-question comparison: 41 correct before
training and 80 afterward. The full500 and prompt controls are pending.
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
