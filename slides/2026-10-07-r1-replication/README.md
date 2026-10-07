# Five-slide R1-Zero-style reproduction update

`research-update.tex` is the 16:9 Beamer source. `results.tex` holds the numbers
and the explicit pending/final switch. `speaker-guide.md` explains the ideas
in plain language; `speaker-notes.json` holds concise laptop presenter notes.
The draft includes the completed pilot and clearly labels the larger comparison
as pending. It is not a reproduction of the paper's full benchmark scores.

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
