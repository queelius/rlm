# Five-slide R1-Zero-style reproduction update

`research-update.tex` is the 16:9 Beamer source. `results.tex` holds the numbers
for the checked final comparison. `speaker-guide.md` explains the ideas
in plain language; `speaker-notes.json` holds concise laptop presenter notes.
The deck reports three short training runs and all eight model/input combinations.
Chat-style training scored 308/500 in chat style and 317 with questions alone;
question-only training scored 168 and 314. A fresh repeat scored 166 and 306.
The starting model scored 154 and 305.
The original separate 128-question result (41 to 80) remains in the guide and
evidence. The two question-only training runs improved by nine and one answer;
we have not established a reliable gain beyond that stronger starting setup.
It is not a reproduction of the paper's full benchmark scores.

At the 17:45 UTC update, a separate check of the authors' released model scored
366/500 (73.2%), compared with their reported 74.2%. Those are the authors'
weights, not a model trained by us. Slide 1 identifies that check separately.
Slide 5 shows the ongoing longer run: 20, 42, and 42 correct out of the same
64 questions before training, after 32 updates, and after 64 updates.
Five answers improved and five regressed between the last two points.
These are progress checks, not a final 500-question result.
The third attempt failed after 12 updates without a saved checkpoint. A fourth
attempt uses an isolated memory repair and more frequent saving, with the same
4,080-question and 255-update budget. Slide 3 states those planned figures. The
short-run table is unchanged. Read the linked research record for later live
status rather than treating this dated PDF as a process monitor.

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
