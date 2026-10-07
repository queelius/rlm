# Lessons from math RL for training RLMs

Read the [six-slide PDF](research-update.pdf), the
[speaker guide](speaker-guide.md), or the
[self-contained overview](../../docs/r1-replication-2026-10-07/START-HERE.md).
The [learning guide PDF](../../docs/r1-replication-2026-10-07/learning-guide.pdf)
retains the detailed experiments and adds the RLM research discussion.

The small-scale reproduction occupies one slide. The rest explains useful
reward variation, curriculum, model/interface fit, and a proposed sequence for
teaching RLM tool use and decomposition, followed by a portfolio of small tests.
Diagrams and examples distinguish
observed results from hypotheses.

Scientific cutoff: **7 October 2026, 20:22 UTC**. The fixed 64-question monitor
scores 20, 42, 42, 43, 43, 48, 47 through update192. The latest check has one
gain and two regressions. This is not a full500 result.
The full500 prompt comparison remains 154 to308 for chat input and305 to317
for questions alone, using the same short-run weight pair. All other outcomes
and authors' reference scores remain in the learning guide.

`research-update.tex` is the 16:9 Beamer source. `speaker-notes.json` supplies
concise laptop presenter notes. The longer training run and its fixed final
evaluations are unchanged; the RLM curriculum is a proposed follow-up.

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
window. `make rehearse` uses a single-screen presenter view. The six-minute timer
is advisory. Rebuild notes after changing their text or slide titles.

Result updates must follow `evidence.md`; change visible slides, notes, guide,
evidence, and cutoff together. No result should be inferred from a running job.

For CPU rendering and text-bound checks with the existing presentation environment:

```sh
/project/alex_phd/envs/rlm-advisor-figures-20260911/bin/python verify_deck.py
```

Inspect all six PNGs in `rendered/` after rebuilding. This checker does not
establish scientific validity and does not replace visual inspection.
