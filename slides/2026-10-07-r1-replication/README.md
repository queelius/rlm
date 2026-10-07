# Lessons from math RL for training RLMs

Read the [six-slide PDF](research-update.pdf), the
[speaker guide](speaker-guide.md), or the
[self-contained overview](../../docs/r1-replication-2026-10-07/START-HERE.md).
The [learning guide PDF](../../docs/r1-replication-2026-10-07/learning-guide.pdf)
retains the detailed experiments and adds the RLM research discussion.

## Open the speaker and audience views on your laptop

Run these commands **locally, not inside the SSH session**, from your local
`rlm` repository. The PDF and slide-by-slide notes are already included;
you do not need LaTeX, Python, or the research environment to present.

```sh
git pull --ff-only origin main
make -C slides/2026-10-07-r1-replication present
```

The Makefile launches **pdfpc** with two movable windows: audience slides,
and your presenter view with talking points, slide previews, and a timer.
Share only the audience window in a video meeting. With a projector, use
extended displays rather than mirroring, and move the audience window onto it.

For private rehearsal with just the presenter view:

```sh
make -C slides/2026-10-07-r1-replication rehearse
```

Install pdfpc once if it is missing: on Ubuntu/Debian,
`sudo apt-get install pdf-presenter-console`; on macOS with Homebrew,
`brew install pdfpc`. See the [official installation instructions](https://github.com/pdfpc/pdfpc#installation)
for other systems. The launch targets do not install software automatically.

Use the arrow keys to change slides, `+`/`-` to adjust note size, and
`Ctrl+Q` to quit. The six-minute timer is only a guide; it does not advance
slides. These controls and launch options are documented in the
[pdfpc manual](https://github.com/pdfpc/pdfpc/blob/master/man/pdfpc.in).
The [speaker guide](speaker-guide.md) has longer explanations for preparation.

Keep `research-update.pdf` and `research-update.pdfpc` together. The latter
contains the private notes; opening the PDF in an ordinary viewer shows only
the audience slides. pdfpc may save local note/settings changes to that file;
keep any personal edits before pulling later updates.

## Research summary

The small-scale reproduction occupies one slide. The rest explains useful
reward variation, curriculum, model/interface fit, and a proposed sequence for
teaching RLM tool use and decomposition, followed by a portfolio of small tests.
Diagrams and examples distinguish
observed results from hypotheses.

Updated: **7 October 2026, 20:56 UTC**. Training was stopped by choice after
219 updates; checkpoints are retained and final evaluations were canceled.
The fixed 64-question monitor
scores 20, 42, 42, 43, 43, 48, 47 through update192. The latest check has one
gain and two regressions. This is a successful small-scale reproduction of
RL-driven improvement, not a match to the paper's full benchmark result.
The full500 prompt comparison remains 154 to308 for chat input and305 to317
for questions alone, using the same short-run weight pair. All other outcomes
and authors' reference scores remain in the learning guide.

`research-update.tex` is the 16:9 Beamer source. `speaker-notes.json` supplies
concise laptop presenter notes. The RLM curriculum is a proposed follow-up,
not a result of the stopped math training run.

## Rebuild after editing

From this slide directory, build on a laptop with LaTeX installed:

```sh
make
make notes-check
```

The existing cluster compiler can be used without installation:

```sh
make tectonic TECTONIC=/project/alex_phd/research-cache/tools/tectonic-0.17.0-musl/tectonic
make notes notes-check
```

Rebuild notes after changing their text or slide titles. Presenting the committed
files does not require any of these build commands.

Result updates must follow `evidence.md`; change visible slides, notes, guide,
evidence, and cutoff together. No result should be inferred from a running job.

For CPU rendering and text-bound checks with the existing presentation environment:

```sh
/project/alex_phd/envs/rlm-advisor-figures-20260911/bin/python verify_deck.py
```

Inspect all six PNGs in `rendered/` after rebuilding. This checker does not
establish scientific validity and does not replace visual inspection.
