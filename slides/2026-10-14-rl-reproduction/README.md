# From RL reproduction to RLM research

[Open the advisor slides](research-update.pdf). Five main slides and one optional
appendix explain the research arc, completed result, lessons and proposed RLM work.
Evidence cutoff: **14 October 2026, 18:10 UTC**.

Our own training reached **58.19%** on GSM8K, compared with the paper's **56.7%**
and our matched starting model's **39.77%**. The result recovers the paper's
performance level in one run with disclosed adaptations. It establishes a
working RL reference, not demonstrated transfer to RLM.

## Present

The PDF works in any PDF viewer. For slides plus private laptop notes, keep
`research-update.pdf` and `research-update.pdfpc` together and use the repository's
existing pdfpc setup. From your **local laptop checkout**, not the cluster terminal:

```sh
git pull --ff-only origin main
make -C slides/2026-10-14-rl-reproduction present
```

Share only the audience window. Use extended displays rather than mirroring with
a projector. For a single-screen presenter rehearsal:

```sh
make -C slides/2026-10-14-rl-reproduction rehearse
```

These targets require pdfpc already installed and do not install anything.
Opening the PDF alone does not show private notes. The nine-minute timer is a
guide, not automatic slide advancement. The appendix is for questions.

## Contents

- [Editable slide source](research-update.tex) and [curve data](learning-curve.csv).
- [Speaker guide](speaker-guide.md) for preparation.
- [Short presenter notes](speaker-notes.json), also compiled into the pdfpc file.
- [Evidence and interpretation](evidence.md), with [curve provenance](curve-provenance.json).
- [Verification and consolidation record](VERIFICATION.md).

This concise deck supersedes the older deck for this advisor narrative; it does
not replace the underlying research record.

## Rebuild

From this directory:

```sh
make all TECTONIC=/project/alex_phd/research-cache/tools/tectonic-0.17.0-musl/tectonic PYTHON=/project/alex_phd/envs/rlm-advisor-figures-20260911/bin/python
```

On another machine, set `TECTONIC` and `PYTHON` to the installed executables.
The notes build reuses `../2026-09-11-advisor-meeting/present.py`, so retain
the repository layout. Presenting the committed PDF and pdfpc files needs no
Python or TeX. Rebuild and inspect both slides and notes after editing.
