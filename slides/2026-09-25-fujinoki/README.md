# Informal discussion with Fujinoki

Four slides for roughly five minutes, followed by discussion. Designed to be
read without narration by someone unfamiliar with AI/ML. Evidence cutoff:
September 24, 18:30 UTC. The [detailed five-slide deck](../2026-09-25-advisor-meeting/README.md)
is preserved separately.

Open **research-update.pdf**. This version deliberately keeps experimental counts
off the audience slides; bounded pdfpc notes and the [speaker guide](speaker-guide.md)
provide the numbers, qualifications, and evidence links if asked.

## Presenting on one laptop screen

From this directory, run `make present`. It opens audience and presenter windows
with a five-minute timer. Share only the audience window, not the desktop.
If someone can see your physical screen, they can still see the notes window.
Use `make rehearse` for private single-screen presenter practice. The PDF also
works by itself in any PDF viewer.

Build with `make` (latexmk) or `make tectonic`. The deck has no external figures.
Portable pdfpc notes are checked in alongside editable `speaker-notes.json`.
The launch options reuse the previous deck's setup; no live laptop GUI test was
performed on the cluster.

## Story

1. We want models to find steps and complete tasks; a game lets us check success.
2. RL produced some gains, but has not reliably improved planning.
3. Teaching discovery and assigning exact execution details to code both helped.
4. Test their interaction, then ask whether it makes RL and delegation easier.

The distinction between measured results and future hypotheses must remain visible.

Verified: four-page PDF compiled and visually inspected, no overfull boxes,
all evidence links resolve, and all four notes match their slides. Notes are
wrapped to42 characters and at most12 lines each. The detailed deck is unchanged
apart from a pointer to this discussion version.
