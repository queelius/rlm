# Informal discussion with Fujinoki

Five short slides for roughly five minutes, followed by discussion. Designed to
make sense without narration for someone unfamiliar with AI/ML. Evidence cutoff:
September 24, 20:29 UTC. The [detailed five-slide deck](../2026-09-25-advisor-meeting/README.md)
is preserved separately.

Open **research-update.pdf**. Each result is labeled in plain language on the
slide; the bounded pdfpc notes and [speaker guide](speaker-guide.md) add context,
qualifications, and evidence links.
Start with [TextCraft explained](TEXTCRAFT-EXPLAINED.md) for a plain-language
walkthrough of what the model does and what the experiments establish.

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
3. Compare two ways of making teaching examples: plan from known recipes, or discover them.
4. Show how code can fill in exact ingredients after the model chooses what to make.
5. Explain the current comparison, proposed RL follow-up, and later delegation research.

Slides 3 and 4 use simplified examples to make the two research ideas concrete.
Their numbers are tied to named conditions so a reader does not need to infer
what each side of a comparison means. The first additional-goal result and its
limitations are documented in [the result note](new-goal-result.md).

The distinction between measured results and future hypotheses must remain visible.

Verified: five-page PDF compiled and visually inspected, no overfull boxes,
all evidence links resolve, and all five notes match their slides. Notes are
wrapped to 42 characters and at most 18 lines each. The detailed deck is unchanged
apart from a pointer to this discussion version.
