# Present with notes on a one-screen laptop

The audience PDF is `research-update.pdf`. It contains **only slides**, not
speaker notes. Send that PDF when sharing the deck with your advisors.

The short notes live in `speaker-notes.json`; the portable pdfpc version is
`research-update.pdfpc`. The longer [speaker guide](speaker-guide.md) explains
the examples, figures, and likely questions in more detail.

## Start here

Install pdfpc on the laptop. On Debian or Ubuntu:

```bash
sudo apt-get install pdf-presenter-console
```

On macOS with Homebrew:

```bash
brew install pdfpc
```

These are the [upstream installation instructions](https://github.com/pdfpc/pdfpc#installation).
Use pdfpc 4.6 or newer. The launcher also needs Python 3.10 or newer and `make`.
No GPU, training environment, LaTeX installation, or research-store access is
needed to present the included PDF.

From the repository root:

```bash
make -C slides present
```

**Then click the presenter window and press `w` to fill your laptop screen.**
pdfpc initially opens this window at half size, which is too small for the notes.
The checked layout fits at an effective resolution of 1280×720 or larger. The
audience window remains separate when you enlarge the presenter view.

[Preview the tested presenter view](presenter-preview.png): the current slide
is on the left, with the next slide and notes on the right. This is the
presenter view, **not** the window to share with the audience.

This opens two windows on your laptop: the audience slides and your presenter
console, which includes the current slide, next slide, timer, and notes. For a
video meeting, **share only the audience slide window**. Do not share the entire
desktop or the presenter window. Check the meeting application's share preview
before starting, and keep the audience window open rather than minimized.
Arrange the two windows as you prefer; the notes view can occupy most of the
screen while the audience window remains available to the meeting application.

If the audience is looking at the same physical laptop screen, or a projector
mirrors that entire screen, private notes are not possible on that shared view.
Use window sharing, an extended second display, or separate printed/device notes.

For private rehearsal with only the presenter console:

```bash
make -C slides rehearse
```

This rehearsal mode opens fullscreen directly.

For an extended laptop-plus-projector setup:

```bash
make -C slides present-dual
```

The laptop is normally the presenter screen; use the display settings to extend,
not mirror. Run `pdfpc --list-monitors` if you need to diagnose screen selection.
This dual-display mode is optional; the default `present` target is intended for
your one-screen laptop.

## Readable notes and useful keys

Each slide has three short paragraphs: **SAY**, **IF ASKED**, and **CAUTION**
(**DISCUSS** on the final slide). These are cues, not a script to read aloud.
The launcher gives the notes more room by reducing the next-slide preview.
It starts with a 15-minute timer; discussion can continue after that timer.

- Right arrow or Space advances; Left arrow goes back.
- `g` lets you jump to a slide number. Tab opens the slide overview.
- `+` and `-` adjust note size while in normal mode; `1` returns to normal mode.
- Shift+C opens layout adjustment; drag the dividers, then press Escape.
- `w` toggles presenter fullscreen; Shift+W toggles audience fullscreen.
- `p` pauses or resumes the timer. Ctrl+Q exits.

If text feels cramped on your laptop, first enlarge the presenter window or
give notes more room with Shift+C. Do not shrink the audience slides to fit
speaker notes. Large desktop scaling can require a different layout; rehearse
on the actual laptop before the meeting.

For a different timer, run from this directory:

```bash
python3 present.py --duration 20
```

The [upstream manual](https://github.com/pdfpc/pdfpc/blob/master/man/pdfpc.in)
documents window modes, notes, and keyboard controls.

## Edit and keep notes in sync

Edit `speaker-notes.json`, then run `make notes`. Every note includes its slide's
exact title, so a changed slide order cannot silently attach the wrong notes.
Keep the cues short; move extended explanations to `speaker-guide.md`.

The launcher uses a fresh working copy under `.presenter/session-*/` so that
pdfpc's saved state or live note edits do not overwrite the published notes.
If you edit notes with Ctrl+N in pdfpc, that working copy is retained; transfer
any changes you want to keep into `speaker-notes.json`. Each launch prints the
working file's location. These local session files are not pushed to GitHub.

The laptop layout is in `pdfpcrc`. It is copied into a session-local configuration
directory; your global pdfpc configuration is left unchanged. This also means
your personal pdfpc customizations are not loaded by this launcher.

After changing visible slides, compile with `make` or `make tectonic`. Both also
regenerate the portable notes. `make check` checks the PDF, slide-note mapping,
and text bounds; visual inspection remains necessary for overlap and readability.

To open the portable files directly, without the launcher:

```bash
pdfpc --windowed=both --note-format=plain research-update.pdf
```

Keep `research-update.pdfpc` beside the PDF. This direct command uses pdfpc's
default layout and may save changes into that sidecar; the `make present`
launcher is the preferred laptop setup.

## Ongoing research updates

At each completed-result review, ask whether the finding changes the main
conclusion, a necessary qualification, or the next research question. If it
does, update the relevant slide and these notes together, along with the guide,
evidence links, numerical data, and evidence cutoff. Otherwise keep the result
in the supporting report. An active training process is not itself a finding.

Recompile and inspect changed slides and notes before pushing a new version.
Prefer replacing a weaker figure or clarifying an existing message over adding
another slide. Preserve readable complete sentences and essential caveats in
the audience PDF: it should make sense even without this presenter console.
