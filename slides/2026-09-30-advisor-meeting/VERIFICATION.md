# Latest ten-slide revision: concrete examples and broader-dataset lessons

The audience PDF now has ten slides. Added an invented MuSiQue question and
its two smaller questions, and an invented financial calculation. The crafting
example shows both the task and the change to its training history. Simplified
examples are explicitly labeled and are not presented as actual model returns.

- Tectonic compilation succeeded, with no overfull or underfull boxes.
- All ten pages have their text inside the page bounds. All pages were reviewed
  visually; the final question-plan and arithmetic pages were rechecked at full
  resolution after wording changes. No clipping or overlap was found.
- Slide 4 now spells out what the learner sees and the unchanged action in
  each example, with the main teaching lesson stated on the audience slide.
  A first expanded layout overflowed; the final two-column layout was rebuilt
  and visually checked, with no overfull boxes or out-of-page text. Its matching
  eight-line note fits a 16-point, 1366×768 static preview.
- Current PDF SHA256:
  `0e710f910be4b084eeb7e5d1f6d5798ed80a04684fc07d570e74758467ed35ca`.
- The reading guide, dataset Q&A, evidence, and presenter notes distinguish
  programmed splits, learned question lists, and learned crafting actions.
- Final-page previews are in `/tmp/rlm-final-examples-pm_tq30h/`.
  The subsequent slide 4 and notes preview is in `/tmp/rlm-slide4-clear-hf4sx5ki/`.
- This verification is local. The managed session cannot push to GitHub.
  Rerunning `publish-main.sh` from the user's SSH terminal publishes the scoped
  documents and explicitly stages the compiled PDF despite the ignore rule.

## September 29 afternoon revision (earlier eight-slide draft)

Compiled and visually reviewed at approximately 14:04 UTC. This local revision
has not been committed or pushed: Git metadata and the external research store
are read-only under the current managed-session permissions.

- Eight-page Beamer PDF, SHA256:
  `e9d6664abaf5d97899f8af298c0e305e0bdfe229daa8073b8171e5b52f3afed1`.
- Final Tectonic build exits successfully with no overfull boxes. The first
  revised compile exposed crowded pages; shorter wording fixed them without
  shrinking the body font. The independent-reader revision initially added a
  line on slide 3, which was shortened and rebuilt before this final check.
- Main visually inspected all eight final pages in a contact sheet and the
  revised training-example, execution-assistance, RL and helper pages at full
  resolution. No overlap or clipping found; all extracted text spans stay
  within page bounds. PDF ligatures were normalized for title matching.
- An independent reader reviewed all eight preceding renders and the guide.
  The final version incorporates its essential fixes: explicitly SFT-trained
  baseline, unequal-dose control, and the helper screen's unresolved outcome.
- Eight matching pdfpc notes; at most nine wrapped lines. All fit in static
  1366×768 presenter previews at 16 points. Main inspected the final page's
  preview. This is not a live pdfpc GUI test.
- Eleven evidence source hashes checked. The raw RL counts are 4/4/5 of 16;
  extra SFT is 4/16. The ingredient comparison is 323/381 of 768. The guide
  additionally records assisted RL2 at 4/16, completed 14:02 UTC, without
  changing the audience PDF's explicit 13:26 snapshot.
- The accompanying retrospective checks 384 small native receipt bindings
  across eight diagnostic cells, retains all fixed checkpoints and provides
  ten structured claims with evidence, limitations and next decisions.
- Local inline-link scan: 134 links resolve directly; eight preexisting
  external-store links in the research index resolve from the main checkout
  rather than the deeper worktree. All ten claims' evidence paths resolve.
- `git diff --check` passes. No earlier deck, GPU owner, sealed source,
  training input or scientific queue was changed.

Temporary final previews: `/tmp/rlm-advisor-final-j6cxa94e/`. The durable
deliverables are the PDF, source, notes, guide and pinned evidence in this folder.

## Earlier published draft verification

Visually checked 29 September 2026, approximately 06:14 UTC, then rebuilt and
rechecked before publication at 06:23 UTC. This is a draft for the
September 30 meeting, not a promise that no subsequent result will change it.

- Eight-page native Beamer PDF. Final PDF SHA256:
  `7a6fd1720bb4ea4b77fd6d8aa7a987bc8913d7b43350116ac1224a77e82998e7`.
- Built with Tectonic 0.17.0; final log has no overfull boxes.
- All text spans remain within page bounds. Main reviewed all eight rendered
  slides, then reinspected the materially changed pages 4, 6, and 7.
- An independent reviewer read all eight slide renders, the source and guide.
  Its five substantive clarity/claim recommendations were incorporated.
- `make notes-check` passes: eight matching titles, at most nine actual note
  lines, wrapped at 42 characters; configured maximum 14 lines.
- All eight static slide-plus-notes previews fit notes at 16 points. The longest
  final-page preview was visually inspected. This is not a live pdfpc GUI test.
- Teaching chart and RL table were read directly from the committed analysis
  JSON, with complete native-outcome denominators checked. Exact source hashes
  and values are in `evidence.json`.
- Earlier September 29 and September 25 decks were not modified.
- Source JSON, guide links, Makefile and ordinary staged whitespace checked
  before publication. No model, GPU job, scientific queue or environment changed
  to build these slides.

## Commands

```sh
make -C slides/2026-09-30-advisor-meeting tectonic notes notes-check \
  TECTONIC=/project/alex_phd/research-cache/tools/tectonic-0.17.0-musl/tectonic \
  PYTHON=/project/alex_phd/envs/rlm/bin/python
```

PyMuPDF from the existing advisor-figures environment rendered the PDF and
checked every text span. Rendered PNGs and compiler logs remain generated local
artifacts; the portable PDF, source, notes, guide and evidence are published.
