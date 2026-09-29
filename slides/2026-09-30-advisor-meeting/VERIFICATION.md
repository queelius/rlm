# First readable draft verification

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
