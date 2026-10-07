# Presentation verification

## Completed fixed128 comparison, 7 October, 10:24 UTC

- Tectonic built the updated five-slide PDF successfully, without overfull or
  underfull boxes. The five matching pdfpc notes pass the 14-line check.
- Fresh `verify_deck.py` run found five pages, matching titles, and all text
  within page bounds. Root visually inspected changed slides 3, 4, and 5.
  An independent reader inspected all five rendered pages and found no clipping,
  overlap, or important audience/scientific ambiguity.
- Both standalone evaluations exited successfully. Root matched all 128 paired
  questions/references against the frozen test and independently regraded all
  saved scores with the authors' full checker. Result: 41 to 80 correct, 44
  gains, five losses. Hashes and settings are in fixed128-scoring-receipt.json.
- Root and independent reader inspected actual test row 5. Its repeated
  fraction question became a division explanation ending in 0.15, as the slide
  summarizes. The invented slide-2 example remains clearly labeled.
- The updated five-page learning PDF passed text-bound checks. Its author
  inspected its visual layout, and root checked its changed source and counts.
- No live pdfpc GUI was tested. The saved notes were checked structurally.

The larger evaluation remains explicitly pending. One training run does not
establish new mathematical knowledge, a Dr. GRPO advantage over another
algorithm, or successful learning of recursive decisions.

## Earlier pilot package

7 October 2026, pilot final-rescoring version; main 128-question result pending.

- Tectonic 0.17.0 built the PDF successfully. Final log contains no overfull or
  underfull boxes and no engine warnings.
- `make notes notes-check` generated and checked five matching pdfpc notes.
  Each wraps at 42 characters and occupies at most 14 lines.
- `verify_deck.py` found five PDF pages, matching slide/note titles, and no
  text outside page bounds. It rendered every page through the existing CPU
  PyMuPDF environment.
- All five rendered pages were inspected. The final change to slide 4's caution
  was rebuilt, re-rendered, and inspected again. No visible clipping or overlap.
- The portable notes were checked structurally; no live pdfpc GUI session was
  run for this draft. The guide explains the material beyond the concise notes.
- Root independently rescored all three native pilot evaluations with the
  official full checker and checked paired prompts and references. The linked
  scoring receipt includes answer hashes and changed rows. Checkpoint hashes
  confirm the two post-training evaluations used identical saved weights.
- Slides 3 and 5 were revised to name MATH500 and show a shortened actual fuel
  example; rebuilt, notes checked, all page bounds checked, changed pages inspected.

The draft contains no new-method claim, no fabricated pending result, no claim
to reproduce the paper's full scores, and no claim that math learning establishes
recursive credit assignment. The learning example is explicitly illustrative.
