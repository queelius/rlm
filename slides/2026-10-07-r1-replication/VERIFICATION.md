# Presentation verification

## Memory-repair handoff, evidence cutoff 7 October, 15:55 UTC

- The third larger attempt failed after 12 updates with no saved checkpoint.
  The five-slide PDF and eight-page learning PDF distinguish that failure from
  completed short-run results and from the authors' released-model reference.
  Attempt 4 is explicitly prepared, not launched at this cutoff.
- Fresh Tectonic builds passed. Root reran the five-page/title/bounds check and
  matching pdfpc-note check; all notes fit within 14 lines. The documentation
  agent checked both PDFs' text bounds and visually inspected revised pages;
  root additionally inspected slide 5. No live pdfpc GUI was tested.
- No partial broader-benchmark scores were promoted into this dated PDF. Its
  supporting research record can carry later verified results without implying
  that they were known at 15:55.

## Smaller-collection retry, evidence cutoff 7 October, 15:25 UTC

- Current guide, notes, speaker guide, and evidence distinguish two retained
  failures from the third fresh-base attempt. The first exhausted memory during
  optimization; completed inner updates are unknown. The second failed during
  initialization because expandable segments were incompatible with vLLM's pool.
- Attempt 3's budget is 4,080 questions, 32,640 responses, and 255 updates in
  16-question collections. The earlier 3,968/248 budget is explicitly historical.
  Both final scores remain pending. The slide's rounded budget and all existing
  results remain accurate; its cutoff and the notes were synchronized.
- Fresh Tectonic builds produced five slides and eight guide pages without
  warnings. PDF bounds and five matching notes pass; note lengths are
  7, 7, 7, 6, and 9 lines. Final slide 5 and guide page 8 were visually checked
  without clipping or overlap. No GPU action or live source edit was performed.

## Authors' released-model reference check, 7 October, 15:15 UTC

- The independent author-reference receipt regrades all 500 saved answers:
  366 correct, with every ordered prompt and reference matching the official
  source. Slide 5 labels these as the authors' released weights, not our
  training result. The five-answer difference from their reported score is
  retained in the guide and evidence.
- The deck remains five slides and preserves the completed short-run table.
  Its 15:14 UTC cutoff includes the reference result and loader correction;
  speaker guide, notes, README, and evidence agree.
  The larger run is explicitly pending; the slide rounds its effective budget
  to roughly 4,000 questions and about 250 planned updates. The guide records
  the loader-derived 3,968 questions and 248 updates without claiming completion.
- Fresh Tectonic builds of the deck and eight-page learning guide completed
  with no engine warnings or overfull/underfull boxes. An initial guide overflow
  from the full revision string was fixed by displaying the short identifier
  while retaining the pinned link and full receipt.
- Five matching presenter notes pass the 14-line check; actual line counts are
  7, 7, 7, 6, and 9. All PDF text spans are within page bounds in both artifacts.
  Revised slides 3 and 5 and guide pages 1, 2, and 8 were visually inspected;
  no clipping or overlap was found. No live pdfpc GUI was run.
- CPU-only documentation work changed no live owner, training source, GPU
  process, or experiment output. The result does not reproduce the authors'
  training or establish a reliable gain from our pending larger recipe.

## Fresh training repeat, 7 October, 12:37 UTC

- Root independently regraded all 1,000 new final answers and matched their
  prompts and references to the official source. The repeat scored 306/500
  with questions alone (18 gains, 17 losses) and 166/500 in chat style
  (25 gains, 13 losses). The prescribed final checkpoint and all 4,096
  training responses were verified; model and answer hashes are in the two
  new repeat receipts.
- The deck remains five slides. Slide 3 describes three fresh-base runs.
  Slide 4 shows all eight model/input combinations, including both
  question-only training outcomes. Slide 5 proposes a longer-training check
  without promising an improvement. The conclusion distinguishes working
  learning updates from a reliable gain over the stronger starting setup.
- Tectonic rebuilt the final PDF without warnings. An initial crowded
  results-slide layout was fixed by shortening redundant prose, not shrinking
  the font. All five titles match the notes, all notes fit the 14-line limit,
  and every PDF text span is within page bounds. Root inspected revised
  slides 3–5; an independent reader reviewed all five and the final result
  slide, finding no unresolved audience-clarity or layout issue.
- The six-page learning guide was rebuilt. Its author checked changed pages
  and all page bounds. Root reviewed its source diff, independently checked
  all six page bounds and final counts/cutoff, and visually inspected pages
  1, 4 and 5. The guide retains the earlier separate128 comparison unchanged.
- The evidence cutoff is 12:35 UTC. A further fixed-weight evaluation repeat
  is pending and is not counted as a new training run or a completed result.
  No live pdfpc GUI was tested; presenter notes were checked structurally.

## Both training formats, 7 October, 11:38 UTC

- Root regraded all 1,000 final answers from the new question-only-trained
  model and checked every prompt and reference against the official dataset.
  Scores are 314/500 with questions alone and 168/500 in chat style. The new
  receipt retains both outcomes, paired changes and data roles.
- Slide 4 now shows all six model/input combinations. Slide 3 distinguishes
  the two fresh-base training runs; slide 5 retains its clearly identified
  actual example and explains the next repeatability check.
- Tectonic rebuilt five pages with no warnings. An initial slide-3 overflow
  was corrected by shortening redundant wording, not shrinking the text.
  Five matching notes pass the 14-line check; PDF text bounds pass.
- Root visually inspected revised slides 3–5, including the final table.
  An independent reader inspected all five and found no material layout,
  audience-clarity or scientific-claim issue.
- The six-page learning guide was rebuilt without warnings. Its author
  inspected changed pages; root checked all text bounds, the changed source,
  and the page showing the full results table.
- Same-weight evaluation repeats and a further training repeat are not used
  as completed evidence in this version. No best checkpoint was selected.
  No live pdfpc GUI was tested; notes were checked structurally.

## Completed prompt comparison, 7 October, 10:42 UTC

- Both full500 evaluation pairs completed successfully. Root independently
  rescored all 1000 question-only outputs with the official full checker and
  verified source questions and references. The chat pair was checked earlier.
  The new receipt records all four counts, hashes, length measures and data roles.
- The deck now presents all four cells: 154 to 308 in chat style, 305 to 317
  with the question alone. It uses the same final trained checkpoint throughout.
- Fresh Tectonic builds completed without overfull or underfull boxes. All five
  slide titles match their notes; notes stay within 14 lines. PDF text bounds pass.
  Root visually inspected slides 3–5. An independent reader approved the same
  layout and scientific explanation; its suggested clarification that the
  fraction example used chat-style input was applied, rebuilt and inspected.
- The six-page learning guide was rebuilt, its latest table checked, and every
  text span checked in bounds. Its author visually inspected changed pages.
- The new training result remains pending. Its first attempt was rejected after
  saved inputs exposed a stale data cache. The restart isolates that cache without
  changing the authors' training algorithm. The completed evaluation is unaffected.
- No live pdfpc GUI was tested; presenter notes were checked structurally.

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

At that earlier cutoff, the larger evaluation was still pending. One training run does not
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
