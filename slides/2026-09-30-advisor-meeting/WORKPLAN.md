---
meeting_date: 2026-09-30
meeting_time_user_local: "15:30"
review_ready_deadline_user_local: "14:00"
timezone_status: "User did not specify; produce first draft now and use 14:00 UTC as a conservative internal cutoff."
status: iterative_draft
target_slides: 8
user_latest_length_direction: "Up to eight; eight or nine allowed if clear examples genuinely need the extra space. Fewer is preferred when equally clear."
---

# Purpose and decisions

The user will not have time to rehearse. Advisors should understand the audience
PDF without narration; a separate reading guide should teach the user enough to
explain it and answer basic questions. Use complete sentences, ordinary words,
explicit examples, and a small number of defensible findings. No run nicknames,
unexplained acronyms, hidden essential qualifications, or cluttered methods.

This is a bounded presentation update using the existing Beamer/pdfpc pipeline.
The user's standing autonomous-work instruction replaces approval pauses.
Keep earlier decks intact and continue GPU experiments and actual result reviews.

## Story

1. RLM motivation and the immediate question about choosing useful next steps.
2. A concrete crafting goal, recipes, starting stock, and verifiable success.
3. Learning from worked examples versus learning from outcomes, in the same game.
4. The fixed-answer teaching repair, using a clearly labeled simplified example.
5. Its replication, with attempts distinguished from distinct goals.
6. Familiar RL gains versus limited transfer, with before/after comparisons.
7. A fair whole-task helper comparison, clearly labeled as planned.
8. Candidate paper direction, missing evidence, and an accessible discussion question.

Eight is useful because the methods primer has its own space. Remove a slide if
it only repeats another; never add one just to reach a count. A ninth is only
justified by a material new result or explanatory example that otherwise crowds
essential content.

## Iteration passes

- [x] Audit recent evidence and earlier presentations for what is actually new.
- [x] Draft the story and separate first-read learning guide.
- [x] Compile; inspect every audience slide and the short presenter notes.
- [x] Unfamiliar-reader pass: identify the first confusing word, missing premise,
  and likely mistaken inference on each slide; revise concrete examples.
- [x] Scientific pass: distinguish tested results, proposed mechanisms, and
  planned work; verify denominators, comparisons, fixed evidence and limits.
- [x] Read-aloud pass: remove sentences that require explanation to be meaningful.
- [x] Publish a readable draft (`403cd90`, research branch); continue targeted
  reviews as new evidence arrives.
- [ ] By the conservative review deadline, freeze a ready-to-read snapshot for
  the user's 2 pm review; preserve any later findings separately until integrated.

## Evidence that would change the deck

- Phi repaired-model results change the scope of the strongest teaching claim.
- Fresh-A RL and extra-SFT comparisons may replace or qualify slide 6, but report
  every prospectively fixed endpoint, not the most favorable checkpoint.
- Complete-goal helper results change slide 7 from a proposed test to a result
  only after whole-task outcomes and shared budgets are verified.
- A new compelling mechanism probe can replace a weaker slide; do not grow a
  run log. Sparse rewards in the first fresh training batch belong in the guide
  until an intervention or comparison makes them a central finding.

At each actual research review, assess these triggers and record the editorial
decision. Update visible claims, data, concise notes, guide, cutoff and PDF
together when needed. Current target is this September 30 package, not the older
September 29 or emailed September 25 decks.

## Draft revision record, September 29

### Research review around 07:11 UTC

The first varied-goal RL update finished and saved a checkpoint. Training-batch
actions changed in the intended average direction, but task improvement is
not measured yet. Keep the audience PDF, notes and guide unchanged; optimizer
diagnostics are not a new task-success result. The assisted fit and fixed
comparisons continue. The [review and receipts](../../experiments/selective_delegation/research_review_20260929/reviews/2026-09-29-0711.md)
record the result, its limits and the decision without adding another slide.

### Initial presentation edits

The first compile revealed one LaTeX line-break mistake and five crowded pages.
The syntax was corrected, then duplicate sentences and long block headings were
shortened. The paired training-example boxes now position below their actual
text heights rather than fixed overlapping coordinates. The chart was shortened
vertically without shrinking its labels. All eight pages now fit.

A second independent reader reviewed all actual slide renders and the guide.
Several material changes followed: clarify that helpers are already part of RLMs;
describe programmed assignment without implying learned or guaranteed success;
explain how the repair reorders existing lookups; replace physical-sounding
“supplies ingredients” with “writes ingredient list”; count successful, not merely
completed attempts; and put a dated evidence snapshot in the audience footer.
The guide now uses the same lantern example and explains the actual tiny helper
pilot separately. Technical details remain optional rather than being required
to understand a slide.

## Automatic research review, September 29, around 06:24 UTC

The new recovery-training examples were independently qualified on CPUs, but
there is no new learning result. The ongoing assisted collection is unfinished;
Phi repair and the whole-goal helper comparison remain pending. Keep the PDF,
notes and guide at their existing evidence cutoff. The supporting
[review](../../experiments/selective_delegation/research_review_20260929/reviews/2026-09-29-0624.md)
records the data milestone, prior art and conditional follow-ups without adding
another slide or presenting preparation as a result.

## Automatic research review, September 29, around 06:48 UTC

Both fresh training collections are now complete. With unchanged model weights,
code assistance increases successful attempts from 5/32 to 12/32 and reduces
calls, but both settings still have only three goals with mixed rewards. Those
are different groups with different sampled actions and credit, not identical
learning signals. The first new weight update is in progress, not complete.

Add this distinction to the optional reading-guide explanation and pin its
separate evidence cutoff. Keep the audience PDF and short notes unchanged until
the fixed RL/extra-SFT readouts establish what the model learned. The existing
slide 6 already introduces code-filled ingredients; a third numerical headline
now would repeat that idea and risk confusing execution assistance with learning.
