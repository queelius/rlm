---
title: Standalone-reader revision and review
date: 2026-09-11
status: addressed_and_visually_verified
audience: advisors and presenter learning the research
evidence_cutoff_utc: 2026-09-11T03:45:00Z
---

# What changed for the reader

Each result slide now identifies what was tested, what changed, and what the
numbers count. Main-model training, isolated helper tests, and helper training
are presented as separate experiments. A concrete observation/action example
explains supervised training before the first results chart.

The diagrams retain the big question: can a model solve an unfamiliar problem
by selecting and combining familiar steps? The deck distinguishes that research
goal from the limited routines tested so far.

# Independent review and resolution

| Reader risk | Change made |
|---|---|
| The same model appears to have inconsistent tag scores. | Slide 7 scores every answer in nested batches; slide 8 uses different records and scores later answers. The three models share slide 8's records with each other. |
| The poor final-task result appears to test the matching-tag improvement. | Slide 10 explicitly describes a separate question-type training test with a fixed calculation, and says it is not a matching-tag test. |
| Reward-training totals appear to treat missing answers as wrong. | Slide 11 displays possible totals as large ranges and explains missing results in the main text. |
| Training on two example sets is confused with evaluating new input. | Slide 5 now explicitly reports the new-input comparison of the same three model versions. The task types stayed familiar; the guide limits the meaning of “newly selected.” |
| The new tag result seems to contradict the earlier unrelated-name result. | Slide 9 shows its actual tag-pair manipulation; its guide explicitly separates the two experiments and warns against comparing their absolute scores. |

An independent agent reviewed the earlier slides 8–10 and confirmed the first
three issues resolved. After insertion of the new slide9, an independent review
also found no material blocker in its figure, guide, methods and claim wording,
and confirmed slide5's evaluation limitation is prominent. MAIN visually inspected every page during the revision and
re-inspected pages changed after that pass. Compilation, numerical checks, and
exact final hashes are recorded in [VERIFICATION.md](VERIFICATION.md).

This is a clarity and consistency review, not a human comprehension study or
independent replication of the experiments. Detailed methods remain in the guide
and evidence document, not in dense slide footnotes.

## Earlier wording pass (02:15 evidence snapshot)

A further independent review of rendered slides 4–10 found three remaining
ambiguities. Slide5 now says the two runs started from the same model, and that
the evaluation records came from earlier tests; this avoids implying sequential
training or silently equating evaluation exposure with training-set membership.
Slide8 now says the reading task stayed the same while the records changed.
Slide10 explicitly says that our fixed Python code combined the helper labels;
the reference bar is labeled "Known-correct labels," without implying a model
achieved perfect labeling. Slide7 directly identifies its bars as later-answer
accuracy. No measured values or numerical cutoff changed.

Supporting format-control and partial whole-task checks are recorded separately
in later-findings.md, linked from the guide. They do not add slides.

## Current new-evidence revision (03:45 cutoff)

The new-input evaluation replaces slide 5's older comparison. Slide 7 now shows
accuracy as batch size grows, with direct line labels instead of a crowded legend.
The guide explains every axis, gives a matching-tag example, and distinguishes
full-batch accuracy from the later-answer metric on slides 8 and 9.
Slide 12 acknowledges completed new-input testing and asks about unfamiliar
combinations and complete-task usefulness. The deck stays at 14 slides.

An independent agent's design review recommended exactly these distinctions.
MAIN incorporated them, compiled the revised deck, corrected two overflows, and
visually inspected all 14 final pages. The final numerical and artifact checks
are in VERIFICATION.md. This review does not establish comprehension by a human
audience; the guide remains available for questions without carrying the main story.

A final independent agent review inspected rendered slides 5, 7, 8, and 12,
their guide sections, and the portable counts. It found no material audience-facing
issue, numerical inconsistency, or substantive clipping/overlap. It specifically
checked fresh records versus familiar tasks, missing-outcome ranges, all-position
versus later-position accuracy, and component findings versus whole-system claims.
The reviewed PDF hash begins `a83bdaab`, matching VERIFICATION.md.
