---
title: Advisor package and final short-experiment checkpoint
recorded_utc: 2026-09-11T13:40:00Z
status: presentation_verified_pilot_terminal
audience_deck_evidence_cutoff_utc: 2026-09-11T12:55:00Z
research_store: /project/alex_phd/runs/rlm-research-r4
---

# What to read and present

The [advisor package](../../slides/2026-09-11-advisor-meeting/README.md) has eight
main slides for a ten-minute discussion and six optional backups. The main story
is a visual explanation of RLMs, training progress, a helper-answer matching
effect in two model families, the measured quality/time tradeoff, and a proposed
record-level handoff component. The earlier August deck remains unchanged.

The [speaker guide](../../slides/2026-09-11-advisor-meeting/speaker-guide.md)
explains each slide and likely questions in plain language. It distinguishes
the larger hope of reusing a decomposition across domains from the narrower
results we have actually measured. The [publication options](../../slides/2026-09-11-advisor-meeting/publication-options.md)
state what would make each direction a credible contribution.

The current audience PDF was compiled and checked visually, including the actual
pdfpc presenter console at 1280×720. All 14 slide/launcher tests passed; numerical
sources and evidence hashes were checked. Details and artifact hashes are in
[VERIFICATION.md](../../slides/2026-09-11-advisor-meeting/VERIFICATION.md).
Run `make -C slides present`, click the presenter window, and press `w`.
Share only the audience window. `make -C slides rehearse` opens private practice.

# Findings added during the two-hour window

- Training transferred to described combinations of familiar operations:
  correct-and-faithful outcomes were 2/72 before and 29/72 after training, with
  missing-result bounds 2–13 and 29–30. The plan was supplied, not discovered.
- At 64 records per helper call, arbitrary matching names changed Qwen accuracy
  from 44% to 83% and Mistral from 35% to 52% on the same exposed panel.
- Changing the requested order of keyed answers retained about 83%, 82%, and
  81% accuracy. This is exploratory order-retention evidence, not equivalence.
- On the same 768 records, large unnamed / large named / smaller unnamed /
  singleton policies reached 49/85/81/87% in 19/91/19/31 local workload seconds.
  Fewer requests did not mean less time; singletons repeated more input text.
- Compact keyed replies reduced output work and local time in both tested
  models. Qwen stayed near 85%; Mistral's improvement mainly reflected fewer
  malformed replies. The full paired counts are in supporting notes.
- The longer reward-training pair did not improve its secondary final-answer
  count: 57 before, 55–57 with answer reward, and 54 with an additional check.
  Both runs together took about 4 hours 39 minutes. Full population review of
  calculation faithfulness remains unfinished; this is not a general result
  against reinforcement learning.

The main PDF stops at the declared cutoff. Later supporting findings are dated
separately; they do not silently change the figures. Git milestone
`a343e1a6d9fa55a0e517abe7987012a27afa7893` was confirmed on origin/main at 13:27 UTC.

# Final pilot: terminal and audited

The prescribed record-map comparison is in
`sidecars/root-record-map-batch-handoff-v1`. Attempt 001 completed all 32 helper
calls and saved all 16 maps, then failed before root execution. Attempt 002 made
zero root calls: a task-setup callback mismatch was followed by a scorer error.
These are integration failures, not model failures. Both attempts are immutable
and released. Their lifecycle times were 134.48 and 128.00 seconds.

The helper-only counts are 321/384 for large named calls versus 255/384 for
smaller unnamed calls. There were 32 physical requests, 43,541 input tokens,
9,432 output tokens, and 1,584 cached input tokens. Acquisition was sequential;
do not import the four-worker/cache-off timing assumptions from the other study.

Attempt 003 began at 13:29:01.335 UTC under PTY 19582, with a 600-second outer
cap. It reuses those exact maps and the original root prompts, seeds, and 32
planned endpoints. It ended at 13:35:34.944 UTC after 393.61 seconds and released
the service; MAIN collected parent exit 1 and confirmed no remaining GPU process.
The fixed deadline left four established outcomes (three empty replies and one
wrong answer) and 28 unknown outcomes, including 20 unstarted episodes.
This is an inconclusive comparison, not a 0/32 observed-failure rate.

MAIN read and reran the independent native reader byte-for-byte (audit SHA
`fd141cbd3065322a1e8ac0b045a974a02aa13c7b2a4aa506882b683f60b3056f`).
Sampled programs repeatedly misused the synchronous dictionary interface as an
older asynchronous/object interface. A clearer executable calling example is the
next targeted check, not another identical run. All 227 physical model requests
and seven requests with unknown usage remain in the supporting evidence.

This compares fixed helper-acquisition packages behind the same prescribed
root-facing dictionary. It does not compare a new adaptive architecture against
the current RLM, and the interrupted lifecycle cannot establish an uninterrupted
pipeline speedup. Analysis lives in
`analyses/root-record-map-batch-handoff-live-2026-09-11`.

# Resume without losing the scientific record

Read the research store's live RESEARCH_QUEUE.md and inspect GPU owners before
launching anything. The user's presentation-focused window ends around 13:44
UTC; do not infer a new long training campaign from remaining scheduler time.
There is no ready successor or active GPU run in this completed window.
Prioritize a usable complete-task comparison over
another isolated identifier study. Keep smaller-call baselines and full physical
costs. The record-level handoff and cross-task training paths remain exploratory.

At 13:41 UTC the shared Codex account had 25% remaining. Another session uses
the same allowance. The existing monitor checks every 15 minutes; wind-down
and pause thresholds are 20% and 15%, preserving roughly 10% for user revisions.
There is no hard cross-session reservation.

GitHub preserves the deck, source, figures, compact evidence, and summaries.
Raw trajectories, model and optimizer checkpoints, environments, caches, and the
live research notebook remain on project storage. Their linked paths are not a
backup of their contents.
