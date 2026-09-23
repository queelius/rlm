# Corrected-original teacher control

This is a mechanism control for the known `train.1029` target-quantity mismatch,
not a new teacher.  The original privileged action is corrected from six `raw_t8`
to four and output 12 to output 8, then **all 32 native trajectories** are replayed.
The corrected immutable input has 366 rows: only one action target differs, while
29 prompts differ because public craft feedback changes.  Thus editing a single
label would have been invalid.

Proposed fixed jobs use the unchanged source048 SFT recipe, seeds 2026092208 and
2026092291, one epoch/23 updates, fixed checkpoint 23, and a 30-minute cap each.
Each endpoint is read on the already exposed fresh32 panel using the same two
sampling seeds.  Existing public-teacher endpoints are reused only as the paired
reference; a native public-minus-corrected-original comparison is then scored.

The comparison answers whether the public package gain remains after correcting
the known quantity mismatch.  It does not isolate action order, history format,
or token dose, and the fresh panel is exploratory rather than untouched
confirmation.  No job is accepted or launched by this proposal.

The CPU interpreter available here lacks `peft`, so native `--prepare-only` could
not enter the source048 trainer.  Input replay and wrapper-contract tests passed;
the eventual qualified training environment must run its normal prepare-only
validation before GPU acquisition.
