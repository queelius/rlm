---
status: operational_snapshot_not_performance_claim
evidence_cutoff_utc: 2026-09-21T19:08:00Z
study_store: /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921
metric: sum_of_recorded_terminal_owner_elapsed_seconds
completed_owner_receipts_with_metric: 46
elapsed_seconds_rounded_sum: 27934.3
---

# Where the GPU time went

At this cutoff, 46 execution owners have terminal receipts containing an
`elapsed_seconds` field. Their recorded durations sum to 27,934.3 seconds, or
about 7 hours 46 minutes. The ongoing fresh ALFWorld run is excluded. This is a
reproducible lower-scope accounting of run wall time, **not measured GPU kernel
utilization**, pure optimizer time, or a claim that all planned work succeeded.
Some runs ended under a declared stopping rule. Initial runs lacking this field
are not silently assigned a duration.

The measured work includes model loading, native generation, environment stepping,
replay, optimizer updates where applicable, and artifact writes within each owner.
For example, the full-pass planner-RL owner took 5,063.2 seconds, whereas one helper
SFT owner took 594.3 seconds. A short SFT update phase therefore does not describe
the compute needed to sample RL trajectories, compare checkpoints, run controls,
and test transfer. Scientific value still depends on what those comparisons show;
long runtime alone is not evidence of a useful result.

Reconstruction: scan only immediate study directories for `TERMINAL-*.json`, retain
receipts with `elapsed_seconds`, pair each with its matching `OWNER-*.json`, and
sum durations. This snapshot rounded each duration to 0.1 seconds before summing.
Do not count copied source fixtures, queued supervisors, or historical calls
reused in an analysis as additional GPU execution. The study uses the common
exclusive coordinator lock; this note does not independently audit all interval
overlaps or turn lease occupancy into utilization.

The detailed scientific reports distinguish new physical calls from reused
comparison data and report input/output tokens and native service time when those
receipts exist. Those measures answer different questions and should remain
separate in any advisor report or paper.
