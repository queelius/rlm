# CURL replication results

Exploratory; three planned training seeds per arm. Evaluation episodes are not training replicates.

Scores use the fixed 100,000 training-environment-step endpoint, averaging ten evaluation episodes of one unchanged policy. Missing results remain missing.

| Arm | Completed seeds | Mean return | Across-seed SD |
| --- | ---: | ---: | ---: |
| curl | 3/3 | 434.63 | 48.81 |
| no_curl | 3/3 | 433.44 | 236.86 |

## Matched pairs

Seed 123: CURL 482.83, control 241.28; difference +241.55.
Seed 456: CURL 385.23, control 360.96; difference +24.27.
Seed 789: CURL 435.83, control 698.06; difference -262.23.

## Attempts

- no_curl seed 123, segment 0: completed; completed. Source: `/project/alex_phd/repos/rlm/.worktrees/curl-replication-20261004/docs/curl-replication-2026-10-04/walker-data/first-control`.
- curl seed 123, segment 0: completed; completed. Source: `/project/alex_phd/repos/rlm/.worktrees/curl-replication-20261004/docs/curl-replication-2026-10-04/walker-data/first-reference`.
- no_curl seed 456, segment 0: completed; completed. Source: `/project/alex_phd/repos/rlm/.worktrees/curl-replication-20261004/docs/curl-replication-2026-10-04/walker-data/second-control`.
- curl seed 456, segment 0: completed; completed. Source: `/project/alex_phd/repos/rlm/.worktrees/curl-replication-20261004/docs/curl-replication-2026-10-04/walker-data/second-reference`.
- no_curl seed 789, segment 0: completed; completed. Source: `/project/alex_phd/repos/rlm/.worktrees/curl-replication-20261004/docs/curl-replication-2026-10-04/walker-data/third-control`.
- curl seed 789, segment 0: completed; completed. Source: `/project/alex_phd/repos/rlm/.worktrees/curl-replication-20261004/docs/curl-replication-2026-10-04/walker-data/third-reference`.

Pilot runs and invalid native inputs excluded: []

Repeated arm/seed attempts and resumed segments remain visible, but are excluded from endpoint aggregates. Pair differences require matching scientific configurations. The control retains crops; it is not the paper's Pixel SAC baseline. This comparison uses equal interactions, not equal wall time.
