# CURL replication results

Exploratory; three planned training seeds per arm. Evaluation episodes are not training replicates.

Scores use the fixed 100,000 training-environment-step endpoint, averaging ten evaluation episodes of one unchanged policy. Missing results remain missing.

| Arm | Completed seeds | Mean return | Across-seed SD |
| --- | ---: | ---: | ---: |
| curl | 3/3 | 398.06 | 128.56 |
| no_curl | 3/3 | 230.61 | 93.23 |

## Matched pairs

Seed 234: CURL 502.95, control 200.43; difference +302.52.
Seed 567: CURL 254.63, control 335.20; difference -80.56.
Seed 890: CURL 436.58, control 156.22; difference +280.37.

## Attempts

- curl seed 234, segment 0: completed; completed. Source: `/project/alex_phd/repos/rlm/.worktrees/curl-replication-20261004/docs/curl-replication-2026-10-04/walker-additional-data/curl-seed234`.
- curl seed 567, segment 0: completed; completed. Source: `/project/alex_phd/repos/rlm/.worktrees/curl-replication-20261004/docs/curl-replication-2026-10-04/walker-additional-data/curl-seed567`.
- curl seed 890, segment 0: completed; completed. Source: `/project/alex_phd/repos/rlm/.worktrees/curl-replication-20261004/docs/curl-replication-2026-10-04/walker-additional-data/curl-seed890`.
- no_curl seed 234, segment 0: completed; completed. Source: `/project/alex_phd/repos/rlm/.worktrees/curl-replication-20261004/docs/curl-replication-2026-10-04/walker-additional-data/no_curl-seed234`.
- no_curl seed 567, segment 0: completed; completed. Source: `/project/alex_phd/repos/rlm/.worktrees/curl-replication-20261004/docs/curl-replication-2026-10-04/walker-additional-data/no_curl-seed567`.
- no_curl seed 890, segment 0: completed; completed. Source: `/project/alex_phd/repos/rlm/.worktrees/curl-replication-20261004/docs/curl-replication-2026-10-04/walker-additional-data/no_curl-seed890`.

Pilot runs and invalid native inputs excluded: []

Repeated arm/seed attempts and resumed segments remain visible, but are excluded from endpoint aggregates. Pair differences require matching scientific configurations. The control retains crops; it is not the paper's Pixel SAC baseline. This comparison uses equal interactions, not equal wall time.
