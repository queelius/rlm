# CURL replication results

Exploratory; three planned training seeds per arm. Evaluation episodes are not training replicates.

Scores use the fixed 100,000 training-environment-step endpoint, averaging ten evaluation episodes of one unchanged policy. Missing results remain missing.

| Arm | Completed seeds | Mean return | Across-seed SD |
| --- | ---: | ---: | ---: |
| curl | 1/3 | 482.83 | missing |
| no_curl | 1/3 | 241.28 | missing |

## Matched pairs

Seed 123: CURL 482.83, control 241.28; difference +241.55.

## Attempts

- no_curl seed 123, segment 0: completed; completed. Source: `/project/alex_phd/repos/rlm/.worktrees/curl-replication-20261004/docs/curl-replication-2026-10-04/walker-data/first-control`.
- curl seed 123, segment 0: completed; completed. Source: `/project/alex_phd/repos/rlm/.worktrees/curl-replication-20261004/docs/curl-replication-2026-10-04/walker-data/first-reference`.

Pilot runs and invalid native inputs excluded: []

Repeated arm/seed attempts and resumed segments remain visible, but are excluded from endpoint aggregates. Pair differences require matching scientific configurations. The control retains crops; it is not the paper's Pixel SAC baseline. This comparison uses equal interactions, not equal wall time.
