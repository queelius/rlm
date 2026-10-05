# Cartpole image-correspondence comparison

Exploratory three-arm comparison. Ten fixed evaluation starts are not training replicates. All nine matched runs must complete before pooling. Shuffled gradients are not mechanism proof.

Fixed 100k training-step endpoints; means of ten starts 10000–10009. No best-point selection.

| Seed | Condition | Status | Endpoint |
| ---: | --- | --- | ---: |
| 123 | CURL with image matching | completed | 678.020818 |
| 123 | Same crops without image matching | completed | 454.471029 |
| 123 | Shuffled image matching | completed | 0.113920 |
| 456 | CURL with image matching | completed | 446.171087 |
| 456 | Same crops without image matching | completed | 240.538928 |
| 456 | Shuffled image matching | completed | 68.102779 |
| 789 | CURL with image matching | completed | 587.994323 |
| 789 | Same crops without image matching | completed | 463.258501 |
| 789 | Shuffled image matching | completed | 107.786184 |

## Cohort averages

CURL with image matching: 570.728743; training-seed SD 116.885199.
Same crops without image matching: 386.089486; training-seed SD 126.127033.
Shuffled image matching: 58.667627; training-seed SD 54.452692.
Paired mean differences: {"curl_minus_no_curl": 184.639256683948, "shuffled_minus_curl": -512.0611154498088, "shuffled_minus_no_curl": -327.42185876586086}

## Sources and accounting

- curl seed 123: completed; source `/project/alex_phd/repos/rlm/.worktrees/curl-replication-20261004/docs/curl-replication-2026-10-04/data/first-reference`.
- no_curl seed 123: completed; source `/project/alex_phd/repos/rlm/.worktrees/curl-replication-20261004/docs/curl-replication-2026-10-04/data/first-control`.
- shuffled_curl seed 123: completed; source `/project/alex_phd/repos/rlm/.worktrees/curl-replication-20261004/docs/curl-replication-2026-10-04/correspondence-data/shuffled_curl-seed123`.
- curl seed 456: completed; source `/project/alex_phd/repos/rlm/.worktrees/curl-replication-20261004/docs/curl-replication-2026-10-04/data/second-reference`.
- no_curl seed 456: completed; source `/project/alex_phd/repos/rlm/.worktrees/curl-replication-20261004/docs/curl-replication-2026-10-04/data/second-control`.
- shuffled_curl seed 456: completed; source `/project/alex_phd/repos/rlm/.worktrees/curl-replication-20261004/docs/curl-replication-2026-10-04/correspondence-data/shuffled_curl-seed456`.
- curl seed 789: completed; source `/project/alex_phd/repos/rlm/.worktrees/curl-replication-20261004/docs/curl-replication-2026-10-04/data/third-reference`.
- no_curl seed 789: completed; source `/project/alex_phd/repos/rlm/.worktrees/curl-replication-20261004/docs/curl-replication-2026-10-04/data/third-control`.
- shuffled_curl seed 789: completed; source `/project/alex_phd/repos/rlm/.worktrees/curl-replication-20261004/docs/curl-replication-2026-10-04/correspondence-data/shuffled_curl-seed789`.

Equal training interactions do not imply equal wall time or optimizer work. Original baselines and the later shuffled cohort remain separately identified.
