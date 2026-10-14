# Evaluation 560 and protected recovery state

At 8,976 updates, independently regraded test mean pass@1 is 56.856520%
(11,999/21,104), validation 72.067694% (4,301/5,968), training subset
88.222446% (5,251/5,952). All 33,024 answers and 21 metrics match.
Test changes by only minus three correct responses from evaluation 550.
This remains intermediate learning, not the fixed 690/11,040 endpoint.

The preselected full 8,960 state is protected on the same home filesystem:
ten source/copy hashes match; trainer/actor/active Adam steps 8,960,
scheduler 8,959, skipped steps zero. Not a GPU restoration or finiteness test.
See [checkpoint receipt](evidence/checkpoint8960-protection-20261014.json)
and [evaluation interpretation](evidence/restart-eval560-interpretation-20261014.json).

Resumption at 15:03 UTC observes 10,256 updates under original owners and
seven unchanged source pins. Later scheduled evaluations await regrading.
Some step labels are rounded to 1e+4, preventing exact contiguous-step proof
from console labels alone; this does not show skipped optimization.
Next: audit the completed evaluation backlog and protect the preselected
10,240 full state after admission checks. The endpoint remains unchanged.
