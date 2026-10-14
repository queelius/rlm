# Progress through evaluation 640

Cutoff: October 14, 15:07 UTC. Latest independently verified test mean pass@1:
**57.666793% at 10,256 updates**, versus paper 56.7% and original SFT 39.774450%.
The fixed 690-collection / 11,040-update endpoint remains unfinished.

| Updates | Test | Validation |
| ---: | ---: | ---: |
| 9136 | 57.188211% | 71.565013% |
| 9296 | 57.226118% | 72.201743% |
| 9456 | 56.984458% | 71.330429% |
| 9616 | 57.737870% | 71.900134% |
| 9776 | 57.846854% | 72.687668% |
| 9936 | 57.500948% | 71.699062% |
| 10096 | 57.808946% | 71.095845% |
| 10256 | 57.666793% | 72.302279% |

All eight scheduled evaluations were independently regraded: 264,192 saved
responses and 168 native metrics, with original identity/prompt/response checks.
Latest test: 12,170/21,104. Every checkpoint is reported; none is selected by test
performance. These are repeated questions from one training seed, not independent
replications or statistical equivalence evidence. BOS causal attribution and
transfer to RLM remain untested.

[Audit index with receipt hashes](evidence/evaluation-backlog570-640-20261014.json).
The original owner, fixed endpoint, and disclosed adapters remain unchanged.
Advisor package remains explicitly dated 07:06; this report supersedes its numbers,
not its limitation that the endpoint reproduction is unfinished.
