# SPO progress through evaluation 550

Scientific cutoff: October 14, 2026, 09:10 UTC. Own training scores **56.870735%**
test mean pass@1 after **8,816 updates**, versus paper56.7% and original SFT39.774450%.
The gain over SFT is17.096285 percentage points. This is an intermediate result;
the prescribed690 collections/11,040 updates and final evaluation remain unfinished.

| Updates | Test | Validation | Training subset (not held out) |
| ---: | ---: | ---: | ---: |
| 8,336 | 55.875663% | 70.492627% | 87.281586% |
| 8,496 | 56.633813% | 70.928284% | 88.004032% |
| 8,656 | 56.235785% | 71.481233% | 87.617608% |
| 8,816 | 56.870735% | 70.894772% | 87.903226% |

All33,024 saved answers and21 native metrics were independently regraded at each
new point. The latest test has12,002/21,104 correct responses; validation4,231/5,968
and training5,232/5,952. Changes from540 are+134/−35/+17 responses. Each audit ran once.

The test curve fluctuates near the paper result; validation and test do not always
move together. These are reused questions and one training seed. We retain every
scheduled result and the fixed endpoint; numerical exceedance does not establish
statistical equivalence, historical implementation fidelity or a causal BOS effect.

A [fresh training audit](evidence/restart-training540-20261014.json) checks all512
responses in collection540 against original questions/gold and saved trajectories.
It finds450 correct rewards,62 incorrect, zero reward or generation/training query
token mismatches, finite saved log-probabilities/advantages,15 mixed-reward groups
and1,153 nonzero advantage tokens. This supports a functioning learning signal;
it is not a new model/optimizer tensor scan or held-out benchmark.

Evidence: [530 audit](evidence/restart-eval530-20261014.json),
[540 audit](evidence/restart-eval540-20261014.json),
[550 audit](evidence/restart-eval550-20261014.json),
[550 interpretation](evidence/restart-eval550-interpretation-20261014.json), and
[execution](evidence/restart-eval550-execution-20261014.json).

Next: continue the original whole owner to the fixed endpoint, preserve the
preselected8,960-update full state, then independently regrade the final test.
The advisor PDF retains its explicitly dated07:06 cutoff through520; its central
message remains that substantial learning is measured and the reproduction is
unfinished. This report contains newer evidence. Algorithm choice and transfer
to future RLM work remain untested.
