# TRAIN execution-credit finding

This completed diagnostic is **TRAIN-only**. It does not estimate held-out
performance, a deployable selector, or a general benefit of helpers.

Across the fixed 16 TRAIN parents, five execution settings, and four candidate
plans per group, all 320 planned plan-only finals returned. Executed full-source
helper/final trajectories were correct on 198/320; matched empty-helper-report
plan-only finals were correct on 99/320 (difference +30.94 percentage points;
15 component clusters, bootstrap 95% interval +12.81 to +52.00 points). Among
both-valid comparisons, execution changed 105 slots from wrong to right and 6
from right to wrong.

That is evidence that the **executed helper report**, conditional on these
already-generated TRAIN plans and their action-dependent final prompts, often
recovers a failure of the empty-report plan-only path. It does not establish
that the learned planner caused the gain: the five settings and four candidates
share parents, documents, source plans, and final seeds; they are not 320
independent cases. Nor does it establish that helpers add value over a
no-plan/no-helper direct answer. That missing direct comparison is the next
small control.

Credit and terminal-reward ranking differed in 26/320 candidate slots
(`terminal_only` 20, `execution_only` 4, `opposite` 2); most slots were tied
or zero. Therefore larger terminal reward should not be read as clean credit
for a better plan. The analysis's ten factual-replay controls all reproduced
their saved output tokens, which checks this particular final-client seam but
does not remove the TRAIN/action-dependence limitation.

The experiment added 330 physical final calls (931,283 tokens) and retained
the historical 1,109-call acquisition cost separately; summed service latency
is not GPU kernel time. Source report:
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/analysis-training-execution-credit-001.json`.
