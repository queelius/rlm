# RL improved answers on a small training replay

The last checkpoint from the full-pass RL run solved **49 of64 attempts**, compared
with **45 of64 before RL**. These are the same16 training questions, four sampled
plans per question, with the original sampling and execution seeds. All four
improvements had valid outputs before and after training; none became incorrect.
This is evidence of training-set reward improvement, not generalization.

The root model changed31 of64 plans. The helper remained the fixed supervised
checkpoint36, and the final answerer retained its unchanged base weights and
access to all source documents. Therefore improved final answers do not establish
that the root learned more faithful or more general decomposition.

Exact match rose6.25 percentage points; the paired component-bootstrap interval
is+1.47 to+11.76. Token F1 rose5.31 points, interval+0.94 to+10.83. These are
exploratory intervals on a selected training panel, not evidence from64 independent
questions or a confirmatory test. The terminal checkpoint was chosen by the
stopping rule, not by this replay score.

Collection used274 new model calls,615,832 tokens and332.2 wall-clock seconds.
All64 attempts were recorded, with no failed or unknown calls. The report is
`R/analysis-rl-trainfit-001.json` and its Markdown sibling, where `R` is the
September21 selective-delegation external research store.

## What changed in the four wins

A native-trace audit identifies four distinct parents, one improved candidate
each. Two changes sharpen the requested answer: a year becomes a full date,
and a county becomes the specific park requested. One repairs a wrong dependency
route, and one removes an irrelevant ship-construction step. None uses more
steps. All four improved finals agree with their terminal helper answer; this
is trace-consistent improvement, not proof that the helper caused the final.

Trace identities are `c423…/c0`, `03a6…/c0`, `2e14…/c1`, and `9b78…/c1`,
fully resolved in the report's paired rows. Baselines are under
`R/rl-fullpass-001/batch-0001/{episodes,calls}` and new observations under
`R/rl-trainfit-001/{episodes,calls}`. The abbreviated IDs are analysis pointers,
not intended as audience-facing result labels.

## What this changes

It weakens the explanation that the small root updates cannot improve reward
at all. It does not settle whether reward is reliable across downstream seeds,
whether additional training helps new questions, or whether the helper execution
is necessary. The active frozen-execution diagnostic and accepted plan-only
comparison address those questions before selecting further training.
