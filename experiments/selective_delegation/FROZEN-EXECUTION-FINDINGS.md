# Repeating a plan usually preserved its reward

We ran each of64 frozen training plans four times with new downstream sampling
seeds. The planner did not regenerate its plan. Helpers used the same supervised
weights and the final answerer used unchanged base weights throughout.

Of64 plans,32 were correct every time and19 were wrong every time. The remaining
13 changed outcome: five succeeded once, four twice and four three times.
Across384 within-plan seed pairs,43 disagreed (11.2%). These384 pairs are not
independent observations; the panel contains16 training questions and15 connected
components.

The training credit was not uniformly stable. Across384 candidate/seed-pair
comparisons,224 had zero credit in both executions,68 changed between zero and
nonzero credit,86 retained the same nonzero sign, and6 reversed sign. Candidate
pairwise ranking never strictly reversed, although63 comparisons changed between
a tie and a strict ordering. Sign changes and pairwise reversals measure different
things: credit also depends on the other candidates.

## Did more executions help choose a plan?

For each held-out execution seed, we selected among the four plans using either
one other execution or the mean of the other three, then scored only on the
held-out seed. One-execution selection solved46/64 held-seed attempts; three-
execution selection solved47/64. The paired difference is+1.56 percentage points,
with an exploratory component-bootstrap interval of0 to+5.00.

This uses answer labels for the same training questions, not a deployable
selector or evidence of generalization. Averaging four exclusions within each
parent prevents treating them as independent questions. The small gain does not
currently justify tripling downstream reward collection for RL. Some reward
noise exists, but this panel does not identify it as the dominant bottleneck.

## Decision and provenance

Together with the training-replay improvement45→49/64, this supports considering
a bounded continuation at the same update size rather than immediately changing
the RL objective or adding repeated reward sampling. The separate plan-only
comparison still needs to establish whether executing helpers earns its cost.

All256 executions completed:836 calls,2,345,503 tokens, no failed or unknown
calls. Native call service time totals649.4 seconds; it is not total wall time.
Source and scoring contracts are in [the diagnostic design](FROZEN-EXECUTION-DIAGNOSTIC.md).
The immutable report is `R/analysis-frozen-execution-001.json` and its Markdown
sibling. Seeds differ from training and separate helper/final offsets; this is
a diagnostic of the fixed policies, not an exact replication of training seeds.
