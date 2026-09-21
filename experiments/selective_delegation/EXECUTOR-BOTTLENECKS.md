# Executor bottlenecks: restricted trace audit

## Scope and rule

This audit uses all 256 `rl-planner-001` TRAIN trajectories and the completed
64-episode held-SFT validation readout. A generated plan is called *exact* only
when its complete ordered `subquestions` list byte-matches the TRAIN parent's
annotated reference-question list. Annotated step answers are compared only in
that exact-plan subset, using MuSiQue normalized exact match against the single
stored step answer. This deliberately does not turn an arbitrary final helper
answer from a semantically similar plan into an annotated-chain result.

## Exact-plan outcomes

There were 31 exact plans among 256 RL trajectories (7 of the 16 exposed TRAIN
parents). They partition as follows:

| Outcome | Trajectories | Interpretation |
|---|---:|---|
| All annotated step answers correct; final correct | 25 | Executed reference plan and chain succeeded. |
| One or more annotated step answers wrong; final wrong | 4 | Wrong fact/relation execution. |
| Helper JSON invalid before a complete trace | 2 | Strict protocol/format failure. |
| All annotated step answers correct; final wrong/invalid | 0 | No annotation-defined final-overturn case in this restricted subset. |
| Wrong annotated chain; final correct | 0 | No annotation-defined final rescue in this subset. |

Thus the exact-plan failures are 4 wrong-fact executions plus 2 helper-format
failures, not evidence that a final routinely overturns a fully correct
*annotated* chain. The small denominator and repeated exposed parents preclude a
rate claim.

Public examples illustrate the distinction. For the public question “What is
the Charles I. Barber's place of death the capitol of?”, opaque trajectory
`u01-06064851ec6aa5a06e1ac625-c0` used the exact two reference questions but
the first helper supplied a date where the requested place was needed; the
second and final consequently followed the wrong entity. For the public Seeley
Booth agency question, all 16 RL trajectories had strict helper JSON failures;
only two happened also to reproduce the exact annotated question list, so this
is broad protocol evidence but only two exact-plan format observations.

The Colorado River/La Poudre Pass example is a useful executor warning but is
outside the exact-list numerator: several semantically similar generated plans
obtained those two helper answers and the final nevertheless returned Colorado
River. Its generated question wording does not exactly equal the annotation, so
calling it an annotation-proven “correct chain overturned” would overstate the
evidence. It motivated the completed full-source/trace-only aggregation test.

## Broader protocol signal, not answer-chain scoring

Across all RL trajectories, 29/256 had `invalid_helper`; held SFT had 12/64.
The completed held-out RL comparison changes this only weakly (SFT 12/64 helper
invalid versus RL 10/64) and its answer EM was 18/64 versus 19/64 (paired
+1.56 points, interval −6.25 to +10.94). Its three wins all involved protocol
outcomes, while both losses were scored episodes. This is not a meaningful
reasoning gain or evidence that four planner updates repaired execution.

As a descriptive prompt-shape observation, plans whose every question used the
terse `X >> relation` form had 0 invalid-helper episodes in RL (0/20) and held
SFT (0/11), versus 29/236 and 12/53 for other plans. These rows are clustered by
parent/repeat, selected by the model, and differ in question content and length;
the comparison is not randomized and says nothing about helper factual accuracy.
Do not train toward this surface form based on these counts alone.

## Decision implication

Helper-SFT is a justified component diagnostic, not an assumed improvement: the
large broad helper-format burden and exact-plan wrong-fact failures leave a
plausible frozen-executor bottleneck. The completed aggregation probe favors
retaining full-source final access; it does not remove the helper bottleneck.
Keep planner checkpoint selection and all transfer labels sealed; helper training
must not expose gold answers, aliases, support labels, or reference decomposition
in inference prompts.
