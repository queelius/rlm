---
date: 2026-09-13
status: exploratory
question: Can organizing the input and reducing each helper's scope improve selection?
evidence_cutoff_utc: '2026-09-13T18:28:00Z'
---

# Organize the information, then give each helper a smaller decision

The strongest new lead is a change to the model's working environment, not a
training gain. On a small fresh comparison, asking one helper about each
candidate recovered the entire correct set in 10 of 12 attempts. Asking the
same model about all candidates at once recovered it in 4 of 12 attempts.
The advantage persisted on twelve new problems: 22 of 24 complete answers,
compared with 10 of 24 for whole-input true/false decisions. The smaller requests
cost much more input overall. Both tests remain within one generated task family.

This follows the [earlier research checkpoint](2026-09-13-from-answer-delivery-to-information-selection.md).
All experiments below use the released Qwen3-4B-Instruct-2507 model without an
adapter. They concern one selection step in our generated database task, not
the complete planning problem or learned recursive decomposition.

“Complete correct” here means exactly the right candidates, ignoring list order.
Sorting violations are recorded separately; these are not strict-format scores.

## What does the model have to do?

The input contains candidates, records of their changes and checks, and a rule
describing which candidates qualify. The answer must include every qualifying
candidate and no others.

For example, suppose the rule requires capacity of at least 12 and a passed
security check. A candidate starts with capacity 10, receives an applied update
of +2, and passes its check. It qualifies. Another candidate reaches capacity
12 but fails the check. It does not qualify. This is a simplified illustration,
not a quoted experimental example.

Python can bring related records together and calculate their current state.
It does not decide which candidates qualify. The model still applies the rule.

## Grouping records helps even when it makes the input longer

We compared three presentations of the same 12 new problems, with two attempts
per problem. Every candidate remained available in every condition.

| What the model sees | Complete correct answers | Input tokens, all attempts |
|---|---:|---:|
| Original separate tables | 1/24 | 102,048 |
| Related records grouped together, with updates still unresolved | 4/24 | 108,788 |
| Each candidate's current values and latest checks | 8/24 | 48,856 |

Grouping gained three complete answers and lost none. Resolving the updates
gained another four and lost none. Grouping reduced incorrect inclusions but
also missed a few more eligible candidates; resolving updates mainly recovered
eligible candidates. Neither transformation improved every individual metric
on every problem.

The grouped input was longer than the original tables. That makes “the model
only benefited from shorter input” an insufficient explanation for this result.
However, wording also changes; this is not a perfectly isolated test of one
mechanism. There are 12 independent problems, not 24 independent problems.

## Shorter answers alone did not solve the problem

On a separate 12-problem panel, we kept the organized input unchanged but changed
the answer format: either list the qualifying candidates or make one true/false
decision for every candidate.

Complete correct answers fell from 16/24 for lists to 10/24 for true/false
arrays. There were three gains and nine losses. All nine losses occurred on
arrays of the correct length, so this was not just malformed output. Three
other arrays were one decision short and were counted as failures, not repaired.

The arrays used far fewer output tokens, but only about 4% fewer total tokens
once the input was included. Simpler output is not necessarily better reasoning.

## Giving each helper one candidate is more promising

We then fixed six new problems before generating any model answers. They have
6, 12, or 20 candidates, with two problems at each size and two attempts each.
Every condition uses the same organized current-state records and eligibility
rule. The singleton condition gives each helper exactly one candidate and then
combines all of their true/false decisions. No candidates are filtered out.

| How the work is divided | Complete correct answers | Model calls | Input tokens |
|---|---:|---:|---:|
| One call returns a list | 4/12 | 12 | 27,446 |
| One call returns all true/false decisions | 4/12 | 12 | 27,850 |
| One helper decides about each candidate | 10/12 | 152 | 103,002 |

The helpers made 150 of 152 individual decisions correctly. They found every
eligible candidate and made two incorrect inclusions: the same candidate failed
the quality threshold but was accepted in both attempts. They recovered the
complete set on all four twenty-candidate attempts; both whole-input conditions
recovered none of those four. All 176 calls returned, and every singleton
response was usable. Whole-list and whole-array invalid outputs remain failures
in their full denominators.

This result does not demonstrate learned planning, a learned stopping rule, or
recursive depth. The decomposition is fixed and the local rule is separable
across candidates. Input cost is about 3.8 times the list condition, and the
prompts and output contracts differ. Six problems are too few for a strong
generalization claim. An independent raw-response audit found no discrepancies.

### The advantage persisted on fresh cases

A second test fixed twelve new problems covering 6, 12, or 20 candidates, shorter
or longer update histories, and one or three check revisions. Each was answered
twice. Every candidate was retained, and all answers were collected.

| Candidate count | One call makes every decision | One helper per candidate |
|---|---:|---:|
| 6 | 4/8 | 8/8 |
| 12 | 3/8 | 8/8 |
| 20 | 3/8 | 6/8 |
| All attempts | 10/24 | 22/24 |

There were twelve gains and no losses, spread across seven problems. Even after
restricting the comparison to usable whole-input answers, helpers corrected
fifteen individual decisions and damaged none. They made 302 of 304 individual
decisions correctly. The two errors were the same candidate accepted twice
despite capacity of −4 when the rule required at least 8. An independent audit
checked every actual response and found no discrepancies.

The cost is still substantial: 304 helper calls instead of 24 whole-input calls,
207,004 input tokens instead of 56,088, and about 3.7 times as many total tokens.
These are not equal-compute comparisons. More complicated record histories
test the complete Python-plus-model system: Python, not the model, resolves
the latest records. We have not established transfer to a different dataset,
learned planning, or that splitting as much as possible is generally optimal.

## What this changes about the research plan

We now have a concrete question about the harness: can it identify the decisions
that need a smaller view, rather than pay for a separate helper for every item?
A useful next comparison would give targeted and randomly chosen refinements
the same number of helper calls. We must measure both complete answers and cost.

In parallel, a fresh batch of 64 model attempts provides variation for an RL
comparison: give feedback to individual decisions versus giving every decision
the same overall-answer feedback. The cases and future evaluation inputs were
fixed in advance. This batch produced 33 complete answers; it is training data,
not a training improvement. No new RL improvement is claimed here.

The bigger research question remains whether a model can learn to choose an
effective decomposition. These experiments establish useful comparisons and
expose failure modes; they do not yet answer that bigger question.

## Evidence and reproducibility

Immutable run artifacts live outside Git under
`/project/alex_phd/runs/rlm-research-r4/sidecars/`:

- `b05-state-representation-v1`: 72 calls; result SHA-256
  `07fa408cedaceaf7f3882c0e55aab8b379f157617ce05260c8c52f28ec525fb2`.
- `b05-decision-vector-v1`: 48 calls; result SHA-256
  `813025db329b910488c4966a76980029b5ed4d114c27ce63a1c0660230dfccfa`.
- `b05-singleton-decomposition-v1`: 176 calls; frozen admission and per-call
  requests, responses, token IDs, log probabilities, costs, and owner terminal.
- `b05-varied-vector-rollouts-v1`: 64 training attempts; separately frozen
  12-case future evaluation panel. No evaluation calls in this collection.
- `b05-singleton-decomposition-replica-v1`: 328 calls on twelve fresh cases;
  independent audit report SHA-256
  `f969d2d649721d38a1c0c280dadbe43c0b9672d2fb1a7d32e96b2079f915c6f9`.

Each result is `outputs/attempt-001/RESULT.json`. Public-input manifests and
source closures preserve generation and sampling seeds. The representation
and initial vector and singleton comparisons have independent raw-response audits under
`analyses/` in the same research store. GitHub documents are not a backup of
external run data or model checkpoints.
