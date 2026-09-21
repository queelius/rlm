---
question_id: selective-delegation-20260921
status: exploratory
cutoff_utc: 2026-09-21T10:05:00Z
model: Qwen3-4B-Instruct-2507
training_performed: false
---

# What the first controlled comparison tells us

An extra reasoning step sometimes helps, but this first small experiment does
not yet show that choosing a different step for each question is better than
using one fixed strategy. We also found answer-format and evidence-quality
issues that need to be separated from genuine reasoning errors before training.

## Completed: same-state pilot

We used 32 MuSiQue training questions. Each produced one initial attempt, then
four alternatives started from that same attempt. Three seeds repeated the
continuations. All final answerers retained the original documents. No model
weights were trained.

| Choice after the initial attempt | Exact answer match | Token F1 | Mean tokens per hypothetical attempt |
|---|---:|---:|---:|
| Finish using the current evidence | 28.1% | 37.9% | 5,817 |
| Ask a helper to reconsider | 28.1% | 38.4% | 9,479 |
| Ask the proposed evidence question | 32.3% | 42.8% | 9,449 |
| Work through the proposed subquestions | 34.4% | 45.0% | 9,416 |

These are 96 continuations per choice on 32 questions, **not 96 independent
questions**. The subquestion-minus-finish difference is 6.25 percentage points;
the exploratory parent-bootstrap interval is approximately −1.0 to +15.6 points.
The estimate is uncertain. Two parents also share an atomic question component,
so these conditional parent-bootstrap intervals are not a full independence
guarantee. The extra step uses about 62% more input-plus-output tokens. This
token measure is not a direct estimate of monetary cost or FLOPs.

A diagnostic chose each question's action using its other two repeats, then
scored the excluded repeat. It achieved 33.3%, compared with 34.4% for the fixed
action selected on those same other repeats. Even this selector has privileged
outcomes from the same question; it is not deployable. Its failure to beat the
fixed strategy weakens the case for immediately training an action router on
this small table. It does not prove routing can never help.

## Completed: changing only the final-answer instruction

All 384 new final calls completed, with the initial attempts and helper reports
reused unchanged. The additional instruction asks for a short answer phrase,
without an explanatory sentence, while preserving necessary qualifiers.

| Choice | Original instruction | Short-answer instruction |
|---|---:|---:|
| Finish | 28.1% | 43.8% |
| Reconsider | 28.1% | 43.8% |
| Targeted evidence | 32.3% | 46.9% |
| Proposed subquestions | 34.4% | 45.8% |

The 11–16 percentage-point changes are substantially larger than the differences
between helper strategies in this pilot. The subquestion-minus-finish gap falls
from 6.25 to 2.08 points. This is an instruction effect, **not training or a
delegation improvement**. Although the instruction targets answer format, it may
also change which answer the model selects; we have not isolated a purely
cosmetic mechanism.

With the new instruction, other-repeat per-question selection reaches 50.0%
versus 46.9% for the corresponding fixed strategy, an exploratory +3.1-point
difference with conditional interval 0 to 8.3 points. This is weak possible routing
headroom, not established learnability.

## Completed: 32 different questions under the clearer answer contract

On a fresh exploratory validation block, finishing scored 37.5%, reconsidering
31.3%, targeted evidence 40.6%, and proposed subquestions 36.5%. Each percentage
uses 96 continuations on 32 questions. All 704 calls completed successfully.
The helper choices required about 60% more tokens than finishing.

The privileged other-repeat selector reached 41.7%, versus 40.6% for its fixed
strategy; the conditional parent-bootstrap interval for that difference is
0 to 3.1 points. These results do not justify a claim that a learned action
router is likely to help. They motivate testing the quality and execution of
the actual subquestions before optimizing a choice among these four actions.

## Completed: supplying reference subquestions

For the 26 two-hop training questions, fresh helper/final calls scored 35/78
(44.9%) with model-generated questions and 39/78 (50.0%) with annotated questions.
There were four improved continuations and none harmed, but the gains came from
only two parent questions. All 312 calls completed. Annotated answers were never
provided. Reference questions are still privileged information, so this is a
diagnostic, not a deployable improvement.

Inspection revealed a crucial limitation: a helper can ignore the supplied plan.
For a question about when a car's manufacturer ended, even the helper given
manufacturer-focused reference questions answered with the car's final production
year. The original provisional answer and composed question remained visible.
Therefore this small comparison cannot cleanly distinguish poor plans from a
failure to execute the plans.

The next comparison crosses model/reference plans with bundled/step-by-step
execution. In the step-by-step condition, each helper sees only its current
subquestion and the documents; a `#1` reference is replaced by the first helper's
actual answer. Both final answerers retain the original checkpoint and full
documents. This changes visibility, call granularity, and helper response format
together, not just the number of calls. The helper output allowance is 384 tokens
in both conditions, but step-by-step execution pays for an additional input pass.
Generated plans have no literal `#1` links in this panel, whereas all reference
plans do; this execution-compatibility difference must be reported, not repaired
silently or mistaken for a pure content-quality effect.

## What changed our next experiment

Some final strings contain the right fact inside a long explanation and fail
exact matching. Other failures are real: a helper sometimes invents an evidence
contradiction, or mistakes a manufacturer's closing date for a product's final
production year. One dataset example supplies a paragraph containing the right
city but not the relation needed to establish a performer's birthplace.

The final-answer replay above preserves the same final sampling seed and token
limit, with no gold-dependent answer cleanup. The harder four-hop transfer panel
remains unused. See
[DATA-AUDIT.md](DATA-AUDIT.md) for the source-data caveat.

We investigated whether the proposed subquestions themselves are wrong.
For example, the model asks about a product's production dates when the question
asks about its manufacturer. The completed reference-question diagnostic used
for the 26 two-hop training parents, holding the number of subquestions at two.
Both model and reference plans receive fresh matched helper/final calls. Only
annotated questions, never annotated answers, enter the reference helper prompt.
This is still privileged annotation assistance, not a deployable result. The
execution limitation above motivates the more explicit follow-up.

## Evidence and scope

External root: `/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.
Completed report: `analysis-pilot-001.json` and `.md`; raw receipts: `pilot-001`.
Final-instruction report: `analysis-format-001.json` and `.md`; new receipts:
`answer-format-001`. Total acquisition across these two studies: 1,088 calls,
3,491,340 tokens; only 384 calls were newly executed for the second study.
All 704 planned pilot calls returned; no missing checkpoints, malformed finals, or
unknown call usage. Pilot collection consumed 2,253,467 input-plus-output tokens;
shared initial attempts are counted once in that physical total. Per-policy
cost includes one initial attempt and only its chosen continuation.

This is a prerequisite diagnostic, not an RL improvement, a generalization claim,
or evidence that learning to delegate is new. The advisor deck is unchanged:
the current finding is too provisional and technical to replace its main story.

Further completed artifacts: `analysis-validation-001.json` and `.md`, raw
`validation-span-001`; `plan-probe-001/SUMMARY.json` and its immutable calls and
episodes. The live execution comparison is `execution-probe-001` with sealed
`source-004`. No new optimizer has run as of this cutoff.
