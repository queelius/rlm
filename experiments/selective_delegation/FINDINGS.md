---
question_id: selective-delegation-20260921
status: exploratory
cutoff_utc: 2026-09-21T10:44:29Z
model: Qwen3-4B-Instruct-2507
training_performed: true
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

## Completed: explicit step-by-step execution

The four conditions scored 24/52 (model plan, bundled), 28/52 (reference plan,
bundled), 24/52 (model plan, isolated), and 26/52 (reference plan, isolated).
These are 26 training questions with two repeats. Two reference-isolated helper
responses failed the JSON contract; they remain failures in the denominator.
The job produced 518 actual calls, not the520-call maximum: those two failed
episodes did not reach their final call.

There is no overall advantage for isolated execution in this panel. The car
manufacturer example nevertheless reveals a useful failure mode: both isolated
reference helpers found the right manufacturer and closing date (1954), but
the final model returned the earlier mistaken production year (1932).
This motivates removing the provisional answer or the whole original checkpoint
from otherwise identical final prompts. This follow-up reuses actual helper
outputs; it does not invent improved reports or alter the evidence.

## Completed: question-plan supervised training

The planner completed48 optimizer updates on256 training examples, with frozen
base-model helpers reserved for evaluation. Its input is the original question
and public document-title index; its target is a list of linked subquestions,
not an answer. Training used a fresh rank8 LoRA (16,515,072 trainable parameters),
AdamW at1e-4, and three passes over the examples. Optimizer work took297.2 seconds;
this is not the duration of the full research session or evaluation campaign.

The token-weighted training loss fell from2.250 in the first epoch to0.663 and
0.468 in the next two. This establishes fitting to the training examples, **not
better answers or generalization**. Update48 is the fixed primary checkpoint.
The completed comparison used base and trained planners on32 validation questions,
two seeds, identical isolated execution, and unchanged base helpers/final model.
Isolated execution was chosen to make the dependency contract explicit, not
because it won the preceding comparison. The new planner has no provisional
answer, so it must not be compared naively with the old checkpoint architecture.

## Completed: supervised planner validation

The unchanged planner answered16/64 attempts correctly (25.0%); the supervised
planner answered19/64 (29.7%). The paired improvement is4.7 percentage points,
with a parent-bootstrap interval of−4.7 to+15.6 points. This is a small, uncertain
gain, not established better reasoning. The trained planner eliminated five
invalid root plans, but helper-format failures persisted (11 base,12 trained).
Of six improved attempts, four previously failed the protocol; of three harmed
attempts, two newly failed the protocol. All520 model calls returned.

The trained condition used585,179 tokens versus660,077 for the unchanged
planner, partly because it usually produced shorter plans. This is not a
compute-matched comparison. See `analysis-planner-sft-001.json/.md` in the external
study root for receipts, paired intervals, and failure accounting.

Subsequently completed simple controls on these same32 questions scored17/64
for one direct answer (26.6%) and14/64 for an original-question helper followed
by a final answer (21.9%). The direct system used2,996 tokens per attempt versus
9,143 for the supervised planner. We therefore do not yet have convincing evidence
that the learned decomposition system earns its extra computation. These controls
share the evidence and final sampling contract but have different prompts and
call budgets; the original-question helper is not a learned decomposition.

## Completed: removing the earlier answer did not solve the final-step problem

We made412 new final calls using the execution probe's saved helper outputs.
Removing only the earlier answer, or removing the whole initial attempt, did not
improve any arm's aggregate exact-match count. The manufacturer example still
returned1932 despite a helper reporting1954. Thus the motivating example does
**not** establish that the earlier answer caused the mistake. The final model
may instead misunderstand the question or disregard useful helper evidence.

This distinction matters: the final model also rescued wrong last-helper answers
in23 episodes across the model/reference isolated conditions, while overturning
a correct last-helper answer in only two. Simply trusting the last helper is
not an evidence-supported default. These are descriptive conditional counts,
not a causal intervention on the final model.

## Completed training; evaluation pending: learn plans from answer rewards

An actual on-policy RL run updated only the planner adapter, starting from
the supervised checkpoint. Each update samples four new plans for each of16
fixed training questions, executes them with frozen helpers, and rewards exact
final answers. The maximum is four updates; the final committed update will be
evaluated without selecting it by validation performance. All four updates
changed the adapter weights, with nonzero gradients and measured
before/after likelihood changes. This proves that optimization ran, not that it
improves held-out answers.

The run completed256 fresh trajectories and1,053 model calls in17.1 minutes,
with no generation failures or unknown token usage. Training rewards were
37/64,36/64,40/64,37/64 across the four batches: no monotonic improvement, and
these changing sampled trajectories are not a fixed evaluation set.

The other32 validation questions are reserved for a matched SFT-versus-RL
readout. Additional frozen readouts cover64 four-hop MuSiQue questions and32
HotpotQA explorer-sample questions. The latter is a small diagnostic, not the
canonical HotpotQA development benchmark. Simple direct-answer controls are
included so a more elaborate system does not win merely by lacking a baseline.

A separate frozen-trace experiment asks whether showing the final model all
documents hides the consequences of good and bad plans. It compares new finals
with documents against new finals with only the question, plan, and actual
helper answers. Higher reward variation alone would not establish better credit
assignment; answer quality and repeat noise must be examined together.

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
episodes. The completed execution comparison is `execution-probe-001` with sealed
`source-004`. Planner training: `planner-sft-001`, committed checkpoints every
eight updates through `checkpoint-0048`; sealed trainer/evaluator: `source-005`.
The completed earlier-answer replay is `checkpoint-replay-001` from `source-006`:
412 actual new finals after the two source helper failures. The SFT validation
report is `analysis-planner-sft-001.json/.md`. Actual RL training is
`rl-planner-001`, source007, checkpoints0001..0004. Its held-out readouts and
aggregation/transfer queue have not yet established an RL benefit at this cutoff.
