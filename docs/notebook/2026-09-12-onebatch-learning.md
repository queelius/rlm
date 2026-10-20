---
date: 2026-09-12
cutoff_utc: "2026-09-12T04:27:00Z"
question: rq:rl-effective-feedback
status: training_complete_paired_evaluation_running
evidence_level: exploratory
---

# Can the model learn from its own successful and unsuccessful attempts?

We have completed a new feedback diagnostic and trained two small updates from
it. The answer-quality comparison is running; we do not yet know whether either
update helps. This continues the [approved research direction](../research-plans/2026-09-11-learned-decomposition.md).

## What the diagnostic established

We asked the existing model to attempt each of twelve training tasks four times.
All 48 attempts produced authenticated, usable training records: 26 answers were
correct and 22 were wrong, with no missing outcomes or recorded integrity failures.
Five tasks produced both successes and failures, five were always solved, and two
were never solved in these four samples.

The five mixed tasks provide a direct contrast for the current RL objective:
increase the probability of the successful attempts relative to unsuccessful
attempts on the same task. The all-success and all-failure groups supply no
within-group final-answer contrast for this particular objective. That does not
mean they could never be useful to another learning method.

These are previously used training contexts, not evidence of generalization.
The diagnostic took 604.125 seconds, including model-service startup, collection,
export and release, and recorded 230 physical model calls.

## What was trained

Both copies started from the same supervised-trained Qwen3-4B root model. The
helper model stayed fixed. We changed only the small trainable adapter on the root
model, using one optimizer update at learning rate 0.00001.

| Learning method | Examples used | Root output tokens | Loading, training and saving |
|---|---:|---:|---:|
| Reward-based learning (RL) | All 20 attempts from the five mixed tasks | 9,495 | 109.632 seconds |
| Learning from correct examples (self-SFT) | All 26 correct-answer attempts | 7,134 | 55.031 seconds |

Training loss applies only to the root model's generated actions, not to the
question, environment observations or helper outputs. Both updates produced finite
gradients, changed weights, and saved adapter, optimizer and random-state files.
Those checks show that learning code executed, not that the model improved.

The two methods share the same 48 collected attempts but select different subsets.
They therefore do not have equal training-token doses. Correct final answers also
do not prove that every selected program followed the intended method.

## What happens next

An automatic sequence compares the unchanged model and both updated models on
the same 72 questions, with matched prompts, seeds and helper. These evaluation
contexts are separate from the new training batch but have already been inspected
in earlier research. This is an exploratory comparison, not a fresh confirmation.

At 04:24 UTC the baseline had 40 attempt records, including one timeout and 39 saved
episodes. We preserve unavailable answers separately from wrong answers. The two
trained evaluations are queued automatically after baseline release; results are
not yet complete at this document's cutoff.

A second prepared diagnostic asks: **If the helper returned correct local answers,
which mistakes would the root model still make?** The helper receives only the
requested local records; it does not supply the final task answer. This artificial
condition is a diagnosis tool, not a deployable improvement or a training result.

Together these comparisons guide whether the next effort belongs in the learning
objective, helper reliability, or the root model's decomposition and aggregation.
One small update cannot settle whether RL works. A promising difference needs
additional batches, seeds, method checks and new task/source data.

## Execution and resume pointers

All paths below are relative to `/project/alex_phd/runs/rlm-research-r4`.

- Completed diagnostic: `sidecars/root-qs6-feedback-diagnostic-v1/outputs/attempt-001/`.
  Read `DIAGNOSTIC_SUMMARY.json`, `OWNER_TERMINAL.json` and
  `collection/export/EPISODES.json`.
- Saved learning updates: `sidecars/root-qs6-onebatch-learning-v1/outputs/`.
  Checkpoints are `rl/checkpoint-0001` and `sft/checkpoint-0001`.
  Their state-file SHA-256 hashes are respectively
  `f7df3b5bc0787c9c6f80e680c48e738be80b3350f15a8c890548defcd8d91452` and
  `0470dd62aabbded73ed1234fc00cce2e229e40728fca04ee3be5dea2647a49cb`.
- Evaluations: `sidecars/root-qs6-matched-eval-v1/outputs/`.
  The running supervisor is
  `operations/2026-09-12-onebatch-evaluation/outputs/attempt-001/`.
  Read its `STATUS.json` before launching another GPU owner.
- Perfect-helper diagnostic: `sidecars/root-qs6-oracle-diagnostic-v1/READY_V2.json`.
  This is prepared, not launched, at the cutoff.
- Detailed training report: `analyses/qs6-onebatch-learning-2026-09-12/TRAINING_STAGE.md`.
  Consult `RESEARCH_QUEUE.md` for later status; this checkpoint is intentionally dated.

No successor ran after the diagnostic finished the previous evening. The roughly
9-hour-43-minute gap before resumption was not training time; it was an operational failure to
keep this session's useful GPU work moving. The new accepted evaluation sequence
does not depend on another chat message to start each next stage.

The user explicitly authorized spending the remaining shared account allowance,
including the earlier slide reserve. Monitoring continues; an expected extra reset
is not assumed until the account endpoint reports it.

Editorial decision: do not update the advisor slides based only on successful
gradient computation. Revisit them when completed comparisons change the findings
or the next research decision. Git preserves this report and resume pointers, not
the external checkpoints, environments or raw run data.
