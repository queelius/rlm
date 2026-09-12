---
schema: research-checkpoint-v1
updated_utc: 2026-09-12T20:15:00Z
status: exploratory
question: Which failures should we ask reinforcement learning to fix?
---

# Preserve correct answers before training from their rewards

The strongest new result is a small but consequential harness correction.
Seven answers were correct when the model generated them, but were changed
before grading. Preserving them improved the completed task. Meanwhile, a
second check weakened the earlier tiny RL gain, and another decomposition
comparison did not support simply replacing summaries with original text.

These are preliminary findings. We are using them to choose better learning
experiments, not claiming a finished general-purpose decomposition method.

## 1. A controlled return-path change recovered seven correct answers

The task asks the model to find a particular earlier reply in a conversation
stored in a Python-accessible file and return it exactly, with a supplied
prefix. Exact includes spaces, punctuation, and line breaks. Supervised
training on 32 worked examples had taught a useful retrieval routine: on
16 separate conversations, with two attempts each, it printed the correct
target text in 30 of 32 attempts. It returned only 17 answers exactly.

Inspection showed that two runtime components removed whitespace from the
ends of model responses. In eight attempts, the model had generated the exact
answer but this processing removed required trailing spaces. That made a
correct model action receive an incorrect final reward.

We then ran a separate, prospective comparison with both operations disabled.
The checkpoint, tasks, requested sampling seeds, prompts, limits, and grader
were unchanged. We did not repair answers using the reference answer.

| Measurement | Original return path | Whitespace preserved |
|---|---:|---:|
| Exact returned answers | 17/32 | 23/32 |
| Available outcomes | 32/32 | 32/32 |

There were seven improvements and one regression. All seven improvements had
identical model action tokens, prompt tokens, generated programs, and tool
observations across the pair. They are direct recoveries of correct generated
answers. The regression had a different final generation. Overall, 29 of the
32 paired generation paths were identical; matching seeds did not guarantee
identical sampled output in every attempt.

The earlier token-level diagnostic found 25 exact generated answers. It was
not a prediction that a new run would score 25, and it does not replace the
original score of 17. The intervention scored 23. The independent comparison
keeps these three measurements separate.

This is a harness improvement, not new learning. It also gives us a better
reward signal for subsequent RL. The experiment changes two terminal whitespace
operations, not every parsing or reasoning transformation in the runtime.
The 32 attempts represent 16 context units with two decoding seeds, not 32
independent problems. This panel had been examined during earlier research.

Evidence: `analyses/openai-mrcr-terminal-clamp-paired-2026-09-12/MAIN_TERMINAL_001.json`
in the research notebook; source and detailed external-run pointers are
retained there. The earlier procedure result is explained in the
[preceding report](2026-09-12-procedure-transfer-and-exact-rewards.md).

## 2. The earlier one-step controller RL advantage did not persist

We evaluated all three fixed checkpoints again on the same 16 conversations
with a second set of decoding seeds. No checkpoint was selected after seeing
which performed best.

| Fixed model | First decoding block | Second decoding block |
|---|---:|---:|
| Starting controller | 0/15 available | 2/15 available |
| Smaller RL update | 0/15 available | 1/16 available |
| Larger RL update | 1/15 available | 1/16 available |

Every condition planned and recorded 16 attempts. Unavailable outcomes came
from prompts exceeding the service's input limit, not from missing result
files. They remain unknown rather than being relabeled as wrong answers.

The larger update repeated its original successful answer, but the starting
model and smaller update also answered that question correctly in the second
block. These successes followed broad conversation dumps, not a learned
target-selection routine. The one additional starting-model success also
followed a broad dump. There is no stable answer advantage from this recipe.

We are retiring this particular old training corpus and one-step update
comparison. This does not establish that RL or token-level importance
correction generally fails. Our next attempts start from the supervised
controller that actually learned retrieval, with the corrected return path.

Evidence: `analyses/openai-mrcr-token-tis-held-seed2-2026-09-12/REPORT.md`.

## 3. Original passages did not clearly outperform summaries

On 12 previously examined multi-hop question-answering problems, two helpers
selected passages from separate halves of the source. We compared final
answers using summaries of those selected passages with answers using the
same original passages. A summary helper saw only its selected passages, so
it could not silently summarize additional source material.

All 72 physical calls returned valid results. Summaries answered 4/12
questions exactly; original passages answered 3/12. Moving to original text
made one answer correct and two incorrect. This is a small mechanism screen,
not evidence of a general advantage for either representation.

The deployed strategies have different costs: original passages need two
selection calls and a final call; summaries also need two summarization
calls. Shared acquisition reduced experimental duplication, but did not make
the strategies equal-cost or equal-input-length. Independent source-coverage
and answer-use analysis is still underway at this report's cutoff.

Evidence: `sidecars/musique-evidence-preservation-v1/outputs/attempt-001/RESULT.json`.

## 4. A newer model was not an immediate replacement

A paired usability screen compared the released Qwen3-4B controller with
Qwen3.5-4B on eight fixed training contexts. Neither produced an exact answer
within the deliberately short two-turn allowance. The newer model emitted
valid first tool-call schemas in 8/8 cases versus 7/8, but it produced no final
text answers within that allowance. This does not rank their general ability.

An earlier attempt at this comparison was invalidated by a CPU validation
bug that repeatedly measured the tokenizer vocabulary inside a token loop.
That attempt supplies no paired capability result. The repaired comparison
completed both models in about 287 seconds; the failed attempt is retained.

## Next decisions

Fresh training-only attempts will distinguish reward collapse, final copying
errors, and retrieval errors. If successful retrieval is already consistent
but final answers vary, we will compare credit assigned to the entire model
trajectory with credit restricted to the actual final answer. If reward
variation is absent, we will change the exploration or training examples
instead of repeatedly taking uninformative updates.

A separate, outcome-blind panel of 16 longer conversations will test transfer
of the fixed supervised checkpoint. Its first two launches failed before
any model queries because of a service-binding error; a narrow repaired
comparison is queued. These failures are operational evidence, not model
accuracy results.

The publication opportunity is to show which model–harness boundaries limit
useful learning, fix a specific limitation, and demonstrate improved complete
answers on new problems. We now have a concrete boundary intervention and a
competent learning starting point. We do not yet have a compelling general RL
improvement or a learned recursive decomposition policy.
