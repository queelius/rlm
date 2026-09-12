---
date: 2026-09-12
evidence_cutoff_utc: 2026-09-12T19:35:00Z
status: exploratory
questions:
  - Can demonstrations teach the controller a useful retrieval procedure?
  - Are exact-answer rewards measuring what the model actually produced?
  - Do focused helper requests improve answers over broad requests?
---

# A learned procedure works; the return path also matters

The clearest new result is supervised learning, not RL yet. After more training
on the same 32 demonstrations, our small model learned how to find a requested
reply in a conversation. It then used that procedure on separate conversations.
We also found that the runtime sometimes changed an exact answer before grading
it. That matters directly for the next reinforcement-learning experiment.

These findings update the [earlier controller report](2026-09-12-controller-learning-and-transfer.md).
All comparisons below are exploratory. Checkpoints were fixed in advance, but
the separate evaluation conversations had already appeared in other experiments.

## What the model learned from demonstrations

The task is to recover a particular reply from a long conversation stored in a
Python-accessible file. A simplified example is:

```text
User: Write a poem about rain.
Assistant: First poem ...
User: Write a poem about rain.
Assistant: Second poem ...

Task: Return the second poem about rain, exactly as it appeared.
```

The training examples demonstrate a procedure: read the message list, find user
requests matching the task, select the requested occurrence, and print the
following assistant reply. The model learns to generate the Python program
and then the final answer. This is supervised fine-tuning (SFT): learning from
worked examples, not discovering the procedure through rewards. The example
above is simplified; it is not a quoted training record.

Four updates had not taught a working procedure. We continued that exact run
to a fixed total of 32 updates, restoring the weights, optimizer and random
state. We did not change its examples, learning rate, or training objective.
The additional 28 updates took 16.1 minutes; subsequent evaluations took longer
and are separate from training time.

| Evaluation | Before this procedural training | After 32 updates |
|---|---:|---:|
| Separate conversations: exact returned answers | 2 / 29 completed | 17 / 32 completed |
| Separate conversations: planned attempts | 32 | 32 |

There are 16 separate conversations, each tried with two random seeds—not 32
independent conversations. Among the 29 pairs with both outcomes available,
training corrected 15 answers and lost none. Three starting-model trajectories
grew beyond the model's context limit and did not produce a final outcome;
we retain these separately, rather than counting them as ordinary wrong answers.

On the training conversations, all 32 first programs matched the demonstrated
procedure and printed the correct target. On the separate conversations,
30 of 32 attempts printed the correct target without dumping the whole input.
The remaining two attempts concern the same conversation: the task asked for
an **email**, but the generated program searched for a **message**. It found no
matching request. This is a concrete remaining selection error, not a failure
to understand Python's list structure.

This supports transfer of a narrow procedure within this public task family.
It does not yet establish general problem decomposition, recursive delegation,
or transfer to another dataset. None of these attempts called a helper.

## Finding the right text is not the same as returning it unchanged

Consider an exact target ending in two spaces:

```text
Required answer: "The answer is here.  "
Returned answer: "The answer is here."
```

An exact-text scorer marks the second string wrong. Inspection of saved model
tokens found that, in eight separate-conversation attempts, the model actually
generated the exact required text. The response parser and the harness's final
return path removed surrounding whitespace before scoring it.

We replayed the saved responses through the installed parser. The original
reported score remains **17 exact returned answers**. A separate diagnostic
finds **25 exact generated answers** before these trimming operations. This is
not a replacement score or a completed intervention result.

Among attempts that retrieved the right text, five failures are real model errors:
four omit spaces already in the
generated tokens, and one changes a digit despite having printed the right text.
Together with the two failed retrieval attempts, these account for all 15
non-exact returned answers after training.

Our next controlled test disables only the two terminal trimming operations in
a separate runtime process, keeping the model, tasks, seeds and scorer fixed.
It must not repair the model's own punctuation, digits or whitespace. Before
more exact-reward RL, we need a return path that does not erase some successes.
This is a contract mismatch for exact-text tasks, not a claim that trimming is
wrong for every application.

## The controller RL result is still weak

We trained two one-update branches from the same starting checkpoint and saved
gradient. Their learning rates differ tenfold, and their actual parameter
movements differ tenfold. This rules out a silently missing update.

In the first evaluation block, the starting model and smaller update each
returned zero exact answers among 15 available attempts; the larger update
returned one. All three had the same additional unavailable attempt. The sole
new success followed a whole-conversation printout, not a correct retrieval
program. It is not yet persuasive evidence of useful RL learning.

Only two training question groups supplied a reward contrast. A second fixed
random-seed block has completed for all three models and awaits analysis.
We will retain every arm,
rather than testing only the successful model or conversation. The now-usable
SFT controller offers a better next starting point: fresh rollouts can reveal
whether failures concern retrieval, copying, reward informativeness, or which
part of the trajectory receives the update.

## Focused follow-up questions did not help this first decomposition screen

We tested 12 public MuSiQue questions that require combining facts. Two helpers
each read half the source paragraphs and produced reports. All report-based
conditions reused the identical first reports and planning step.

| What the final answering model receives | Exact answers / 12 |
|---|---:|
| The initial reports; stop gathering information | 1 |
| Initial reports plus two broad follow-up reports | 1 |
| Initial reports plus two focused follow-up reports | 1 |
| The full original source, in a separate control | 3 |

The broad and focused conditions used the same number of extra helper calls.
They solved the same one question. All 48 final outcomes were available, and
the audit reconstructed all 132 physical model calls. There are 12 independent
question contexts, not 48 independent examples.

This does not support spending more calls on this particular follow-up recipe.
The full-source control sees different information and is not a cost-matched
proof that decomposition is generally harmful. The inputs fit the model's
context window, and we supplied the call graph: the model did not learn when
to delegate. We are checking whether reports omitted important facts, focused
questions requested the wrong facts, or the final model failed to use them.

## What looks worth pursuing

The strongest near-term direction is a controlled study of the path from
retrieval to a verifiable answer. We now have a model that usually retrieves
the right text, identifiable model errors, and an identifiable runtime change
that can alter the reward. That gives RL a much more informative starting point
than repeated attempts from a controller that cannot use the input.

Next, test the unchanged procedure on longer, non-overlapping conversations;
compare fresh RL learning contrasts after the narrow runtime fix; and use the
helper-report failures to design a sharper information-preservation test.
Sixteen longer conversations have been selected without inspecting model
outcomes. Their external inputs span roughly 16,500–30,500 tokens; this does
not mean those tokens are all inserted into the controller's neural prompt.

For publication, SFT alone is a prerequisite, not a novel claim. A stronger
result would identify a specific bottleneck, change it under a matched
comparison, improve complete answers, and repeat that improvement on genuinely
new material. We have promising ingredients, not that finished claim yet.

## Evidence and checkpoints

The [research notebook](https://github.com/queelius/rlm-research) preserves selected
source, reports and artifact hashes. The external store is
`/project/alex_phd/runs/rlm-research-r4`.

- SFT continuation: `sidecars/openai-mrcr-procedural-sft-continue32-v1`; final step commit SHA `3e002acdadc0256097efedd62cdc6af5c7d219b5f7937c9bc952a9ec345102ff`.
- Independent SFT readout: `analyses/openai-mrcr-procedural-sft-dose32-readout-2026-09-12`; final report SHA `32b72266cf8d2484f4c21db55650d38274990afc88f60dfe2ccfeb2a16bc3a51`.
- Held token/parser replay: `HELD_DECODER_SEAM_V2.json` in that folder, SHA `ba15522db4bf25620a4fae2061d25336eae651eeeeec3d6e1302bb2e95f14dee`.
- RL readout and mechanism: `analyses/openai-mrcr-token-tis-held-three-arm-2026-09-12`.
- MuSiQue independent outcome: `analyses/musique-task-directed-followup-independent-2026-09-12/outcome/REPORT.json`, SHA `92e5ae4f5bf4114a672ac44bb60d19cf229be4762de3ead0641ac1ee44fdb11b`.

The model and optimizer checkpoints remain outside Git. A GitHub push is not
a backup of those heavyweight artifacts.
