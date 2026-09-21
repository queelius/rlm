# A bounded on-policy planner learning test

This run asks whether fresh task rewards can improve the content of a question
plan after the initial supervised training. It updates only the planner adapter.
The helpers and final answering model retain the released base weights. This is
an exploratory learning test, not a confirmatory benchmark result.

Start from the committed SFT checkpoint at update 48. Sort the complete 256-parent
training pool by opaque parent ID, shuffle with seed 2026092108, and take the first
16 parents. This selection is independent of their observed outcomes. Keep these
same 16 parents for at most four updates. Each update collects four newly sampled
planner outputs per parent, giving 64 rollouts per update and at most 256 total.

The planner sees the original question and document title index, with the exact
prompt and strict JSON parser used by `eval_planner.py`. It receives no reference
questions, annotated answers, supporting-source labels or previous rewards.
Sample roots at temperature 0.8 with at most 128 emitted tokens. Execute each
valid plan through the evaluator's isolated helper prompts and dependency binder.
Distribute a total helper output budget of 384 tokens as `floor(384/n)` per step.
The final has a 128-token cap and the evaluator's short-answer instruction. All
helpers and finals sample at temperature 0.5 with the adapter disabled.

Within each parent and update, use common helper/final seeds across all four
candidates, while varying the root seed. These common seeds reduce one source of
variation but do not make the downstream outcomes independent or deterministic.
Plans of different lengths receive different per-helper output caps and incur
different repeated-input costs. This is the existing evaluator's execution
contract, not a comparison with matched total input computation.

## Reward and the admission decision

Reward is official MuSiQue alias-max exact match, either zero or one. Returned
invalid plans, invalid dependency references, invalid helper JSON and invalid
final answers receive zero. A generation exception is different: its group is
missing and the bounded run halts immediately. It is never converted into a zero
reward, silently retried, or repaired with a fallback.

Before each update, require at least two of the 16 groups to contain at least two
distinct valid plans and both reward values among the valid plans. Variation
caused only by valid versus invalid root syntax does not admit an update. Record
all groups, including invalid outcomes. Also report whether valid-plan variation
includes downstream protocol failures, and whether fully scored valid final
answers themselves show a reward difference. Distinct serialized question lists
are an operational diversity measure, not a claim of semantic diversity.

If admission fails, preserve the complete pilot batch and stop without creating a
fresh optimizer or taking another update. When a later batch fails admission,
retain the earlier committed updates and stop. Admission is a small exploratory
signal check, not a significance test or evidence that learning will generalize.

## One fresh policy-gradient pass

For candidate i, subtract the mean reward of the other three candidates from its
reward. Optimize the negative sum of that advantage times the sum of its emitted
root-token log probabilities, divided by the fixed batch size 64. Use logits
divided by 0.8 when computing these probabilities. Include every emitted token,
including EOS when it was generated; do not invent EOS after a length-limited
completion. There is no token-length normalization, PPO ratio, KL penalty or
helper/final loss. Zero-advantage examples may skip backward computation, but the
denominator remains 64 and no groups are selected out of the batch.

Use one fresh gradient pass per batch, AdamW with learning rate 0.00002, zero
weight decay, and gradient clipping at norm 1. The base is BF16; rank-8 LoRA
parameters and optimizer state are FP32. Only LoRA parameters are trainable.
Generation uses the HF cache. Gradient computation disables the cache and enables
nonreentrant gradient checkpointing, with dropout disabled.

Save actual generation token IDs and selected sampling log probabilities. Before
each update, recompute all root token probabilities locally under the unchanged
current adapter, and after the update score those same token sequences again.
Record finite differences between cached generation, full forward scoring, and
the gradient pass; BF16 kernel differences are not treated as exact equality.
These differences need inspection before making precise policy-likelihood claims.
Save the gradient norm, adapter parameter change, reward variance, plan validity,
plan diversity, all call seeds and observed physical token costs.

## Ownership, artifacts and resume

Only main launches this job, after acquiring the existing exclusive GPU
coordinator lock. Stop at three cumulative hours or allocation end minus ten
minutes, whichever comes first. Each generation has the evaluator-style soft
90-second limit; HF checks it between decoding iterations, so an in-progress
forward can finish after that limit. Every returned native token sequence is
checked immediately and written to its own receipt. An inference exception stops
the owner rather than continuing through failing groups.

Use the existing training interpreter:
`/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python`.
The script records Python, torch, transformers, PEFT, CUDA and GPU identity;
source dependencies, input cases, base manifest and adapter files are hashed.
The run does not install packages or mutate another environment.

The CLI is:

```text
python rl_planner.py --cases <inputs-001/cases.jsonl> \
  --adapter <planner-sft/checkpoint-0048> --output <new-rl-output> \
  --updates 4 --hours 3
```

Each admitted update is committed atomically through the existing trainer's
checkpoint writer, including adapter files, Adam state, Python and torch CPU/CUDA
RNG state, completed update and next cursor. No NumPy sampling is used. The
checkpoint is committed before optional post-update diagnostics can be interrupted.
An explicit resume restores the latest committed optimizer and RNG state, and
retains the cumulative time budget. An existing uncommitted rollout batch causes
resume to stop: it must not silently regenerate failed or interrupted rollouts.

Outputs preserve batches, calls, starts, episodes, admission diagnostics,
before/after log probabilities, atomic checkpoints, ownership and terminal
receipts. If interruption prevents post-update scoring, the checkpoint remains
available and the terminal receipt records the failure; do not claim complete
before/after diagnostics for that update. Evaluation is a separate later job.

Sixteen reused training parents and four updates are too small to establish
generalization. RLOO differences can reflect plan content, plan length, protocol
compliance or downstream sampling. Report those distinctions and compare a fixed
RL checkpoint with its SFT parent on untouched evaluation parents and matched
execution contracts. A training reward increase alone is not that comparison.
