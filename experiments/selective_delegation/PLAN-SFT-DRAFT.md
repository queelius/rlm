# Question-plan SFT design

Update, September21 10:09UTC: The small48-update run is accepted and queued behind
the execution diagnostic. The reference-question signal is weak (two improving
parents), not an established effect. This inexpensive training comparison tests
whether executable question plans are learnable; it does not presume downstream
improvement. Primary evaluation uses the fixed final48 checkpoint. CPU preparation
was materialized in `planner-sft-inputs-001`; no optimizer had run at this update.

The falsifiable question is whether a small planner learns useful decomposition
from human-authored training questions, improving held-out downstream answer
accuracy under the same helper and final-answer contract. Its motivation is
diagnostic, not a claim that reference plans reliably improve accuracy. Preparation
performs no optimizer steps; actual receipts must establish that training happened.

## Frozen observation and target

`prepare_plan_training.py:planner_prompt(case)` exposes only the original question
and every document's public `id`/`title` index. It never exposes document bodies,
gold answers, source IDs, support flags, support IDs, component IDs, or hop count.
Its JSON response contains only `subquestions`. `parse_plan(text)` requires exactly
that object, with 1–8 nonempty strings and no duplicate keys, extra fields, fenced
text, or repair. The schema supports variable plan length; training references
have two or three questions. List position numbers the steps. The helper must
interpret `#1` as its own inferred answer to step 1, never an annotated answer.

The target is the ordered list of
`metadata.question_decomposition[*].question` strings from the 256 train parents.
This is explicitly annotation-privileged supervision, not a deployable oracle.
Annotated step answers and final gold answers are not exported. Human questions
can themselves carry privileged decomposition information; absence of a separate
answer field does not make these targets ordinary model-generated supervision.

`examples.jsonl` has exactly `id`, `split`, `prompt`, `target`: `id` and `split`
are host fields; both `prompt` and `target` are strings. `target` is serialized
JSON. Training order sorts opaque parent IDs then shuffles with
`random.Random(20260921)`. Validation 64 remains reserved for later selection;
transfer outputs remain unopened until training choices are fixed. No validation
or transfer targets enter the exported examples.

## Architecture and controlled comparison

The new planner has no provisional answer. Compare released-base and trained
planners using the identical `planner_prompt`/`parse_plan` contract, sampling,
helper, final prompt, full-source evidence, and token budgets. Freeze helper and
final-answer models to released Qwen3-4B-Instruct-2507. Both evaluation arms use
their own predicted questions, never reference questions. Preserve failures.

This changes the earlier architecture, whose checkpoint read full documents and
produced a provisional answer plus exactly two subquestions. Comparisons against
that old checkpoint mix architecture and training effects; report them separately.
The oracle-plan diagnostic also differs: it edits questions inside the old shared
checkpoint. It motivates this test but cannot prove this new planner will improve.

## Proposed bounded optimizer recipe

Reuse the proven TREC leaf SFT mechanics, not its trained adapter weights:
`/project/alex_phd/runs/rlm-research-r4/sidecars/trec-leaf-sft-v1/source/experiment.py`.
Load the pinned Qwen3-4B-Instruct-2507 base with BF16/SDPA and a freshly initialized
rank 8, alpha 16, dropout 0 FP32 LoRA targeting q/k/v/o/up/down/gate projections.
Use AdamW LR 1e-4, weight decay 0, gradient clip 1, microbatch 1, 16 examples per update,
three epochs: 48 real updates on 256 examples. Normalize supervised CE by target
tokens in the accumulated batch. Mask all prompt tokens. Checkpoint every 8 updates
and epoch boundary, including adapter, optimizer, RNG, cursor and immutable input
identity. Keep optimizer state across updates and resume only committed cursors.

Native tokenization uses one user message, thinking disabled, and generation
prompt enabled. Append separately encoded target JSON plus exactly one native EOS.
Audit the proposed 512-token prompt and 256-token target budgets and 2048 total cap
on actual CPU tokenizer outputs. Never truncate, silently drop long examples, or
infer a successful cap from character counts. Budget overruns must be reported;
the total context cap and generation cap are different constraints.

Actual CPU audit on 2026-09-21 used the existing training environment and frozen
`inputs-001/cases.jsonl` (SHA256
`0aebb983cf5c91b38f8bbee93e6fbf1ebe86588fd59048f54c140ae17bf95ef8`).
All 256 training examples fit: 0 exceed 512 prompt tokens, 0 exceed 256 target tokens,
and 0 exceed 2048 total tokens. Maximum prompt 456, target 47 including EOS (token
ID 151645), total 488. These maxima may come from different examples. This verifies
tokenizer feasibility, not optimizer correctness or GPU
performance. Preparation used `--audit-only`; no external training artifact or
optimizer run was created during this check.

The existing training Python is
`/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python`.
Base/tokenizer assets are
`/project/alex_phd/research-cache/models/Qwen--Qwen3-4B-Instruct-2507--cdbee75f17c01a7cc42f958dc650907174af0554`.
No new install is required for CPU tokenization. Model weights and training
artifacts stay outside Git.

## Evaluation and admission

If launched, report optimization curves and checkpoint16/32/48 held-out downstream
EM/F1, plan validity/length, helper/final failures and all deployed token/call costs.
Use validation for the declared selection rule only; do not train on validation
outcomes. Keep test/transfer results unopened until selecting the checkpoint.
Prefer a fixed final 48 primary comparison with earlier checkpoints as learning
curves to limit selection optimism. Report parent/component dependence and the
small validation sample. A negative oracle-plan diagnostic or a non-improving
learning curve can motivate revising the question rather than extending dose.

CPU preparation command (new external output directory only):

```text
TRAIN_PYTHON prepare_plan_training.py --cases CASES_JSONL --output NEW_EXTERNAL_DIRECTORY
```

`--audit-only` runs the actual tokenizer audit without writing artifacts. The
immutable manifest records source/cases/tokenizer hashes, deterministic order,
supervision scope, all per-row lengths and overflow counts. Main alone launches
GPU training after the diagnostic decision.
