# Compact action interface — ready CPU assets, 2026-09-28

Question: can the model learn more effectively or generate more cheaply when it
chooses craft target/output count, while code supplies ingredient arguments from
an already observed public recipe? This is an exploratory interface/SFT comparison,
not an RL result or a new policy-gradient estimator.

Preparation is complete. No GPU or scientific model generation was used here;
parent launches all jobs. Original active sources remain unchanged.

## Exact intervention

Craft output is `{"action":"craft","target_item":"product","output_count":2}`.
An experiment-local bridge requires exactly this schema, with the existing optional
`note`; old ingredient fields are rejected. It binds ingredients only from a single
recipe returned by actual prior `get_info` in that frame. Missing/ambiguous recipes,
bad types and nondivisible quantities are explicit rejections. It never changes
target, output count, inventory, or recipe choice; native code still checks stock.
Get-info, inventory, finish and schema rejection retain their existing contracts.
This first probe is flat-only. Ingredient maps remain visible in recipe observations.

Public action histories retain the compact requested action, not silently expanded
arguments. The collector and saved-receipt auditor use the same explicit bridge
binding; original native methods execute and score both.

## Frozen training dose

Data: external `textcraft-compact-actions-20260928-001/` under the active campaign
store. All32 original public-discovery TRAIN tasks and366 steps survive, with exact
task-file bytes, row order, targets/counts and native feedback. Preparation replays
the original actions to rebuild instructions, compact histories, inventory and
remaining-token counters; it does **not** just edit target strings.

| Quantity | Original discovery | Compact projection |
| --- | ---: | ---: |
| Tasks / rows / optimizer updates | 32 / 366 / 23 | 32 / 366 / 23 |
| Supervised output tokens incl. EOS | 8,820 | 6,396 (−27.48%) |
| Prompt tokens | 438,065 | 435,264 (−0.64%) |
| Native successful teacher tasks | 32 | 32 |

Same Qwen3-4B-Instruct-2507 base revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, seed2026092208, one epoch, LoRA rank8,
alpha16, LR1e-4, effective batch16 (last14), unchanged target-only loss and optimizer.
Fixed checkpoint23; no endpoint selection. Equal epochs/rows/updates are **not**
equal tokens, FLOPs or wall time. Original SFT took207s; estimate3–6min on one A100,
hard cap30min, every update checkpointed.

Data manifest SHA:
`dcee1f13eee43d74b5b72414eb202bf4e0b8dca02b7a09eea0bcb80759ea32c7`.
Prepared training PLAN SHA:
`445a7b5762b4c2c7ea6892083757128148b83a00ce435f1f6d1516200b3e3f56`.
`COMPACT-CONTRACT.json` binds the actual compact validator/wrapper and original
optimizer source; it never substitutes an old dataset manifest.

## Smallest comparison

First existing breadth panel00, worlds42 and50,8 identical goal roots, two rollout
seeds2026092204/05,16 episodes per world. Panel00 was chosen by index, not outcomes.
Depth5 is included. Use the already completed fullformat+observed-binder publiccp23
outputs as the first baseline:

- `textcraft-breadth-p00-w42-soriginal-binder-001`:
  PLAN `53aa1ad38d3ebf41d0ab4a0ff38955129e3142f5c5e94a5c85b525c00b862719`.
- `textcraft-breadth-p00-w50-soriginal-binder-001`:
  PLAN `e2664bf24bf61c11db3c6aa3073047df15f587c4b14f3064881829310889ff7e`.

Compact readouts preserve each world's task bytes/order, job order, seeds, sampling,
96-call/8192-output-token budget,256-token response cap,8192 context and45min cap.
Initial prompt+cap is at most828 tokens in world42 (world50 is independently checked).
Existing binder readouts took19.1 and25.7min: expect roughly20–30min per compact
readout, not a promised speedup. Baseline replication is supported with `--mode binder`.

September28 integration note: the optional `--mode binder --audit` branch currently
passes `expected_task_count` to a sealed older auditor that does not accept it.
Do not use that optional audit command until a separate adapter is provided. The
accepted **compact** runtime/audit and the existing independently audited binder
baselines are unaffected. No live/pinned source was changed for this unused branch.

Primary endpoints: paired native task success, generated tokens, prompt tokens,
native calls and service time. Report errors/caps and unavailable episodes separately.
The compact source change is not only token deletion: full binder falls through to
raw native craft when no recipe was observed, whereas compact rejects it. Existing
world42 binder crafts were291 bound/5 nondivisible; world50 were342 bound/4
nondivisible/15 unobserved. A strict-recipe full-format control is warranted if this
boundary appears mechanistically important. Also, matching sampling seeds does not
match sampled continuations across models/interfaces.

Promote only if useful success/cost gains recur in a second training seed and fresh
fixed roots/changed worlds. Retire as a learning direction if no reproducible
success benefit and only trivial savings; keep an engineering simplification if
meaningful token savings preserve success. One training seed, tiny reused panels,
shared-world recipe exposure and changed token dose preclude a robust learning claim.

## Verified runtime seam

`runtime-fixture-001/VERIFICATION.json` passed worlds42/50. Scripted **CPU** output
tensors pass through the real NativeClient request/save/token/decode machinery,
real native episode and saved-call auditor:12/10 calls,score1 each, with2 schema
errors,2 action rejections and1 insufficient-stock native error each, all charged.
No ingredients enter compact action histories. This is a fixture, not measured
model competence. Five focused strict-interface tests also pass. Data/preparer and
training source hashes were rechecked unchanged after preparation.

## Ready commands

From this worktree, with the parent's assigned allocation/owner environment:

```bash
COMPACT_PY=/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python
COMPACT_RUNS=/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921
COMPACT_SRC=experiments/selective_delegation/compact_actions_20260928

# --resume is required because CPU prepare-only already froze PLAN; no owner was launched.
$COMPACT_PY $COMPACT_SRC/train.py --output "$COMPACT_RUNS/textcraft-compact-sft-20260928-001" --hours 0.5 --resume

# Run after authenticated complete checkpoint0023. Parent queues these separately.
$COMPACT_PY $COMPACT_SRC/evaluate.py --world 42 --output "$COMPACT_RUNS/textcraft-compact-p00-w42-20260928-001" --hours 0.75
$COMPACT_PY $COMPACT_SRC/evaluate.py --world 50 --output "$COMPACT_RUNS/textcraft-compact-p00-w50-20260928-001" --hours 0.75

# CPU audit each terminated owner; exact receipts, no reward substitution.
$COMPACT_PY $COMPACT_SRC/evaluate.py --output "$COMPACT_RUNS/textcraft-compact-p00-w42-20260928-001" --audit
$COMPACT_PY $COMPACT_SRC/evaluate.py --output "$COMPACT_RUNS/textcraft-compact-p00-w50-20260928-001" --audit
```

Before the checkpoint exists, `evaluate.py --world 42 --output <path>
--validate-inputs-only` checks actual panel/world/tokenizer inputs without fabricating
an adapter binding or writing a pending PLAN. `--prepare-only` requires the real
finished checkpoint and freezes the actual launch plan without loading the model.
Never rerun a scientific evaluation output directory: use a new numbered owner.

Source provenance: original TextCraft `ApGa/platoon` commit
`d9c5857d3a0a056ebc9b047241a2a0c9515aafbe`, MIT, native method/generator hashes and
world digests retained. No new acquisition or environment modification.
