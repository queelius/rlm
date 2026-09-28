# Public-demand table pilot

Status: CPU-qualified and prepared, no GPU launch by this agent. The retrospective
[finding](FINDING.md) is complete; its analyzed source/report bytes are frozen separately.

## Scientific contrast

Both arms use the original discovery-trained checkpoint23, native observed-recipe binder,
full public history, public fact table, identical instructions and alphabetically ordered rows.
`demand` displays remaining root-goal demand, stock deficit, batch count and explicit frontier
deficit type. `masked` replaces those computed columns with nulls. At every state, the shorter
serialized prompt is padded with semantically empty `x` tokens until both complete encoded
input lengths match exactly. Both actual inputs are charged normally.

The calculation accepts only the already serialized public prompt. It has no native-world
argument or access to hidden recipes. Unknown recipes stop propagation. Only an observed
`is_base=true` response can label a deficit `known_base`; an unqueried item from initial stock
is labeled `unqueried_initial_stock`, not assumed to be an oracle-known raw material.
No selected next action, ready-action list, priority order, automatic query/craft or additional
ingredient repair is exposed. Shared-child demands are aggregated before recipe batch rounding.

This is host computational assistance, **not** a pure memory or independent reasoning claim.
The treatment intentionally performs a public-graph calculation. It does not prove that a full
native recipe graph is known. Exact per-state prompt length matching does not match cumulative
trajectory cost once actions diverge; padding is itself a presentation caveat.

## Frozen population, actor and dose

- Panel00/world42, all four declared depth4/5 identities, selected before the retrospective
  primary-slice outcomes: `.628`, `.599`, `.401`, `.294`.
- Two fresh paired repeat seeds: `2026092804`, `2026092805`; eight attempts per arm.
  This is adaptive reuse of exposed VAL tasks, not held-out evaluation.
- Actor: `R/textcraft-public-discovery-sft-001/checkpoint-0023`, trained with seed2026092208.
  Adapter SHA256 `f029d36967eb1cef30e102fe82e52f22f90d5eacdaf91e82937b60218ba5b00f`.
- Native runtime: `R/source-textcraft-recipe-binder-003`; same BF16 base, frozen adapter,
  temperature0.5, 96 calls, 8,192 cumulative generated tokens, 256 tokens/call,
  8,192 input-plus-output context, no truncation. No training/checkpoints are created.
- The historical reference is `R/textcraft-breadth-p00-w42-soriginal-binder-001`.
  It has different repeat seeds and no new table, so it is not the paired control.
- Fixed cap: 2,700 seconds/arm, including model load; allocation margin can shorten it.
  Historical eight-attempt slice took16.5 minutes. Estimate35–55 minutes for both new arms,
  hard maximum90 minutes. If additional context cost causes caps, that is an outcome.

`R` is `/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.
Prepared plans are under `R/textcraft-public-demand-20260928-001`:

| Arm | Relative PLAN | SHA256 |
|---|---|---|
| demand | `demand/PLAN.json` | `dc01db54dacf5eee2f58331ee00dcea2cf1d2d4276e4c82493e201215bfb58c6` |
| masked | `masked/PLAN.json` | `d28fde32106b7f37ca264a7a677f33ff3cfaa5e8f3f1165757a05a399e25935c` |

Plans contain exact source hashes, Python/package versions, actor lineage, task/world hashes,
sampling and budget rules. Freeze `readout.py`, `demand_table.py`, and `analyze.py`.
Do not rewrite accepted PLANs or retry into an existing scientific attempt.

## Parent launch commands

From the active worktree, using its already installed one-A100 training environment:

```bash
GPU_PY=/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python
TABLE_ENTRY=experiments/selective_delegation/inventory_bottleneck_20260928/readout.py

# Parent's serialized GPU owner only. Both PLANs already exist; no --resume flag is needed.
$GPU_PY "$TABLE_ENTRY" --mode demand
$GPU_PY "$TABLE_ENTRY" --mode demand --audit
$GPU_PY "$TABLE_ENTRY" --mode masked
$GPU_PY "$TABLE_ENTRY" --mode masked --audit

# CPU paired readout after both terminal audits.
$GPU_PY experiments/selective_delegation/inventory_bottleneck_20260928/compare.py
```

Each collection uses the existing native coordinator lock and emits OWNER/TERMINAL/LOAD,
starts, calls, nodes, episodes and SUMMARY. The wrapper corrects the old collector's hardcoded
16-slot summary to the actual eight planned slots. `--audit` replays requested/executed binder
transitions, requests/tokens/decodes, full histories, stock, scoring and endings. It records
all eight planned slots even if some were not reached; missing/interrupted outcomes stay unknown.
The primary endpoint is paired native completion. Secondary readouts are repeated stock errors,
root-ready failed finishes, calls, actual input/output tokens, context caps and elapsed time.

## CPU qualification

Eleven focused tests pass (five unchanged public-math, four interface, two paired-readout tests); Ruff
passes on new Python files. Prepared initial input-token counts match in both arms:
`[1360, 1084, 1170, 1369]` for the four selected roots.

`R/textcraft-public-demand-20260928-001/runtime-fixture-002/VERIFICATION.json` is the current
saved-request/native replay fixture. In both arms, 34 real saved responses from `.599` repeat1
pass through actual NativeClient request construction, tokenizer, decode, starts/calls and
independent native replay. Each reproduces 10 queries,23 crafts,one finish,11 native action
errors,1,045 output tokens, identical final inventory/history and the original failed score.
All34 same-state prompt lengths match (first1,084; final4,784 tokens). These are replayed model
responses on CPU, **not** new model evaluation evidence. Fixture001 also passed but preceded a
fixture-only lint correction; fixture002 is the final-source receipt.

Re-run the fixture only into a new external output directory:

```bash
$GPU_PY experiments/selective_delegation/inventory_bottleneck_20260928/fixture.py \
  --output /absolute/new/runtime-fixture
python -m pytest -q experiments/selective_delegation/inventory_bottleneck_20260928/test_public.py \
  experiments/selective_delegation/inventory_bottleneck_20260928/test_table.py \
  experiments/selective_delegation/inventory_bottleneck_20260928/test_compare.py
```

Do not expand just because loops decline without better completion. A positive pilot requires
an unread task-panel and second-actor replication before generalization. Four shared-world task
identities are a mechanism screen, not confirmatory evidence.
