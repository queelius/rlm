---
date: 2026-09-28
status: completed_CPU_witness_study
scope: frozen_TRAIN_diagnostic_B_synthetic_counterfactuals_only
model_calls: 0
native_continuations: 48
training_on_B: false
results_sha256: b6186cc6a0f323ada826297105858fbcd2fdcc7631c15ec166e47d9bcc8f0eb9
provenance_sha256: 0f8e3f77310b9c2b26bd91e3228498900cffc42f4c3c43574a058d0100016ed6
analysis_sha256: e558afcc3c5e683166ca2f0fe41110ee523f0a2355e84c042c79159c558f6ffa
---

# Conflicting oracle queries were harmless for success in this witness study

All eight frozen diagnostic-B goals produced two completions with **identical
complete public prompts and input token IDs after the same root query**. Oracle
next labels differed in all eight pairs, yet all **48/48** bounded continuations
succeeded natively: either oracle suggestion, or the common public suggestion,
followed by the unchanged public-only planner. Every forced suggestion was a
valid `get_info` call. There were no failed constructions, native errors or caps;
no goal was replaced.

Decision: deprioritize these query-label conflicts as an explanation for native
failure. They constrain exact imitation, not the ability to choose a successful
action. This does not explain the original teacher/SFT performance gap.

## Cost and label results

Each arm is evaluated on both completions of all eight goals; costs include the
shared root-query prefix in each episode.

| Forced next suggestion | Success | Calls | Queries | Emitted tokens | Extra queries vs public |
|---|---:|---:|---:|---:|---:|
| Oracle for world42 | 16/16 | 439 | 214 | 11,128 | 5 |
| Oracle for hybrid | 16/16 | 442 | 217 | 11,164 | 8 |
| Public-only planner | 16/16 | 434 | 209 | 11,043 | 0 |

Thirteen of the 32 oracle-forced episodes cost one extra query; the other 19
match the public control's call count. None costs more than one extra query.
Maximum episode use is 54/96 calls, 1,430/8,192 output tokens and 5,266/8,192
prompt-plus-generation-cap tokens; per-call cap remains 256 including native EOS.
Exactly 48 continuations ran, with 16 separately constructed native prefix calls
and 1,267 new native calls after those prefixes. The charged episode total is
1,315 calls and 33,335 output tokens. Scientific execution took 7.72 CPU seconds.

On the deliberately balanced two-completion empirical distribution, literal and
semantic oracle labels have `H_emp(A|X)=1 bit` and maximum expected exact-label
accuracy 50%. Public labels have 0 bits/100%. These are full-action label
statistics over eight exact-input groups, not token losses or task-success bounds.

**Important limitation:** 0/16 oracle next-query item names were present in the
shared public prompt. Thus this run does not exhibit conflicting choices among
already-visible candidate names. It tests a hidden-name query schedule, not a
harmful craft decision or a complete acceptable-action set.

## Construction and exact goals

The pair is native world42 versus a synthetic completion retaining world42's
root recipe and public metadata while substituting same-name, same-tier recipes
strictly below the root tier from world43. Both worlds are regenerated from the
cached pinned generator, 25 items/domain/tier. Common initial stock is the
componentwise maximum of base amounts consumed by the two deterministic official
plans; no root/intermediate stock is added. These are **not untouched official
task instances**. Same-tier and higher recipes, base membership, item identities
and native depth metadata remain world42's. Full recipe snapshots and per-pair
deltas are saved. Native root replies themselves are equal, not redacted.

All IDs have prefix `textcraft_synth.train.`. Diagnostic-B remains diagnostic only.

| ID suffix | Exact root goal | Root tier | Public calls world42 / hybrid |
|---|---|---:|---:|
| 1796 | 3 × `o2_i4_12` | 4 | 15 / 9 |
| 256 | 3 × `a3_i3_23` | 3 | 11 / 11 |
| 672 | 2 × `c6_i4` | 4 | 23 / 27 |
| 1273 | 2 × `o1_i4` | 4 | 19 / 13 |
| 1847 | 1 × `c0_i5` | 5 | 17 / 23 |
| 38 | 1 × `c1_i5_21` | 5 | 35 / 41 |
| 201 | 2 × `m8_i5_18` | 5 | 53 / 53 |
| 964 | 3 × `a1_i5_11` | 5 | 43 / 41 |

Concrete example: after the identical root query for `o2_i4_12`, the world42
oracle asks about `o1_i1_11`, while the hybrid oracle asks about `o9_i1_19`.
The common public planner queries the visible `o1_i2_11`. In world42 the two
oracle/public continuations take 15/16/15 calls; in the hybrid they take 10/9/9.
All six succeed. Choosing the other world's initial query wastes at most one
query here, not the task.

The original official solver's stable increasing-tier order and recursive
insertion-order ties are unchanged. World names, task IDs, hidden recipes and
sampler seeds are host-only; the continuation receives only targets,
initial/current stock and prior public recipe replies. There is no sampler or
model. The one forced oracle suggestion is the explicit intervention, not a
claim that the whole episode is public-derived. No hidden plan is supplied after it.

## Evidence and reproduction

The small [result summary](RESULT-SUMMARY.json) is a byte-identical copy of the
external `ANALYSIS.json`, included here so aggregate evidence remains available
without cluster access. Full native traces remain in the external store.

Artifacts are in
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/analysis-teacher-counterfactual-20260928-001/`:
`PROVENANCE.json`, `RESULTS.json`, `ANALYSIS.json`, eight `pairNN.json` records,
world42/43 recipe snapshots, eight recipe deltas and 48 complete saved traces.
Pair records include exact prompts/IDs, native replies, inventories, canonical
world digests, privileged schedules, query-name visibility and next labels.
Source revision is Platoon `d9c5857d3a0a056ebc9b047241a2a0c9515aafbe` (MIT).
Tokenizer is cached Qwen3-4B-Instruct-2507 at
`cdbee75f17c01a7cc42f958dc650907174af0554`; no weights were loaded.

```bash
PY=/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python
E=/project/alex_phd/repos/rlm/.worktrees/selective-delegation-20260921/experiments/selective_delegation/teacher_counterfactual_20260928
R=/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/analysis-teacher-counterfactual-20260928-001
# Inspect existing evidence; no additional native continuation.
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PY" "$E/analyze.py" --output "$R"
"$PY" -m pytest -q "$E/test_study.py"
# Original command, already completed; immutable existing output is rejected.
# CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PY" "$E/study.py" --output "$R"
```

Three focused tests passed after failing on the missing implementation. The
saved-trace audit checks all 1,315 rows: exact branch-start prompt/IDs, public-only
continuation decisions, observed-recipe quantities, stock arithmetic, shared
budget accounting and final native receipts. It makes no additional native
continuations. No GPU, network, training, frozen-source changes or Git operations.

Two artificial completions, generous common stock, correlated goals and a capable
deterministic continuation limit the conclusion. Success provides a witness;
failure would not establish unsalvageability. No novelty, optimality, general task
unsolvability, visible-name ambiguity or learned-policy efficacy is established.
