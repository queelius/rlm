---
date: 2026-09-28
status: completed_native_audited_first_update_matrix
scope: familiar_TRAIN8_two_new_sampling_seeds_one_RL_update_per_interface
outcomes_sha256: 74f15cc95a51b3a8e4c180229812a6a4e55fe6cfac5bc3cf995496cec4344470
compact_snapshot_sha256: 705c37f5c5ffc6c9a115255afa93862f9eb78d2fbe57cf7b5692974ffe6316b9
GPU_or_new_native_calls: 0
---

# A real first-update success signal, with uneven behavior and higher cost

Strongest defensible claim: **one actual terminal-reward update improved observed
success with the execution interface held fixed on these familiar goals**. Raw-RL
went 9→13/16 with ordinary tools; binder-RL went 14→16/16 with assistance. This is
more than an execution-only improvement, but not yet robust RL efficacy, held-out
transfer, superiority over extra SFT, or evidence that assistance improves learning.

| Weights / execution | Success | Calls | Output tokens | Service seconds | Native / schema errors |
|---|---:|---:|---:|---:|---:|
| Warm / raw | 9/16 | 326 | 9,610 | 737.5 | 112 / 0 |
| Raw-RL / raw | 13/16 | 340 | 10,224 | 792.1 | 127 / 0 |
| Binder-RL / raw | 12/16 | 372 | 11,508 | 893.5 | 137 / 0 |
| Warm / binder | 14/16 | 277 | 8,462 | 650.6 | 41 / 17 |
| Raw-RL / binder | 15/16 | 319 | 10,613 | 828.9 | 49 / 41 |
| Binder-RL / binder | 16/16 | 303 | 10,325 | 806.7 | 30 / 53 |

No transport failures. Raw-RL/raw gains four paired slots and loses none across
three goals; binder-RL/binder gains two slots and loses none across two goals.
Existing eight-task-cluster bootstrap intervals are +6.25…50 percentage points
and 0…31.25 points respectively. These intervals condition on this adaptive panel
and one realized update; they do not include training-seed uncertainty. There are
eight goals, not 16 independent tasks. Raw/binder updates used 19,787/8,440 credited
tokens from 790/556 collection calls: equal one-step optimization is not equal
data, token dose or compute.

## What changed in the actions?

All IDs below have prefix `textcraft_synth.train.`. Counts are warm→trained.

- Raw-RL/raw rescues **1680 r1** (24→18 calls), **465 r0** (17→37),
  **465 r1** (22→25), and **860 r1** (40→30). It does not rescue 429 or404 r0.
  This mixes useful recipe correction with longer recovery, not a uniform planner gain.
- **465 r0:** both actors initially add an extraneous base ingredient. The updated
  actor makes *more* native mistakes (5→15), but eventually queries `t9_i1_19`,
  supplies only `raw_t3`, builds the missing intermediary, and completes all three
  roots. Warm finishes without any root. Success is bought with 20 additional calls.
- **465 r1:** warm finishes at 2/3 roots. Raw-RL eventually crafts more `t0_i2`
  and finishes at3; the first13 responses are identical. Ingredient and stock errors
  remain. **860 r1** instead recovers an ingredient-map error: it eventually removes
  extraneous `raw_a0` from `a8_i2_18`, enabling the missing chain, and saves10 calls.
- **Binder-RL/binder,1680 r1:** exactly15 calls for both. At zero-based call12,
  after an identical complete input prefix, the updated actor requests6 rather
  than3 `c6_i2`, then produces2 rather than1 root. This is a concrete quantity/goal
  completion change, not simply refusing to stop or adding search calls.
- **Binder-RL/binder,429 r0:** 38→80 calls. It eventually crafts the missing
  `c0_i1_10` and `c3_i1_13`, then the root, despite a much longer invalid-response
  loop. The warm actor instead stops with no root.
- Cross-interface binder-RL/raw has four wins and one loss versus warm/raw:
  wins1680 r1,404 r0,465 r0/r1, but **loses404 r1** by finishing at2/3 roots.
  Its429 r1 reaches context cap after78 calls/58 native errors. No outcome is omitted.

Premature finishes fall raw7→3 and binder2→0. That is descriptive, not an isolated
stopping mechanism: all four raw wins change earlier actions at exact matched
input IDs. Under raw-RL/raw, ingredient-map errors fall only67→62, stock errors
rise45→64, and one nondivisible-count error appears. Query repeats fall23→15, while
successful craft calls rise92→100. Later trajectories visit different states; their
error or entropy rates are not controlled same-state policy comparisons.

There is also inefficient persistence *after* success. Native audits count20→45
calls beginning with sufficient root stock under warm/raw→raw-RL/raw (including
required finish). TRAIN404 r1 alone goes12→33 such calls. Terminal success does
not directly penalize this. These post-goal actions remain fully charged.

## Why only binder has the schema-error spike

**Every one of the 111 schema errors is the same response on TRAIN429 r0**, across
the three binder cells. It contains duplicate `c9_ore` keys:

```json
{"action":"craft","ingredients":{"c9_ore":1,"c9_ore":1,"c3_i1_13":2,"c8_i2":2,"c0_i1_10":1},"target_item":"c9_i3_19","output_count":1}
```

These are completed EOS responses,65 output tokens each, not transport errors or
truncation. The strict parser correctly rejects them **before** recipe binding,
leaves inventory unchanged, and appends explicit duplicate-field feedback. The
model repeats the response anyway. No permissive parsing or deduplication is justified.
The loop already occurs17 times with the warm actor; RL did not originate it.

A saved-call witness shows why raw/binder do not visit the same histories. At
call9, warm raw and warm binder have identical input IDs and emitted text, but
raw rejects extraneous ingredients while binder executes the observed recipe.
Inventory and public history then diverge. That is evidence of a changed visited
trajectory, not proof of the precise cause of duplicate generation.

Warm versus binder-trained call9 differs only in requested ingredient payload;
the binder executes the same action. At call27 the trained model picks a missing
intermediary instead of another already-made product. History is still identical,
but remaining output budgets differ by14 tokens, so this later choice is **not**
an exact-input causal contrast. `LOOP-WITNESS.json` preserves both distinctions.

The duplicate loop alone consumes1,105/2,665/3,445 output tokens and85/212/278
service seconds in warm/raw-trained/binder-trained binder cells. For binder-RL it
accounts for33.4% of output tokens and34.5% of service time. Total own-interface
cost regresses: raw-RL calls+4.3%, output+6.4%, time+7.4%; binder-RL calls+9.4%,
output+22.0%, time+24.0%. A post-hoc sensitivity check excluding429 r0 has binder
calls239→223 and tokens6,783→6,061; the main result must retain that costly episode.

## Independent compact-interface pilot: efficiency, not a clear success gain

A one-time snapshot found both world42 and50 readouts completed and native-audited.
Using actual task/repeat/seed pairs, compact versus full-format+binder is8 vs10/16
on world42 (2 wins/4 losses), and10 vs9/16 on world50 (2 wins/1 loss). Output tokens
fall39.4%/39.3%, service time37.5%/36.2%, but calls rise4.1%/2.6%. Compact has48/104
explicit action rejections and58/116 native errors; these are not silently repaired.

This is a separate SFT/interface comparison on the same eight VAL roots across
two worlds, not32 independent goals and not RL. Both use23 SFT updates, but compact
has27.5% fewer supervised output tokens; history/schema and unobserved-recipe
handling also differ. It supports generation-cost savings, not preserved or improved
success. A later strict-full-format comparator can separate that contract difference.

## Ranked next decisions

1. **Fresh diagnostic-B fixed-endpoint transfer** (another agent is preparing it).
   Keep weights fixed and raw/binder tools crossed; zero-shot performance outside
   TRAIN8 is the strongest immediate check against familiar-goal overinterpretation.
2. **Repeat the learning contrast with an independent collection/update seed**, then
   compare with the accepted extra-SFT control. One updated model and two rollout
   seeds cannot establish reproducibility or a uniquely RL advantage.
3. **Read the accepted demand-table and compact-RL probes before adding machinery.**
   Quantity completion and repeated payload generation are concrete remaining errors.
   Keep native stock/count/schema failures visible; do not count compact's lower
   output length as a learning gain or claim the payload mask saves inference compute.
4. **Only after replication, test a cost-sensitive learning/stopping contrast** with
   success and cost reported separately. The404 post-goal persistence and429 rejection
   loop offer targeted diagnostics. A call penalty can encourage early failure, so
   retain native completion as the primary endpoint and inspect the tradeoff.

## Reproducibility

External output: `R/analysis-rl-outcomes-20260928-001/`, where
`R=/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.
`OUTCOMES.json` contains all96 episode censuses, changed-slot witnesses, error
categories, cost sensitivity, copied uncertainty and input/source pins.
`COMPACT-SNAPSHOT.json` and `LOOP-WITNESS.json` are separate pinned observations.
Every consumed RL call/node/episode is checked against its saved native-audit hash;
no large model ancestry is rehashed and no new native/model call is made.

```bash
PY=/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python
E=/project/alex_phd/repos/rlm/.worktrees/selective-delegation-20260921/experiments/selective_delegation/rl_outcomes_20260928
"$PY" -m pytest -q "$E/test_analyze.py"
# Original completed CPU commands; immutable output paths refuse overwrite:
# CUDA_VISIBLE_DEVICES= "$PY" "$E/analyze.py"
# CUDA_VISIBLE_DEVICES= "$PY" "$E/compact_snapshot.py"
# CUDA_VISIBLE_DEVICES= "$PY" "$E/loop_witness.py"
```

Three focused tests passed after failing on missing behavior; the actual saved-data
census validates all six native totals. Scoped Ruff passes. No frozen/runtime source,
GPU owner, live queue, dataset, environment, Git commit or push was changed.
