# Fresh official TRAIN curriculum — 2026-09-28

Two root-disjoint eight-task groups are frozen under
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/textcraft-fresh-train-20260928-001`.
Use `train/` for optimization and `diagnostic/` for a held-away **TRAIN-only** readout.
Each directory contains original-order `tasks.jsonl`, its own `MANIFEST.json`, and
public/native qualification traces. No GPU, model generation, installation, or
changes to active collectors were involved.

## Selection and boundary

- Official `ApGa/platoon` commit `d9c5857d3a0a056ebc9b047241a2a0c9515aafbe`, MIT;
  original world42. Every selected JSONL line is copied byte-for-byte, not
  reserialized; dictionary order is part of prompt identity.
- Seed `2026092803`; each group has actual dependency-depth quotas `{3: 2, 4: 3,
  5: 3}`, restricted to declared source tiers3–5. Within each stratum choose
  ascending SHA256(`seed:official_id`), group train before diagnostic, skipping
  used roots. Quotas and rows were frozen before native feasibility; no replacements.
- Exclude SFT32/readiness8 and every root/ID from48 existing task inventories,
  including reserved breadth06/07 and reserve worlds47–49. Also exclude all632
  official VAL rows'256 roots; this stronger boundary removed no extra TRAIN rows.
  The293 existing top-level TextCraft plans' selected IDs are all covered.
  Exact paths, IDs, roots and file hashes are in `EXCLUSIONS.json`.
- After exclusion:2,432 TRAIN rows. Among declared tiers3–5, actual depths3/4/5
  have279/252/240 rows and93/84/80 distinct roots;36 other rows have actual depth2.
  Structural counts use native public recipe replies, not gold or model outcomes.
  These diagnostic graph data are not supplied to the actor.

## Qualification and limitations

All16 public-only solver traces finish with native score1; no construction failure
or source-evaluation row was filtered into either group. Train native action
counts are `[11,9,25,27,31,33,57,43]`; diagnostic counts are
`[15,11,23,19,17,35,53,43]`. Cached Qwen3-4B tokenizer checks give maximum initial
prompt plus256 output tokens of821 and859 respectively, below8192.

This is native solvability and **initial** prompt qualification, not a guarantee
that a model or full-history teacher fits96 calls,8192 emitted tokens, or8192
context. Same-world ingredient recipes may overlap old tasks even though selected
goal roots do not. The diagnostic group is not official VAL/HOLDOUT and is not a
confirmatory generalization benchmark. Preserved gold exists only in the original
task schema; bridge actor prompts and the qualifying solver do not consume it.

`ATTEMPT-001.json` preserves an initial CPU failure: the shell's default Python
lacked `transformers`. All frozen inputs survived unchanged. Qualification resumed
with `--qualify-frozen` using the existing training environment; no re-selection
or environment installation. `VERIFICATION.json` records the final saved-trace
replay and loader check.

## Adapter contract / pins

| File | SHA256 |
| --- | --- |
| Root `MANIFEST.json` | `76e387378c9a89b0d8e26180874db61ad40086721c6cfef9e3b78dd77f915e05` |
| `train/MANIFEST.json` | `07c410047a740b429912df9dbb0fb9c8cea3063b1e185c51f87ca645c5ab9d90` |
| `train/tasks.jsonl` | `22e65d9017711dd117ef7884b5a7e51c6152fa791c47666382dda05b7407bf76` |
| `diagnostic/MANIFEST.json` | `65e67e57063ee43d1c8c072988cca2081531b96aae4a47d582c2550f419c1c1a` |
| `diagnostic/tasks.jsonl` | `12b0381fa3bcc17aee92e8677a713f8db3ba613cad7faa8229cc63411ed6d0ff` |

Group manifests expose `group`, `split`, `task_ids`, `tasks_sha256`, `world_seed`,
`world_sha256`, `ready`, `all_native_feasible`, `initial_context_qualified`, per-task
`audits`, and true selection/exclusion path+hash references. The lightweight
`prepare.load_group(root, group, expected_root_manifest_sha256)` preserves order,
rejects non-original source rows and overlaps, and rejects missing qualification;
call once in CPU preflight, outside GPU ownership.

Focused checks from this worktree:

```bash
python -m unittest discover -s experiments/selective_delegation/fresh_train_20260928 -p 'test_*.py'
python experiments/selective_delegation/fresh_train_20260928/check.py
```

The test-driven selection fixture covers source-VAL rejection, alias roots,
order-independent deterministic selection, and refusing unavailable root quotas.
The separate CPU verification fixture checks actual frozen files, public actions,
native feedback, finish/score, prompt identity, and tamper rejection. Broad tests
were deliberately unnecessary for this additive data-only preparation.
