# Frozen A-only recovery/ordinary teaching data — 29 September 2026

**Data prepared and CPU-qualified. No training adapter, GPU launch or queue is prepared.** This implements exactly the selection/matching rule approved in [EXPERIMENT.md](../fresh_reward_diagnosis_20260929/EXPERIMENT.md); the diagnosis and all accepted sources remain unchanged.

External immutable output: `/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/textcraft-recovery-data-20260929-001`.

| Frozen training input | Recovery | Ordinary control |
|---|---:|---:|
| Rows / official TRAIN A roots | 32 / 8 | 32 / 8 |
| Queries / crafts / finishes | 7 / 17 / 8 | 7 / 17 / 8 |
| Target tokens, including EOS | 823 | 823 |
| Prompt tokens | 80,120 | 65,044 |
| Maximum full prompt + response cap | 6,220 | 5,452 |
| Native-qualified complete teacher trajectories | 8 / 8 | 8 / 8 |

All32 matched pairs have the same target-token **length**; 24/32 have identical target JSON. Eight labels differ. Prompt-token counts, histories, token identities and FLOPs are not matched. The manifest's generic prospective `unmatched`/“equal-token” caveat is clarified by the measured fields and additive `VERIFICATION.json`: target-token counts are in fact equal here. This is not a same-label-multiset comparison.

The recovery arm uses each A task's repeat0 state immediately after its first native error. Selection is fixed and independent of final model success. Each actual saved next actor request was reconstructed byte-for-byte and re-encoded to exactly the saved input-token IDs. Original initial/current inventory, full history and spent call/output budgets were retained. Teacher decisions use only copied public goals/inventories and observed `is_base`/recipes; native metadata is discarded from the teacher recipe map. No gold plans, hidden recipe graph, native scores, task IDs or future replies enter teacher decisions. Previously public native replies stay in the unmodified actor history.

Four distinct recovery rows per root are selected: first suffix action, first unused craft, last unused craft, finish, then chronological order. Ordinary rows come from fully qualified clean public A trajectories and are matched without replacement by task/action type, minimum absolute target-token-length difference, then original step index. No root or failed attempt was substituted. The full qualifying pools (155 recovery +236 ordinary rows) are retained for inspection but are **not** the proposed training sets.

## Artifacts and hashes

- `MANIFEST.json`: `64f53f5c972bb7bd1fdea7887396e7d57c0341841a6d2f0c69c7804915301f34`
- `recovery-rows.jsonl`: `11dcc356d758fc96681e5e04f11d79319baf0c4127ae0f4587879d9975a7d673`
- `ordinary-rows.jsonl`: `10bef3dbeff91e6340f39a27703d4acfee66b6ba20e38a523dcd6a302230c863`
- `PREFIXES.json`, `MATCHING.json`, `NATIVE-AUDIT.json`, per-task complete teacher traces/audits, unchanged official `tasks.jsonl`, and additive `VERIFICATION.json` provide the checks and lineage.

The manifest records all selected row/label IDs, task/source hashes, original collection/publiccp23 identity, official source revision/license, pinned tokenizer assets and template, per-arm token/context counts, source hashes, Python/package versions and limits. No weights were loaded or rehashed. Existing environments were not changed; `uv` was not invoked. Outputs occupy about16.7MB, with no weights or new third-party downloads.

## Verification and use

Five focused tests were written first and observed failing, then passed. A separate read-only artifact check verified all manifest artifact/source hashes, recomputed the 64 selected teacher labels from their public prompt state, checked all eight exact actor prefixes, all eight selection rules and all32 greedy matches without replacement. All16 actual native continuations finished successfully under original96-call/8,192-output-token/8,192-context caps. No broad suite was run.

Formatting check passed. One cosmetic Ruff `E501` remains in the already pinned preparer (`prepare.py:410`, a102-character provenance string); it is explicitly recorded and was not changed after source freeze.

Executed CPU-only command from the isolated worktree:

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python experiments/selective_delegation/recovery_teaching_prepare_20260929/prepare.py --output /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/textcraft-recovery-data-20260929-001
```

The preparer refuses to overwrite the existing output. The scoped tests can be repeated without writing it:

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /project/alex_phd/envs/rlm/bin/python -m pytest -q -p no:cacheprovider experiments/selective_delegation/recovery_teaching_prepare_20260929/test_prepare.py
```

Rows use the existing encoder convention: `input_ids = prompt_ids + target_ids`, including EOS; `labels = [-100] * prompt_length + target_ids`. A future trainer must apply its normal causal shift, **not** pass these unshifted full sequences directly to an emitted-token replay helper. `feedback` and native audit details are qualification metadata only, never extra model inputs. Only the frozen32 selected rows per arm are proposed training inputs. The one-step fit and paired full-start A readouts remain conditional on first-A/SFT evidence; B was not used and no GPU efficacy is claimed.
