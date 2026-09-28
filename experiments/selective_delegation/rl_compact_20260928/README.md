# One compact-interface RL cycle

CPU-prepared on September 28, 2026; parent owns dispatch. Nothing here launches on import or
preparation. The four scientific stages are collection, one update, warm diagnostic, updated
diagnostic. No sweep, continuation, diagnostic-based checkpoint selection, or extra rollout reuse.

Question: does native-success RL improve the compact observed-only actor beyond its own warm
checkpoint, and how does that gain differ from the first fresh full-format+binder update?

## Fixed comparison

- Starting actor: the actual committed compact matched-SFT checkpoint 23. Admission reuses the
  existing compact evaluator's completed-owner/commit/23-step/row checks. Future hashes are unknown
  until that succeeds. `PREPARED-JOBS.json` is only a pending-weight preparation contract;
  scientific `PLAN.json` files and `WARM-ENDPOINT.json` bind real bytes at runtime.
- Official TRAIN groupA: the frozen fresh eight goals, four execution seeds
  `202609280110`–`202609280113`, exact task/order pairing with fresh-v002 update 1.
- Official TRAIN groupB: eight disjoint diagnostic goals, seeds `202609280900/901`;
  warm and updated readouts are fixed and never optimization data. No VAL/HOLDOUT optimization.
- `execution_mode=compact_observed`: compact craft action/target/count, strict observed-only
  recipe expansion, native inventory/terminal score, and compact public action histories.
  Flat interface only; expanded ingredient arguments never replace sampled likelihood targets.
- Native binary terminal reward, signed leave-one-out advantage against the other three
  same-task rewards, full original sampled-token sum including invalid actions and EOS, `/32`
  trajectory normalization. One fresh AdamW update, LR `2e-5`, weight decay 0, gradient clip 1.
  Behavior and replay both use temperature 0.5, top-p 1, top-k 0; original generation/replay
  probability checks and numerical guards are unchanged. This is one on-policy update.
- Limits: 96 physical model calls and 8192 generated tokens per trajectory; 256 tokens per
  response; 8192 input+output context, no truncation. Base FP16, LoRA FP32.

Incomplete collections are not trained. No within-task native reward variation skips the update.
Numerical failure prevents endpoint admission even if a boundary was written. Warm groupB is
independent of those outcomes; the updated readout skips honestly when no usable endpoint exists.

## Parent dispatch

External root: `textcraft-compact-rl-20260928-001` under the active selective-delegation store.
`PREPARED-JOBS.json` contains absolute commands, all source/data pins and the shared coordinator
lock. It is not an `ACCEPTED` receipt and does not race existing owners. Parent should dispatch all
four fixed stages in order, even when earlier stages report conditional skips:

```bash
GPU_PY=/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python
SOURCE=/project/alex_phd/repos/rlm/.worktrees/selective-delegation-20260921/experiments/selective_delegation/rl_compact_20260928
"$GPU_PY" "$SOURCE/compact_stage.py" --kind collect
"$GPU_PY" "$SOURCE/compact_stage.py" --kind train
"$GPU_PY" "$SOURCE/compact_stage.py" --kind warm
"$GPU_PY" "$SOURCE/compact_stage.py" --kind updated
```

Scientific caps are 150/120/90/90 minutes; supervisor caps add five minutes per stage. CPU
comparison is capped at ten minutes. The full descriptor maximum is 8 hours; useful work is
estimated at 2–4 hours, not measured compact throughput. The native collector retains saved
scientific responses, generation log-probabilities/entropy, owners, caps and independent replay.

Preparation uses `compact_rl_fixture.py --output <new-directory>` followed by
`prepare_compact_rl.py`; neither loads a pretrained model or uses a GPU. The pinned external
fixture is `runtime-fixture-001/VERIFICATION.json`. Existing sealed code is reused through private
module dependencies, never edited.

## Readout and interpretation

`compact_compare.py --output <new-report>` reports warm/post success, physical calls, errors,
unsuccessful explicit/early finishes, sampled-token entropy and response diversity. It computes
own-interface paired gains and compact-minus-full-binder learning gain using the same 16 diagnostic
slots; missing or skipped cells remain unknown. Training records include mixed groups, credited
token counts, gradient size and pre/post selected-token probabilities.

Matched SFT means 23 updates, not equal token exposure: compact has 6396 supervised tokens versus
8820 full-format tokens. Strict unobserved-recipe rejection also differs from full binder fallback.
Warm actors, instructions, action histories and trajectory/token doses differ. A gain interaction
is therefore a co-adapted-interface result, not an isolated schema effect. Raw cross-interface
scores are not RL benefits. Eight exposed TRAIN clusters in one world are exploratory; no unseen
generalization or RLM-recursion claim follows.

Focused CPU verification covers fixed pairing, pending endpoint rejection, honest skip propagation,
diagnostic exclusion, missing-cell arithmetic, and a saved compact native replay. The latter uses
scripted outputs, not model competence: invalid old-format JSON, rejected unobserved craft,
native insufficient-stock craft, then successful observed-recipe execution. It verifies original
token targets/log-probabilities and the actual signed trainer loss: analytical rewards `[1,1,0,0]`
give advantages `±2/3` and per-token derivatives `∓1/48`; native receipt labels are untouched.
