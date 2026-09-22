# Fresh-root teacher comparison: CPU-ready proposal

2026-09-22. Inputs are frozen; **GPU comparison is not accepted or launched**.
Question: does public-discovery supervision retain its advantage over privileged
action-order supervision on previously unevaluated root goals in the same recipe
world? This is not independent-world or unseen-prerequisite generalization.

## Frozen selection and qualification

Official MIT Platoon VAL inventory at commit
`d9c5857d3a0a056ebc9b047241a2a0c9515aafbe`: 632 rows. Select the first
SHA256(`2026092222:task_id`) rows within depth strata 2/3/4, taking 4/4/8
**distinct root goals**. Exclude every official TRAIN root goal, all task/root
identities in existing `R/textcraft*/tasks.jsonl`, and official VAL IDs in
existing TextCraft PLAN jobs. The exact input paths, hashes and exclusion lists
are recorded in SELECTION.json; no outcome or feasibility filtering. This is
an explicit local inventory, not an assertion about every historical experiment.
Twenty-four VAL rows were excluded through exposed roots; zero PLAN IDs were
unresolved. Eligibility remained 69 depth-2, 66 depth-3 and 60 depth-4 rows.

`R/textcraft-fresh-inputs-001/tasks.jsonl` was written before any selected-task
replay. All 16 native quantity-correct gold replays succeeded; all 16 existing
public-observation teacher traces also succeeded. No replacements. Gold traces
use at most 35 calls, 54 output tokens per action including EOS allowance, and
3,413 input tokens (plus 256 response cap); largest initial flat prompt is 577.
This establishes a constructive feasible path, not a minimum path or model
success. Policy loops may still consume the full budget/context.

World42 recipe hash is unchanged:
`f76ce3978c038be9624b3c7387aa30033970508c6afecb1f8f08315aa0693808`.
Fourteen of 16 new roots have prerequisite recipes shared with the actual
32-task SFT training panel: 49 of 109 distinct reachable recipe products overlap.
Root exclusion therefore does not imply recipe independence. Preserve these
overlaps in analysis; do not remove tasks retrospectively.

## Prospective matched readout and qualified wrapper

Two fixed endpoints only: privileged048 checkpoint23 and public056 checkpoint23.
Original flat prompt only, shared base4B/tokenizer, adapter enabled and frozen;
no reminder, training, repair, profile selection, or recursive arm. Use existing
episode seeds 2026092204/2026092205 and native per-task/call seed formula for
both adapters. Sixteen roots × two seeds × two adapters = 64 episodes; 32 per
arm. Preserve T=.5/top-p=1/top-k=0, 256 per-call cap, global96 calls/8192 output
tokens, input+cap8192, all invalids charged, no truncation, exact native finish
checker, and missing/transport unknown distinct from observed budget failures.
Propose one hour per adapter, two hours total, at most 6,144 native calls.
Completion within the cap is not guaranteed; never substitute shorter tasks.

Implemented `eval_textcraft_fresh.py` reuses the qualified native run and endpoint
validator, pins this exact new manifest, and constructs 32 original/flat jobs
per teacher. The historical eight-task prepare function remains unchanged.
Main authorized a minimal **worktree-only** shared correction: summary counts
now derive from PLAN jobs; the native auditor accepts an explicit
`expected_task_count=16` while retaining the historical default8. No sealed/live
source was edited. `analyze_textcraft_fresh.py` replays both complete native
receipt trees and reports full-panel uncertainty and depth strata; it does not
mislabel these as052/057. Main still owns scientific acceptance and launch.

Primary statistic: paired root-level mean success difference over the full
planned panel, with coverage alongside it; two seeds are not independent roots.
Report depth strata, invalid/nonexistent-query rates, root-first discovery,
all-call costs, and shared-dependency caveat. Sixteen task bootstrap units remain
exploratory/conditional on a single recipe world, not independent-world evidence.

## Most discriminating TRAIN-only RL readiness test (not implemented)

Before choosing an optimizer/reward, freeze a small disjoint **official TRAIN**
root panel and collect four bounded trajectories per root from fixed public056,
without touching this VAL panel. Inspect native-success variation **among valid
query/craft executions**, not just syntax failures, and where failures occur:
discovery, quantity aggregation, or revisiting known state. Include a small
same-plan/downstream-seed repeat to assess whether terminal credit is mostly
execution noise. Mixed valid success with stable discovery failures supports a
small terminal-reward RL pilot; uniformly successful tasks need harder TRAIN
coverage, while uniformly failing/noisy tasks need a targeted execution or
memory control before spending on RL. No arbitrary minimum mixed-group gate,
no policy tuning from fresh VAL, no new novelty claim. A finite sampled-block
budget with zero-advantage skips is preferable to stopping at the first flat
batch or promising indefinite additional training.

## Reproduction and source pins

`R` is `/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.
Source: `R/source-textcraft-fresh-panel-001`; exact six source hashes are in its
SOURCE.json. Standalone entrypoint `prepare_textcraft_fresh.py`; focused tests
cover exclusion/unique roots/order invariance, prerequisite closure, and one
actual official native/public-token trace. Three tests passed from the worktree;
the same three tests passed from the sealed source in5.44s. Ruff passes. No broad suite.

Input SHA256s:

- tasks: `da9f7498ffc136d24cc348523fe7be63586fc09bb2220ec05c35e86474382e8c`
- selection: `8792bffd3c3da6b57d22a8757b53ca1923cc4d0a37eac1eddb5a13af03e5e66e`
- manifest: `a145cb33aa6d2d62566c75bfcf432763369bf648f766bde6c99cda4bf6369a33`
- preparer: `43c31da5d9742023bcc1698b2aef988458c9f98e5115ea941cbff0d2b5a4d630`

With the existing training Python environment and CUDA disabled, reproduce only
into a **new absent output directory**; never overwrite001:

```sh
python source-textcraft-fresh-panel-001/prepare_textcraft_fresh.py \
  --root "$R" --output "$R/textcraft-fresh-inputs-NEW" --stage freeze
python source-textcraft-fresh-panel-001/prepare_textcraft_fresh.py \
  --root "$R" --output "$R/textcraft-fresh-inputs-NEW" --stage audit
```

The exclusion inventory is prospective: a later rerun after new panels appear
will include them. Exact001 reconstruction uses the paths/hashes recorded in
its SELECTION.json, not an assertion that future filesystem discovery is frozen.

## Readout seal and exact command

`R/source-textcraft-fresh-readout-001` contains the narrow wrappers and exact
shared dependencies. Six sealed tests passed in8.12s, including old8/new16/8×4
summary denominators, default8 versus explicit16 native audit, both real fixed
checkpoint23 identities and matched schedules, full unknown-pair handling, and
an actual052 request's native IDs/decode/sampling/adapter checks. Both sealed
`--prepare-only` invocations produced 32-episode PLANs with GPU_loaded=false.
Only PLANs exist in `textcraft-fresh-privileged-001` and `textcraft-fresh-public-001`;
no scientific responses, owner or GPU process were created.

`R/TEXTCRAFT-FRESH-READOUT-PROPOSED-001.json` binds the source/input/PLAN hashes
and both exact future argv lists. It is explicitly proposed, not accepted.
Future main-owned command for each teacher (existing training Python):

```sh
python "$R/source-textcraft-fresh-readout-001/eval_textcraft_fresh.py" \
  --prepared "$R/textcraft-fresh-inputs-001" --teacher privileged \
  --output "$R/textcraft-fresh-privileged-001" --hours 1
# Repeat for --teacher public and output textcraft-fresh-public-001.
```

After both owners terminate, CPU-only analysis writes one new immutable JSON
and Markdown report; partial/capped coverage remains explicit:

```sh
python "$R/source-textcraft-fresh-readout-001/analyze_textcraft_fresh.py" \
  --privileged-output "$R/textcraft-fresh-privileged-001" \
  --public-output "$R/textcraft-fresh-public-001" \
  --report "$R/analysis-textcraft-fresh-teachers-001.json"
```
