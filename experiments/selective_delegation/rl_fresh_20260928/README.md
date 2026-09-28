# Fresh TRAIN optimization and transfer diagnostic, September28

The prepared study is
`R/textcraft-fresh-rl-20260928-002`, where
`R=/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.
No GPU stage was launched during preparation. The parent dispatches the finite
queue after the two teacher controls and their readouts.

## Question and fixed comparison

Starting from the same public-discovery SFT checkpoint23, does deterministic
binding of previously observed recipe arguments change reward learning on new
optimization goals? Does that learning transfer to different TRAIN target roots,
and does it outperform one additional public-teacher SFT optimizer step?

Each of the raw/binder arms samples four trajectories on each of eight training
goals, then performs one signed terminal-RLOO update. Up to four prospectively
numbered cycles use fresh trajectories from that arm's immediately preceding
checkpoint and restore its Adam state. Each completed update has an independently
audited16-episode readout on the same disjoint diagnostic group and paired seeds.
Every prospective checkpoint is retained and reported. Diagnostic outcomes do not
select favorable checkpoints or gate continuation.

Collection, scoring, sampled-token likelihood and checkpoint code reuse the
unchanged `../rl_resume_20260928/` implementation. A privately loaded module gets
explicit fresh-manifest dependencies; the original module files and live sources
are never edited. Both record paths and hashes identify the actual fresh data.

The policy objective remains full original sampled-token signed RLOO atT0.5,
with denominator32, fresh AdamW2e-5 on cycle1, clip1 and FP16 base/FP32 LoRA.
Binder ingredients are environment transitions, never substituted learning targets.
No importance weights, repeated old-batch optimization, reward normalization,
positive-only filtering or ingredient-token masking are introduced.

## Data and independent control

The dataset root is `R/textcraft-fresh-train-20260928-001`, frozen separately by the
fresh-data agent. Root manifest SHA256 is
`76e387378c9a89b0d8e26180874db61ad40086721c6cfef9e3b78dd77f915e05`.
Each group has actual dependency-depth quotas3/4/5 of2/3/3 tasks. Original official
JSONL bytes and dictionary order are preserved. Both groups exclude original SFT
and previous task roots/IDs; their roots/IDs are disjoint from each other. The
entire official VAL target-root inventory was also excluded during selection.

All IDs have prefix `textcraft_synth.train.`:

| Group | ID suffixes | Use |
|---|---|---|
| A / `train` |132,1423,2281,1753,1921,1051,2338,2026|RL trajectories/gradients only|
| B / `diagnostic` |1796,256,672,1273,1847,38,201,964|Warm, RL checkpoint and extra-SFT readouts only|

All16 goals have successful public-only native CPU feasibility traces. Initial
prompt+256 lengths are at most821/859 tokens. Full model histories can still hit
the unchanged8192 context limit; native feasibility is not full-context solvability.
All training and readouts use the original seed42 recipe world.

The extra-SFT control starts at the same publiccp23 and takes one fresh AdamW2e-5
step over all original366 public teacher-action rows,8820 target tokens including
EOS, with clip1 and the same arithmetic. Its conventional T1 token-mean NLL differs
from T0.5 trajectory-sum RL. Data, histories, token dose and compute also differ.
This is matched on optimizer-step count/LR/start/arithmetic, not an equal-token or
equal-FLOP control. At RL checkpoints2–4 it remains a fixed one-step SFT reference.

## Dispatch and caps

The default predecessor is the parent's last teacher GPU result:
`R/textcraft-teaching-order-20260928-001/eval-random_visible-s2026092208-p00-w50-raw`.
The existing generic supervisor waits for its authenticated owner to terminate,
then runs individually capped jobs serially. Every scientific stage acquires the
same `sidecars/root-rlvr-campaign-v1/COORDINATOR.lock` as the current campaign.

```bash
PY=/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python
P=/project/alex_phd/repos/rlm/.worktrees/selective-delegation-20260921/experiments/selective_delegation/rl_fresh_20260928
$PY "$P/launch_campaign.py"
# Parent dispatch, after reviewing the prepared receipt:
$PY "$P/launch_campaign.py" --updates 4 --launch
```

`--predecessor`, `--study` and `--queue-output` are explicit optional arguments.
The default study ends`-002`; `-001` contains preserved, unused90-minute preparation
plans and a supersession receipt. No scientific owner ever ran those old plans.
Do not reuse attempted output directories or alter pinned source after dispatch.

| Stage | Scientific cap | Expected useful duration |
|---|---:|---:|
|32-episode fresh collection|150min|roughly50–90min|
|One accumulated RL update and probability diagnostics|120min|roughly20–60min|
|16-episode diagnostic readout|90min|roughly25–50min|
|One extra-SFT update and NLL diagnostics|45min|roughly5–15min|

The existing world42 breadth depth3/4/5 per-episode means were39/90/181seconds raw
and34/65/166seconds assisted. Applying2/3/3 quotas×4 predicts59.4/50.8minutes per
training collection before variation/loading. Depth5 had maxima of357/372seconds,
so the larger collector caps avoid making a partial batch the likely result.

The33-job four-cycle queue has57.83hours of upper job caps and a64-hour overall
bound including predecessor wait, also limited by the allocation end. Expected
useful execution is roughly5–8hours for the first cycle and controls,14–24hours
for all four cycles if both arms stay informative. These are estimates, not a
promise that caps sum to allocation coverage. Skipped arms reduce useful coverage.

If a collection is incomplete, it cannot enter a partial gradient update. If all
task groups have zero terminal-reward variation, the trainer records a conditional
skip without loading the model. An unusable or failed optimizer endpoint stops
only that arm's later stages. The other arm, both warm groupB readouts and extra-SFT
remain independent jobs. The supervisor refuses to proceed past an unresolved
scientific owner. There are no automatic retries.

## Individual stages and analysis

For independent parent dispatch, the following commands run a single stage after
resolving its actual dependency. They use fixed caps and preserve explicit skips:

```bash
S=/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/textcraft-fresh-rl-20260928-002
$PY "$P/stage.py" --study "$S" --kind collect --mode raw --update 1
$PY "$P/stage.py" --study "$S" --kind train --mode raw --update 1
$PY "$P/stage.py" --study "$S" --kind readout --mode raw --actor warm
$PY "$P/stage.py" --study "$S" --kind readout --mode raw --actor rl --update 1
$PY "$P/stage.py" --study "$S" --kind sft
$PY "$P/stage.py" --study "$S" --kind readout --mode binder --actor sft
```

Repeat the raw stages with`--mode binder`; cycles2–4 automatically bind their own
previous committed checkpoint. Independent scientific scripts also accept
`--prepare-only`. First collection, training, warm-readout and extra-SFT plans
already exist; updated checkpoint/readout plans are created only when their
actual committed adapter files exist.

Each cycle emits`COMPARISON-000N.json`, reporting RL-minus-warm, RL-minus-extra-SFT
and the difference between interface learning gains. Missing cells produce
unknown pairs, not zeros or selectively dropped cases. Reports include cost,
error counts, generation entropy, response diversity, mixed reward groups,
gradient dose, adapter changes and sampled-token probability changes. Intervals
cluster by eight diagnostic goals; these share one recipe world.

GroupB is a prospectively disjoint TRAIN transfer diagnostic, not official held-out
confirmation. These flat actors do not establish recursion benefit. Promising
effects still need replication, equal-dose controls where relevant, and a separately
frozen confirmatory test. The binder itself is not claimed as novel.

## Verification

Focused CPU tests exercise true original group manifests, role separation, byte
identity, paired seeds, an actual newTRAIN native trace with deliberately wrong
ingredients and independent replay, original sampled likelihood targets, failed-arm
gates, the conventional extra-SFT loss, all fixed checkpoint readouts, and unknown
comparison cells. A first integration run caught a script name shadowing Python's
standard`queue` module; it was renamed before any GPU owner or final preparation.
No existing/live source, environment, checkpoint or model weights were modified.
