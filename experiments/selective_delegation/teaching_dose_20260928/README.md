# Matched additional optimization: fixed teacher dose

Ready CPU preparation; only the parent launches GPU jobs. The question is whether
the quantity-corrected **known-recipe** teacher mainly needs more optimization.
`known_recipe` never denotes a corrected version of the discovery model.

Both teachers resume their original seed2026092208 checkpoint23, with BF16 base,
FP32 LoRA rank8/alpha16/dropout0, and the saved SFT AdamW moments, step counter and
Python/torch/CUDA RNG. There are exactly two new epochs:46 new optimizer updates,
732 one-row microbatches, and17,640 additional supervised JSON+EOS tokens. Total
endpoints46 and69 are fixed prospectively, not chosen using validation outcomes.
Each epoch uses source048 `epoch_order(366,2026092208,epoch)` with epoch1 then2,
batch16 and a final14-row batch. LR remains1e-4, weight decay0 and norm clip1.
Each update uses source048 summed target loss divided by that minibatch's target
tokens. Every update commits adapter, AdamW, RNG and cumulative state.

This retains each teacher's original row order and histories. It matches total
action/label token dose, not row-aligned minibatch targets or prompt-token dose.
It does **not** remove the teacher observability/distribution confound.

## Original losses and feasibility

| Original teacher | First five online minibatch NLL mean | Last five | Final minibatch | Original owner wall time | Predicted two-new-epoch owner time |
|---|---:|---:|---:|---:|---:|
| Discovery |0.7161|0.06340|0.05135|207.24s|395.46s|
| Quantity-corrected known-recipe |0.7573|0.17602|0.18742|200.83s|382.87s|

The known teacher's residual loss is higher, leaving underfitting plausible.
These are pre-update losses on changing minibatches, not fixed-endpoint NLL,
and teacher conditioning histories differ. The final losses alone cannot show
that additional fitting will improve public execution rather than strengthen
memorized privileged naming. Estimate = twice original gradient runtime plus
original non-gradient overhead; expected roughly6–7minutes per teacher, with a
hard cumulative30-minute cap each. All23 loss receipts and their hashes are
included in the continuation PLAN's `original` record.

## Exact inputs and plans

Common root:

```text
R=/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921
D=$R/textcraft-teaching-dose-20260928-001
```

| Teacher | Original complete checkpoint | Frozen dataset | New continuation PLAN |
|---|---|---|---|
| discovery |`$R/textcraft-public-discovery-sft-001/checkpoint-0023`|`$R/textcraft-public-discovery-prototype-001`|`$D/train-discovery/PLAN.json`|
| known_recipe |`$R/textcraft-quantity-matched-seed2026092208-001/checkpoint-0023`|`$R/textcraft-quantity-matched-inputs-004`|`$D/train-known_recipe/PLAN.json`|

Original PLAN hashes are respectively
`5a34080562a39dc92dee2a813078e730b883fb60944dad96b613598c31f9624c` and
`368b0b3df7367129a78760010dbaf9664f780b41e123cf8902e077453a6cbb60`.
Original rows hashes are respectively
`dd152038a7f7da337c6cffe89a5a83a016d405cb251f4e26d3c3ec0d8f1da30a` and
`24ea72cb1242f2e0d819d8fb115737864de03fb750e064f145f9ec48a245e6d6`.

Prepared continuation PLAN hashes are respectively
`127923cb7504a7a47a4340f8475ee532500faa92ccf6b4ca6203081e84e93a7c` and
`86a4f1eecff14f3a0330e62a2fc061f41d46d84e18cc18afc1105189578ac5db`.
Direct CPU inspection found504 Adam parameter states at step23, LR1e-4, and one
saved CUDA RNG state in each original checkpoint.

No original inputs, source048 files, checkpoints, or queued teaching-order files
were edited. New training code imports pinned source048 tokenization, objective,
shuffle and checkpoint serialization. Preparation verifies original adapter,
optimizer/RNG boundary, input and small model-manifest hashes, without walking
multi-gigabyte base-model ancestry.

## Ready parent commands

```bash
PY=/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python
P=/project/alex_phd/repos/rlm/.worktrees/selective-delegation-20260921/experiments/selective_delegation/teaching_dose_20260928
$PY "$P/train.py" --teacher discovery --resume
$PY "$P/train.py" --teacher known_recipe --resume
```

`--resume` is required because both immutable plans already exist. Source pins
include `train.py` and `dose_common.py`; do not edit them after queue acceptance.
The output checkpoints are `$D/train-TEACHER/checkpoint-0046` and
`$D/train-TEACHER/checkpoint-0069`. Counters are cumulative from the original
training: checkpoint46 is two total epochs, checkpoint69 three total epochs.
Both commands acquire the existing coordinator lock, honor allocation margin,
and save authenticated owner/terminal receipts. No automatic failed-job retry.

Read each fixed endpoint on existing panel00, worlds42 and50, with the original
two execution repeats (16 attempts per cell). The accepted primary readouts are
raw; binder execution is supported but is a separate named extension.

```bash
$PY "$P/readout.py" --teacher discovery --step 46 --world 42 --execution raw
$PY "$P/readout.py" --teacher discovery --step 46 --world 42 --execution raw --audit
```

Repeat for both `--teacher discovery|known_recipe`, both `--step 46|69`, and
both `--world 42|50`:eight primary cells. Each16-attempt readout is expected to
take15–25minutes and retains the original45-minute cap. Outputs are
`$D/eval-TEACHER-cpSTEP-p00-wWORLD-raw/{PLAN.json,NATIVE-AUDIT.json}`.
`--prepare-only` is CPU-only; future endpoint preparation becomes available only
after the actual training checkpoint exists. `--step23` correctly binds each
teacher's original endpoint, but completed matched baseline runs already exist:

| Teacher | World42 raw baseline | World50 raw baseline |
|---|---|---|
| discovery |`$R/textcraft-breadth-p00-w42-soriginal-raw-001`|`$R/textcraft-breadth-p00-w50-soriginal-raw-001`|
| known_recipe |`$R/textcraft-breadth-p00-w42-soriginal-corrected-raw-001`|`$R/textcraft-breadth-p00-w50-soriginal-corrected-raw-001`|

Their `PLAN.json` and `NATIVE-AUDIT.json` bind baseline adapter, tasks, world,
sampling seeds, prompts and caps. Readout templates come from the discovery
baseline solely for this shared task/world/harness contract; new plans replace
teacher, adapter, checkpoint dose and condition explicitly. Never call the
known-recipe endpoint a discovery baseline merely because that template is reused.

## Qualification and interpretation

Focused CPU tests passed for fresh epoch shuffles, cumulative microbatch/token
state, rejection of partial/nonprospective endpoints, exact saved-AdamW/RNG
roundtrip, and unchanged paired jobs with truthful teacher/checkpoint labels.
Both real training `--prepare-only` commands passed. Actual original-checkpoint
readout preparation passed for known/world42 and discovery/world50. Actual GPU
training remains a runtime check, not something established by CPU tests.

Compare each fixed endpoint to its own checkpoint23 baseline, with paired task
and execution-seed differences. These readouts reuse exploratory panel00 and
contain only eight task identities, not32 independent goals across both worlds.
Report both fixed checkpoints even if one is worse; do not select the better one.
A later training failure does not erase an earlier prospectively fixed committed
boundary: readout qualification retains terminal statuses and requires the owner
to have released the resource. Any such partial campaign must be labeled.

Optional fixed-endpoint teacher-forced NLL is a separate measurement, not a
checkpoint selector and not a substitute for native task success.

## Optional NLL commands

```bash
$PY "$P/nll.py" --teacher discovery --step 23
$PY "$P/nll.py" --teacher known_recipe --step 23
```

Repeat at the prospectively fixed steps46 and69 after those checkpoints exist.
Each model is evaluated on its own frozen teacher rows by default; an optional
`--dataset discovery|known_recipe` enables a separately labeled cross-dataset
diagnostic. NLL is summed over the full original JSON+EOS targets at temperature1,
then divided by tokens within each stratum. It is not name-token-only NLL.

The strata are all366 rows,167 get-info queries,167 crafts,32 finishes and32
first actions. Structured public-name visibility divides discovery queries167/0
visible/unseen and known-recipe queries32/135. Schema/example substrings do not
count as public observations. Metrics and per-row details are written to
`$D/nll-TEACHER-cpSTEP-on-DATASET/NLL.json`.

The two original-checkpoint NLL plans have passed real CPU preparation; no NLL
GPU measurement has run. Expected forward-only cost is roughly1–3minutes per
366-row cell, extrapolated from the original gradient runtime, with a15-minute
owner cap. This optional queue must not delay the dose trainings/readouts.
All six focused tests now pass, including the structured-visibility fixture.
