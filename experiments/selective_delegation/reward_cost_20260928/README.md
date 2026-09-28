# One-step native success minus error cost

Question: does a tiny error penalty make the first RL update more useful than terminal reward on
the *same* warm-policy trajectories, or mainly encourage inexpensive unsuccessful stopping?

This is a changed multiobjective reward, **not** reward-equivalent potential shaping or process
credit. Native success remains the primary outcome. The two cost branches do not collect new
training rollouts and do not continue the terminal-trained checkpoint.

## Fixed comparison

| Dimension | Terminal control | Cost branch |
|---|---|---|
| Initial actor | Public discovery SFT checkpoint0023 | Exactly the same checkpoint |
| Data | First fresh v002 raw/binder groupA32 collection | Exact same per-interface collection |
| Reward | Native binary success | Native success −0.001×(invalid_schema + native_action_error) |
| Credit | Signed RLOO, other three same-task samples | Same estimator with composite rewards |
| Update | One fresh AdamW, LR2e-5, clip1 | Same; no inherited Adam and no continuation |
| Targets | All original sampled tokens atT0.5 | Identical targets, including EOS and errors |
| Readout | Fixed diagnostic groupB8×2 | Same groupB goals, seeds, world and caps |

Each original trajectory is used at most once for this new objective/actor. Both raw and binder
have96-call/8192-output-token caps. Their realized trajectory lengths and token credit differ;
the loss is a trajectory-token **sum** divided by32, not length normalized. Composite rewards can
also make previously native-flat groups nonzero, so credited-token dose is not equal to the
terminal control. Executed binder ingredients remain state transitions, never likelihood targets.

All costs are nonnegative integer counts, mutually exclusive and bounded by physical calls≤96.
Thus successful rewards are≥0.904 and failed rewards≤0. Original `native_score` labels and native
audit files are never modified. `REWARD-TABLE.json` is a separate derived artifact. The literal
specified formula excludes the bridge's separate `rejected_action` category, which is still
reported. The full action likelihood gets trajectory credit, not only the erroneous action.

Per-trajectory reward ordering does not guarantee policy-level success preservation. In an
all-failure group, cheap immediate stopping may earn positive advantage. Adam's normalization and
gradient clipping mean that coefficient0.001 does **not** guarantee a tiny parameter update.

## Inputs, conditional execution and outputs

External root:
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.

- Inputs/control: `textcraft-fresh-rl-20260928-002/{raw,binder}/collect-0001`, `train-0001`,
  `readout-0001` and `readout-warm`.
- Frozen data: `textcraft-fresh-train-20260928-001/{train,diagnostic}`. Both are official TRAIN;
  diagnostic roots/IDs are disjoint from optimization. No official VAL/HOLDOUT goals enter RL.
- New outputs: `textcraft-error-cost-20260928-001/{raw,binder}/{train-0001,readout-0001}`,
  `PREPARED-JOBS.json`, then `COMPARISON.json`.
- The environment receipt is reused and hashed from the fixed fresh study. No environment changes.

Only the complete native-audited first collection under publiccp23 is admitted. Incomplete or
missing first batches skip that arm; no partial batches, replacement tasks, invented failure
rewards, or alternate warm actors. A flat composite-reward batch skips without GPU/model loading.
Numerical replay/gradient failures retain the existing trainer's failure/boundary policy. Only a
usable committed first-step endpoint gets read out. Missing terminal-only endpoints remain unknown,
not warm replacements. One failed branch does not prevent the other from running.

`train_cost.py` privately loads the unchanged existing trainer and replaces only its preparation
and batch-credit functions. It retains native receipt checks, generation-versus-replay probability
checks, train/eval checks, original sampled token likelihood, gradient clipping, checkpoint/Adam/RNG
commits and before/after probability diagnostics. Positive/negative diagnostic tokens refer to
*composite advantage*, not successful/failed trajectories. No live/pinned files are edited.

## Parent dispatch

Preparation is CPU-only and may run before the future batches exist:

```bash
PY=/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python
C=/project/alex_phd/repos/rlm/.worktrees/selective-delegation-20260921/experiments/selective_delegation/reward_cost_20260928
$PY "$C/prepare_jobs.py"
```

`PREPARED-JOBS.json` contains all five exact job argv arrays, source/data/plan pins and caps. It is
**not** an accepted queue and cannot launch itself. The parent must schedule these after existing
owners release, not create an independent race with the already queued fresh four-cycle owner.
All GPU stages use the existing shared `root-rlvr-campaign-v1/COORDINATOR.lock`.

Once both original first-collection attempts finish, the exact sequential stage commands are:

```bash
$PY "$C/stage_cost.py" --kind train --mode raw
$PY "$C/stage_cost.py" --kind train --mode binder
$PY "$C/stage_cost.py" --kind readout --mode raw
$PY "$C/stage_cost.py" --kind readout --mode binder
CUDA_VISIBLE_DEVICES= $PY "$C/analyze_cost.py" --output /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/textcraft-error-cost-20260928-001/COMPARISON.json
```

Training PLANs are prepared now, conditional on future complete audited batches. Readout PLANs are
created at launch from the *actual* committed cost checkpoint; `READOUT-LINEAGE.json` binds its
training PLAN/SUMMARY and fixed groupB manifest. Never guess a future checkpoint hash.

Expected useful work: roughly15–40min training and25–50min readout per arm, total1.3–3h.
Long-trajectory safety caps are120min per training and90min per readout (7h scientific total).
Supervisor caps include5min per-stage overhead plus10min analysis:7.5h total maximum job caps.
No new warm/terminal baseline rollouts are requested. A failed or native-flat terminal update is
not silently promoted, while an admitted cost update can proceed if its composite groups vary.

## What would change the research decision?

Report paired native successes on all fixed16 diagnostic slots against terminal-only step1 and
warm separately, with task-cluster uncertainty. Preserve unknown/incomplete cells. Report calls,
tokens, each error category, entropy, response diversity, native/composite-mixed groups and
all-failure cost-varying groups. Also report unsuccessful explicit finishes, unsuccessful first-call
finishes, and unsuccessful finishes within≤4 calls. `get_info` calls and per-episode distinct
queried items are descriptive exploration proxies, not an efficacy claim.

Promote only a credible success or useful exploration benefit, never lower errors/calls alone.
Retire this weight/objective if it is cheaper without success benefit and increases failed early
finishes or reduces information exploration; the report emits that prespecified warning. Review
the actual trajectories before attributing causality. One small TRAIN-transfer readout is not
held-out confirmation, and no diagnostic score selects favorable checkpoints.

## Focused verification

```bash
$PY -m pytest "$C/test_cost.py" -q
ruff check "$C"
```

The hand-calculated four-reward fixture uses rewards `[1,.994,-.002,-.008]`, RLOO advantages
`[.672,.664,-.664,-.672]`, log-probability sums `[-1,-2,-3,-4]` and actual accumulated loss
`−.08375` with gradients `[-.021,-.02075,.02075,.021]`. It runs through complete32 sample credit
validation and the unchanged trainer objective. A separate differentiable replay checks sampled
token IDs andT0.5. A real saved-batch loader fixture checks native audit/receipt/generation-logp
admission and proves that native scores and original call files are unchanged.
