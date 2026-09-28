# September 28: does observed-recipe assistance change reward learning?

This is an executable, exploratory TRAIN-only comparison. The starting actor is
public-discovery SFT checkpoint23, with FP16 base arithmetic and FP32 LoRA. Both
arms get the same eight existing TRAIN goals, four new rollout seeds and identical
96-call/8192-output-token episode limits. One arm executes original JSON arguments;
the other invokes the existing deterministic recipe binder. No evaluation goal is
used for training or for these diagnostic readouts.

The binder preserves the model's requested target and output count. It replaces
only ingredients, using a single recipe already returned in public history when
the requested count is divisible by the recipe yield. It does not supply missing
inventory, choose subgoals, or repair malformed schema. The existing sealed binder
collector and auditor are imported directly; no source tree was copied or edited.

## Prior evidence and why this is different

The earlier terminal objective had real reward variation: the FP16 cp1 batch had
five mixed groups, with success counts `[2,2,2,3,3,4,4,4]` out of four. Raising the
learning rate from 2e-5 to 1e-4 changed weights five times as much but produced
16/32 versus15/32 in its exposed evaluation. Positive-only credit then tied signed
credit at15/32. See [RL credit assignment](../RL-CREDIT-ASSIGNMENT-20260923.md) and
[the completed positive-only readout](../MORNING-UPDATE-20260924.md).

The original TRAIN8 warm rollout was25/32, with three mixed groups. This panel is
familiar and may become too easy with assistance. The present comparison uses new
rollout seeds and starts again from the public SFT actor; it changes the transition
interface during collection and learning. It does not repeat the cp1 saved-batch
LR or positive-only comparison. If rewards are flat, the trainer records a CPU
conditional skip, never invents an update. Fresh prospectively chosen official
TRAIN goals should then outrank repeatedly sampling this panel.

## Objective and on-policy status

For task `g`, four independent execution seeds produce terminal rewards `r_gi`.
The advantage is `A_gi = r_gi - sum(j != i, r_gj)/3`, and the minimized objective is
`-sum(g,i)[A_gi * sum(all emitted actor tokens) log pi_T0.5(token|prefix)] / 32`.
There is no reward standardization, positive-only filtering, token mean, PPO
clipping, KL term, or importance weighting. All-success/all-failure groups receive
zero credit but remain in denominator32. Invalid-action paths keep native terminal
rewards; missing/transport-failed episodes cannot enter the batch.

Each update reloads exactly the FP16 collection checkpoint, verifies captured
generation probabilities against unchanged-weight replay, accumulates one full
signed gradient, clips its norm to1, and takes one AdamW step at2e-5. The first
optimizer is fresh; later explicitly requested updates restore the preceding
committed Adam state and require fresh trajectories from those updated weights.
There is no repeated optimization of an old batch.

Crucially, likelihood targets are the ORIGINAL emitted token IDs, including wrong
ingredient fields and EOS. The deterministic binder acts downstream as part of
the environment transition. Substituting executed arguments as targets would
turn this into a different, generally off-policy/imitation objective. Simply
masking ingredient tokens would also change the estimator: autoregressive
ingredient tokens can condition subsequent decisions and schema validity.

## Ready commands

The environment is the existing training interpreter:

```bash
PY=/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python
P=/project/alex_phd/repos/rlm/.worktrees/selective-delegation-20260921/experiments/selective_delegation/rl_resume_20260928
S=/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/textcraft-rl-assist-20260928-001
```

For each `MODE=raw` and `MODE=binder`, run in the order below through the parent
coordinator. Both first collection/training plans and both warm readout plans are
already CPU-prepared. Remove no artifacts on failure; use a separately named,
explicit retry after diagnosing the observed seam.

```bash
$PY "$P/collect.py" --mode "$MODE" --output "$S/$MODE/collect-0001"
$PY "$P/train.py" --collection "$S/$MODE/collect-0001" --output "$S/$MODE/train-0001"
$PY "$P/collect.py" --mode "$MODE" --phase readout --hours .75 --output "$S/$MODE/readout-warm"
$PY "$P/collect.py" --mode "$MODE" --phase readout --hours .75 --checkpoint "$S/$MODE/train-0001/boundaries/sample-0001/checkpoint-0001" --output "$S/$MODE/readout-0001"
CUDA_VISIBLE_DEVICES= $PY "$P/compare.py" --root "$S" --output "$S/COMPARISON-0001.json"
```

Only run the updated readout if `train-0001/SUMMARY.json` says `endpoint_usable`.
The warm readout is independently useful even if training fails or has flat reward.
All GPU stages acquire the existing
`sidecars/root-rlvr-campaign-v1/COORDINATOR.lock` under the campaign store. Preparation
runs before that lock; it pins existing task/model manifests and verifies adapter
bytes, without repeatedly walking multi-gigabyte ancestral model files. Use
`--prepare-only` for CPU-only invocation. No script retries transport errors.

Collection caps at75minutes for32 episodes, readout at45minutes for16, and training
at60minutes for one update. Expected useful times from prior native runs are
25–45,15–30 and15–25minutes respectively. These are estimates, not lease promises.
The first real scientific response is reported through native STATUS.json and
must be checked promptly by the parent. Every stage preserves owner and terminal
receipts. Training commits adapter, optimizer, RNG and state after each update.

## Measurements and decisions

The primary diagnostic is within-interface native success gain on16 paired
readout episodes, then the difference between binder and raw gains. Eight task
clusters, not32 or64 independent worlds, define the exploratory bootstrap.
Generation sidecars save sampled-token log probabilities and full-vocabulary
entropy atT0.5. Reports include first-response diversity, action-error counts,
calls, tokens, native service time, mixed reward groups, credited-token counts,
adapter movement and positive/negative sampled-token probability changes.
Those probability changes are not KL divergence or generalization metrics.

The same call/token caps do not match realized gradient dose: shorter binder
trajectories produce different token sums. A promising signal requires a matched
extra-SFT control and new TRAIN goals before scaling; a generalization claim
requires a separately frozen held-out test. Own-interface readouts do not show
cross-interface transfer. Crossed endpoint/harness evaluations can be added as a
later named comparison. These actors are flat, so no result here establishes
RLM recursion benefit. This tests learnability under exact observed-recipe
execution, not novelty of deterministic binding.

Conditional scaling is supported with `collect.py --update 2 --checkpoint` set to
the previous committed endpoint, followed by `train.py` on that fresh collection;
the maximum prospective update index is4. This is a capability, not automatic
approval to continue on the near-ceiling panel. Select the next question from
complete reward/diversity/readout evidence and the extra-SFT control.

## CPU verification

Seven focused tests cover TRAIN-only paired scheduling, native repair and replay,
original sampled targets versus executed arguments, real tiny-model generation
scores/entropy versus causal replay, signed-token objective gradients, probability
diagnostics and unknown-aware paired summaries. Full project tests were not run:
this research adapter is isolated, and the repository's GPU-first policy requests
proportional focused verification. No GPU job was launched during preparation.
