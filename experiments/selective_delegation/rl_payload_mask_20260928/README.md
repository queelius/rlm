# Same-batch ingredient-payload loss diagnostic

September 28 CPU preparation; **parent alone launches**. This is one biased
gradient intervention, not an unbiased executed-action policy-gradient estimator,
not a new collection, and not a correction to valid full-token REINFORCE.

The completed binder batch has 28/32 native successes and three mixed task groups.
Of its 8,440 nonzero-credit tokens, 2,370 are strictly interior to eligible
ingredient values. Their absolute advantage-weighted mass is 1,186.67 / 4,352
(27.27%). This passes the prospective gate: at least one mixed group, at least
100 credited masked tokens, and at least 1% of absolute credited token mass.

## Exact contrast

The existing full-token update and this masked update start independently from
the same actual public-discovery checkpoint23, with fresh AdamW at 2e-5, clipping1,
FP16 base/FP32 LoRA, one optimizer step, and exactly the same native-audited binder
batch and signed leave-one-out terminal advantages. Denominator remains 32. The
masked endpoint is not initialized from the full-updated actor.

Only tokens **strictly interior to the JSON value** of `ingredients` lose their
direct log-probability loss term, and only when a prior public observation makes
the call `bound_observed_single_recipe`. Eligibility is recomputed from native
public history and checked against the saved assist. It does not depend on native
craft success, changed-versus-already-correct ingredients, or final reward.
All other tokens keep credit: fixed key/colon, boundary-overlap tokens, EOS,
action/target/count, queries, finish, unbound/nondivisible craft and schema errors.

All original emitted IDs and the entire autoregressive prefix remain present.
The original full-token generation/replay and train/eval checks, their tolerances
and their counts are unchanged. Retained loss tokens are recorded separately as
`retained_loss_tokens`; the original numerical-check count is
`original_replay_tokens`. Both reported objectives use the actual mask, including
the post-update objective. The inherited `positive_tokens`/`negative_tokens` and
probability-change diagnostics still cover all original nonzero-credit tokens.

The private adapter loads the pinned original trainer and changes four exact,
counted source expressions; hashes and substitutions are in
`PRIVATE-ADAPTER.json`. It never edits the live trainer. A privately loaded RL
module keeps stop/guard state intact. Its first actual saved-call replay writes
`FIRST-REPLAY.json` promptly and rejects nonfinite/excessive gaps before continuing
the full-batch replay. All full numerical safeguards remain active.

## Ready parent commands

```bash
PY=/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python
E=/project/alex_phd/repos/rlm/.worktrees/selective-delegation-20260921/experiments/selective_delegation/rl_payload_mask_20260928
S=/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/textcraft-rl-payload-mask-20260928-001

# Parent-scheduled scientific jobs: 60-minute train, then 45-minute readout.
"$PY" "$E/train.py" --hours 1
CUDA_VISIBLE_DEVICES= "$PY" "$E/analyze.py" --kind train --output "$S/train-0001/ENDPOINT-AUDIT.json"
"$PY" "$E/readout.py" --hours 0.75
CUDA_VISIBLE_DEVICES= "$PY" "$E/analyze.py" --kind compare --output "$S/COMPARISON.json"
```

`PREPARED-JOBS.json` has exactly two scientific jobs, source/data pins, CPU audit
and comparison commands, and explicit output roots. It is not an acceptance or
launch receipt. The training job requires the actual usable full-token baseline;
the readout requires the actual usable, CPU-audited masked endpoint. Future
adapter/readout-plan hashes remain unknown until those endpoints exist. There is
no warm or partial-checkpoint fallback. Both scientific jobs use the existing
shared GPU coordinator lock and parent-supplied allocation deadline.

Only the masked endpoint's 16 readout episodes are new. They pair the same TRAIN8
task/repeat/seed identities and binder interface with the accepted warm/full
controls at `textcraft-rl-assist-20260928-001/binder/readout-{warm,0001}`. Missing
controls stay unknown. Estimated additional GPU time is 15–35 minutes for the
update plus 15–30 minutes for readout, with hard caps 60/45 minutes. No extra
gradient-decomposition backward or cosine diagnostic is included.

## What would change the decision?

The primary screen is masked-minus-full native success on the 16 paired slots,
then masked-minus-warm. Also inspect task-cluster uncertainty, cost, schema/native
errors, and premature finish. A prospective promotion rule is a net gain of at
least 2/16 across at least two tasks with no schema-error increase. A tie/loss or
only favorable same-prefix likelihood movement retires this one-step candidate;
it does not prove general equivalence of objectives.

If payload-off wins, add a **matched-size random-token mask or matched
gradient-scale control** before claiming specificity to overwritten fields.
Fewer loss terms, Adam and clipping can produce generic scale/regularization
effects. No such extra arm is prepared or run here. Ingredients also condition
later target/count tokens and consume parsing/budget resources, so masking is
not generally an unbiased estimator even though the binder overwrites them.

Broad action/argument gradient separation is already prior art in
[Harness-RL v1](https://arxiv.org/html/2608.29641v1). This narrower contrast removes
one execution-overwritten payload while holding the actor, batch and inference
interface fixed; it is not CAPO parameter routing. The queued compact RL cycle
changes schema, history, warm SFT and fresh task groups and can save generation
cost. This loss mask does not remove inference tokens or compute those savings.

## CPU qualification

Six focused tests cover payload boundaries, JSON-string decoys, signed `/32`
loss derivatives, public-recipe eligibility, fail-closed source rewrites, and
early numerical-gap rejection. The actual scripted six-call saved-request/loss
fixture includes schema rejection, unobserved craft, nondivisible count, repaired
observed craft and native root success. It checks captured/replayed original token
probabilities and exact causal inputs; analytical reward examples do not relabel
the native episode. Earlier CPU fixtures are preserved; final
`runtime-fixture-003` pins the audit-gated readout and every imported comparison
source. These are scripted qualification attempts, not scientific model runs.

The experiment does not establish held-out generalization, gradient conflict,
or RLM recursion. See the [binder census](../rl_signal_20260928/BINDER-RESULT.md).
