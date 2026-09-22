# One additional FP16 training step, explicit continuation

## Repaired attempt accepted, September 22 at 17:12 UTC

Queue 014 is now waiting behind the complete second-seed queue 013. It uses the
same scientific settings as the failed attempt. The repair restores the exact
compatible analyzer from the earlier training source; no numerical tolerance,
reward, learning rate, input task or checkpoint-selection rule changes.

The saved-plan regression reproduced the failure; main verified the repaired
snapshot with the actual training plan and native 32-attempt/770-call replay. Six focused
continuation/composition tests passed in 12.03 seconds; the endpoint test passed
in 0.09 seconds. All 75 unique proposed job pins matched. The only changed
common training Python file is the analyzer; the only training PLAN difference
is its source identity. The new snapshot includes the regression fixture.
Source-level verification is not a claim that the future GPU update succeeds.

The receipt is `INDEPENDENT-TRAINING-QUEUE-014.json`, SHA256
`4c7945a6ded49fbc40f6f5756e16d27c46b200acfd975aa4e03489db036ffa27`.
Training output is `textcraft-terminal-fp16-continuation-002`; readout is
`textcraft-fresh-fp16-cp2-bf16-002`; final native report is
`analysis-textcraft-fp16-cp2-minus-cp1-002.json`. Each uses the existing evidence
root below. Five hours of waiting and five hours of execution are separately
capped, with the allocation deadline still limiting both. Training retains its
three-hour cap, evaluation one hour, and CPU analysis twenty minutes.

The failed 001 attempt, source and dependent failures remain intact. Only a
genuine new, finite optimizer step may produce an eligible checkpoint2.

## First attempt failed before collection, September 22 at 17:05 UTC

The restored model passed the finite-loss/finite-gradient backward probe, with
no optimizer update during that probe. The run then failed after 9.55 seconds:
the sealed readiness helper calls `analyze_textcraft.validate_runtime_tasks`,
but that function is absent from the analyzer copied into its source snapshot.
There were no new rollouts or optimizer steps. The copied starting checkpoint
is not a new training result, and the dependent evaluation correctly rejected it.
The failure record is preserved. Repair must exercise the actual saved-plan
readiness interface before a new, separately recorded attempt. Queue 013 has
already advanced to independent teacher-seed training; no need to hold the GPU.

## Accepted queue, September 22 at 15:30 UTC

Training and its fixed BF16 evaluation are now accepted as queue012. They wait
for queue011's one-step comparison and resource release. The sequence is:

1. Restore checkpoint1 and run at most one additional RL update, capped at three hours.
2. If a normal, finite checkpoint2 is committed, evaluate all 32 fixed attempts,
   capped at one hour. Failed or flat-only training does not get relabeled as
   a usable checkpoint2.
3. Run CPU-only native analysis against the checkpoint1 evaluation. Missing
   outcomes remain unknown, not failures or zeros.

Queue012 has a four-hour wait limit and a separate five-hour execution allowance.
Main ran ten sealed trainer tests (5.27 seconds), four sealed readout tests
(0.12 seconds), checked all 103 job pins and validated the actual queue handoff.
A second agent independently reviewed both training and endpoint qualification
and found no material blocker. The future checkpoint2 integration is necessarily
still untested until that checkpoint exists.

Evidence root:
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.
Accepted receipt: `INDEPENDENT-TRAINING-QUEUE-012.json`, SHA256
`a9d37327307ffc7d6fe9ee7533006311bf9733df43b7510aa63e1e91bcb77cf4`.
Training output: `textcraft-terminal-fp16-continuation-001`.
Evaluation output: `textcraft-fresh-fp16-cp2-bf16-001`.
Final report: `analysis-textcraft-fp16-cp2-minus-cp1-001.json`.
The report does not exist yet; queue acceptance is not a training result.

In summaries, `new_sampled_batches` counts newly **committed** batch boundaries.
An interrupted collection can leave physical calls without such a boundary;
those calls remain saved and are counted separately in physical compute usage.

## Prospective specification (preserved)

Prospective CPU preparation only; main must accept before GPU launch. Diagnostic
evidence motivates a dtype intervention, not automatic promotion: FP16 cached/full
max .0437 versus BF16 .5 on the same47 prefixes; this did not qualify backward.

Restore RL002's sole committed checkpoint1, FP32 LoRA, actual Adam moments/step1,
and Python/Torch/CUDA RNG. Base arithmetic changes to FP16/SDPA; objective, T=.5,
LR2e-5, wd0, clip1, all32 denominator, strict native environment, task inventory,
and budgets stay fixed. An opt-in flag leaves existing default training unchanged,
apart from saving numerical receipts before rejection and rejecting nonfinite gaps.

Before new collection, run one finite-loss/gradient backward qualification on the
fixed longest saved BF16 call as a **numerical probe only**. No Adam step; clear all
gradients and restore RNG afterward. Any failure halts this owner promptly. Collect
only new FP16 on-policy rollouts: historical sample cursor1 is preserved; new sample
directories2/3 use seeds2026092253 +10*sample +repeat. The old BF16 sample2 remains
immutable and excluded from training. Transport-unavailable episodes halt without
retry or update. Main checks the first actual response promptly.

Target one additional committed optimizer step (cumulative2), at most two fresh
8-task x4-candidate batches, three-hour cap. A flat batch advances the cursor but
not Adam; two flat batches terminate without a usable cumulative2 endpoint. Save
all own-generation/full replay scores before unchanged max.25/mean.025 checks.
Finite objective/gradients, frozen base, teacher-forced train/eval check and commit
checks remain required. Each committed boundary records cumulative versus new steps.
There is no implicit retry/resume of this new owner or fresh-Adam substitution.

Follow-on comparison: only a successfully committed cumulative2 endpoint gets
the fixed BF16 fresh16 x2 readout. Compare with the currently running cp1 BF16
readout at identical test arithmetic; no RL-specific advantage claim without a
future cp1-warmstarted SFT control. The current056-warm one-step SFT control does
not itself match this additional step. A separate thin cp2 endpoint qualifier is
still required; the old normal-RL reader must not silently accept this new schema.

The separate qualifier is now implemented and included in queue012. It verifies
one new and two cumulative updates, Adam step2, finite update statistics,
checkpoint/batch ancestry and the unchanged BF16 evaluation contract.

Focused CPU tests cover actual Adam moments/step/RNG restore, new seed blocks,
and persistence before threshold/NaN rejection, alongside the existing native
gradient/checkpoint fixtures. All old seals, failed RL002 and FP16 probe receipts
remain unchanged. The new source/PLAN are prospective and no GPU is launched by
the preparing agent.

## How the outcomes should change the next experiment

- If FP16 still fails before an update, use the saved failing scores or gradients
  to isolate the arithmetic problem. Do not repeat the same rollout collection
  unchanged or raise the numerical tolerance simply to obtain a completed run.
- If the update completes without a task gain, distinguish insufficient update
  size from unhelpful reward assignment. A paired learning-rate comparison using
  the same saved FP16 on-policy batch could test update size without paying to
  collect that batch again. This is a candidate, not an accepted next job.
- If checkpoint2 improves, replicate and add an equally dosed supervised
  continuation starting from checkpoint1 before claiming an RL-specific benefit.
  The existing control starts earlier and cannot answer that question by itself.

The revised teaching procedure remains the stronger performance lead. A clean
comparison holding the one corrected training quantity and other data differences
fixed, followed by another training seed or recipe world, would be more useful
for a mechanism claim than simply accumulating scores on this exposed panel.

Downstream readiness amendment: `source-textcraft-fp16-cp2-readout-001` now supplies
the separate qualifier, unchanged BF16 fresh32 collector, and native cp2-minus-cp1
comparison. Four sealed focused tests and actual frozen-input validation pass.
The qualifier requires one **new** optimizer step, two cumulative/committed/Adam
steps, exact terminal boundary, finite update, pinned training PLAN and unchanged
base/LoRA ancestry. Flat-only, initial-copy, failed and partial endpoints reject.
`TEXTCRAFT-FP16-CONTINUATION-READOUT-PROPOSED-001.json` lists training (3h), readout
(1h), and CUDA-hidden native analysis (20min); main must bind the preceding queue
and accept. No readout PLAN is fabricated before an eligible cp2 actually exists.
