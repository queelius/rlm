# Conditional next step after a zero-advantage batch

Design only, 2026-09-21. No implementation or GPU run accepted here. At this
inspection neither `fresh-contract-rl21-001` nor `training-execution-credit-001`
had a terminal receipt; no partial outcomes were read. Revisit the decision below
after their completed, official analyses. The fresh003 panel is now reused
development evidence, not an untouched test.

## What actually stopped

`rl-continuation-001/batch-0022` has 64/64 scored finals: twelve parents have
four correct candidates and four have four incorrect candidates (48/64 EM).
Every root plan is valid; each group contains 2–4 distinct question lists.
All 64 RLOO advantages are exactly zero. This is not a transport failure,
malformed-output artifact, or evidence that distinct strings imply useful choices.

Last committed step21 has `cursor=0,next_update=22`; the loop uses one `update`
for parent schedule, RNG seeds, batch directory, and optimizer checkpoint. It
breaks on failed admission and creates batch directories with `exist_ok=False`.
Thus ordinary resume cannot advance past saved batch22, and incrementing the
checkpoint number without an optimizer step would misrepresent training.
Admission additionally requires two groups with distinct valid plans and mixed
rewards. The existing loss is summed root-token log probability times
leave-other-three-out advantage, divided by **all64** candidates; it is not GRPO.

## Smallest terminal-reward option: bounded cursor separation

Only if the completed step21 readout makes additional terminal-reward learning
worth testing, fork its exact committed adapter/Adam/RNG into a new output:

- Maintain `optimizer_step=21` separately from `sample_batch_cursor=22`.
  Authenticate batch22 as already consumed under checkpoint21, preserving its
  receipts and zero result. New sample cursors23 and24 use exactly the existing
  second-shuffle parent blocks and original seed formula with **sample cursor**.
  No reseeding batch22 to hunt a favorable draw; no new parent selection by score.
- Hard cap **two new blocks ×16 parents ×4 candidates =128 trajectories**,
  at most1,280 calls, at most two updates, and30 minutes cumulative. A64-trajectory
  version is a separately predeclared one-block screen, not a posthoc cap choice.
  Two admitted blocks would end at optimizer step23, **not24**.
- Freeze weights throughout each block. Preserve current rootT=.8,
  helper/finalT=.5, caps, helper36, base final, LR2e-5, strict scores and one-pass
  loss. Uniform groups remain in the denominator and acquisition accounting;
  skipping their zero backward terms is already exact in this implementation.
- If the entire block has zero advantages, record `sampled_no_signal`, advance
  only the sampling cursor, and do **not** call Adam: its existing moments could
  move weights even with a zero gradient. If the block has nonzero advantages
  but fails the existing admission gate, stop and report that distinct reason;
  do not silently discard its nonzero gradient or refill until it passes.
- On admission, use all64 candidates/denominator64 and one optimizer step;
  commit adapter/Adam/RNG plus both cursors and exact contributing batch hash.
  Save a small immutable sampling-boundary receipt after zero blocks, linked to
  the unchanged checkpoint and parent schedule; include current RNG snapshot,
  physical cost and completed batch hashes. Resume resolves the latest committed
  sampling boundary, not `step+1`. A partial/error block remains unresolved and
  unavailable, never reward0 or an automatically replaced block.

This avoids *new* outcome-conditioned refill/renormalization; the existing
admission rule remains a selection rule, so do not claim an unbiased end-to-end
training procedure. Stop at the fixed acquisition/time cap regardless of how
many updates it yields. Compare the terminal endpoint with saved step21 under
identical evaluation requests; select no intermediate checkpoint by dev score.
This tests whether usable signal exists beyond one uniform block, not whether a
new sampling algorithm beats an equal-compute training control.

## Why not simply call this DAPO?

[DAPO v1 §3.2 and Algorithm1](https://arxiv.org/html/2503.14476v1) explicitly
oversample and retain mixed-reward groups until the effective batch is filled.
The [official project](https://dapo-sia.github.io/) links its implementation.
That mechanism is established prior art. Its clipped, standardized-advantage,
token-normalized objective is not our unstandardized RLOO sum-logp objective.

A genuine bounded refill variant could acquire at most32 groups/128 candidates
under **one unchanged policy**, retain up to16 mixed groups, and refuse further
sampling if the reservoir is unfilled. But retention depends on observed reward;
normalizing by survivors changes the sampled distribution/effective weighting
and is not justified by saying “discarded advantages were zero.” It also risks
spending the whole cap without an update. Prefer the cursor-only screen first.
If later testing refill, declare a new estimator, retain every acquisition and
protocol/transport category, and compare equal rollout budgets—not update counts.

## Different question: learn whether to execute a fixed plan

For a **frozen** generated plan x, define action utility as
`U(execute)=E−lambda*C_execute`, `U(skip)=P−lambda*C_skip`; then
`Delta=E−P−lambda*(C_execute−C_skip)` is the paired action-value target.
The root generator and both answer paths stay fixed. A gate sees only the public
question, document index, generated plan and prospective cost features before
execution—not answers, outcomes, helper reports, or host annotations. Compare
always-skip, always-execute, a prespecified plan-length rule, and a small learned
gate on absolute accuracy versus actual native token/call cost. Fix lambda before
evaluation (or report a prespecified frontier, without picking its winner on test).

Do not train this gate on16 diagnostic parents and call repeated seeds independent
examples. If credit results show repeatable *both signs* of execution benefit,
first acquire64 additional component-separated TRAIN parents, one frozen plan
each, both actions, two common downstream seeds. At most64 root +128 skip finals
+128×(8 helper+1 final)=1,344 calls; one A10040GB, ≤30 minutes acquisition cap.
Use component-held-out TRAIN cross-validation for a cheap logistic gate before
considering LoRA. A full-information gate objective can directly maximize
`p_theta(execute|x)*Delta`; there is no need to pretend this is root-planner RLOO.
An additional binary gate call must be charged if that is the deployed interface.

Applying E−P to **train the plan generator** is a different objective: it can
favor a worse absolute answerer with larger relative helper benefit. It is not
an unbiased baseline subtraction. Do not conflate the gate and planner experiments.

## Conditional decision

If step21 improves both-valid development answers credibly enough to justify a
small dose probe and E/P credit mostly agrees, the128-rollout cursor-only screen
is the smallest continuation. If step21 is flat and execution changes few outcomes,
retire more terminal-only updates on this panel; cheap plan-only execution or an
information-constrained task is more informative. If E−P has stable useful/harmful
variation and terminal credit often disagrees, prioritize the bounded gate screen
over changing root reward. If signs vary mostly across seeds/protocol failures,
do neither yet: report noisy/contract-mediated credit rather than manufacture
labels. These are decisions about exploratory value, not p<.05 promotion gates.

Source receipts (R is the September21 selective-delegation store):

- `R/rl-continuation-001/checkpoint-0021/STATE.json` SHA256
  `f39ba1305b492766e11d951578acaa0980c568e249a0c5c92a6f555892afe5ae`.
- Same checkpoint `COMMIT.json` SHA256
  `703b30fd49b7d5bf954caa08be788a84139178aacf25f98cc07bf7e55efc5fb8`.
- `R/rl-continuation-001/batch-0022/BATCH.json` SHA256
  `fe660f1eeafdeb702816c6aa1de1f1ac5bf953ca5e4d67a55742696a8da9fb6f`.
- `R/rl-continuation-001/PLAN.json` SHA256
  `0365da267c0d9e371ed88e19b786a4e09bd4eab0684650dc353939cbfceb6c3b`.

Inspected actual `rl_planner.py` loss, diagnostics, loop, checkpoint and resume
logic; no weight rehash, model load, training change or downloaded code execution.
