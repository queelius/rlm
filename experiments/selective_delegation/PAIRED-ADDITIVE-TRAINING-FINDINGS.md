# Additive reward changes the update, but does not establish better paired answering

Source051 finished cleanly on2026-09-22 at07:14:14 UTC: all eight sampled blocks
committed eight real optimizer updates, with1,024 valid native calls and512 valid
paired candidates. No protocol failures, unavailable calls, missing pairs,
unresolved starts or unsampled slots. The owner released before the independent
sealed CPU audit ran. No active source or GPU queue was changed.

## Actual learning signal, not the earlier offline projection

The primary scientific score remains official paired exact correctness `P*N`.
The optimized additive scalar is `(P+N)/2`: `P` requires the positive label and
exact answer; `N` requires the negative refusal label. They are not interchangeable.

| Block | Paired successes /64 | Additive reward sum /64 | Additive-active groups /16 | Gradient norm before clip1 | Actual LoRA delta L2 |
|---|---:|---:|---:|---:|---:|
| 1 | 4 | 24.5 | 10 | 1.8190 | .080894 |
| 2 | 12 | 32.5 | 11 | 3.1988 | .059558 |
| 3 | 5 | 32.0 | 6 | 1.1362 | .052154 |
| 4 | 4 | 32.5 | 6 | 1.0962 | .046325 |
| 5 | 9 | 36.5 | 6 | 2.4674 | .037254 |
| 6 | 4 | 33.0 | 4 | 1.2336 | .037902 |
| 7 | 4 | 32.0 | 6 | 1.7602 | .037221 |
| 8 | 6 | 33.5 | 5 | 9.4547 | .033323 |

These are changing TRAIN question blocks, not a fixed-question learning curve.
Actual additive credit covered54/128 parent groups,412 responses and4,767
emitted tokens. Twenty-one active groups had no positive exact success.
On these new trajectories the product reward alone would vary in29 groups.
The earlier **98/128** additive-active figure was an offline regrading of the
old product-policy trajectories, not a prediction or observation of this run.

| Whole-run on-policy TRAIN quantity | Original product037 | Additive051 |
|---|---:|---:|
| Official paired successes /512 | 78 | 48 |
| Positive exact successes /512 | 127 | 61 |
| Correct negative labels /512 | 340 | 452 |
| Positive abstentions /512 | 228 | 375 |
| Negative overanswers /512 | 171 | 60 |
| Malformed positive /negative responses | 1 /1 | 0 /0 |
| Actual objective-active groups /128 | 42 | 54 |
| Credited emitted tokens | 4,165 | 4,767 |

Additive051 accumulated256.5/512 additive reward, but only48/512 paired exact
successes. Its on-policy responses are substantially more conservative than the
product run. This is consistent with the prospective concern that additive
credit could favor easy negative-label accuracy while positive competence stays
weak. It is **not evidence of held-set harm**, nor proof of an improvement or
decline on a fixed TRAIN panel: policies and later sampled responses differ.

## Same first samples, materially different update direction

The first128 native calls match exactly between037 and051 in prompt, model,
seed, sampling, input IDs, output IDs **and generation token log-probabilities**.
All504 initial LoRA tensors are identical. Thus this first update really is a
same-warmstart, same-sample objective contrast, not merely matched labels.

The actual checkpoint0-to1 update norms are .08082924(product) and
.08089355(additive), but their cosine is only **.07239**;53.61% of coordinates
nonzero in both updates share a sign. The objective change materially changed
the optimizer's direction at almost the same norm. This is not quality evidence,
and does not establish a gradient-noise or convergence theorem. The small
keyed-tensor/native-receipt comparison took0.62 CPU seconds; no base weights or
GPU inference were needed.

## Audit, numerical limitation and cost

The independent audit regraded native responses, reconstructed additive RLOO
advantages and the all64-pair emitted-token loss, verified checkpoint/PLAN
identities, measured every adjacent saved LoRA delta, and checked the actual
Adam state steps1 through8 across504 parameter states. All16,515,072 saved
adapter parameters remain in the same LoRA-only tensor inventory. None of the
eight blocks was a zero-Adam skip.

Advantage-weighted post-update log-probability movement is positive for blocks1–7
but **−.0860 on block8**, whose pre-clipping gradient norm is9.4547. Do not replace
this with a claim that every update improved its sampled objective. Adam momentum,
clipping and the nonlinear update need not improve every fixed sampled surrogate.
The audit found no token/advantage alignment defect.

BF16 cached-generation versus full-replay token gaps are nonzero: maximum
absolute token gap .26488; credited sequence-sum signed gaps range
−.33728 to+.32008. This is approximate replay, not exact numerical equivalence.
Full per-block token-gap summaries and per-response sequence gaps are retained.

Cost:1,024 calls,2,896,748 input tokens,11,120 output tokens,1,086.35 native-service
seconds and1,648.22 owner-wall seconds. Original037 cost the same call/input
count,12,387 output tokens and1,627.75 wall seconds. Equal updates/calls are not
equal credited tokens, reward distribution, information or FLOPs.

The endpoint stays the predeclared checkpoint8. No tuning, extra update or
checkpoint substitution is justified by these TRAIN diagnostics. The primary
four-arm readout was pending at this training-audit cutoff: frozen050's32 parents/29 component clusters,
warm joint32 versus original productRL8, original matched extraSFT8 and additiveRL8.
`PAIRED-REWARD-READOUT-PLAN.md` specifies that comparison. Interpret paired EM,
positive answering and negative overanswer together; do not infer its result here.

Subsequent update: the [held readout is complete](PAIRED-REWARD-CONTROL-FINDINGS.md).
It gives6/10/10/4 correct pairs out of64 for warm/product-RL/extra-SFT/additive-RL.
That separate fixed-policy evaluation, not the changing TRAIN counts above,
provides the evidence that this additive continuation underperformed product RL.

## Durable evidence

R denotes `/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.

- Primary audit: `R/analysis-sufficiency-additive-training-001.json/.md`, JSON SHA256
  `f65463706fe15a8a5b8b73cb9edf873ff9f07682bb5322383f3304f7b71dfd01`.
- Same-first-update receipt: `R/analysis-sufficiency-first-update-objective-001.json`.
- Native run/terminal endpoint: `R/sufficiency-additive-rl-001/`,
  `boundaries/sample-0008/checkpoint-0008`.
- Analyzer: `R/analysis-source-sufficiency-additive-training-001/`, script SHA256
  `92bd01d0c4980209bec37d32fcc08a6580f2bda5c712b1dc771deb84009b07bd`;
  five focused sealed tests passed. Actual037 Adam-state schema smoke also passed.
- Original comparison audit: `R/analysis-sufficiency-rl-training-001.json`;
  marginal counts independently retained in `R/analysis-paired-reward-pairing-002/REPORT.json`.

The sealed terminal audit completed without a repair or overwritten report.
Its import path is the immutable003 analyzer dependencies, then source051 and
source053 (the latter supplies the existing cost/helper-analysis dependencies).
