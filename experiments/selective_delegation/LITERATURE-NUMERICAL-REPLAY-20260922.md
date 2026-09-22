---
status: diagnostic_motivation_not_causal_finding
retrieved_utc: 2026-09-22T13:38:00Z
question: Why do generation and replay assign different probabilities to identical saved actions?
related_run: textcraft-terminal-rl-002
---

# A focused numerical check before more RL

## What happened here

The trainer completed one optimizer update, then collected a second batch of
32 attempts. Before updating again, it compared saved generation-time token
log probabilities with a full-forward recomputation under the same weights.
That check failed its predeclared maximum-or-mean tolerance. The old code
raised before saving the numerical gaps, so **we do not yet know their size
or which threshold failed**. No second update occurred. This is not a measured
decline in task performance.

The existing first-step check compared training and evaluation forwards and
matched exactly. That is a different comparison from cached generation versus
full-forward replay. Only one fresh generation request qualified the reused
first batch; the second batch tests many more requests and token positions.
Consequently, this failure does not by itself show that the update caused the
discrepancy to grow.

## Two relevant primary papers

[Qi et al., *Defeating the Training-Inference Mismatch via FP16*, v1](https://arxiv.org/html/2510.26788v1)
reports substantial BF16 discrepancies between rollout and training probability
calculations. The paper discusses autoregressive generation versus parallel
training as one source of numerical differences and finds that FP16 reduces
them in its experiments, including LoRA experiments. Its
[official implementation](https://github.com/sail-sg/Precision-RL) is a possible
reference if our diagnostic identifies this mechanism. We have read the paper,
not inspected or executed that repository. This evidence motivates a precision
comparison; it does not establish our failure's cause or guarantee a fix.

[Zhang et al., *Beyond Precision*, v1](https://arxiv.org/html/2602.01826v1)
finds that mismatch can also grow with optimization dynamics during longer
RL runs. In its experiments FP16 alone does not prevent eventual instability;
a response-length-informed learning-rate schedule helps. Our one-update run
does not establish that phenomenon. This is a caution against treating a
precision change as a universal solution, not a reason to change our learning
rate before diagnosing the observed failure.

## Smallest informative follow-up

Keep the committed first checkpoint, saved input/output token IDs, temperature,
model configuration and numeric types fixed. Compare recorded generation
probabilities with both a full-forward pass and a cached, token-by-token pass
over the **same saved tokens**. Select requests without looking at rewards.
Save token-level differences, quantiles, maxima and exact offending positions
before applying any acceptance rule. No optimizer step or new task rollout is
needed for this comparison.

If cached replay agrees with generation but full replay does not, that supports
a numerical-path explanation. If neither agrees, investigate loading, captured
scores, model mode and generation configuration before changing precision.
Any later FP16 comparison must compare FP16 cached and full calculations with
each other—not demand that they reproduce BF16 probabilities. A restored RL
run would need explicit provenance and requalification, not a silent tolerance
increase or relabeling of the failed endpoint.

The base-model fresh-task comparison runs independently while this diagnostic
is prepared. The numerical issue stays in the supporting research record; it
does not replace the substantive demonstration-transfer result or become a
claim of a new RL method.
