# Diagnosing the SPO training failure and testing a corrected run

Status: October 12, 2026, 19:11 UTC, after the fifth scheduled restart evaluation.
The native full checkpoint at 640 updates is also verified.
Our primary objective is to reproduce the
paper's **training result** on one A100. It is still unfinished. We have matched
the authors' released-model evaluation, diagnosed concrete discrepancies in our
training pipeline, and started a fresh run that tests the strongest one.

## The result we need to reproduce

The [SPO paper](https://arxiv.org/html/2505.23564v2) reports 56.7% on GSM8K for
SPO-chain with interval 5, compared with 40.3% for its supervised-fine-tuned
starting model. Our matched evaluations give 56.69% and 39.77%, respectively.
Those results show that we can evaluate the supplied models; they do not show
that our own training recreates the improvement.

Our previous training improved initially, then deteriorated. The latest
scheduled test was **22.19% after 7,856 optimizer updates**. The run failed
before reaching the prescribed 11,040-update endpoint. The failure was not an
allocation timeout, and an earlier favorable checkpoint cannot replace it.

## What changed our explanation

We independently regraded 66,048 saved answers from two scheduled checkpoints
across test, validation, and the training subset. Every checked original
question, saved response, and native metric agrees. The test lost 5,978 correct
responses. Extraction failures rose only from 14 to 18, and length-limit endings
from 16 to 19. The large decline is therefore not explained by those bookkeeping
or formatting issues. The 16 samples per question are not independent training
replications.

A more consequential discrepancy appeared in the input tokens. In all 2,560
examined saved training responses, the generation engine saw a beginning-of-sequence
token that the training calculation omitted. In plain language, the model was
asked to generate an answer under one input, then assigned probabilities for
that answer under a slightly different input.

We tested identical saved responses both ways on the A100. On the first 16
distinct saved questions, chosen without filtering by correctness, adding this
token barely changed the original SFT model's total response log-probability:
mean −0.10 nats. For the late model it changed the mean by +53.98 nats, with the
same direction on all 16 questions. A separate check with both Hugging Face and
vLLM corroborated the effect. **These are likelihood diagnostics, not accuracy
improvements or proof of the cause of the collapse.**

The immediate crash was a separate issue: PPO can reject a high-ratio minibatch
and produce zero loss, while the installed BF16 optimizer asserts that the
gradient norm must be strictly positive. A focused fixture reproduced this
boundary. An isolated adapter accepts finite zero while still rejecting the
nonfinite sentinel and preserving the optimizer step. A real A100 test restored
all 3.30 billion saved FP32 master/moment elements exactly and exercised that
zero-step path. It saved no modified model. The historical failing norm itself
was not logged, so the fixture establishes a mechanism, not its exact value.

## The new comparison

The fresh run started October 12 at 15:22:49 UTC from the original RhoMath 1.1B
SFT weights, not a favorable intermediate checkpoint. It preserves the author
data splits, seed 42, 64 questions with 8 responses each, 9 Monte Carlo
continuations, two training epochs per collection, learning rate 1e-6, and the
1,000-iteration learning-rate horizon. The comparison endpoint remains
**690 collections / 11,040 optimizer updates**.

The private adapter adds the configured beginning token before training
likelihoods, intermediate reward calculations, and PPO. End-of-sequence
behavior is unchanged. The separate zero-norm adapter only addresses the
verified optimizer boundary. Seven focused CPU tests and independent reviews
passed; the first live collection then generated 512 responses and completed
16 finite optimizer updates. Its completed scheduled evaluation is described below.

At the fixed endpoint, a small hook verifies the already-saved full model and
optimizer metadata and invokes the native test without adding an optimizer
update. We retain the native 1,319 questions × 16 responses, temperature 0.35,
effective top-p 1.0, and **mean pass@1**, not best-of-16. Every scheduled result
will remain visible. A 68-hour whole-run cap includes evaluation; full optimizer
checkpoints are kept for an allocation handoff if needed.

## Scheduled evaluations so far

The scheduled tests measure the model after **16, 176, 336, 496, and 656 optimizer updates**.
None is the untrained model or the prescribed final checkpoint.
Every saved answer was independently regraded, including original question
identities, prompts, tree and dataset agreement, and all seven native metrics.
Each evaluation's 33,024 answers across the three splits passed these checks.

| Split | After 16 updates | After 176 updates | After 336 updates | After 496 updates | After 656 updates |
| --- | ---: | ---: | ---: | ---: | ---: |
| Test | 40.03% | 41.42% | 42.59% | 44.93% | 45.56% |
| Validation | 53.10% | 54.74% | 55.04% | 56.77% | 58.14% |
| Training subset (not held out) | 57.39% | 61.19% | 62.67% | 64.47% | 64.77% |

The latest test has **9,615 correct answers out of 21,104**, compared with 8,394
for the original SFT model: **+5.79 percentage points**. It gained 133 correct
answers, or 0.63 points, since the preceding scheduled test. Validation rose
1.37 points in that interval. This is another early measured
improvement in our run, not a completed reproduction or evidence that the token
repair causes a lasting benefit. The previous run also improved initially before
deteriorating. We retain the native decoding and mean single-answer accuracy;
solved-at-least-once coverage and majority voting do not replace that metric.

Test extraction failures fell from 53 to 30 and length-limit endings from 54
to 37. These diagnostics accompany the correctness result; they do not identify
its learning mechanism. Individual questions move both ways: 406 have more
correct samples, 340 fewer, and 573 the same count. Even the first original
test question fell from 10/16 to 6/16 correct despite the overall gain.

**Higher single-answer accuracy is not improvement on every measure.** Test
solved-at-least-once coverage over 16 samples fell from 76.95% to 76.50%;
majority voting fell from 54.06% to 53.68%. The average number of distinct parsed
answers also fell, from 5.65 to 5.44. Validation coverage and majority voting
edged down too. This is consistent with changing how probability is distributed
among answers, but does not establish that mechanism or a loss of underlying
capability. We retain the prescribed mean single-answer metric and report the
other measures alongside it. This is one training run, not independent
replications or a statistical-significance claim. See the
[full audit](evidence/restart-eval40-20261012.json) and
[interpretation](evidence/restart-eval40-interpretation-20261012.json).

Training completed 656 updates before this evaluation. The accompanying
batch audit, collection 40, contains 328 correct and 184 incorrect responses,
with mixed terminal outcomes on 31 of
64 questions. All 512 terminal rewards regrade correctly. Its training query
tokens match the actual generation requests, and saved actor and reference log
probabilities and advantages are finite. Fifteen groups with uniform terminal
outcomes still have nonzero intermediate advantages: SPO can supply a signal
within a solution even when the final rewards all agree. All first 656 update numbers are
present and logged numeric values are finite. These checks support continuing
the current run, not declaring learning success; the training batches contain
different questions and are not a benchmark progress curve.

The next decision remains the same: keep the recipe and endpoint fixed, inspect
subsequent scheduled evaluations (next after 816 updates), and preserve complete
recovery states. This
early result stays in the supporting report; it does not change the advisor
deck's main conclusion at its explicitly dated 15:28 UTC cutoff.

## A protected recovery point

The first full checkpoint finished saving at 16:18 UTC after 160 updates.
Saved trainer, actor, optimizer, and learning-rate counters agree. All saved
FP32 master parameters and Adam moments are finite. We copied all ten files
(17.60 GB) outside native checkpoint cleanup, then verified every source and
destination SHA256. Training continued under the same owner throughout.

This is operational protection, not an accuracy result. The copy is on the
same home filesystem, so it is not a disaster backup. The live launcher is
fresh-start-only: an interruption would still require a separately prepared
resume entrypoint. We inspected this state on CPUs without displacing training;
we have not claimed a new GPU restoration test of it. The earlier full-state
restoration test used the previous run's 7,840-update checkpoint.

The next native full checkpoint completed at 320 updates. Its trainer, actor,
and scheduler counters agreed, and all ten required files were present at that check. This
[metadata check](evidence/restart-full320-metadata-20261012.json) did not scan
every tensor or restore it on the GPU.

The following full checkpoint completed at 480 updates. Trainer and actor counters
are 480, the scheduler counter is 479, and both active Adam state counters are
480. All ten files were stable during CPU inspection. This
[metadata check](evidence/restart-full480-metadata-20261012.json) did not scan
every tensor, copy the checkpoint, or restore it on the GPU. Native checkpoints
remain subject to automatic cleanup; the separate 160-update copy remains protected.
An initial inspection mistakenly treated two legacy optimizer group counters
as active counters; explicit state-path inspection and the installed optimizer
source resolved this. The inspection failure is preserved in the receipt and
was not a training failure.

At 18:57 UTC, the next scheduled full checkpoint was verified at **640 updates**:
trainer iteration 40/global step 640, actor step 640, scheduler counter 639,
and both active Adam state counters 640. No optimizer steps were skipped.
All ten required files were present and stable during the CPU check; the
protected 160-update copy also retained its recorded file sizes. All first
640 logged updates and numeric values are finite, with no logged traceback.
The same authenticated owner continued generating collection 40 with no HTTP
errors at the 18:58 cutoff. See the
[640-update metadata receipt](evidence/restart-full640-metadata-20261012.json).
This did not scan every tensor, copy the new checkpoint, or test a GPU resume.
Native cleanup still applies, and a separate faithful resume entrypoint remains
necessary if interrupted. That checkpoint check itself is not a benchmark;
the subsequent 656-update evaluation is reported above. The fixed endpoint
remains 11,040 updates.

A CPU recovery audit at 16:54 UTC independently reconstructed the original
question order for all 14 checked collections (896 question appearances).
This recipe selects each collection with seed `42 + iteration`; it does not
depend on an accumulating question-sampler cache. Starting from the protected
checkpoint's iteration counter reconstructs the same next 64 questions. A
focused fixture also ran the native minibatch sampler on 512 synthetic episode
IDs: both epoch orders matched after perturbing the process RNG. These checks
reduce data-order uncertainty for a future continuation, but do not demonstrate
an end-to-end resumed GPU run or bitwise-identical generation. No resume was
launched and the live training code was untouched. See the
[recovery data-order audit](evidence/restart-recovery-data-order-20261012.json).

A further [CPU recovery check](evidence/restart-recovery-selection-20261012.json)
found that the native "resumable" test can accept a directory containing only
an actor marker, with no model or optimizer state. Our stronger handoff gate
requires complete files and matching counters, not that flag alone. The real
320-update checkpoint passed the recorded file checks at that audit. The incomplete example
was a tiny test fixture, not a live training failure; no resume was launched.

## What this can and cannot establish

If the run recovers the published improvement, that supports a practical
reproduction under the documented implementation. It does not establish
bit-for-bit historical equivalence: the authors' original training predates the
public source, which already contains the omitted token operations. Their exact
source revision and checkpoint-selection rationale remain unresolved.

The old run also had an interruption, whereas this run starts fresh and includes
the optimizer compatibility repair. A better result would therefore support the
overall corrected pipeline, not isolate a causal effect of one token by itself.
If the result stays poor, we will examine the remaining EOS and historical-recipe
differences rather than quietly change the target or claim success.

The lesson for later RLM training is concrete: having rewards is necessary but
not sufficient. The trajectories we reward must also match the inputs and
probabilities used to update the model. That is a testable pipeline requirement,
not evidence yet that an RLM teaching intervention works.

## Evidence and continuation

- [Portable diagnostic summary](evidence/restart-20261012.json)
- [Independent late regrade](evidence/late-regrade-20261012.json)
- [Sixteen-question paired likelihood comparison](evidence/bos-breadth-20261012.json)
- [Frozen source and configuration manifest](evidence/bos-preparation-20261012.json)
- [First collection audit](evidence/startup-audit-20261012.json), checked after
  the 15:28 cutoff: all 512 training queries match generation token IDs;
  16 updates completed. Training rewards are not benchmark gains.
- [First scheduled evaluation audit](evidence/restart-eval0-20261012.json):
  all three splits, 33,024 saved answers, no scoring discrepancies.
- [Second scheduled evaluation audit](evidence/restart-eval10-20261012.json)
  and [interpretation](evidence/restart-eval10-interpretation-20261012.json):
  another 33,024 answers checked; early improvement, not the fixed endpoint.
- [Third scheduled evaluation audit](evidence/restart-eval20-20261012.json)
  and [interpretation](evidence/restart-eval20-interpretation-20261012.json):
  all 33,024 answers and 21 metric comparisons agree; test gain was 2.81
  points over the matched starting model at 336 updates.
- [Fourth scheduled evaluation audit](evidence/restart-eval30-20261012.json)
  and [interpretation](evidence/restart-eval30-interpretation-20261012.json):
  all 33,024 answers and 21 metric comparisons agree; test gain is now 5.16
  points at 496 updates, still early in the fixed run. Before executing the
  audit, a stale descriptive label was corrected from evaluation20 to
  evaluation30; the actual data bindings and scoring were already correct.
  The original preparation and regression evidence are preserved externally.
- [Research review and next decision](evidence/restart-review-20261012.json).
- [Full 160-update checkpoint protection](evidence/restart-full160-protection-20261012.json)
  and [second review](evidence/restart-review160-20261012.json).
- [Experimental adapters and tests](../../experiments/spo_reproduction_20261008/)

The live handoff is
`/project/alex_phd/runs/rl-paper-reproduction-20261012/SESSION_CHECKPOINT.md`,
with decisions in `QUEUE.json`. New weights and native outputs are in
`/home/atowell/research-runs/rl-paper-reproduction-20261012/spo-bos-only-from-sft-attempt1/native`.
The owner covers training and final evaluation; no competing GPU job may start
until all its processes leave. Previous failed artifacts remain untouched.
GitHub contains source and reports, not model-weight backups.
