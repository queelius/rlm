# Diagnosing the SPO training failure and testing a corrected run

Status: October 12, 2026, 22:27 UTC. The ninth scheduled restart evaluation
is fully regraded at 1,296 updates. The native full checkpoint at 1,280 updates is
verified, and logged learning rates are independently checked through 768 updates.
The byte-verified 960-update copy remains protected outside automatic cleanup;
the scientific recipe and fixed endpoint are unchanged.
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

One recipe detail is now verified in the running optimizer: the learning-rate
schedule still spans **1,000 collections / 16,000 updates**, with 480 warmup
steps, even though our comparison stops at 690 collections. All 768 contiguous
audited update displays match that schedule at their displayed precision;
76 full-precision optimizer reports (152 parameter-group values) match too.
An independent formula check corroborates this. The initialization report is
checked separately from post-update values. This rules out an accidentally
shortened schedule in the audited interval; it does not establish the exact
historical author runtime or add a new learning result. See the
[learning-rate audit](evidence/restart-learning-rate-fidelity-20261012.json).

## Scheduled evaluations so far

The scheduled tests measure the model after **16, 176, 336, 496, 656, 816, 976, 1,136, and 1,296 optimizer updates**.
None is the untrained model or the prescribed final checkpoint.
Every saved answer was independently regraded, including original question
identities, prompts, tree and dataset agreement, and all seven native metrics.
Each evaluation's 33,024 answers across the three splits passed these checks.

| Completed updates | Test | Validation | Training subset (not held out) |
| ---: | ---: | ---: | ---: |
| 16 | 40.03% | 53.10% | 57.39% |
| 176 | 41.42% | 54.74% | 61.19% |
| 336 | 42.59% | 55.04% | 62.67% |
| 496 | 44.93% | 56.77% | 64.47% |
| 656 | 45.56% | 58.14% | 64.77% |
| 816 | 46.39% | 59.50% | 65.64% |
| 976 | 47.64% | 60.54% | 66.55% |
| 1,136 | 48.25% | 60.39% | 66.23% |
| 1,296 | 48.50% | 60.89% | 67.91% |

The latest test has **10,235 correct answers out of 21,104**, compared with 8,394
for the original SFT model: **+8.72 percentage points**. It gained 53 correct
answers, or 0.25 points, since the preceding scheduled test. Validation gained
30 correct answers (+0.50 points), and the training subset gained 100 (+1.68 points).
This is an early measured test improvement in our run, not a completed
reproduction or evidence that the token repair causes a lasting benefit.
The previous run also improved initially before
deteriorating. We retain the native decoding and mean single-answer accuracy;
solved-at-least-once coverage and majority voting do not replace that metric.

Test extraction failures fell from 16 to 12 and length-limit endings from 21
to 13. The gain of 53 correct answers is not explained solely by the net
reduction of four missing extractions; these diagnostics do not identify a
learning mechanism. Individual questions move both ways: 355 have more
correct samples, 333 fewer, and 631 the same count. This small checkpoint-to-
checkpoint increment is descriptive, not a statistical-significance claim.

**Secondary measures can move differently.** Test solved-at-least-once coverage
over 16 samples rebounded from 77.56% to 77.63%, and majority voting from 55.34%
to 55.80%. Both remain below the 976-update values of 78.47% and 56.86%, despite
higher mean accuracy. Validation and training-subset coverage and majority
voting also rose this time. The test's average number of distinct parsed answers
rose from 4.74 to 4.84. Neither diversity alone nor one improving score
establishes beneficial concentration or collapse.
We retain the prescribed mean single-answer metric and every scheduled result.
This is one training run, not independent replications, a statistical-significance
claim, or proof of the learning mechanism. See the
[full audit](evidence/restart-eval80-20261012.json) and
[interpretation](evidence/restart-eval80-interpretation-20261012.json).

At the same 1,296 updates, the earlier eventually failed run scored 46.61% test,
60.67% validation and 67.88% on the training subset. The fresh run is higher
by 1.89, 0.22 and 0.03 points, respectively. Its earlier all-answer audit and
unchanged native metric hashes were rechecked. Matching the update count is
useful context, but the earlier run had a resumed ancestry and other runtime
differences: this is not an isolated test of the BOS correction. Its later
collapse is a reason to keep the endpoint fixed, not declare recovery now.

Training completed 1,296 updates before this evaluation. The accompanying
batch audit, collection 80, contains 315 correct and 197 incorrect responses,
with mixed terminal outcomes on 35 of
64 questions. All 512 terminal rewards regrade correctly. Its training query
tokens match the actual generation requests, and saved actor and reference log
probabilities and advantages are finite. Ten groups with uniform terminal
outcomes still have nonzero intermediate advantages: SPO can supply a signal
within a solution even when the final rewards all agree. All first 1,296 update numbers are
present and logged numeric values are finite. These checks support continuing
the current run, not declaring learning success; the training batches contain
different questions and are not a benchmark progress curve.

The latest 160-update diagnostic window has a mean policy ratio of 1.0031,
approximate KL of 0.0635, clipping fraction of 0.0983, and gradient norm of 19.10.
There are no logged high-ratio guard warnings or NaN/Inf loss anomalies through
1,296 updates. These rounded aggregates are not a guarantee of future numerical
or learning stability. Their definitions and earlier windows are retained in the
[numerical audit](evidence/restart-training-windows1296-20261012.json).

The next decision remains the same: keep the recipe and endpoint fixed, inspect
subsequent scheduled evaluations (next after 1,456 updates), and preserve complete
recovery states. This incremental result stays in the supporting report. The
synchronized advisor slide, pdfpc notes and learning-guide status page retain
their 20:53 UTC cutoff: the main story, material limits and next comparison have
not changed. They keep the previous failure visible and the full reproduction
explicitly unfinished.

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

At 19:47 UTC, the scheduled **800-update full state** passed the same CPU
metadata check: trainer iteration 50/global step 800, actor and both active
Adam counters 800, scheduler 799, and zero skipped steps. All ten files were
stable; the protected 160-update copy retained its recorded sizes. This is
saved progress, not a tensor scan, new backup, GPU resume, or benchmark. See the
[800-update receipt](evidence/restart-full800-metadata-20261012.json).

At 20:39 UTC, the scheduled **960-update full state** passed the same CPU
metadata check: trainer iteration60/global960, actor and active Adam counters960,
scheduler959, skipped0. All ten files were stable and the protected160 copy's
sizes still matched. This is not a tensor scan, new backup or GPU resume. See
the [960-update receipt](evidence/restart-full960-metadata-20261012.json).

At 21:18 UTC, all ten files of that **960-update full state** were also copied
outside the trainer's automatic cleanup directory. Every source/destination
SHA256 matches, the source files remained stable during verification, and
trainer/actor/active Adam counters remain 960 with scheduler counter 959.
The 17.60 GB copy was selected by completed update count, not benchmark score;
the earlier protected 160-update copy is retained. See the
[protection receipt](evidence/restart-full960-protection-20261012.json).
This is same-filesystem recovery protection, not an off-system backup, a full
numerical tensor scan, or a new GPU restoration test. A separate faithful resume
entrypoint is still needed if interrupted. An initial CPU metadata check hit
the installed PyTorch loader's requirement for a string filename with memory
mapping; a failing/passing saved-state fixture verified that narrow correction,
and the complete hash/counter check then passed. Training was unaffected.

At 21:29 UTC, the next scheduled **1,120-update full state** passed the CPU
metadata check: trainer iteration 70/global step 1,120, actor and both active
Adam counters 1,120, scheduler 1,119, and no skipped steps. All ten files were
stable; protected 960-update file sizes still matched. This does not scan every
tensor, copy the new state or exercise a GPU restore. The new state remains
subject to native cleanup. See the
[1,120-update receipt](evidence/restart-full1120-metadata-20261012.json).

At 22:14 UTC, the **1,280-update full state** passed the same CPU metadata
check: trainer iteration 80/global step 1,280, actor and both active Adam
counters 1,280, scheduler 1,279, and no skipped steps. All ten files were stable.
See the [1,280-update receipt](evidence/restart-full1280-metadata-20261012.json).
No new protected copy or GPU restore was performed; protected states remain at
160 and 960 updates.

We also compared the same two layer-zero attention projections used in the
earlier saved-weight check. Their selection was recorded before the new
checkpoint existed. Between protected update 960 and native update 1,280,
1,440,143 of 4,194,304 query-projection elements and 2,207,966 of 4,194,304
output-projection elements changed. Both saved tensors are finite; the maximum
absolute difference is 0.00024414 in each. This checks actual stored weight
updates as well as counters. It does not show that these changes are beneficial,
scan all parameters or optimizer moments, or establish a new benchmark result.
The [selection](evidence/restart-parameter-update1280-selection-20261012.json)
and [CPU comparison](evidence/restart-parameter-update1280-20261012.json)
retain the fixed tensor identities and exact values. The same training owner
continues unchanged toward the prescribed endpoint.

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

At 20:28 UTC, a [native custom-state loading check](evidence/restart-recovery-custom-state-20261012.json)
used the saved 800-update files on CPUs. The real native loader replaced
deliberately incorrect trainer counters and running-score statistics with their
saved values. However, `FixedKLController.load_state_dict` deliberately does
nothing: an incorrect configured coefficient of 123 stayed 123 despite the
saved value of `0.0001`. Repeating the check with the original configured
coefficient of `0.0001` retained the intended value. Thus a faithful resume must
pin the original configuration, not infer all settings from successful checkpoint
loading. This did not expose a mismatch in the live run, whose configured and
saved coefficients agree; the native behavior is left unchanged.

This probe bypassed model/engine construction and initialized only a one-process
CPU context. It restored the small custom state and registered the deferred
engine-restoration path, not the model or optimizer on a GPU. Its initial
CPU fixture lacked the required Accelerate logging context; adding CPU
`PartialState` resolved that setup failure without changing training. The source
audit also found that the trainer ignores the engine loader's return value.
A future separately admitted resume must verify actual engine restoration and
actor/scheduler/Adam counters before further updates, as well as complete files
and the original recipe. No resume launcher or new GPU restore was performed.

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
