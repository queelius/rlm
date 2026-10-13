# Diagnosing the SPO training failure and testing a corrected run

Status: October 13, 2026, 03:15 UTC. The fifteenth scheduled restart evaluation
is fully regraded at 2,256 updates. The complete 1,440-update checkpoint is
byte-verified outside automatic cleanup, alongside the earlier 160- and
960-update copies. A further CPU scan finds all saved model parameters,
FP32 master weights and Adam moments finite at 1,440. Logged learning rates are
independently checked through 768 updates;
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

The scheduled tests measure the model after **16, 176, 336, 496, 656, 816, 976, 1,136, 1,296, 1,456, 1,616, 1,776, 1,936, 2,096, and 2,256 optimizer updates**.
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
| 1,456 | 48.86% | 62.55% | 69.47% |
| 1,616 | 49.39% | 62.35% | 68.16% |
| 1,776 | 49.88% | 63.64% | 69.14% |
| 1,936 | 50.13% | 63.22% | 70.50% |
| 2,096 | 51.05% | 65.03% | 70.48% |
| 2,256 | 51.37% | 66.86% | 71.72% |

The latest test has **10,842 correct answers out of 21,104**, compared with 8,394
for the original SFT model: **+11.60 percentage points**. It gained 69 correct
answers, or 0.33 points, since the preceding scheduled test. Validation gained
109 correct answers (+1.83 points), and the training subset gained 74 (+1.24 points).
This is measured intermediate improvement in our run, not a completed
reproduction or evidence that the token repair causes a lasting benefit.
The previous run also improved initially before deteriorating. We retain the
native decoding and mean single-answer accuracy; solved-at-least-once coverage
and majority voting do not replace that metric.

Test extraction failures rose from 11 to 12 while length-limit endings fell from
15 to 12, alongside 69 more correct answers. Validation extraction failures and
length-limit endings both fell from three to zero. These aggregates are
not a causal decomposition. Individual questions move both ways: 354 have more
correct samples, 350 fewer, and 615 the same count.
For the predetermined first test question, the model gets 12 of 16 answers right,
up from 9. Its first response accounts for both the three breakfast eggs and
four baking eggs, returning (16 − 3 − 4) × 2 = 18. It still misses 4 of 16
attempts on that question. This illustration is not a representative sample,
a separate benchmark, or a statistical-significance claim.

**Secondary measures do not replace the paper's metric.** Test solved-at-least-once
coverage over 16 samples slipped from 79.53% to 79.38%, while majority voting rose
from 58.68% to 59.06%. Validation coverage also fell, from 89.81% to 89.54%,
while majority voting rose from 72.65% to 73.73%. Training-subset coverage is
unchanged and majority voting rises. Better average correctness therefore does
not establish broader sampled problem coverage. Test distinct parsed answers per
question fell from 4.606 to 4.487. Neither diversity nor an improving
score alone establishes beneficial concentration or lasting stability. We retain
every metric and scheduled result. This is one training run, not independent
replications or proof of a learning mechanism. See the
[full audit](evidence/restart-eval140-20261013.json) and
[interpretation](evidence/restart-eval140-interpretation-20261013.json).
The [preceding evaluation](evidence/restart-eval130-20261013.json) remains in the record.

At the same 2,256 updates, the earlier eventually failed run scored 48.54% test,
60.87% validation and 69.25% on the training subset. The fresh run is higher by
2.83, 5.98 and 2.47 points, respectively. The earlier all-answer audit and
unchanged native metric hashes were rechecked. Matching the update count is
useful context, but the earlier run had a resumed ancestry and other runtime
differences: this is not an isolated test of the BOS correction. Its later
collapse is a reason to keep the endpoint fixed, not declare recovery now.

Training completed 2,256 updates before this evaluation. The accompanying
batch audit, collection 140, contains 368 correct and 143 incorrect responses,
plus one unfinished response penalized at −1. There are mixed terminal outcomes
on 25 of 64 questions. All 512 terminal rewards
regrade correctly. Training query tokens match actual generation requests, and
saved actor/reference log probabilities and advantages are finite. Eleven groups
with uniform terminal outcomes still have nonzero intermediate advantages:
SPO can supply a signal within a solution even when the final rewards all agree.
All first 2,256 update numbers are present and logged numeric values are finite.
Changing training questions are not a benchmark progress curve.

**A correct outcome need not imply correct arithmetic.** The first rewarded
response in the preceding audited batch, collection 130, ends with the gold
answer, $200, but omits the remaining $40 and incorrectly evaluates $20 / 0.30
as $200. The correct equation
is `0.4 × 0.75 × initial_money − 20 = 40`. The native reward of one is correct
under the paper's outcome rule; it does not certify the rationale. This is one
illustrative response, not a prevalence estimate or evidence of reward hacking.
We preserve the paper's grader. The
[previous interpretation](evidence/restart-eval130-interpretation-20261013.json)
records its exact selection and external evidence pointer.

The first penalized response in collection 140 illustrates a different failure:
asked for the number of fish after five years, it keeps adding fish for far more
years and reaches the 1,024-token limit without a final answer. The native −1
unfinished penalty agrees. This is one example (one of 512 responses in this
batch), not a new evaluation metric or a causal account of training progress.

The latest 160-update diagnostic window has a mean policy ratio of 1.0032,
approximate KL of 0.0631, clipping fraction of 0.0948, and gradient norm of 18.49.
Its largest approximate KL is 0.355 at update 2,159, below the previous window's
0.736 at update 2,047; the older 0.523 excursion at update 1,712 also remains in
the record. The latest peak falls to 0.207 on the next update. We retain these
excursions as watchpoints; improved test scores do not
establish that the numerical concern is resolved.
There are no logged high-ratio guard warnings or NaN/Inf loss anomalies through
2,256 updates. These rounded changing-batch aggregates are not exact
distributional KL measurements or a guarantee of future stability.
Earlier windows and definitions remain in the
[numerical audit](evidence/restart-training-windows2256-20261013.json).

The scheduled full checkpoint at 2,240 updates has all ten required files;
trainer, actor and active Adam counters agree at 2,240, scheduler at 2,239,
with no skipped steps. This is a
[CPU metadata check](evidence/restart-full2240-metadata-20261013.json),
not a new tensor scan, protected copy or GPU restore. File identities were
unchanged across this check. Native cleanup rotated the older 2,080-update full
state; its metadata receipt remains historical. The separately protected
1,440-update recovery state remains available.

**Saved-weight check, October 13, 03:35 UTC:** the same two layer-zero projection
tensors used in earlier audits contain 1,854,493 and 2,630,490 changed BF16
elements, respectively, out of 4,194,304 each, between protected 1,440 and native
2,240. Both tensors are finite and both files stayed unchanged during inspection.
This [CPU comparison](evidence/restart-parameter-update2240-20261013.json) verifies
actual saved updates, not their usefulness, all-model finiteness or BOS causality.
The checkpoint pair was chosen by availability, not test score; unlike the tensor
names, this pair was not selected before the checkpoints existed. No GPU was used.

The next decision remains the same: keep the recipe and endpoint fixed, inspect
subsequent scheduled evaluations (next after 2,416 updates), and preserve complete
recovery states. This incremental result stays in the supporting report. The
synchronized advisor slide, pdfpc notes and learning-guide status page retain
their October 12, 20:53 UTC cutoff: the main story, material limits and next
comparison have not changed. They keep the previous failure visible and the
full reproduction explicitly unfinished.

## Time and storage headroom

An operational check at **October 13, 01:15 UTC** used all 119 completed collection
timings and all 12 completed evaluation intervals. Including 58 remaining
evaluations and the in-flight collection counted in full, all-completed,
recent-20 and recent-10 rate scenarios leave **10.82, 12.34 and 10.09 hours**
before the 68-hour whole-owner cap, respectively. Every interval from the earlier
85-collection analysis reproduces exactly; no slow intervals were excluded.
These are descriptive extrapolations, not confidence bounds or completion
guarantees: response length, Monte Carlo branching, failures and filesystem
contention can change the cost. The recipe remains unchanged. See the
[updated timing receipt](evidence/restart-pacing119-20261013.json) and
[earlier receipt](evidence/restart-pacing85-20261012.json).

Actual directory quotas, rather than the shared filesystem's free-space
display, showed about **498 GB available under home and only 16 GB under
project**. Large 17.60 GB full checkpoints must therefore stay under home.
An observed-rate projection leaves roughly 365 GB there after remaining
artifacts, retained model files, a transient full state and one additional
protected full copy. Even twice the observed artifact growth leaves about
303 GB. File-count limits also matter; the receipt includes a deliberately
larger stress scenario that exhausts the file quota before the byte quota.
These are approximate scenarios, not reserved capacity, and other work can
consume space. See the [storage receipt](evidence/restart-storage-headroom-20261012.json).

Source inspection confirms that native checkpoint cleanup runs **before**
the next full save. Retaining a model directory does not retain its optimizer
state. The existing protected 160-, 960-, and 1,440-update copies are therefore
useful even while training is healthy. The 1,440-update state was selected at
22:45 UTC, before that state or its subsequent evaluation existed. Its completed
copy passed all ten source/destination hashes and counter checks at 23:09 UTC;
home had about 477 GB of directory-quota space remaining after the copy.
No live source, cleanup policy, training owner or endpoint was changed.
This operational check does not change the scientific result or advisor story.

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

At 23:09 UTC, the complete **1,440-update state** was protected outside native
cleanup. All ten source/destination hashes match across 17.60 GB; trainer,
actor and both active Adam counters are 1,440, with scheduler counter 1,439 and
no skipped steps. The source remained stable, and older protected copies were
retained. Selection preceded the checkpoint and its subsequent evaluation.
See the [protection receipt](evidence/restart-full1440-protection-20261012.json).

Two inspection failures are preserved in the external receipt directory:
library log lines initially prevented parsing the metadata JSON, and a later
JavaScript round-trip rounded nanosecond timestamps. The latter verifier
stopped before hashing. The exact original metadata and narrow regression
checks resolved both reporting issues; the full hash/counter verification then
passed. No model, optimizer, training source or configuration was changed.
That byte/counter check was not an all-tensor finite scan, a GPU restore,
or an off-system backup.
The separate faithful resume entrypoint remains conditional and unimplemented.

At **23:28 UTC**, a separate read-only CPU scan checked all six optimizer
tensors: 1,100,048,384 FP32 master-weight elements and the same number in each
of the two Adam moments. It also checked all 201 model tensors (1,100,048,384
elements) in each of the actor and Hugging Face exports. **No non-finite values
were found.** A small injected NaN/positive-infinity/negative-infinity fixture
checked the scanner across chunk boundaries. All ten protected file identities
and the earlier protection receipt were unchanged before and after; the scan
took 28 seconds with one CPU thread and did not initialize CUDA. See the
[finite-state receipt](evidence/restart-full1440-finite-20261012.json).
This supports keeping the saved recovery point; it does not prove future
stability, establish an end-to-end resume, or change the measured learning
result. Training continued unchanged and had reached 1,536 updates at 23:29 UTC.

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
