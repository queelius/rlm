# Diagnosing the SPO training failure and testing a corrected run

Status: October 13, 2026, 15:05 UTC; scientific cutoff 15:05, native-state check 15:03.
The thirtieth scheduled restart evaluation is fully regraded at 4,656 updates;
the next evaluation follows 4,816 updates.
Test falls to 54.75% and validation to 67.04%, while the non-held-out training
subset rises to 78.95%. Two successive test declines and this interval's
held-out/training divergence merit continued attention; they do not establish
sustained overfitting or renewed collapse. Overall improvement remains
intermediate learning, not completed reproduction or proof of lasting stability.
Native 4,640 counters and ten-file stability are verified; that is not a
protected copy, full-tensor scan or GPU restoration.
The complete 3,840-update checkpoint remains byte-verified outside cleanup,
alongside the earlier 160-, 960-, 1,440- and 2,560-update copies. A separate CPU scan
found all saved model parameters, FP32 master weights and Adam moments finite
at 1,440; no such full scan is claimed for newer states. Logged learning rates are
independently checked through 768 updates; the scientific recipe and fixed
endpoint are unchanged. Our primary objective remains reproducing the paper's
**training result** on one A100. We have matched the authors' released-model
evaluation, diagnosed concrete discrepancies in our training pipeline, and
started a fresh run that tests the strongest one.

The broader purpose, reaffirmed by the user, is methodological calibration for
continuing RLM research: check that our RL implementation, measurement and
experimental reasoning can recover a published learning result. Success would
strengthen confidence in those practices, not establish that SPO is the right
algorithm for RLM or that all implementation assumptions are correct. We will
report the fixed endpoint's exact score gap, gain from the starting model and
remaining protocol differences; no acceptance tolerance is being chosen from
an intermediate score.

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

The scheduled tests measure the model after **16, 176, 336, 496, 656, 816, 976, 1,136, 1,296, 1,456, 1,616, 1,776, 1,936, 2,096, 2,256, 2,416, 2,576, 2,736, 2,896, 3,056, 3,216, 3,376, 3,536, 3,696, 3,856, 4,016, 4,176, 4,336, 4,496, and 4,656 optimizer updates**.
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
| 2,416 | 51.81% | 65.48% | 72.06% |
| 2,576 | 52.76% | 66.72% | 72.24% |
| 2,736 | 53.13% | 67.95% | 72.40% |
| 2,896 | 53.57% | 68.05% | 74.26% |
| 3,056 | 53.59% | 66.82% | 75.03% |
| 3,216 | 53.31% | 67.02% | 75.84% |
| 3,376 | 53.57% | 66.64% | 76.43% |
| 3,536 | 54.02% | 67.59% | 76.43% |
| 3,696 | 54.48% | 67.11% | 75.52% |
| 3,856 | 54.13% | 66.71% | 76.14% |
| 4,016 | 54.20% | 67.24% | 76.39% |
| 4,176 | 54.74% | 66.54% | 77.12% |
| 4,336 | 55.34% | 67.81% | 78.53% |
| 4,496 | 55.09% | 68.03% | 78.73% |
| 4,656 | 54.75% | 67.04% | 78.95% |

### Latest evaluation: 4,656 updates

All 33,024 saved answers and 21 native metrics agree in the
[full audit](evidence/restart-eval290-20261013.json). Test has **11,554/21,104 correct
answers, 54.7479%**, or **+14.9735 percentage points** versus original SFT.
It loses 73 correct answers (-0.3459 points) since 280. Validation loses 59
(-0.9886 points) to **4,001/5,968, 67.0409%**. The training subset gains 13
(+0.2184 points) to **4,699/5,952, 78.9483%** and is not held out.

The last four tests are 54.74%, 55.34%, 55.09% and 54.75%. Both held-out means
fall this interval while training rises, after the preceding test decline.
That is a generalization watchpoint, not yet sustained-overfitting or collapse
evidence. Test once-hit slips 79.76% to 79.68%, with majority unchanged at 61.26%.
Test questions 294 improve / 326 worsen / 699 stay the same; 50 become once-hit
and 51 lose all correct samples. Validation coverage falls 89.54% to 86.33% and
majority 74.26% to 71.85%; 72 improve / 91 worsen / 210 stay the same, with 6 newly
once-hit and 18 lost. Training coverage falls 94.62% to 94.35% and majority 84.68%
to 84.41%, despite its higher mean; 74 improve / 67 worsen / 231 stay the same.

Test extraction failures rise 4 to 7 and length-limit endings 4 to 8.
Those small count changes alone cannot explain 73 fewer correct answers.
The predetermined first test question improves 8/16 to 14/16 despite aggregate
decline. Secondary numeric-equivalence potential rescues are 91/41/50 for
test/validation/train versus 89/38/44 before; primary scores remain unchanged.
These paired counts are descriptive, not significance tests or independent
training replications. Keep all 30 points and fixed 690/11,040; the current
1.9761-point gap to supplied 56.724% is not the final endpoint comparison.
No posthoc tolerance, causal BOS effect, lasting stability or RLM transfer is
established. See the [interpretation](evidence/restart-eval290-interpretation-20261013.json).

The [native 4,640 metadata check](evidence/restart-full4640-metadata-20261013.json)
verifies ten stable files and matching trainer/actor/active Adam counters,
scheduler 4,639 and zero skipped steps. It is distinct from evaluated 4,656,
and is not a protected copy, full finite scan or GPU restore. Native 4,480
optimizer files rotated at 14:46 UTC; its receipt remains historical. Protected
160/960/1,440/2,560/3,840 remain; prepared 5,120 protection retains all gates.

All 2,065 evaluation HTTP returns succeeded, including 2,064 generation POSTs;
no recorded transport errors or tracebacks. The first answer was checked within
eight seconds. Target 290 was selected earlier, but its wrapper was prepared
after native generation started. Focused RED/GREEN and independent binding review
passed; the full CPU audit ran once after all three completed split gates.
The read-only checkpoint inspection's string-path correction and the successor
watcher's accidental PID retarget were observation repairs, not training fixes.
The successor covers 300/4,816. It checked 291's correct first answer 18 after
113 seconds, with 747 successes and no errors; this missed the 90-second target
but does not show idle GPU time. Original owners and recipe are unchanged.

### Preceding evaluation: 4,496 updates

All 33,024 saved answers and 21 native metrics agree in the
[full audit](evidence/restart-eval280-20261013.json). Test has **11,627/21,104 correct
answers, 55.0938%**, or **+15.3194 percentage points** versus original SFT.
It loses 51 correct answers (-0.2417 points) since the preceding check.
Validation gains 13 (+0.2178 points) to **4,060/5,968, 68.0295%**.
The training subset gains 12 (+0.2016 points) to **4,686/5,952, 78.7298%**
and is not held out.

The last four scheduled tests are 54.20%, 54.74%, 55.34% and 55.09%.
This interrupts the preceding interval's aligned held-out increases.
Test once-hit coverage falls from 81.05% to 79.76%, and majority from 62.17%
to 61.26%. There are 314 questions with more correct samples, 338 fewer and
667 unchanged; 51 become once-hit and 68 lose all correct samples.
Validation coverage rises from 88.20% to 89.54%, and majority from 73.19%
to 74.26%, with 89 improved / 81 worsened / 203 unchanged questions.
Training coverage rises from 93.82% to 94.62%, and majority from 83.87%
to 84.68%; 75 questions improve, 73 worsen and 224 are unchanged.
The two held-out splits do not give a uniform direction.

Test extraction failures and length-limit endings each rise from two to four.
The two additional extraction failures alone cannot explain 51 fewer correct
answers. The predetermined first test question falls from 12/16 to 8/16 despite
its correct first response. Secondary numeric-equivalence potential rescues are
89 test, 38 validation and 44 train, versus 73, 31 and 49 previously; primary
scores are unchanged. These descriptive paired counts are not significance
tests or independent training replications.

Keep all 29 scheduled points, mean pass@1 and the fixed 690/11,040 endpoint.
The current score is 1.6302 points below the supplied 56.724% Figure 3 reference,
but is not the endpoint comparison. This interval neither establishes renewed
collapse nor rules it out. No posthoc acceptance tolerance, BOS causal claim,
lasting stability or measured RLM transfer is established.
See the [interpretation](evidence/restart-eval280-interpretation-20261013.json).

The [native 4,480-state metadata check](evidence/restart-full4480-metadata-20261013.json)
verifies ten stable files, trainer/actor/active Adam counters, scheduler 4,479
and zero skipped steps. This differs from the evaluated 4,496 weights and
is not a protected copy, full-tensor finite scan or GPU restore.
Native 4,320 rotated out at 14:03 UTC; its metadata receipt remains historical.
Protected 160/960/1,440/2,560/3,840 remain; prepared 5,120 protection awaits
the full-save, owner, stability and fresh byte/file quota gates.
The full reward/token audit remains collection 200, the two-tensor comparison
2,240 updates, and the all-saved finite scan 1,440.

All 2,065 evaluation HTTP returns succeeded: 2,064 generation POSTs and one
model-discovery GET, with no transport errors or tracebacks. The first test
response was checked within 20 seconds.
The audit target was selected before outputs; wrapper preparation at 13:59:07
preceded the 14:00:30 check showing no native evaluation directory.
The first logged request followed at 14:07:46. Focused RED/GREEN and independent
binding review passed. The full CPU audit ran once after all three split gates;
it and the completed read-only watcher exited normally.
The successor covers 290/4,656 and found collection 281's first answer correct
at 24, with 747 successful returns and no errors. Its 176-second inspection
delay missed our usual 90-second target; no GPU waste is inferred from that
observation gap. Original owner, sources and recipe are unchanged.

### Preceding evaluation: 4,336 updates

All 33,024 saved answers and 21 native metrics agree in the
[full audit](evidence/restart-eval270-20261013.json). Test has **11,678/21,104 correct
answers, 55.3355%**, or **+15.5610 percentage points** versus original SFT.
It gains 125 correct answers (+0.5923 points) since the preceding check.
Validation gains 76 (+1.2735 points) to **4,047/5,968, 67.8117%**.
The training subset gains 84 (+1.4113 points) to **4,674/5,952, 78.5282%**
and is not held out.

The last four scheduled tests are 54.13%, 54.20%, 54.74% and 55.34%.
Both held-out means and auxiliary coverage/majority scores rise this interval.
Test once-hit coverage rises from 79.30% to 81.05%, and majority from 61.64%
to 62.17%. There are 348 questions with more correct samples, 311 fewer and
660 unchanged; 70 become once-hit and 47 lose all correct samples.
Validation coverage rises from 86.33% to 88.20%, and majority from 72.92%
to 73.19%, with 98 improved / 76 worsened / 199 unchanged questions.
Training coverage falls from 94.35% to 93.82% despite its higher mean and
majority. The metrics remain distinct and improvement is not uniform.

Test extraction failures and length-limit endings each fall from three to
two. The predetermined first test question stays at 12/16; its first response
now correctly answers 18. Secondary numeric-equivalence potential rescues are
73 test, 31 validation and 49 train, versus 66, 33 and 40 previously; primary
scores are unchanged. These descriptive paired counts are not significance
tests or independent training replications. Keep all 28 scheduled points,
mean pass@1 and the fixed 690/11,040 endpoint. The current score is 1.3885
points below the supplied 56.724% Figure 3 reference, but is not the endpoint
comparison; no posthoc acceptance tolerance, BOS causal claim, lasting
stability or measured RLM transfer is established.
See the [interpretation](evidence/restart-eval270-interpretation-20261013.json).

The [native 4,320-state metadata check](evidence/restart-full4320-metadata-20261013.json)
verifies ten stable files, trainer/actor/active Adam counters, scheduler 4,319
and zero skipped steps. This differs from the evaluated 4,336 weights and
is not a protected copy, full-tensor finite scan or GPU restore.
Native 4,160 rotated out at 13:18 UTC; its metadata receipt remains historical.
Protected 160/960/1,440/2,560/3,840 remain; prepared 5,120 protection awaits
the full-save, owner, stability and fresh byte/file quota gates.
The full reward/token audit remains collection 200, the two-tensor comparison
2,240 updates, and the all-saved finite scan 1,440.

All 2,065 evaluation HTTP returns succeeded, with no transport errors or
tracebacks. The first test response was checked within 31 seconds.
The audit target was selected before outputs; wrapper preparation preceded
the first logged request by 25 seconds. The native directory already existed
at the first post-preparation check, so preparation before native initialization
is not established. Focused RED/GREEN and independent binding
review passed, and the full CPU audit ran once after all three split gates.
The completed read-only watcher exited normally. Its successor covers
280/4,496 and checked collection 271 within 59 seconds: correct 100, 313
successful returns and no errors. Original owner, sources and recipe are unchanged.

### Preceding evaluation: 4,176 updates

All 33,024 saved answers and 21 native metrics agree in the
[full audit](evidence/restart-eval260-20261013.json). Test has **11,553/21,104 correct
answers, 54.7432%**, or **+14.9687 percentage points** versus original SFT.
It gains 114 correct answers (+0.5402 points) since the preceding check.
Validation loses 42 (−0.7038 points) to **3,971/5,968, 66.5382%**.
The training subset gains 43 (+0.7224 points) to **4,590/5,952, 77.1169%**
and is not held out.

The last four scheduled tests are 54.48%, 54.13%, 54.20% and 54.74%.
Test improvement alongside validation decline does not establish a sustained
trend, overfitting, lasting recovery or completed reproduction. Test once-hit
coverage falls from 79.61% to 79.30%, while majority rises from 60.35% to
61.64%. There are 329 questions with more correct samples, 313 fewer and 677
unchanged; 50 become once-hit and 54 lose all correct samples. Validation
coverage falls from 89.28% to 86.33%, while majority rises from 72.65% to
72.92%. Training coverage rises while its majority falls. These metrics are
not interchangeable.

Test extraction failures and length-limit endings each fall from four to
three. The predetermined first test question rises from 9/16 to 12/16 despite
its first response being wrong. Secondary numeric-equivalence potential
rescues are 66 test, 33 validation and 40 train, versus 88, 37 and 42 previously;
primary scores remain unchanged. These descriptive paired counts are not
independent training replications or significance tests. Keep all 27 scheduled
points, mean pass@1 and fixed 690/11,040 endpoint; no favorable checkpoint,
posthoc acceptance tolerance, BOS causal claim or demonstrated RLM transfer.
See the [interpretation](evidence/restart-eval260-interpretation-20261013.json).

The full reward/token audit remains collection 200. A separate
[native 4,160-state metadata check](evidence/restart-full4160-metadata-20261013.json)
verifies ten stable files, trainer/actor/active Adam counters, scheduler 4,159
and zero skipped steps. This is not a protected copy, full-tensor finite scan
or GPU restore. It is distinct from the evaluated 4,176 weights. The first
read-only inspection encountered the installed PyTorch string-filename
requirement for mmap; correcting that argument completed the check without
changing training or checkpoint contents. Native 4,000 rotated at 12:34 UTC;
its receipt is historical. Protected 160/960/1,440/2,560/3,840 remain. The
step-selected 5,120 protection is prepared but awaits its full-save, owner,
stability and fresh byte/file quota gates.

All 2,065 evaluation HTTP returns succeeded, with no transport errors or
tracebacks. The first test response was inspected within 28 seconds and
wrongly answered 6 instead of 18. The completed read-only observer exited
normally after all three splits. A subsequent collection-262 check was late
at 123 seconds versus the normal 90-second goal, with 711 successes and zero
errors; no wasted GPU interval is inferred. Collection 263 was checked within
15 seconds and correctly answered 200. A new read-only observer covers
270/4,336. Original owner, sealed sources and scientific recipe are unchanged.

### Preceding evaluation: 4,016 updates

All 33,024 saved answers and 21 native metrics agree in the
[full audit](evidence/restart-eval250-20261013.json). Test has **11,439/21,104 correct
answers, 54.2030%**, or **+14.4285 percentage points** versus original SFT.
It gains 15 correct answers (+0.0711 points) since the preceding check.
Validation gains 32 (+0.5362 points) to **4,013/5,968, 67.2420%**.
The training subset gains 15 (+0.2520 points) to **4,547/5,952, 76.3945%**
and is not held out.

The last four scheduled tests are 54.02%, 54.48%, 54.13% and 54.20%.
This small rebound is not a sustained-trend, lasting-recovery or completed-
reproduction claim. Test once-hit coverage rises from 79.08% to 79.61%, but
majority falls from 61.03% to 60.35%. There are 326 questions with more correct
samples, 318 fewer and 675 unchanged; 57 become once-hit and 50 lose all
correct samples. Validation coverage rises from 88.47% to 89.28%, while
majority falls from 73.46% to 72.65%. Training coverage and majority remain
unchanged despite its higher mean. These metrics are not interchangeable.

Test extraction failures rise from three to four; length-limit endings fall
from five to four. The predetermined first test question falls from 11/16 to
9/16 correct samples despite the aggregate gain. Secondary numeric-equivalence
potential rescues are 88 test, 37 validation and 42 train, versus 80, 40 and 39
previously; primary scores are unchanged. These descriptive paired counts are
not independent training replications or significance tests. Keep all 26
scheduled points, mean pass@1 and the fixed 690/11,040 endpoint; no favorable
checkpoint, posthoc acceptance tolerance or BOS causal claim.
See the [interpretation](evidence/restart-eval250-interpretation-20261013.json).

The full reward/token audit remains collection 200. A separate
[native 4,000-state metadata check](evidence/restart-full4000-metadata-20261013.json)
verifies all ten stable files, matching trainer/actor/active Adam counters,
scheduler 3,999 and zero skipped steps. This is not a protected copy,
all-tensor finite scan or GPU restore. Protected 160/960/1,440/2,560/3,840 remain.
Native cleanup removed the 3,840 optimizer at 11:48:59 UTC, while retaining its
HF weights and custom metadata; the protected copy's ten identities still match.
Periodic directory retention is not optimizer retention. A stale assumption in
the publication check failed safely and was corrected before publishing; the
original protection receipt and the failed check are retained. The evaluated
4,016 weights are distinct. The step-selected 5,120 procedure is prepared but
not executed; fresh complete-state, owner, stability and byte/file quota
gates remain mandatory.

All 2,065 evaluation HTTP returns succeeded with no transport errors or
tracebacks. The first test response was inspected within 29 seconds and
wrongly answered 2 instead of 18; it does not determine the aggregate result.
After the completed 250 observer, the initial collection-251 check was late
at 139 seconds (normal goal 90 seconds), with 779 successful returns and
zero errors. Record the monitoring gap, not a wasted GPU interval. The new
read-only observer covers 260/4,176, and checked collection 252 within 38
seconds. The original whole owner, sealed sources and scientific recipe
remain unchanged.

A predetermined collection-247 example correctly answers `1600`, but
native gold `1,600` yields reward zero: the exact native method normalizes
commas in the prediction only. The saved trajectory and episode agree.
The [source-bound check](evidence/restart-reward-formatting247-20261013.json)
is one concrete instance, not a prevalence estimate. Do not silently alter
the reproduction's reward or inflate its primary score. For later RLM work,
validate reward formatting independently of mathematical and procedural
correctness; this is a methodological lesson, not demonstrated transfer.

### Preceding evaluation: 3,856 updates

All 33,024 saved answers and 21 native metrics agree in the
[full audit](evidence/restart-eval240-20261013.json). Test has **11,424/21,104 correct
answers, 54.1319%**, or **+14.3575 percentage points** versus original SFT.
It loses 74 correct answers (−0.3506 points) since the preceding check.
Validation loses 24 (−0.4021 points) to **3,981/5,968, 66.7058%**, its second
consecutive mean decline. The training subset gains 37 (+0.6216 points)
to **4,532/5,952, 76.1425%** and is not held out.

The last three scheduled tests are 54.02%, 54.48% and 54.13%; the previous
two-increase pattern did not persist. Both held-out means falling while
training-subset mean rises is worth following, but does not establish
overfitting, lasting deterioration, collapse or a cause. Test once-hit coverage
falls from 79.83% to 79.08%, and majority from 61.18% to 61.03%. There are
299 questions with more correct samples, 336 fewer and 684 unchanged; 46
become once-hit and 56 lose all correct samples. Validation coverage and
majority rise despite its lower mean; training coverage falls despite its
higher mean and majority. Do not interchange these metrics.

Test extraction failures decrease from eight to three and length-limit endings
from eight to five: increased extraction failures do not explain this decline.
The predetermined first test question falls from 12/16 to 11/16 correct samples.
These are descriptive paired counts, not significance tests or independent
training replications. Keep all 25 scheduled points, the unchanged 690/11,040
endpoint and mean pass@1; no favorable checkpoint or BOS causal claim.
See the [interpretation](evidence/restart-eval240-interpretation-20261013.json).

The full reward/token audit remains collection 200. The protected 3,840-update
state below is distinct from the 3,856-update evaluated weights. No new full
tensor scan or GPU restore is claimed. Native 3,680 rotated at 11:04 UTC;
its receipt remains historical. The 240 CPU audit and read-only observer
both exited successfully; a new read-only observer covers evaluation 250
after 4,016 updates. Its initial collection-241 response check was late
(187 seconds versus the normal 90-second goal); the inspected response was
wrong (9 instead of 15) and correctly received zero reward. At that check
there were 856 successful returns and no transport errors; no wasted GPU
interval is inferred from the late inspection.

### Preceding evaluation: 3,696 updates

All 33,024 saved answers and 21 native metrics agree in the
[full audit](evidence/restart-eval230-20261013.json). Test has **11,498/21,104 correct
answers, 54.4826%**, or **+14.7081 percentage points** versus original SFT.
It gains 97 correct answers (+0.4596 points) since the preceding check.
Validation loses 29 (−0.4859 points) to **4,005/5,968, 67.1079%**;
the training subset loses 54 (−0.9073 points) to **4,495/5,952, 75.5208%**,
and is not held out.

The last three scheduled tests are 53.57%, 54.02% and 54.48%. Two increases
do not establish a durable trend, and the directions across splits are mixed.
Test once-hit coverage falls from 80.29% to 79.83%, while majority rises
from 60.20% to 61.18%. There are 334 questions with more correct samples,
312 fewer and 673 unchanged; 50 become once-hit and 56 lose all correct
samples. Mean sample accuracy improves without covering more questions.
Validation mean, coverage and majority all fall. Training-subset coverage
rises while its mean and majority fall. These descriptive comparisons are
not significance tests or independent training replications.

The predetermined first test question rises from 9/16 to 12/16 correct samples.
Test extraction failures increase from six to eight and length-limit endings
from seven to eight. One live example is not an aggregate accuracy estimate.
Mean pass@1 remains the target, not once-hit or majority voting.
Continue the unchanged 690/11,040 endpoint; do not select an earlier checkpoint
or infer BOS causality, overfitting or completed reproduction.
See the [interpretation and paired counts](evidence/restart-eval230-interpretation-20261013.json).

The full reward/token audit remains collection 200. The
[native 3,680-state inspection](evidence/restart-full3680-metadata-20261013.json)
verifies ten stable files and matching trainer/actor/active Adam counters,
scheduler 3,679 and zero skipped steps. It is not a new protected copy, full
tensor scan or GPU restoration; evaluated 3,696 weights are distinct.
Native 3,520 rotated out at 10:19 UTC under existing cleanup; its preceding
receipt is historical. Protected 160/960/1,440/2,560 remain; next protection
is preselected at 3,840. No live source, configuration, environment or GPU
owner changed. The next scheduled evaluation follows 3,856 updates.

### Preceding evaluation: 3,536 updates

All 33,024 saved answers and 21 native metrics agree in the
[full audit](evidence/restart-eval220-20261013.json). Test has **11,401/21,104 correct
answers, 54.0229%**, or **+14.2485 percentage points** versus original SFT.
It gains 95 correct answers (+0.4502 points) since the preceding check.
Validation gains 57 (+0.9551 points) to 67.5938%; the training subset is
unchanged at 4,549/5,952, 76.4281%, and is not held out.

The last five scheduled tests are 53.57%, 53.59%, 53.31%, 53.57% and 54.02%.
The latest score moves above the preceding roughly flat band, but does not
establish a durable upward trend. Test once-hit coverage rises from 79.98%
to 80.29%, while majority rises from 60.05% to 60.20%. There are 342 questions
with more correct samples, 307 fewer and 670 unchanged; 56 become once-hit
and 52 lose all correct samples. Coverage remains below its earlier 81.27%
observation. Validation mean, coverage and majority all rise. Training-subset
mean and coverage are unchanged while majority rises. These descriptive
comparisons are not significance tests or independent training replications.

The predetermined first test question falls from 10/16 to 9/16 correct samples
despite the aggregate gain. Test extraction failures remain six and length-limit
endings seven; these aggregate counts do not explain the 95-sample increase.
The native mean pass@1 rule remains the target, not once-hit or majority voting.
Continue the unchanged 690/11,040 endpoint; do not select an earlier checkpoint
or infer BOS causality, overfitting or completed reproduction.
See the [interpretation and paired counts](evidence/restart-eval220-interpretation-20261013.json).

The full reward/token audit remains collection 200. The newer
[native 3,520-state inspection](evidence/restart-full3520-metadata-20261013.json)
verifies ten stable files and matching trainer/actor/active Adam counters,
scheduler 3,519 and zero skipped steps. It is not a new protected copy, full
tensor scan or GPU restoration; evaluated 3,536 weights are distinct.
Native 3,360 rotated out at 09:32 UTC under existing cleanup; its receipt below
is historical. Protected 160/960/1,440/2,560 remain; next protection is preselected
at 3,840. No live source, configuration, environment or GPU owner changed.

An additional operational observation at 09:50 UTC is instructive: collection
222's predetermined first training response received reward 1 for the correct
answer, four weeks, while omitting the overtime pay factor and incorrectly
equating 4,640/1,040 with four. Correct weekly pay is 1,160, giving exactly four
weeks. This [checked example](evidence/restart-outcome-reasoning222-20261013.json)
illustrates the boundary of final-answer rewards, not an evaluator defect or a
prevalence estimate. It motivates caution about reasoning validity in future
RLM reward design without changing this reproduction's scientific recipe.

### Preceding evaluation: 3,376 updates

All 33,024 saved answers and 21 native metrics agree in the
[full audit](evidence/restart-eval210-20261013.json). Test has **11,306/21,104 correct
answers, 53.5728%**, or **+13.7983 percentage points** versus original SFT.
It gains 55 correct answers (+0.2606 points) since the preceding check.
Validation loses 23 (−0.3854 points) to 66.6387%; the training subset gains
35 (+0.5880 points) to 76.4281%, not held out.

The last four scheduled tests are 53.57%, 53.59%, 53.31% and 53.57%: roughly
flat. This rebound does not establish a renewed upward trend or durable recovery.
Test once-hit coverage falls from 81.12% to 79.98%, while majority rises from
59.89% to 60.05%. There are 331 questions with more correct samples, 333 fewer
and 655 unchanged; 46 become once-hit and 61 lose all correct samples.
Higher mean sample accuracy therefore does not mean broader question coverage.
Validation coverage rises while majority falls; training-subset coverage and
majority both fall. These are descriptive comparisons, not significance tests
or independent training replications.

The predetermined first test question has 10/16 correct samples versus 11/16.
Test extraction failures fall from ten to six and length-limit endings from
eleven to seven. These changes do not identify the cause of the net rebound.
The native mean pass@1 rule remains the target, not once-hit or majority voting.
Continue the unchanged 690/11,040 endpoint; do not select an earlier checkpoint
or infer BOS causality, overfitting or completed reproduction.
See the [interpretation and paired counts](evidence/restart-eval210-interpretation-20261013.json).

The full reward/token audit remains collection 200. The newer
[native 3,360-state inspection](evidence/restart-full3360-metadata-20261013.json)
verifies ten stable files and matching trainer/actor/active Adam counters,
scheduler 3,359 and zero skipped steps. It is not a new protected copy, full
tensor scan or GPU restoration; evaluated 3,376 weights are distinct.
Native 3,200 optimizer state rotated out at 08:48 UTC under existing cleanup.
Protected 160/960/1,440/2,560 states remain; next protection is preselected at 3,840.

### Preceding evaluation: 3,216 updates

All 33,024 saved answers and 21 native metrics agree in the
[full audit](evidence/restart-eval200-20261013.json). Test has **11,251/21,104 correct
answers, 53.3122%**, or **+13.5377 percentage points** versus original SFT.
It loses 58 correct answers (−0.2748 points) since the preceding check.
Validation gains 12 (+0.2011 points) to 67.0241%; the training subset gains
48 (+0.8065 points) to 75.8401%, not held out.

Test once-hit coverage falls from 81.27% to 81.12% and majority from 60.12%
to 59.89%. Validation coverage falls from 88.74% to 87.13%, while majority rises
from 74.53% to 75.34%. Training-subset coverage rises to 93.01% while majority
falls to 83.06%. Test has 344 questions with more correct samples, 349 fewer
and 626 unchanged; 61 become once-hit and 63 lose all correct samples.
These descriptive counts do not establish significance or independent replication.

The predetermined first test question has 11/16 correct samples versus 13/16.
Test extraction failures rise from six to ten and length-limit endings from
nine to eleven. Neither change identifies the reason for the net test decline.
The native mean pass@1 rule remains the target, not once-hit or majority voting.
The latest two scheduled test increments are flat/down. Continue the unchanged
690/11,040 endpoint; do not select an earlier checkpoint or infer collapse,
overfitting, BOS causality or durable recovery from this one modest decline.
See the [interpretation and paired counts](evidence/restart-eval200-interpretation-20261013.json).

### Preceding evaluation: 3,056 updates

All 33,024 saved answers and 21 native metrics agree in the
[full audit](evidence/restart-eval190-20261013.json). Test has **11,309/21,104 correct
answers, 53.5870%**, or **+13.8125 percentage points** versus original SFT.
It gains only three correct answers (+0.0142 points) since the preceding check:
the test result is effectively flat. Validation loses 73 answers (−1.2232 points)
to 66.8231%; the training subset gains 46 (+0.7728 points) to 75.0336%, not held out.

Secondary metrics again differ. Test once-hit coverage rises from 79.91% to
81.27%, while majority accuracy falls from 60.42% to 60.12%. Validation coverage
rises from 87.94% to 88.74% and majority remains 74.53%; training-subset coverage
falls from 92.20% to 91.94%, while majority rises from 81.99% to 83.33%.
On test, 359 questions have more correct samples, 334 fewer and 626 unchanged,
net only three answers; 67 become once-hit and 49 lose all correct samples.
Validation has 84 improving, 92 declining and 197 unchanged question counts.
These observations do not establish significance, a cause, or uniform improvement.

The predetermined first test question has 13/16 correct samples versus 14/16.
Test extraction failures increase from four to six and length-limit endings from
eight to nine. The native mean pass@1 rule is unchanged; once-hit and majority
are secondary diagnostics, not replacement targets.
Continue the original training horizon and fixed 690/11,040 endpoint.
The substantial gain from SFT is still intermediate learning, not a completed
reproduction, independent-seed replication, BOS causal proof, or durable recovery.
See the [interpretation and paired counts](evidence/restart-eval190-interpretation-20261013.json).

The full reward/token audit remains collection 170 and the detailed numerical
receipt ends at 2,736. A newer CPU/meta inspection verifies native full state at
3,040; it does not claim a new tensor scan, protected copy or GPU restoration.
The live log check through 3,067 is contiguous/finite with no failure markers,
but retains the approximate-KL 0.678 excursion at 2,952 and older larger peaks.

### Preceding evaluation: 2,896 updates

All 33,024 saved answers and 21 native metrics agree in the
[full audit](evidence/restart-eval180-20261013.json). Test has **11,306/21,104 correct
answers, 53.5728%**, or **+13.7983 percentage points** versus original SFT.
It gains 93 correct answers (+0.4407 points) since the preceding scheduled check.
Validation gains six (+0.1005 points) to 68.0462%; the training subset gains 111
(+1.8649 points) to 74.2608%, which is not held-out performance.

The secondary measures do not move uniformly. Test once-hit coverage rises
from 79.38% to 79.91%, but majority-vote accuracy falls from 61.49% to 60.42%.
Validation coverage stays at 87.94% while majority rises from 73.99% to 74.53%;
training-subset coverage and majority also rise. On test, 337 questions have
more correct samples, 341 fewer and 641 unchanged, with 65 newly once-hit
questions and 58 losing all correct samples. Validation has 86 more, 90 fewer
and 197 unchanged. Neither the positive total nor these small differences
establish statistical significance, a cause, or uniform improvement.

The predetermined first test question has 14/16 correct samples versus 12/16.
Test extraction failures fall from 14 to four and length-limit endings from
18 to eight. These observations do not change the native mean pass@1 rule.
Continue the original training horizon and fixed 690/11,040 endpoint; this is
further intermediate learning, not completed reproduction or durable recovery.
See the [interpretation and paired counts](evidence/restart-eval180-interpretation-20261013.json).
The separate full reward/token audit remains collection 170, the detailed
numerical receipt ends at 2,736, and native full state was inspected at 2,880;
none is relabeled as a new audit of collection 180 or its 2,896 evaluated weights.

### Preceding evaluation: 2,736 updates

All 33,024 saved answers and 21 native metrics agree in the
[full audit](evidence/restart-eval170-20261013.json). Test has **11,213/21,104 correct
answers, 53.1321%**, or **+13.3577 percentage points** versus original SFT and
78 answers (+0.3696 points) versus the preceding scheduled evaluation. Validation
gains 73 answers (+1.2232 points) to 67.9457%; the training subset gains nine
(+0.1512 points) to 72.3958%, and is not held out.

The secondary measures are mixed. Test once-hit coverage rises from 79.30% to
79.38% and majority accuracy from 60.80% to 61.49%. Validation coverage falls
from 89.28% to 87.94% and majority accuracy from 74.26% to 73.99%; training-subset
coverage and majority also fall. On test, 372 questions have more correct samples,
329 fewer and 618 unchanged. There are 67 newly once-hit questions and 66 losing
all correct samples. Validation has six newly once-hit and eleven losing all
correct samples despite its higher mean score. These observations do not establish
the cause of the changes or statistical significance.

The predetermined first test question has 12/16 correct samples, versus 9/16
previously, and its first response correctly returns 18. Test extraction failures
fall from 15 to 14 while length-limit endings rise from 15 to 18. These examples
and aggregates do not replace the native mean pass@1 metric. The result is further
intermediate learning, not uniform coverage improvement, durable recovery, an
isolated causal BOS test, or completed reproduction. Keep every scheduled point
and the fixed 690/11,040 endpoint. See the
[interpretation and question-level comparisons](evidence/restart-eval170-interpretation-20261013.json).
The separately completed collection 170 reward/token audit and numerical window
through 2,736 are documented below; earlier audits remain available.

### Preceding evaluation: 2,576 updates

All 33,024 saved answers and 21 native metrics agree in the
[full audit](evidence/restart-eval160-20261013.json). Test has **11,135/21,104 correct
answers, 52.7625%**, or **+12.9881 percentage points** versus original SFT and
200 answers (+0.9477 points) versus the preceding scheduled evaluation. Validation
gains 74 answers (+1.2399 points) to 66.7225%; the training subset gains 11
(+0.1848 points) to 72.2446%, and is not held out.

Test once-hit coverage rises from 78.62% to 79.30% and majority accuracy from
59.59% to 60.80%. Validation coverage rises from 87.40% to 89.28% and majority
from 72.65% to 74.26%. Training-subset coverage falls slightly and majority is
unchanged. Neither mean accuracy nor these secondary measures improves uniformly:
376 test questions have more correct samples, 300 fewer, and 643 unchanged.
There are 63 newly once-hit test questions and 54 losing all correct samples.
The predetermined first test question still returns 18 on its first sample,
but its total falls from 10/16 to 9/16 correct. Test extraction failures rise
from 13 to 15 while length-limit endings fall from 18 to 15.

This is further measured intermediate learning, not the fixed endpoint, durable
recovery, a clean causal BOS test, or an independent training-seed replication.
Keep the recipe, all scheduled results and fixed 690/11,040 endpoint unchanged.
The protected 2,560 recovery state is distinct from the 2,576 evaluated weights.
See the [interpretation and question-level comparisons](evidence/restart-eval160-interpretation-20261013.json).

### Latest collection audit and saved state: 200 and 3,200 updates

The [collection 200 audit](evidence/restart-training-batch200-20261013.json)
verifies all 512 terminal rewards, original gold answers, saved response identities
and generation/training token bindings: 354 correct and 158 incorrect, without
malformed or unfinished penalties. Twenty-eight of 64 groups have mixed terminal
rewards; eleven terminal-uniform groups still have nonzero intermediate advantages,
with 2,300 nonzero advantage tokens overall. Saved actor/reference log probabilities
and advantages are finite. The completed generation log has 949 HTTP successes,
zero errors and no tracebacks; thirteen completed dataset files remain unchanged
through the audit. These changing training batches are not held-out comparisons.

An inspected incorrect response double-counts weekly egg collections, mixes dozens
with individual eggs and then takes the absolute value of a negative remainder:
64 versus gold 48, correctly rewarded zero. A correct example gives 36 candies.
These examples check actual returned reasoning and reward assignment, not a
sampled accuracy estimate. Raw responses remain in the external evidence store.

The [full 3,200-update CPU inspection](evidence/restart-full3200-metadata-20261013.json)
verifies ten stable files, trainer/actor/active Adam counters 3,200,
scheduler 3,199 and zero skipped steps, without initializing CUDA. This is not
a protected copy, all-tensor scan or GPU restoration. It precedes the evaluated
3,216-update weights. Native 3,040 optimizer state rotated at 07:57:42 under
the existing cleanup policy; its receipt remains historical.
Protected 160/960/1,440/2,560 states remain. The next 3,840 protection is preselected
and prepared, subject to actual-save, state, owner and fresh storage gates.

At 08:22 UTC the same owner has completed collection 203: all 1..3,264 update
numbers are contiguous, the 23 logged numeric fields are finite, and no failure
markers are recorded. The rounded 2,577–3,264 window has mean ratio 1.00455 and
maximum approximate KL 0.678; older larger excursions remain watchpoints.
This is not a new detailed-window tensor audit. The last actual two-tensor weight
comparison remains 2,240; the last all-saved-tensor finite scan remains 1,440.
No live source, recipe, environment or owner has been changed.

### Preceding training audit through 2,736 updates and historical state at 2,880

The [collection 170 audit](evidence/restart-training-batch170-20261013.json)
verifies all 512 terminal rewards, original gold answers, and saved response and
generation/training token identities. There are 387 correct and 125 incorrect
responses, without malformed or unfinished penalties; 34 of 64 question groups
have mixed terminal rewards. Five terminal-uniform groups still have nonzero
intermediate advantages, and 2,825 advantage tokens are nonzero. Saved actor and
reference log probabilities and advantages are finite. The completed generation
log contains 982 HTTP successes, no errors and no tracebacks. Thirteen completed
dataset files have stable identities and recorded hashes. This checks pipeline
integrity, not a held-out learning trend: successive training batches differ.

The first correct response computes 112 − 66 = 46. The first wrong response
misapplies separate child and adult pool admission rates to everyone, producing
1,154 instead of 588; it also contains an arithmetic error. The zero reward
agrees with the original gold answer. Raw responses remain external; the public
receipt retains only their row, prediction, gold answer and native reward.

The [2,577–2,736 numerical window](evidence/restart-training-windows2736-20261013.json)
has mean policy ratio 1.0047, approximate KL 0.06395, clipping fraction 0.09443,
and gradient norm 18.7435. Its approximate-KL peak is 0.416 at update 2,699,
followed by 0.0738. All first 2,736 logged update numbers are contiguous and all
23 numeric fields are finite; no logged NaN/Inf loss anomaly or ratio-guard
warning appears. Earlier windows, including the larger 0.736 and 0.541 peaks,
remain in the receipt. These rounded changing-batch proxies do not establish
future stability or exact distributional KL.

At its recorded inspection, the
[full 2,880-update checkpoint](evidence/restart-full2880-metadata-20261013.json)
had ten stable files, matching trainer, actor and active Adam counters of 2,880,
scheduler counter 2,879 and zero skipped steps. This CPU/meta inspection leaves
CUDA uninitialized. It is not an all-tensor scan, GPU restore or protected copy.
The earlier native 2,720 optimizer state rotated out at 06:17:48; its inspection
below is historical. Preserve protected 160/960/1440/2560. The next coarse copy
remains step-preselected at 3,840, subject to complete-save and storage gates.
Native 2,880 optimizer state rotated out at 07:08:16; the receipt is historical.
That recovery state is distinct from the evaluated weights at 2,896.

The newer [full 3,040-update inspection](evidence/restart-full3040-metadata-20261013.json)
at 07:12 UTC verifies ten stable files, trainer/actor/active Adam counters 3,040,
scheduler 3,039 and zero skipped steps. CUDA remains uninitialized in this
CPU/meta audit. It is not a new protected copy, all-tensor scan or GPU restoration.
Its weights precede the completed scheduled evaluation at 3,056 updates.
Native cleanup removed its optimizer state at 07:57:42; this receipt is historical.

Two reporting issues were corrected before publication: CPU-selection log lines
preceded the otherwise successful checkpoint JSON, and an in-memory numerical
draft retained the prior window's peak list. Parsing the schema-tagged JSON and
checking every peak against the new window resolved them. The native checkpoint
inspection succeeded throughout; neither issue changed training or live sources.

### Preceding training audit through 2,576 updates and saved state at 2,720

The [collection 160 audit](evidence/restart-training-batch160-20261013.json)
rechecks all 512 rewards and generation/training query-token identities against
the original data. It has 386 correct and 126 incorrect responses, no malformed or
unfinished penalties, and 30 of 64 questions with mixed terminal rewards. Nineteen
terminal-uniform groups nevertheless have nonzero intermediate advantages;
2,311 advantage tokens are nonzero. Saved actor/reference log probabilities and
advantages are finite. The completed generation log has 805 HTTP successes and
no HTTP errors or tracebacks. Training questions change between batches, so this
is an integrity check, not a held-out progress curve.

The first wrong response in saved order reverses yesterday and today: nine
songs yesterday and five more today should give 9 + 14 = 23, but the response returns
13. The native reward of zero agrees. The first correct response computes
3 × 18 × (2/3) = 36 remaining leaves. These examples illustrate both outcomes, not their
prevalence or a causal explanation of the learning gain. Raw outputs remain in
the external research store.

The [new 160-update numerical window](evidence/restart-training-windows2576-20261013.json),
updates 2,417–2,576, has mean policy ratio 1.0036, approximate KL 0.06836,
clipping fraction 0.09892 and gradient norm 18.9254. Its largest approximate KL is
0.338 at update 2,480, falling to 0.00118 on the next update. All 1–2,576 logged update
numbers are present and finite, without logged NaN/Inf loss anomalies or ratio
guard warnings. Earlier peaks 0.736 and 0.541 remain watchpoints: a smaller maximum
on different batches does not establish lasting stability. Approximate KL and
the reference penalty are native rounded proxies, not exact distributional KL.

The [full 2,720-update state](evidence/restart-full2720-metadata-20261013.json)
had ten stable files and matching trainer/actor/active Adam counters 2,720,
scheduler 2,719 and skipped 0. Native cleanup removed the previous 2,560 optimizer
files at 05:21:21; the separately protected 2,560 copy retains all ten recorded file
identities and its prior complete hashes. An old check requiring both source
and copy failed on that expected rotation; the failure is retained in the
review record. This is not a training failure, new full tensor scan, GPU restore,
or additional protected copy. Keep protected 160/960/1440/2560; the next coarse
protection is preselected 3,840, subject to actual-save and storage gates.

### Preceding evaluation and training audit: 2,416 updates

The preceding test has **10,935 correct answers out of 21,104**, compared with 8,394
for the original SFT model: **+12.04 percentage points**. It gained 93 correct
answers, or 0.44 points, since the preceding scheduled test. Validation lost
82 correct answers (−1.37 points), while the training subset gained 20 (+0.34 points).
This is mixed intermediate progress in our run, not a completed
reproduction or evidence that the token repair causes a lasting benefit.
The previous run also improved initially before deteriorating. We retain the
native decoding and mean single-answer accuracy; solved-at-least-once coverage
and majority voting do not replace that metric.

Test extraction failures rose from 12 to 13 and length-limit endings rose from
12 to 18, alongside 93 more correct answers. Validation extraction failures and
length-limit endings both rose from zero to three. These aggregates are
not a causal decomposition. Individual questions move both ways: 344 have more
correct samples, 345 fewer, and 630 the same count.
For the predetermined first test question, the model gets 10 of 16 answers right,
down from 12. Its first response accounts for both the three breakfast eggs and
four baking eggs, returning (16 − 3 − 4) × 2 = 18. It still misses 6 of 16
attempts on that question. This illustration is not a representative sample,
a separate benchmark, or a statistical-significance claim. A correct first
response is an operational check, not a replacement for the full evaluation.

**Secondary measures do not replace the paper's metric.** Test solved-at-least-once
coverage over 16 samples fell from 79.38% to 78.62%, while majority voting rose
from 59.06% to 59.59%. Validation coverage also fell, from 89.54% to 87.40%,
and majority voting fell from 73.73% to 72.65%. Training-subset coverage and
majority voting rise. Better test average correctness therefore does not
establish uniform improvement or broader sampled problem coverage. Test distinct
parsed answers per question fell from 4.487 to 4.478. Neither diversity nor an improving
score alone establishes beneficial concentration or lasting stability. We retain
every metric and scheduled result. This is one training run, not independent
replications or proof of a learning mechanism. See the
[full audit](evidence/restart-eval150-20261013.json) and
[interpretation](evidence/restart-eval150-interpretation-20261013.json).
The [preceding evaluation](evidence/restart-eval140-20261013.json) remains in the record.

At the same 2,416 updates, the earlier eventually failed run scored 48.49% test,
60.59% validation and 71.12% on the training subset. The fresh run is higher by
3.32, 4.89 and 0.94 points, respectively. The earlier all-answer audit and
unchanged native metric hashes were rechecked. Matching the update count is
useful context, but the earlier run had a resumed ancestry and other runtime
differences: this is not an isolated test of the BOS correction. Its later
collapse is a reason to keep the endpoint fixed, not declare recovery now.

Training completed 2,416 updates before this evaluation. The accompanying
batch audit, collection 150, contains 371 correct and 141 incorrect responses,
with no unfinished or malformed-response penalties. There are mixed terminal
outcomes on 20 of 64 questions. All 512 terminal rewards
regrade correctly. Training query tokens match actual generation requests, and
saved actor/reference log probabilities and advantages are finite. Twenty groups
with uniform terminal outcomes still have nonzero intermediate advantages:
SPO can supply a signal within a solution even when the final rewards all agree.
All first 2,416 update numbers are present and logged numeric values are finite.
Changing training questions are not a benchmark progress curve.

**A correct outcome need not imply correct arithmetic.** The first rewarded
response in an earlier audited batch, collection 130, ends with the gold
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

The first wrong response in collection 150 makes an ordinary arithmetic error:
it confuses the first- and second-day mushroom counts, then divides a count by
the dollar price. It returns 24 instead of `58 / 2 + 12 + 24 = 65`; native reward
zero agrees. This first-in-saved-order example is not a prevalence estimate.

The preceding 160-update diagnostic window has a mean policy ratio of 1.0030,
approximate KL of 0.0678, clipping fraction of 0.0931, and gradient norm of 18.50.
Its largest approximate KL is 0.541 at update 2,415, above the previous window's
0.355 at update 2,159. Earlier excursions of 0.736 at update 2,047 and 0.523 at
update 1,712 also remain in the record. The latest peak falls to 0.162 on the
next update. We retain these excursions as watchpoints; improved test scores do not
establish that the numerical concern is resolved.
There are no logged high-ratio guard warnings or NaN/Inf loss anomalies through
2,416 updates. These rounded changing-batch aggregates are not exact
distributional KL measurements or a guarantee of future stability.
Earlier windows and definitions remain in the
[numerical audit](evidence/restart-training-windows2416-20261013.json).

The scheduled full checkpoint at 2,400 updates had all ten required files;
trainer, actor and active Adam counters agreed at 2,400, scheduler at 2,399,
with no skipped steps. This was a
[CPU metadata check](evidence/restart-full2400-metadata-20261013.json),
not a new tensor scan, protected copy or GPU restore. File identities were
unchanged across this check. Native cleanup has since rotated that state and
the older 2,240-update full state; their receipts remain historical. The newer
2,560-update state is now separately protected, as detailed below; older
protected states remain available.

**Saved-weight check, October 13, 03:35 UTC:** the same two layer-zero projection
tensors used in earlier audits contain 1,854,493 and 2,630,490 changed BF16
elements, respectively, out of 4,194,304 each, between protected 1,440 and native
2,240. Both tensors are finite and both files stayed unchanged during inspection.
This [CPU comparison](evidence/restart-parameter-update2240-20261013.json) verifies
actual saved updates, not their usefulness, all-model finiteness or BOS causality.
The checkpoint pair was chosen by availability, not test score; unlike the tensor
names, this pair was not selected before the checkpoints existed. No GPU was used.

The next decision remains the same: keep the recipe and endpoint fixed, inspect
subsequent scheduled evaluations (next after 3,376 updates), and
preserve complete recovery states. The preselected 2,560-update protection has
passed full source/copy hashes and protected-copy counter checks. This mixed
incremental result and recovery milestone stay in the supporting
report. The synchronized advisor slide, pdfpc notes and learning-guide status page retain
their October 12, 20:53 UTC cutoff: the main story, material limits and next
comparison have not changed. They keep the previous failure visible and the
full reproduction explicitly unfinished.

## Time and storage headroom

An operational check at **October 13, 10:33 UTC** uses all 231 completed
collection timings and all 24 completed evaluation intervals. Including 46
remaining evaluations (with the fixed final evaluation) and counting the
in-flight collection in full, all-completed, recent-20 and recent-10 rate
scenarios leave **11.02, 13.99 and 13.10 hours** before the 68-hour whole-owner
cap. Their projected final-evaluation completion times are October 15 00:21,
October 14 21:23 and October 14 22:16 UTC, respectively. The whole cap remains
October 15 11:22:49 UTC. Every interval in the prior 119-collection/12-evaluation
analysis reproduces exactly; no slow intervals were excluded.

These are descriptive extrapolations, not confidence bounds or completion
guarantees: response length, Monte Carlo branching, failures and filesystem
contention can change the cost. The final test is conservatively budgeted at
the mean duration of a full three-split scheduled evaluation. No scientific
recipe or schedule was shortened. See the
[latest timing receipt](evidence/restart-pacing231-20261013.json),
[119-collection receipt](evidence/restart-pacing119-20261013.json) and
[85-collection receipt](evidence/restart-pacing85-20261012.json).

The following earlier storage scenarios are not fresh capacity guarantees;
the 05:09 check found about 450 GB available under home and 16 GB under project
after checkpoint copies and further artifacts. Recheck both byte and file-count
quotas before the preselected 3,840-update protection.

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

At **October 13, 04:36 UTC**, the preselected **2,560-update full state** was
also protected outside automatic cleanup. All ten source/copy hashes match
across 17.60 GB. Reading the protected copy confirms trainer iteration 160,
global/actor/active Adam step 2,560, scheduler counter 2,559 and no skipped steps.
Both source and copy stayed unchanged during verification. Selection was
recorded at 03:35 UTC, before the checkpoint and evaluation existed, and was
not based on score. Earlier 160-, 960-, and 1,440-update copies are retained.
See the [2,560-update receipt](evidence/restart-full2560-protection-20261013.json).

Home directory quota had about 451 GB and 502,457 file slots remaining after
the copy. Before copying, the earlier deliberately larger two-times-growth
scenario still left about 273 GB and 195,493 file slots after projected demand;
that scenario includes the new protected copy, retained model files and a
transient full state. These are estimates, not reserved space. No native
cleanup, live source, environment, owner, learning-rate horizon or endpoint
was changed. This is same-filesystem protection, not an off-system backup,
all-tensor finite scan, GPU restoration test or new accuracy result. The next
evaluation at 2,576 updates subsequently passed its full audit; the reproduction
remains unfinished.

At **October 13, 11:15 UTC**, the preselected **3,840-update full state** was
protected by the same additive procedure. All ten source/copy SHA256 hashes
match across 17.60 GB; source and destination have different inodes and remained
stable throughout verification. The copied trainer is at iteration 240,
trainer/actor/both active Adam states at 3,840, scheduler 3,839 and skipped steps
zero. No CUDA initialization or GPU restore occurred.
See the [3,840-update receipt](evidence/restart-full3840-protection-20261013.json).

Selection was recorded at 04:43 UTC before the checkpoint and evaluation, not
chosen from a test score. The source finished saving before inspection; the
previous native 3,680 state rotated at 11:04 UTC. Protected 160/960/1,440/2,560
remain in place. An initial read-only preflight failed because its old-state
lookup omitted leading zeros in 0160/0960. The failed attempt is retained; the
minimal corrected lookup and full preflight passed before copying. No model,
source, configuration, environment or owner changed.

The [fresh preflight](evidence/restart-full3840-preflight-20261013.json) found
about 438 GB and 481,815 file slots available at home. Reusing the earlier
conservative 605-collection/61-evaluation two-times-growth scenario, including
one additional full copy, left about 243 GB and 174,635 file slots after projected
demand. Fewer collections now remain, but this is not a new growth measurement,
reservation or guarantee. After copying, home reported about 420 GB and 480,939
file slots available. Project still had only about 16 GB and was not used for
the checkpoint. Preselect the next coarse protection at 5,120 updates, with
fresh complete-state and storage gates. Recovery protection is not new accuracy,
an all-tensor finite scan, end-to-end resumption or an off-system backup.

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
