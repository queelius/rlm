# Reproducing an LLM reinforcement-learning experiment on one A100

## Current status on October 13, 17:17 UTC

Completed scientific-result cutoff: 17:17 UTC. Live-update check: 17:16 UTC;
protected 5,120-update recovery copy verified at 17:12 UTC.

**The own-training reproduction is incomplete, not successful.** The previous run
continued after the last written review below, deteriorated, and exited with
an error on October 10 at 00:49 UTC. Its last logged optimizer step was 7,995,
short of the prescribed 11,040; no final checkpoint marker exists.

The previous run's latest scheduled evaluation, after 7,856 updates, reports **22.19% test,
22.18% validation and 26.81% training-subset accuracy** in the native metric
files. These are the latest results, not the earlier 50.52% test score. The
terminal receipt, metric dictionaries and traceback were checked. Independent
regrading now matches all 66,048 saved answers and all seven native metrics
across the two audited checkpoints and three splits. See the
[terminal audit](evidence/terminal-audit-20261012.json).

The traceback ends at a BF16 optimizer gradient-norm assertion after four
microbatches were rejected by the PPO ratio guard. The owner did not time out.
The immediate crash and the preceding accuracy collapse require separate
diagnosis; neither is established to be caused by allocation expiry.

The advisor's primary objective remains reproducing a published training
result on our A100. Evaluating the authors' released weights successfully
recovered 56.69%, but that is not our training reproduction. RLM teaching
exploration is secondary.

**A fresh training run started at 15:22 UTC from the original SFT model.** We
found that generation included a beginning-of-sequence token which training
omitted. Real A100 tests show that this mismatch strongly changes the late
model's probabilities, although it had little effect on the starting model.
The new run corrects that mismatch and separately handles a verified
zero-gradient optimizer boundary. The shared source and environment are
unchanged; both corrections are isolated and documented.

The latest scheduled test, after **5,136 optimizer updates**, is **55.61%**, versus
**39.77%** for the original model: **+15.83 percentage points**. This is the
thirty-third scheduled evaluation; all 33,024 saved answers and 21 native metrics
independently agree. The [complete scheduled series](diagnosis-and-restart-20261012.md#scheduled-evaluations-so-far)
retains every earlier result.

Test gains 12 correct answers (+0.057 percentage points), validation loses
16 (−0.268 points) to 67.54%, and the training subset gains 26 (+0.437 points)
to 80.63%; training is not held out. Recent tests are 55.69%, 55.55% and 55.61%:
nearly flat while validation falls twice and training rises. Keep this
generalization watchpoint without claiming sustained overfitting, collapse,
a renewed upward trend or a cause. Overall improvement is **intermediate
learning, not completed reproduction**. The fixed 690-collection /
11,040-update comparison remains required; no acceptance tolerance is being
chosen from interim scores.

There are 326 test questions with more correct samples, 297 with fewer and
696 unchanged: 760 gained correct answers versus 748 lost. The first test
question falls from 15/16 to 14/16 despite the aggregate increase. Validation
once-hit coverage rises from 87.40% to 88.47%, and majority accuracy from
72.92% to 73.46%, despite lower mean accuracy. These auxiliary scores cannot
replace native mean pass@1. Test extraction failures and length-limit endings
fall from four to one; validation has none, so increased extraction failure
does not explain its decline. These are descriptive comparisons, not
independent training replications or significance tests. See the
[latest evaluation audit](evidence/restart-eval320-20261013.json) and
[interpretation](evidence/restart-eval320-interpretation-20261013.json).

An additional training observation at 09:50 UTC illustrates a scoring limit:
the first saved response in collection 222 received the correct-answer reward
despite erroneous intermediate arithmetic. This is faithful to the paper's
answer-only reward, not an evaluator failure. One example cannot estimate how
often this happens; see the [arithmetic check](evidence/restart-outcome-reasoning222-20261013.json).
For later RLM work, improved final-answer accuracy must not be treated as proof
that every reasoning or tool-use step is sound.

A second predetermined example exposes a formatting boundary: collection 247
answers `1600` correctly in mathematical terms, but receives zero reward against
native gold `1,600`. Executing the exact source method and checking the saved
trajectory and episode confirms that the native grader strips commas from the
prediction only. We preserve that rule for this reproduction and keep numeric-
equivalence diagnostics secondary. This is a concrete reward-validation lesson
for later RLM experiments, not a measured RLM transfer effect or a prevalence
estimate. See the [native reward check](evidence/restart-reward-formatting247-20261013.json).

A third predetermined training example contains an inconsistency in the supplied
label itself: original author row 754's worked solution computes 108, while its
terminal label says 106. All eight sampled answers give 108, receive native zero
reward and have zero saved advantages. Saved text/token bindings and leading BOS
agree. This is one record, not an estimate of label-error prevalence or aggregate
training impact. Preserve the author labels and native scores for reproduction;
the lesson for later RLM work is to audit reward oracles, not a measured transfer
effect. See the [label inconsistency evidence](evidence/restart-training-label310-20261013.json).

The separate full collection 200 audit verifies all 512 terminal rewards and
generation/training token bindings: 354 correct / 158 incorrect, with finite
saved log probabilities and advantages. Twenty-eight question groups have mixed
terminal rewards; eleven terminal-uniform groups still have nonzero intermediate
advantages. These changing training questions do not form a held-out progress
curve; see the [training audit](evidence/restart-training-batch200-20261013.json).
The last detailed numerical window, updates 2,577–2,736, remains in the
[numerical audit](evidence/restart-training-windows2736-20261013.json).
The live log check at 17:16 UTC verifies contiguous updates through 5,152,
all 23 logged numeric fields finite, and no recorded failure markers.
In updates 4,977–5,152, ratios span 0.962–1.12 and gradient norms reach 70.5.
Approximate KL reaches 1.06 at update 5,135; that update's ratio is 0.987
and gradient norm 11.5. Preserve this finite numerical watchpoint without
inferring its cause, full-tensor finiteness or future stability. Earlier
excursions remain recorded.

**Recovery update, October 13, 17:12 UTC:** the complete **5,120-update checkpoint**
is copied outside native cleanup. All ten files (17.60 GB) match by SHA256;
trainer, actor and both active Adam counters are 5,120, scheduler 5,119,
with no skipped steps. This was preselected by training step at 11:55 UTC,
before it existed. Earlier protected 160/960/1,440/2,560/3,840 states remain.
The saved recovery point is not the evaluated 5,136-update state.
See the [checkpoint receipt](evidence/restart-full5120-protection-20261013.json)
and [storage/state preflight](evidence/restart-full5120-preflight-20261013.json).
Copy-time home quota and conservative future-growth gates passed. This is
same-filesystem protection, not an off-system backup, all-tensor finite scan
or tested faithful GPU restart. A separate resume entrypoint remains required
after complete owner departure if interrupted. Next protection is preselected
at 6,400, still subject to fresh state, stability and storage gates.

**Earlier recovery update, October 13, 11:15 UTC:** the complete **3,840-update checkpoint**
is now copied outside the trainer's automatic cleanup directory. All ten files
(17.60 GB) match their originals by SHA256; trainer, actor and both active Adam
counters are 3,840, and the scheduler is 3,839, with no skipped steps. Selection
at 04:43 UTC preceded the checkpoint and its evaluation. Earlier 160-, 960-,
1,440- and 2,560-update
copies are retained. This protects a newer recovery point, not a new scientific
result, full numerical scan, GPU restoration test, or off-system backup.
See the [new checkpoint receipt](evidence/restart-full3840-protection-20261013.json)
and [storage/state preflight](evidence/restart-full3840-preflight-20261013.json).
An initial read-only preflight stopped on an unpadded lookup of the older 0160/0960
directory names. The lookup was corrected and the full preflight passed before
copying; the failed inspection is preserved and training was untouched.
The [earlier 2,560-update protection](evidence/restart-full2560-protection-20261013.json)
survives native cleanup of its optimizer files at 05:21 UTC.
A separate [CPU metadata check of native 4,960](evidence/restart-full4960-metadata-20261013.json)
at 16:21 UTC verifies all ten stable files, matching trainer/actor/active Adam
counters, scheduler 4,959 and zero skipped steps. This is not a protected copy,
full-tensor scan or GPU restore; it precedes the 4,976-update evaluated weights.
Native 4,960 was subsequently removed entirely by normal cleanup at 16:58:25 UTC;
its receipt is now historical. The protected 5,120 copy above is the current recovery point.
Native 4,800 was removed entirely by normal cleanup at 16:16 UTC; neither its
optimizer nor HF export survives there. Its [metadata receipt](evidence/restart-full4800-metadata-20261013.json)
is historical.
Native 4,640 optimizer files rotated out at 15:31 UTC; its
[metadata receipt](evidence/restart-full4640-metadata-20261013.json) is historical.
That earlier read-only inspection's string-filename correction remains
recorded; it did not alter training or checkpoint data.
Native 4,480 optimizer files rotated out at 14:46 UTC; its
[metadata receipt](evidence/restart-full4480-metadata-20261013.json) is historical.
Native 4,320 rotated out at 14:03 UTC; its
[metadata receipt](evidence/restart-full4320-metadata-20261013.json) is historical.
Native 4,160 rotated out at 13:18 UTC; its
[metadata receipt](evidence/restart-full4160-metadata-20261013.json) remains historical,
including the preserved string-filename mmap inspection failure and correction.
Native 4,000 was removed by normal cleanup at 12:34 UTC; its
[metadata receipt](evidence/restart-full4000-metadata-20261013.json) is historical.
Native cleanup removed the 3,840 optimizer at 11:48:59 UTC; only HF weights and
custom metadata remain there. Its separate ten-file protected copy is intact.
Native 3,680 rotated at 11:04 UTC; its
[CPU inspection receipt](evidence/restart-full3680-metadata-20261013.json) is historical.
Native 3,520 rotated out at 10:19 UTC; its
[inspection receipt](evidence/restart-full3520-metadata-20261013.json) is historical.
Native 3,360 rotated out at 09:32 UTC; its
[inspection receipt](evidence/restart-full3360-metadata-20261013.json) is historical.
Native 3,200 optimizer state rotated out at 08:48 UTC; its
[inspection receipt](evidence/restart-full3200-metadata-20261013.json) is historical.
Native 3,040 optimizer state rotated out at 07:57 UTC; its
[inspection receipt](evidence/restart-full3040-metadata-20261013.json) is historical.
Native 2,880 optimizer state rotated out at 07:08 UTC, and 2,720 at 06:17 UTC;
their [2,880 inspection](evidence/restart-full2880-metadata-20261013.json) and
[2,720 inspection](evidence/restart-full2720-metadata-20261013.json) remain historical.
The [earlier 1,440-update copy](evidence/restart-full1440-protection-20261012.json)
was verified at 23:09 UTC. Its additional **CPU scan at 23:28 UTC** checked every
element of the saved model parameters in both exports, the FP32 master weights,
and both Adam moments:
all are finite. The [finite-state receipt](evidence/restart-full1440-finite-20261012.json)
records the counts and unchanged file identities. This strengthens the recovery
check; it is not an end-to-end GPU restore or a new accuracy result. A separate
faithful resume entrypoint is still required if interrupted; the live launcher
is fresh-start-only.
Training continues under the same owner. A **CPU check at October 13, 03:35 UTC**
of the [same two previously selected weight tensors](evidence/restart-parameter-update2240-20261013.json)
confirms actual saved changes from 1,440 to 2,240 updates; both tensors are finite.
This is not an all-parameter scan or a measure of learning effectiveness.
The [earlier 960-to-1,280 check](evidence/restart-parameter-update1280-20261012.json)
remains in the record. Protected recovery states are distinct from the
5,136-update weights evaluated above. The earlier native full 2,400-update state
passed a [CPU metadata check](evidence/restart-full2400-metadata-20261013.json);
it was not separately protected or GPU-restored. Native cleanup has now rotated
it and the older 2,240-update full state; their receipts remain historical.
Evaluation 320 is complete and independently regraded. All 2,065 evaluation
HTTP returns succeeded: 2,064 generation POSTs and one model-discovery GET.
No recorded transport errors or tracebacks occurred. Training resumed under
the same owner.

The target was fixed earlier. The audit wrapper was created at 17:13:27 UTC,
after the first model request at 17:03:11 and completion of all three native
splits at 17:12:05; this was not prospective wrapper preparation. The focused
binding test and independent review passed, and the CPU audit ran once after
all three split gates, exiting normally at 17:16:15. The prior observation
handle also completed normally. No scoring, training or live-source changes
were made.

The first evaluation answer was correct (18), checked 17 seconds after the
first request with 58 successful returns and no errors. The first
post-evaluation training answer was also correct (39), checked after 78 seconds
with 572 successful returns and no errors. Both checks meet the 90-second
operational target. The preceding evaluation 310 handoff's 311-second check
remains a recorded historical miss, not evidence of GPU idleness or service
failure. The successor read-only observer was started before publication work.

The next scheduled evaluation, 330, follows 5,296 updates. Its read-only observer
is active; prepare the CPU audit from pinned evaluation 320 and run only
after all three native splits complete. The full 5,120-update recovery state
is now protected and verified as documented above. The next step-selected
6,400-update protection procedure is prepared; no copy has started. Require
complete-save, owner, file-stability and fresh byte/file quota gates.

**Operational check, October 13, 10:33 UTC:** all 231 completed collection
timings and 24 evaluation intervals leave about 11–14 hours before the whole-run
cap under the recorded rate scenarios, including the fixed final evaluation;
completion is not guaranteed. All earlier intervals reproduce without exclusion.
The 11:11 preflight found about 438 GB under home, versus only 16 GB under project.
Reusing the earlier deliberately larger two-times-growth scenario, including this
copy, leaves about 243 GB and 174,635 file slots of projected headroom. After the
copy, the live home quota reported about 420 GB and 480,939 file slots remaining.
These are shared, potentially lagging counters, not reserved space; recheck before
another copy.
The [headroom analysis](diagnosis-and-restart-20261012.md#time-and-storage-headroom)
records assumptions and limits. The scientific result above is unchanged.

The original 1,000-iteration learning-rate
schedule and fixed 690-collection / 11,040-update endpoint remain unchanged.
The schedule is now checked against actual optimizer logs, not just the
configuration: all 768 audited update displays and 152 full-precision
parameter-group values match the original 16,000-update horizon. The learning
rate was not shortened to decay over 11,040 updates. This is
[runtime fidelity evidence](evidence/restart-learning-rate-fidelity-20261012.json),
not a new benchmark or proof of the authors' exact historical implementation.
The corrected run has a 68-hour cap including final evaluation; an interruption
will remain an incomplete result.

A CPU recovery check also confirms why the original configuration must travel
with the checkpoint: the native loader restores training counters and running
statistics, but its fixed KL controller retains the configured coefficient.
Our saved and configured coefficient agree at `0.0001`. This is a
[recovery constraint](evidence/restart-recovery-custom-state-20261012.json),
not a new accuracy result or proof that a complete GPU resume is ready.

[Diagnosis, evidence, and restart plan](diagnosis-and-restart-20261012.md)
explain the decision and remaining limits. In particular, the exact historical
author implementation remains unavailable, and this restart is not proof that
the token mismatch caused the decline. We will report the full scheduled series,
not choose the most favorable test checkpoint.

## Historical progress at the last October 9 review

Status: 9 October 2026, 06:32 UTC. **Our latest evaluated weights scored 50.52% after 4,336 optimizer updates, versus 39.77% before training: +10.75 percentage points.** Test accuracy fell 0.53 points, while validation edged up to 63.99%. The training-subset result is now 75.32%, up 0.24 points; this is not held-out accuracy. Larger finite gradient spikes were followed by lower norms; their cause and effect on test accuracy remain unestablished. We report every scheduled result, not a selected best checkpoint, and the reproduction is incomplete. Training continues with the scientific recipe unchanged toward the prescribed 11,040-update endpoint. The released author model scored 56.69%, matching the paper's 56.7% reference.

## What we are trying to reproduce

We selected **Segment Policy Optimization (SPO)**, using the authors' RhoMath 1.1B experiment on GSM8K arithmetic word problems. The paper describes this experiment on one A100 40 GB—the GPU available to us. We use their implementation, their already-supervised-fine-tuned starting model, and their existing data splits. We are reproducing the **RL stage**, not the model's pretraining or its earlier supervised training.

The published comparison is:

| Model or training method | Paper's GSM8K score |
| --- | ---: |
| Starting supervised-fine-tuned model | 40.3% |
| GRPO baseline | 44.6% |
| SPO-chain, interval 5 | 56.7% |

The question is not simply whether we can load the authors' trained model. It is whether **our own run, starting from their initial model, recovers the improvement under the same recorded recipe and scoring procedure**. Evaluating their released weights is an important check, but it is not our training result.

Sources: [paper](https://arxiv.org/html/2505.23564v2), [official code and Figure 3](https://github.com/AIFrameResearch/SPO/tree/6c5f94723c20e4de68578545b8bb98186e956797), [author experiment logs](https://wandb.ai/my-wandb-team/SPO-experiments).

## Why this is useful for our RLM research

Ordinary outcome-based RL says whether a whole answer succeeded. SPO tries to give more local feedback: from an intermediate part of an answer, sample possible continuations and estimate how often they succeed. This helps estimate which parts of the reasoning improved the chance of success.

For example, a model may correctly identify the quantities in a word problem but make an arithmetic mistake later. A final failure alone does not tell us which part went wrong. Sampling continuations from intermediate points can provide more information, at the cost of extra generation.

The possible connection to a recursive language model (RLM) is attractive: a trajectory includes decisions about Python code, subagent calls, and decomposition. Can we learn which decisions help, instead of assigning the same final outcome to every action? **That is a later research question—not something this math reproduction establishes.** First we want a trustworthy training-and-evaluation workflow.

## What has happened so far

| Our experiment | Result | What it establishes |
| --- | ---: | --- |
| Original starting model, full native test | **39.77%** | Close to the paper's 40.3% starting reference; evaluator and saved answers checked |
| Authors' released SPO weights, same full test | **56.69%** | Recovers the published 56.7% reference to one decimal place; not our training gain |
| Our weights after 1 collection / 16 optimizer updates | **40.01%** | First full intermediate test; +0.24 percentage points versus our starting model, not a demonstrated substantive gain |
| Our weights after 11 collections / 176 optimizer updates, before interruption | **41.20%** | +1.42 percentage points versus our starting model; early signal from one training run, not the fixed final endpoint |
| Resumed weights after repeating updates 161–176 | **41.24%** | +1.47 percentage points versus our starting model; shares the first 160 updates with the interrupted branch, not an independent replication |
| Our weights after 21 collections / 336 optimizer updates | **42.44%** | +2.66 points versus our starting model and +1.19 points since the resumed 176-update test; still one ongoing training ancestry |
| Our weights after 31 collections / 496 optimizer updates | **43.56%** | +3.79 points versus our starting model and +1.12 points since 336 updates; still far short of the prescribed 11,040-update endpoint |
| Our weights after 41 collections / 656 optimizer updates | **44.21%** | +4.44 points versus our starting model; +0.65 points since 496 updates, while validation is nearly flat |
| Our weights after 51 collections / 816 optimizer updates | **45.15%** | +5.38 points versus our starting model; +0.94 points since 656 updates, with validation improving again |
| Our weights after 61 collections / 976 optimizer updates | **45.84%** | +6.07 points versus our starting model; +0.69 points since 816 updates, with validation also rising |
| Our weights after 71 collections / 1,136 optimizer updates | **46.71%** | +6.94 points versus our starting model; +0.87 points since 976 updates, while validation is essentially flat |
| Our weights after 81 collections / 1,296 optimizer updates | **46.61%** | +6.83 points versus our starting model; test is essentially flat (−0.10 points), while validation rises to 60.67% |
| Our weights after 91 collections / 1,456 optimizer updates | **47.31%** | +7.54 points versus our starting model; +0.71 points since 1,296 updates, with validation rising to 61.36% |
| Our weights after 101 collections / 1,616 optimizer updates | **47.46%** | +7.69 points versus our starting model; a small +0.15-point test increment, with validation at 61.90% |
| Our weights after 111 collections / 1,776 optimizer updates | **47.83%** | +8.06 points versus our starting model; a modest +0.37-point test increment, with validation at 62.40% |
| Our weights after 121 collections / 1,936 optimizer updates | **46.64%** | +6.86 points versus our starting model, but −1.20 points since the previous test; validation also falls to 61.13% |
| Our weights after 131 collections / 2,096 optimizer updates | **47.99%** | +8.21 points versus our starting model; test rebounds by 1.35 points, but validation recovers only slightly to 61.46% |
| Our weights after 141 collections / 2,256 optimizer updates | **48.54%** | +8.77 points versus our starting model; test rises another 0.55 points while validation falls to 60.87% |
| Our weights after 151 collections / 2,416 optimizer updates | **48.49%** | +8.72 points versus our starting model; test is essentially flat (−0.05 points), while validation edges down to 60.59% |
| Our weights after 161 collections / 2,576 optimizer updates | **48.75%** | +8.98 points versus our starting model; test rises 0.26 points and validation rebounds to 61.46%, below its earlier 62.40% |
| Our weights after 171 collections / 2,736 optimizer updates | **49.31%** | +9.54 points versus our starting model; test rises 0.56 points and validation recovers to 62.55% |
| Our weights after 181 collections / 2,896 optimizer updates | **49.98%** | +10.20 points versus our starting model; test rises 0.66 points and validation improves to 63.27% |
| Our weights after 191 collections / 3,056 optimizer updates | **50.42%** | +10.64 points versus our starting model; test rises 0.44 points and validation improves to 64.29% |
| Our weights after 201 collections / 3,216 optimizer updates | **50.74%** | +10.97 points versus our starting model; test rises 0.33 points while validation slips to 64.02% |
| Our weights after 211 collections / 3,376 optimizer updates | **50.56%** | +10.78 points versus our starting model; test slips 0.18 points while validation rises to 65.55% |
| Our weights after 221 collections / 3,536 optimizer updates | **51.18%** | +11.41 points versus our starting model; test rises 0.63 points while validation slips to 65.18% |
| Our weights after 231 collections / 3,696 optimizer updates | **50.92%** | +11.15 points versus our starting model; test falls 0.26 points and validation falls to 64.08% |
| Our weights after 241 collections / 3,856 optimizer updates | **50.43%** | +10.65 points versus our starting model; second test decline, while validation rebounds to 64.59% |
| Our weights after 251 collections / 4,016 optimizer updates | **51.15%** | +11.38 points versus our starting model; test recovers to its earlier range, validation essentially flat at 64.46% |
| Our weights after 261 collections / 4,176 optimizer updates | **51.05%** | +11.27 points versus our starting model; test essentially flat, validation falls to 63.82% |
| Our weights after 271 collections / 4,336 optimizer updates | **50.52%** | +10.75 points versus our starting model; second consecutive test decline, validation edges up to 63.99% |
| Matched GRPO training control | [Provenance audited; further preparation needed](grpo-control-plan.md) | Current repo defaults and reward handling cannot be assumed to reproduce the historical baseline |

The starting-model run saved all **1,319 questions × 16 answers = 21,104 answers**. We checked every question against the original test split and recomputed all seven native metrics exactly. Its launcher returned an error **after** scoring and cleanup because it passed an unsupported command-line argument. That command failure is retained separately; subsequent launches use DeepSpeed's supported `--no_local_rank` option. We did not change model outputs or the grader to make the score look better. See [native metrics](evidence/base-reference-native.json) and [command receipt](evidence/base-reference-command.json).

The released-author evaluation also covered all 21,104 answers, with every question identity and native score recomputed, and exited successfully. It scored 56.68594%, versus the historical log's 56.72384% (a difference of 0.038 percentage points). The difference between our two **reference models** is 16.91 percentage points. That confirms we can measure the released models' performance difference; it does not show that our fresh training has reproduced it. See [author-model metrics](evidence/author-reference-native.json), [command receipt](evidence/author-reference-command.json), and [validation receipt](evidence/author-reference-validation.json).

One concrete observation: on the first test question, the starting model gave the correct answer in 4 of 16 attempts. It also made identifiable arithmetic and interpretation mistakes. This illustrates usable variation in outcomes; it is not evidence of an RL improvement, because these are the **starting** weights.

The first training attempt generated 512 responses, then ran out of GPU memory while calculating their model probabilities. The inference server had reserved 26.27 GiB alongside the trainer. We preserved the failure and restarted from the same original weights with the inference-memory reservation limited to 30%. This changes resource allocation, not the dataset, training batch, objective, sampling settings, or learning-rate schedule. Configuration equality was checked after excluding only that memory setting and output paths. It may affect inference scheduling, so it is documented—not presented as bit-for-bit equivalence. See the [failed-attempt receipt](evidence/training-attempt1-command.json) and [single-setting overlay](evidence/train-memory03.jsonnet).

The second attempt passed that failure point. It generated the 512 initial responses, completed 1,066 distinct intermediate-state continuation requests, and finished two training epochs: **16 actual optimizer updates**. The recorded loss and gradient norm were finite, with zero reported NaN/Inf loss anomalies. Updated weights were saved and evaluated. This is a successful first training iteration, not a completed 690-iteration experiment. See the [first-update receipt](evidence/first-training-iteration.json).

The first own-model test produced **8,444 correct answers out of 21,104**, compared with **8,394** before training: **40.01% versus 39.77%**, or **+0.24 percentage points**. We rechecked all question identities, all answers, and all seven native metrics. This small difference could reflect sampling variation; it is too early to infer a useful training effect. The same checkpoint scored 53.08% on the separate 373-question validation split, but that is not comparable to the test score and we have not measured its matched starting-model validation score. We retain these observations without choosing a checkpoint or changing the recipe. See the [evaluation and regrading receipt](evidence/first-own-evaluation.json).

The next scheduled test, after **176 updates**, produced **8,694/21,104 correct answers (41.20%)**. Its **+1.42 percentage-point** gain over the starting model is encouraging, but it is one intermediate observation from one training run. All test and validation answers were regraded with the native evaluator; validation was 54.41%, on its separate question distribution. We keep the endpoint fixed at 690 collections. The log calls this evaluation `iteration__10`, because iteration numbering begins at zero; the evaluated weights have completed **11** collections. See the [176-update evidence](evidence/progress-176-updates.json).

At 160 updates, we also saved and verified a full **17.6 GB model/optimizer/scheduler checkpoint**. Copying it inside project storage filled the project's separate 1 TB quota—an operational mistake: the filesystem-wide free-space display did not show that quota. Training subsequently stalled during evaluation cleanup. We relocated only the new backup to the home research store, preserved the interrupted attempt, and restarted from the full 160-update state in a fresh output directory. The original files were not moved or deleted. No old evaluation or rollout cache was reused, and configuration equality was checked apart from output paths. The completed 176-update test remains valid, but the repeated work after resumption is not a new independent training run. Exact random-state continuity is not established. See the [checkpoint hashes and relocation map](evidence/checkpoint-160-preservation.json).

The resumed branch's scheduled 176-update test scored **8,704/21,104 (41.24%)**, only ten more correct answers than the interrupted branch. All seven native metrics were rechecked from the saved answers on test, validation, and the training subset. This is consistent with the earlier positive signal, but it does not establish a benefit from restarting or an independent replication. We retain both observations and continue the same recipe. See the [resumed evaluation and training evidence](evidence/resumed-176-evaluation.json).

The next scheduled test, at **336 updates**, scored **8,956/21,104 (42.44%)**. We regraded every saved test answer and all seven metrics. Validation also rose, from 54.05% at the resumed 176-update checkpoint to **55.56% (3,316/5,968)**, with the same checks. The agreement across these scheduled observations supports continued learning, but neither establishes the final paper result. We keep the recipe and endpoint unchanged. See the [336-update review](evidence/progress-336-updates.json).

The full **320-update checkpoint** is now protected outside the native cleanup directory, with all ten copied files verified by checksum. It includes optimizer and learning-rate scheduler state; it is distinct from the 336-update weights evaluated above. We checked the home-storage quota before copying and retained only the initial and latest recovery points. See the [preservation manifest](evidence/checkpoint-320-preservation.json).

At **496 updates**, the scheduled test reached **9,193/21,104 (43.56%)**. Validation rose to **56.94% (3,398/5,968)**. All question identities, saved answers, and seven native metrics were checked on test, validation, and the training subset. The training-subset score is a diagnostic, not held-out performance. The full **480-update model/optimizer/scheduler state** was also saved and inspected; at that review, the separately protected recovery copy was at 320 updates. See the [496-update review](evidence/progress-496-updates.json).

At **656 updates**, the scheduled test reached **9,331/21,104 (44.21%)**. Validation was **56.99% (3,401/5,968)**—just three more correct answers than before, so effectively flat. Again, all seven metrics were regraded on all three splits. The consecutive test increment is modest; we do not claim each interval separately proves improvement. The full **640-update model/optimizer/scheduler checkpoint** is now protected outside native cleanup, with all ten copied files matching checksums. Earlier protected states remain; nothing was deleted. See the [656-update review](evidence/progress-656-updates.json) and [640-update preservation manifest](evidence/checkpoint-640-preservation.json).

At **816 updates**, the scheduled test reached **9,529/21,104 (45.15%)**, and validation improved to **58.43% (3,487/5,968)**. Saved answers and all seven native test and validation metrics were regraded. This supports the continuing positive learning signal, but remains an intermediate observation from one resumed run. A new native full checkpoint exists at 800 updates; the separately protected recovery copy remains at 640. See the [816-update review](evidence/progress-816-updates.json).

At **976 updates**, the scheduled test reached **9,674/21,104 (45.84%)**, and validation rose to **59.65% (3,560/5,968)**. Every saved test and validation answer and all seven native metrics were regraded. The test gain is **+6.07 percentage points** versus the starting model and **+0.69 points** since 816 updates. This strengthens the evidence for early learning in this run; it does not establish that each small consecutive increment is a repeatable effect. We retain the fixed endpoint and all earlier results. A full native checkpoint was saved at 960 updates; the separately protected recovery copy remains at 640. See the [976-update review](evidence/progress-976-updates.json).

At **1,136 updates**, the scheduled test reached **9,858/21,104 (46.71%)**. Validation was **59.62% (3,558/5,968)**, two fewer correct answers than the preceding test—essentially flat. Every saved test and validation answer and all seven native metrics were regraded. Test accuracy is **+6.94 points** above the starting model, but the latest increment is not uniform across splits. This is evidence of continued progress in one run, not a reason to change the recipe or declare the endpoint reproduced. A full native checkpoint was saved at 1,120 updates; the separately protected copy remains at 640, with the next coarse backup planned at 1,280. See the [1,136-update review](evidence/progress-1136-updates.json).

At **1,296 updates**, the scheduled test scored **9,836/21,104 (46.61%)**: 22 fewer correct answers than at 1,136 updates, or **−0.10 percentage points**. Validation rose to **60.67% (3,621/5,968)**, with 63 more correct answers. All question identities, saved answers and seven native metrics were checked on all three splits. The small test dip does not establish a reversal, and the validation rise does not make improvement uniform. We keep both observations and continue the fixed recipe without choosing the better test checkpoint. See the [1,296-update review](evidence/progress-1296-updates.json).

The full **1,280-update model/optimizer/scheduler checkpoint** is now preserved outside training cleanup. All ten files, totaling 17.6 GB, match their source checksums. We checked the home-storage quota first; earlier recovery points remain and nothing was deleted or moved. This recovery state is distinct from the 1,296-update weights evaluated above. Next coarse protection is planned at 2,560 updates, or sooner if interruption risk warrants it, with another quota and retention check. See the [1,280-update preservation manifest](evidence/checkpoint-1280-preservation.json).

At **1,456 updates**, the scheduled test scored **9,985/21,104 (47.31%)**, up **0.71 percentage points** since the preceding test. Validation rose to **61.36% (3,662/5,968)**. Every saved test and validation answer and all seven native metrics were regraded. This supports continued learning, but neither proves every consecutive gain is repeatable nor completes the paper reproduction. A full native checkpoint was saved at 1,440 updates; the separately protected recovery state remains at 1,280. See the [1,456-update review](evidence/progress-1456-updates.json).

At **1,616 updates**, the scheduled test scored **10,017/21,104 (47.46%)**, just 32 more correct answers than at 1,456 updates (**+0.15 percentage points**). Validation rose to **61.90% (3,694/5,968)**, also 32 more correct answers. Every saved test and validation answer and all seven native metrics were regraded. The overall gain remains encouraging, but this small consecutive test increment does not separately establish improvement, nor does it establish a plateau. We keep the recipe and endpoint fixed. A full native checkpoint was saved at 1,600 updates; the separately protected recovery state remains at 1,280. See the [1,616-update review](evidence/progress-1616-updates.json).

At **1,776 updates**, the scheduled test scored **10,095/21,104 (47.83%)**, 78 more correct answers than at 1,616 updates (**+0.37 percentage points**). Validation rose to **62.40% (3,724/5,968)**, 30 more correct answers. Every saved test and validation answer and all seven native metrics were regraded. This continues the overall positive learning curve, without establishing that this small consecutive gain is independently reliable. A full native checkpoint was saved at 1,760 updates; the separately protected recovery state remains at 1,280. See the [1,776-update review](evidence/progress-1776-updates.json).

At **1,936 updates**, the scheduled test fell to **9,842/21,104 (46.64%)**, 253 fewer correct answers than at 1,776 updates (**−1.20 percentage points**). Validation also fell to **61.13% (3,648/5,968)**, 76 fewer correct answers. Every saved test and validation answer and all seven native metrics were regraded. This dip is larger than the earlier 0.10-point dip and appears in both splits, so we should not simply dismiss it as noise. It does not, by itself, establish a lasting reversal or its cause. Training rewards and gradients still pass their checks; answer-extraction failures did not increase. The smallest useful follow-up is the next prescribed evaluation under the unchanged recipe. A full native checkpoint was saved at 1,920 updates; the protected recovery state remains at 1,280. See the [1,936-update review](evidence/progress-1936-updates.json).

At **2,096 updates**, the scheduled test rebounded to **10,127/21,104 (47.99%)**, 285 more correct answers than at 1,936 updates (**+1.35 percentage points**). Validation recovered only 20 correct answers, to **61.46% (3,668/5,968)**, still below 62.40% at 1,776 updates. Every saved test and validation answer and all seven native metrics were regraded. The test decline did not persist into this scheduled observation, but the new score is only **0.15 points above the pre-dip test**. We should not turn the rebound into a claim of steady progress or a known cause. Continue the fixed recipe and retain all observations. A full native checkpoint was saved at 2,080 updates; the protected recovery state remains at 1,280. See the [2,096-update review](evidence/progress-2096-updates.json).

At **2,256 updates**, the scheduled test rose to **10,244/21,104 (48.54%)**, 117 more correct answers (**+0.55 percentage points**). Validation fell by 35 correct answers to **60.87% (3,633/5,968)**. All saved answers and seven native metrics were regraded on test, validation, and the training subset. A focused comparison of the validation questions found 88 improved, 111 worsened, and 174 unchanged; missing extracted answers decreased from four to one. The decline is therefore not explained by increased extraction failures or just one difficult question. Its descriptive paired standard error is 0.71 points versus a 0.59-point decline, so this comparison alone does not establish a persistent deterioration or its cause. We retain the conflicting split results and the unchanged recipe. Full native state exists at 2,240 updates; protected recovery state remains at 1,280. See the [2,256-update review and validation diagnostic](evidence/progress-2256-updates.json).

At **2,416 updates**, the scheduled test scored **10,234/21,104 (48.49%)**, just ten fewer correct answers (**−0.05 percentage points**). Validation edged down by 17 correct answers to **60.59% (3,616/5,968)**, while the training subset rose to **71.12% (4,233/5,952)**. All saved answers and seven native metrics were regraded on all three splits. The test is essentially flat; this does not establish a plateau. The training/validation divergence is worth watching, but does not yet explain the change or establish overfitting. Continue the unchanged recipe and next prescribed evaluation. Full native state exists at 2,400 updates; protected recovery state remains at 1,280. See the [2,416-update review](evidence/progress-2416-updates.json).

At **2,576 updates**, the scheduled test scored **10,289/21,104 (48.75%)**, 55 more correct answers (**+0.26 percentage points**). Validation recovered 52 correct answers to **61.46% (3,668/5,968)**; the training subset rose to **71.74% (4,270/5,952)**, which is not held-out accuracy. All saved answers and seven native metrics were regraded on all three splits. The validation rebound weakens a simple story of uninterrupted deterioration, but remains below 62.40% at 1,776 updates. The small consecutive test increase does not independently establish a reliable gain. Keep the recipe and endpoint fixed. Full native state exists at 2,560 updates, distinct from the evaluated 2,576-update weights. See the [2,576-update review](evidence/progress-2576-updates.json).

At **2,736 updates**, the scheduled test scored **10,407/21,104 (49.31%)**, 118 more correct answers (**+0.56 percentage points**). Validation rose by 65 correct answers to **62.55% (3,733/5,968)**—only nine more than at 1,776 updates, so best described as recovering its earlier range. The training subset slipped by 26 correct answers to **71.30% (4,244/5,952)**; this is not held-out accuracy. All saved answers and seven native metrics were regraded on all three splits. Test and validation improved this time, weakening a simple story of persistent validation deterioration. This remains one ongoing training ancestry; the consecutive test increment alone does not establish a reliable gain or its cause. Full native state exists at 2,720 updates; the protected recovery state remains at 2,560. See the [2,736-update review](evidence/progress-2736-updates.json).

At **2,896 updates**, the scheduled test scored **10,547/21,104 (49.98%)**, 140 more correct answers (**+0.66 percentage points**) than at 2,736 updates. This brings the measured gain over our matched starting model to **10.20 points**. Validation rose by 43 correct answers to **63.27% (3,776/5,968)**; the training subset rose by 22 to **71.67% (4,266/5,952)**, which is not held-out accuracy. All original question identities, saved answers and seven native metrics were checked on all three splits. Improvement on both test and validation supports continued learning and weakens the earlier concern about persistent deterioration. This remains intermediate progress from one ongoing training ancestry, not a completed reproduction or a causal explanation. Full native state exists at 2,880 updates; the protected recovery state remains at 2,560. See the [2,896-update review](evidence/progress-2896-updates.json).

At **3,056 updates**, the scheduled test scored **10,640/21,104 (50.42%)**, 93 more correct answers (**+0.44 percentage points**) than at 2,896 updates. The gain over our matched starting model is now **10.64 points**. Validation rose by 61 correct answers to **64.29% (3,837/5,968)**; the training subset rose by 56 to **72.61% (4,322/5,952)**, which is not held-out accuracy. All original question identities, saved answers and seven native metrics were checked on all three splits. Joint test/validation improvement continues this time, weakening the earlier deterioration account without establishing a cause or guaranteeing steady future progress. This remains one ongoing training ancestry, not a completed reproduction. Full native state exists at 3,040 updates; the protected recovery state remains at 2,560. See the [3,056-update review](evidence/progress-3056-updates.json).

At **3,216 updates**, the scheduled test scored **10,709/21,104 (50.74%)**, 69 more correct answers (**+0.33 percentage points**) than at 3,056 updates. The gain over our matched starting model is **10.97 points**. Validation slipped by 16 correct answers to **64.02% (3,821/5,968)**; the training subset slipped by 50 to **71.77% (4,272/5,952)**, which is not held-out accuracy. All original question identities, saved answers and seven native metrics were checked on all three splits. The latest changes are mixed: the earlier joint improvement does not continue uniformly, but these small differences do not establish a reversal or its cause. Keep the prescribed recipe and endpoint. Full native recovery state exists at 3,200 updates; the protected recovery state remains at 2,560. See the [3,216-update review](evidence/progress-3216-updates.json).

At **3,376 updates**, the scheduled test scored **10,670/21,104 (50.56%)**, 39 fewer correct answers (**−0.18 percentage points**) than at 3,216 updates. The gain over our matched starting model is **10.78 points**. Validation rose by 91 correct answers to **65.55% (3,912/5,968)**. The training-subset evaluation, completed and checked at 01:45 UTC, scored **73.03% (4,347/5,952)**, up 75 correct answers (**+1.26 points**); this is not held-out performance. All original question identities, saved answers and seven native metrics have now been checked on all three splits. The small test dip does not establish a reversal, and the validation gain argues against interpreting it as general deterioration. Neither proves a cause or completes the reproduction. Keep the prescribed recipe and endpoint. Full native recovery state exists at 3,360 updates; the protected recovery state remains at 2,560. See the [initial 3,376-update review](evidence/progress-3376-updates.json) and [completed training-subset check](evidence/progress-3376-completion.json).

At **3,536 updates**, the scheduled test scored **10,802/21,104 (51.18%)**, 132 more correct answers (**+0.63 percentage points**) than at 3,376 updates. The gain over our matched starting model is **11.41 points**. Validation slipped by 22 correct answers to **65.18% (3,890/5,968)**; the training subset rose by 43 to **73.76% (4,390/5,952)**, which is not held-out performance. All original question identities, saved answers and seven native metrics were checked on all three splits. The previous small test dip did not persist, but the split directions differ again: neither a uniform gain nor a causal explanation is established. This remains one ongoing training ancestry, not a completed reproduction. Keep the prescribed recipe and endpoint. Full native recovery state exists at 3,520 updates; the protected recovery state remains at 2,560. See the [3,536-update review](evidence/progress-3536-updates.json).

At **3,696 updates**, the scheduled test scored **10,747/21,104 (50.92%)**, 55 fewer correct answers (**−0.26 percentage points**) than at 3,536 updates. The gain over our matched starting model is **11.15 points**. Validation fell by 66 correct answers to **64.08% (3,824/5,968)**, its second consecutive decline. The training-subset evaluation, completed and checked at 03:21 UTC, rose by 19 correct answers to **74.08% (4,409/5,952)**; this is not held-out performance. All original question identities, saved answers and seven native metrics have now been checked on all three splits. A question-level validation comparison found 83 improved, 98 worsened and 192 unchanged. Missing extracted answers decreased from two to one, so increased extraction failures do not explain this decline. The training/held-out divergence warrants checking persistence, but does not establish overfitting, a lasting reversal or its cause. Keep the prescribed recipe and endpoint. Full native recovery state exists at 3,680 updates; the protected recovery state remains at 2,560. See the [initial 3,696-update review and diagnostics](evidence/progress-3696-updates.json) and [completed training-subset check](evidence/progress-3696-completion.json).

At **3,856 updates**, the scheduled test scored **10,642/21,104 (50.43%)**, 105 fewer correct answers (**−0.50 percentage points**) than at 3,696 updates. The gain over our matched starting model is **10.65 points**. Validation rebounded by 31 correct answers to **64.59% (3,855/5,968)**, partially recovering its previous decline. The training subset rose by 56 to **75.02% (4,465/5,952)**, which is not held-out performance. All original question identities, saved answers and seven native metrics were checked on all three splits. The second test decline merits continued attention, but the validation rebound argues against a simple claim of uniform deterioration. Missing extracted test answers decreased from 17 to 11, so increased extraction failures cannot explain the test dip. Keep the prescribed recipe and endpoint; no overfitting or causal explanation is established. Full native recovery state exists at 3,840 updates; the protected recovery state remains at 2,560. See the [3,856-update review and diagnostics](evidence/progress-3856-updates.json).

At **4,016 updates**, the scheduled test scored **10,795/21,104 (51.15%)**, 153 more correct answers (**+0.72 percentage points**) than at 3,856 updates. The gain over our matched starting model is **11.38 points**. This ends the two-test decline, but is only seven correct answers below the 3,536-update score: a recovery to the earlier range, not a new established advance. Validation is essentially flat at **64.46% (3,847/5,968)**, eight fewer correct answers (**−0.13 points**). The training-subset evaluation, completed and checked at 04:53 UTC, fell by 56 correct answers to **74.08% (4,409/5,952)**; this is not held-out performance. All original question identities, saved answers and seven native metrics have now been checked on all three splits. Test recovery, essentially flat validation and falling training-subset accuracy give mixed directions, not a uniform improvement or a causal explanation. Keep the prescribed recipe and endpoint. Full native recovery state exists at 4,000 updates; the protected recovery state remains at 2,560. See the [initial 4,016-update review and diagnostics](evidence/progress-4016-updates.json) and [completed training-subset check](evidence/progress-4016-completion.json).

At **4,176 updates**, the scheduled test scored **10,773/21,104 (51.05%)**, 22 fewer correct answers (**−0.10 percentage points**) than at 4,016 updates. The gain over our matched starting model is **11.27 points**. Validation fell by 38 correct answers to **63.82% (3,809/5,968)**, or **−0.64 points**. All original question identities, saved answers and seven native metrics were checked on test and validation. The training-subset result was pending at the earlier 05:30 UTC cutoff. It is now complete and independently checked: **4,469/5,952 (75.08%)**, up 60 correct answers (**+1.01 points**) from 4,016 updates. All 372 original training-question identities, saved tree answers and seven metrics match. Test and validation hashes remain unchanged. Training accuracy rising while validation falls is worth tracking, but does not establish overfitting. See the [completion review](evidence/progress-4176-completion.json); the earlier pending-cutoff receipt is preserved. The test remains near its earlier range, but this is not enough to establish a plateau or a lasting reversal. Native training diagnostics show renewed finite excursions, described below, without establishing a cause of the score changes. Keep the prescribed recipe and endpoint. Full native recovery state exists at 4,160 updates; the protected recovery state remains at 2,560. See the [4,176-update review and safeguard diagnostic](evidence/progress-4176-updates.json).

At **4,336 updates**, the scheduled test scored **10,662/21,104 (50.52%)**, 111 fewer correct answers (**−0.53 percentage points**) than at 4,176 updates. The gain over our matched starting model is **10.75 points**. Validation edged up by 10 correct answers to **63.99% (3,819/5,968)**, or **+0.17 points**. All original question identities, saved tree answers and seven metrics were checked on test and validation. The training-subset result, pending at the earlier 06:16 UTC cutoff, is now complete: **4,483/5,952 (75.32%)**, up 14 correct answers (**+0.24 points**). All 372 original identities, saved tree candidates and seven metrics were independently checked. Test and validation hashes are unchanged. See the [completion review](evidence/progress-4336-completion.json); the earlier pending-cutoff receipt remains preserved. This is the second consecutive test decline, but earlier dips recovered and validation moves differently. Neither a lasting reversal nor overfitting is established. Larger numerical excursions deserve follow-up without assuming they caused the dip. Keep the prescribed recipe and endpoint. Native recovery state exists at 4,320 updates; protected recovery remains at 2,560. See the [4,336-update review and focused diagnostic](evidence/progress-4336-updates.json).

**A useful distinction:** single-answer accuracy, question coverage and majority voting need not move together. This time, test questions solved at least once in 16 attempts fell from **1,047 to 1,025 out of 1,319**, while majority-vote accuracy fell from **59.21% to 56.79%**. Average distinct answers changed little, from 4.61 to 4.61 when rounded. Missing extracted answers fell from 16 to 14, so increased extraction failures cannot explain the test dip. These are descriptive checks, not causal explanations. We retain all metrics without substituting the more favorable one. The paper comparison remains single-answer accuracy.

## What exactly is the score?

The native evaluator samples 16 answers for each of the 1,319 held-out questions. It scores each answer with the authors' answer-extraction and comparison rule, then averages the fraction correct across questions. Because every question has 16 answers, this also equals total correct answers divided by 21,104.

This is an estimate of **single-answer accuracy (pass@1)**. It is **not** the fraction of questions solved at least once in 16 tries, and it is not majority voting. Those other scores are saved but are not used for the paper comparison. Evaluation uses the authors' prompt, temperature 0.35, and generation limits.

We preserved the authors' actual data split: **7,100 training, 373 validation, and 1,319 test questions**. We found no identical questions shared between these splits. Sixteen answers per test question do not give us sixteen independent training runs.

## What “faithful reproduction” requires here

The audit found that today's README defaults differ from the original experiment logs. We recovered the historical configuration instead of assuming the newest example command was the published recipe. We also found a decoding discrepancy: the written evaluation configuration says top-p 0.9, but the actual historical template uses the library default 1.0. For the reference comparison we preserve the effective native behavior and document the discrepancy.

The recorded result matching 56.7% comes from checkpoint **690**, after **11,040 optimizer updates**, within a **1,000-iteration learning-rate schedule**. We fixed checkpoint 690 as our comparison point before training; we will not choose whichever local test checkpoint looks best. The original authors' reason for selecting that checkpoint and their exact historical source revision remain unresolved. Consequently, this is an attempt to reproduce the recovered experiment—not yet a claim of exact historical or bit-for-bit equivalence.

The original run segments took about **60 hours** in total. Our current allocation had roughly **47 hours** left at launch. Nine measured non-evaluation iterations average 4.33 minutes each, which would imply about 49.8 hours for 690 collections **before periodic evaluations**, if that rate persisted. Continuation beyond this allocation is therefore likely. If time runs out, we will report an incomplete run and preserve available checkpoints, rather than shorten the learning-rate schedule and call it the same experiment. The restart retains the original cumulative training cap; it does not grant a fresh 45 hours.

Full technical provenance, caveats, and source links are in the [protocol audit](protocol-audit.md). The [environment notes](evidence/environment.md), [environment lock](evidence/environment-lock.txt), [data provenance](evidence/dataset-provenance.json), and [recovered training settings](evidence/author-training-config.json) are available for inspection.

## Unattended operation and the next update

Jobs run serially under detached, time-capped owners. The observer requests a research review when results arrive and approximately hourly otherwise. Reviews examine actual answers, rewards, updates, failures, and checkpoints; useful milestones are documented and pushed to GitHub. Dispatch stops before consuming the reserved account allowance. A failed command stops its queue so that the next review can diagnose it rather than silently change the recipe.

At the 06:30 UTC native-data check, the resumed run had completed **4,384 updates**. In the fully inspected collection, 317 of 512 responses earned a positive reward and 195 earned zero; all rewards were checked against the native grader. Saved advantages and model log probabilities were finite. One actual response multiplied by five twice when counting four groups of five actors, answering 100 instead of 20. It correctly earned zero. These are changing training questions, not a matched benchmark learning curve. The next collection had 486 successful HTTP responses, no HTTP 400/500 errors, and real saved model answers at the check.

Numerical excursions need continued attention. After the earlier gradient norm of 242, two much larger reported norms appeared: **2,290 and 6,850 at updates 4,271–4,272**, followed by **17.5 and 36.9**. At 4,336 updates the reported norm was 15.1; at 4,384 it was 12.8. We checked the saved advantages and log probabilities in the affected collection and its two neighbors: all were finite, and all 512 terminal rewards in the affected collection regraded correctly. No new average-ratio safeguard warning fired; reported loss NaN/Inf counts remain zero. Gradient clipping remains configured at 1, but the logged norm is not a measurement of parameter displacement. These checks do not establish the spikes' cause or their effect on test accuracy. We will track recurrence and investigate the affected batches without changing the live recipe. The earlier safeguard zeros affected loss contributions rather than removing entire optimizer steps from the counter. Native recovery state exists at 4,320 updates, distinct from the evaluated 4,336-update weights. A further 48 recorded updates had a maximum gradient norm of 57.5, with no new ratio warnings or reported nonfinite losses. This limited interval does not explain the earlier spikes or guarantee future stability.

The full **2,560-update model and optimizer state** is now copied outside the trainer's cleanup directory. All ten source/backup file checksums match; earlier recovery states remain preserved. This is a recovery checkpoint, not the evaluated 2,576-update state. Next coarse protection is at 5,120 updates, or sooner if interruption risk rises.

Cluster resume pointer: `/project/alex_phd/runs/spo-reproduction-20261008/SESSION_CHECKPOINT.md`. GitHub contains source, reports, configuration, and small evidence receipts—not model-weight backups.
