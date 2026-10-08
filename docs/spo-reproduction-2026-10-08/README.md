# Reproducing an LLM reinforcement-learning experiment on one A100

Status: 8 October 2026, 12:22 UTC. **Our weights scored 44.21% after 656 optimizer updates, versus 39.77% before training: +4.44 percentage points.** This is continued early improvement, not a completed reproduction. The same resumed run scored 41.24%, 42.44%, and 43.56% at earlier scheduled evaluations; all results are retained. Training continues with the scientific recipe unchanged. The released author model scored 56.69%, matching the paper's 56.7% reference.

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

**A useful distinction:** single-answer accuracy improved, but the number of questions solved at least once in 16 attempts fell from **1,013 to 996 to 986 out of 1,319** across the last three tests. The average number of distinct answers also fell, from 5.85 to 5.66 to 5.38. This could reflect answers becoming more concentrated, or sampling variation; it does not establish a loss of capability. We track both views without substituting the more favorable metric. The paper comparison remains single-answer accuracy.

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

The resumed run has completed updates through 656 with finite diagnostics. In the latest inspected 512-response training collection, 305 responses earned a positive reward and 207 earned zero; all rewards were checked against the native grader. Saved advantages and model log probabilities were finite. These are changing training questions, not a matched benchmark learning curve. Next are later prescribed evaluations and refreshed checkpoint protection within both storage quotas. A completed reference check, an intermediate training score, an interrupted attempt, and a completed reproduction remain clearly distinguished.

Cluster resume pointer: `/project/alex_phd/runs/spo-reproduction-20261008/SESSION_CHECKPOINT.md`. GitHub contains source, reports, configuration, and small evidence receipts—not model-weight backups.
