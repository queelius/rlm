# Reproducing an LLM reinforcement-learning experiment on one A100

Status: 8 October 2026, 08:15 UTC. **The released author model scored 56.69%, matching the paper's 56.7% to one decimal place. Our fresh training has completed its first 16 optimizer updates; no benchmark improvement from our own weights has been measured yet.** This campaign follows the request to reproduce a published experiment faithfully, rather than treat an encouraging small pilot as an exact reproduction.

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
| Fresh SPO training from the original starting model | **16 updates completed** in attempt 2 | Training path works; own-model benchmark improvement not yet measured |
| Matched GRPO training control | [Provenance audited; further preparation needed](grpo-control-plan.md) | Current repo defaults and reward handling cannot be assumed to reproduce the historical baseline |

The starting-model run saved all **1,319 questions × 16 answers = 21,104 answers**. We checked every question against the original test split and recomputed all seven native metrics exactly. Its launcher returned an error **after** scoring and cleanup because it passed an unsupported command-line argument. That command failure is retained separately; subsequent launches use DeepSpeed's supported `--no_local_rank` option. We did not change model outputs or the grader to make the score look better. See [native metrics](evidence/base-reference-native.json) and [command receipt](evidence/base-reference-command.json).

The released-author evaluation also covered all 21,104 answers, with every question identity and native score recomputed, and exited successfully. It scored 56.68594%, versus the historical log's 56.72384% (a difference of 0.038 percentage points). The difference between our two **reference models** is 16.91 percentage points. That confirms we can measure the released models' performance difference; it does not show that our fresh training has reproduced it. See [author-model metrics](evidence/author-reference-native.json), [command receipt](evidence/author-reference-command.json), and [validation receipt](evidence/author-reference-validation.json).

One concrete observation: on the first test question, the starting model gave the correct answer in 4 of 16 attempts. It also made identifiable arithmetic and interpretation mistakes. This illustrates usable variation in outcomes; it is not evidence of an RL improvement, because these are the **starting** weights.

The first training attempt generated 512 responses, then ran out of GPU memory while calculating their model probabilities. The inference server had reserved 26.27 GiB alongside the trainer. We preserved the failure and restarted from the same original weights with the inference-memory reservation limited to 30%. This changes resource allocation, not the dataset, training batch, objective, sampling settings, or learning-rate schedule. Configuration equality was checked after excluding only that memory setting and output paths. It may affect inference scheduling, so it is documented—not presented as bit-for-bit equivalence. See the [failed-attempt receipt](evidence/training-attempt1-command.json) and [single-setting overlay](evidence/train-memory03.jsonnet).

The second attempt passed that failure point. It generated the 512 initial responses, completed 1,066 distinct intermediate-state continuation requests, and finished two training epochs: **16 actual optimizer updates**. The recorded loss and gradient norm were finite, with zero reported NaN/Inf loss anomalies. Updated weights are being saved. This is a successful first training iteration, not a completed 690-iteration experiment or a benchmark gain. See the [first-update receipt](evidence/first-training-iteration.json).

## What exactly is the score?

The native evaluator samples 16 answers for each of the 1,319 held-out questions. It scores each answer with the authors' answer-extraction and comparison rule, then averages the fraction correct across questions. Because every question has 16 answers, this also equals total correct answers divided by 21,104.

This is an estimate of **single-answer accuracy (pass@1)**. It is **not** the fraction of questions solved at least once in 16 tries, and it is not majority voting. Those other scores are saved but are not used for the paper comparison. Evaluation uses the authors' prompt, temperature 0.35, and generation limits.

We preserved the authors' actual data split: **7,100 training, 373 validation, and 1,319 test questions**. We found no identical questions shared between these splits. Sixteen answers per test question do not give us sixteen independent training runs.

## What “faithful reproduction” requires here

The audit found that today's README defaults differ from the original experiment logs. We recovered the historical configuration instead of assuming the newest example command was the published recipe. We also found a decoding discrepancy: the written evaluation configuration says top-p 0.9, but the actual historical template uses the library default 1.0. For the reference comparison we preserve the effective native behavior and document the discrepancy.

The recorded result matching 56.7% comes from checkpoint **690**, after **11,040 optimizer updates**, within a **1,000-iteration learning-rate schedule**. We fixed checkpoint 690 as our comparison point before training; we will not choose whichever local test checkpoint looks best. The original authors' reason for selecting that checkpoint and their exact historical source revision remain unresolved. Consequently, this is an attempt to reproduce the recovered experiment—not yet a claim of exact historical or bit-for-bit equivalence.

The original run segments took about **60 hours** in total. Our current allocation has roughly **47 hours** left at launch. Actual throughput may differ. If time runs out, we will report an incomplete run and preserve available checkpoints for continuation, rather than shorten the learning-rate schedule and call it the same experiment.

Full technical provenance, caveats, and source links are in the [protocol audit](protocol-audit.md). The [environment notes](evidence/environment.md), [environment lock](evidence/environment-lock.txt), [data provenance](evidence/dataset-provenance.json), and [recovered training settings](evidence/author-training-config.json) are available for inspection.

## Unattended operation and the next update

Jobs run serially under detached, time-capped owners. The observer requests a research review when results arrive and approximately hourly otherwise. Reviews examine actual answers, rewards, updates, failures, and checkpoints; useful milestones are documented and pushed to GitHub. Dispatch stops before consuming the reserved account allowance. A failed command stops its queue so that the next review can diagnose it rather than silently change the recipe.

The next milestones are the first own-model evaluation, steady-state iteration time, and the first permanent optimizer checkpoint. A completed reference check, an intermediate training score, and a completed reproduction will remain clearly distinguished.

Cluster resume pointer: `/project/alex_phd/runs/spo-reproduction-20261008/SESSION_CHECKPOINT.md`. GitHub contains source, reports, configuration, and small evidence receipts—not model-weight backups.
