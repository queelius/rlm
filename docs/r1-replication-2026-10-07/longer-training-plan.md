# Moving from a pipeline check toward reproducing the training result

Current protocol, 7 October 2026 at 15:25 UTC. Attempt 3 started at 15:24 UTC.
This is an exploratory reproduction, not a new algorithm. Results are pending.

Two earlier attempts failed and are retained. [Attempt 1](longer-attempt1-failure.json)
ran out of GPU memory during its first collection's optimization. [Attempt 2](longer-attempt2-failure.json)
tested an expandable allocator, but that setting is incompatible with the
generation engine's memory pool. Neither produced a completed training result.

Attempt 3 uses the earlier successful collection size of 16 questions and the
default allocator. The dataset, learning rule, prompt, checker, and learning
rate are unchanged. This changes how often updated weights reach the generator
relative to the unsuccessful large-collection attempt; it is not just a memory
implementation change.

The native loader excludes two selected questions longer than 1,024 input
tokens. Of the remaining 4,094, it uses 255 complete 16-question batches and
drops the final 14 questions. Thus the fixed effective budget is **4,080
questions, 32,640 responses, and 255 optimizer updates**, with final checkpoint
`step_00256`. These counts follow from the loader, not from any test score.
The [earlier loader receipt](longer-loader-receipt.json) identifies the same
two excluded rows and preserves the abandoned large-collection accounting.

## What the short runs left unresolved

Our first chat-style run rose from 30.8% to 61.6% on MATH500. But the original
model already scored 61.0% when given the question without the chat wrapper.
The short runs therefore established a working training pipeline and strong
prompt sensitivity, not the paper's larger learning improvement.

We next evaluated the authors' released trained model using our evaluator.
Its native score was 366/500, or 73.2%, against the paper's 74.2%. This is a
check of the released model and evaluation setup. **We did not train that model.**
The separate [scoring receipt](author-reference-receipt.json) records the checks.
It makes insufficient training a more useful hypothesis to investigate, but
does not rule out differences in our training settings or software versions.

## The next experiment

**Question:** Does a longer run, closer to the authors' released-model recipe,
produce a useful gain beyond our stronger untrained baseline?

| Setting | New run |
| --- | --- |
| Starting weights | Original Qwen2.5-Math-1.5B base, not an earlier trained checkpoint |
| Training data | 4,096 selected from official 8,523 MATH level 3–5 questions; 4,080 used after native filtering/batching |
| Selection | First 4,096 rows after a Python Random(42) shuffle |
| Learning method | Authors' unchanged Dr. GRPO implementation, full-model BF16 updates |
| Input and reward | Qwen-Math conversation template; `math_verify` answer checker |
| Sampling | Eight attempts per question, temperature 1, up to 3,000 response tokens |
| Training budget | 255 collections, 255 optimizer updates, 32,640 sampled attempts |
| Learning rate | Constant 0.000001 |
| Hardware and limit | One A100 40 GB; 10-hour training cap |
| Saved state | Every 32 updates; keep the latest two model and optimizer checkpoints |
| Final evaluation | All 500 MATH500 questions, in both chat and question-only formats |

Each collection contains 16 questions and 128 responses. The learner takes
one optimizer update, then sends its new weights to the generator, as in the
successful short runs. The native final folder is named `step_00256`; it means
final bookkeeping after 255 updates, **not** a 256th weight update.

The unchanged official source is commit
`dfca49dd460ee7cc8e4a5a162c876a7fd6993b87`. Base weights use revision
`4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2`.
The learner seed is 42; the original actor uses time-based randomness, so this
is not a promise of an exactly repeatable training trajectory.

Compared with the short runs, we changed data coverage and difficulty,
training duration, and the reward checker to move closer to the reference recipe.
Consequently, this run cannot tell us which change caused any improvement.
That is a later comparison if the stronger recipe works.

## How we will decide what to do afterward

1. Finish the fixed training budget and evaluate its final weights in both
   input formats. Intermediate scores help diagnose learning but do not choose
   a checkpoint. Preserve failures and partial runs separately.
2. Inspect what changed: newly correct and newly wrong answers, completion
   rates, answer lengths, mixed-success training groups, and actual updates.
3. If the gain is promising, repeat the training and test another benchmark.
   If it is weak, use the diagnostic evidence to distinguish insufficient
   coverage, weak learning signals, small updates, or damaging updates before
   choosing the next run. Do not simply keep testing checkpoints until one wins.

MATH500 has now been examined repeatedly and is a development benchmark for
this campaign. It contains no exact training-question overlap, but that alone
does not make repeated experimentation on it an independent confirmation.
Repeating evaluation of the same weights is also not a new training replicate.

## What would count as reproducing the paper?

There are three distinct claims, and we should not substitute one for another:

1. **Released-model evaluation:** Run the authors' saved model and approach its
   reported score. Our 73.2% versus 74.2% is evidence for this limited claim;
   we have not explained the remaining five-answer difference.
2. **Learning from the original model:** Train from the base weights and obtain
   a dependable improvement beyond the strongest untrained input format.
   This is the immediate priority. Fresh training repeats, not repeated tests
   of the same model, check whether the gain survives training randomness.
3. **The proposed method's advantage:** Under a matched training budget, compare
   Dr. GRPO with GRPO for both accuracy and response length, especially length
   of wrong answers. A high math score alone does not reproduce this claim.

Before inspecting the new endpoint, rank AMC (83 questions) and Minerva
(272 questions) as the first broader checks using the authors' existing data.
Compare the untrained model, our prescribed final model, and the authors'
released model with the same decoding and answer checker. These can test
whether a gain extends beyond MATH500; they do not establish broad reasoning
ability or absence of pretraining overlap. Keep all outcomes, not just the
benchmark that improves. The remaining AIME and OlympiadBench collections are
available if the narrower comparisons justify the time.

## Evidence and resumption

- External protocol: `LARGER4096_PLAN.md` in
  `/project/alex_phd/runs/r1-zero-replication-20261007/`.
- Frozen selection and file hashes: `larger4096-data-manifest.json` there.
- Native training and optimizer outputs:
  `/home/atowell/research-runs/r1-zero-replication-20261007/larger4096-chat-drgrpo42-attempt3/`.
- The same GPU owner runs both final evaluations automatically. Its whole
  sequence is capped at 11 hours. Do not launch a competitor at trainer exit.
- The current sequence checks actual completion of 255 updates and natural-end
  checkpoint 256 before its two final tests. No separate successor is needed.
- Operational state and current decisions: `SESSION_CHECKPOINT.md` and
  `reviews-longer-v3/` in the external run store.
- The allocation ends 9 October at 20:39 UTC. The earlier presentation deadline
  is not a reason to stop this newly authorized research phase.

Heavy model files and raw data are not in GitHub. The published reference is
[the authors' pinned model card](https://huggingface.co/sail/Qwen2.5-Math-1.5B-Oat-Zero/blob/a98e477854071157a450e57dd45fd0684b6fa38a/README.md);
the scientific reference is [the paper](https://arxiv.org/abs/2503.20783).
