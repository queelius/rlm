# Moving from a pipeline check toward reproducing the training result

Launch update: attempt 4 started 7 October at 16:02 UTC under owner exec30612,
after the [isolated GPU memory check](memory-repair-receipt.json) passed.
The protocol below was fixed before that launch. The four broader reference
controls are complete; [all outcomes](broader-reference-receipt.json) are retained.

Prelaunch protocol fixed on 7 October 2026 at 15:55 UTC; the launch update
above records what happened afterward. This is an exploratory reproduction,
not a new algorithm. Final training results remain pending.

Two earlier attempts failed and are retained. [Attempt 1](longer-attempt1-failure.json)
ran out of GPU memory during its first collection's optimization. [Attempt 2](longer-attempt2-failure.json)
tested an expandable allocator, but that setting is incompatible with the
generation engine's memory pool. Neither produced a completed training result.

Attempt 3 used the earlier successful collection size of 16 questions and the
default allocator. It completed 12 updates, then ran out of memory while
computing an entropy statistic during collection 13. No checkpoint had been
saved. The [failure receipt](longer-attempt3-failure.json) preserves this result;
it is not a benchmark score or a recoverable trained endpoint.

Attempt 4 keeps that training recipe but uses a private copy of the Oat package
with two memory changes: compute the diagnostic entropy in 128-token pieces,
and release old-policy prediction tensors after extracting their probabilities.
The training objective and its gradients are not intentionally changed. Three
focused CPU tests check matching entropy values/gradients, identical actual
learning-step updates in a fixture, and subprocess loading of the private copy.
A full-shaped GPU memory check preceded training and passed, as recorded above.
These checks are not proof that all later training batches fit. The shared environment and official clone
remain unchanged. More frequent saving limits the cost of another interruption.

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
| Learning method | Authors' Dr. GRPO objective, full-model BF16; private memory adapter described above |
| Input and reward | Qwen-Math conversation template; `math_verify` answer checker |
| Sampling | Eight attempts per question, temperature 1, up to 3,000 response tokens |
| Training budget | 255 collections, 255 optimizer updates, 32,640 sampled attempts |
| Learning rate | Constant 0.000001 |
| Hardware and limit | One A100 40 GB; 10-hour training cap |
| Saved state | Every 8 updates; keep latest two model and optimizer checkpoints; monitor every 32 |
| Final evaluation | MATH500 (500), AMC (83), Minerva (272), each in both chat and question-only formats |

Each collection contains 16 questions and 128 responses. The learner takes
one optimizer update, then sends its new weights to the generator, as in the
successful short runs. The native final folder is named `step_00256`; it means
final bookkeeping after 255 updates, **not** a 256th weight update.

The official training source is commit
`dfca49dd460ee7cc8e4a5a162c876a7fd6993b87`. Base weights use revision
`4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2`.
The learner seed is 42; the original actor uses time-based randomness, so this
is not a promise of an exactly repeatable training trajectory.

Compared with the short runs, we changed data coverage and difficulty,
training duration, and the reward checker to move closer to the reference recipe.
Consequently, this run cannot tell us which change caused any improvement.
That is a later comparison if the stronger recipe works.

This is still a reduced schedule. The pinned repository's example script uses
eight GPUs and permits 20 passes over its training collection; we use one GPU
and one pass over a subset. That example also uses the broader `math_12k` data,
whereas the released model card specifies level 3–5 data. The card does not
specify its exact number of completed updates. Therefore 255 updates is our
fixed exploratory budget, not an asserted match to the released model's full
training history. A remaining score gap would not, by itself, falsify the paper.

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
- Native training and optimizer outputs for the prepared attempt:
  `/home/atowell/research-runs/r1-zero-replication-20261007/larger4096-chat-drgrpo42-attempt4/`.
- Adapter and 63-file manifest: `runtime-memory-v1/` in the external run store;
  only `oat/algorithms/ppo.py` differs from the installed package.
- The same GPU owner runs both MATH500 and both AMC/Minerva final evaluations
  automatically. Its whole sequence is capped at 13 hours, including a 10-hour
  training cap. Do not launch a competitor at trainer exit.
- The current sequence checks actual completion of 255 updates and natural-end
  checkpoint 256 before four evaluation jobs covering six dataset/input-format
  conditions. No separate successor is needed.
- Operational state and current decisions: `SESSION_CHECKPOINT.md` and
  the current observer directory named in that checkpoint. Old stopped observer
  directories are historical owner replacements, not research pauses.
- The allocation ends 9 October at 20:39 UTC. The earlier presentation deadline
  is not a reason to stop this newly authorized research phase.

Heavy model files and raw data are not in GitHub. The published reference is
[the authors' pinned model card](https://huggingface.co/sail/Qwen2.5-Math-1.5B-Oat-Zero/blob/a98e477854071157a450e57dd45fd0684b6fa38a/README.md);
the scientific reference is [the paper](https://arxiv.org/abs/2503.20783).
