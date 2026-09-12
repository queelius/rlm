---
date: 2026-09-12
cutoff_utc: "2026-09-12T07:16:00Z"
question: rq:rl-effective-feedback
status: one_helper_update_and_paired_evaluations_complete
evidence_level: exploratory_single_update
---

# Helper RL now runs, but its first accuracy change is negligible

We completed a reinforcement-learning update to the small model that answers
local helper questions. Its first evaluation does **not** show a useful accuracy
improvement. That distinction matters: we have a working experiment to build on,
not yet a successful learning method.

| Evaluation | Before the update | After the update |
|---|---:|---:|
| New question-category examples | 119 correct out of 128 | 120 out of 128 |
| New news-category examples | 112 correct out of 128 | 112 out of 128 |
| Familiar questions, each answered twice | 245 correct out of 256 answers | 244 out of 256 answers |

There were no missing answers or formatting failures in these comparisons.
The new-question change was exactly one wrong answer becoming correct; no new
question became wrong. News predictions were unchanged. On familiar material,
one answer improved and two became wrong. Both losses concerned the same question
on its two repeats, so they are not two independent examples.

## The earlier supervised training has a stronger transfer result

A newly completed control tests the original model without our research adapter
on the same new examples. This separates the small extra RL change from the
much larger difference associated with the earlier supervised-trained helper.

| New examples | Original model | Supervised-trained helper | Helper after one RL update |
|---|---:|---:|---:|
| 128 question-category examples | 92 correct | 119 correct | 120 correct |
| 128 news-category examples | 113 correct | 112 correct | 112 correct |

This is encouraging evidence that the supervised-trained helper can handle new
questions of the same kind. It is not evidence of a general improvement across
tasks: the news-category score did not improve. Against the original model,
the supervised-trained helper corrected 28 question-category answers and spoiled
one; it corrected three news answers and spoiled four. All three models received the
same four-record requests and category instructions within each dataset. The
original model's serving configuration had no adapter and no prompt caching;
the adapted models used adapters and prompt caching. This is a comparison of
the deployed model conditions, not an isolated test of every numerical execution
detail. Eager execution and the maximum number of served sequences also differed.
We do not claim a speed gain from their different runtimes.

The original-model control first failed before sending any model requests because
an outdated driver-library path was selected. Its additive repair completed all
64 requests with valid replies. The failed attempt remains recorded separately.

## What the model learned from

We started with the existing supervised-trained helper, based on
Qwen3-4B-Instruct-2507. For each of 32 training questions, it sampled four answers.
The task was to return the question's category in the required JSON format.
An answer received one point for the correct category and zero otherwise.

The update compared each answer's reward with the average reward of the other
three answers to that question. This rewards relatively successful attempts and
discourages relatively unsuccessful ones. It does not provide a new teacher-written
answer or train the root's decomposition plan. The root model was unchanged.

Of the 128 sampled answers, 116 were correct. Only four of the 32 questions
produced a mixture of correct and incorrect answers. The other questions offered
no within-question learning contrast in that batch. We still kept the original
128-answer loss denominator; we did not amplify the update by silently discarding
the no-contrast groups.

This was one small update, using a learning rate of 0.00001. It is too little
evidence to conclude either that helper RL works well or that it cannot work.

## Why this update is technically different from the paused attempt

An earlier approach collected answers through the fast serving system and then
evaluated their probabilities in the training system. We found unresolved
differences in those probabilities. That older attempt remains on hold.

Here, sampling and training used the same model implementation, grammar support,
precision and batch shape. Before changing weights, we replayed every sampled
answer and checked that its probabilities matched those recorded at sampling.
All 32 groups passed the original checks; the recorded selected-token and
whole-answer differences were exactly zero. An independent saved-state audit
also reconstructed the single optimizer update from its stored moments and
confirmed that the adapter weights changed.

The first attempt ran out of GPU memory before updating. We preserved its sampled
answers and reduced training memory use by recomputing intermediate activations.
The repaired run verified and reused those same pre-update answers. It did not
change the numerical acceptance criteria or quietly sample a different batch.

The failed attempt took 407 seconds; the successful repair took 362 seconds.
Together these were about 12.8 minutes of owner time, excluding the subsequent
evaluation runs and CPU preparation. The successful repair peaked at about
9.83 GiB of allocated GPU memory. Its saved random-generator state belongs to
the repair, not to the interrupted original sampling stream.

## What “new” means here

The new panel contains 128 TREC test questions and 128 AG News test examples.
We froze their selection before scoring and checked that neither their source
identifiers nor normalized text occurred in the helper's actual supervised-training
inputs or the 32 new RL training questions. This is **not** a claim that the base
model had never encountered them during its original pretraining.

The separate familiar panel contains 128 questions that the helper previously
saw twice in supervised training. They were excluded only from this new 32-question
RL update. That panel checks behavior on familiar material, not unseen-example
generalization. It uses individual-record requests and two samples per question;
the new panel uses four-record requests and one sample. Compare before and after
within each panel, not the panels' absolute scores as a controlled contrast.

## What this changes about our next experiment

We are preparing a four-update run with newly sampled answers at every step,
one optimizer carried across the steps, and a checkpoint after each successful
update. The fourth checkpoint is the declared primary result; we will not select
whichever checkpoint looks best on the evaluation panel.

If this also produces little signal, promising follow-ups include a broader
training set and a separately declared comparison of how training questions are
chosen. We should check whether the model encounters enough informative successes
and failures, rather than simply running more identical easy examples. Evaluation
questions must not enter that selection. Further reuse of this scored test panel
will be labeled exploratory, with fresh data needed for confirmation.

In parallel, the [helper-size experiment](2026-09-12-helper-size-downstream.md)
remains the stronger harness lead: actual replies generated in smaller groups
improved final answers on two familiar contexts. A natural next question is
whether we can learn **when smaller helper requests are worth their extra cost**.
That restricted decision is a concrete first step toward learned decomposition,
not yet a claim of general recursive planning.

## Evidence and reproducibility

The external research store is `/project/alex_phd/runs/rlm-research-r4`.

- Training: `sidecars/helper-hf-onpolicy-v2/outputs/attempt-001`.
  Checkpoint state SHA-256:
  `24d607e7651c60c1b9952d092635fb5d95f0d00dc1d0eea7f2137bac6b9ccd9e`.
- Independent update audit:
  `analyses/helper-hf-v2-update-stage-local-review-2026-09-12/REPORT.json`, SHA-256
  `72f6c81238ce998c72c9ad0e3be39b5f78bf32da8f605ea19d77c4ceb31080ae`.
  MAIN independently reproduced its result, excluding its creation timestamp.
- Independent evaluation audit:
  `analyses/helper-hf-update-generalization-2026-09-12/REPORT.json`, SHA-256
  `e2b63eff8c8c1d05fc4cf5e0734344c1a77ae5a10f37c43708557388e7230bcc`.
  It checks the frozen requests, ordered schemas, record identities and decoded
  answers, including all 512 familiar-panel raw request/response/call records.
- The new panel's acquisition and overlap receipts are in
  `sidecars/helper-unseen-generalization-panel-v1`. Source license limitations
  remain recorded there; raw dataset text and model weights stay outside Git.
- Original-model control:
  `sidecars/helper-unseen-generalization-base4b-v1/outputs/attempt-002/RESULT.json`,
  SHA-256 `b10032ee7e0aa23e625fa2027a76ee0f24c82a0cbcaae8c973fb9ecaf4196d1f`.
  The completed run used `READY_REPAIR.json`; attempt-001 is the startup failure.
- Paired original-versus-supervised-trained analysis:
  `analyses/helper-unseen-baselines-base-repair-2026-09-12/REPORT.json`, SHA-256
  `a2a965fa0c8a507d3ceb8bffdcec5a3dbb302b87399777aada7c448469f2cb87`.

The delivered September 11 slides retain their meeting cutoff. This is a new
post-meeting report. A GitHub push preserves this report, not the external run
store or model checkpoints.
