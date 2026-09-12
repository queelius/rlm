# A broader RL gain repeats with a second training seed

Evidence cutoff: September 12, 2026, 16:00 UTC. Two training seeds show a
similar gain on the same test panel. A separate-panel test is queued. The
retention and controller findings below are newer than the first report.

## What we tried

We trained a small model that serves as an RLM helper. Its job was to read four
news articles and return each article's category: world news, sports, business,
or science and technology. This tests the helper's decisions, not the RLM's
ability to devise a plan or choose a tree of subtasks.

Both training methods started from the same Qwen3-4B helper, previously adapted
with supervised examples. We froze 1,024 additional training articles and 512
separate test articles, excluding known local training exposure and normalized
duplicates. Prior exposure during the original model's pretraining is unknown.

Supervised training learned from correct category maps. Reinforcement learning
(RL) sampled four maps for each four-article request, scored each map by how
many labels it got right, and used better-versus-worse comparisons to update
the model. Both methods made eight updates on the same article groups. Their
losses, sampled answers, token counts and computation were not matched.

## What happened

| Model | Correct out of 512 | Accuracy |
|---|---:|---:|
| Before this training | 422 | 82.4% |
| After supervised training | 427 | 83.4% |
| After RL | 437 | 85.4% |
| After RL, second training seed | 436 | 85.2% |

Every model returned all 512 answers. The test used the preselected eighth
checkpoint; we did not choose the best checkpoint after looking at test scores.
All 512 raw test responses were decoded again and checked against their saved
results and model identities.

The second RL run corrected 16 starting-model mistakes and introduced the same
two regressions: a net gain of 14 answers, or 2.73 percentage points. The two RL
models agree on 511 of 512 predicted labels. All 16 second-run corrections also
occur in the first run. This is a repeatable pattern on these examples, not just
two similar totals. Its descriptive request-cluster interval is +1.17 to +4.30
percentage points. We kept both preselected final checkpoints; we did not choose
the better seed.

RL corrected 17 starting-model mistakes and introduced two new mistakes.
It changed 21 labels in total; two changes replaced one wrong label with another.
The net gain was 15 answers, or 2.93 percentage points. Supervised training
corrected seven mistakes and introduced two. Against the supervised model,
RL had 12 wins and two losses.

Most of the RL gain came from science-and-technology articles: correct answers
rose from 71 to 85 out of 128. Sports gained three, business lost two, and the
world-news count was unchanged. That concentration gives us a concrete
follow-up question, but does not by itself explain the learning mechanism.

The descriptive uncertainty interval for RL's gain over the starting model
was +1.37 to +4.49 percentage points, resampling the 128 four-article requests.
This describes uncertainty within this panel; it does not account for our
larger exploratory search or variation between training runs. It is not a
confirmatory significance claim.

## What this does—and does not—tell us

The earlier one-update gain of one answer did not repeat. This broader result
is substantially more encouraging and now repeats across two training seeds.
Both seeds saw the same training data and test panel, so this does not yet show
that the gain transfers to different examples or tasks.
More varied training data and more updates changed together, so we cannot yet
say which caused the improvement. Only 43 of 256 training request groups had
different scores among their four samples; useful learning did not require
every group to provide a contrast.

All eight RL updates changed weights and passed the recorded probability,
replay and optimizer-state checks. Its full owned training workflow took about
35 minutes, including answer generation and loading. Supervised training took
about four minutes. Each complete model evaluation took another seven minutes.
These are workflow times, not isolated GPU-compute measurements. The accuracy
table is not evidence that RL is more compute-efficient or generally superior
to supervised training.

## Did the helper lose its earlier skill?

On 128 previously evaluated question-classification examples, the starting
helper got 121 right, the first RL model got 122, and the supervised model
got 121. Every answer was available. RL changed exactly one answer, from
wrong to right; supervised training changed none.

This small check found no accuracy loss. It does not establish broad retention
or transfer: the panel has historical evaluation exposure and omits one of
the dataset's six categories. We are also preparing an encyclopedia-description
test with 14 different categories, which will be a more distinct task.

## Why we are not yet training the controller with the same recipe

The helper result does not mean that the controller can already plan well.
In a separate 32-trial search of one long conversation, the controller returned
no exact requested answers. Many attempts copied a user's request instead of
the assistant's reply. Its low, varying text-overlap scores mostly measured
differences between wrong answers. We did not treat those scores as sufficient
reason to begin RL.

We are instead testing shorter conversations and preparing examples that teach
a clear retrieval procedure: identify the requested message, find the matching
user request, and return the following assistant reply. The demonstration
program solves all 32 training conversations using their public text and
questions. That validates the demonstrations, not the model's ability. The
model still needs to learn the procedure and be tested on held-out conversations.

A new record-selection interface exposed a similar problem. Only 11 of 48
episodes produced a strictly formatted final answer; only nine also agreed
with the declared finish action. The original exporter rejected the modified
prompts, so these are separately audited raw-trace diagnostics, not an accuracy
comparison. A paired syntax-example experiment now asks whether the model can
learn to use this interface before we assess the quality of its selection policy.

## What we will do next

Evaluate all four fixed models on 512 separately selected articles from the
dataset's official test partition. Their selection was frozen before the second
RL test score was known. This checks different examples within the same task,
not transfer to a new domain. All four models remain in the comparison; we
have not selected a better seed based on the earlier scores.

A training run now repeats the first 128 articles eight times. Comparing
it with eight different blocks helps separate training breadth from update count.
If repetition stops early because the sampled answers all receive the same
reward, we will report that stop rather than call it an eight-update comparison.

Another queued experiment keeps the controller unchanged and uses live helper
calls on eight new news contexts. It asks whether better local category labels
actually lead to better final counts and sums. The controller must write its
own aggregation code. A helper gain with no final-answer gain would point us
toward a different bottleneck than a failure to improve the helper itself.

The strongest potential research story is therefore not simply that RL works.
It is identifying which component improves, under what training conditions,
and whether that improvement survives changes in examples and reaches the
whole system's answers. We have promising evidence for the first part; the
remaining parts are experiments in progress, not publication-ready conclusions.

## Evidence

The [research notebook](https://github.com/queelius/rlm-research) contains
the plain-language synthesis, fixed study sources, training audits, final paired
readout and limitations. The local source-to-raw audit is
`analyses/helper-agnews-eightstep-live-audit-2026-09-12/outcomes/RAW_AUDIT-003.json`,
SHA256 `57606da1e52a5b694cad6d39dbff9febf5c66a7f6783ce3cf99e440e961f7bbb`.
The external research store is `/project/alex_phd/runs/rlm-research-r4`.
Published source and reports are not a backup of model checkpoints or raw traces.

The second-seed readout is
`analyses/helper-agnews-seed-replication-findings-2026-09-12/FINDINGS.json`,
SHA256 `970ba89011ccbeb91ed8065c49dc81615d91fa31b723a76424745fec1d7ee395`.
Its source-to-raw audit is
`analyses/helper-agnews-eightstep-seed2-live-audit-2026-09-12/outcomes/RAW_AUDIT.json`,
SHA256 `c17259f5951693f50773ab78f7295cda2b9147230d0780251ca5f57fc59b834b`.
