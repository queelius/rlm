# A broader training run produced a promising RL gain

Evidence cutoff: September 12, 2026, 14:07 UTC. This is an exploratory result
from one training run. A different-seed replication is being prepared.

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

Every model returned all 512 answers. The test used the preselected eighth
checkpoint; we did not choose the best checkpoint after looking at test scores.
All 384 raw test responses were decoded again and checked against their saved
results and model identities.

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
is substantially more encouraging, but that history makes replication important.
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

## What we will do next

First repeat the same RL recipe with different training and sampling seeds.
Keep the data, update count and fixed final checkpoint unchanged. Reusing the
512-article panel tests training repeatability, not transfer to another dataset.
If the gain holds, test another helper input size and another task, including
whether improving news classification damages the helper's earlier skills.

In parallel, a separate experiment lets the RLM choose which records to ask
about and explicitly decide when to finish. Another conversation-search study
targets the controller's own Python-based procedure. Those are ongoing
experiments, not demonstrated benefits. A new 32-train/16-test conversation
split is ready for a later test on different underlying conversations.

## Evidence

The [research notebook](https://github.com/queelius/rlm-research) contains
the plain-language synthesis, fixed study sources, training audits, final paired
readout and limitations. The local source-to-raw audit is
`analyses/helper-agnews-eightstep-live-audit-2026-09-12/outcomes/RAW_AUDIT-003.json`,
SHA256 `57606da1e52a5b694cad6d39dbff9febf5c66a7f6783ce3cf99e440e961f7bbb`.
The external research store is `/project/alex_phd/runs/rlm-research-r4`.
Published source and reports are not a backup of model checkpoints or raw traces.
