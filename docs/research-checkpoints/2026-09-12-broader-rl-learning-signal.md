# RL gains repeat, but weaken on fresh examples

Evidence cutoff: September 12, 2026, 17:03 UTC. Two training seeds show a
similar gain on the earlier panel. On 512 fresh examples, the gains are smaller
and do not clearly beat supervised training. This changes our interpretation:
we have evidence of modest helper improvement, not a general RL advantage.

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

Each column below contains 512 different examples. The earlier panel had been
examined during this research; the fresh panel was separately frozen from the
dataset's official test partition. These are not training-set scores.

| Model | Earlier panel: correct / 512 | Fresh panel: correct / 512 |
|---|---:|---:|
| Before this training | 422 | 422 |
| After supervised training | 427 | 426 |
| After RL | 437 | 427 |
| After RL, second training seed | 436 | 429 |

Every model returned all 512 answers. The test used the preselected eighth
checkpoint; we did not choose the best checkpoint after looking at test scores.
All four models' raw four-article responses on each panel were decoded again
and checked against their saved results and model identities.

On the fresh panel, RL corrected 11 starting-model mistakes but introduced six
or four new mistakes, depending on the training seed. The net gains were five
and seven answers. Supervised training corrected four mistakes and introduced
none. RL therefore exceeded supervised training by only one and three answers;
both descriptive uncertainty intervals for that difference include zero.

Both RL models gained ten correct science-and-technology labels out of 128,
but lost two business labels and two to four world-news labels. This is
consistent with a shift in category preferences, rather than a broad new
capability. We have not established what caused the smaller gain on this panel.
The two RL models still agree on 508 of 512 labels. Repeating across training
seeds and transferring to fresh examples are different requirements.

### Details of the earlier panel

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
Both seeds saw the same training data. The fresh-example test now shows a
smaller positive difference from the starting helper, without a clear advantage
over supervised training. Neither news panel establishes transfer to a new task.
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

## The controller needs a different investigation

The helper result does not mean that the controller can already plan well.
In a separate 32-trial search of one long conversation, the controller returned
no exact requested answers. Many attempts copied a user's request instead of
the assistant's reply. Its low, varying text-overlap scores mostly measured
differences between wrong answers. We did not treat those scores as sufficient
reason to begin RL.

On eight shorter conversations, 32 trials produced no exact answers; 30 trials
were available and two exceeded the model's context limit. Four answers were
nearly correct, with literal backslash-n characters instead of actual newlines.
However, their Python programs had not correctly extracted the answer. They
printed much of the conversation, after which the root model picked out the
requested passage. We must not call this successful programmatic retrieval or
learned decomposition. We preserved the original scores without repairing text.

We are testing examples that teach
a clear retrieval procedure: identify the requested message, find the matching
user request, and return the following assistant reply. The demonstration
program solves all 32 training conversations using their public text and
questions. That validates the demonstrations, not the model's ability. The
model still needs to learn the procedure and be tested on held-out conversations.
A fixed four-update supervised run and a separate one-update RL comparison are
queued. The latter gives partial reward only for a near-exact answer, and full
reward for an exact answer. It may still reward broad conversation printing, so
we will inspect the procedure and observation size as well as answer scores.

A new record-selection interface exposed a similar problem. Only 11 of 48
episodes produced a strictly formatted final answer; only nine also agreed
with the declared finish action. The original exporter rejected the modified
prompts, so these are separately audited raw-trace diagnostics, not an accuracy
comparison. A paired syntax-example experiment has completed, but the same
exporter mistake rejected its modified prompts. All 48 actual initial prompts
match the condition-specific prompts frozen before the run. A separate audit is
recovering what can be learned without rerunning the model. Raw traces show fewer
rejected interface actions after the syntax example (35 to eight), but not better
agreement between the model's declared finish and final answer. This is not yet
a positive architecture result.

## What we will do next

The new official-test comparison above is complete, with all four fixed models
reported. A queued encyclopedia-description test asks whether changes survive a
different task with 14 categories. A released-base model reference will help
distinguish new capability from undoing earlier specialization.

A training run completed eight updates by repeating the first 128 articles.
Its fixed final evaluation is queued. Comparing it with eight different blocks
helps separate training breadth from update count. All eight updates had usable
reward contrasts and saved optimizer checkpoints. The training score increased,
but that is not yet a held-out improvement.

Another queued experiment keeps the controller unchanged and uses live helper
calls on eight new news contexts. It asks whether better local category labels
actually lead to better final counts and sums. The controller must write its
own aggregation code. A helper gain with no final-answer gain would point us
toward a different bottleneck than a failure to improve the helper itself.

We also queued a small multi-hop question-answering comparison: does allowing
helpers to call their own helpers improve answers under the same total-call
limit? We will record whether deeper calls actually occur. Merely allowing
recursion is not evidence that the model learned when to use it.

The strongest potential research story is therefore not simply that RL works.
It is identifying which component improves, under what training conditions,
and whether that improvement survives changes in examples and reaches the
whole system's answers. We have promising evidence for the first part, but
fresh-example evidence is weaker than the first panel suggested. The remaining
parts are experiments in progress, not publication-ready conclusions.

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

The fresh official-test synthesis is
`analyses/helper-agnews-official-test-transfer-findings-2026-09-12/FINDINGS.json`;
its independent four-arm source-to-raw audit is in
`analyses/helper-agnews-official-test-fresh512-independent-2026-09-12/RESULT.json`.
The controller procedure correction is
`analyses/openai-mrcr-short32-outcomes-2026-09-12/PROGRAM_VS_TERMINAL_ADDENDUM.json`,
SHA256 `858fb090ea54796da506b1a155461dd7291e0da8022b115238868bfddffbcebb`.
