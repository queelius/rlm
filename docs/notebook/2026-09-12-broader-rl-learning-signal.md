# RL gains repeat, but weaken on fresh examples

Later evidence: the [18:23 follow-up](2026-09-12-controller-learning-and-transfer.md)
reports the completed repetition control on fresh examples and new controller
tests. The dated report below retains its earlier evidence cutoff.

Evidence cutoff: September 12, 2026, 17:42 UTC. Two training seeds show a
similar gain on the earlier panel. On 512 fresh examples, the gains are smaller
and do not clearly beat supervised training. This changes our interpretation:
we have evidence of modest helper improvement, not a general RL advantage.
The first test on a different task shows almost no change. Better helper labels
also failed to improve the complete RLM's final answers in a small paired test.

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
The initial comparison changed both data variety and update count. A new control
now separates them more directly: eight updates repeating the same 128 articles
scored 417/512, compared with 437/512 after eight different article groups and
422/512 before either run. Against the varied-data model, repetition lost 23
answers and won three; every answer was available and raw-response audited.
This supports using varied examples in this setting, rather than simply doing
more updates. It remains a single repetition run on the earlier test panel;
its first update was not bitwise identical to the varied-data run despite
matching sampled inputs, and the choice of the repeated block may matter.
A fresh-panel comparison is being prepared before making a stronger claim.
Only 43 of 256 training request groups in the varied-data run had
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
the dataset's six categories.

On a different task, classifying 224 encyclopedia descriptions into 14 categories,
the starting helper got 209 correct, supervised training got 208, and the two RL
models got 209 and 210. All answers were available and independently checked
against the saved raw responses. Each trained model changed only one starting
prediction. The first RL model changed one wrong label to another wrong label;
the second corrected one error. This small test does not establish useful
cross-task improvement, although it found little change in existing performance.

## Do better helpers improve the complete RLM?

Not yet in our small test. We gave the unchanged controller eight fresh sets of
16 news articles and asked two questions per set, such as counting articles in
a category or adding their numerical weights. The controller had to call the
helper and write its own Python code to combine the returned labels.

The helper's correct labels rose from 103 to 108 out of 128 distinct articles,
but the complete RLM answered only 3 of 16 questions correctly with either
helper. Without helper calls, the tested controller answered none correctly.
All 48 scheduled outcomes were available. These are eight context groups,
not 48 independent problems or evidence about unfamiliar question types.

The saved programs explain some of the gap. Four count attempts assigned a
Python variable but did not print its value. The controller therefore received
an empty observation, then answered anyway. Some programs also counted the
wrong subset of articles. In contrast, all 20 supported programs that returned
a scalar observation had final answers matching their computed value—even
when that value was wrong. The controller does not generally ignore its helper.

There is another caution: the only newly exact helper-derived aggregate came
from two label mistakes cancelling each other, not from a faithful set of labels.
Six other unique label mistakes were genuinely corrected, but this did not
produce another exact final answer. We should measure correct intermediate
work as well as final scores.

The controller's authenticated training examples all printed their intermediate
results. Silent assignment was not a demonstrated target. We will therefore
test whether the model retains the execution procedure on fresh inputs and
whether a clearer result-return mechanism helps. Simply adding more examples
of the same routine may not fix incorrect interpretation of the question.

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
A fixed four-update supervised run is queued. A separate proposed RL update
gives partial reward only for a near-exact answer, and full reward for an exact
answer. Its first attempt stopped before changing weights: the probabilities
computed by the training and answer-generation engines differed enough that
whole-trajectory importance weights became highly uneven. This is a failed
training qualification, not evidence that RL learned nothing. We are checking
whether small numerical differences accumulate across long action sequences,
and whether a standard lower-variance correction is appropriate. The reward may
still favor broad conversation printing, so procedure and observation size
remain necessary diagnostics.

A new record-selection interface exposed a similar problem. Only 11 of 48
episodes produced a strictly formatted final answer; only nine also agreed
with the declared finish action. The original exporter rejected the modified
prompts, so these are separately audited raw-trace diagnostics, not an accuracy
comparison. A paired syntax-example experiment also encountered the exporter
mistake. Its corrected, call-free audit now authenticates all 48 initial prompts
against the condition-specific prompts frozen before the run; it does not change
model outputs. Correct endpoint answers fell from 5 to 2 out of 24, with two and
three provider-error outcomes respectively kept unknown. Under the interface's
stricter finish contract, usable answers fell from five to zero. Rejected
interface actions fell from 35 to eight, but this did not improve answers.
We are retiring this syntax-example variant rather than spending more GPU time
on the same intervention.

## What we will do next

The news and encyclopedia comparisons above are complete, with all four fixed
models reported. A released-base model reference will help
distinguish new capability from undoing earlier specialization.
Its first launch failed before making any model calls because a helper-specific
launcher was incorrectly reused for the unadapted model. A narrow, additive
repair will retain the same frozen questions and preserve the failed attempt.

The repeated-data run improved its sampled training score but reduced its test
score, as reported above. We will test its fixed checkpoint on the same fresh
panel already used for the other four models; no best-checkpoint selection is
involved. This is a stronger follow-up than repeating the same training recipe
without checking where its improvement applies.

The whole-RLM comparison above now points toward controller interpretation and
execution as additional bottlenecks. A queued conversation-retrieval comparison
tests supervised procedural training against a small RL update once its
training-engine mismatch is addressed. A separate training-only measurement
is conditional on an actual update and will check whether it changes
the probability of the sampled actions. This distinguishes a negligible update
from a substantial update that does not help on new examples.

We also queued a small multi-hop question-answering comparison: does allowing
helpers to call their own helpers improve answers under the same total-call
limit? We will record whether deeper calls actually occur. Merely allowing
recursion is not evidence that the model learned when to use it.

The strongest potential research story is therefore not simply that RL works.
It is identifying which component improves, under what training conditions,
and whether that improvement survives changes in examples and reaches the
whole system's answers. We have promising evidence for the first part, but
fresh-example evidence is weaker than the first panel suggested, and the first
whole-system and cross-task comparisons show little or no endpoint improvement.
These limitations are guiding the next experiments, not being hidden behind
the most favorable initial score. The broader story is not publication-ready.

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

The encyclopedia audit is
`analyses/helper-dbpedia224-transfer-independent-2026-09-12/RESULT.json`.
The whole-RLM mechanism and training-corpus audit is
`analyses/root-qs6-ag-live-helper-transfer-mechanism-2026-09-12/REPORT.json`;
it uses static recognition and trusted host reductions, never executing saved
generated programs. The corrected syntax readout is in
`analyses/root-qs6-budgeted-evidence-syntax-corrected-readout-2026-09-12/`.
The repetition control is
`analyses/helper-agnews-repeat128-live-audit-2026-09-12/outcomes/RAW_AUDIT.json`,
SHA256 `510c93109e2db100c844086d73d81395cb620326688516f9435bb237452ac9a4`.
