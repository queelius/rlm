# Evidence and result update

Cutoff: 7 October 2026, 18:20 UTC. The fourth longer-training attempt is
underway. Its final benchmarks are pending; completed short-run scores are unchanged.

## Longer training progress

Slide 5 uses the [intermediate receipt](../../docs/r1-replication-2026-10-07/interim-monitor-1820.json):
20/64 before training, 42/64 after 32 completed updates, 42/64 after 64 and
43/64 after 96. Questions, reference answers, chat inputs, full checker and
response limit are matched. All answers at every later point independently
regrade without a mismatch. Between the last two points, five answers improved
and four regressed. A net gain of one is not yet a reliable new improvement.

This is a repeatedly examined development subset, one ongoing run, and not a
full500 endpoint. It does not establish improvement beyond the stronger base
or the short trained models. We retain every scheduled point, not a best point.
Native labels 32, 64 and 96 follow that many completed updates; the forced final
label 256 instead follows 255 updates without adding another update.

## Authors' released model: an evaluation check, not our training result

The [independent reference receipt](../../docs/r1-replication-2026-10-07/author-reference-receipt.json)
records 366/500 (73.2%) for `sail/Qwen2.5-Math-1.5B-Oat-Zero`, revision
`a98e477854071157a450e57dd45fd0684b6fa38a`. All 500 ordered prompts and references
match the official evaluation data and Qwen-Math template; all saved rewards
match independent regrading with the official full checker. Generation used
one greedy answer per question and the same 3,000-token limit. The saved-answer
SHA256 is `68ee233dd7b5044030018ad03a8ffed057d8198ba4b2741352bcf731998926d5`.

The paper reports 74.2% for this released model, equivalent to 371/500. Our
reference evaluation is five answers lower; that discrepancy is unexplained.
This checks whether our evaluator reaches a similar score on the authors'
weights. It does not reproduce their training or turn our own 308/500 chat
result into 366/500. The full checker grades final answers, not each reasoning
step. No generated code was executed.

The [first larger attempt](../../docs/r1-replication-2026-10-07/longer-attempt1-failure.json)
ran out of GPU memory during the first collection's
optimization. No collection completed; inner optimizer-update count is unknown.
The [second attempt](../../docs/r1-replication-2026-10-07/longer-attempt2-failure.json)
failed during initialization with "Expandable segments are
not compatible with memory pool" in vLLM. The
[third attempt](../../docs/r1-replication-2026-10-07/longer-attempt3-failure.json)
completed 12 updates and sampled 13 collections, then ran out of memory while
computing the diagnostic entropy statistic. No checkpoint was saved. All three
failures are retained, not completed benchmark results. The earlier successful
startup checks did not establish stability across later variable-length batches.

Attempt 4 started at 16:02 UTC. It uses a private copy of
Oat with diagnostic entropy computed in 128-token pieces and unused old-policy
logits released. Three focused CPU tests passed, including a fixture with
identical actual learning-step updates. An isolated A100 test reduced extra
entropy-allocation peak from about 3.7 GiB to 0.15 GiB, with close numerical
agreement. This is not proof all future batches fit. The shared environment
and official clone remain unchanged.

The retry starts from the original base model, using the default allocator and
16-question collections. It retains the 4,096
selected MATH level 3–5 questions, `math_verify` training verifier, chat prompt,
and learning rate. Eight answers per question give 128 responses and one
optimizer update per collection; weights transfer after each update.
Filtering two overlong prompts leaves 4,094 eligible questions. The incomplete
final batch is dropped, giving 255 planned collections and updates, 4,080
questions and 32,640 responses, with prescribed final checkpoint `step_00256`.
The former 128-question collection plan (3,968 questions and 248 updates after
filtering) was abandoned after the failures, not completed.

This is a loader-derived budget, not completed training evidence. Training has
a ten-hour cap and the full sequence a thirteen-hour cap. Checkpoint saving
occurs every eight updates; progress checks occur every 32. MATH500 (500 questions),
AMC (83), and Minerva (272) will each be evaluated in both input formats using
the prescribed final checkpoint. All six final tests remain pending. Separate
base-model and authors' released-model checks on the broader benchmarks are
complete in the [broader reference receipt](../../docs/r1-replication-2026-10-07/broader-reference-receipt.json);
those scores do not test our training.
Data, verifier, and duration change
together, while the smaller collection size matches the short runs. This is
not an isolated test of training length. The
[original loader receipt](../../docs/r1-replication-2026-10-07/longer-loader-receipt.json)
records the two excluded prompts and the abandoned larger-batch accounting.
Actual completion must still be checked. The current
[accepted protocol](../../docs/r1-replication-2026-10-07/longer-training-plan.md)
also points to the external plan at
`/project/alex_phd/runs/r1-zero-replication-20261007/LARGER4096_PLAN.md`.

The learning-guide appendix explains the distinction. The five-slide deck
retains the completed short-run table, identifies the reference model on
slide 1, and shows the current monitor on slide 5.

## Evaluation repeatability update, 12:53 UTC

The [completed repeatability checks](../../docs/r1-replication-2026-10-07/evaluation-repeatability-receipt.json)
reproduced all five tested conditions byte-for-byte: base/question-only 305,
first question-only training 314, repeated question-only training 306,
base/chat-style 154, and chat-style training 308. These are repeated tests of
unchanged weights, not additional training runs. Earlier variation under
different evaluation batch composition remains a limitation. No slide score
or conclusion changed at that point, so the audience PDF then retained its
12:35 result cutoff. The later reference check is now identified separately.

## Later objective comparison, 13:48 UTC

The optional [standard GRPO comparison](../../docs/r1-replication-2026-10-07/grpo-comparison-plan.md)
finished training and scored **318/500** with questions alone, versus305base.
There were24 gains and11 losses. All500 answers were independently regraded
and checked against source questions/references; see the
[scoring receipt](../../docs/r1-replication-2026-10-07/grpo-final-scoring-receipt.json)
and [training audit](../../docs/r1-replication-2026-10-07/grpo-training-receipt.json).
Its chat-format score is **169/500**, versus 154 before training (22 gains,
7 losses). All 500 chat answers were also regraded and source-checked;
see the [chat receipt](../../docs/r1-replication-2026-10-07/grpo-chat-scoring-receipt.json).
This is a modest single-run gain,
not evidence that GRPO is better than the two Dr. GRPO runs (314 and306).
The switch changes two objective components together, and randomness is not
fully matched. The five-slide deck retained its 12:35 cutoff and core prompt-
comparison story at that point; this supporting result did not overturn that
lesson. The later reference check is now identified separately.

## Fresh training repeat: the smaller gain is not reliable yet

The [repeat result receipt](../../docs/r1-replication-2026-10-07/question-only-repeat-scoring-receipt.json)
records 306/500 with questions alone, compared with the fixed starting score
of 305. Eighteen answers became correct and seventeen became incorrect.
Root independently regraded all 500 new answers and the baseline and checked
every question and reference against the official dataset. The
[training receipt](../../docs/r1-replication-2026-10-07/question-only-repeat-training-receipt.json)
records the final checkpoint hash and all 4,096 training responses.

The first question-only training run scored 314. Both runs started from the
same original model and used the same questions, settings and update count.
Their learner seeds differed; the authors' time-based actor sampling was
retained. These are two training realizations, not repeated evaluations of one
model or an isolated test of the effect of one random seed.

The repeat's 308 additional questions scored 188 versus 187 before training.
Its overall near-zero gain is therefore not only a feature of the 64 monitored
questions. Results on those 64 change with generation batch composition: the
final monitor scored 41, whereas their answers in this full500 batch scored 37.
The corresponding base scores are 38 and 39. We retain each observation in its
own context; we do not substitute a more favorable response or checkpoint.

The result weakens a claim of stable improvement beyond the stronger starting
setup. It does not prove that RL cannot help or that a longer run would fail.
The repeated-training result is exploratory, on an already-inspected test set.
The slide table now places each model in a row and each input format in a
column, so both training outcomes remain visible.

The same repeated-training final model scored 166/500 in chat style, against
154 before training, with 25 gains and 13 losses. All500 saved answers were
independently regraded and matched to the exact official chat template and
reference answers. Its capped-response count is198 versus205 before training.
The receipt retains both input formats, not whichever gives the larger gain.

| Model tested | Chat-style input | Question alone |
| --- | ---: | ---: |
| Starting model |154|305|
| Trained with chat style |308|317|
| Trained with questions alone |168|314|
| Repeated question-only training |166|306|

Every entry counts correct answers out of the same500 questions. Each trained
row is a separate fresh-base run, evaluated at its prescribed final checkpoint.

## New follow-up: train with the question alone

The [question-only final receipt](../../docs/r1-replication-2026-10-07/question-only-final-scoring-receipt.json)
records the second training run: 314/500 with question-only input, versus 305
before training, and 168/500 with chat-style input, versus 154. Root regraded
all 1,000 new answers and verified prompts/references against the source.
The raw-input comparison has 24 gains and 15 losses; the chat comparison has 21
gains and seven losses. These are small one-run changes, not reliable effects yet.

The trained rows on slide 4 are different models, each trained afresh from
the original base on the same 512 questions for 32 updates. Each model is tested
in both formats using the same final checkpoint. No checkpoint was selected
for its score. The new model's capped responses are 19 with questions alone and
201 in chat style, compared with 15 and 205 in the starting model. Thus this
training did not produce the large chat-response recovery of chat-style training.
We have not isolated every cause of the contrast between the training runs.

Update at 11:42 UTC: both first-run fixed-weight question-only evaluations repeated
byte-for-byte, retaining scores of 305 and 314. All repeated answers were
regraded and source-checked. The later fresh training repeat is reported above.
Neither check may replace the first scores with better ones.

## Main slide result: same weights tested with two prompt formats

The [prompt-control receipt](../../docs/r1-replication-2026-10-07/prompt-control-scoring-receipt.json)
records chat-style scores of 154/500 before and 308/500 after training;
question-only scores are 305/500 before and 317/500 after the same training.
Both rows use identical before/after questions, answer checking, and length
limits. The trained checkpoint is the same in both rows. Raw-question outputs
were independently rescored and their prompts and references checked against
the original dataset. That comparison has 18 gains and six losses.

All 500 questions were excluded from weight training. The 64 monitoring
questions were repeatedly inspected, so they are not an untouched test. Under
question-only prompting, their scores changed 39 to 37; the original 128 in
this fresh batch changed 79 to 84; the additional 308 changed 187 to 196.
The [chat-style receipt](../../docs/r1-replication-2026-10-07/full500-scoring-receipt.json)
separates the corresponding roles for that format.

This shows that prompt choice changes the measured size of the RL gain. It
does not establish either no learning or newly acquired reasoning skills.
The question-only follow-up is exploratory and uses the authors' existing
no-template option, not a change to their learning algorithm. Its first attempt
at 10:35 was rejected for stale cached chat inputs; the valid run started 10:40
and finished 11:24. Actual training inputs were audited, not just configuration.

## Main separate test

The [fixed-test receipt](../../docs/r1-replication-2026-10-07/fixed128-scoring-receipt.json)
records **41/128 before training and 80/128 afterward**, with 44 newly correct
and five newly incorrect answers. Every saved reward was independently rescored
with the authors' full checker. All prompts and references match the fixed
test questions; both models used the same generation settings and prompt.

We used the prescribed final checkpoint after 32 updates, saved as step_00033.
No 33rd optimizer update occurred; the directory name includes a bookkeeping
increment. The [training receipt](../../docs/r1-replication-2026-10-07/subset512-training-receipt.json)
records its model hash and the audit of all 4,096 training responses.

Of the 44 gains, 36 had a baseline response reaching the 3,000-token cap, and
37 had no parseable baseline answer. Those groups overlap. Total capped
responses fell from 56 to 12. These are descriptive measurements, not a causal
identification of what training learned. Correctness checks final answers,
not every explanation step. No generated Python code was executed.

The separate 128 test was generated before the full500 comparison. In the
fresh full500 batch, those same questions score 41 to 82 under chat prompts.
Retain 41 to 80 as the original result; do not select the higher repeat.
A larger test does not create more training replicates.

## Earlier pilot, retained for context

The [source protocol](../../docs/r1-replication-2026-10-07/README.md) and
[scoring receipt](../../docs/r1-replication-2026-10-07/pilot-scoring-receipt.json)
record a fresh rescore of the native saved answers: official final checker 20/64 before, 22/64
on the final evaluation after four updates; three newly correct and one newly
incorrect. The earlier post-update evaluation was 23/64, with the same weights.
Eight of 64 generated texts varied between these two evaluations. Native fast
checker: 19/64 before, 22/64 at the first check and 21/64 at the final repeat.
The receipt records source hashes, exact grader, paired-question check, and changed rows.

Saved pilot answers are under:

`/project/alex_phd/runs/r1-zero-replication-20261007/pilot-attempt2/debug_1007T09:09:11/eval_results/`

The relevant checkpoint is the model after four updates. Do not infer update
count from a directory number alone; the owner should bind the exact receipt.
The fifth-numbered saved directory is not automatically a fifth update.

Official source: `dfca49dd460ee7cc8e4a5a162c876a7fd6993b87`.
Base model revision: `4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2`.
External experiment store:
`/project/alex_phd/runs/r1-zero-replication-20261007`.

When another result becomes available, update the visible result,
cutoff, concise notes, speaker guide, and learning guide together. Keep the
original fixed128 result; do not replace it with a favorably selected subset.

Slide 2 is an invented learning illustration. Slide 5 uses the actual separate
test question “Write 3/20 as a decimal” (row 5), with responses summarized.
Its arithmetic
is independently checkable; it is not a verbatim transcript. No fabricated model
output appears as experimental evidence. The paper and official code were checked read-only on
7 October 2026; the deck does not reproduce the paper's numerical results.
