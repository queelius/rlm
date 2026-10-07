# Evidence and result update

Cutoff: 7 October 2026, 10:36 UTC, completed four-way prompt comparison.

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
The question-only training follow-up started at 10:35 UTC and is pending.
It is exploratory and uses the authors' existing no-template option, not a
change to their learning algorithm.

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
