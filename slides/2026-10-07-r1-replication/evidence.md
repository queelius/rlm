# Evidence and result update

Cutoff: 7 October 2026, 09:27 UTC, final pilot rescoring.
The main 128-question comparison is not yet available.

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

For the final 128-question result, keep `\finalresultsfalse` in `results.tex`
until both model evaluations and the paired change counts are checked. Then
set `\finalresultstrue` and populate `\FinalBefore`, `\FinalAfter`,
`\FinalNewCorrect`, and `\FinalNewWrong`. Set `\FinalInterpretation` from the
observed evidence. Keep interpretation preliminary for this one short run.
Update `\EvidenceCutoff`, speaker guide, slide-five interpretation if needed,
and the concise notes. Rebuild PDF and notes, check all five rendered slides,
and attach paths and hashes for the exact answers and scoring receipt here.

Slide 2 is an invented learning illustration. Slide 5 shortens an actual pilot
question and summarizes the observed before/after behavior (row 29). Its arithmetic
is independently checkable; it is not a verbatim transcript. No fabricated model
output appears as experimental evidence. The paper and official code were checked read-only on
7 October 2026; the deck does not reproduce the paper's numerical results.
