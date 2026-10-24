# E58 pre-registration: a powered non-inferiority test of three-label bootstrapping (written 2026-10-24 04:35 UTC, before either run)

Motivation: E39 (two bootstrap seeds on the same 30 families) gave 26 against 29 correct libraries, p 0.61; its
family-bootstrap 95% interval for the difference in success rate is about -0.20 to +0.13, so it does not establish
non-inferiority. An independent review (Codex, codex-reviews/2026-10-24-claims-review.md in the run store) ranked a
powered test on untouched families first.

Families: the held-out families of data seeds 2 to 5 from family_gen.family_ids("heldout", (2, 3, 4, 5)) that no
earlier run or training set used: 141 families (32 need the helper). Excluded: the 7 gen-chat-*-s2 families, which
were in the 8B run whose libraries trained the 4B. List: e58-prereg/families.txt in the run store, and below.

Arm A (e58a-4br5b-s16-best8): proposer-rl-16-40, pre-declared because it is the round-five recipe's median checkpoint
on held-out discovery (36 of 74), not its best under E39 (that was seed 17); 8 round-0 candidates, no repair, no early
stopping; accept the first valid candidate (runs on all 30 teaching inputs, varied outputs) that reproduces the 3
shown examples. Arm B (e58b-8b-standard): qwen3-8b-awq thinking, standard protocol (4 candidates, 1 labelled repair
round, accept at teaching mean 0.9 or more over 30 labels). Both boot seed 0.

Outcome per family and arm: success = the accepted library scores 0.9 or more on 30 fresh tasks (split "fresh", 400
records). Primary analysis: paired difference in success rate, A minus B, with a 95% percentile interval from 10,000
family-bootstrap draws (e39_analysis.py --margin 0.10). Non-inferiority is claimed only if the interval's lower bound
is above -0.10. Expected power about 70 percent if the arms are truly equal (assuming about 25 percent discordant
families). Secondary: false acceptances, proposer output and input tokens per family, exact McNemar. All results are
reported whatever they show.

Families: gen-prose-count-plain-s2,gen-prose-count-answer_line-s3,gen-prose-count-prefix-s4,gen-prose-count-json-s5,gen-prose-total-plain-s2,gen-prose-total-answer_line-s3,gen-prose-total-prefix-s4,gen-prose-total-json-s5,gen-prose-top_user-json-s2,gen-prose-top_user-plain-s3,gen-prose-top_user-answer_line-s4,gen-prose-top_user-prefix-s5,gen-prose-nth_item-json-s2,gen-prose-nth_item-plain-s3,gen-prose-nth_item-answer_line-s4,gen-prose-nth_item-prefix-s5,gen-prose-last_city-plain-s2,gen-prose-last_city-answer_line-s3,gen-prose-last_city-prefix-s4,gen-prose-last_city-json-s5,gen-prose-first_after-prefix-s2,gen-prose-first_after-json-s3,gen-prose-first_after-plain-s4,gen-prose-first_after-answer_line-s5,gen-prose-distinct-json-s2,gen-prose-distinct-plain-s3,gen-prose-distinct-answer_line-s4,gen-prose-distinct-prefix-s5,gen-prose-day_span-json-s2,gen-prose-day_span-plain-s3,gen-prose-day_span-answer_line-s4,gen-prose-day_span-prefix-s5,gen-prose-semantic_total-answer_line-s2,gen-prose-semantic_total-prefix-s3,gen-prose-semantic_total-json-s4,gen-prose-semantic_total-plain-s5,gen-xmlish-count-answer_line-s2,gen-xmlish-count-prefix-s3,gen-xmlish-count-json-s4,gen-xmlish-count-plain-s5,gen-xmlish-total-answer_line-s2,gen-xmlish-total-prefix-s3,gen-xmlish-total-json-s4,gen-xmlish-total-plain-s5,gen-xmlish-top_user-plain-s2,gen-xmlish-top_user-answer_line-s3,gen-xmlish-top_user-prefix-s4,gen-xmlish-top_user-json-s5,gen-xmlish-nth_item-plain-s2,gen-xmlish-nth_item-answer_line-s3,gen-xmlish-nth_item-prefix-s4,gen-xmlish-nth_item-json-s5,gen-xmlish-last_city-answer_line-s2,gen-xmlish-last_city-prefix-s3,gen-xmlish-last_city-json-s4,gen-xmlish-last_city-plain-s5,gen-xmlish-first_after-json-s2,gen-xmlish-first_after-plain-s3,gen-xmlish-first_after-answer_line-s4,gen-xmlish-first_after-prefix-s5,gen-xmlish-distinct-plain-s2,gen-xmlish-distinct-answer_line-s3,gen-xmlish-distinct-prefix-s4,gen-xmlish-distinct-json-s5,gen-xmlish-day_span-plain-s2,gen-xmlish-day_span-answer_line-s3,gen-xmlish-day_span-prefix-s4,gen-xmlish-day_span-json-s5,gen-xmlish-semantic_total-prefix-s2,gen-xmlish-semantic_total-json-s3,gen-xmlish-semantic_total-plain-s4,gen-xmlish-semantic_total-answer_line-s5,gen-chat-count-plain-s3,gen-chat-count-answer_line-s4,gen-chat-count-prefix-s5,gen-chat-total-plain-s3,gen-chat-total-answer_line-s4,gen-chat-total-prefix-s5,gen-chat-top_user-json-s3,gen-chat-top_user-plain-s4,gen-chat-top_user-answer_line-s5,gen-chat-nth_item-json-s3,gen-chat-nth_item-plain-s4,gen-chat-nth_item-answer_line-s5,gen-chat-last_city-plain-s3,gen-chat-last_city-answer_line-s4,gen-chat-last_city-prefix-s5,gen-chat-first_after-prefix-s3,gen-chat-first_after-json-s4,gen-chat-first_after-plain-s5,gen-chat-distinct-json-s3,gen-chat-distinct-plain-s4,gen-chat-distinct-answer_line-s5,gen-chat-day_span-prefix-s2,gen-chat-day_span-json-s3,gen-chat-day_span-plain-s4,gen-chat-day_span-answer_line-s5,gen-chat-semantic_total-plain-s2,gen-chat-semantic_total-answer_line-s3,gen-chat-semantic_total-prefix-s4,gen-chat-semantic_total-json-s5,gen-pipe-day_span-prefix-s2,gen-pipe-day_span-json-s3,gen-pipe-day_span-plain-s4,gen-pipe-day_span-answer_line-s5,gen-pipe-semantic_total-plain-s2,gen-pipe-semantic_total-answer_line-s3,gen-pipe-semantic_total-prefix-s4,gen-pipe-semantic_total-json-s5,gen-kv-day_span-plain-s2,gen-kv-day_span-answer_line-s3,gen-kv-day_span-prefix-s4,gen-kv-day_span-json-s5,gen-kv-semantic_total-prefix-s2,gen-kv-semantic_total-json-s3,gen-kv-semantic_total-plain-s4,gen-kv-semantic_total-answer_line-s5,gen-jsonl-day_span-json-s2,gen-jsonl-day_span-plain-s3,gen-jsonl-day_span-answer_line-s4,gen-jsonl-day_span-prefix-s5,gen-jsonl-semantic_total-answer_line-s2,gen-jsonl-semantic_total-prefix-s3,gen-jsonl-semantic_total-json-s4,gen-jsonl-semantic_total-plain-s5,gen-csv-day_span-answer_line-s2,gen-csv-day_span-prefix-s3,gen-csv-day_span-json-s4,gen-csv-day_span-plain-s5,gen-csv-semantic_total-json-s2,gen-csv-semantic_total-plain-s3,gen-csv-semantic_total-answer_line-s4,gen-csv-semantic_total-prefix-s5,gen-table-day_span-json-s2,gen-table-day_span-plain-s3,gen-table-day_span-answer_line-s4,gen-table-day_span-prefix-s5,gen-table-semantic_total-answer_line-s2,gen-table-semantic_total-prefix-s3,gen-table-semantic_total-json-s4,gen-table-semantic_total-plain-s5

Secondary arm (added 2026-10-24 05:30 UTC, before it ran and before any E58 result; exploratory, not part of the primary
test): arm A's protocol with the weakest checkpoint on held-out discovery, `proposer-rl-9-40` (round four, seed 9;
22 of 74), run `e58a2-4br4-s9-best8`, analysed the same way against arm B.

## Amendment 1 (2026-10-24 08:50 UTC, before unblinding)

At the time of this amendment, arm B had 97 of 141 families finished, and neither arm's accepted libraries had been
scored on fresh tasks or inspected. The amendment follows an independent design review by Codex
(`codex-reviews/2026-10-24-design-and-next-experiments.md` in the run store). It does not change the primary analysis,
the margin or the arms.

- **Completeness.** The analysis runs only with `--require-complete`, which refuses to run unless all 141 families
  have a finished run in both arms. The earlier code counted an unfinished family as not accepted, which would
  have favoured the arm that finished first. An interrupted run resumes; it is never scored as a failure.
- **Errors.** Fresh-task errors are counted per accepted library and reported apart from wrong answers. The
  registered outcome, fresh mean of 0.9 or more, is unchanged.
- **Tokens.** Input tokens are now reported alongside output tokens, as the registration promised. Candidates
  without usage records are counted.
- **Sensitivity analyses.** These are fixed now and do not change the primary verdict:
  1. The same interval, resampling the 37 format-question templates instead of the 141 families. Families that
     share a template are not independent.
  2. The paired difference in fresh score, counting an unaccepted family as 0, with a family-bootstrap interval.

  Percentile intervals can mislead when few families are discordant, and they say nothing about variation across
  training seeds.
- **Checkpoint rationale, corrected.** Seed 16's 36 of 74 is the highest of the round-five mixed-RL seeds (32, 36,
  33), not the median. The checkpoint was still chosen before any E58 result, but it is not a typical checkpoint,
  and the result applies to it alone.
- **Secondary arm.** The secondary arm (seed 9) stays exploratory. It cannot rescue a primary failure, and no claim
  will rest on either checkpoint passing.
- **Scope.** A pass establishes non-inferiority at a 10-point margin for this checkpoint, this protocol and this
  generator panel. It does not establish equivalence or label efficiency in general.

## Result (unblinded 2026-10-24 ~13:20 UTC; arm B finished 12:56; analysis with `--require-complete`)

Full output: `H/e58-analysis/RESULT.txt`.

**Primary.**

| Arm | Accepted | Correct on fresh tasks | False acceptances | Proposer tokens per family (output / input) |
| --- | --- | --- | --- | --- |
| A | 85 | 75 of 141 | 10 | 4.1k / 16.9k |
| B | 73 | 73 of 141 | 0 | 50.2k / 13.6k |

- Paired: +22/-20, exact McNemar p 0.878.
- A - B = +0.014, 95% family-bootstrap interval [-0.078, +0.106]. The lower bound is above -0.10: **non-inferior,
  as pre-registered.**
- No accepted library in either arm had fresh-task execution errors.

**Pre-specified sensitivity analyses** (Amendment 1).
- Resampling the 37 format-question templates: [-0.113, +0.139]. The lower bound crosses the margin, so the verdict
  is not robust to treating templates as the unit.
- Fresh score with unaccepted families counted as 0: +0.064 [-0.025, +0.154].

**Secondary arm (exploratory; weakest checkpoint, round-four seed 9).**
- A 67 of 141 (76 accepted, 9 false acceptances) against B 73.
- +19/-25, p 0.451; -0.043 [-0.135, +0.050]. Non-inferiority not shown.
- Template-clustered interval [-0.167, +0.079].
- As registered, the secondary arm does not rescue or qualify the primary verdict, but it bounds its scope to the
  tested checkpoint.

**What the result licenses** (per Amendment 1's scope): non-inferiority at a 10-point margin for this checkpoint,
this protocol and this generator panel, with families as the unit. It is not established with templates as the
unit, not for the weakest checkpoint, and not as equivalence or general label efficiency. It comes at the cost of
10 false acceptances, against none for the labelled protocol.

## E72: a label-free metamorphic check for day-span libraries (added 2026-10-24 13:36 UTC, before the run; exploratory, after E58's unblinding)

This follows Codex's E58 review. All three of E58 arm A's day-span false acceptances subtract full timestamps
instead of calendar dates.

**Check** (`e72_metamorphic.py`). For each accepted day-span library in E58 arms A and B, generate 30 tasks on a
new split, "metamorphic", with 400 records. Then make a variant of each task with the same records, dates, order,
question and answer, but each record's time of day redrawn (sorted within its date). The library is flagged if any
output changes. No labels are used.

**Outcome.** Flags on wrong libraries (from E58's fresh scoring) and on correct ones. A posthoc exploratory test of
one targeted check, not a change to E58.

**Prediction.** It flags all 3 wrong arm-A libraries and none of the correct ones.

**E72 result (13:37 UTC).**
- Flagged 3 of 3 wrong libraries: `gen-pipe-day_span-*`, `gen-csv-day_span-*` and `gen-table-day_span-plain-s3`.
  The last scores 0.5 on fresh tasks and changed 4 of 30 answers.
- Flagged 0 of 22 correct libraries (arm A 4, arm B 18).
- Prediction met.
