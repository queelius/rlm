# E39 pre-registration (written 2026-10-23 08:25 UTC, before either run started)

Question: does a small trained proposer with best-of-8 sampling and three labels bootstrap as many libraries as the
thinking 8B with the standard protocol and 30 labels, on families neither has seen?

Families: the 30 held-out families of data seed 1 that no earlier run used (family_gen.family_ids("heldout", (1,))
minus the 7 families any earlier run used, all gen-chat-*-s1 from the 8B's round-two run and hence possibly in the
4B's SFT data):
gen-prose-count-json-s1,gen-prose-total-json-s1,gen-prose-top_user-prefix-s1,gen-prose-nth_item-prefix-s1,gen-prose-last_city-json-s1,gen-prose-first_after-answer_line-s1,gen-prose-distinct-prefix-s1,gen-prose-day_span-prefix-s1,gen-prose-semantic_total-plain-s1,gen-xmlish-count-plain-s1,gen-xmlish-total-plain-s1,gen-xmlish-top_user-json-s1,gen-xmlish-nth_item-json-s1,gen-xmlish-last_city-plain-s1,gen-xmlish-first_after-prefix-s1,gen-xmlish-distinct-json-s1,gen-xmlish-day_span-json-s1,gen-xmlish-semantic_total-answer_line-s1,gen-chat-day_span-answer_line-s1,gen-chat-semantic_total-json-s1,gen-pipe-day_span-answer_line-s1,gen-pipe-semantic_total-json-s1,gen-kv-day_span-json-s1,gen-kv-semantic_total-answer_line-s1,gen-jsonl-day_span-prefix-s1,gen-jsonl-semantic_total-plain-s1,gen-csv-day_span-plain-s1,gen-csv-semantic_total-prefix-s1,gen-table-day_span-prefix-s1,gen-table-semantic_total-plain-s1

Arm A (e39a-4brl4-best8): proposer-rl-5-40 (round four), 8 round-0 candidates per family, no repair, no early
stopping (discovery_bench --threshold 2 --repair-rounds 0 --candidates 8), boot seed 0. Accept, per family, the
first candidate (in sampling order) that is valid (runs on all 30 teaching inputs, varied outputs) and reproduces the
3 examples shown in its prompt. No other label is used for acceptance.
Arm B (e39b-8b-standard): qwen3-8b-awq, thinking, standard protocol (4 candidates, 1 repair round, acceptance at
teaching mean >= 0.9 over all 30 labels, repair prompts show failing labelled cases), boot seed 0.

Primary outcome: families with an accepted library whose score on 30 fresh tasks (split "fresh", 400 records) is
>= 0.9. Secondary: proposer output tokens per family; false acceptances (accepted but fresh < 0.9).
Prediction: A >= B - 3 (non-inferior within 3 families) at <= 1/10 of B's tokens; A's false acceptances <= 1.
Test: exact McNemar on per-family success. A result outside the prediction is reported as such.

## Replication addendum (written 2026-10-23 15:05 UTC, before either replication run started)

E39r repeats E39 unchanged except `--boot-seed 1` in both arms (different shown examples, sampling and, for arm B,
repair draws), on the same 30 families: arm A `e39r-a-4brl4-best8`, arm B `e39r-b-8b-standard`, analysed with the
same `e39_analysis.py`. The same three predictions apply. The pooled result (60 family attempts per arm) is reported
alongside each seed.

## Exploratory arm C (added 2026-10-23 17:55 UTC, before the run; not part of the registered predictions)

E39 changed the proposer and the protocol together. Arm C (`e39c-8b-best8`) runs the thinking 8B under arm A's
protocol (8 round-0 candidates, no repair, no early stopping, acceptance by the 3 shown examples), boot seed 0, same
30 families, scored with the same acceptance rule and fresh tasks. A against C isolates the proposer; C against B
isolates the protocol (three labels and sampling against thirty labels and repair).

Arm C with boot seed 1 (`e39rc-8b-best8`, added 2026-10-23 19:50 UTC before the run; exploratory) completes the
arms across both seeds.

## Checkpoint robustness (added 2026-10-23 22:00 UTC, before the runs; exploratory)

E43, E45 and E46 showed that arm A's checkpoint (round four, RL seed 5) is the best of four RL seeds of its recipe
(42 of 74 held out; the others 29, 22, 33). Arm A is therefore rerun unchanged (boot seed 0, same 30 families, same
acceptance rule and fresh-task scoring) with round four's other checkpoints (`proposer-rl-6-40`, `-9-40`, `-10-40`)
and the round-five recipe's seed-17 checkpoint (`proposer-rl-17-40`): runs `e39a-4br4-s{6,9,10}-best8` and
`e39a-4br5b-s17-best8`, each scored against arm B with `e39_analysis.py`. Question: does the three-label 4B match the
8B with thirty labels for a typical checkpoint, or only for the best one?

Further checkpoints (added 2026-10-24 00:20 UTC, before the runs; exploratory): arm A unchanged with the round-five
recipe's other checkpoints (`proposer-rl-16-40`, mixed RL; `proposer-rl-25-40`, code-only RL) and round seven's seed
27 (`proposer-rl-27-40`): runs `e39a-4br5b-s16-best8`, `e39a-4br5a-s25-best8`, `e39a-4br7-s27-best8`.

## E50: the cheap protocol on a real family (added 2026-10-24 01:00 UTC, before the runs; exploratory)

MRCR is program-solvable with exact answers. Under the standard protocol (8 candidates, 2 labelled repair rounds,
30 labels) the thinking 8B found a correct library on 2 of 5 bootstrap seeds (E21). E50 gives the same 8B, on the same
five seeds (0 to 4), 16 round-0 candidates with no repair and no early stopping (`bootstrap_library.py --candidates 16
--repair-rounds 0 --threshold 2`): runs `e50-mrcr-8b-best16-s{0..4}`. Accept, per seed, the first valid candidate that
reproduces the 3 shown examples; check accepted libraries held out (`run_routine.py --family mrcr`, 2 and 4 needles).
Question: does three-label acceptance find MRCR's library as often as the labelled protocol, without false acceptances?

## E51: more samples under the cheap protocol (added 2026-10-24 01:05 UTC, before the runs; exploratory)

Arm A's protocol with 32 round-0 candidates instead of 8 (boot seed 0, same 30 families, same acceptance rule and
fresh-task scoring), for the strongest checkpoint under E39 (`proposer-rl-17-40`) and the weakest on held-out
discovery (`proposer-rl-9-40`): runs `e51-4br5b-s17-best32`, `e51-4br4-s9-best32`. Question: does sampling more push
the 4B past the 8B with thirty labels (13 of 30), and does it close the gap between checkpoints?

## E52: three-label repair on MRCR (added 2026-10-24 01:55 UTC, before the runs; exploratory)

E50 found no MRCR library in 4 of 4 finished seeds with 16 round-0 candidates, and both standard-protocol successes
(E21, seeds 0 and 4) came from the first repair round, which showed failing teaching cases with their answers. E52
keeps the standard budget (8 candidates, 2 repair rounds, same 8B, seeds 0 to 4) but uses only the three shown
examples' labels: `bootstrap_library.py --shown-labels-only` ranks candidates by shown examples reproduced, inputs
answered and varied outputs, repairs with the failing shown examples plus inputs that crash or return nothing (no other
expected answers), and accepts a candidate that reproduces all three shown examples, answers every input and varies.
Runs `e52-mrcr-8b-shownrepair-s{0..4}`; accepted libraries checked held out (`run_routine.py --family mrcr`, 2 and 4
needles). Question: does three-label repair recover what labelled repair finds (2 of 5)?

E53 (added 2026-10-24 02:20 UTC, before the run; exploratory): arm A's families and checkpoint `proposer-rl-17-40` with
three-label repair (`discovery_bench.py --candidates 8 --repair-rounds 2 --shown-labels-only`, boot seed 0): run
`e53-4br5b-s17-shownrepair`, scored with the arm A rule on its accepted libraries. Question: does three-label repair
add to best-of-8 for the 4B on exact families, as labelled repair did for the 8B on MRCR?

E52b (added 2026-10-24 02:45 UTC, before the runs; exploratory): E52's five runs continued from their 24 saved
candidates to 4 repair rounds (rounds 3 and 4 added, same three-label rules): runs `e52b-mrcr-8b-shownrepair4-s{0..4}`
(copies of E52's candidates; E52's own directories are unchanged).

## E54 and E55 (added 2026-10-24 03:15 UTC, before the runs; exploratory)

E52b (four three-label repair rounds) found fully correct MRCR libraries on 2 of 5 seeds (1, 2; seed 0's accepted
library fails beyond the teaching lengths), as did the standard protocol with two labelled repair rounds (seeds 0, 4),
but the budgets differ. E54 compares the protocols at a matched budget on fresh seeds 5 to 9: the thinking 8B,
8 candidates, 4 repair rounds, labelled (`e54-mrcr-8b-labelled4-s{5..9}`, threshold 0.9 over 30 labels) against
three-label (`e54-mrcr-8b-shown4-s{5..9}`, `--shown-labels-only`). E55 runs the 14B with three-label repair, 4 rounds,
seeds 0 to 4 (`e55-mrcr-14b-shown4-s{0..4}`; the 14B had 2 of 5 under the standard protocol). Accepted libraries are
checked held out at 2 and 4 needles; success = 0.9 or more held out at both.

E56 (added 2026-10-24 03:45 UTC, before the runs; exploratory): E54's matched-budget comparison with the 14B (8008),
seeds 5 to 9: `e56-mrcr-14b-labelled4-s{5..9}` against `e56-mrcr-14b-shown4-s{5..9}`; same rules and held-out check.
The 14B finds MRCR's program more often, which gives the protocol comparison more power. Primary measure for E54 and
E56 (decided now, before results): seeds with a library correct held out (0.9 or more at 2 and 4 needles), per arm,
plus per-candidate rates of correct libraries in round 0 and in repair rounds.

E51r (added 2026-10-24 04:10 UTC, before the runs; exploratory): E51 repeated with boot seed 1 (`e51r-4br5b-s17-best32`,
`e51r-4br4-s9-best32`), scored against arm B's boot-seed-1 run (`e39r-b-8b-standard`); pooled with E51 over 60 family
attempts per checkpoint.

**E51r result (2026-10-24 08:55 UTC).** Boot seed 1, against the 8B's 16 of 30: seed 9 scores 16 (+4/-4, p 1.0), seed 17 scores 19 (+6/-3, p 0.51, one false acceptance). Pooled with E51 against the 8B's 29 of 60: seed 9 gets 32 (+9/-6, p 0.61), seed 17 gets 37 (+12/-4, p 0.077). The 60 attempts are 30 families under two boot seeds, and these are exploratory comparisons after E39; E58 is the powered test.

## E57: repair feedback with sampling luck removed (added 2026-10-24 05:20 UTC, before the runs; exploratory)

Following the independent review's third recommendation: for MRCR seeds 10 to 14, the thinking 8B draws one shared
pool of 8 round-0 candidates (`e57-mrcr-8b-pool-s{sd}`, no repair, no early stopping). Each pool is copied into two
branches that resume from it with 4 repair rounds: labelled (`e57-mrcr-8b-labelled-s{sd}`, acceptance at teaching mean
0.9 or more, repair shows failing teaching cases with answers) and three-label (`e57-mrcr-8b-shown-s{sd}`,
`--shown-labels-only`). Both branches see identical round-0 candidates, so differences come from repair and acceptance
alone. Measure: per seed and arm, an accepted library correct held out (0.9 or more at 2 and 4 needles).

**E57 serving change (2026-10-24 08:36 UTC, before any E57 run started).** GPU1 was idle while E57 waited behind E58's
8B arm on GPU0, so all 15 E57 runs now use a second copy of the same 8B server on GPU1 (`qwen3-8b-awq`, same weights,
vLLM build and flags, port 8010, gpu-memory-utilization 0.44 instead of 0.45). Pools and both branches of every seed use
this one server, so the within-seed matched comparison is unaffected. Pools no longer wait for E58.

**E57 analysis amendment (2026-10-24 08:50 UTC).** At this point three pools were running, and the branches of
the other two (s13, s14) had just started. No E57 outcome had been inspected. The amendment follows the Codex design review
(`codex-reviews/2026-10-24-design-and-next-experiments.md`).

What E57 can and cannot attribute:
- The two branches share a round-0 pool. Beyond that, they differ in repair feedback, in how they rank the pool
  when picking the parent to repair, in acceptance and in stopping. Continuation sampling is not seeded (`--seed`
  only selects teaching examples).
- So E57 estimates a whole-protocol contrast given a shared pool. A difference cannot be attributed to the feedback
  text alone.

Reported per seed:
- the pool's own best held-out result;
- each branch's trajectory: rounds, candidates, and input and output tokens consumed;
- acceptance;
- held-out means at 2 and 4 needles at full precision, and exact-match counts out of 64;
- false acceptances.

Each branch's first rows will be checked against its pool's `candidates.jsonl` by hash. The analysis reports
discordant pairs and an exact McNemar test. With five seeds, even five of five discordances favouring one branch
give p = 0.0625, so the result is descriptive only: no equivalence or non-inferiority claim. A follow-up isolating
the feedback (one frozen parent per pool, with matched continuations under thirty-label, three-label and generic
feedback) is in the resume document's work list.

**Held-out scorer fix (same time).** `mrcr_heldout_batch.py` used to parse the rounded mean from the log line, and it
did not notice that `run_routine.py` appends to an existing `rows.jsonl`. It now:
- recomputes the mean from per-task rows;
- checks the row count against the tasks run;
- refuses to reuse a work directory;
- reports exact matches and errors.

Recomputing the earlier E52b, E55 and E56 checks changes no conclusion:
- Every library that scored 1.0 matched all 64 held-out tasks exactly at both needle counts.
- E52b s0 scores 0.786 and 0.807 (exactly correct on 50 and 51 of 64).
- E55 s0's work directories had held each row twice, from a rerun; all rows scored 1.0, so its mean was unaffected.

## E59: do accepted libraries survive meaning-preserving shifts? (added 2026-10-24 08:52 UTC, launched in the same step; exploratory)

This was proposed by the Codex design review (priority 2 in both reviews). E39's "unseen" families are new variants
of the same generator, and fresh tasks keep each family's wording and nuisance settings. E59 freezes the accepted
libraries of E39's pre-registered arms and scores each one on 30 tasks per condition from a new split, "stress"
(`stress_gen.py`). The arms are A (`e39a-4brl4-best8`, `e39r-a-4brl4-best8`, rule a) and B (`e39b-8b-standard`,
`e39r-b-8b-standard`, rule b).

Conditions:
- **control:** as generated, 400 records.
- **paraphrase:** the same records and answers, with the question reworded once per question form.
- **nuisance:** the same records and answers, with field order, separator and question prefix redrawn. For prose and
  chat, only the question prefix changes.
- **long:** 2000 records.

Success is 0.9 or more. Outcome: among libraries that succeed on control, the share that also succeeds under each
shift, by arm.

Predictions, written after a single-library smoke test that scored 1.0 on control, 0 on paraphrase, 0 on nuisance
(only the question prefix changed) and 1.0 on long:
- **Long:** at least 80% retained.
- **Paraphrase:** at most 30% retained.
- **Nuisance:** at most 50% retained.
- **A vs B:** no directional prediction. Retention is compared with Fisher's exact test.

If paraphrase or nuisance retention is low, then "bootstrapped library" means a template-bound program, and every
generalization claim must say so. Possible follow-ups are teaching sets that vary wording and nuisance settings, or a
helper-mediated question parser. Run: `RUNS/../e59-stress` in the run store (`H/e59-stress/rows.jsonl`).

**E59 result (2026-10-24 08:53 UTC).** Of 55 libraries correct on control, the number still correct under each
condition:

| Condition | Arm A (4B, 26) | Arm B (8B, 29) | Fisher p |
| --- | --- | --- | --- |
| Long | 26 | 29 | 1.0 |
| Paraphrase | 0 | 3 | 0.24 |
| Nuisance | 4 | 7 | 0.51 |

- **Question prefix only:** on prose and chat, where nuisance changes only the question prefix, 5 of 26 are retained.
- **Paraphrase survivors:** day span (2 of 12) and first-after (1 of 5). Their single slot (a name or a time) is
  extracted by a generic pattern.
- **Errors:** 13 library-condition runs raised errors (8 nuisance, 5 paraphrase). They are counted as failures.

All three predictions were met. The libraries are template-bound programs. Generalization claims for bootstrapping
must say "to new instances and lengths of the same template", not "to the task".

## E60: can bootstrapping produce template-robust libraries? (added 2026-10-24 09:03 UTC, before the runs started at 09:05; exploratory)

(The comment in `stress_gen.py` dating paraphrase 2 to 09:10 is wrong: it was drafted at 08:58 and revised at 09:04, before any E60 run.)

The design was revised after a Codex review of a draft (`codex-reviews/2026-10-24-e59-e60-review.md`). In the draft,
seed 0's three shown examples were all in the original wording. Nuisance 2 repeated nuisance 1's rendering on four
families. The teaching tasks differed from E39's, and the outcome was conditional on control survivors. All four
problems are fixed below.

Families: E39's 30, boot seed 0. The proposer and protocol match E39 arm A: `proposer-rl-5-40`, helper `qwen3-4b`
on 8001, 8 round-0 candidates, no repair, no early stop, and acceptance by the first valid candidate that reproduces
the 3 shown examples.

**Arms:**

- **M (mixed demonstrations and validation):** E39's own 30 teaching tasks (same records, slots, answers and order),
  re-rendered 10 each in the original wording, paraphrase 1 and nuisance 1. The three shown positions are E39's
  shown tasks, one per wording (`stress_gen.make_teach`, family `mix:<fid>`). This tests the mixed package as a
  whole: demonstrations and the validity filter both change.
- **C (contract only):** E39's teaching set and shown examples exactly. The proposer prompt ends with a note that
  question wording, question prefix and field order or separator may differ in use, and that it should extract the
  question's parameters in a way that survives rewording.
- **Baseline (A0):** E39 arm A's frozen boot-seed-0 libraries (`e39a-4brl4-best8`).

**Scoring.** Every accepted library is scored on 30 stress-split tasks under control, paraphrase 1 and nuisance 1
(variants the M arm was shown), paraphrase 2 and nuisance 2 (never shown to any arm), and long. Paraphrase 2 is
frozen at this commit, with "qty" and "a furniture" reworded.

**Primary outcome:** robust yield over all 30 families, counting non-acceptance as failure. A family counts as robust
when the library is accepted and scores 0.9 or more on both control and paraphrase 2. M vs A0 and C vs A0 are
compared paired by family: discordances and exact McNemar.

**Secondary outcomes:**
- the same yield with nuisance 2 in place of paraphrase 2;
- known-variant correctness (paraphrase 1, nuisance 1);
- acceptance, control correctness, long, and execution faults.

**Predictions:**
- A0's robust yield is 0, since E59 found 0 of 26 arm A libraries survived paraphrase 2.
- M's yield is above 0, with nuisance-2 yield higher than paraphrase-2 yield.
- M accepts fewer libraries than A0's 12, because the validity filter now runs on varied inputs.
- C: no directional prediction.

About 12 accepted libraries per arm can show a difference from 0 but cannot establish equivalence. Thirty tasks per
library are not independent replicates. Each shift has one wording per question form.

**E60 arm MC (added 2026-10-24 09:24 UTC; E60 M and C were still running and no E60 library had been scored or
inspected).** Mixed teaching and the contract note together (`e60mc-mixcontract-4brl4-best8`). Everything else is the
same as M and C. It completes the 2x2 design: neither (A0), mixed teaching only (M), contract note only (C), both (MC).
It is scored and analysed the same way. No directional prediction beyond M's.

**E60 result, arms M and C (2026-10-24 09:31 UTC; `H/e60/analysis-m-c.txt`).** All counts are out of 30 families.

| Measure | A0 | M | C |
| --- | --- | --- | --- |
| Accepted | 12 | 4 (+1/-9 vs A0, p 0.021) | 14 (+2/-0, p 0.5) |
| All accepted libraries correct on control | yes | yes | yes |
| Robust, control + paraphrase 2 (primary) | 0 | 0 | 0 |
| Robust, control + nuisance 2 | 2 | 0 | 7 (+5/-0, p 0.062) |
| Known variants (paraphrase 1 and nuisance 1) | 0 | 4 | 0 |
| Long | 12 | 4 | 14 |
| Execution faults | 0 | 0 | 0 |

Predictions:
- A0's yield is 0: met.
- M's yield is above 0: not met.
- M's nuisance-2 yield exceeds its paraphrase-2 yield: not met (both 0).
- M accepts fewer libraries: met.

Reading, revised after the Codex review (`codex-reviews/2026-10-24-e60-results-and-h-arm.md`):
- M's four accepted libraries fit the demonstrated renderings and fail both unseen shifts. They are regex
  alternatives for the two demonstrated phrasings (one uses a shared substring, one adds a looser pattern), and
  none calls the helper. The data do not show that the varied demonstrations caused this.
- C's 7 against 2 on nuisance 2 is suggestive (five gains, no losses, p 0.0625), not established. Its 14
  against 12 acceptances is not shown to differ.
- No arm produced a library robust to the unseen rewording.

## E60 arm H: the helper parses the question, code computes the answer (added 2026-10-24 09:38 UTC, before the run)

The draft was revised after the Codex review (`codex-reviews/2026-10-24-e60-results-and-h-arm.md`).

**Setup.** As arm C (E39's teaching set and shown examples, `proposer-rl-5-40`, 8 round-0 candidates, arm A's
acceptance rule), with a different prompt suffix. The suffix asks the proposer to:
- define the parameter keys its code needs;
- call `ask()` once for question parsing, passing only the question and that schema, without the records;
- request JSON parameter values only, decode them with `json.loads`, and compute and format the answer in Python;
- make further `ask()` calls only for judgments code cannot make.

Run: `e60h-helperparse-4brl4-best8`.

**Primary comparison:** H against C, paired by family, with A0 also reported. The outcome is the E60 robust yield
(accepted, and 0.9 or more on control and on paraphrase 2), over all 30 families.

**Prediction (unconditional):** H's robust yield is at least 5 of 30.

**Also reported:**
- adoption: the share of accepted libraries whose code sends the question alone to `ask()` and decodes JSON, judged
  by reading the code;
- helper calls per task on each condition (now logged in the stress rows);
- acceptance and execution faults.

**Analysis requirements:**
- every accepted library must have all six condition rows, with no duplicates (enforced by `e60_analysis.py`);
- each run must have finished all 30 families before scoring.

A failure would not show that neural question parsing cannot work. It jointly tests whether the 4B adopts the recipe,
finds the question, and implements the parser and the computation.

## E61: can a small model parse reworded questions into a family's parameters? (added 2026-10-24 09:40 UTC; exploratory)

This sets the ceiling of the hybrid design: the helper parses the question, code computes the answer.

**Method.**
- Inputs: E39's 30 families, 30 stress tasks each under control, paraphrase 2 and nuisance 2.
- The parser receives only the question's final paragraph or message (label and answer instruction included, no
  records), plus the family's key schema with a one-line description per key (`e61_parse.py`).
- It returns JSON, compared with oracle parameters read from the same task's original wording.
- Parsers: `qwen3-4b` (helper, port 8006) and `qwen3-0.6b-nt` (port 8004).

**Smoke test, which also fixed the prompt.**
- Bare key names were ambiguous: the 4B returned the whole question as "user", and "fruit items" as the category.
- With key descriptions, 2 tasks per condition: control 59/60, paraphrase 2 58/60, nuisance 2 59/60.

**Predictions.**
- The 4B parses at least 95% exactly in every condition.
- The 0.6B parses at least 80%.
- If both hold, E59's collapse is an architectural problem (question handling written as regex) and not a capacity
  limit of small models.

**E61 result (09:46 UTC; `H/e61/`).** Exact parameters, out of 900 tasks per condition:

| Parser | Control | Paraphrase 2 | Nuisance 2 |
| --- | --- | --- | --- |
| `qwen3-4b` | 881 (97.9%) | 861 (95.7%) | 884 (98.2%) |
| `qwen3-0.6b-nt` | 556 (62%) | 524 (58%) | 572 (64%) |

4B failure modes:
- first-after under paraphrase 2: 31/60; the time is returned with 'or later' attached;
- day-span: 13 control failures, where 'user' is returned literally;
- nth-item under paraphrase 2: 54/60.

The 0.6B often returns empty or literal key values.

Predictions:
- 4B at least 95% in every condition: met (barely, on paraphrase 2).
- 0.6B at least 80%: not met.

Reading: question parsing at 4B scale is robust to the shifts that broke the regex libraries. The hybrid design is
feasible with a 4B helper, not with this 0.6B prompt. Whether the proposer *writes* such libraries is arm H.

**E60 arm H2 (added 2026-10-24 10:05 UTC, before any arm H result was inspected).**
- Arm H repeated with the round-five seed-16 checkpoint (`proposer-rl-16-40`, E58's arm A checkpoint), as
  `e60h2-helperparse-4br5b-s16-best8`. It is queued to run after H.
- It is scored the same way. Its baseline is C, and it is compared with H for checkpoint dependence.
- No new prediction: H's prediction (robust yield at least 5 of 30) applies.

**E60 result, arms MC and H (10:37 UTC; `H/e60/analysis-all.txt`).** All counts are out of 30 families.

| Measure | MC | H |
| --- | --- | --- |
| Accepted | 9 | 11 |
| Robust, control + paraphrase 2 (primary) | 1 | 1 (C 0; pre-registered prediction of at least 5 not met) |
| Robust, control + nuisance 2 | 8 (vs A0 +7/-1, p 0.070) | 4 (vs C +1/-4) |
| Known variants | 7 (+7/-0, p 0.016) | 0 |
| Execution faults | 0 | 0 |

Adoption in H:
- 41 of 240 candidates contain `ask(`.
- 1 of 11 accepted libraries uses ask() plus json.loads on the question.
- That library (`gen-xmlish-total-plain-s1`) is H's only robust success: 0.9 on paraphrase 2, with 30 helper calls
  over 30 tasks.

Reading: the hybrid design works when it is written, but the 4B proposer rarely adopts it, and its attempts rarely
pass acceptance. The bottleneck is acquisition, not the interface.

## E68: does a question-canonicalization wrapper repair the template-bound libraries end to end? (added 2026-10-24 10:38 UTC, before the run; exploratory)

This is the frozen-executor test both Codex reviews ranked first, and it follows from E60 arm H: the proposer rarely
writes the hybrid interface, so the harness supplies it.

**Wrapper** (`e68_canon.py`).
- Each frozen E59 library (arms A and B, both boot seeds) is kept unchanged.
- The final paragraph or message of each task is replaced by: the family's original question prefix, the question re-rendered
  in the original template from parsed parameters, and the task's own answer instruction. The instruction is located
  by the generator's four forms; this is family-level template knowledge a library-specific wrapper would have.
- Records are untouched, so under nuisance 2 the field-order or separator change still reaches the library. The
  wrapper repairs the question interface only.

**Parameters.**
- Oracle: from the task's original wording. On paraphrase 2 this reproduces the control prompt exactly (0
  mismatches in 90 checks).
- E61's 4B parses and E61's 0.6B parses, for the same stress tasks; no new helper calls.

**Outcome.** Among libraries correct on control (55), the number still correct (0.9 or more) on paraphrase 2 and on
nuisance 2, for no wrapper (E59) and for each parser.

**Predictions.**
- Paraphrase 2: oracle 55 of 55. 4B wrapper at least 44 of 55 (80%), with first-after the expected loss (E61's 4B
  parses it at 52%). 0.6B wrapper below 30 of 55.
- Nuisance 2: prefix-only families (prose and chat) mostly repaired by the oracle and 4B wrappers. Structured formats
  stay limited by record parsing, so total oracle retention is between 11 (E59) and 40 of 55.

**E68 result (10:44 UTC; `H/e68/rows.jsonl`).** Libraries still correct, out of 55 correct on control:

| Condition | No wrapper (E59) | 4B parses | Oracle | 0.6B parses |
| --- | --- | --- | --- | --- |
| Paraphrase 2 | 3 | 54 (arm A 25 of 26, B 29 of 29) | 55 | 22 |
| Nuisance 2 | 11 | 48 | 48 | 26 |

- The 4B's only paraphrase-2 miss is an nth-item library at 0.87.
- Nuisance 2 by format: prose and chat 26 of 26 with the 4B or the oracle; structured formats 22 of 29, where
  the record layout changed.

Predictions:
- oracle 55: met;
- 4B at least 44: met;
- 0.6B below 30: met;
- nuisance oracle between 11 and 40: not met (48).

**E60 arm H8 (added 2026-10-24 10:46 UTC; queued after E57 on the second 8B server).** Arm H's instruction given to a
stronger proposer.
- Setup: `qwen3-8b-awq` with thinking, and E39 arm C's exact spec (`e39c-8b-best8`: 8 round-0 candidates, no
  repair) plus H's prompt suffix. Run: `e60h8-helperparse-8b-best8`.
- Baseline: E39 arm C, the same protocol without the suffix. Its libraries are scored on the stress conditions in
  the same way.
- Question: does a stronger proposer adopt helper question parsing, and does adoption bring robust yield?
- Prediction: adoption in at least 5 of the 8B's accepted libraries (against the 4B's 1 of 11), and robust yield at
  least 4 of 30.

## E69: can a small proposer edit a working library to use helper question parsing? (added 2026-10-24 10:50 UTC, before the run; exploratory)

This follows E60 H, where a 4B writing from scratch adopted helper parsing in 1 of 11 accepted libraries. It is
Codex's "repair frozen successes" alternative.

**Setup** (`e69_refactor.py`).
- For each of E39 arm A's 12 accepted libraries (boot seed 0), the proposer (`proposer-rl-5-40`) is shown the
  library's code, the same three examples it was bootstrapped from, and an instruction. The instruction: replace
  question parsing with one `ask()` call (the question only, JSON parameters out), and keep the rest.
- Up to 8 candidates per library. Accept the first that is valid and reproduces the shown examples (arm A's rule).
- The accepted edit is scored on all six stress conditions.
- Run: `e69-refactor-4brl4`, after H2.

**Smoke test (1 family, 2 candidates).** Both edits adopted the recipe. Both failed teaching (0.1 and 0.0), because
`json.loads` was called on the helper's raw reply with no handling of code fences.

**Outcome.** Among the 12 families: accepted edits; adoption (ask plus json.loads); robust yield (accepted, and 0.9
or more on control and on paraphrase 2). The comparison is A0 (0) and H (1 of 30 overall).

**Predictions.**
- At least 8 of the accepted edits use the helper.
- Acceptance is at most 8 of 12, because brittle JSON handling breaks many edits.
- Robust yield is at least 3 of 12.

**E60 result, arm H2 (11:12 UTC; `H/e60/analysis-h2.txt`).**
- 18 of 30 accepted.
- Robust on paraphrase 2: 0. Robust on nuisance 2: 3.
- Adoption: 0 of 18 accepted libraries; 30 of 240 candidates call `ask(`.
- Baseline for H8: E39 arm C (8B, no instruction) has 15 accepted, robust 1 on paraphrase 2 and 3 on nuisance 2.

**E69 result (11:50 UTC).**
- 11 of 12 accepted.
- Adoption: 2 of 11 accepted edits use the helper; 16 of the 29 drawn edits do.
- Robust (control and paraphrase 2): 1 of 12 (`gen-xmlish-total-plain-s1`).
- Predictions: none met.

The acceptance rule takes the first edit that reproduces the shown examples, which is usually a non-adopting one.

**E69b (added 11:50 UTC, before the run).** The same, with `--require-helper`: accept only an edit that is valid,
reproduces the shown examples, and uses ask() plus json.loads. Run: `e69b-refactor-adopt-4brl4`.
- Question: can a 4B produce a *working* helper-parsing edit?
- Prediction: at least 6 of 12 families get an accepted adopting edit, and at least 4 are robust on paraphrase 2.

**E69b2 (added 2026-10-24 11:51 UTC, before the run).** E69b with the round-five seed-16 checkpoint
(`proposer-rl-16-40`), editing the same frozen E39 arm A libraries. Run: `e69b2-refactor-adopt-4br5b-s16`, after
E69b. It is a replication across checkpoints and carries E69b's prediction.

**E69b result (12:13 UTC).**
- Accepted adopting edits: 3 of 12 families (prose-count, xmlish-count, xmlish-total).
- Paraphrase 2 scores: 0.63, 0.93 and 1.0, so 2 robust.
- Nuisance 2 scores: 0.97, 0.0 and 0.87.
- 22 of 78 drawn edits adopt.
- Predictions: not met.

**E57 result (12:35 UTC; `H/mrcr-heldout/e57.jsonl`).**
- **Pools:** every branch's first 8 rows are identical to its pool (checked). No pool candidate is correct
  (best teaching scores 0.214, 0.214, 0.076, 0.000, 0.214).

| Seed | 30-label branch | 3-label branch |
| --- | --- | --- |
| s10 | not accepted (40 rows) | accepted in round 3; false acceptance (0.786, 0.789) |
| s11 | not accepted | not accepted |
| s12 | not accepted | not accepted |
| s13 | not accepted | accepted in round 4; correct (1.0, 1.0; 64 of 64 exact at both needle counts) |
| s14 | accepted in round 2; correct (1.0, 1.0) | accepted in round 1; correct (1.0, 1.0) |

- **Correct:** 30-label 1 of 5, 3-label 2 of 5. Discordant pairs: +1/-0 for 3-label; p 1.0.
- **New output tokens per branch:** 30-label 104k to 332k; 3-label 68k to 351k.
- **Reading** (descriptive, as pre-registered): with sampling luck removed at round 0, three-label repair
  matched or exceeded thirty-label repair on these five seeds, and admitted one library that fails beyond the
  teaching lengths.

**E69b2 result (12:40 UTC).**
- 3 of 12 families get a working adopting edit: first-after (paraphrase 2: 0.0), xmlish-count (0.87) and
  xmlish-total (1.0).
- 1 is robust. The xmlish-total family is the one success in E60 H, E69, E69b and E69b2 alike.

**E70 first cell (added 12:40 UTC, before the run): REPL with primitives only, OOLONG.**
- Setup: the untrained 4B in this repository's runtime, default harness (REPL, helper calls, no library), on the
  293 held-out rows. Same lengths and per-cell sampling as E09. Run: `e70-oolong-repl-prims-4b`.
- This is not yet E70's matched design (that needs the adapter, equal budgets and normalized docs). It is the
  plain 'tools alone' reference for OOLONG.
- Reference points: direct 0.208, library 0.672, and this harness on a 72-row sample 0.111.
- Prediction: at most 0.25.

**E70 second cell (added 2026-10-24 12:41 UTC, before the run): REPL with primitives only, LongBench-v2, all 503
questions.**
- Setup: the untrained 4B, default harness (no library), `family_eval.py --family longbench --per-size 300`.
  Queued after the OOLONG cell. Run: `e70-longbench-repl-prims-4b`.
- Reference points: direct reading 0.356 (E32e), the retrieve-then-read library 0.427 (E32e, 0.6B caller), and this
  harness on 150 questions 0.120 (E32).
- Prediction: at most 0.30, below direct reading.

**E69b8 (added 2026-10-24 13:26 UTC, before the run).** E69b with the thinking 8B (`qwen3-8b-awq`, 16k output
tokens) editing the same frozen 4B libraries (E39 arm A, boot seed 0), with adoption required for acceptance.
- Run: `e69b8-refactor-adopt-8b`, on gpu0 after E54.
- Together with H8 (the 8B writing from scratch), it completes the who-can-supply-the-interface table: {4B, 8B} ×
  {write, edit}.
- Prediction: working adopting edits for at least 8 of 12 families, at least 6 of them robust on paraphrase 2.
