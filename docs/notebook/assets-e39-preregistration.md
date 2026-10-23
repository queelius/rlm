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
