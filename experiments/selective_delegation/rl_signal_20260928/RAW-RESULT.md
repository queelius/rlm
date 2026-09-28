---
date: 2026-09-28
status: complete_native_audited_collection
scope: exposed_TRAIN8_four_execution_seeds_no_learning_claim
collection: textcraft-rl-assist-20260928-001/raw/collect-0001
report: analysis-rl-signal-20260928-001/raw-final.json
report_sha256: 6bc45043f95cdc1c2f4db5d63a5dc2090fdf57aac09c22c6fb0dbcef7f814a30
analyzer_sha256: de7dc43a1ed94c6208ff1d9f8472d4a063e70cea80922009818c26f440635afa
---

# Raw collection: useful but concentrated terminal-reward signal

The unchanged public-discovery checkpoint23 succeeds on **24/32** native-audited
episodes. Four of eight task groups have mixed rewards; the remaining four are
all-success. This supplies a real signed update signal but does not establish
learning, unseen-task benefit, or recursion.

| Official TRAIN suffix | 2040 | 1791 | 2294 | 1680 | 429 | 404 | 465 | 860 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Successes / four seeds | 4 | 4 | 4 | 2 | 1 | 3 | 2 | 4 |

## Exact objective exposure

Existing leave-one-out credit is used unchanged: each trajectory receives its
native terminal reward minus the other three same-task rewards' mean. Every
emitted token, including errors and EOS, participates; denominator remains 32.

| Credit | Trajectories | Calls | Emitted tokens | Sum of absolute advantage × tokens |
|---|---:|---:|---:|---:|
| Positive | 8 | 188 | 5,736 | 3,240.33 |
| Negative | 8 | 395 | 14,051 | 7,178.67 |
| Zero | 16 | 207 | 5,425 | 0 |
| Nonzero total | 16 | 583 | 19,787 | 10,419.00 |

Signed advantage-weighted token mass is −3,938.33 before `/32`. These are exposure
counts, not gradient norms. The larger negative mass reflects longer failed
trajectories under a legitimate token-sum score-function objective; it is not an
objective bug or evidence that failures should be length-normalized.

Sixty-two failed craft calls, totaling 2,407 emitted tokens, receive positive
trajectory credit because their episodes eventually succeed. Another 242 failed
craft calls receive negative credit and 44 zero credit. Successful-trajectory
credit is **not causal step credit**, and does not by itself guarantee the
probability of any individual mistaken action increases after shared-parameter
optimization and clipping.

## Behavior and physical cost

790 returned calls, no transport failures; 25,212 output tokens and 1,766,263
prompt tokens; 1,977.89 native-service seconds (32.96 minutes). There are 348 native
craft errors and one schema error. Thirty-one episodes finish explicitly, seven
of those unsuccessfully; the remaining failed episode hits the context cap.

At the identical initial prefix, all four samples for each of all eight tasks
choose exactly the same root `get_info`: empirical initial semantic entropy is
zero. This is not absence of later exploration. Among 758 later calls, there are
661 exact input-token prefixes, including 45 repeated-prefix groups. Fifteen of
those have different full semantic actions; fourteen still differ after removing
craft ingredients. The other histories are different states, not replicates.

Concrete example: at the same fourth-call prefix for TRAIN2040, three seeds craft
`o0_i1_10` with output count 3, while one queries its recipe again. Conversely, at
a repeated prefix for TRAIN860, two different ingredient maps request the same
`a8_i2_18` target and output count 2. The latter is ingredient-only variation,
not a different projected target/quantity choice.

Mean full-vocabulary conditional entropy at sampled token prefixes is 0.03844
nats/token (initial calls 0.01070; later calls 0.03900). This is entropy after
temperature0.5, not sequence-level action entropy, and no entropy change from
learning has been measured. Four samples per initial state cannot characterize
rare choices.

## How much craft output specifies ingredients?

All 542 schema-valid craft calls re-encode to the exact saved non-special token
IDs; none are excluded. Their 21,460 emitted tokens include:

- 9,473 wholly inside the `ingredients` member: 44.14%.
- 1,084 touching an ingredient-member boundary: a conservative inclusive upper
  bound of 49.19% when added to the whole-member count.
- 8,735 wholly inside `action`, `target_item`, or `output_count`: 40.70%.
- The remainder is other boundary punctuation, outer syntax, and actual EOS.

For the 424 nonzero-credit craft calls, ingredients account for 45.75–50.64% of
17,343 emitted tokens. Among the 194 native-successful crafts, the corresponding
share is 36.34–42.14% of 6,697 tokens. These are token cost measurements, not an
argument that masking these gradients is unbiased. Raw ingredients affect native
execution; binder eligibility must be established from its actual public-history
receipts before treating them as overwritten.

## Decision

Keep the already accepted single signed update: this batch has meaningful reward
variation. Do not repeatedly resample the all-success familiar groups by default;
the prepared fresh TRAIN curriculum remains the stronger next collection.
Compare binder's mixed groups, credit exposure, and behavior before deciding
whether assistance removes or improves the reward signal. The proposed
[same-batch gradient diagnostic](MASK-DIAGNOSTIC.md) is optional and CPU design only.

The external report pins the collection PLAN, completed owner terminal, native
audit and summary, every generation sidecar, and analysis/loss sources. The
existing loader verifies all native receipt hashes and original collection source
pins. The skill-guided checks added exact-prefix and token-boundary tests; no
collector or trainer was modified. This mechanism finding belongs in supporting
research evidence, not as an advisor-slide claim of RL improvement.
