# Exposed Hotpot explorer: verified type readout

The comparison-type base/direct advantage over SFT is entirely mediated by
returned helper-protocol failures in these saved runs. It is not evidence that
SFT made valid comparison reasoning worse. This replaces the earlier provisional
interpretation with an executed native-receipt audit.

All 32 exposed explorer parents join by exact original source ID **and exact
question** to the pinned HF mirror: 23 bridge and 9 comparison. Both repeats and
all four policies remain: 256 planned/recorded slots, zero unavailable outcomes.
Official HotpotQA EM/F1, including its yes/no/noanswer rule, were recomputed from
209 actual final receipts; upstream protocol failures retain zero on the planned
denominator. Overall values reproduce the existing official reports exactly.

| Policy | Bridge EM; F1 | Comparison EM; F1 | Valid bridge / comparison |
|---|---|---|---|
| Planner base | 20/46 (43.5%); .4681 | 10/18 (55.6%); .6508 | 33 / 14 |
| Planner SFT48 | 18/46 (39.1%); .4860 | 5/18 (27.8%); .3730 | 38 / 11 |
| Planner RL4 | 19/46 (41.3%); .4990 | 5/18 (27.8%); .3730 | 39 / 10 |
| Direct base | 26/46 (56.5%); .6572 | 10/18 (55.6%); .6508 | 43 / 16 |

## What the matched changes show

All differences below are the named policy minus SFT. Intervals use 20,000
paired-parent bootstrap draws, seed 2026092122; the two repeats stay together.

| Type / contrast | EM difference, 95% interval | Wins / losses | Mechanism of changed rows |
|---|---|---|---|
| Comparison: base − SFT | +27.78pp [0, +55.56] | 5 / 0 | All five wins protocol-mediated |
| Comparison: direct − SFT | +27.78pp [0, +55.56] | 5 / 0 | Same five protocol-mediated wins |
| Comparison: RL − SFT | 0pp [−16.67, +16.67] | 1 / 1 | One both-valid win; one protocol loss |
| Bridge: base − SFT | +4.35pp [−6.52, +17.39] | 5 / 3 | Wins: two both-valid, three protocol; losses: three protocol |
| Bridge: direct − SFT | +17.39pp [+2.17, +32.61] | 9 / 1 | Wins: five both-valid, four protocol; loss: both-valid |
| Bridge: RL − SFT | +2.17pp [−4.35, +8.75] | 3 / 2 | Wins: two both-valid, one protocol; losses: one of each |

The five comparison wins occur on only three parents:
`75f9f469ce2e50118f41a6cc` (both repeats),
`c4ac2a2bc9a8308db68c9dc0` (repeat 1), and
`f1f065b17449068a642ae17c` (both repeats). Every SFT row has
`invalid_helper`; its base and direct counterparts have valid correct finals.
There are no both-valid base/direct wins over SFT in this stratum. The RL
comparison net zero is not identical behavior: one valid-answer improvement
is offset by another helper-protocol failure.

Bridge direct − SFT also improves F1 by +17.12pp [+2.23, +33.17]. Its five
both-valid wins and one both-valid loss show that the broader direct-policy
advantage is not entirely a protocol artifact. This remains a different-prompt,
different-cost policy comparison, not a causal decomposition intervention.

## Decision and limits

These results support checking the already-queued fixed-helper transfer before
motivating a new comparison-specific planner. They do not establish that a
helper adapter will fix these particular failures. Official Hotpot type is not
the generated plan's dependency topology, and nine comparison parents are a
small, already-exposed stratum. Bootstrap intervals are exploratory and
unadjusted; apparent type differences are not a tested interaction or a causal
effect of question type. No new primary checkpoint or panel is selected here.

## Reproduction and immutable evidence

Source: `analyze_hotpot_types.py`; fixture: `test_hotpot_types.py`. Executed from
the selective-delegation worktree with
`/project/alex_phd/envs/prime-rl-5990b1b/bin/python`:

```sh
python -m pytest experiments/selective_delegation/test_hotpot_types.py -q
python experiments/selective_delegation/analyze_hotpot_types.py \
  --root /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921 \
  --mirror /project/alex_phd/research-cache/datasets/hotpotqa-official-dev-20260921/distractor-validation-00000-of-00001.parquet \
  --report /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/analysis-hotpot-types-001.json
```

Three focused tests passed. The second command completed and created immutable
JSON and Markdown; reruns must choose a new report path. JSON SHA256:
`27ba8d341a6a2ed56f1f42d5d4eb7a8be829f656a9ad0dec4e2f1895264bce09`.
The report records 482 source/consumed-file hashes, including 256 episode and
209 final receipts, the official reports, cases, scorer, mirror acquisition and
Parquet. It includes all per-attempt outcomes and paired changed-row identities.
No model weights were audited or GPU calls made.

Mirror revision `1908d6afbbead072334abe2965f91bd2709910ab`, license CC-BY-SA-4.0,
retrieved 2026-09-21; Parquet SHA256
`c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6`.
This is the pinned HF mirror, not a claim of byte identity with official JSON.
Native sources are `transfer-hotpot-sft-001`, `transfer-hotpot-rl-001`, and
`transfer-hotpot-direct-001`, checked against the two existing official regrades.
