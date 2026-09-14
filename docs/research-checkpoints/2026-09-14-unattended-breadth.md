# Unattended dataset and harness screen

On September 14 we started a finite local campaign to discover where small
helper calls help or hurt across different tasks. It runs without Codex or
external model APIs, under the current single-A100 allocation.

The frozen bundle contains 2,277 cases: 512 answerable MuSiQue questions, 512
FinQA questions, 512 BoolQ questions, 238 disjoint AG News counting groups, and
503 LongBench v2 benchmark questions. Inputs beyond the common 24,000-token
full-input bound are excluded from every arm, not truncated. LongBench results
will therefore describe only an admissible-length subset.

The campaign compares direct answers, concise helper summaries, and
evidence-preserving helper returns. FinQA also gets a restricted arithmetic
expression baseline. Two-way and four-way divisions are separate conditions;
every helper and final call is charged. These are fixed-harness ablations, not
learned recursive planning or a compute-matched comparison. Released 4B and 8B
models are queued, without adapters. No new SFT or RL update is in this first
breadth campaign; the purpose is to identify informative training directions.

Initial 16-case blocks are followed by another 64 cases. Further extensions
require earlier results to show both successes and failures. Both fixed arms and failures
remain reportable. This adaptive screening is exploratory, not confirmation.
The four-way block revisits earlier cases with new seeds and is labeled as such.

A successful pilot returned 54/54 model calls across 32 scored or excluded
episodes. Two earlier engineering failures are retained: a Unicode JSONL
reader issue, followed by an installed-tokenizer return-type mismatch. Neither
produced scientific results. The actual native HTTP path now returns and scores
real answers. This establishes operation, not scientific improvement.

## Location and operation

External campaign root:
`/project/alex_phd/runs/rlm-research-r4/sidecars/unattended-breadth-20260914/`

- `LAUNCH.json`: detached owner PID, command and deadline.
- `outputs/campaign-001/STATUS.json`: current phase, returns, errors and scores.
- `outputs/campaign-001/SUMMARY.json`: completed-stage summaries.
- `outputs/campaign-001/models/*/calls/`: raw native requests and responses.
- `outputs/campaign-001/models/*/episodes/`: conditions, scores and costs.
- `README.md`: interpretation limits, stop and resume commands.

The detached supervisor started as PID 1366840. Its deadline is
**September 15, 2026, 17:20 UTC**, ten minutes before the allocation end recorded
in the environment. It may finish earlier if the finite useful queue is exhausted.
It can outlive this Codex conversation, but not cancellation of the allocation.
Only the owner may use this reserved GPU while the campaign is live.

No final campaign results are claimed in this launch note. The September 13
[findings report](2026-09-13-decision-interfaces-and-record-grouping.md) remains
the completed scientific checkpoint. Public source snapshots are not backups
of external raw runs or model weights.
