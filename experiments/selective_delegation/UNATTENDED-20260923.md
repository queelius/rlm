---
date: 2026-09-23
status: accepted-running-and-queued
updated_utc: "09:23"
allocation: 5879
---

# September 23 unattended research handoff

The scripts below run locally without further Codex generation. They save native
attempts, checkpoints and analyses. They are a bounded experimental program, not
an autonomous agent that invents new scientific questions after each result.

Artifact root (`R` below):
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.

The larger RL update completed training (23.7 minutes) and evaluation: **15/32**
versus **16/32** for the smaller update, all outcomes known. The
[credit-assignment analysis](RL-CREDIT-ASSIGNMENT-20260923.md)
records measured parameter and token-probability changes, and the
[repeated-state audit](TEXTCRAFT-REPEATED-STATE-CREDIT-FEASIBILITY.md) assesses a
possible follow-up without spending GPU time. A positive-only credit comparison
is now **accepted and waiting** in queue019, after the running queue016 memory
factorial. Queue019 then retains every job from017 and018. Only the waiting
017/018 supervisors were stopped, with exact identities checked and recorded in
`R/REPRIORITIZE-017-018-TO-019.json`; do not relaunch them.

| Order | Question | Accepted receipt under R |
|---|---|---|
| Completed | Does the teaching result repeat in changed recipes with the second training seed? Yes: 1/16 versus 9/16; see the dated analysis. | `TEXTCRAFT-WORLD43-SEED2291-QUEUE-002.json` |
| 015, completed | A larger update did not establish improvement: 15/32 versus16/32. | `INDEPENDENT-TRAINING-QUEUE-015-AMENDED.json` |
| 016, running | Does a public recipe notebook help, separately from shortening history? | `INDEPENDENT-TRAINING-QUEUE-016.json` |
| 019, waiting | Does omitting negative terminal credit help? Then run the retained quantity controls, actor-memory replications and changed-world comparisons. | `INDEPENDENT-TRAINING-QUEUE-019.json` |

Never launch the original unamended 015 or 017 receipts. Their replacements were
accepted before execution. Earlier proposal documents intentionally retain their
pre-acceptance status; the receipts above establish what is actually queued.
The descriptions of017/018 below document retained jobs, not live supervisors.
New gate019: PID566177, creation1790155303.47, session78032; receipt SHA
`79907103988639046ce23d526f8d38fc791f54461c942ae342218914debf7bc6`.
Invocation SHA `dc4a5b399aca97a1ee2140898845e303e4d8ae411d798ec03cb73821076b7cec`.
It contains63 jobs and respects the lease minus600 seconds. These are caps,
not a guarantee that the whole sequence will fit.

Queue 016 contains the complete four-condition notebook comparison for the
first public-discovery model. Queue 017 orders: corrected-original teaching at
both seeds; both teacher seeds in recipe world 44; the second public-discovery
model's four notebook conditions; worlds 45 and 46; then the RL checkpoint's four
notebook conditions. Analyses follow each complete comparison. Failed independent
jobs do not prevent subsequent jobs from attempting their own preflight, but an
unreleased scientific owner stops the queue rather than risking overlapping jobs.

The GPU lease ends at epoch `1790222352`; each supervisor clamps execution to
600 seconds before that. It declines to start a job whose full cap would not fit.
Thus not every accepted arm is guaranteed to run. A missing or interrupted result
is unknown, not an incorrect model answer. Source snapshots are immutable.

Runtime estimates are uncertain. Worlds 44–46 project roughly 4–6 hours from
completed similar evaluations; all notebook conditions may require several more
hours, especially if shorter histories permit longer action sequences. Maximum
caps must not be reported as actual GPU use. Queue 018 adds worlds 47–49 as
lower-priority robustness checks, projected at roughly another 4–6 hours if
similar to earlier worlds. They were fixed before inspecting their model results.
This provides more prospective evidence across recipe assignments, not new
benchmark families. The queue may reach the allocation cutoff before finishing.

## What to check on resumption

1. Inspect `STATUS.json` in the active scientific output. A rising `returned`
   count and recent timestamp show actual responses; GPU memory alone does not.
2. Inspect `independent-training-queue-015`, `-016`, `-017`, and `-018`. The immutable
   `GATE-INVOCATION.json` records exact PID, creation time and predecessor. Each
   completed job has a log and JSON exit record; `GATE-RESULT.json` marks the
   supervisor's end. Authenticate a PID's creation time before acting on it.
3. Read completed native analyses before raw aggregate summaries. Useful new
   filenames include `analysis-textcraft-lr-replay-001.json`,
   `analysis-textcraft-memory-public1-factorial-001.json`,
   `analysis-textcraft-quantity-matched-seed2026092208-001.json`, and
   `analysis-textcraft-world44-seed2026092208-001.json`.
4. Compare paired wins/losses, missing attempts and cost, not just headline
   success rates. The same task's repeated attempts are correlated. Worlds share
   goal names, vocabulary and generation rules; this is not cross-dataset transfer.

## Decisions the results should change

- A larger RL update that changes behavior but not success weakens the
  “updates are simply too small” explanation. A gain needs replication and an
  equally large supervised-update control before claiming an RL-specific benefit.
- If correcting the original quantity removes the teaching gap, revise the
  causal story rather than attributing it to information discovery.
- If a notebook helps mainly with short history, investigate explicit memory
  as a way to reduce context cost. If it helps neither, do not scale it blindly.
- If the teaching advantage reverses across recipe worlds, investigate what
  properties of a recipe make the demonstrations transferable.

See [the dated findings](RESEARCH-UPDATE-20260923.md) for completed evidence.
Current process pointers also live in the research store's `SESSION_CHECKPOINT.md`.
Source and reports are pushed to `research/selective-delegation-20260921`;
external model weights and native traces are **not** backed up by GitHub.

The last gate uses the additive `source-queue-handoff-generic-004`: its sole
runtime change permits up to a 24-hour predecessor wait instead of 12 hours.
The GPU lease cutoff and all owner checks still apply. This avoids abandoning
the lower-priority queue merely because the preceding useful experiments take
more than 12 hours. Earlier supervisors retain their original sealed source.
