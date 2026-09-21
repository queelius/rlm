---
status: completed_exploratory
evidence_cutoff_utc: 2026-09-21T17:08:55Z
question: Does short-term goal management help once commands are executable?
decision: test_local_deliberation_before_claiming_a_hierarchy_benefit
---

# Short-term goals help on three small placement games

With numbered action choices, the manager-worker policy solves 6 of 16 attempts;
the flat policy solves 1 of 16. These are **eight seen-development games, each
sampled twice**, not 16 independent tasks. The five paired gains occur on three
placement games. There are no paired losses, but neither policy solves a heating
or cleaning game. This is a useful exploratory signal, not broad task mastery.

Both use the same frozen 4B model, public observations, admissible commands,
50-action limit and 2,048-generated-token allowance. The manager writes a short
goal every four executed actions; its worker chooses the next numbered command.
The flat policy chooses a numbered command directly. Neither has been trained
on these games in this study.

| Measure | Flat action choice | Short-term goal manager |
| --- | ---: | ---: |
| Completed tasks / 16 attempts | 1 | 6 |
| Invalid outputs | 0 | 0 |
| Attempts reaching 50-action limit | 15 | 10 |
| Executed actions | 781 | 551 |
| Model calls, including manager | 781 | 697 |
| Total input + output tokens | 1,373,215 | 1,252,404 |
| Summed model-call seconds | 340.0 | 356.5 |

The manager uses fewer calls and about 8.8% fewer tokens because it finishes some
tasks sooner. It does **not** use less measured model-call time; those seconds
increase slightly. The complete collection takes 923.3 wall-clock seconds,
including model loading and environment work. Tokens are not a substitute for
measured time or exact GPU computation.

## A concrete example

The task is to put an egg in the microwave. The flat policy moves a bowl,
collects lettuce, then repeatedly travels between locations until its action
limit. The manager-worker policy searches, finds an egg in the garbage can,
takes it, and puts it in the microwave in 11 actions. The two seeds reproduce
that successful action sequence. This illustrates task-directed search rather
than a repaired JSON response.

The tissuebox-to-toilet task shows a similar contrast: a successful nine-action
search-and-place sequence versus repeated trips among bathroom fixtures. Other
games still produce long valid but ineffective loops. Numbered actions solve
the command-admissibility problem, not all planning problems.

## What the comparison establishes—and what it does not

The paired improvement is 31.25 percentage points; an exploratory bootstrap
over eight games, keeping both seeds together, gives a 95% interval of 6.25 to
62.50 points. The panel is tiny, exposed, and not proven scene-independent.
The interval does not justify a general ALFWorld superiority claim.

The earlier string-command screen was dominated by inadmissible actions. This
new package includes numbered choices and rejection feedback, so comparing the
two screens does not isolate a single interface change. In this completed run,
no response was rejected: the new feedback path was available but never used.

Within the new interface, the manager contrast includes extra goal-generation
calls and a persistent short-term goal. It does not isolate hierarchy from
ordinary deliberation. The next proposed control lets a single flat agent write
a brief reason before each action, under the same episode and generation caps.
A subsequent fresh-game comparison is required before promoting the effect.

## Evidence

External root:
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.
Source `source-026-alfworld-closed-loop`, output `alfworld-closed-loop-001`, and
independently reconstructed report `analysis-alfworld-closed-loop-001.json`.
All 32 episodes are observed, with 1,478 returned calls, no inference errors,
no unresolved starts and no unlinked calls. Analyzer source is preserved under
`analysis-source-alfworld-002`. Bootstrap: 20,000 draws, seed 2026092180.
