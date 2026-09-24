---
date: 2026-09-24
evidence_cutoff_utc: "01:12"
status: exploratory
questions: [learnable-demonstrations, memory-interface, terminal-credit]
---

# Overnight results: teaching holds up; adding memory is not enough

The local queue continued while Codex was inactive. At this cutoff the GPU is
running the last memory condition for the RL-trained actor. The allocation ends
at approximately 03:59 UTC; jobs stop accepting work before the final ten minutes.
Queues 016 and 019 have finished 29 scientific jobs (training or evaluation),
with 55,531.7 seconds—15.4 hours—of combined job wall time, and 22 analysis jobs.
One scientific job and its initial analysis failed, as discussed below. These
times include loading and preflight; they are not measurements of GPU kernel time.

## Teaching advantage survives the quantity correction

We corrected the known quantity error in the original teaching examples and
retrained from the same starting model under each of the two training seeds.
Public-discovery teaching still does better on the original evaluation panel.

| Training repeat | Corrected original teaching | Public-discovery teaching | Paired improvements / regressions |
|---|---:|---:|---:|
| First | 5/32 | 15/32 | 11 / 1 |
| Second | 3/32 | 10/32 | 7 / 0 |

The differences are 31.25 and 21.875 percentage points. Task-cluster bootstrap
intervals are [15.625, 50] and [9.375, 37.5] points. All outcomes are known. The
quantity error therefore does not explain the entire teaching advantage. This
does not isolate a single cause: the teaching packages still differ in actions,
information acquisition and the observations supplied during training.

## The teaching direction also repeats in three additional recipe worlds

Each cell is original-teaching successes → public-discovery successes, out of 16
attempts (eight goals, two attempts each).

| Changed recipe world | First training repeat | Second training repeat |
|---|---:|---:|
| 44 | 4 → 7 | 3 → 6 |
| 45 | 1 → 8 | 1 → 8 |
| 46 | 3 → 8 | 1 → 10 |

All 96 pairs are known. These comparisons still use the **uncorrected** original
teacher, so they do not yet cross the quantity correction with changed recipes.
They also share a synthetic recipe generator and task family; this is not
transfer to a different domain. Do not pool the attempts as independent worlds.
The next targeted control is both teaching packages on a previously unexamined
world, using the quantity-corrected original teacher.

## A notebook is not a reliable improvement without further adaptation

We independently varied whether the model sees all prior interactions or only
the latest four, and whether it also receives a notebook of recipes it actually
looked up. No hidden recipe information was added.

| What the model sees | First trained actor | Second trained actor |
|---|---:|---:|
| Full interaction history | 15/32 | 15/32 |
| Full history plus recipe notebook | 14/32 | 6/32 |
| Latest four interactions only | 10/32 | 10/32 |
| Latest four plus recipe notebook | 14/32 | 8/32 |

All 256 outcomes are known. Shortening history reduces success in both actors.
The notebook recovers four attempts with short history for the first actor, but
loses two for the second. Adding it to full history is especially harmful for
the second actor: nine paired losses, no wins. That difference's cluster interval
is −43.75 to −12.5 percentage points. These are exploratory comparisons on an
already exposed panel, not a general claim that memory is harmful.

The positive notebook-by-history interaction in the second actor means the
notebook is **less harmful** with short history—not that it helps. This distinction
matters when interpreting the machine-readable interaction statistic.

Short-history conditions use 1,971–2,311 calls, versus 942–1,316 with full
history. Calls are not elapsed time or cost by themselves: prompts are shorter,
and full-history attempts can hit context limits earlier. All four arms share
the same memory-description instruction. That instruction was absent from the
historical readouts, so use the newly matched full-history arm as the control.

Decision: finish the already fixed RL-actor comparison. Do not expand the simple
notebook intervention as though it were an established improvement. A subsequent
study could test whether training on the interface changes its usefulness, or
whether exact tool execution matters more than repeating available facts.

## Positive-only RL: training completed; evaluation was interrupted

The positive-only update completed normally. Its evaluation reached the one-hour
cap with 28 completed attempts, one interrupted attempt and three unstarted slots.
The initial analysis then failed on an identity check in the interrupted-record
path. The additive recovered audit finds 15 successes among the 28 complete
attempts; three paired improvements and three regressions against the signed
update. Full-panel success lies between 15/32 and 19/32. These are missing-outcome
bounds, not confidence intervals. No complete-panel confidence interval is valid
yet. Do not treat missing outcomes as losses or claim RL improvement.

The proposed follow-up covers only the four unavailable slots in a separate,
bounded run, preserving the original records and their compute costs. It must
reuse the fixed endpoint, prompts, per-slot seeds and per-episode limits, and
must not rerun already completed successes or failures.

## What looks most promising now?

The strongest repeated signal is still teaching the model how to obtain and use
information it can actually observe. The quantity control strengthens that lead.
Memory alone is not the solution, and meaningful TextCraft RL improvement remains
unestablished. The [earlier action audit](RL-CREDIT-ASSIGNMENT-20260923.md) motivates
separating recipe discovery, tool-argument construction and inventory planning.
All current actors here are flat; these findings do not demonstrate a recursion
advantage. Prior work such as LEAP already studies learnable demonstrations, so
a publication needs a more specific mechanism and stronger controls.

Evidence: [compact reports and immutable source hashes](overnight-results-20260924.json).
Raw reports and weights remain in the external research store, not GitHub.
Deck decision: consolidate these controls in research notes first; do not silently
change the historical advisor deck's cutoff or present a positive memory/RL claim.
