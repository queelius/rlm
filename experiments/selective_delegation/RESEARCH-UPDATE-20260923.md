---
date: 2026-09-23
status: exploratory
cutoff_utc: "07:40"
questions: [learnable-demonstrations, rl-update-size, public-recipe-memory, recipe-world-transfer]
---

# The teaching result repeats; RL has not yet earned a strong claim

## Addendum, 09:23 UTC: larger updates did not solve the RL problem

The five-times-larger update solves15/32 versus16/32 with the smaller update:
two paired improvements, three regressions, all outcomes known. The difference
interval is−18.75 to+12.5 percentage points. It uses more calls and produces
more rejected actions. Thus improving the saved training objective is not enough.

The next accepted diagnostic removes negative whole-attempt credit while keeping
the same saved batch and starting checkpoint. Even this is imperfect: winning
training attempts contain177 rejected actions, including71 immediate identical
action/error repetitions. See the [credit analysis](RL-CREDIT-ASSIGNMENT-20260923.md)
for measured changes, evidence and limits, and the
[decision map](DECISION-MAP-20260923.md) for how results change the plan.
The public-memory factorial is running first, unchanged. Queue019 retains all
previously queued quantity and changed-world controls after the credit diagnostic.

## Addendum, 07:58 UTC: the changed-recipe replication also repeats

The second training seed now completes the changed-world comparison:
**original teaching 1/16; public-discovery teaching 9/16**. There are eight paired
wins, no losses and eight ties. The difference is +50 percentage points, with an
eight-goal cluster-bootstrap interval of [25, 75] points. All 32 outcomes are
recorded and natively checked. The public model reaches its context limit on one
attempt; that is a recorded task failure, not a missing outcome.

The first training seed had given 3/16 versus 10/16 in this same world. Thus the
direction repeats across training randomness in both the original and changed
recipe worlds. The worlds and goals are still exposed, and the teaching-quantity
confound remains. The better second-seed model uses **more**, not fewer, resources:
433 versus 372 calls, and 895 versus 654 seconds elapsed. We should not claim a
consistent efficiency improvement alongside the success improvement.

Evidence: `analysis-textcraft-world43-seed2291-001.json`, SHA256
`33ee458e07d43a67675f462351afbed3e9804d963635bb83c0d9f8eac52ed590`.
The GPU handed off to the accepted learning-rate experiment after this native
analysis completed. The remainder below preserves its original 07:40 cutoff.

The strongest lead remains **how we teach the model to find information before
acting**. Repeating training with different randomness preserves the direction
of the result. An additional RL update completed, but its improvement is only
one extra solved attempt. We should not present that as established RL success.

## Completed results

Each row below compares the same 16 goals, with two attempts per goal. These are
32 attempts, **not 32 independent problems or worlds**. The panel has already
been inspected during exploration. Intervals resample goals, keeping their two
attempts together; they do not measure uncertainty across recipe worlds.

| Comparison | Earlier condition | Later condition | Paired wins / losses | Difference, 95% interval |
|---|---:|---:|---:|---:|
| First training seed: original vs public-discovery teaching | 1/32 | 15/32 | 14 / 0 | +43.8 points [25.0, 62.5] |
| Second training seed: same teaching comparison | 3/32 | 10/32 | 7 / 0 | +21.9 points [9.4, 37.5] |
| First RL update vs its starting model | 15/32 | 15/32 | 2 / 2 | 0 points [−12.5, 12.5] |
| Matched one-step supervised update vs starting model | 15/32 | 17/32 | 4 / 2 | +6.3 points [−6.3, 18.8] |
| Additional RL update vs first RL checkpoint | 15/32 | 16/32 | 1 / 0 | +3.1 points [0, 9.4] |

All outcomes in these completed paired comparisons are recorded and checked by
replaying the environment. Missing results were not counted as failures.

The public-discovery examples show the model how to request recipe information
that it can actually obtain at test time. The original teacher could plan using
information not yet visible to the student. However, these are **teaching
packages**, not a clean isolation of that explanation: their histories differ,
and one original training trajectory requests a different ingredient quantity.
The second seed reduces concern about a lucky training run but does not remove
these confounds. Absolute success also varies substantially between seeds.

The repaired RL continuation performed one genuine new optimizer update. Its
training job took 63.1 minutes, including collecting attempts and replaying their
token probabilities; that is not 63 minutes of optimizer updates. Evaluation
uses the same BF16 arithmetic for both checkpoints. The first-update SFT control
is not a matched control for the second RL update.

## Experiments chosen from this evidence

1. **Is the RL change simply too small?** Replay the exact saved on-policy batch
   from the same starting weights, optimizer state and random state, changing
   only the learning rate from 0.00002 to 0.0001. Compare the fixed endpoints on
   the same panel. This diagnoses update size; a positive result would still
   need replication and a matched supervised-training control.
2. **Does the teaching effect survive a known confound?** Correct the ingredient
   quantity in the original examples and retrain both seeds. Regenerate the
   whole affected trajectory: changing just the target action would leave 29
   later prompts describing the wrong inventory. This control does not isolate
   every remaining difference in the teaching packages.
3. **Can a small notebook help the model use what it has learned?** Cross two
   choices: keep all interaction history or only the last four entries; provide
   a recipe notebook or not. The notebook copies only facts returned by earlier
   public queries. It never reveals unseen recipes or invents a plan. Current
   inventory, goal, action limits and scoring stay fixed. This separates the
   effect of remembering recipes from the effect of shortening history.
4. **Does the result travel to different recipes?** Repeat the second seed in
   the existing changed world, then test both seeds in three further worlds
   fixed before seeing outcomes. Retain every predefined goal; do not replace
   hard examples. This tests recipe changes, not new task types or datasets.

The first changed-world seed already gave 3/16 versus 10/16. The second seed is
running at this cutoff. Later-world inputs passed 24/24 native feasibility
checks, but this is a check of the tasks, **not model success**. Notebook and
quantity-control results are not yet available.

## Publication direction and boundaries

The plausible paper is not “memory helps” or “RL works.” A stronger, testable
question is: **When learning to act with incomplete information, which parts of
an expert demonstration must be made observable and remembered by the learner?**
The teaching controls, recipe changes and notebook comparison can distinguish
several explanations rather than merely accumulate scores.

Related work already studies these themes. [LEAP](https://openreview.net/pdf?id=st7XqFgbAH)
addresses learning from privileged experts and the difficulty of imitating
actions from less informative observations; its full paper still needs a careful
comparison with our setup. [Active Context Compression](https://arxiv.org/abs/2601.07190)
and [SAM](https://arxiv.org/abs/2605.24468) study explicit agent memory and history
management. These are prior-art leads, not evidence that our proposed mechanism
is novel. Only their available abstracts/search descriptions were reviewed at
this cutoff.

The current TextCraft comparisons use a single acting model, without recursive
helpers. They can inform an RLM component, but **do not establish a benefit from
recursion, learned decomposition, or transfer to unrelated benchmarks**. Those
claims require later experiments.

## Evidence and resumption

External artifact root:
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/`.

- `analysis-textcraft-second-seed-001.json`, SHA256
  `5dd0b829fdd0e7d416c3f5955ef27696b2a4c8d0f85e3893b84461a399e16d02`.
- `analysis-textcraft-fp16-cp2-minus-cp1-002.json`, SHA256
  `b091d9809f20cebcc585d00871894e211bebe1599d7f5ef507a6e94109842ee4`.
- [One-step findings](TEXTCRAFT-ONE-STEP-FINDINGS.md),
  [quantity-control analysis](TEXTCRAFT-QUANTITY-MATCH-CONTROL.md),
  [memory experiment](TEXTCRAFT-MEMORY-READOUT-PLAN.md), and
  [learning-rate comparison](TEXTCRAFT-LR-REPLAY-PLAN.md).
- The research-store `SESSION_CHECKPOINT.md` and `RESEARCH_QUEUE.md` contain
  current process identities and accepted queue receipts. GitHub preserves
  source and these reports, not the external model checkpoints.

## Operating failure recorded

The previous queue ended September 22 at 20:07 UTC. Useful GPU work resumed
September 23 at 07:30 UTC: roughly **11 hours 23 minutes of avoidable idle time**.
Large per-job runtime caps did not provide equivalent actual queue coverage.
The remedy is executable follow-ups with automatic saved analyses and realistic
duration estimates, not filler work or an assurance that idle time cannot recur.

The user now authorizes using the remaining account allowance; the earlier 5%
reserve is superseded. At 07:40 UTC the account reported 11% remaining. A banked
reset is available to the user, but no reset has been assumed or requested.

Presentation decision: retain the earlier meeting deck's stated evidence cutoff.
These results strengthen the teaching lead and constrain the RL claim without
changing that main conclusion; this dated report is the current follow-up.
Promote the notebook and quantity-control comparisons together if they materially
change the explanation, rather than adding every exploratory run to the deck.
