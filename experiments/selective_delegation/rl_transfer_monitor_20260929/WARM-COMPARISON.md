---
date: 2026-09-29
snapshot_utc: "2026-09-29T02:10:36Z"
status: complete_warm_controls_rl_transfer_pending
question_id: execution_assistance_on_fresh_goals
claim_level: exploratory_eight_root_diagnostic
---

# On the new goals, execution assistance helps only a little

The same SFT-trained starting model finished **4/16 attempts without ingredient
assistance and 5/16 with it**. The code fills ingredient arguments from recipes
already looked up; the model still chooses what to make and how much. This is
a harness comparison, not an RL improvement. The RL-trained actors' four
crossed readouts remain incomplete in this snapshot.

| Same eight goals, two attempt seeds each | Model supplies ingredients | Code fills known ingredients |
|---|---:|---:|
| Successful attempts | 4/16 | 5/16 |
| Model calls | 778 | 673 |
| Generated tokens | 23,516 | 21,759 |
| Rejected native actions | 348 | 241 |
| Invalid response structures | 1 | 3 |
| Explicit finishes without completing the goal | 10 | 10 |
| Context-limit endings | 2 | 1 |

Both controls are complete and independently natively audited, with zero
transport failures. The paired comparison has one improvement, no regression,
and fifteen unchanged outcomes. The improvement is one repeat of TRAIN1796;
all other goal pairs are unchanged. The task-cluster bootstrap interval for the
6.25-percentage-point difference is 0–18.75 points. With only eight root clusters
and one changed outcome, this is weak evidence of a success advantage.

Assistance reduces calls by 13.5%, generated tokens by 7.5%, and summed model-call
time by 9.7% (1,963 to 1,773 seconds). These are aggregate attempt costs, not
training time or an estimate of whole-allocation utilization. Many rejected
actions and unsuccessful finishes remain. Fewer argument mistakes do not by
themselves demonstrate better planning or useful decomposition.

## Decision

Continue the four fixed RL-transfer readouts, comparing each trained actor with
the matching warm baseline above. Both columns have room to improve. Do not
compare these scores directly with the earlier familiar-goal 9/16 and 14/16 as
if they were paired changes: the tasks differ. Do not select new training labels,
favorable checkpoints, or replacement tasks using these B outcomes.

The already accepted fresh-A collections will reveal whether the new training
goals have useful reward variation. Two identical initial responses per B goal
do not establish a lack of learning signal over complete A trajectories.
The proposed recovery-teaching study remains conditional on those A traces.

Keep this small starting-point comparison in supporting evidence. It does not
change the fixed September29 slide story or establish an RL-transfer result.
The [earlier raw-only snapshot](README.md) retains its original cutoff and
describes the repeated errors in more detail.

## Evidence

[WARM-COMPARISON.json](WARM-COMPARISON.json) preserves the unchanged six-cell
analyzer's output, including explicit unknowns for the four pending RL cells.
It adds source/report hashes, the snapshot time and this editorial decision.
The external original is `R/analysis-rl-transfer-20260929-001/WARM-001.json`,
SHA256 `142d88eaefbe8591f7e3c9c2dc0515b1331c64524d0196692da0e28162dd53dd`.
The analyzer validates actual actors, native audits, task/seed matching, world,
sampling and budgets. No new GPU work or native replay was run for this report.
`R=/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.
