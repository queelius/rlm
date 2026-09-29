# Fresh-B warm baseline: fixed snapshot

Cutoff: 2026-09-29T01:46:38.130385+00:00. Raw is complete; binder native audit was still absent at the final natural check (01:47 UTC) and remains pending here. This is a snapshot, not a watcher.

The warm public-discovery SFT checkpoint 23 succeeds on **4/16 attempts**, covering two of eight roots. This is the pre-RL baseline on fixed TRAIN-only group-B diagnostic goals, not an RL result. The familiar warm baseline is 9/16 with the same actor but disjoint goals and different difficulty; subtracting these scores is not a paired estimate of degradation.

## Fixed goals and outcomes

Each root has two rollout seeds (202609280900/901), world 42. Depth is the preselection dependency-chain metadata, not the item's declared tier; these differ for TRAIN1796 and TRAIN1847.

| TRAIN ID | Actual depth | Success | Calls | Termination |
|---|---:|---:|---:|---|
| 1796 | 3 | 0/2 | 128 | 2 finish |
| 256 | 3 | 2/2 | 25 | 2 finish |
| 672 | 4 | 0/2 | 105 | 1 finish, 1 context cap |
| 1273 | 4 | 0/2 | 74 | 2 finish |
| 1847 | 4 | 2/2 | 39 | 2 finish |
| 38 | 5 | 0/2 | 111 | 2 finish |
| 201 | 5 | 0/2 | 152 | 2 finish |
| 964 | 5 | 0/2 | 144 | 1 finish, 1 context cap |

Actual-depth totals are 2/4 at depth 3, 2/6 at depth 4, and 0/6 at depth 5. Both repeats agree for every root, so there are eight correlated root clusters, not sixteen independent problems. The fixed goals were selected before actor outcomes; these results must not generate B training labels or outcome-based replacements.

## Failures and cost

The 778 calls used 23,516 output tokens and 2,579,699 prompt tokens, with zero transport errors and no unknown token counts. Summed audited service time is 1,963 seconds (32.7 minutes), not a wall-clock allocation estimate.

Native replay reports 348 action errors plus one schema error: 116 extra-ingredient, 84 missing-ingredient, 128 insufficient-stock, and 20 other native errors. Fourteen episodes explicitly finished; ten did so without the goal being met. Two reached the input-context cap. None reached the 96-call or 8,192-output-token ceiling; context growth still mattered.

A strict repeated-error loop means consecutive identical canonical JSON actions with identical public error feedback. There are 45 blocks in 10/16 episodes, comprising 118 calls, including **73 repeats after the first failure (9.4% of all calls)**. The longest block is six calls. Repeated stock errors account for 26 repeats; extra/missing ingredient errors account for 47. This excludes nonconsecutive cycling and is descriptive, not a forecast of repair benefit.

For example, TRAIN1796 repeat 0 crafts `o5_i1` six times with the same extra `raw_o3` argument and identical rejection. TRAIN38 repeat 0 attempts the same craft six times despite feedback that three ingredients each have one unit but require two. Literal public action/feedback records are retained in the JSON.

All four audited post-goal calls are ordinary terminal `finish` actions; there is no observed post-goal crafting waste in this baseline.

## What matters next

1. Complete the paired warm raw/binder comparison on these same B roots and seeds. That estimates execution assistance with the actor fixed, not learning.
2. Compare the queued familiar-RL actors crossed with raw/binder execution against each execution column's warm B baseline. This tests transfer and interface co-adaptation without confusing the familiar 9/16 with B performance.
3. Compare every fixed fresh-A RL endpoint with its own warm B baseline and the fixed extra-SFT reference, within interface. Extra SFT matches one optimizer step/LR/start, not data, token dose or FLOPs; later RL endpoints have more updates. Do not choose a checkpoint using B.

This is neither an aggregate ceiling nor an all-zero floor: twelve failures leave headroom and two roots already succeed. Depth 5 is locally at floor and could make small improvements hard to detect. Prior CPU feasibility does not prove solvability under this actor's context constraints. Group-B outcomes do not reveal whether group-A training supplies useful reward variation.

## Provenance and verification

Source run: `R/textcraft-fresh-rl-20260928-002/raw/readout-warm`, where `R=/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.

[BASELINE.json](BASELINE.json) records all sixteen task/seed outcomes, exact actor identity, sampling/caps, error definitions and 56 small receipt hashes. The actual PLAN has sixteen unique jobs; all sixteen native-audit records are present, saved episode/node hashes match that audit, and all sixteen first-call PLAN hashes match the actual PLAN. Task manifests and the public cp23 actor COMMIT also match. PLAN's `update=1` is bookkeeping, not RL dose.

This analysis reused the completed native audit and inspected saved public histories. No new native replay, model-weight rehash, GPU work, process changes or Git operations were performed. Structural metadata is used only for descriptive stratification, not actor-visible planning or failure labels. Binder has no substituted zeros: its success and observed counts remain unknown.
