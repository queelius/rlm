---
date: 2026-09-24
evidence_cutoff_utc: "16:15"
status: exploratory
question: Does the teaching advantage survive both quantity correction and changed recipes?
---

# Correcting the demonstrations narrows the advantage on changed recipes

The first training repeat now has a completed three-way comparison on world47:

| Teaching package | Successes / attempts |
|---|---:|
| Original | 2/16 |
| Quantity-corrected original | 4/16 |
| Public-discovery | 6/16 |

All outcomes are known. Against corrected teaching, discovery wins three paired
attempts and loses one. Its advantage is12.5 percentage points, with a task-group
bootstrap interval from−12.5 to+43.75 points. This small panel does not establish
a clear advantage. The correction closes half of the observed gap in this
setting; it is not a general decomposition of the teaching effect.

The second training seed's uncorrected comparison is1/16 versus8/16. Its corrected
checkpoint already exists; queue023 now evaluates that checkpoint on the same
world and reuses the completed public comparator. No new training or duplicate
public rollout is needed. Do not examine interim successes to select the endpoint.

## What might explain the difference?

Corrected teaching used491 calls and produced43 invalid actions; discovery used
621 calls and produced312 invalid actions. Their inference service times were
about846 and1562 seconds. These are whole-arm totals, not independent examples
or matched-cost estimates. More completed tasks despite more errors motivates
examining routes, persistence and recovery. It does not show that mistakes help.

Both teaching packages include public recipe lookups. The original teacher
uses underlying recipe knowledge to determine a route; the public-discovery
teacher determines prerequisites through observable lookups. Never describe
the comparison as looking up recipes versus not looking them up.

## Decision

1. Finish the second-seed corrected control.
2. Inspect all four discordant paired attempts, with exact native trace pointers.
3. Pilot automatic ingredient-argument binding from recipes already observed.
   Keep model-selected targets and quantities unchanged. This tests execution
   assistance, not learned planning, a new action schema or recursive decomposition.

The five-slide deck now includes the completed control and its uncertainty.
The same-world result remains stronger than the changed-world evidence.

## Native evidence

Root: `/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.

- `analysis-textcraft-world47-quantity-corrected-seed2208-004.json`
- `analysis-textcraft-world47-seed2026092208-001.json`
- `analysis-textcraft-world47-seed2026092291-001.json`

Corrected collection had a1800-second cap versus2700 for public; both finished
all episodes before the total cap. Per-episode limits are matched. These are
eight exposed goals with two correlated attempts each, within one game generator.
They are not a new independent domain or a final untouched test set.
