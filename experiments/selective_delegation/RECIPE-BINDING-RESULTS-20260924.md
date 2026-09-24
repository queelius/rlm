---
date: 2026-09-24
evidence_cutoff_utc: "17:55"
status: exploratory_replicated_within_task_family
question: Does executing known recipe details in code improve task completion?
---

# A small execution change improved completion without further training

The model still proposes a target item, output quantity and ingredients. Before
executing a crafting action, code replaces the ingredient dictionary using one
recipe the model previously obtained through a public lookup. It preserves the
target and output amount. Unobserved or ambiguous recipes and quantities that
do not divide into whole recipe batches are left unchanged. The native game
still checks inventory, recipe validity and final task success.

This is automatic execution assistance, not a new learned planning policy.
The model-visible history contains the executed action and actual feedback.
Raw requested actions and assistance metadata are saved separately. A native
replayer reconstructs assistance from raw replies and prior observations and
checks subsequent prompts, inventories and scores.

| Comparison | Unchanged execution | Recipe details filled by code | Paired wins / losses | Difference interval, percentage points |
|---|---:|---:|---:|---:|
| World47, first trained model | 6/16 | 12/16 | 6 / 0 | 12.5 to 75 |
| World47, second trained model | 8/16 | 10/16 | 2 / 0 | 0 to 37.5 |
| World48, first trained model | 9/16 | 14/16 | 5 / 0 | 6.25 to 56.25 |

All 48 paired outcomes are known. Do not pool them as independent tasks: these
are eight goal identities, two attempts each, across related recipe worlds and
trained models. Intervals resample task groups within each comparison. The
second-model effect is smaller and its interval touches zero. No paired losses
were observed; this does not guarantee absence of regressions elsewhere.

## Cost and error counts

| Comparison | Calls before / after | Native action errors before / after | Inference service seconds before / after |
|---|---:|---:|---:|
| World47, first model | 621 / 469 | 312 / 123 | 1562 / 1112 |
| World47, second model | 428 / 392 | 84 / 60 | 832 / 842 |
| World48, first model | 469 / 307 | 201 / 73 | 1292 / 665 |

Calls and native action errors fell in all three comparisons. Wall-clock savings
were not uniform: the second model took slightly more inference time despite
fewer calls. Native action errors exclude malformed action-schema responses.
Service time excludes loading and CPU analysis; it is not GPU kernel utilization.

## What this supports

This is the clearest recent result from changing the surrounding execution
system rather than the model weights. Exact command construction can be a
meaningful bottleneck. It is a useful preliminary follow-up to the demonstration
study, and a practical direction for changing the RLM environment.

It does not show learned decomposition, a recursion advantage, or transfer to a
different domain. Execution changes later observations, so this is an end-to-end
interface intervention, not a counterfactual count of individually repaired
errors. General planning/execution separation is established prior art; novelty
requires a more specific claim and wider controls.

## Decisions and limitations

The pilot was motivated by inspected world47 failures. The second model and
world48 follow-ups were queued before examining pilot success totals, but all
goals belong to an already exposed research protocol. The unmodified baselines
were reused without rerolling. Total collection caps differed (30 versus45
minutes), but all episodes completed; per-episode limits were held fixed.

Queue026 is completing the second-model comparison on world48, including its
previously prepared original/public baseline and native audit. This closes the
small two-model-by-two-setting comparison rather than selecting only favorable
cells. Do not expand to a new RL sweep before reviewing that result.

For the five-slide presentation, slide5 now reports these completed results,
with the model's unchanged weights and the limited task family visible. It
replaces the earlier proposed-experiment slide. The fourth cell remains pending.

The adjacent numerical receipt preserves exact source paths, hashes, paired
rows, coverage and costs: [RECIPE-BINDING-RESULTS-20260924.json](RECIPE-BINDING-RESULTS-20260924.json).
