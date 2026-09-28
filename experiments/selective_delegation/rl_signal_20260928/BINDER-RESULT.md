---
date: 2026-09-28
status: complete_native_audited_paired_collections
scope: same_warm_actor_exposed_TRAIN8_no_learning_claim
reports: analysis-rl-signal-20260928-001/{binder.json,COMPARISON.json}
---

# Assistance improves collection outcomes but reduces terminal-credit exposure

With the same warm public-discovery actor, task identities, four execution seeds,
and caps, binder succeeds on **28/32**, versus raw **24/32**: six paired wins, two
losses and 24 ties. This is an interface effect during collection, not RL improvement.

| Metric | Raw | Binder |
|---|---:|---:|
| Native success | 24/32 | 28/32 |
| Mixed task groups | 4/8 | 3/8 |
| Nonzero-credit trajectories | 16 | 12 |
| Nonzero-credit calls | 583 | 294 |
| Nonzero-credit emitted tokens | 19,787 | 8,440 |
| Absolute advantage-weighted token mass | 10,419 | 4,352 |
| All physical calls | 790 | 556 |
| All emitted tokens | 25,212 | 15,761 |
| Prompt tokens | 1,766,263 | 805,652 |
| Native service seconds | 1,977.89 | 1,196.23 |
| Native craft errors / schema errors | 348 / 1 | 106 / 0 |
| Transport failures | 0 | 0 |

Binder's mixed groups are TRAIN429 (3/4), TRAIN465 (2/4), and TRAIN860 (3/4).
The other five groups are all-success. Eight positive trajectories contribute
6,110 tokens and absolute advantage-weighted mass 2,280; four negative trajectories
contribute 2,330 tokens and mass 2,072. Twenty trajectories have zero credit. This
changed exposure follows legitimately from the same token-sum terminal objective;
it is not an objective defect or a causal attribution to individual steps.

All initial prefixes still yield one identical root query across four seeds for
every task. Later calls contain49 repeated exact-prefix groups,16 with distinct
full actions and16 with distinct projected choices. Different histories are not
merged. Mean conditional vocabulary entropy on visited token prefixes is 0.03234
nats versus raw 0.03844; state visitation differs, so this is not an entropy change
caused by learning or a same-state entropy comparison.

All 337 schema-valid binder craft texts align exactly with their saved token IDs.
Ingredients occupy 40.91% of their 12,460 emitted tokens, or at most 46.32% including
explicit boundary-overlap tokens; decision members occupy 42.86%. The binder
changes actual arguments on 41 calls. Its 334 observed-recipe/divisible craft calls
are eligible for payload masking even when the requested ingredients already
equal the deterministic replacement; eligibility is not outcome selection.

The prospective same-batch diagnostic gate passes. Among 179 eligible calls on
nonzero-credit trajectories, a conservative mask wholly inside the ingredient
JSON value removes 2,370 token-loss terms and absolute advantage-weighted mass
1,186.67: 27.27% of total credited mass. Fixed keys/colons, boundary tokens, EOS,
and all unbound/schema-invalid actions remain supervised. Native errors need not
be ingredient errors: stock, requested target and quantity remain model decisions.

Decision: prepare one additive same-batch masked update and paired readout, as a
biased surrogate diagnostic; retain the valid full-token baseline. These eight
exposed TRAIN clusters do not establish generalization, and the already queued
compact-interface cycle remains a different experiment. See
[the diagnostic contract](MASK-DIAGNOSTIC.md) and
[raw details](RAW-RESULT.md). No accepted run or source was changed.
