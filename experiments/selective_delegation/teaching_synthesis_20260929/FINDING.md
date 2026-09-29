# Teaching controls: synthesis

Cutoff: 2026-09-29T01:14:27.611452+00:00

All 40 unique cells are natively audited: 576/576 actual attempts, zero unknowns and zero failed transport calls. Native action/schema errors remain in costs/outcomes.

The strongest current interpretation is improved learnability at the tested optimization budget. Repair gains reproduce across a second fit seed and additional root goals; extra training partially rescues the original known-recipe teacher. Phi tests the original teacher/interface packages, not transfer of the repair method.

## Repair replication (both fixed worlds)

| Measurement stratum | Known | Discovery | Stable repair | Random repair |
| --- | ---: | ---: | ---: | ---: |
| pilot | 1/32 | 14/32 | 14/32 | 12/32 |
| added_fit | 2/32 | 13/32 | 13/32 | 12/32 |
| added_roots | 3/32 | 15/32 | 17/32 | 15/32 |
| added_combined | 5/64 | 28/64 | 30/64 | 27/64 |

Pilot = panel00/fit2208. Added-fit = panel00/fit2291; added-roots = panel01/fit2208. The added measurements were fixed after the pilot but reuse exposed breadth panels. Added-combined has 16 roots; each individual stratum has eight. Do not count worlds, fits or rollout repeats as new compositional problems.

| Selected paired contrast | Wins/losses/ties/unknown | Difference pp | Root 95% pp |
| --- | --- | ---: | --- |
| pilot:stable_visible-known | 13/0/19/0 | +40.62 | [15.62, 65.62] |
| pilot:random_visible-known | 11/0/21/0 | +34.38 | [9.38, 68.75] |
| added_fit:stable_visible-known | 11/0/21/0 | +34.38 | [6.25, 65.62] |
| added_fit:random_visible-known | 10/0/22/0 | +31.25 | [6.25, 62.5] |
| added_roots:stable_visible-known | 15/1/16/0 | +43.75 | [21.88, 65.62] |
| added_roots:random_visible-known | 12/0/20/0 | +37.50 | [18.75, 62.5] |
| added_combined:stable_visible-known | 26/1/37/0 | +39.06 | [21.88, 57.81] |
| added_combined:random_visible-known | 22/0/42/0 | +34.38 | [15.62, 53.12] |
| added_combined:stable_visible-discovery | 4/2/58/0 | +3.12 | [-4.69, 10.94] |
| added_combined:random_visible-discovery | 4/5/55/0 | -1.56 | [-9.38, 7.81] |
| dose:known:46-23 | 8/1/23/0 | +21.88 | [3.12, 46.88] |
| dose:known:69-23 | 7/0/25/0 | +21.88 | [9.38, 34.38] |
| pilot:stable_visible-known69 | 8/2/22/0 | +18.75 | [0.0, 40.62] |
| pilot:random_visible-known69 | 6/2/24/0 | +12.50 | [-9.38, 37.5] |
| phi:raw:discovery-known | 3/0/13/0 | +18.75 | [0.0, 43.75] |
| phi:binder:discovery-known | 6/0/10/0 | +37.50 | [12.5, 75.0] |
| phi:discovery:binder-raw | 2/0/14/0 | +12.50 | [0.0, 37.5] |
| phi:known:binder-raw | 0/1/15/0 | -6.25 | [-18.75, 0.0] |

## Dose, cost and model family

| Fixed cohort | Success/planned | Native calls | Output tokens |
| --- | ---: | ---: | ---: |
| pilot:known | 1/32 | 1206 | 25666 |
| dose:known:46 | 8/32 | 1848 | 39593 |
| dose:known:69 | 8/32 | 1958 | 42329 |
| pilot:discovery | 14/32 | 1201 | 41932 |
| dose:discovery:46 | 15/32 | 1088 | 33932 |
| dose:discovery:69 | 17/32 | 1115 | 35905 |
| pilot:stable_visible | 14/32 | 767 | 19251 |
| pilot:random_visible | 12/32 | 1033 | 27982 |
| added_combined:known | 5/64 | 2118 | 45031 |
| added_combined:discovery | 28/64 | 2024 | 60828 |
| added_combined:stable_visible | 30/64 | 1581 | 42242 |
| added_combined:random_visible | 27/64 | 1990 | 56481 |
| phi:known:raw | 1/16 | 1110 | 26163 |
| phi:known:binder | 0/16 | 1209 | 27430 |
| phi:discovery:raw | 4/16 | 482 | 12940 |
| phi:discovery:binder | 6/16 | 426 | 12780 |

Known continuation reaches 8/32 at both fixed 46- and 69-update endpoints, from 1/32 at update23. World-level outcomes change from 3/16+5/16 to 5/16+3/16, so selecting the better checkpoint per world would inflate evidence. Discovery moves 14→15→17/32. Repair's 23-update pilot reaches 14/32 stable or 12/32 random. This supports a learning-efficiency advantage at tested doses, not an irreparable teacher, an asymptotic advantage, or a dose-independent causal effect. Cost includes every audited attempt, not only successes. Fewer calls can also reflect stopping or errors.
Known cumulative training time was 182→364→545 seconds at these fixed endpoints; stable repair took 184 seconds for update23. These are trainer-state durations, not owner wall time. The direct repair-versus-continued intervals remain broad.

Both repair fit seeds preserve recorded initialization, 366 targets in known-teacher row order, all 23 minibatch target-token denominators, and 8,820 answer tokens including EOS. Original crafting order is retained; schedules use offline gold actions while querying publicly visible names. Input history and input-token dose change (414,682 known; 431,274 stable; 435,710 random). The random schedule is not a globally optimal teacher; both orderings remain in the report without selecting the winner.

Phi: discovery scores 4/16 raw and 6/16 binder, versus known 1/16 raw and 0/16 binder. Eight roots and one rollout seed/world give little precision; both binder gains are VAL263 in different worlds. This supports limited original-package dependence across model families, not repair-method transfer. There is no Phi repaired actor yet. Equal 23 updates across families do not equalize tokenizer, labels, LoRA dimensions or compute. Failed base probes are excluded from these valid trained-cell contrasts.

## Next two comparisons

1. **Phi stable-visible repair**, same 366 known-teacher targets/minibatch order, fixed seed/update23 and two raw world readouts (8 roots/one rollout seed each). Pair with existing Phi-known/discovery. This directly tests repair-method transfer; no rescue should narrow a family-general claim. Add binder only if a competent repaired raw policy leaves an execution-interface question.
2. **A fixed fresh-root/world repair test**, conditional on Phi's result: compare the existing known, discovery and stable actors at one predetermined fit/checkpoint. Before rollouts, select eight roots outside all exposed breadth/pilot goals and all SFT query products, plus an unused world seed; verify feasibility from compact manifests. Retain the same 16-attempt budget and report lower-recipe/graph overlap. This tests whether repair extends beyond the exposed problem family, not wholly unseen composition. If no such root slice exists, say so and prioritize the narrower cross-world claim rather than relabeling shared recipes as unseen structure. A null Phi repair result should narrow the family-general claim before more dose grids.

## Limits and authority

20,000 paired root-cluster bootstrap draws; fixed worlds/fit/rollout repeats stay together. Added-combined resamples eight roots within each panel, retaining panel weights. Intervals describe the exposed fixed design, not benchmark-level uncertainty; shared recipes can correlate roots. Unknown pairs suppress estimates and intervals. No best endpoint.

Existing native audit authority, with small PLAN/receipt lineage checks; not fresh replay or model/trajectory rehash. Actual Qwen16/Phi8 slots per cell.
 All comparisons are exploratory and adaptive; no multiplicity-adjusted claim, absence-of-effect equivalence, unseen-structure claim, or universal novelty claim. Root disjointness does not remove shared lower-level recipes. JSON retains every cell, paired root count, cost, explicit checkpoint and small-receipt hash. Per-attempt scores are retained once per cell so all paired rows can be reconstructed.
