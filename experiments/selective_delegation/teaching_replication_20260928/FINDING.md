# Teaching-order repair readout

Cutoff: 2026-09-28T15:56:14.836212+00:00

Fixed native-evidence cutoff: 2026-09-28T15:11:13.985232+00:00

The matched-target repair recovers much of the known-teacher gap on this fixed exploratory slice. Stable matches discovery's total, not demonstrated superiority.

| Teacher | Successes | Calls | Output tokens |
| --- | ---: | ---: | ---: |
| known | 1/32 | 1206 | 25666 |
| discovery | 14/32 | 1201 | 41932 |
| stable_visible | 14/32 | 767 | 19251 |
| random_visible | 12/32 | 1033 | 27982 |

| Paired contrast | Wins/losses/ties/unknown | Difference pp | Root-bootstrap 95% pp |
| --- | --- | ---: | --- |
| stable_visible_minus_known_w42 | 6/0/10/0 | +37.50 | [12.5, 75.0] |
| stable_visible_minus_discovery_w42 | 1/2/13/0 | -6.25 | [-37.5, 18.75] |
| random_visible_minus_known_w42 | 6/0/10/0 | +37.50 | [12.5, 75.0] |
| random_visible_minus_discovery_w42 | 1/2/13/0 | -6.25 | [-37.5, 18.75] |
| stable_visible_minus_known_w50 | 7/0/9/0 | +43.75 | [18.75, 68.75] |
| stable_visible_minus_discovery_w50 | 2/1/13/0 | +6.25 | [-12.5, 25.0] |
| random_visible_minus_known_w50 | 5/0/11/0 | +31.25 | [6.25, 62.5] |
| random_visible_minus_discovery_w50 | 1/2/13/0 | -6.25 | [-37.5, 18.75] |
| stable_visible_minus_known_fixed_two_worlds | 13/0/19/0 | +40.62 | [15.62, 65.62] |
| stable_visible_minus_discovery_fixed_two_worlds | 3/3/26/0 | +0.00 | [-12.5, 12.5] |
| random_visible_minus_known_fixed_two_worlds | 11/0/21/0 | +34.38 | [9.38, 68.75] |
| random_visible_minus_discovery_fixed_two_worlds | 2/4/26/0 | -6.25 | [-25.0, 12.5] |

Controls: 32 TRAIN tasks, 366 rows, exactly 8,820 label tokens including EOS, 23 updates. Both repairs preserve literal targets and unmasked label IDs in the original known-teacher row order; all 23 actual minibatch target denominators match. Checkpoint-0 COMMIT receipts record identical initial LoRA weights. Base, fit seed, optimizer recipe and original craft order match. Public-name queries increase from 32/167 to 167/167. Both repaired native sequences match discovery on only 8/32 TRAIN tasks. Prompt-token totals differ: known 414,682, stable 431,274, random 435,710.

Offline gold-action scheduler; public query names and stock-feasible native replay. Conditioning histories and input-token dose differ; no visibility-only causal isolation.

| Root task ID | Known successes /4 | Stable successes /4 |
| --- | ---: | ---: |
| textcraft_synth.val.109 | 1 | 4 |
| textcraft_synth.val.263 | 0 | 4 |
| textcraft_synth.val.294 | 0 | 0 |
| textcraft_synth.val.313 | 0 | 3 |
| textcraft_synth.val.401 | 0 | 0 |
| textcraft_synth.val.502 | 0 | 1 |
| textcraft_synth.val.599 | 0 | 0 |
| textcraft_synth.val.628 | 0 | 2 |

Eight exposed root identities, two correlated worlds, two rollout seeds and one fit seed. Bootstrap resamples the eight roots, holding worlds/repeats together; shared recipes remain correlated. No discovery-superiority, unseen-structure or general novelty claim. Fixed dose controls are still separate pending evidence.

Next decision: Retain both orderings; inspect fit-seed and additional-root slices separately with fixed 46/69-update dose controls. A replicated rescue supports this trace-repair package, not isolated visibility causality; mixed replication should narrow the claim before expansion.
