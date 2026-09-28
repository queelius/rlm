# TextCraft breadth: fixed panels00–05, 2026-09-28

Discovery training plus observed-recipe binding reaches 381/768 (49.61%); raw is 323/768 (42.06%). The quantity-corrected **known-recipe** teacher reaches only 48/768 (6.25%) with binding and 39/768 (5.08%) raw. `corrected` never denotes a correction to discovery training.

The cutoff is 192 native-audited runs: six panels × four worlds × two training seeds × two teacher packages × two execution modes, with 16 attempts each. All 3,072 outcomes are observed, all 192 task-world inputs passed native qualification, and small audited PLAN/episode/input hashes match. Panels06–07 are excluded even when completed.

| Teacher / mode | Successes /768 | Calls | Calls / attempt | Native action errors | Schema errors |
|---|---:|---:|---:|---:|---:|
| discovery/raw | 323 | 24,759 | 32.24 | 9,409 | 146 |
| discovery/binder | 381 | 22,342 | 29.09 | 5,690 | 146 |
| known_recipe_corrected/raw | 39 | 25,203 | 32.82 | 2,908 | 134 |
| known_recipe_corrected/binder | 48 | 25,430 | 33.11 | 2,686 | 130 |

All native service calls returned; no transport failures or unknown outcomes. Native action errors are environment feedback, not failed model requests.

| Paired contrast | Difference, percentage points | 95% task-cluster interval | Wins / losses / ties |
|---|---:|---:|---:|
| discovery_binder_minus_raw | +7.55 | [+5.08, +10.16] | 74 / 16 / 678 |
| known_recipe_corrected_binder_minus_raw | +1.17 | [-0.26, +2.86] | 15 / 6 / 747 |
| discovery_minus_known_recipe_raw | +36.98 | [+32.94, +40.89] | 292 / 8 / 468 |
| discovery_minus_known_recipe_binder | +43.36 | [+38.67, +47.92] | 341 / 8 / 419 |
| binder_by_teacher_interaction | +6.38 | [+3.52, +9.24] | 77 / 26 / 665 |

Intervals resample the 48 task identities within declared-depth strata, retaining all worlds, training seeds and repeated attempts. They are conditional on these four worlds and two fitted seeds. Interaction wins/losses count the sign of the difference-in-differences, not a simple arm comparison. Full JSON includes an unstratified task-cluster sensitivity and call/error intervals.

The strongest mechanistic lead is **what the demonstrations ask the model to know**. All 32 task-initial prompts are identical. All 32 per-task target-string multisets are identical after the quantity correction; canonical parsed-action and unmasked label-token multisets also have no differences. Both supervise 8,820 target tokens. The difference lies in action ordering and the public state/history conditioning those same actions; discovery has 438,065 prompt tokens versus 414,682 (+5.64%).

| Training-time query visibility | Discovery | Known-recipe corrected |
|---|---:|---:|
| First query names available in public state | 32/32 | 2/32 |
| All query rows using available names | 167/167 | 32/167 |

Visibility is checked against structured `target_items`, current/initial inventory and previous `get_info` item/recipe-ingredient feedback, excluding schema examples and model-invented names echoed in other actions. A literal-name-in-prompt check agrees on all training queries. It differs on four evaluation first queries per known-recipe mode; structured visibility is the primary measure.

| Evaluation behavior | Discovery raw / binder | Known-recipe raw / binder |
|---|---:|---:|
| First physical action queries root | 765 / 765 of768 | 113 / 113 of768 |
| First query uses available names | 768/768 / 768/768 | 113/762 / 113/762 |
| Ever queries root | 768 / 768 | 232 / 244 |
| Nonexistent-item query mentions | 104 / 97 | 10,552 / 10,549 |

Six known-recipe episodes in each mode never query. Of its 762 queried episodes, 649 first ask for a name absent from the structured public information. This is consistent with learning a privileged leaf-first naming policy that transfers badly; it is still a behavioral association, not an isolated causal result.

Synthetic example: TRAIN `textcraft_synth.train.988` starts with goal `c6_i2` and inventory `c5_ore`, `c0_ore`. Discovery teaches `get_info(c6_i2)`. The corrected known-recipe teacher teaches `get_info(c9_i1_19)` before that prerequisite name appears in public information. On VAL `textcraft_synth.val.313`, panel00/world42/training-seed2026092208/repeat0, the goal is `t8_i2_18`, inventory is `raw_o6`, `raw_t1`, `raw_t3`, but the known-recipe model first asks for unseen `t1_i1_2`. Native feedback reports no recipe and it ultimately fails; discovery queries the visible goal and succeeds. These examples illustrate the full-panel counts; they are not an outcome-selected evaluation subset.

| declared_depth | Discovery raw → binder | Known-recipe corrected raw → binder |
|---|---:|---:|
| 2 | 177 → 181 /192 | 25 → 31 /192 |
| 3 | 108 → 124 /192 | 12 → 11 /192 |
| 4 | 27 → 39 /96 | 2 → 5 /96 |
| 5 | 11 → 37 /288 | 0 → 1 /288 |

| world | Discovery raw → binder | Known-recipe corrected raw → binder |
|---|---:|---:|
| 42 | 95 → 101 /192 | 11 → 7 /192 |
| 50 | 68 → 88 /192 | 8 → 13 /192 |
| 51 | 78 → 103 /192 | 10 → 14 /192 |
| 52 | 82 → 89 /192 | 10 → 14 /192 |

| training_seed | Discovery raw → binder | Known-recipe corrected raw → binder |
|---|---:|---:|
| 2026092208 | 177 → 213 /384 | 23 → 32 /384 |
| 2026092291 | 146 → 168 /384 | 16 → 16 /384 |

| dependency_chain | Discovery raw → binder | Known-recipe corrected raw → binder |
|---|---:|---:|
| 1 | 20 → 20 /20 | 6 → 6 /20 |
| 2 | 183 → 191 /208 | 20 → 25 /208 |
| 3 | 94 → 107 /172 | 11 → 11 /172 |
| 4 | 23 → 47 /148 | 2 → 5 /148 |
| 5 | 3 → 16 /220 | 0 → 1 /220 |

| panel | Discovery raw → binder | Known-recipe corrected raw → binder |
|---|---:|---:|
| 0 | 55 → 64 /128 | 4 → 4 /128 |
| 1 | 53 → 61 /128 | 15 → 19 /128 |
| 2 | 57 → 66 /128 | 5 → 6 /128 |
| 3 | 64 → 71 /128 | 4 → 6 /128 |
| 4 | 48 → 52 /128 | 4 → 6 /128 |
| 5 | 46 → 67 /128 | 7 → 7 /128 |

Binding improves discovery in every world, both training seeds and every declared-depth stratum. It cuts total discovery calls by 9.76% and native action errors by 39.53%; schema errors stay at 146. Context-cap terminations fall from 49 to 32. These paired effects support keeping the binder, with remaining failures concentrated in harder tasks.

Depth5 remains difficult: 37/288 (12.85%) for discovery+binding, versus 181/192 (94.27%) at depth2. The known-recipe package has fewer native action errors yet far fewer successes; low error counts alone are not a competence metric. Exact error/status breakdowns and per-task outcomes are available in the external details artifact.

The primary interpretation is a teacher-package transfer advantage plus an additional binding benefit. It does not establish that query order alone causes the package gap, or that recursive delegation/RLVR has improved.

Binding shifts discovery errors from argument construction toward feasibility: missing/extra/wrong ingredient errors fall from 5,953 to 281, while explicit insufficient-inventory errors rise from 3,155 to 5,132. This change reflects new trajectories and error precedence; it does not prove the binder worsens planning.

Historical controls already completed:

- [Root-first procedure prompt](TEXTCRAFT-PROCEDURE-CONTROL-FINDINGS.md): old uncorrected known-recipe model 0/16 versus default 3/16, despite more root queries; repeated static-recipe queries rose 70→418. A minimal prompt on corrected weights would be a changed replication, not a first test of root-first prompting.
- [One-step extra SFT](TEXTCRAFT-ONE-STEP-FINDINGS.md): discovery 17/32 versus 15/32 baseline, with an uncertain effect; matched one-update/12,074-token RL is also 15/32. This does not settle whether the known-recipe teacher is undertrained. The active-store training-plan scan found no matched multi-epoch teacher readout.

Limitations:

- Known_recipe_corrected is the quantity-corrected known-recipe teacher, not a corrected discovery model.
- Both teacher packages have 32 tasks, 366 rows and 23 updates; query order, histories, prompt-token exposure and trajectory package differ. Target-token dose is identical after the known-recipe quantity correction.
- Forty-eight task identities, not 768 independent examples; worlds, repeated evaluation seeds, training seeds, names and generator induce dependence.
- World42 is the source recipe world; worlds50–52 share generator/item grammar and change recipes plus sufficient inventory. New-to-inventory goal roots are not an unbounded historical exposure guarantee.
- Declared task depth can exceed realized dependency-chain depth. Four depth strata are deliberately reweighted; rates are not an official benchmark average.
- The first six complete panels are an explicit analysis cutoff; panels06–07 are excluded regardless of availability/outcome. The larger campaign is incomplete.
- Binding modifies ingredients only from a previously observed single recipe when output quantity is divisible; it does not choose targets, quantities, decomposition or stopping.
- All arms are flat action SFT; these data do not establish recursive delegation, RLVR gains, other-model transfer or equality of inference token cost.
- Native audits are reused; this synthesis verifies PLAN/episode, node histories, first-query calls and frozen input hashes, not a full new replay or weight rehash.
- Query visibility and failure categories are descriptive correlates, not isolated causal mechanisms; repeated nonexistent-name mentions are not independent errors.

Next research decisions:

1. Highest priority: repair known-recipe teacher query observability while preserving crafting order and action/target multiset, then compare fixed trained endpoints against both existing packages. Audit every query name against public state. This tests whether unobserved-name supervision drives the transfer failure.
2. Separate order from optimization: compare predetermined one/two/three-epoch endpoints of the same frozen teacher packages with matched updates and seeds. The existing one-step discovery SFT control does not answer known-teacher undertraining. Keep all checkpoints in the report; do not choose a best VAL result.
3. Check teacher-forced fit by action type and query visibility on frozen TRAIN/VAL rows: successful fit to hidden-name labels with poor rollout transfer supports a supervision/state-distribution problem; poor fit leaves undertraining open.
4. Retain discovery plus binding for depth4–5 feasibility/quantity interventions. Binding removes ingredient-selection errors but does not ensure enough inventory. Compare bounded quantity planning/decomposition on paired goals before broad RLVR.
5. Do not repeat the completed multi-sentence root-first prompt control as if new. A minimal root-first-only instruction on corrected weights and new goals is a changed replication, lower priority than repairing training-time observability.

Update the breadth/generalization evidence: large teacher-package difference, matched supervised target dose but sharply different query observability, smaller binding gain, and severe depth5 limitation. Parent owns deck integration.

Reproduce from the repository root:

```bash
python experiments/selective_delegation/analyze_textcraft_breadth_20260928.py \
  --report /tmp/textcraft-breadth-20260928.json \
  --details /tmp/textcraft-breadth-20260928-details.json
```

The command also writes Markdown beside the JSON. Outputs are exclusive-create. The checked-in JSON records the script hash and external evidence-details hash; the latter includes every verified input receipt and all 192 run audit hashes.
