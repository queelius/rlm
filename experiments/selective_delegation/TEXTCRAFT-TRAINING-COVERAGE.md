# What the small TextCraft action-SFT set covers

CPU input-only audit, September 22. No source049 partial outcomes were read. The fixed source048 one-epoch training decision and checkpoint23 endpoint were chosen independently of that control's result. This note describes what the frozen source047 data can teach; it does not infer what the adapter actually learned.

## Recipe and item overlap with the eight-task VAL pilot

TRAIN contains 32 tasks with 30 distinct root items. VAL contains eight tasks with eight distinct root items. There is no task-ID or root-item overlap. More strongly, **none of the eight VAL root recipes is queried or crafted anywhere in these TRAIN demonstrations**.

However, all tasks use the same seed42 recipe world. TRAIN queries and crafts 126 distinct recipes; the eight VAL gold plans require 47 distinct recipes. **21/47 are explicitly exposed in TRAIN**, including their exact ingredient quantities and output batch sizes. All 126 saved TRAIN public recipe replies were checked against the pinned native world. The other 28 VAL recipes, including all eight roots, are absent from these demonstrations—not necessarily unfamiliar to the pretrained model. Counting root, crafted, ingredient and initial-inventory item names gives 174 TRAIN items, 75 VAL items, and 49 shared names. Item-name overlap alone is weaker than recipe exposure.

| VAL slot / official ID suffix | Declared depth | Gold craft actions | Distinct required recipes exposed in TRAIN |
|---|---:|---:|---:|
| t00 / 325 | 2 | 3 | 1/3 |
| t01 / 629 | 3 | 5 | 1/5 |
| t02 / 47 | 3 | 4 | 1/4 |
| t03 / 19 | 2 | 2 | 1/2 |
| t04 / 567 | 4 | 11 | 3/11 |
| t05 / 435 | 4 | 5 | 3/5 |
| t06 / 207 | 4 | 9 | 6/9 |
| t07 / 352 | 4 | 10 | 6/10 |

The per-task denominators total 49 because two recipe identities recur across VAL tasks; the unique-panel denominator is 47. Every VAL task mixes exposed and unexposed required recipes. For example, t03 shares intermediate `m0_i1`, but not its root `m0_i2_20`; t05 shares `o0_i1`, `o0_i1_10` and `o1_i2_11`, but not root `o8_i4`.

Earlier source047 documentation reports overlap with **the entire official VAL split**: all 30 selected TRAIN roots appear as intermediates somewhere in that larger split. That must not be confused with root exposure in this particular eight-task pilot, where none of its eight roots appears in selected TRAIN actions.

## Depth and task size

| Declared depth | TRAIN tasks | TRAIN gold crafts: min / median / max | VAL tasks | VAL gold crafts: min / median / max |
|---|---:|---:|---:|---:|
| 2 | 11 | 1 / 2 / 3 | 2 | 2 / 2.5 / 3 |
| 3 | 11 | 3 / 4 / 6 | 2 | 4 / 4.5 / 5 |
| 4 | 10 | 5 / 8 / 17 | 4 | 5 / 9.5 / 11 |

All 40 tasks have one root target item. There is no multi-root-goal supervision or readout here. VAL's 2–11 craft-action range is inside TRAIN's 1–17 range, and all nominal depths are represented in TRAIN. This is not out-of-range depth or size extrapolation.

The prior inventory-aware audit further distinguishes declared recipe tier from the constructive gold dependency chain: TRAIN chain lengths 1/2/3/4 occur in 2/10/14/6 tasks; VAL chain lengths 2/3/4 occur in 3/3/2 tasks. Initial inventories in these 40 tasks contain only native base items; skip-level recipe edges account for shorter chains here. Those constructive chain measurements are not minimum-plan proofs. See `analysis-textcraft-depth-001.json` for the full mapping.

## What the supervision does—and does not—teach

An additive exact-identifier audit covers every query, not only the first query:
135/167 source047 query targets (30/32 tasks) do not occur as a dictionary key
or complete string value anywhere in that row's prior public JSON frame,
including nested history. Source055's public-discovery prototype has 0/167
such absent targets on the identical task set. This identifies teacher-supplied
query names, not illegal queries or input leakage; arbitrary identifiers remain
valid query actions, and the audit is not a model-knowledge test. The immutable
`analysis-textcraft-query-visibility-001.json` records every query, prompt hash,
row-source hash and exact matching method; see `PUBLIC-DISCOVERY-READINESS.md`.
An additive receipt002 broadens matching to identifier-bounded occurrences
anywhere in the entire prior prompt, including free-text feedback; it confirms
the same 135 versus zero counts without overwriting receipt001.

There are 366 strict JSON-plus-EOS action rows: 167 recipe queries, 167 crafts and 32 explicit finishes. Of 8,821 supervised tokens, 2,620 belong to queries, 5,881 to crafts and 320 to finishes. Craft rows therefore dominate target tokens; calling this “only format training” would be inaccurate. The targets also supervise product choice, ingredient dictionaries, batch quantities, ordering and stopping. Prompt/history tokens are masked from the loss.

Each craft is preceded by a public query for that product and its native recipe reply. Histories contain only successful teacher actions; no invalid JSON, rejected craft, error recovery, exploration failure or repetition loop is demonstrated. All 32 selected tasks succeed, so there was no replacement or realized success-filter attrition. Finish is supervised once per task, immediately after the gold sequence.

**Thirty of 32 first actions query an intermediate, not the root.** The teacher's gold plan chooses that intermediate before its connecting root recipe has been publicly discovered. This is intentional privileged action-target supervision, not hidden future data in the prompt. It can teach direct goal-to-prerequisite associations in this fixed world, but it does not demonstrate a general root-first recipe-discovery procedure. The two first-root-query examples are one-craft tasks. Every trajectory follows query/craft pairs and then finish; there are no `view_inventory` or `delegate` targets, child frames, or recursive budget decisions.

Thus a positive fixed-checkpoint result could combine stricter single-action/EOS behavior, better batch execution and termination, remembered intermediate recipes, and improved ordering on new root goals. Aggregate success alone cannot separate those mechanisms. A negative result could reflect limited discovery/recovery coverage rather than inability to learn planning. The already-prepared original/reminder × base/SFT comparison helps distinguish a cheap instruction effect from an additional training effect; it still does not isolate novel planning or establish recursion benefit.

## Exact inputs and counting method

All paths below are relative to `/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/`:

- TRAIN tasks `textcraft-train-inputs-001/tasks.jsonl`: SHA256 `390dff9bb19d0fe71c7bec0505c97608013c66c65aea90a821839f01b615ab30`.
- TRAIN rows `textcraft-train-inputs-001/rows.jsonl`: `caa78390f9d4ac28e600674b26e56375203b72d8cdad1c3f9471da3fb25776a9`.
- TRAIN manifest: `670c808d563ff9fd46aa511089acfb475ce377cea803b7ffcc5d285f364d0a98`.
- VAL tasks `textcraft-inputs-001/tasks.jsonl`: `16a6663385759a9a16fa7ca44e6601b4d491234f8b04ccb9f2bf522c7ced8ab3`.
- VAL manifest: `79ac9326c209e3df3d31cf3a479fec264506fb15e19445f06370a1c3d5baf405`.
- Existing depth audit: `67699aa3e0859c5bf99ae9f88cab5c411adad7d47b16dd9ddc96598f67d13a96`.

Recipe exposure is the set of product IDs in actual TRAIN `get_info` feedback with recipe objects, cross-checked against native ingredients/result counts; VAL requirements are product IDs in its frozen host-only gold craft trajectories. Counts use set intersections, not substring matches. Action/token totals are summed directly from saved target JSON and `target_tokens`. First-query classification compares each task's first target action against that task's root set. Official generator/bridge provenance remains commit `d9c5857d3a0a056ebc9b047241a2a0c9515aafbe` (MIT), already pinned by the manifests. No prompts, inputs, selection or live collectors were changed.
