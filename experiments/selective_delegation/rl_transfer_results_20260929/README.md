# Familiar-goal RL: little transfer on the fixed fresh-B diagnostic

Evidence cutoff: **2026-09-29 05:30:50 UTC**. All six cells are complete: 96/96 native-audited attempts, zero unknown outcomes or transport errors. Each cell repeats the same eight B roots with two fixed seeds; these are not 96 independent problems.

The one-update familiar-goal gains largely do not carry to B. With execution interface held fixed, the raw-trained actor gains **one attempt**, while the binder-trained actor gains **none**. This is limited observed transfer, not proof that RL cannot generalize.

## Complete actor × execution matrix

“Binder” fills ingredient arguments from recipes already observed; it does not choose the target/quantity or provide hidden recipes. Warm means the unchanged discovery-SFT checkpoint23. Both RL actors received one update from that warm actor, on familiar goals, not B.

| Actor | Execution | Success /16 | Calls | Output tokens | Native / schema errors | Context caps |
|---|---|---:|---:|---:|---:|---:|
| Warm | Raw | 4 | 778 | 23,516 | 348 / 1 | 2 |
| Warm | Binder | 5 | 673 | 21,759 | 241 / 3 | 1 |
| Raw-trained | Raw | 5 | 739 | 22,870 | 308 / 1 | 2 |
| Raw-trained | Binder | 4 | 693 | 21,648 | 237 / 2 | 0 |
| Binder-trained | Raw | 5 | 862 | 27,018 | 374 / 1 | 6 |
| Binder-trained | Binder | 5 | 698 | 21,814 | 228 / 0 | 2 |

Paired against the warm actor in the same execution column: raw-trained/raw has 1 win, 0 losses; raw-trained/binder has 0 wins, 1 loss; binder-trained/raw has 1 win, 0 losses; binder-trained/binder has 16 ties. Each single-attempt change is 6.25 percentage points.

## Two roots, not one—and no deep-root breakthrough

Every success disagreement across the six cells lies on two task/seed pairs:

- TRAIN1796, root `o2_i4_12` ×3, actual depth3, seed202609280900: warm/binder, raw-trained/raw and binder-trained/binder succeed; the other three cells fail.
- TRAIN1273, root `o1_i4` ×2, actual depth4, seed202609280901: only binder-trained/raw succeeds.

The other repeat of each root fails everywhere. TRAIN256 and TRAIN1847 succeed twice in every cell. TRAIN672, TRAIN38, TRAIN201 and TRAIN964 fail twice in every cell. Thus every cell remains **0/6 on the same three actual-depth5 roots**. Declared item tiers are not actual dependency depths.

This concentration concerns success, not all behavior or cost. For example, binder-trained/raw still solves TRAIN256 twice but takes 69 calls rather than the warm actor's 25.

## Cost and uncertainty

Binder execution reduces calls relative to raw execution for every fixed actor: 105 fewer for warm, 46 fewer for raw-trained and 164 fewer for binder-trained. Native errors also fall, but success changes are +1, −1 and 0 attempts respectively. These are execution effects, not evidence of better learning.

Relative to their own-interface warm baselines, raw-trained/raw uses 39 fewer calls (−5.0%) and 646 fewer output tokens; binder-trained/binder uses 25 more calls (+3.7%) and 55 more tokens, with identical success. Binder-trained/raw costs 84 extra calls and 3,502 extra tokens and reaches six context caps instead of two.

The four new transfer readouts used 2,992 calls, 93,350 output tokens and 129.7 summed service minutes. Including the two reused warm controls gives 4,443 calls, 138,625 tokens and 191.9 service minutes. These are request-time sums, not allocation wall time or additional training cost.

The descriptive eight-root bootstrap gives [0,+18.75] percentage points for either +1-attempt effect and [−18.75,0] for the −1 effect. The assisted all-tie comparison mechanically returns [0,0]; that degenerate resampling interval does **not** establish equivalence. The crossed training-interface × execution interaction is +6.25 points, interval [−18.75,+37.5]: no stable co-adaptation advantage is established. Intervals omit training-run uncertainty and multiplicity adjustment.

## Familiar gains remain real but narrowly scoped

| Fixed own-interface comparison | Familiar training goals | Different B roots |
|---|---|---|
| Raw warm → raw-trained | 9 → 13 /16 | 4 → 5 /16 |
| Binder warm → binder-trained | 14 → 16 /16 | 5 → 5 /16 |

The actor identities match across panels. Familiar readouts use goals already in the RL batch, with new rollout seeds; B roots were prospectively excluded from gradients. Compare before/after within each panel, not final scores across panels: goals, difficulty and seeds differ.

B remains an exposed official TRAIN diagnostic in one shared recipe world, not held-out VAL/HOLDOUT or unseen lower-level recipes. There is one realized update per interface, with different trajectories and credited-token exposures (19,787 raw versus 8,440 binder). No broad transfer, equal-dose learning advantage or recursive-decomposition benefit follows.

## Decision and slide4

Complete the already accepted first fresh-A update/B readouts and extra-SFT control. They address whether learning from different, difficulty-matched A roots transfers and whether extra example-based training already helps. Extra SFT matches one step/start/LR/arithmetic, not data, target tokens or FLOPs. Report every fixed endpoint; never select checkpoints, training labels, replacement tasks or continuation gates using B outcomes.

Recommend slide 4 title: **“Reward training improved familiar goals, with little transfer so far.”** Use the compact before/after table above. Visible limitation: “One update per interface; eight different TRAIN roots. One extra raw success, no assisted gain.” Keep the crossed matrix and intervals in notes. Fresh-A learning and extra-SFT remain separate pending tests at this cutoff. Update slide/data/cutoff/notes/guide together; this analysis leaves the existing 01:14 deck and reports unchanged.

## Evidence and checks

[RESULTS.json](RESULTS.json) contains all 96 task/seed outcomes, root-level pairing, costs, exact actor identities, effects and 236 small receipt hashes. `R` expands to `/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`; `E` is this worktree's `experiments/selective_delegation`.

Primary source: `R/textcraft-rl-transfer-20260928-001/COMPARISON.json`; familiar source: `R/textcraft-rl-assist-20260928-001/COMPARISON-0001.json`. Actual PLANs, completed audit/SUMMARY counts, matching manifests/runtime contracts, 96 saved episodes, 96 first-call PLAN links and committed actor JSON were checked. All reported transfer effects were recomputed from audited scores; the JSON specifies the bootstrap exactly.

No new native replay, full trace/model rehash, GPU work, queue/owner changes, Git operations or frozen-source edits were performed.
