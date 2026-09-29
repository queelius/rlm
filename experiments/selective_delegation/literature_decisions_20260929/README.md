# Next learning decisions, not another parallel method sweep

Reviewed September 29, 2026, 01:04 UTC. Recommendation: finish fixed-endpoint B transfer and the first fresh-A/extra-SFT comparison before adding optimization. Then prioritize **public-state recovery supervision**; treat **root-aligned recursive credit** as conditional on a completed helper-utility pilot. Neither proposal below is accepted or launched.

## Evidence and duplicate check

The [native-audited first-update analysis](../rl_outcomes_20260928/README.md) finds raw 9→13/16 and binder 14→16/16, but own-interface service cost rises 7.4%/24.0%. Raw post-goal calls rise 20→45. All 111 binder schema errors in the crossed matrix are repeated duplicate-key responses on one goal. These are specific recovery/termination failures, not evidence that every token needs a new advantage estimator.

The actual `R/information-first-tail-20260928-002/ACCEPTED.json` has 120 descriptors: unchanged warm B at indices 62–63, fixed transfer 64–68, fresh cycles/extra SFT 69–99, rejected-action cost 100–104, compact RL 111–115, payload mask 116–119. None should be duplicated. Receipt/input hashes are in [SOURCES.json](SOURCES.json). `R` denotes the existing selective-delegation external store.

Additional admission evidence matters: V2 fixed/adaptive each spawned a real child; their root outcomes are **unknown** at the 32-response screening cap. Fixed child's local score was 1, adaptive child's 0. Those are not completed flat-versus-recursive results. The completed public quantity-table demand/masked screens both scored 0/8; that small exposed panel does not establish that quantity information is useless generally.

## Five primary works that change the decision

| Version checked; read scope | Actual contribution and implication here |
|---|---|
| [Harness-RL, v1, Aug 30, 2026](https://arxiv.org/html/2608.29641v1), §§3.1–3.3,4.1,4.3,5 | CAPO identifies activation-based parameter partitions and routes action/argument gradients separately; its trajectory representation retains exact call tokens and contexts. It is **not** simply dropping ingredient losses. Four/eight A100-80GB training and partition-estimation work make transplantation disproportionate now. Read compact/mask results first. The [official README](https://github.com/jiangxinke/Harness-RL) confirms a distributed StackPlanner/slime stack, not a drop-in one-card trainer. |
| [GACA, v1, Sep 11, 2026](https://arxiv.org/html/2609.12424v1), §§3.3,4.3,5, Appendix B assumptions | Mixes episode and repeated-state credit using sampled-action NLL. Its theory explicitly bounds textual multiplicity and assumes separation of canonical-action values; its conclusion acknowledges biased action-dependent weighting. High NLL of discarded ingredients or repeated malformed JSON is not established decision value. Our sparse exact-prefix groups do not warrant an entropy-credit implementation yet. |
| [Knowing When to Quit, v6, Jul 7, 2026; first Apr 20](https://arxiv.org/html/2604.18419v6), §3.2; §4 oracle assumption; §5 opening; §7 | Studies value-thresholded **abstention**, with oracle-value guarantees and an estimated hidden-state probe in practice. Abstaining on an unpromising reasoning trace is not successfully finishing a state-changing task. Do not award a cheap TextCraft failure as though it were successful termination. A public goal-sufficiency check can supply our finish labels without a learned value oracle. |
| [Guided Policy Optimization, v2, Mar 13, 2026; first May 21, 2025](https://arxiv.org/html/2505.15418v2), §§2.2,3.1–3.3,4.4–5 | Co-trains and constrains a privileged guider toward an imitable learner; TigerDoor shows why a teacher that never gathers information can be unsuitable. The ideal mirror-descent claim assumes its guide/projection conditions, not arbitrary SFT. This motivates learner-reachable, public-history supervision, but does not establish our teacher-history intervention as novel. |
| [Recursive Agent Optimization, v1, May 7, 2026](https://arxiv.org/html/2605.06639v1), §§2.2,3.1, Appendix A.2–A.4 | Uses node-local success, root-group leave-one-tree-out baselines and depth balancing. Crucially, **TextCraft uses delegation bonus λ=0**; λ=0.4 belongs to Oolong. Existing helper uptake is not a reason to add a spawn bonus. Local child completion versus root utility under shared stock/budgets is the useful unresolved contrast, not recursion itself. |

This is a targeted primary-source review, not an exhaustive novelty search. No absence of a search hit establishes originality; no paper result above was reproduced here.

## Candidate 1 — Teach recovery on states the actor actually visits

Question: are repeated public errors and late finishing mainly missing supervision on learner-state histories, rather than a need for another reward penalty?

From the **first fresh-A binder collection only**, freeze at most 48 distinct prefixes by a prespecified task/call order, at most two per task/category: post-schema/stock-error, publicly sufficient root stock, and near-goal insufficient stock. Preserve full original histories, initial scoring inventory and remaining budgets. A public-only teacher supplies one next action using only already disclosed recipes/inventory/goals; unknown recipe details require a visible `get_info`, never hidden-world routing. Strictly validate the proposed action natively on a fork. Do not relabel sampled actions or call this on-policy RL. Empty/unsupported categories are reported, not filled from B.

From the same public cp23, compare one small recovery-SFT update with one ordinary-public-teacher SFT update, prospectively matched for optimizer/LR, action-type mix and supervised target-token dose within one complete example. Reuse unchanged warm B; evaluate both new endpoints on the fixed 16 B attempts through binder. This is a **data intervention**, unlike the already queued extra-SFT step on all 366 original demonstrations; unequal task/history distributions remain an explicit limitation.

Before fitting, probe a frozen 24-prefix TRAIN diagnostic set with four next-action samples per actor: valid finish when sufficient, false finish when insufficient, repeated error and executed-action correctness. Same complete input IDs make this interpretable; trajectory-level entropy comparisons do not. Never fit or select on B outcomes.

Estimate: 1.5–2.5 A100 hours including two small fits, same-prefix sampling and two readouts; prospective cap 3 hours. **Go** only with at least 24 valid distinct training prefixes across six A roots. Promote only if recovery beats both warm and dose-control by at least 2/16 paired B slots across two roots, or preserves every warm success while cutting calls at least 15%, with improved TRAIN recovery diagnostics. These are triage thresholds, not significance tests. **No-go:** only NLL improves, false finishes increase, control matches it, or a deterministic public finish check explains all savings. Novelty is low; usefulness is an inexpensive diagnosis of the remaining learning bottleneck.

## Candidate 2 — Root reward versus child-local credit, only after feasibility

Question: with shared inventory and a fixed global budget, does locally successful child work improve the root, or train an easy subgoal that the root cannot use?

First require completed flat/recursive root outcomes on a prospectively fixed TRAIN panel under the full existing 96-call/8192-output budget. Retain the same observed-recipe hints in flat and recursive arms, permit at most one depth-1 child, and charge all calls/context. The child is optional: required-delegate admission is not evidence of voluntary routing. Stop before training if no voluntary uptake, no root advantage/heterogeneity, or the flat public-hint control suffices. A 16-pair feasibility readout should take roughly 1–2 hours, cap 3; this is the first decision, not a long recursive campaign.

If admitted, collect one new 8-task ×4-tree on-policy batch. Fork the same initial weights into **root-only credit** versus **node-local native-success credit**, without a delegation bonus. Both use the same original per-call token prefixes, token-sum normalization /32, one fresh AdamW step and the other three trees' root-success baseline. Only the reward assigned to each node changes: root score for every node versus its own net-growth score. This is a RAO-inspired reward ablation, **not** a reproduction of its depth-weighted optimizer or an unbiased estimator of one shared objective. No forced child string becomes a sampled likelihood target.

Require at least two mixed root-reward groups and record child-success/root-failure cases before either update. Read warm and both endpoints on a fixed, readout-only diagnostic under the same recursive interface; never compare recursive updated scores only to flat warm scores. Additional training/readout estimate 4–6 hours, cap 8, after feasibility. Promote only a replicated root-success gain beyond the unchanged actor; retire local credit if it raises child scores/spawns while lowering root success or wasting stock. A learned router is premature if the simple hint/flat policy matches it. This extends the earlier resource-boundary proposal, not a new claim that local rewards or recursion are original.

## What fixed-endpoint B transfer changes next

- **Both actors improve under common interfaces:** first replicate the collection/update seed and finish fresh-A plus extra-SFT controls. Generalization beyond familiar roots becomes plausible; it does not prove RL-specific superiority. Candidate 1 then addresses measured cost/recovery, not a missing success signal.
- **Only binder execution retains improvement:** call it interface-conditional transfer. Keep raw crossed readouts; prioritize compact/mask results and recovery training within that interface, not a generic planner claim.
- **Neither transfers, with warm B below ceiling:** familiar-goal adaptation is the leading interpretation. The already accepted first fresh-A cycle is still useful. Analyze it before proposing further scaling or root-credit machinery; do not select a favorable B checkpoint or silently alter the four-stage plan.
- **Warm B is near ceiling, cells incomplete, or effects concentrated in one root:** the probe is inconclusive, not a negative RL result. Report all cells, paired roots, calls and errors; use a separately frozen harder diagnostic only in a later experiment, never optimize on B.

Planning estimates are based on this campaign's small fits and 25–50-minute B readouts, not paper hardware timings. Queue descriptors, frozen transfer code and all accepted sources remain untouched.
