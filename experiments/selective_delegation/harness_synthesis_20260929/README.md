---
title: "Harness results: arithmetic display, action representation, and real child uptake"
date: 2026-09-29
cutoff_utc: "2026-09-29T01:04:17Z"
status: exploratory_native_evidence
scope: CPU-only synthesis; no model execution or sealed-source changes
research_store: /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921
machine_readable: SYNTHESIS.json
---

# Decision

Prioritize a six-root, complete-goal delegation comparison. Relaxing query order admitted real helpers, but the 32-response screening limit censored every root. Do not expand the quantity-table pilot or ALFWorld representation grid from these results.

## Native outcomes and cost

| Comparison | Native successes | Calls | Output tokens | Native service seconds |
| --- | ---: | ---: | ---: | ---: |
| TextCraft masked arithmetic | 0/8 | 413 | 13,055 | 1,134.1 |
| TextCraft demand arithmetic | 0/8 | 418 | 15,686 | 1,327.7 |
| ALF index base | 1/24 | 1,186 | 10,406 | 503.5 |
| ALF index SFT | 2/24 | 1,122 | 8,882 | 773.1 |
| ALF command base | 0/24 | 838 | 8,952 | 409.4 |
| ALF command SFT | 4/24 | 417 | 4,344 | 343.5 |

Every listed outcome is observed and independently native-audited; transport failures are zero. Calls, generated tokens and service time are distinct costs. Model loading/CPU audits are not included in service time.

### Public arithmetic: valid negative completion result

The paired [quantity report](../inventory_bottleneck_20260928/README.md) uses the same discovery checkpoint23, observed-recipe binder, four exposed depth4/5 roots (.628/.599/.401/.294), world42 and two seeds2026092804/05. Both arms retain full history and a fact table; only computed demand/deficit/batch columns are masked in the control. Same-state input lengths are exactly padded to match. The calculation cannot inspect unobserved recipes or select actions.

Demand minus masked has eight ties, zero wins/losses. Its displayed bootstrap [0,0] is a degenerate four-root sample, **not** a population no-effect bound. Demand reduced unchanged recipe re-queries35→9 and above-demand crafts20→8, but stock errors168→173 and repeated identical-action/stock failures90→106 did not improve. Demand cost +20.2% output tokens and +17.1% service time. Six versus five context caps and two versus three failed explicit finishes are valid outcomes under the unchanged limits, not transport invalidity.

Concrete failure: [.599 repeat1, calls32–41](</project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/textcraft-public-demand-20260928-001/demand/nodes/t05-r1-flat-original-n0.json>) repeats the same failed craft ten times: produce3 o0_i2 using raw_o9, with none remaining. The [actual request](</project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/textcraft-public-demand-20260928-001/demand/calls/t05-r1-flat-original-c032.json>) already displays o0_i2 stock3, remaining demand1, deficit0 and new batches0. Thus blindly delegating every reported shortage could amplify an obsolete craft; the problem is not merely absent arithmetic.

Budget:96 shared calls,8192 generated tokens,256 per response,8192 full context;45 minutes/arm. Actual owners took22.6/19.4 minutes. No scaling of this display is justified. [Authoritative paired evidence](</project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/textcraft-public-demand-20260928-001/PAIRED.json>).

### Action representation: narrow learning-package signal, substantial legal-action failures

The four [ALF cells](</project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/alfworld-representation-20260928-001/COMPARISON.json>) use12 prospectively new valid_unseen games, two seeds2026092813/14 and only four shared scenes. Index and exact-command SFT use the same524 successful TRAIN rows/native commands, fit seed2026092200 and33 updates, but command targets contain5362 versus4060 tokens (+32.1%); token-mean example weights also change. Both show the same current indexed admissible list and full public history. Exact-command matching does not repair wording.

Index learning gain:2 wins/1 loss (+4.17pp; game-cluster95% −12.5..25pp). Command gain:4 wins/0 losses (+16.67pp;0..41.67pp). Interaction:+12.5pp (game0..33.33pp;scene0..27.78pp). This is exploratory package evidence, not isolated symbol binding, robust superiority or hierarchy.

All four command-SFT successes are games5 and8, both scene10, at both seeds. In [game5](</project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/alfworld-representation-20260928-001/command-trained/episodes/game-05-seed-2026092813-flat.json>), the actor initially places an unclean egg, then finds another egg, cleans it and places it in the microwave: native success in19 actions/20 responses. Game8 heats an apple and returns it to the fridge in10 actions.

Counterevidence: command SFT has73 invalid outputs versus54 base; all127 are syntactically valid command JSON whose literal command is absent from the CURRENT admissible list. In [game0 call009](</project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/alfworld-representation-20260928-001/command-trained/calls/game-00-seed-2026092813-flat-call-009-flat.json>), it repeats “go to drawer3” although only “examine drawer3” is available for that object. Strict rejection is correct. Twenty of24 trained-command episodes terminate after three invalid outputs; fewer calls cannot be called efficiency dominance. Index SFT removes immediate repeated commands571→4 but still wins only2/24; fewer loops are insufficient.

Budget:50 native actions/2048 generated tokens/128 per response/8192 full context,24 minutes/cell; no history trimmed, no unknowns. Keep these results; do not broaden the grid from4/24.

### Delegation: uptake demonstrated, root efficacy still unknown

| Relaxed-query screen | Returned responses | Real child responses | Output tokens | Recorded child result |
| --- | ---: | ---: | ---: | --- |
| Flat | 32 | 0 | 906 | none |
| Fixed | 32 | 9 | 691 | +2 a2_i2_12; local score1 |
| Adaptive | 32 | 13 | 650 | +2 a8_i2_18 versus target4; local score0 |

All use exposed val494/world42/seed2026092806, publiccp23 plus binder. All three hit the planned32-response admission ceiling; all root scores are **unknown**, not0/1. The [fixed audit](</project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/textcraft-decomp-flexible-fixed-20260928-001/ADMISSION-AUDIT.json>) and [adaptive audit](</project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/textcraft-decomp-flexible-adaptive-20260928-001/ADMISSION-AUDIT.json>) authenticate saved requests/tokens and native bindings, but explicitly do not fully replay interrupted root transitions. Child scores above are saved native records, not an independent full-root replay result.

The fixed child produces its requested2 units using one raw_a1. The adaptive child finishes after producing only2 of4 requested a8_i2_18, consuming3 raw_a0. Parent resumes from that same shared inventory. There is no physical stock reservation; child-entry snapshots define net production and root scoring retains the original snapshot. No hidden dependency depth drives routing: its depth is a lower bound from observed recipes. This is deterministic harness routing, not trained recursion.

The original V1 query-order admission failure is no longer the relevant barrier. The revised screen has real uptake; the unanswered question is whether that work helps the root within the common budget.

## Ranked follow-ups: at most two

1. **Complete-goal delegation; selected for CPU preparation.** Same original val494/world42, flat/fixed/adaptive, two fresh paired seeds: six roots, all newly run. Keep publiccp23+binder, one child and original96-call/8192-output/8192-context limits. Remove the admission-only32-response ceiling; retain strict schemas, actual native errors and charged responses. No hidden recipes, forced successful actions or stock repair. Run to native finish or ordinary resource termination; genuine owner/transport failures remain unknown. Report every arm, native root success, child-local success, root/child calls and prompt/output tokens separately. A CPU public-only construction takes23 actions, so96 is feasible headroom, not a guarantee of model success.

   Reuse the existing collector and recursive auditor privately. The missing seam is small: the admission wrapper audits only jobs[0] and interrupted roots take a receipt-only branch. The new wrapper must audit BOTH jobs with full native transition replay, shared inventory, original sampled payloads versus binder-executed actions, root–child call order and return feedback. Existing completed scripted fixtures already exercise native child replay; add a saved-screen-request/two-job fixture. No host-spawn architecture is needed. Expected15–30 GPU minutes; conservative45–60; hard science cap20 minutes/arm=60 minutes, separate owner/audit margins. Do not scale without full outcomes. Flat ties at lower cost or local child success without root benefit should narrow the next question, not justify extra depth. One root/two seeds remains a mechanism diagnostic.

2. **Conditional reactive deficit child versus identical hint; defer.** Only if the complete-goal run exposes a relevant recoverable stock failure, compare public arithmetic/hint-only with the same hint plus one bounded local child. Select only goal-relevant positive net shortage from observed recipes; reject obsolete/zero-deficit targets like .599. Preserve shared resources and global budgets. This would need explicit host-spawn event provenance and public-fact handoff in collector/auditor; never fabricate sampled delegate tokens. A later fixed eight-slot exposed deep panel could screen it in45–90 GPU minutes. Retire if eligibility is absent, hints match completion more cheaply, or local completion consumes stock without root gains. Not queued or implemented by this synthesis.

## Provenance and scope

[SYNTHESIS.json](SYNTHESIS.json) includes exact source and evidence hashes, cell counts, concrete witnesses and inference limits.62 unique pinned source files matched; primary reports/native-audit/PLAN bindings and16 quantity episode hashes matched. The ALF strict-invalid census was recomputed from saved requests/responses. No model, native continuation, existing analyzer entry point, GPU job or sealed source was executed/changed; no weights were rehashed. Existing native audits remain authoritative. The brainstorming skill kept proposals bounded and separated observed results from capability hypotheses. Advisor-story implication: “real helper uptake, no root efficacy yet”; “arithmetic display alone did not rescue deep tasks.” No deck or queue edits here.

