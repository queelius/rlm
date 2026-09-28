# Reactive subgoals: conditional harness test, not learned decomposition

September 28, 2026. Design only; no implementation, GPU work or acceptance. Prioritize only if
[V2](decomposition_flexible_20260928/PLAN.md) reaches eligible boundaries without helper uptake,
and the [quantity-table comparison](inventory_bottleneck_20260928/README.md) leaves a meaningful
native-completion deficit. No helper because no candidate is not an emission failure.

The motivation is concrete: the [stock audit](inventory_bottleneck_20260928/FINDING.md) finds
repeated stock errors in 222/308 deep binder failures. In `.628`, five identical failed crafts
need two units but have one; an already observed recipe can replenish the shortage. This
motivates local recovery, not a claim that recursion is necessary.

## Minimal contrast

Use the same frozen discovery checkpoint for every root/child; no new `delegate` response is
required. Compare binder-flat, binder plus a public missing-ingredient hint, and the identical
hint plus host-triggered child. Primary effect: child minus hint in native root completion;
hint minus flat measures public arithmetic/recommendation assistance. Reuse the same eight
predeclared deep task/seed slots across arms, explicitly exploratory. Report trigger eligibility
for every slot, including zero-trigger cases; do not select only recoverable failures.

After the first actual native stock-deficiency error, retain and charge the failed response.
Require an observed recipe for the failed craft and public evidence that its target supports
the outstanding root goal. Do not trigger on obsolete/over-requested crafts or known uncraftable/base
shortages; otherwise error-triggering could amplify a wrong plan. Select one missing ingredient
by a fixed lexical rule. For required stock `n` and current stock `s`, delegate net production
`d=max(0,n-s)`, **not n**. The hint arm receives the same item, arithmetic and proposed subgoal.
No hidden recipe lookup, free query, automatic craft or automatic parent retry.

## Reuse and genuine seams

Existing [Frame.delegate](textcraft_bridge.py) already shares inventory and the global Budget,
snapshots child-start stock, and applies native net-growth checks. Reuse the same model client
sequentially; give the child a normal flat-action prompt with a local textual goal and targets.
Parent root targets and its original scoring snapshot remain unchanged.

This is not a Frame-only patch: the sealed collector enters its nested execution loop only
after a sampled `delegate`; replay makes the same assumption. A later additive adapter needs
an explicit **host-spawn event**, linked to the unchanged failed craft, and matching audit
support. Never relabel sampled tokens or invent a model delegation call. Binder lookup is
node-local, so inherited parent recipes need provenance-aware public-fact handoff, visible to
hint control and child; do not fabricate child queries. Charge transferred prompt tokens.

## Resources, stopping and interpretation

Child success can consume stock needed by the parent or siblings. Show the pending parent
craft's other ingredient commitments and root goal in both hint/child contexts; decline known
public stock conflicts. Keep native inventory shared: no hidden stock partition, rollback or
child-only craft veto. Informational reservations cannot guarantee global feasibility when
recipes remain unknown; measure consumed resources and local-success/root-failure separately.

Permit one child, no grandchildren; cap it at 16 responses/2,048 generated tokens while
reserving at least four responses/1,024 tokens for the parent within the unchanged 96-call/
8,192-token episode budget. Return once on repeated unchanged-stock failure or child cutoff;
preserve status/inventory and let the parent decide. Native finish remains required.
Keep 8,192-token contexts untruncated.

Retire if the hint/table matches completion more cheaply, or children merely reduce loops
without root gains. Ordinary public arithmetic may already solve the motivating cases.
[RAO already studies recursive TextCraft agents](LITERATURE-OPPORTUNITIES-20260928.md);
[RLM context-isolation results](LITERATURE-CONTEXT-ISOLATION-20260922.md) require adequate retained
state. Shared resource commitments make that qualification material. This fixed error router
tests goal reframing/context packaging, not learned decomposition or a new recursive method.
Cached primary-source reviews suffice; no additional paper check was needed.
