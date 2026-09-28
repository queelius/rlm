# Remaining depth4/5 failures: public inventory bookkeeping

Cutoff: 2026-09-28; discovery-trained actors only, completed breadth panels00–05.
`corrected` known-recipe actors and unmatched panel06 are excluded.

The strongest next probe is **computed quantity bookkeeping**, not another recipe notebook or
craft validator. This is an exploratory mechanism hypothesis, not a demonstrated remedy.
Recipe discovery remains a substantial, separate bottleneck.

## Selection and evidence

Before examining outcomes, the primary slice was fixed to panel00/world42/original training
seed2026092208, all four declared depth4/5 task identities, both repeats: eight attempts.
Secondary broadening includes all 24 deep task identities in panels00–05, four recipe worlds,
two training seeds and two repeats: 384 attempts per execution arm. Worlds, seeds and repeats
are correlated realizations of those 24 identities, not 384 independent problems.

Every selected node and episode matches its existing native audit hash; task and PLAN hashes
also match. There are 96 audit receipts and 1,752 hashed receipt/task files. Final inventory
was independently reconstructed from executed crafts and checked against each audited node;
queried current quantities were checked chronologically. No hidden native recipe graph, model
weights, or full prompt files were read by the analysis. Binder history contains executed
actions; the separately saved originally requested ingredients are not mistaken for execution.

## Counts

| Diagnostic | Primary binder | All deep binder | All deep raw |
|---|---:|---:|---:|
| Native success | 2/8 | 76/384 | 38/384 |
| Failures with complete observed recipe closure | 6/6 | 152/308 | 156/346 |
| Failures with a publicly feasible completion | 6/6 | 147/308 | 152/346 |
| Failures leaving a craftable needed prerequisite | 5/6 | 277/308 | 334/346 |
| Failures leaving a goal-completing root craft ready | 1/6 | 13/308 | 5/346 |
| Episodes repeating an identical stock-deficient craft at unchanged stock | 7/8 | 247/384 | 199/384 |
| Duplicate stock-error calls, excluding first occurrence | 38/125 stock errors | 1,672/4,641 | 1,121/2,789 |
| Queries returning only previously seen static facts | 11/119 queries | 1,189/6,920 | 1,685/7,126 |
| Context-cap endings | 0/8 | 32/384 | 48/384 |

The remaining binder failures are not all recipe-complete: 146 have both closure and publicly
feasible completion, 155 have neither, six have closure but resource deficits, and one can
complete using current stock despite unknown subrecipes. Thus missing recipe information is
still present in 156/308 failures. Public feasibility is a constructive arithmetic statement
about available recipes and stock, not evidence that the model understood or retained them.

For binder, 222/308 failed attempts contain a repeated stock error. All 24 task identities
have at least one such failure, one publicly feasible failure, and one failure with an available
needed prerequisite. Exploratory 95% task-cluster bootstrap intervals are 67.6–76.3%,
39.5–56.2%, and 85.4–93.7%, respectively. They describe this reused task population, not a
prospective intervention effect.

| Binder depth | Success | Failed but publicly feasible | Failed with ready prerequisite | Context caps |
|---|---:|---:|---:|---:|
| 4 | 39/96 | 40/57 | 47/57 | 0/96 |
| 5 | 37/288 | 107/251 | 230/251 | 32/288 |

There are 1,292 consecutive repeated stock-error calls; the longest identical consecutive run
is 23. “Same stock and executed craft” does **not** mean the same policy state: full history,
recipe knowledge and remaining budget may change. Of the 4,641 stock errors, 1,491 occur after
complete recipe closure, and 4,198 while some needed prerequisite is currently craftable.

Most failures explicitly finish: 276/308 binder failures versus 32 context caps. All caps are
depth5; 31/32 also contain stock-error repetition. This association does not establish that
loops caused caps. Caps are 8,192 input-plus-output tokens with no truncation; the call and
cumulative generated-token budgets remain 96 and 8,192. Original versus second training seed
has 29 versus three caps, so pooled cap rates also hide substantial actor differences.

## Shared ingredients and resource loss

There are 2,978 successful consumptions of publicly known shared ingredients. A later shortage
at another already-known parent, without intervening replenishment of that ingredient, occurs
on 355 error calls in 75/384 episodes (72/308 failures). This is a temporal consumption witness,
not proof that the earlier craft was wrong: both branches may be needed and more stock may
still be producible.

Only four successful crafts change public completion feasibility from true to false with
the same observed recipes. All four are failed episodes with complete recipe closure.
Two additional closed failures were already infeasible when their recipes became visible.
Successful production exceeds current public batch-rounded remaining demand on 381/5,372
successful craft calls, across 111/384 attempts; excess production is not necessarily harmful.

These observations do not support treating destructive exhaustion as the dominant observed
failure mode. They also do not rule out exhaustion before recipe discovery: missing recipes
prevent that diagnosis. We never complete the graph using an oracle.

## Representative primary traces

Literal synthetic names and zero-based saved call IDs below; examples were chosen after the
primary-slice counts, not used to define the analysis population. Exact paths, hashes and
selected actions/feedback are in `FINDING.json`.

- `.628`, repeat1: `c040–c044` repeat the same craft of `c8_i3_18`; every response says
  `c0_i1_20: need 2, have 1`. Its recipe had been returned at `c005`: one `c2_ore` produces
  three `c0_i1_20`. Three `c2_ore` remain. It finishes at `c045` without replenishing it.
- `.599`, repeat1: `c000` reveals that one `o9_i3_19` plus one `o8_i4` produces two
  `o0_i5_20`, against a goal of one. After successful `c032`, stock is one and two,
  respectively. `c033` finishes without crafting the root. This needs no hidden graph.
- `.401`, repeat0: `c003` reveals that `a5_i2:1, a8_i2:1, a3_i2:2` produces two `a1_i4`.
  At the final failed root craft, stock is 3, 2 and 6, respectively, while `a1_i4` is absent.
  The actor finishes with that immediately executable prerequisite still uncrafted.

Supplemental resource-loss example outside the primary slice: panel01/world52/original,
`.147` repeat0, `c054` consumes one `c4_ore` to produce four `c5_i1` when the latter's
remaining public demand is zero. Public completion feasibility changes to false with a
one-unit `c4_ore` deficit. It is one of four witnessed transitions, not the typical failure.

## Historical controls and smallest next experiment

The earlier [recipe-memory factorial](../TEXTCRAFT-MEMORY-READOUT-PLAN.md) already tested a
fact-only notebook with full/recent-four history. Full-history success went 15→14/32 for
the original actor, 15→6/32 for the second seed, and 14→14/32 for the RL actor. Those arms
already clarified that `can_craft` means recipe existence, not stock sufficiency. Neither
another notebook nor that sentence is a new mechanism experiment. The earlier
[failure audit](../TEXTCRAFT-REMAINING-FAILURES.md) proposed—but did not establish—a separate
derived quantity ledger. [Repeated-state work](../TEXTCRAFT-REPEATED-STATE-CREDIT-FEASIBILITY.md)
also explicitly rejects inventory-only state identity.

Proposed and now authorized for CPU preparation: two arms, eight attempts each, on the same
already-exposed primary deep slice, using fresh paired repeat seeds2026092804/2026092805.
Same frozen discovery actor, binder, full history, schema, native semantics and budgets:

- Both display the same alphabetically ordered public fact table and identical instructions.
- Treatment adds host-computed outstanding goal demand, inventory deficit and batch counts;
  sum shared-child demand before batch rounding. Unobserved recipes remain explicitly unknown.
- Control masks computed columns and pads its serialized prompt to the same token length at
  that current state. Padding and all displayed tokens count against the unchanged context cap.
- No action recommendations, priority ordering, extra recipe queries, inventory reservations,
  automatic crafts or model-output repair beyond the existing binder.

This is **host computational assistance**, not pure memory. Prompt presentation and arithmetic
are not fully isolated; equal state-conditional input length is not equal total trajectory cost.
Primary endpoint is paired native root completion, with all planned slots accounted for.
Secondary endpoints are repeated stock errors, root-ready failed finish, calls, actual
input/output tokens, context caps and wall time. Unknown outcomes remain unknown.

The primary historical slice averaged 124 seconds/attempt: 16 attempts imply 33 GPU minutes
before added prompt overhead. Estimate 35–55 minutes total, capped at 45 minutes per arm.
Any improvement needs an unread task-panel and second-actor replication. Do not expand solely
because repeated errors decline without better completion. No GPU job was launched here.

## Reproduce

`analyze.py` checks receipts and reconstructs all 768 deep traces; `report.py` produces compact
`FINDING.json` and task-cluster intervals. Rich artifacts are external at
`R/analysis-inventory-bottleneck-20260928/{SUMMARY,DETAILS}.json` (265 MiB detailed trajectories).
Scripts refuse to overwrite existing outputs:

```bash
python experiments/selective_delegation/inventory_bottleneck_20260928/analyze.py \
  --output /absolute/new/analysis-directory
python experiments/selective_delegation/inventory_bottleneck_20260928/report.py \
  --analysis /absolute/new/analysis-directory --output /absolute/new/FINDING.json
python -m pytest -q experiments/selective_delegation/inventory_bottleneck_20260928/test_public.py
```

Five focused tests cover shared-demand rounding, missing-recipe visibility, stock-based root
completion without subrecipes, repeated-stock versus full-state identity, and executed-versus-
requested binder actions. Source and artifact hashes are in `FINDING.json`.
