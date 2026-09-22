---
status: completed_cpu_audit
evidence_date: 2026-09-22
question: What behavior changed on the fresh crafting goals, and what still fails?
claim_strength: descriptive_correlates_not_isolated_causal_mechanisms
---

# Fresh TextCraft behavior audit

This is a descriptive audit of the completed frozen fresh007 comparison, not a new outcome
selection or a mechanism experiment. The primary result remains all 16 root tasks × two seeds:
the public-information teacher package won 15/32 versus 1/32 for the old privileged-teacher
package (14 paired wins, no losses; parent-bootstrap CI +25 to +62.5 points).

## Behavior across the full primary panel

| Saved native behavior | Old privileged teacher | Public-information teacher |
|---|---:|---:|
| First valid query asks the root | 5/32 | 32/32 |
| Ever queries the root | 14/32 | 32/32 |
| Queries containing a nonexistent item | 586 | 3 |
| Repeated nonexistent-item mentions | 468 | 1 |
| Requeries an earlier returned static recipe | 174 calls | 124 calls |
| Native action errors | 119 | 550 |
| Successes | 1/32 | 15/32 |

Thus the public package is strongly associated with root-first discovery and almost eliminates
nonexistent-name querying. It does **not** eliminate requerying returned recipe facts, and its
larger action-error count is not evidence of worse overall execution by itself: it made 1,206
calls versus 1,340 and attempted substantially more crafts (797 versus 259 in the authoritative
report). These are trajectory correlates, not proof that a specific query behavior caused success.

The 550 public craft errors are native feedback categories: 220 insufficient-inventory, 216
extra-ingredient, 68 missing-required-ingredient, 35 wrong-amount, one no-recipe, and 10 other.
Main inspected all ten feedback strings in the last category: each requests an output count
that is not a multiple of the recipe's batch size. No additional unknown error family was found.
They point to feasibility/quantity accounting after discovery, rather than a remaining
nonexistent-name problem.

## Where failures remain

Public teacher outcomes decline with task depth: depth 2 has 7/8 successes and no action errors;
depth 3 has 5/8 successes and 55 action errors; depth 4 has 3/16 successes, 13 failures, 495
action errors, and 991 calls. All 16 depth-four episodes still queried their root first. The
largest remaining bottleneck is therefore executing multi-step recipes and quantities after a
plausible discovery start—not demonstrated evidence that another memory prompt or recursive child
would fix it.

For example, the two public `textcraft_synth.val.285` depth-four runs both queried the root first
and never queried a nonexistent name, yet failed with 30 and 36 native action errors respectively.
This is an illustrative retained failure, not a hand-scored diagnosis of all deep episodes.

## Predeclared identifier-exposure supplement

`TEXTCRAFT-FRESH-SCOPE.md` was completed before paired outcomes were opened. Its three exact
training-intermediate root items are retained in the primary panel. On the remaining 13 roots ×
two seeds, whose exact root identifiers were absent from both local SFT prompts/targets, old has
1/26 successes and public has 11/26. This supports that the primary improvement is not confined to
those three identifiers, but is supplementary: shared prerequisites and the recipe world remain,
and it is not a pretraining-clean or independent fresh panel.

## Provenance

Machine-readable receipt:
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/analysis-textcraft-fresh-behavior-002.json`
(SHA-256 `00e9dfa25150602ed8ca188b68e111ebbdc321fc98519428b9c1b2d0deae9986`). It binds the authoritative
fresh result `analysis-textcraft-fresh-001.json` (SHA-256
`574b545bfb6bd7b00c3e83758d7bc4007464afca86ca9a958d4679f96995e480`), the scope document, and every
used raw receipt through the completed native audit's per-arm hash maps. Analysis code:
`audit_textcraft_fresh_behavior.py`; focused checks cover the full-primary versus supplemental
split and the native craft-error classifier. Main ran both focused tests and independently
reran the complete audit to report003, which is byte-identical to report002. Main also verified
that all 32 public episodes have the root query as their very first action, not merely as the
first query after some earlier non-query action.
