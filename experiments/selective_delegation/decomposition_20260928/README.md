---
question_id: OP28-2-admission
date_utc: 2026-09-28
status: cpu_qualified_parent_gpu_launch_only
scientific_model_calls: 0
novelty: diagnostic_only_not_trained_recursion
---

# Can the trained flat actor actually use one worthwhile helper?

Ready: **one fresh task × flat/fixed/adaptive, at most96 real responses total**.
This is an admission screen, not a recursion-benefit result or a new architecture.
The earlier [pilot](../TEXTCRAFT-PILOT-FINDINGS.md) produced zero delegates/child
calls; all366 public-discovery SFT targets are flat actions
([inspection](../TEXTCRAFT-TRAINING-CONTROLS.md)). Merely allowing depth2 would
repeat an uptake test without addressing its likely format/selection bottleneck.

The falsifiable first question is whether concrete, public-state-derived delegate
instructions produce a valid helper that can act and return under the existing
shared budget. If admitted, the next question is whether skipping trivial or
resource-coupled subgoals helps root success per total cost. RAO already studies
Qwen3-4B/TextCraft recursion; this screen has no standalone method-novelty claim.
See [the literature opportunity](../LITERATURE-OPPORTUNITIES-20260928.md).

## Paired contract

| Arm | One-boundary rule |
|---|---|
| Flat | Keep the public candidate in the parent; no helper permitted. |
| Fixed | Require one helper for the first lexical missing direct prerequisite with an observed recipe. |
| Adaptive | Use that same candidate only if public branching, remaining budget and known shared-stock checks admit it; otherwise continue in the parent and reconsider after actual actions. |

Every arm uses the same public-discovery checkpoint23, world42, original task
bytes, sampling/seed and public candidate facts. All perform the same mandatory
public discovery prefix: query the root and each missing immediate prerequisite.
These are actual actor-emitted `get_info` actions, **not free host queries**.
Each delegation must also be emitted by the model; the harness does not fabricate it.
Flat receives the same subgoal hint, controlling for the information in that hint.

Adaptive routing uses only recipes returned to that frame, current inventory and
remaining budget. Let `d` be depth in the **partially observed** recipe graph and
`u` the number of inputs still short. Require `u >= 2`, no observed sibling-input
stock shortage, at least `8 + 4*d + 2*u` calls and `max(1024, 256*(2+d))` output
tokens remaining. These are declared heuristics, not calibrated cost estimates.
Unknown dependencies are not evidence of independence. The native depth field
is not read by the routing rule. This is **oracle-free deterministic harness
routing, not a learned decomposition policy**.

The unchanged native episode uses one client and shared96-call/8192-output-token
budget; each response is capped256, and input+cap8192 without truncation. The
admission wrapper stops before a next request after32 returned responses per arm
or two required-instruction rejections, including schema rejections. All root,
child, delegate, finish and error responses count. No retry is free. One helper
boundary only; no grandchildren. Partial admission stops are unknown root
outcomes, never substituted reward0.

**Native inventory is shared without stock reservation.** Each child snapshots
inventory at entry and must produce its requested quantity in addition to that
snapshot. The root retains its original baseline. Delegated targets are the
missing quantity, not the total parent requirement. No stock, target or quantity
repair is introduced; the existing public-recipe ingredient binder is used in all
arms. Child stock deltas and root/child errors are reported separately. Parent
returns are unchanged mechanical status/message/inventory, not free model summaries.

## Concrete task and example

Official `textcraft_synth.val.494`: produce2 `a8_i4_18`. Selection seed2026092805
chose the first SHA256(seed:ID) among12 unused declared-depth4 rows/four roots,
excluding every current selected task inventory (including future breadth/HOLDOUT)
and all official TRAIN roots. No fresh depth5 root remained. No model outcome or
native success filtering/replacement occurred; original JSONL bytes are preserved.

After four real public queries, `a2_i2_12:2` is the first prerequisite. Its
observed recipe can already run using one `raw_a1` in stock. Fixed requests:

```json
{"action":"delegate","targets":{"a2_i2_12":2},"context":"Craft this prerequisite using public recipes; finish when its net target is met."}
```

Adaptive initially keeps that trivial step in the parent. In the CPU public-action
witness, it later admits `a8_i2_18:4`, requiring `a3_i1_13:2` and `a6_i1_16:4`,
both then missing; the partially observed depth is2. This is a concrete expected
policy distinction, not a claim about what the model will do.

## Stop/promote rules and interpretation

- If both delegation arms produce zero child calls or reject their instructions
  twice, stop this branch. Do not expand a flat/interface grid. Diagnose schema
  refusal versus a successful delegate followed by failed child execution from
  saved requests. A tiny TRAIN-only delegation-demonstration dose would be a
  separate, newly specified experiment, not an automatic continuation.
- One valid child with returned responses establishes admission, not benefit.
  Child success is never the headline: retain root success, child consumption,
  all calls/tokens/service time, and unknowns. A successful child with worse root
  outcome or higher cost is evidence against that boundary on this task.
- Only if admitted should a further fixed fresh-root panel test value, with
  complete96-call budgets and paired flat/fixed/adaptive seeds. Use the remaining
  unused depth4 roots under a new frozen selection; do not silently reuse protected
  depth5 roots. Retire this deterministic gate if it repeatedly creates extra
  work without root/cost benefit. One task/seed cannot establish efficacy or CIs.

## Ready commands and immutable inputs

`P`, `R`, `E` below are task-specific variables; run the three scientific commands
serially under the parent's one-GPU owner, never concurrently. They reuse the
existing collector's GPU lock, allocation margin and first-real-response check.
Each has a15-minute hard owner cap; estimated useful cost is roughly2–5minutes per
arm (6–15minutes total) on one A100, with no optimizer or checkpoint creation.
This is an estimate; inspect the first real response within90seconds.

```bash
P=/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python
R=/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921
E=/project/alex_phd/repos/rlm/.worktrees/selective-delegation-20260921/experiments/selective_delegation/decomposition_20260928
"$P" "$E/run.py" --mode flat --output "$R/textcraft-decomp-flat-20260928-001"
"$P" "$E/run.py" --mode fixed --output "$R/textcraft-decomp-fixed-20260928-001"
"$P" "$E/run.py" --mode adaptive --output "$R/textcraft-decomp-adaptive-20260928-001"
```

After each owner terminates, CPU-audit with the same command plus `--audit`.
The three PLANs already exist from `--prepare-only`; launch without that flag.
They contain complete source, native revision/license, world, endpoint and data
pins. No accepted/live source was edited. Dataset and fixtures are under
`R/textcraft-decomposition-admission-20260928-001`.

| Artifact | SHA256 |
|---|---|
| Data MANIFEST | `03587d44e0822c4501899f064b29a42c343fc3d8b7d85987eb869e158ab554bd` |
| Original task bytes | `18ce889c74beee362abb51ea43df29b78d3daf48b082204df7b2fa122b0c5f6d` |
| Flat PLAN | `46d18e91871a5204119ecf82496579c302ebdf1063ee3d2383db76c047ce01c0` |
| Fixed PLAN | `d898740a97de568557b0c2632135d4ddb868ce8322952d206098c5198b476615` |
| Adaptive PLAN | `242fdd052affc43842e128d669bb1b805e66c22fc0ec27a3f11e56437b275153` |
| CPU runtime VERIFICATION | `6e42e347bc8235c57ac0b35aadb55bf25b611bd8b9a6bd7afa8a1917d8f40a3a` |

## Verification scope

Six focused unit tests and scoped Ruff pass. The scripted CPU fixture uses actual
NativeClient request construction, tokenizer/response decoding, saved starts and
responses, original native episode/score, and independent saved-call audit.
Flat/fixed/adaptive complete to root score1, including a real helper in each
delegation mode. A fixed-arm fixture injects one root schema error, one child
schema error and one child native-stock error; all are charged and fully replayed.
Two separate fixtures verify two refusals and a three-response cap stop before
the next request, with root outcome unknown. These are not model-quality results.
Initial prompt+256 is931 for each arm. A public-only CPU planner solves the selected
native task in23 tool actions; neither witness is a compute-matched model baseline.

Complete observed episodes receive full native transition replay. The existing
auditor only binds request/response receipts for externally interrupted episodes;
that narrower guarantee is stated explicitly for admission stops. The wrapper
also checks start/response identities and summed actual output tokens. It does
not claim full partial-state replay.

The compact package remains frozen. A strict-fullformat comparator is a later
compact follow-up to separate unknown-recipe rejection from schema/dose effects,
not a reason to delay the accepted compact launch.
