# Source043 CPU bridge and frozen inputs

Update, September21 18:51 UTC: main accepted the separately sealed source044
pilot after four focused native/recursive fixtures passed. It is queued after the
trained ALFWorld readout. There are no TextCraft model results yet. The readiness
account below records the earlier CPU-only stage; see [the pilot plan](TEXTCRAFT-PILOT-PLAN.md).

No GPU accepted/launched; source044 collector remains a separate step.
Eight official VAL tasks were frozen **before** any replay, using selection
seed2026092203 and hash order within the predeclared depth strata. No tasks were
replaced or dropped. Root is depth0; recursive children may reach1 and2.

| Task suffix | Recipe depth | Gold craft actions | Info+craft+finish calls | Gold public-history maximum input tokens |
|---|---:|---:|---:|---:|
|325|2|3|7|957|
|629|3|5|11|1300|
|47|3|4|9|1083|
|19|2|2|5|783|
|567|4|11|23|2345|
|435|4|5|11|1249|
|207|4|9|19|1965|
|352|4|10|21|2143|

All eight quantity-correct gold traces passed the exact trusted upstream craft and
finish-checker bodies, including when recipe queries were inserted. This establishes
CPU environment feasibility, not model competence or a policy success rate. The
constructive traces use5–23 global calls and107–584 output tokens including an EOS
allowance; these are not proven minimum costs. They fit the shared96-call/8192-token
budget. No oracle-length filtering occurred.

Actual cached Qwen3-4B tokenizer: initial prompts445–531 tokens, identical counts
across the two policies; maximum audited gold-history prompt2345 tokens, largest
gold JSON action48 tokens including EOS. The pre-outcome per-call cap256 accommodates
short notes/delegation context while retaining the same global8192 output budget.
Native collector must still enforce8192 input+output, no truncation, and retain policy
context/budget failures. Gold traces never enter model prompts.

## Trusted code boundary

Pinned upstream commit `d9c5857d3a0a056ebc9b047241a2a0c9515aafbe`, MIT license.
`textcraft_bridge.py` verifies whole source hashes before loading the reviewed stdlib
recipe generator and compiling only the four statically allowlisted AST method bodies:
`TextCraftCodeExecutor.craft/get_info/view_inventory`, `TextCraftEnv.evaluate`.
No model-provided text can enter this loader/compiler: model text only passes through
strict JSON parsing and exact-field/action/type validation. This reuses native method
bodies, **not** the complete Platoon/IPython lifecycle. No IPython interpreter starts.
No packages or environments were installed; no clone or active training env changed.

The exact seed42/items25 regenerated world contains1402 recipes, canonical world hash
`f76ce3978c038be9624b3c7387aa30033970508c6afecb1f8f08315aa0693808`.
It is not the smaller checked-in recipe-directory world. Tests cover method AST hashes,
native positive gold success, wrong quantity/no mutation, nonempty explicit finish,
net-inventory growth, strict schema/bool/duplicate rejection, public projection,
shared recursive inventory, and one global call/token counter including grandchildren.

Collector responsibility: charge every returned model response **before** parsing,
including invalids, delegation, child finish and the next parent response. No free
model-generated return summary; child finish text may be returned mechanically.
Budget/depth/action exhaustion is observed policy failure; transport/native exception
is unknown and halts the owner. Never turn external errors into reward zero.

Inputs live under
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/textcraft-inputs-001`:
`SELECTION.json`, `tasks.jsonl`, `REPLAY.json`, `TOKEN-AUDIT.json`, and preserved exact
selection/replay/token-audit source snapshots. The proposed source043 seal adds the
final bridge/preparer/tests and upstream MIT notice; manifest binds them all.

The main comparison remains32 episodes =8 tasks×2 seeds×2 policies, not32 independent
tasks; max3072 model calls and one-hour owner cap. Shared recipe-world compositional
transfer and interface qualification only—not RAO reproduction or a novel method.
