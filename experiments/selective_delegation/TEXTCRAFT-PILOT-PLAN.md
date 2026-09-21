# Source044: native flat/recursive TextCraft qualification

Accepted by main September21 18:51 UTC; supervisor2094 waits for042b. The
acceptance receipt and immutable PLAN are in the external study store. No model
outcomes are available at this cutoff.

Question: does bounded recursive delegation improve native root success on longer
compositions under the same global model-call/output budget and frozen base4B?
Use immutable source043 and `textcraft-inputs-001` unchanged. Eight tasks×two seeds
2026092204/2026092205×flat/recursive =32 planned episodes; no post-replay selection.

Implementation: one native HF frozen-base client; one synchronous recursive executor;
one shared `Budget(96,8192)` and inventory per episode. Root depth0, descendants1/2.
Each frame snapshots its own initial inventory, while root retains its original
snapshot for exact native score. Every returned model response is charged including
EOS before strict JSON parsing. Invalid actions consume budget and provide explicit
rejection feedback, no repair/retry/early invalid limit. Mechanical child returns
contain actual child finish text/status and inventory, never a free model summary or
hidden native score. Parent's next model call is charged normally.

All roles use T.5/top-p1/top-k0, fresh GenerationConfig, no penalties/adapters,
max256 output reduced only by remaining global tokens. Native chat tokenization,
input+allowed-output<=8192, no truncation. Context exhaustion is an observed policy
failure of that frame; global budget exhaustion prevents any further node calls.
Transport/model/environment exceptions are unknown and stop the owner. Preserve
requests, emitted IDs, roles, node/parent/depth, global call index and actual costs.

Paired seed formula: repeat seed + first8 SHA256(taskID) hex as integer +
1000×global zero-based model-call index. Both policies share the formula; after
their behavior differs the same index need not denote the same state/role. Initial
public prompts differ only max_agent_depth0 versus2; later contexts necessarily
differ because delegation creates new goals/histories. This is the tested harness
intervention, not matched internal computations.

One-hour cumulative owner cap, exclusive existing GPU lock, allocation end−600s,
no silent retries/resume, native first-response check within90s. Lightweight per-call
progress, summary every4 episodes and terminal. Planned16/policy denominators retain
missing/unknown separately. CPU smoke: native tiny-model request/cap/frozen weights,
actual trusted bridge nested returns/global exhaustion, strict invalid accounting.

Conditional launcher waits authenticated complete48-episode
`alfworld-trained-actor-001` (accepted source042b/decision002), with exact source/input/PLAN acceptance
hashes. Main alone accepts/seals/launches; no GPU run is part of this preparation.

CPU provenance note: normal trusted-generator import created an ignored Python
`__pycache__/synth_recipe_generator.cpython-312.pyc` in the external upstream clone.
It is retained, tracked upstream sources are unchanged, and the generated world is
checked against the frozen source043 manifest. The launcher disables further
bytecode writes. This does not change source043 or its qualification evidence.
