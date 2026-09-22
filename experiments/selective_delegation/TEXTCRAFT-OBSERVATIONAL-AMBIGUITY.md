# Identical public starts can conceal different recipe worlds

CPU feasibility only,2026-09-22. All eight existing VAL roots/quantities were
retained; no model calls, outcome-based replacement, new GPU acceptance or edits
to057/the changed-world proposal. The deterministic replay itself took1.71seconds.

| Check | Result |
|---|---:|
| Byte-identical initial public prompts across worlds42/43 | 8/8 pairs |
| Different native root-query replies | 8/8 pairs |
| Existing public-query teacher native success | 16/16 |
| Regenerated privileged-order teacher native success | 16/16 |
| Public teacher queries root first | 16/16 |
| Privileged teacher first-query labels differ across identical inputs | 8/8 pairs |
| Public teacher maximum prompt+256-token cap | 2922/8192 |

## Construction and observed example

Regenerate the pinned official worlds with seed42/43,25 items/domain-tier. Preserve
each original public goal and target quantities. For each world, obtain a native
legacy plan from a conservative sufficient base-material supply; sum its actually
consumed base ingredients. Give both worlds the componentwise maximum of those
two base inventories, with no initial crafted items or root stock. Regenerate
the legacy plan under this common inventory. Compare complete first public prompt
bytes—not merely goal strings—and replay both teachers through the unchanged
native bridge, strict action schema, nonempty finish and net-inventory checker.

For `val.325`, both public goals require3×`c3_i2_23`. World42's root query reveals
`c7_i1_17:1` plus `c1_i1_11:1`; world43 reveals `c5_i1:2` plus `c6_ore:1`.
Both recipes yield3 root items. Before any feedback, the privileged teacher first
queries `c7_i1_17` in42 but `c5_i1` in43. The public teacher instead asks for
`c3_i2_23` in both, then succeeds in7 and5 actions respectively.

Across all pairs the public teacher uses238 actions, including native finish.
It receives only targets, inventories and its previous public recipe replies.
World identity, oracle dependency graph and reference action order are not its
inputs. Existing cached tokenization was CPU-only; no model weights were loaded.

## What this establishes—and does not

The deterministic privileged teacher's first label is not identifiable from
the initial observation: the same input has different labels. The hidden world
therefore supplies demonstrator information that the deployed policy lacks at
that moment. This is a sharper **supervision information-gap** diagnostic than
same-world memorization alone. Native query feedback distinguishes the worlds,
and the existing public-only teacher shows that both are solvable without that
initial privilege.

Different teacher labels are **not uniquely required optimal actions**. Both
first queries are legal in either world, other successful plans may exist, and
querying the shared root first works in both. Thus this is not proof that the
task requires different first actions or answer-conditioned branching; even a
generic static algorithm with feedback-bound recipe arguments may suffice. Nor
does it establish that teacher order caused052's model failures.

Common inventory is sufficient but not proved minimal. It overprovisions each
world and may remove exhaustion failures, alter prompt length and make irrelevant
materials distracting. World-specific plan length, branching and recipe counts
still differ. Any later model study must disclose this changed task distribution,
retain both worlds/all pairs, report same-frame matching and separate recipe
discovery from planning/execution. It must not silently replace the existing
changed-world panel or use these exposed VAL cases as new TRAIN examples.

## Provenance and decision

Immutable receipt: `R/analysis-textcraft-observational-ambiguity-001.json`, SHA256
`212410a12a6bb7e6fa772953664e3f6f8b94a9e785abf3a63ecd1eb27cb744d1`.
It records per-pair inventories, teacher actions/results, public prompt hashes,
world hashes and every executed source hash. Both world hashes match the earlier
`analysis-textcraft-worlds-001.json`.

Execution reused sealed055 `prepare_textcraft_public.trajectory` and its qualified
legacy teacher/bridge. Only the fully read official
`extract_base_materials_synth` and `solve_crafting_task_synth` function ASTs were
extracted from cached `synth_tasks.py`, avoiding the full Platoon lifecycle;
their source/AST hashes are in the receipt. No model-generated code was executed.
Upstream commit `d9c5857d3a0a056ebc9b047241a2a0c9515aafbe`, MIT.

The exact construction is now preserved in
`audit_textcraft_observational_ambiguity.py`, sealed with four byte-identical055
dependencies at `R/analysis-source-textcraft-observational-ambiguity-001`.
SOURCE SHA256 `fd3f800b19be94495b976ca76383c82bb2b2bd9b5a97904117b417dc46cf2993`;
driver SHA256 `54595c2da4fd5ebd7bd33d7844088aabeb4a1a0b935204aa538c7f79e9c3c5a5`.
One actual sealed replay verified both world hashes, all pair inventories,
initial-prompt byte equality, public/legacy outcomes and original001 fields.
Additive `analysis-textcraft-observational-ambiguity-replay-001.json` SHA256
`3dfff0ee9e264b79a24d034fe6ba9739185a529853f8a1f0ff9b02cb7f8945f3`
also records hashes of fully reconstructed public/legacy teacher rows.001 was
not overwritten; no broad tests or model calls were needed.

Reproduction CLI: with the existing training Python and
`CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1`,
run `R/analysis-source-textcraft-observational-ambiguity-001/audit_textcraft_observational_ambiguity.py --verify-against R/analysis-textcraft-observational-ambiguity-001.json`.
No PYTHONPATH override is needed. Optional `--report NEW_JSON_PATH` writes once;
an existing destination is rejected before replay. SOURCE.json records the full
Python executable and expanded command paths.

Decision: feasible conditional question preparation, not a new branch or accepted
model run. It can later test whether public-feedback supervision transfers across
observationally ambiguous worlds better than oracle first-action imitation. Wait
for existing057 evidence; do not claim novelty for partial observability or
learning to query before acting.
