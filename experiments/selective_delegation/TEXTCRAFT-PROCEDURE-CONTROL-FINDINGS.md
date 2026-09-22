---
status: exploratory_completed
evidence_cutoff_utc: 2026-09-22T10:00:00Z
question: Can a fixed procedural instruction recover the older trained model's failures?
result: negative_for_this_prompt_package
evaluation_task_parents: 8
exposure: previously_examined_validation_tasks_in_shared_recipe_world
publication_status: supporting_control_not_general_prompting_claim
---

# TextCraft fixed procedure-prompt control

## Result

On the same eight previously examined validation parents and two fixed seeds, adding one fixed
procedure instruction to the **old TextCraft action-SFT adapter** produced **0/16** successes,
versus **3/16** for that adapter under its original prompt. The paired result is 0 wins, 3 losses,
and 13 ties: **−18.75 percentage points**, with a 20,000-draw task-parent bootstrap interval of
**[−37.5, −6.25] points**. All 16 planned episodes were observed; no transport calls failed.

The appended sentence package was exactly:

> First query each requested target's recipe. Discover needed ingredients from returned recipes
> before crafting. Use current inventory and returned batch sizes; do not repeatedly query an item
> whose recipe is already known. Finish only after the net target is met.

This rejects that particular packaged procedure prompt for this fixed adapter, exposed panel, and
fixed evaluation protocol. It does **not** show that every procedural prompt, training-time
procedure target, or public-information teaching approach is harmful. This is not the newer
public-discovery SFT adapter (`textcraft-public-discovery-sft-001`); it reuses the older
`textcraft-action-sft-001` adapter.

## Cost and trajectory behavior

The procedure run consumed 850 native calls, 3,148,734 prompt tokens, 15,135 completion tokens,
and 1,407.0 summed service seconds. Its terminal statuses were 8 `finished`, 7 `context_cap`, and
1 `global_call_cap`; it made 770 `get_info`, 70 `craft`, 8 `finish`, and 1 `delegate` action. It
had 30 native action errors, one invalid-schema output, and one rejected action. These are observed
trajectory costs, not a matched-compute causal comparison.

A read-only replay of the saved public histories is a useful diagnostic but not a new outcome
metric. Procedure control made its first physical action a root query in 10/16 episodes and ever
queried the root in 15/16, versus 3/16 and 11/16 for the old original prompt. It also reduced
nonexistent queries (240 versus 344) but made substantially more calls and repeated already-returned
static recipes more often (418 versus 70). Thus target-first querying alone did not rescue
execution; the prompt was associated with query loops and cap terminations in this readout.

For a concrete paired loss, `textcraft_synth.val.19`, repeat 0, succeeded with the old original
prompt in 18 calls but hit context cap under procedure control after 94 calls. The procedure run
queried the target at call `t03-r0-flat-procedure_control-c001`, then repeatedly queried the same
returned `m0_i1_20` recipe while also trying to craft a nonexistent item; it never finished. This
is an illustrative trace, not proof that the appended instruction caused the loop.

## Provenance and limits

The authoritative report is
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/analysis-textcraft-procedure-control-001.json`
(SHA-256 `020c12fcb69ec57aa1a2889cad69182bc78acc90b64611af00daeb546bf400b3`). The behavior counts
are persisted in `analysis-textcraft-procedure-behavior-002.json` (SHA-256
`d932efa3d26a103c71f2b264d1b2b234e2ec5ec30208d32aa1f06389065852a1`). Its tiny driver,
`experiments/selective_delegation/audit_textcraft_procedure_behavior.py`, binds the exact two
labels, the two report hashes, and the strict “recipe returned in earlier public feedback” repeat
semantics before reusing `audit_textcraft_query_transfer.py`.

The eight task parents share a recipe world and have two correlated seeds each. The panel was
already exposed while designing the intervention. Accordingly, the result is a sharp local
negative control, not a fresh generalization estimate or a causal result about prompt wording in
general.
