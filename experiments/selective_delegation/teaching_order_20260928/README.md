# Matched-action teaching-order controls, 2026-09-28

The quantity-corrected known-recipe and discovery teachers have identical per-task action,
target-string and unmasked label-token multisets: 32 tasks, 366 rows, 8,820 target tokens.
Their conditioning histories differ. Known-recipe training queries names available in public
state on only 32/167 query rows; discovery does so on 167/167. The first six breadth panels
show 39/768 versus 323/768 raw successes. See
[the fixed-cutoff analysis](../finding_textcraft_breadth_20260928.md).

These controls test whether repairing the availability of queried names helps transfer while
holding the original craft order and optimizer targets fixed. The general teacher/student
information gap is established prior work; any contribution would require evidence for this
particular matched-action trace repair and its transfer effect.

## Prepared comparisons

All assets live at
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/textcraft-teaching-order-20260928-001`.

| Teacher | Query selection when next original craft is not ready | Native successes | Prompt tokens | Same native sequence as discovery |
|---|---|---:|---:|---:|
| `stable_visible` | Earliest original query with a public name | 32/32 | 431,274 | 8/32 tasks |
| `random_visible` | Fixed-seed random currently visible query | 32/32 | 435,710 | 8/32 tasks |

Both retain every original craft in its original order, initial stock, goal, action multiset,
literal target strings, and label-token multisets. Every query is grounded in targets, inventory
or earlier recipe feedback. A craft needs a previously queried recipe and sufficient current stock.
The random seed is `2026092801 + int(sha256(task_id)[:8], 16)`.

This is an **offline oracle action schedule with publicly observable query names**, not a claim
that the scheduler constructs quantities or chooses among gold actions without privileged input.
Stable selection greedily retains the earliest original query; it is not claimed globally minimal
under an edit-distance objective.

Training rows remain in the original known-teacher action-index order; `step` retains native
chronology and `source_action_index` explains the permutation. Thus the unchanged source048 shuffle
presents exactly the same target sequences and token denominator in each paired optimizer batch.
Only the conditioning histories differ. Prompt-token dose and computation are not identical.

`prepare.py` constructs rows by native replay. `audit.py` independently replays the saved rows,
reconstructs every prompt/mask and checks visibility without invoking the scheduler. `test_schedule.py`
guards hidden-name queries, recipe observations, stock feasibility and original craft order.
Preparation and training inputs are immutable; failed attempts must be retained under a new name.

## Ready one-A100 commands

Use the existing GPU environment and the allocation's normal single-GPU assignment. Parent owns
all GPU launches. No frozen source/import package was edited.

```bash
GPU_PY=/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python
TEACHING=/project/alex_phd/repos/rlm/.worktrees/selective-delegation-20260921/experiments/selective_delegation/teaching_order_20260928
"$GPU_PY" "$TEACHING/train.py" --mode stable_visible --seed 2026092208 --resume
"$GPU_PY" "$TEACHING/train.py" --mode random_visible --seed 2026092208 --resume
```

The CPU `--prepare-only` pass has already created both training PLANs. **`--resume` is required**
by source048 even before checkpoint0 exists. Each run starts from the pinned base/LoRA seed, does
23 updates at LR1e-4, effective batch16 (last14), and retains every checkpoint. The fixed endpoint
is checkpoint23. Expected owner wall time is **205 seconds per arm**, based on matching earlier
one-epoch jobs at 200.5–207.2 seconds; this is an estimate, with a 1,800-second hard cap. Both supported
training seeds are 2026092208 and 2026092291; prioritize distinct first-seed controls before replication.

After a complete released training owner, evaluate raw mode first on panel00/world42, then world50.
These reused exploratory panels have completed matched known-recipe/discovery baselines.

```bash
"$GPU_PY" "$TEACHING/readout.py" --mode stable_visible --seed 2026092208 --panel 0 --world 42 --execution raw --prepare-only
"$GPU_PY" "$TEACHING/readout.py" --mode stable_visible --seed 2026092208 --panel 0 --world 42 --execution raw
"$GPU_PY" "$TEACHING/readout.py" --mode stable_visible --seed 2026092208 --panel 0 --world 42 --execution raw --audit
```

Substitute `random_visible`, `--world 50`, or `--execution binder` for distinct follow-ups. Each
readout is 8 fixed goals ×2 evaluation seeds, with unchanged prompt/sampling and global caps96calls,
8,192 output tokens and 8,192 context tokens. Expected useful execution is **15–25 minutes**, with
a 45-minute hard cap; do not count the cap as expected coverage. Native audits are CPU-only. The
wrapper cannot bind an unfinished training endpoint. Existing scientific attempts cannot be silently
rerun. Verify a real returned model response promptly after startup using the normal owner checks.

Primary comparison: stable-visible versus matched known-recipe, retaining all16 attempts. Random
versus stable tests robustness to a distinct visible frontier; discovery is a descriptive reference
whose original training minibatch target order differed. Report task-clustered paired outcomes,
queries of unseen/nonexistent names, calls and failure categories. A restored root-first query alone
does not establish restored planning: the old procedure-prompt control already improved root-query
behavior while reducing success from3/16 to0/16.

## Decision and later controls

Promote the repair if success and name grounding improve across worlds and the second training
seed. If name grounding improves without success, focus on feasibility/quantity planning. If it
does not learn the repaired policy, inspect teacher-forced fit before more rollout sampling.

A two/three-epoch matched continuation remains a useful undertraining control, but is **not ready
in this wrapper**: source048 deliberately accepts exactly one epoch and treats checkpoint23 as
terminal. Do not bypass that guard or relabel a duplicate epoch as a continuation. The historical
one-step extra-SFT experiment involved the discovery model and does not settle known-teacher
undertraining. A later discovery dataset reindexed into the same target presentation order would
separate minibatch-order effects from conditioning-history effects without changing its prompts.
