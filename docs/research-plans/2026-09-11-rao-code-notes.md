---
date: 2026-09-11
status: source_inspection_not_a_reproduction
question_ids:
  - rq:rl-effective-feedback
  - rq:depth-value
upstream_commit: d9c5857d3a0a056ebc9b047241a2a0c9515aafbe
---

# A practical lesson from the published recursive-agent code

The [RAO project page](https://apga.github.io/RAO/) links a specific
[Platoon snapshot](https://github.com/ApGa/platoon/tree/d9c5857d3a0a056ebc9b047241a2a0c9515aafbe).
We downloaded that snapshot into the external research cache and inspected selected
training and environment code. We have not installed or run it.

The useful distinction is **whether a whole solution succeeded versus whether one of
its smaller tasks succeeded**. Our first diagnostic measures whole-solution feedback
for root-only RL. It must not be treated as a universal test of whether all forms of
recursive RL can learn.

In the snapshot's Tinker workflow, each trajectory starts with its own reward. The
workflow subtracts a baseline computed from other root rollouts. Its constant-reward
filter considers child trajectories as well as root trajectories. Therefore, all roots
can have the same score while independently scored children still provide nonzero
learning signals. This is an implementation observation, not evidence that copying
the objective will improve our model.

Relevant source:

- [`group_rollout_workflow.py`](https://github.com/ApGa/platoon/blob/d9c5857d3a0a056ebc9b047241a2a0c9515aafbe/platoon/train/tinker/workflows/group_rollout_workflow.py): constant-reward filtering and root-baseline subtraction.
- [`rl.py`](https://github.com/ApGa/platoon/blob/d9c5857d3a0a056ebc9b047241a2a0c9515aafbe/platoon/train/tinker/rl.py): inverse trajectory-frequency weighting by depth, rescaled using action-token totals.
- [`env.py`](https://github.com/ApGa/platoon/blob/d9c5857d3a0a056ebc9b047241a2a0c9515aafbe/plugins/textcraft/platoon/textcraft/env.py): records immediate children's success separately from the parent outcome.

For us, this suggests a follow-up when a root can execute a sensible calculation but
its helper supplies wrong labels: compare training only the root with training the
helper on independently checked local tasks. Keep root correctness primary, and do not
reuse agreement with a potentially wrong helper as if it were a true local answer.
For unrestricted natural-language subtasks, reliable local scoring is itself an open
problem; exact labels are available only in particular controlled task families.

Do not copy the supplied configuration and expect it to fit our allocation. The
[8K TextCraft AReaL configuration](https://github.com/ApGa/platoon/blob/d9c5857d3a0a056ebc9b047241a2a0c9515aafbe/plugins/textcraft/platoon/textcraft/configs/areal/textcraft_synth_ctx8192_recursive_medium_areal.yaml)
requests eight GPUs and disables LoRA. A one-A100 adapter experiment would be a reduced
adaptation, not a faithful reproduction. AReaL and Tinker are separate implementations;
the selected-code inspection is not an audit of their equivalence.

## Acquisition and next decision

Exact commit: `d9c5857d3a0a056ebc9b047241a2a0c9515aafbe`.
The root license is MIT. The external acquisition receipt records the source, tree,
license and lockfile hashes, cache path and execution status:
`acquisitions/2026-09-11-platoon-rao.json` under the active research store.

The clone includes synthetic crafting data and a recipe generator. No examples have
entered our experiments. Inspect dataset rights, split identities and shared recipe
families before using it for a depth-generalization test. Do not let this acquisition
delay the already prepared GPU diagnostic.
