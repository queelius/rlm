# World47 recipe-execution assist: sealed-source pointer

The launchable, immutable source is
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/source-textcraft-recipe-binder-003`.
Its `SOURCE.json` is the authoritative runtime manifest.  This worktree retains the pure,
reusable binder at `textcraft_recipe_binder.py` (SHA-256
`2729dc225da2bda00ff1d15ab5acbd87ea9d98a8aef6ebde67c02e0934546695`).

Source003 changes the public World47 collector only at the execution seam.  A model's parsed
`craft` action has its ingredient map filled only when exactly one recipe for that target has
already appeared in public `get_info` feedback and its requested output count is divisible by
that recipe's batch size.  It does not choose targets, quantities, recipes, or actions, and does
not check hidden recipes or feasibility.  The model-visible history retains only the executed
action and native feedback.  Requested/executed action pairs and the binding reason are stored in
the non-prompt-visible `execution_assists` receipt field.

The source003 replayer recomputes the binding from prior public history, verifies every stored
receipt, and applies the executed action.  Its paired unchanged-public baseline is reused from
the authenticated native audit
`analysis-textcraft-world47-seed2026092208-001.json`; it is not replayed with the changed binder.
The source checks the baseline PLAN and every baseline episode against that audit's hash map.

The focused native fixture in the sealed source is
`test_recipe_binder_replay.py` (SHA-256
`3bb6c365ce4de4180d861c128eb892eb2b6d50413ba5cd4fecff9ece8637c8ff`).  It uses two independent
native frames: one collector-side binding step and one raw-JSON parse/replay step.  It asserts
identical real feedback, history, and next prompt, and rejects a tampered executed-action receipt.
This is a changed execution-interface pilot, not a learned planner or an unchanged-prompt policy
comparison.
# Git backup of the changed execution code

The adjacent `recipe-binder-source003.patch` preserves the collector, independent
replayer, fixed runner, baseline-comparison analysis and focused native fixture.
It is a unified patch against `source-textcraft-reserve-readout-001`, not a patch
to an arbitrary current checkout. The reusable binder and its unit tests are also
checked in directly. This backs up the source changes, not model weights or traces.
