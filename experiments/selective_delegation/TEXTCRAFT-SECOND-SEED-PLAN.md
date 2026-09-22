# Second shared training seed: teacher-package replication

## Accepted execution, September 22 at 15:48 UTC

The comparison is accepted and waiting behind the RL continuation and its
evaluation. It will run automatically after that entire queue releases its
resources. Waiting is not GPU training: the GPU is currently evaluating the
earlier one-update RL checkpoint.

The accepted receipt is `INDEPENDENT-TRAINING-QUEUE-013-ANALYSIS-AMENDED.json`
under `/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.
Its SHA256 is `9ddee303c36ab5719daa487c7918fcc3c7f3b2316a6ceac839c95b1381c25cf9`.
It supersedes the unlaunched queue-013 receipt only to replace its CPU analysis;
all four training and evaluation jobs are identical. The queue may wait at most
12 hours, then has a separate four-hour execution allowance. Both respect the
allocation deadline, with ten minutes reserved for shutdown.

The fixed training seed is **2026092291**. Each teacher uses the original 366
examples and 23 updates. The output directories are
`textcraft-teacher-seed2291-{privileged,public}-001` for training and
`textcraft-fresh-seed2291-{privileged,public}-001` for evaluation. The final
CPU-verified comparison is `analysis-textcraft-second-seed-001.json`.

Source snapshots are `source-textcraft-teacher-seed2291-001` for training and
evaluation, and `analysis-source-textcraft-teacher-seed-002` for analysis.
The latter removes irrelevant within-arm flat-versus-recursive summaries that
could otherwise show misleading zeros. Native action checks, costs, missing
outcomes and the actual comparison between teachers are retained. Use the
separate `analyze_textcraft_teacher_seed.py` entry point for this analysis.

Main verification: 11 sealed seed/input/readout tests passed in 8.06 seconds;
the analysis-summary regression passed in 0.09 seconds; 64 unique job pins
matched. Independent review found no further material training or readout
blocker. The queue's eight focused tests passed; an actual preparation check
confirmed it cannot start while its predecessor remains active. Ruff and format
checks passed using this repository's configuration. These are focused checks,
not a claim that the full repository test suite was run.

## Question and interpretation

Prospective seed **2026092291**, frozen before either new training run or outcome.
Question: does public-discovery versus privileged-teacher fresh-goal performance
replicate across a second training seed, rather than depending on the original
2026092208 initialization/order?

Use the exact original two 366-row/32-TRAIN-task datasets. Both policies start from
base4B, one epoch/23 updates, LR1e-4, rank8/alpha16/dropout0; fixed cp23 only. No
task deletion, new teacher, world43, direct baseline or new panel. Known train.1029
quantity difference remains: this tests the teacher package/history intervention,
not pure query-order causality. Histories, prompt tokens and target tokens differ;
equal examples/updates are not equal information or FLOPs.

The thin wrapper explicitly configures the unchanged source048 recipe's one SEED
constant, records wrapper/seed/input identity, and restores the constant afterward.
Recipe bytes, public055 qualification and input hashes remain strict. The seed
controls both initial LoRA randomness and example order, as in the original recipe.
Each trainer is capped at30minutes; preflight prepares both immutable PLANs and
pins them before any launch. `--resume` at launch is the unchanged recipe's required
way to use an already prepared identical PLAN, not permission to change inputs.

Read both fixed endpoints under identical BF16 original prompts on the frozen
fresh16 x2 slots, one hour each. Independently replay native actions and report the
paired public-minus-privileged effect with16-task bootstrap and unknowns retained.
This is a reused exposed panel/shared recipe world, not a new independent test set.
Replication would strengthen seed robustness, not identify teacher order as the
sole cause. A null/reversed second-seed effect limits confidence; do not select the
better seed or retune from these results.

Accepted queues 011/012 and their source snapshots are untouched. The existing
handoff runner has one bounded extension: an explicitly accepted wait can be
up to 12 hours rather than four. Its default, process-identity checks, resource
release checks and execution caps are unchanged. Two training caps plus two
evaluation caps total three hours, followed by bounded CPU analysis.
