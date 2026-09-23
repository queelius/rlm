# Repeated-state step-credit feasibility

This audit checks whether saved sample-0002 rollouts contain genuinely repeated
decision states for a GACA-style anchor contrast. Exact model-input tokens ignore
sampling seed. A second, deliberately conservative public-state key retains goal,
inventories, budget, and full public history (including returned recipes); it does
not count inventory-only matches as state matches.

Repeated calls within one trajectory can identify a loop but share the same terminal
reward and therefore do not provide an independent episode-level contrast. A
cross-episode repeated state with different terminal reward and action would be the
minimum saved-data signal for a new step-credit estimate. Otherwise, reusing root
RLOO advantage at matching calls would merely copy episode credit, not identify a
new step effect. This is a feasibility result for one 32-episode, four-samples/task
batch, not evidence about general repeated-state credit.

## Sample-0002 result

The 32 completed episodes contain 909 available calls. Both exact token identity
and the conservative public-state summary yield 808 unique states and 44 repeated
cross-episode groups (145 calls). There are no within-episode repeated-state groups.
Of the 44 cross-episode groups, 23 have terminal-reward variation and 16 have
different decoded actions somewhere in the group. Thus saved data can support a
strictly observational anchor diagnostic, but it cannot establish a causal step
credit estimator without specifying how repeated samples, terminal rewards, and
trajectory dependence enter the objective. It is not merely duplicated root RLOO
in every matching group: there are 23 distinct reward-variable anchor groups, not
909 calls. These groups remain
correlated: repeated states can be nested shared prefixes of the same four sampled
trajectories. Of those groups, 12 also have different actions and per-action mean
returns; 11 differ only in later outcomes after the same action. Exact anchor LOO
differs from task episode-RLOO for 35 non-root records. These are descriptive
credit differences, not causal root or step effects.

## Prospective compressed-state diagnostic

A future policy could receive goal, initial/current inventory, a notebook of
actually returned `get_info` facts (`item`, `can_craft`, `is_base`, depth, recipes),
and remaining call/token/depth budget—but not raw history. This is a prospective
recipe-notebook interface, not a claim of sufficiency or current-policy validity.
On saved trajectories it is only an aliasing diagnostic; merged groups cannot be
used for current on-policy credit and require fresh rollouts. Inventory-only remains
invalid when matching frames differ in recipe knowledge or other public state.

The notebook key gives 50 cross-episode groups covering 160 calls, compared with
44 groups and 145 calls for exact prompts. Fourteen groups have different actions
and different mean returns. This modest increase does not yet justify prioritizing
compressed-state training over the pending memory and credit comparisons.
Inventory-only matching hides differences in recipe knowledge in 36 groups and
other public state or remaining budget in another 65 groups.

## Decision

Keep repeated-state credit as a secondary diagnostic. Its changed credit reaches
only 35 of 909 saved decisions, and correlated shared prefixes do not supply 35
independent experiments. If memory helps and exact-state coverage remains sparse,
a more informative future experiment may deliberately sample several continuations
from selected non-root decisions. That requires a new rollout design and a matched
compute comparison; it is not evidence already contained in this batch.

Receipt: `R/analysis-textcraft-repeated-state-credit-006.json`; the report records
the batch, all call-file hashes, and audit-source hash.
