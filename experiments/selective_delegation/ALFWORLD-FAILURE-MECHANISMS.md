# Where executable actions still failed (completed source026)

Inspected all16 heating/cleaning attempts: four games x two seeds x two policies.
All reached50 actions without a native win; all responses were valid. Mechanism
descriptions below use only public initial tasks, observations, available commands,
model goals and selected actions. Native `won` supplies the outcome only; no hidden
PDDL state or expert trajectory is used to explain a failure. These are trace-based
hypotheses, not identified internal causes.

Across all eight heating attempts, **no heat command appeared in a worker/flat
decision's current admissible list**. Thus the traces do not show a policy ignoring
a currently executable heat action. Only one of eight cleaning attempts reached a
visible clean action; it then executed that action33 times. The dominant problems
include reaching the right state, object/task interpretation, and leaving a completed
subgoal—not malformed command syntax.

## Three concrete contrasts

1. **Hot egg: task interpretation and action routing, not just object search.**
   Game05/seed2026092178's public task requests a hot egg in the fridge. Flat finds
   egg4 in the fridge, takes/puts/takes it, then selects `cool egg 4 with fridge 1`
   at action9. It later repeatedly examines a stoveburner. The manager initially
   proposes cooking with a stove/pot, eventually retrieves the egg, then explicitly
   changes its goal to “Cool the egg4 … to complete the task.” Its worker instead
   puts/takes the egg, and the manager subsequently returns to stove goals. Public
   `go to microwave 1` is available at the egg-in-hand states; `heat` is not yet
   available. This supports a wrong transformation/route hypothesis, not proof that
   the model lacks heating knowledge. Some other heating attempts never find the
   requested object at all (game04 repeatedly searches for a tomato).

2. **Butterknife: target found, but premature put/take loop; manager fails earlier.**
   Game06/seed2026092178 flat spends14 actions on `look`, then public feedback at
   countertop2 explicitly lists butterknife1. It takes the knife at action22 and
   immediately puts it back at23, before any clean action or cleaning feedback.
   It alternates taking/putting through action50:15 takes and14 puts. The same
   index43 maps alternately to those two commands; `go to sinkbasin 1` is available
   while holding the knife. This separates finding the target from satisfying its
   required condition. At the same seed, manager/worker instead takes a bowl and
   cycles among cabinets, including21 examinations of cabinet18; its observed
   failure occurs before finding the correct object. A single “planning” label
   would conflate these bottlenecks.

3. **Cloth: completed transformation, stale goals, and worker noncompliance.**
   Game07/seed2026092179 manager/worker finds cloth1, goes to the sink, and cleans
   it at action18. The next public feedback explicitly says it cleaned the cloth.
   Nevertheless actions18–50 are the same clean command (index0). At actions19–50,
   current feedback, available-command list and selected action are identical;
   full public histories still grow, so these are not literally identical prompts.
   The manager repeatedly renews a cleaning goal despite that feedback. Crucially,
   it correctly asks to put the cleaned cloth in drawer1 after actions36 and48,
   and the worker still cleans it while `go to drawer 1` is visibly available.
   This is not solely a stale-manager problem. The inspected requests dropped zero
   history entries. Flat at this seed finds the cloth but repeatedly puts/takes it
   in the drawer without ever going to clean it. At the other seed, manager goals
   also conflate the requested cloth with handtowels: another object-category seam.

## Smallest useful control before outcome RL

After source032 completes, choose a few **TRAIN-only public prefixes**, fixed without
success-based selection, spanning object-in-hand before transformation and explicit
transformation-confirmed feedback. Compare the unchanged next-action policy with
two separately declared prompt aids: (a) a generic native operator/task glossary
from public help/documentation, with no instance-specific solution, and (b) a compact
verbatim ledger of already observed task-relevant events, with no new operator
knowledge or hidden state. Keep commands/budgets fixed and charge added tokens.
Test local actions and bounded continuation, not only a manually judged rationale.

If the glossary changes wrong transformation/routing while the ledger does not,
task/interface knowledge is a plausible target. If the ledger prevents repeating
an explicitly completed action without glossary help, progress tracking is a more
plausible target. Neither result proves an internal mechanism; these aids alter
prompting. Since repeated indices0/43 are also striking, a small deterministic
**reindexing** check at the exact same public state is an inexpensive nuisance control
before attributing everything to memory. Preserve all commands and map indices
exactly; never remove a legal action or repair a response. No such follow-up is
implemented or accepted here. Sparse terminal RL alone would not distinguish these
knowledge, search, index-selection and progress-tracking failures.

## Exact native trace pointers

R=`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`;
all files below are in `R/alfworld-closed-loop-001/calls/` (append `.json`).
Corresponding full trajectories are in `episodes/` and the completed audited report
is `R/analysis-alfworld-closed-loop-001.json`.

- Heating: `game-05-seed-2026092178-flat-call-008-flat`;
  `game-05-seed-2026092178-manager_worker-call-030-manager` and `call-031-worker`.
- Knife: `game-06-seed-2026092178-flat-call-021-flat`, `call-022-flat`, `call-023-flat`.
- Cloth: `game-07-seed-2026092179-manager_worker-call-022-worker`, `call-023-worker`,
  `call-025-manager`, `call-045-manager`, `call-046-worker`, `call-060-manager`,
  `call-061-worker`. Suffix-only names retain the immediately preceding episode prefix.

The decisive correct-goal/ignored-goal worker receipt (`…call-046-worker`) SHA256 is
`712758c6188abeaa1ba150c11b3d69de964436471e458b16a6622b8fd6adf0a8`.
No future evaluation outcomes or live collector source were inspected/changed.
