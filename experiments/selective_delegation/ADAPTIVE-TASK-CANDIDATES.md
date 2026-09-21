# Observation-conditioned task candidates

2026-09-21; optional future queue input, **not a pivot or accepted GPU job**.
Read `COMPOSITION-ASSETS.md`: do not repeat B05, purchase joins, or fixed operator
composition under a new name. Their facts permit advance composition. The two
candidates below instead provide native action/observation environments. Neither
establishes novelty for adaptive planning, hierarchical policies, or tool use.

## 1. ALFWorld text-only: strongest small screening candidate

[Paper, ICLR 2021](https://arxiv.org/abs/2010.03768) and
[official repository](https://github.com/alfworld/alfworld). Use text-only tasks,
not visual THOR or a new agent framework. The official
[playthrough](https://github.com/alfworld/alfworld/blob/aaba6870f86c5be6a08a491f32a50b906227bc3e/scripts/README.md)
illustrates a concrete branch: while obtaining a hot potato, inspecting the first
countertop does not reveal a potato, so the next action searches elsewhere;
inspection of another countertop reveals potatoes, enabling pickup and heating.
The number/location of searches is observation-conditioned, not an extra fixed
chain of question placeholders.

**Interface/reward.** Goal + current textual observation + past public actions
and observations → one native command (`go to`, `open`, `take`, `heat`, `move`).
Use the simulator's exact `won` flag, not termination alone. A flat reactive
controller receives exactly the same information and action interface as an
adaptive subgoal/worker controller. Both may branch; the question is whether
explicit subproblem management improves success/cost, not whether open-loop
actions lose when information is hidden. Full world state would be an explicitly
privileged oracle, not the primary direct baseline.

**Important implementation seam.** The
[official environment adapter](https://github.com/alfworld/alfworld/blob/aaba6870f86c5be6a08a491f32a50b906227bc3e/alfworld/agents/environment/alfred_tw_env.py)
can expose admissible commands and, under training wrappers, expert plans/facts.
Never put expert plans, PDDL state, gamefile paths, or hidden facts in prompts.
Either expose the same admissible-command list to both arms or expose it to
neither; record that choice because the list itself supplies affordance evidence.
Preserve ordinary initial room/receptacle observations—do not hide more to make
decomposition look necessary. Exclude engine-invalid/unsolvable instances by a
predeclared CPU check, not after seeing model performance.

**Smallest screen, if later accepted.** Eight predeclared development games,
covering search-and-place and search-then-transform goals, two sampling seeds,
two arms =32 episodes. Both use the same frozen 4B model, public history, 50
environment-action limit, and 2,048 generated-token episode ceiling; count
manager calls against that ceiling. Flat one-action policy versus an explicit
adaptive subgoal policy with a bounded worker; no new SFT/RL initially. Keep
official train/development membership and group shared scenes/layouts when
making later held-out claims. Estimated one A10040GB, one resident BF16 4B plus
KV cache, roughly30–60 minutes; hard60-minute collection cap, not a measured
throughput promise. CPU simulator/dependency setup is separate and unperformed.

**Promote/retire.** First verify several native traces in which newly revealed
object absence/presence changes the useful next subgoal. Promote only for clear
paired successes or fewer actions/tokens at comparable success against the
equally informed reactive baseline, with errors attributable to subgoal choice
rather than parser syntax. Retire as a decomposition screen if flat control is
already near ceiling, both arms fail predominantly on command grounding, or the
only advantage comes from extra information/budget. Existing contingent scripts
may suffice; there is no claim that a prewritten conditional policy is impossible.

**Decomposition boundary.** Finding a potato changes the useful action, but may
not require inventing a new decomposition: a fixed find→heat→place recipe with
a reactive search routine could suffice. The first screen measures the value of
explicit subgoal management versus flat reaction, not learned recursive depth.
Before attributing a later gain to adaptive decomposition, compare against that
fixed-recipe controller with equally capable, equally informed reactive workers.
That is a conditional follow-up, not an additional arm silently added here.

## 2. ScienceWorld unknown-material experiments: conditional, lower priority

[Paper, EMNLP 2022](https://arxiv.org/abs/2203.07540) and
[official repository](https://github.com/allenai/ScienceWorld). The unknown-material
melting-point task is a plausible richer branch probe. Its
[official task implementation](https://github.com/allenai/ScienceWorld/blob/e8216d6044e8e39be9fcb185e3b2dfb602584b52/simulator/src/main/scala/scienceworld/tasks/specifictasks/TaskUseInstrumentThermometer3.scala)
contains heating versus cooling procedures, repeated phase/temperature checks,
and a broken-stove fallback to a furnace. Illustrative candidate: after attempting
heating, observations establish that it is not working, so obtaining another
heater replaces continued measurement. A phase transition instead makes further
heating unnecessary and permits the threshold decision.

**Do not oversell this example.** These branches were inspected in the official
gold policy, not demonstrated by a new public-observation rollout. That policy
reads simulator internals. Initial object names may already say “liquid,” so that
particular branch is not evidence of new information acquired mid-episode. Before
promotion, verify that selected native variations genuinely reveal a needed
branch only through public action feedback, and that device-failure/state-change
branches occur; otherwise this is merely another preplannable measurement chain.

**Interface/reward and risk.** Public task, `look`/inventory, command feedback and
measurement results → one native action; no object tree, hidden property values,
gold sequence, or privileged goal-progress annotations. The
[Python API](https://github.com/allenai/ScienceWorld/blob/e8216d6044e8e39be9fcb185e3b2dfb602584b52/scienceworld/scienceworld.py)
returns score and termination separately; retain exact official score/success,
plus a separately labeled measurement-evidence audit. Source goals permit the
correct answer-box decision without enforcing all measurements. Thus exact
success alone does not prove adaptive experimentation; guess/name leakage is a
real competing explanation. An
[open official-repository issue](https://github.com/allenai/ScienceWorld/issues/81)
reports problematic melting-point gold traces; it is not proof the current
revision still fails. Current README also warns that1.3.0 changed deterministic
object order and physics/gold trajectories. Pin the revision and simplifications;
do not silently reuse old trajectories.

**Smallest conditional screen.** Only after a CPU trace audit passes: four
predeclared development variations with verified public-information branches,
two seeds, the same two equally informed 4B policies =16 episodes. Use80 actions
and2,048 generated tokens per episode; estimate30–60 minutes on one A10040GB,
hard60-minute cap. Java simulator runs on CPUs; installation is not done here.
Use official train/dev/test variation APIs, preserve substance-family separation,
and report this deliberately selected branch-bearing subset—not whole-benchmark
performance. Promote only if observed evidence causally changes useful actions
and the adaptive policy beats matched flat control without gold/API leakage.
Retire if branch state is already in the initial prompt, if success comes from
guessing answer boxes, if unchanged scripts suffice, or if physics/annotation bugs
dominate. No reward redesign is proposed for this first screen.

## Inspection provenance and scope

Retrieved small official pages/source text at2026-09-21T14:09:50Z; branch heads
resolved with read-only `git ls-remote`. No repository clone, dataset/model
download, installation, downloaded-code execution, or GPU job occurred.

- ALFWorld revision `aaba6870f86c5be6a08a491f32a50b906227bc3e`;
  [MIT license](https://github.com/alfworld/alfworld/blob/aaba6870f86c5be6a08a491f32a50b906227bc3e/LICENSE).
  README separately identifies TextWorld as MIT and Fast Downward as GPLv3;
  benchmark/assets/dependencies must retain their own notices when acquired.
- ScienceWorld revision `e8216d6044e8e39be9fcb185e3b2dfb602584b52`;
  [Apache-2.0 license](https://github.com/allenai/ScienceWorld/blob/e8216d6044e8e39be9fcb185e3b2dfb602584b52/LICENSE).

Selected retrieved-text SHA-256 values:

| Repository/file | SHA-256 |
|---|---|
| ALFWorld `LICENSE` | `0bdf8c0558499c192b6ab55818e99e91ef0eb02c6e2d19d907b3f2e14df40590` |
| ALFWorld `scripts/README.md` | `3117ab081cd42d5a28ac644f238645d60d8f4595a47052c2ef040473803e81b2` |
| ALFWorld `alfred_tw_env.py` | `4d9166363e32f0fa70c63530a900785ddb45c79387a02ebcb9a21252869a8924` |
| ScienceWorld `LICENSE` | `d85fd829c236347c5e1febbf2ba844f706dd34e4a3ca4c43b9b1e5fa1667fadf` |
| ScienceWorld `README.md` | `2cdb71716ca2b4b6a34bdadf67f4673b511e489bb2c4143fe0b6be6a5aef7d73` |
| ScienceWorld `TaskUseInstrumentThermometer3.scala` | `518eb5bfb2719b729481af9ad5a946d694bca80dd65240cf5f88fbd9186dd1b5` |
