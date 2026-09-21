# ALFWorld screen: command admissibility dominates this tiny comparison

Completed 2026-09-21 at16:34:48UTC. All32 episodes observed; no missing outcomes,
inference failures, unresolved starts or unlinked calls. The first real responses
and two-arm executed-action smoke check passed within about50seconds of startup.
This was frozen base4B inference, not RL or a learned hierarchy.

| Measure | Flat | Manager/worker |
|---|---:|---:|
| Native wins / planned episodes | 1/16 | 2/16 |
| Three-consecutive-invalid stops | 13 | 12 |
| 50-action cap stops | 2 | 2 |
| Generated-token cap stops | 0 | 0 |
| Invalid command outputs | 40 | 36 |
| Executed environment actions | 250 | 230 |
| Model calls | 290 | 334 |
| Input tokens | 288,171 | 329,502 |
| Output tokens | 2,997 | 4,166 |
| Total tokens | 291,168 | 333,668 |

Manager/worker gained two paired game/seed outcomes and lost one; both gains
were repetitions of the **same game**. Difference +6.25percentage points,
exploratory95% game-bootstrap interval [−18.75,+37.50]. Bootstrap:20,000draws,
seed2026092180, resample eight games with both seeds together. Eight exposed
seen-development games are not16 independent examples, and scene/component
independence is not established. This does not demonstrate a hierarchy benefit.

All76 invalid outputs were **well-formed JSON but commands absent from the
current admissible list**; none were JSON/schema failures. Lists were supplied
in every public prompt and native requests/actions/feedback reconstructed
successfully. They provide affordance information, not merely syntax hints;
this assisted setting should not be described as unassisted ALFWorld.

Three representative paired traces explain why an interface control and a
semantic control are different:

- Game1, seed2026092178: flat found and took the tissuebox, then tried to move it
  to the toilet while still at the drawer, repeating the inadmissible command
  three times. Manager/worker navigated to the toilet and completed the task.
- Game3, seed2026092179: flat put the mug on the sidetable in five actions.
  Manager/worker instead collected a cellphone, repeatedly examined furniture,
  moved the wrong object, then repeated an inadmissible move. Command compliance
  alone would not correct the wrong-object strategy.
- Game6, seed2026092178: both policies spent long stretches repeating admissible
  examine/look actions. Manager/worker reached the50-action cap without any
  invalid output. Preventing inadmissible actions therefore need not solve the
  underlying search/no-progress problem.

The first smoke episodes show the same distinction: flat repeatedly tried to
go to its current cabinet; worker repeatedly tried to go to its current fridge.
Rejected responses are logged but are **not included in subsequent public
history or feedback**. The next request therefore sees unchanged public context
with a different sampling seed. This matches the sealed no-repair contract;
it is a material interface limitation, not a broken native environment.

Total acquisition:624calls,624,836tokens,310.4summed model-service seconds,
441.1wall seconds. The hierarchical total includes68manager calls and66,926
manager tokens; they are not free. Neither policy approached the2048-output-token
ceiling often enough to terminate there. Task types were four placement games,
two heating games and two cleaning games; these labels remained host-only.

## Smallest conditional follow-up

Before further hierarchy or RL claims, test a frozen **flat-only indexed-command
interface** on the same eight games/two seeds against these saved flat outcomes:
identical public information and budgets, explicit stable per-state command-ID
mapping, strict JSON ID output, no repair or invented commands. Charge all calls
and any constraint machinery; do not conflate indexed prompting with guaranteed
valid decoding. This isolates command selection more cheaply than retraining or
adding another manager. It is not yet accepted or implemented.

Separately exposing the previous invalid attempt and its rejection is a different
intervention; do not combine it with IDs in the first comparison. Promote an
interface only if it reduces protocol stops **and** improves native task progress/
success without hiding extra cost. If it merely turns early stops into repetitive
50-step failures, retire the interface explanation and address search/memory or
base-agent competence. A better interface on these seen games is not generalization.

## Immutable evidence

R=`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.

- Native source/output: `R/source-022-alfworld/`, `R/alfworld-screen-001/`.
- Reproducible analyzer/test: `R/analysis-source-alfworld-001/`; imports the sealed
  source022 dependencies using PYTHONPATH. Three focused fixtures passed, including
  actual native smoke reconstruction. No live collector edits or GPU launch.
- Official native-won readout: `R/analysis-alfworld-screen-001.json` and `.md`.
  JSON SHA256 `c7276230cc750a0ce30ee68e5f94c669b99778ba88d9ac79871630cc44831953`.
- Analyzer SHA256 `151bc1c196e981f4db4d04637a890b91f649d325b2ee2cb8b212747d107cad1f`.
- Collector PLAN SHA256 `98b43576837e59f2616bcd9f2f883875ff1612424c51ec1ece35d66957e00fe0`.

The JSON retains individual native source/receipt hashes, per-role costs, strict
termination counts, paired outcomes, public trace audits and bootstrap method.

### Main's subsequent preparation decision

The single-intervention recommendation above is retained as the historical
readout. Main instead requested CPU preparation of a **combined numbered-action
interface plus explicit rejection feedback**, shared by flat and manager/worker,
on all32 original game/seed/policy slots. This qualifies a normal closed-loop
agent interface before spending on hierarchy or RL; it is not a claim that
either change alone causes improvement. No game reselection, new benchmark or
novelty claim is intended. A later strong result would require separate mechanism
ablations; none is inferred from this combined qualification. GPU acceptance and
launch remain separate main-owned decisions.
