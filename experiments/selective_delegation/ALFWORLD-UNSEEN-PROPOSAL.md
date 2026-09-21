# Proposed unseen-game qualification, not a new architecture

CPU preparation only. Source035 is reserved, but no runtime, decision acceptance,
or GPU launch is provided here. Review source032's completed local-reason result
before accepting this comparison.

The question is whether the exposed eight-game manager advantage survives new
official unseen games and broader task families when compared with local
per-action deliberation. Reuse source026's exact flat/manager functions and
source032's exact local-reason function, frozen base model and native client.
This is baseline/generalization qualification, not new hierarchical planning.
The novelty constraints in `LITERATURE-UPDATE-1721.md` remain binding.

## Frozen selection and actual public resets

The official filtered `valid_unseen` inventory has 134 games: 18 light-use,
24 placement, 31 cleaning, 21 cooling, 23 heating, and 17 two-object placement.
Within each native family, rank by SHA256(`2026092194:` + dataset-relative game
path) and retain the first two. No result, scene, annotation policy, reset success,
or length filter was used. No previously exposed seen/manual game path overlaps.

| Slot | Task family | Scene | Public goal |
|---|---|---:|---|
| 0 | Light use | 308 | Examine alarmclock with desklamp |
| 1 | Light use | 308 | Examine CD with desklamp |
| 2 | Placement | 219 | Put vase in safe |
| 3 | Placement | 219 | Put watch on safe |
| 4 | Cleaning | 10 | Clean bowl, put in cabinet |
| 5 | Cleaning | 10 | Clean plate, put on countertop |
| 6 | Cooling | 10 | Cool pan, put on countertop |
| 7 | Cooling | 10 | Cool mug, put in cabinet |
| 8 | Heating | 10 | Heat mug, put in coffeemachine |
| 9 | Heating | 10 | Heat apple, put in fridge |
| 10 | Two objects | 424 | Put two soapbars in garbagecan |
| 11 | Two objects | 424 | Put two soapbars in cabinet |

All 12 official text-engine resets succeeded, initially `won=false`. No model
calls or environment actions were performed. Initial observations contain
366–523 characters and 16–30 native admissible commands. Only four distinct
scenes occur; six games share kitchen scene10. **Do not treat these as twelve
independent scenes or silently reselect them.** The balanced two-per-family panel
also does not estimate the natural full-benchmark task mixture.

Immutable inventory/public-reset receipts:
`R/alfworld-unseen-inputs-001/MANIFEST.json`, SHA-256
`4d69b397bb98e77f988af445f7fbdb7310e0369531495f7df4886d575cbcd7a9`.
`R` is `/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.
The preparation script is `prepare_alfworld_unseen_panel.py`; its bytes, original
readiness manifest, selected game files, bridge, environment lock and each raw
public-reset receipt are hash-bound. Two harmless line-length lint warnings were
left after preparation to preserve the exact manifest-bound script bytes.

## Small accepted-candidate design

Twelve games × two proposed fresh seeds (2026092195/2026092196) × three policies
gives **72 planned episodes**, at most two cumulative hours. Retain 50 executed
actions, 2,048 generated tokens counting every manager/reason/worker output,
128 tokens per request, 8,192 context, temperature .5/top-p1/top-k0, and the
existing per-action/manager seed progression. Native `won` is the outcome.
All three arms receive identical admissible-command assistance; no hidden state,
reward, native goal family, game path, gold plan or host labels enter prompts.

Manager refresh remains every four actions; local reasons remain one-call,
reason-first JSON and do not persist into accepted-action history. Preserve
source026 rejection feedback, strict parsing, no repair/fallback, and three
consecutive-invalid stopping. Preserve the exact shared neutral history trimming;
the local-reason arm must fail explicitly rather than trim extra context.
The first fixed game/seed across all three arms is a smoke check inside the
accepted budget, using an executed-action gate rather than a win gate.

Primary contrasts: manager minus flat, local reason minus flat, and manager
minus local reason, all 24 slots per arm. Report unknown versus observed failure,
protocol, token/action caps, calls/tokens/latency and both seeds. Bootstrap games
with repeats together, but show scene-cluster sensitivity and emphasize that
four clusters cannot support precise generalization claims. Family results are
descriptive. Do not train on these unseen evaluation games.

Promote only if an informative advantage persists beyond protocol differences
without unacceptable cost; mixed/uncertain results motivate a narrow interface
control, not an automatic hierarchical optimizer. A null result retires the
exposed-game advantage as insufficient architecture evidence.
