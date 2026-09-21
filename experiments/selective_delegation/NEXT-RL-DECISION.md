# Next RL decision: isolate reward reliability before extending dose

## Current evidence

`rl-planner-001` completed its precommitted four fresh on-policy updates: terminal
EM rewards were 37/64, 36/64, 40/64, and 37/64. Every root returned, the adapter
changed after every update, and the admission rule passed (3, 5, 4, and 6
qualifying parent groups). This establishes non-flat observed rewards and real
updates, not a learning curve: each batch contains newly sampled plans and only
16 exposed train parents. Invalid helper paths also contributed explicit reward
zeroes.

The most decision-relevant evidence is already queued: frozen SFT-versus-RL
held-out readouts and the full-source versus trace-only final aggregation probe.
Read those before selecting another checkpoint or launching more planner RL.
PyRAG's role-by-role findings make executor quality a credible alternative to
planner dose; they do not make a planner-RL claim novel.

## Ranked next action

1. **Interpret queued held-out and aggregation results; do not extend RL first.**
   The aggregation probe uses the actual 58 valid batch-one helper traces, two
   fresh final seeds, and full-source/trace-only finals. It is the smallest test
   of final bypass: documents can either rescue a bad helper trace or let the
   final answerer bypass it. Its 24 invalid-helper source rows remain headline
   zeroes, while conditional trace variation excludes them. This is not a test
   of improved planning.

2. **If the result remains ambiguous, run one no-update, two-final-repeat
   on-policy reliability batch before any dose extension.** Fixed current RL-4
   checkpoint; fixed same 16 train parents; four fresh root plans each at the
   existing temperature; frozen isolated helpers; one actual helper trace per
   plan; two paired final seeds per trace. Do not optimize. Primary outcome is
   each candidate's two-repeat EM disagreement and the variance of candidate
   mean EM within parent. This separates terminal-final sampling noise from
   variation that survives a repeated final answer. It does **not** prove that
   greater between-plan variance is useful credit: candidate differences still
   include plan length and frozen helper realizations.

   Expected work is about 64 roots + about 128 isolated helper calls + 128 final
   calls, roughly 320 model calls (about 5--8 minutes at the observed
   `rl-planner-001` throughput). Stop without another optimizer step if fewer
   than two parents have mixed candidate *mean* EM, or if most apparent
   single-final mixed groups disappear after the second final. Pivot to executor
   diagnosis rather than averaging noisier rewards into another update.

3. **If final-repeat reliability is adequate and held-out RL-4 is not clearly
   worse than SFT, consider a predeclared dose comparison:** continue the fixed RL-4 checkpoint for
   exactly four further fresh updates on the same 16 parents, then compare fixed
   RL-4 versus fixed RL-8 once on untouched validation parents with matched
   seeds/execution. This is a dose diagnostic, not a generalization estimate;
   it costs approximately the original 1,053 calls and 17 minutes plus the
   held-out readout. The current64 validation parents will all have been read;
   genuinely fresh validation requires a new label-blind, component-separated
   panel. Otherwise label a repeated panel as reused development data. A null
   small-sample difference is not proof that dose cannot help; use it to rank
   alternatives rather than launching an open-ended dose sweep.

   Trace-only need not outperform full-source finals for planner RL to be viable.
   A final model that repairs mistakes may be the better deployment interface.
   Interpret the aggregation result as a mechanism diagnostic, not an automatic
   admission requirement for all subsequent RL.

## Falsifiable branches

| Observation | Most supported limitation | Next action |
|---|---|---|
| High paired-final disagreement; mixed groups collapse on repeat | Noisy terminal reward | Do not add planner updates; consider repeat-mean reward only after a fresh precommitted budget decision. |
| Trace-only reduces EM or fails to preserve candidate differentiation; full source rescues helper mistakes | Final/executor bypass or weak helper evidence | Freeze planner RL; improve/test executor or helper before retraining planner. |
| Reliable repeated rewards and trace-sensitive differentiation, but RL-4 is flat versus SFT | Dose is plausible but unproven | Run the single RL-4 versus RL-8 comparison above. |
| Reliable repeated rewards, trace-sensitive differentiation, and RL-4 improves held-out | Planner signal exists | Replicate on fresh train parents before transfer; do not select checkpoints on transfer. |

## Contamination and reporting constraints

Keep all Hotpot explorer and MuSiQue four-hop transfer labels sealed until fixed
checkpoint and protocol choices are made. Do not place gold answers, aliases,
support labels, reference decomposition, or reward history in planner/helper/final
prompts. Treat the 16 RL parents as exposed training data, not confirmation data.
The Hotpot explorer sample is a small official-sample diagnostic with possible
pretraining contamination, not canonical-dev or clean-OOD evidence. Report all
missing/invalid calls separately; never convert transport failures into reward
zeroes or choose a favorable mixed-reward coordinate after collection.
