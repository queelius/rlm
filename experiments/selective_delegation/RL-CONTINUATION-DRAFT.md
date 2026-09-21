# Conditional eight-update continuation, preserving the optimizer

September21,15:00UTC. CPU preparation accepted; **no GPU training accepted yet**.
TRAIN fit rose45→49/64 with four both-valid wins and no losses on16TRAINparents;
fresh performance rose53→56/128 but remains uncertain. Finish repeated-execution
and plan-only diagnostics before accepting a training launch.

Extend the existing `rl_planner.py` flow, not a second training framework. A new
explicit continuation option loads the completed `rl-fullpass-001/checkpoint-0016`
into a fresh output directory. Preserve its LoRA weights, Adam state, Python/Torch/
CUDA RNG state and global update number. Do not alter the ancestor directory,
PLAN, checkpoint or sealed source. This is not an ordinary resume of that run.

The intended endpoint is global update24: eight additional fresh on-policy
updates,16parents x4candidate plans each =512newtrajectories, at most5,120calls,
with expected wall time about40–45minutes and a hard90-minute cumulative cap.
Keep learning rate2e-5, binary terminal-EM RLOO, rootT=.8, downstreamT=.5,
128/384/128generation caps, frozen separate helper36 and adapter-disabled base
final. Keep per-batch admission unchanged. Save weights/optimizer/RNG after
every update; last committed endpoint is selected by cap/stopping, not dev score.

Parent schedule: first16blocks reproduce the ancestor's sorted-then-shuffled
256TRAINparents with seed2026092108. Blocks17–24 take the first128parents of
a second deterministic shuffle with seed2026092109. Record the complete schedule
and actual repeated-parent counts. Use global update17–24 in the existing seed
function; never recycle update1 seeds. Existing common downstream seeds across
candidates and helper2/final seed collision are deliberately retained.

The new PLAN binds ancestor PLAN/STATE/COMMIT hashes, original schedule prefix,
starting step16, optimizer/RNG identities, additional-update count, caps, and
frozen component contract. Validate ancestor component identity against its own
PLAN before mapping state to the new planned schedule. Reject mismatched helper,
base, optimizer hyperparameters, incomplete ancestor, or divergent schedule
prefix. An explicit resume of the new continuation must use its own immutable
PLAN/committed checkpoint, not silently restart at16.

Focused CPU checks should establish old16-update behavior unchanged, exact
second-pass ordering/global seeds, source-contract rejection, optimizer/RNG
restoration through the existing code path, and resume/update boundaries. Inspect
the actual completed checkpoint16's small state and optimizer/RNG metadata on
CPU. Do not load the4Bmodel or run GPU work during preparation.

Conditional readout: compare the endpoint against already frozen checkpoint16
on allfresh00364parents x2 under identical helpers/final/caps/seeds. Reuse the
same immutable panel as development, not a new held-out test. Include direct and
SFT controls already collected, actual compute, both-valid changes and any
checkpoint truncation. A further fresh independent panel is warranted only
after a credible effect; no held-score checkpoint selection or broad sweep.
