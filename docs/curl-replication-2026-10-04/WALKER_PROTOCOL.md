---
date: 2026-10-04
status: prepared_not_launched
question: Does the early benefit of the extra image-matching update extend to a walking task?
primary_endpoint: 100000_training_simulator_steps
training_seeds: [123, 456, 789]
external_campaign: /project/alex_phd/runs/curl-walker-replication-20261004
---

# A second task: learn to walk from images

The first cartpole comparisons favor CURL after short training, but the two
completed longer comparisons have different winners. We should check the scope
of the early result before building an explanation around one later reversal.
Walker/walk asks a simulated body to stay upright and move forward. It changes
the control problem while keeping the paper's learner and our matched comparison.
This is another reproduction experiment, not an RLM result or a novelty claim.

## The comparison, chosen before walker results

Train CURL and the same-crops/no-contrastive control from scratch for each seed
123, 456 and 789. Pair the settings and seeds within walker; matching a seed
number across tasks does not create identical experience. Run all three declared
pairs regardless of the first pair's score. Do not initialize from cartpole
weights or choose a favorable checkpoint.

The primary score is mean episode reward from ten fixed starts, 10000--10009,
at the final 100,000 training simulator steps. Report individual seeds, paired
differences and learning curves. Higher reward is better, not a percentage.
Evaluation does not update weights; its interactions and time are separate costs.
Incomplete or failed runs have no final score, rather than a zero score.

## Settings and what is different

Use walker/walk with action repeat 2. Thus 100,000 simulator steps require
50,000 model decisions, including 1,000 initial random-action decisions, leaving
49,000 ordinary update calls. Keep 100-by-100 rendered images, 84-by-84 crops,
three stacked frames, four convolution layers, 32 filters, 50-dimensional
features, batch 128 and replay capacity 100,000. Actor, critic and encoder
learning rates stay 0.001. Other learner settings remain in the pinned manifest.
These task-specific choices follow [supplement Table 3](https://proceedings.mlr.press/v119/laskin20a/laskin20a-supp.pdf),
not the repeat-4 setting used in some supplementary ablations.

Evaluate every 2,000 decisions, so evaluation points remain 4,000 simulator
steps apart, as in the cartpole campaign. This schedule is our declared choice,
not a verified historical-paper setting. It yields 26 batches including initial
and final evaluation, or 260,000 evaluation simulator interactions per run.
Each natural walker episode is 500 decisions / 1,000 simulator steps.

The paper reports CURL **403 (SD 24)** at 100k and **902 (SD 43)** at 500k,
across ten training seeds. These are context, not acceptance thresholds for our
three-seed modern-stack experiment. Our same-crops control is not necessarily
the paper's Pixel SAC baseline. [Paper Table 1](https://proceedings.mlr.press/v119/laskin20a/laskin20a.pdf)

## Compute, checkpoints and launch boundary

Use the existing single A100, sequential jobs, the existing shared GPU lock,
a two-hour cap per run, and the current allocation deadline 1791387366.
Checkpoint every 900 seconds at a natural episode boundary and at completion.
Record native rewards, updates, errors, timing, config, source hashes and runtime.
The original source commit remains `8416d6e3869e38ca0e46fcbc54a2f784dc09d7fc`.
Keep the currently tested modern environment and document its differences from
the partly unpinned historical environment; do not claim an exact historical run.

Planning estimate: 45--90 minutes per run, around 4.5--9 GPU-hours for six;
measure actual duration on the first pair. Allow about 56 GB for final replay
checkpoints and at least 10 GB temporary-save headroom. The existing protocol-4
save repair matters here: one pixel array exceeds 4 GiB by the 100k endpoint.

The driver already supports walker. The queue now forwards domain, task
and action repeat and checks those fields when recognizing completed jobs.
That small change passed nine focused tests and independent review, separately
from the sealed live cartpole sources.
Use a distinct campaign and source snapshot. Do not start a competing owner
while the current cartpole owner still has work. No walker run has launched
as of this protocol. Cartpole-specific continuation admission remains unchanged;
any later walker continuation requires its own explicit protocol.

The six-job queue and source snapshot are prepared at
`/project/alex_phd/runs/curl-walker-replication-20261004`. Source commit:
`bad5b90a3d8113faad987ff034806282b7f05888`; the task-specific manifest changes
and file hashes are recorded in its `SOURCE.json`. The simulator check completed
a natural 500-decision / 1,000-step episode with six action components and the
expected image stack. This checks the adapter, not learned walking performance.
Its `HANDOFF.md` records the launch boundary and observer update still required.

## How the evidence will change our plan

If all or most paired early effects also favor CURL, the result has broader
support within this small two-task study. If they are mixed or favor the control,
limit the earlier claim to cartpole and investigate what differs. Neither case
alone establishes why the effect occurs. Only then decide between more training
seeds, a longer walker horizon, or a shared-checkpoint test of whether continuing
the matching objective helps. Do not turn a favorable single seed into a claim.
