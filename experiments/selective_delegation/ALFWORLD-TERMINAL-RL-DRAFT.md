# Small terminal-reward ALFWorld actor-RL screen (prospective)

This is conditional on a clean one-epoch action-SFT endpoint (`041b`) and the
fixed-checkpoint flat/manager trained-actor readout (`042b`). It is not a queued
GPU job, a data freeze, or a claim about the active 035 results. The narrow
question is whether a competent **flat** public indexed-action policy supplies
enough within-game terminal-reward variation for a real on-policy update. Starting
flat avoids assigning an effect to manager goals, worker routing, or their joint
adapter behavior.

## Smallest credible comparison

Select 8 official TRAIN games label-blind, with a fixed hash seed and no overlap
with demonstration games or any held/development game. For each game, reset the
same native public state four times and sample one flat trajectory per candidate
from the current policy (the fixed action-SFT checkpoint for block one). The policy sees exactly the existing public
indexed-action prompt: initial/current feedback, executed public history, and the
admissible-command list. It never sees expert actions, PDDL, task family, reward,
`won`, or an oracle plan. Keep temperature 0.5, action cap 50, total output cap
2,048, per-call cap 128 and context cap 8,192. Native `won` is host-only terminal
reward `R in {0,1}`.

Use two predeclared four-game groups (16 trajectories each), each with a fresh
on-policy rollout and then a possible update. Do **not** retain only mixed groups:
all four candidates from every selected parent enter acquisition, accounting and
the block-level objective. A uniform group has zero RLOO signal but remains a
costly observed result. Skip its optimizer step only if the **whole four-game
block** has zero advantage; a mixed block includes uniform groups with zero weight. A
matched action-SFT control begins from the identical checkpoint and receives the
same number of **real** optimizer updates on fixed public-only demonstration rows.
Use the qualified AdamW contract (LR 2e-5, weight decay 0, clip 1, dropout 0), not
a new SGD optimizer. The comparison controls actual update count and adapter
initialization, not environment rollout cost or supervised-token count. Neither
arm uses replay from a prior policy, PPO clipping, expert targets in rollouts, or
a manager/worker update.

For candidate `k` in game `p`, score every emitted action-response token plus EOS
along the realized trajectory, including malformed JSON or controller-rejected
responses. Do not condition the loss on strict-valid actions only: those observed
responses are part of the sampled policy trajectory. The existing three-consecutive-
invalid rule remains an episode termination rule, not a one-invalid-action terminal.

`A[p,k] = R[p,k] - mean(R[p,j] for j != k)`

`loss = - mean_{p,k}(A[p,k] * sum_{action calls and emitted tokens} log p_theta(token | public prefix))`.

Replay uses the same action policy temperature: logits divided by 0.5 before
token log-softmax. Record generated token ids, prompt ids, action boundaries,
EOS, trajectory/action lengths and the exact checkpoint digest, so the sampled
sequence and replay score are auditable. The trajectory token-logprob sum is the
standard terminal-REINFORCE score estimator; there is intentionally **no arbitrary
length normalization**, which would change the objective. Variable trajectory
length can nevertheless increase variance and make action credit harder, so report
its association with advantage/reward and emitted action-token count. Separate an
observed invalid policy action/schema failure (a terminal outcome under the
controller contract) from unavailable/unknown inference or environment failure;
never silently turn the latter into reward zero.

## Cost and decision rule

Source026's completed flat arm used 781 calls, 1,366,335 prompt tokens and 6,880
completion tokens for 16 trajectories: about 48.8 calls, 85.4k prompt tokens and
430 completion tokens per trajectory. Thus this 8-game × 4-candidate screen is
about 1,562 calls, 2.73M prompt tokens and 13.8k completion tokens. Its flat
closed-loop work was roughly half of a 923s mixed-policy 32-episode run, so a
conservative one-A100 allocation is 30 minutes for rollouts plus a small margin
for at most two replay/AdamW updates; cap the whole job at 45 minutes. A 12-game extension
is 2,343 calls and 4.10M prompt tokens, so it should not be silently appended.

After the first four-game rollout, log its four-candidate reward histograms and
valid-action diversity. If it has no mixed available-reward group, preserve that
observed block and skip its update. Otherwise require a finite nonzero replay
gradient before its update and record actual finite nonzero adapter movement after.
Roll out the second four-game block
under the resulting policy (or the unchanged policy if the first block was
uniform), then apply the same rule. These one or two updates only qualify whether
the terminal-reward implementation has usable signal; they cannot establish a
meaningful RL gain. If the two fixed blocks provide nonzero signal and allocation
permits, an explicitly separate 4--8-update expansion and matched-SFT comparison
may be proposed before any held evaluation. If fewer than two of the eight fixed
groups are mixed, report sparse signal and do not quietly append games or declare
terminal RL generally dead. Any learning claim still requires a fresh TRAIN block
or untouched development panel.

The earlier 16-slot flat source026 arm had only 1 win and mostly hit the action
cap, while manager-worker had 6 wins; this is why flat-only terminal reward may be
uniformly zero despite valid syntax. The result would test terminal-reward
availability for a public action actor, not novelty of hierarchy, ALFWorld
planning, or RL itself. Evidence links: `R/alfworld-closed-loop-001/SUMMARY.json`,
`R/alfworld-local-reason-001/SUMMARY.json`, and
`SUFFICIENCY-RL-CONDITIONAL.md`.
