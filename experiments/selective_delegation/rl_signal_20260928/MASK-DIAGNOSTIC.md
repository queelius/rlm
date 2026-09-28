---
date: 2026-09-28
status: cpu_design_only_not_implemented_or_queued
question: does_overwritten_ingredient_payload_gradient_help_or_hurt_one_binder_update
prior_art:
  name: Harness-RL
  arxiv: 2608.29641
  version: v1
  submitted: 2026-08-30
  checked: 2026-09-28
---

# One same-binder-batch ingredient-payload gradient diagnostic

Recommendation: make this a small mechanism ablation **after** the binder census,
not a new collection or a substitute for the queued compact-interface RL cycle.
The raw collection has substantial ingredient output, but that alone proves
neither harmful gradients nor negligible semantic exploration.

## Smallest interpretable comparison

Reuse the completed first binder collection under the actual public-discovery
checkpoint23. Compare its already planned full-token one-update endpoint with one
new endpoint started independently from exactly that same warm actor and fresh
AdamW state. One masked update uses the same 32 episodes, original sampled token
IDs, terminal rewards, leave-one-out advantages, FP16 base/FP32 LoRA, temperature
0.5 likelihood, LR2e-5, clipping1, and trajectory denominator32. No replay of the
full-updated checkpoint and no second optimization of either arm's batch.

Reuse the warm/full-update binder readouts on the fixed TRAIN8 diagnostic seeds
2026092890/891. Add only the masked endpoint's 16 paired native episodes. No new
rollout batch, sweep, reward change, target replacement, or inference-harness
change. A separate additive adapter would be required; frozen trainers stay
untouched. This document is not a launch descriptor.

## Mask contract

Call the intervention **ingredient-payload-off**, not generic decision-only RL.
On a schema-valid craft whose saved native assist says
`bound_observed_single_recipe`, zero only loss terms whose exact emitted token
offsets lie wholly within the JSON **value** of `ingredients`. This is independent
of native craft success: eligible insufficient-stock attempts remain eligible.
The recipe must have been uniquely observed through prior public `get_info`, and
the requested output count must be divisible by its yield. No hidden recipes.

Keep all other loss terms: the fixed `ingredients` key/colon, tokens crossing the
value boundary, EOS, action/target/count, queries, finish, malformed actions, and
every unbound or nondivisible craft. Keep the complete original token sequence as
teacher-forced context, including masked tokens. A decoder must locate the JSON
value span; naive substring search is insufficient. Re-encoding must exactly
match saved emitted IDs before using offsets. Reject unalignable eligible calls
at CPU admission rather than silently dropping them or changing denominators.

For episode `i` and emitted token `t`, the minimization target is
`−sum_i A_i sum_t m_it log pi_T0.5(y_it | original_prefix_it) / 32`.
No division by retained-token count or per-trajectory length. Record eligible
calls, masked/retained token counts, positive/negative/zero credit, absolute
advantage-weighted token mass, and boundary-token counts before GPU dispatch.
The raw 44.14–49.19% whole-member measurement is **not** the expected masked dose:
this conservative value-only mask retains keys/boundaries and applies only where
the binder actually overwrites ingredients.

## What the diagnostic can and cannot mean

Full-token REINFORCE remains the valid baseline for the fixed raw-output MDP.
Ingredient tokens precede and condition target/count tokens autoregressively;
JSON validity and consumed budgets also depend on their generation. Therefore
the masked update is a **biased surrogate/gradient intervention**, not an unbiased
gradient of executed-action likelihood. Exact executed-action likelihood would
require marginalizing raw preimages. Do not advertise masking as a theorem or a
correction to the current objective.

The tight question is whether this biased intervention gives a more useful local
update under the same binder. First record unclipped gradient norms and cosine
between retained and removed gradient components if obtainable with one extra
backward pass; otherwise omit the conflict mechanism claim. Parameter movement
and post-update same-prefix log-probability deltas on retained versus ingredient
positions are descriptive diagnostics, not task success. Adam/clipping make
token counts alone insufficient evidence for gradient domination.

Primary outcome: masked-minus-full native successes across the 16 paired readout
episodes; also compare each to warm, task-cluster differences, calls/output tokens,
schema errors and premature finishes. Eight exposed task clusters remain
exploratory. As a prospective screening rule, promote only a net gain of at least
2/16 spanning at least two distinct tasks, with no schema-error increase; require
a fresh-task replication before efficacy claims. Retire this one-step candidate
if it ties or loses to full-token training, if only token log-probabilities move,
or if apparent benefit comes with more schema failures. A tie does not establish
global equivalence of objectives.

Skip GPU work if there are no mixed binder groups, no eligible nonzero-credit
payload tokens, a failed native audit, or an unavailable/numerically invalid full
baseline. Expected incremental cost is one A10040GB for about 15–35 minutes of
update/diagnostics plus 15–30 minutes for the paired readout; hard caps60/45minutes.
These are estimates, not measured binder throughput. Gradient decomposition can
add another backward pass and must stay within the update cap. Existing warm/full
jobs are reused, not rerun to match a preferred outcome.

## Prior art and relation to queued compact RL

Harness-RL already motivates action/argument gradient interference and uses
activation-derived parameter partitions to route their token gradients. It also
aligns saved interface calls with outcome/process rewards. Its future work asks
about partition robustness/cost and more varied harnesses. Thus broad novelty
for separating semantic decisions from argument updates is **low**. This proposed
payload removal is not CAPO: no learned partition, no parameter routing, and no
new process reward. Its narrower value is a cheap same-batch intervention where
the public harness deterministically overwrites one argument field.
[Harness-RL v1, §§3.2–3.3 and5](https://arxiv.org/html/2608.29641v1).

The already queued compact RL cycle changes the warm SFT actor, output schema,
instructions/history, and fresh TRAIN groups; it also removes ingredient
generation cost. This diagnostic keeps those fixed and changes only loss terms,
so it addresses a different, narrower optimization question. It does not remove
inference tokens or contextual dependence on ingredients. Even if both improve,
that alone does not isolate a universal interface principle. See the
[compact RL contract](../rl_compact_20260928/README.md).
