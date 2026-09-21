---
status: primary_source_review
retrieved_utc: 2026-09-21T17:21:00Z
purpose: constrain_novelty_and_choose_next_interactive_experiment
---

# Interactive-agent research changes the novelty boundary

The new small ALFWorld result is worth investigating, but neither a manager,
learned subgoal switching, nor learned workspace access is a new idea by itself.
The following primary papers materially constrain our next claims.

## HiPER: learned switching is already an explicit design

[HiPER, version2, May29](https://arxiv.org/html/2602.16165v2) separates subgoal
decisions from primitive actions and trains them with credit assigned at their
respective time scales. Its interface explicitly chooses whether to keep or
switch the current subgoal. The [official implementation](https://github.com/JonP07/HiPER-agent)
is available. Our inference: replacing our fixed four-action refresh with a
learned switch is a baseline-worthy extension, not sufficient novelty. First
complete the simpler local-deliberation control; do not implement a new
hierarchical optimizer merely because the exposed eight-game screen is positive.

## EvoHarness-RL: learning how to use external state is also established

[EvoHarness-RL, version1, August5](https://arxiv.org/html/2608.05446v1) teaches
policies to access and update an external workspace for world information,
progress and reusable experience. It combines supervised demonstrations with
cost-aware RL on ALFWorld. This closely overlaps our broad harness/weights
co-adaptation motivation. Our inference: a memory tool plus RL is not enough
for a new contribution. Any proposed interface should answer a narrower
question about which information is needed, when it should be requested, and
whether simpler automatic bookkeeping achieves the same result.

## SPACE: variable-length action grouping needs a meaningful boundary

[SPACE, version1, September2](https://arxiv.org/html/2609.02042v1) learns action
chunks using boundaries induced from successful programmatic skills. It reports
failure modes of naive variable-length training: almost never grouping actions,
or grouping too many. Our inference: adaptive depth/length is not automatically
a useful action space. We should measure whether candidate boundary decisions
change success before investing in their RL optimization.

## Two useful training references

[GiGPO, version3](https://arxiv.org/html/2505.10978v3) uses repeated states
across trajectories to obtain finer-grained relative credit without a separate
critic. It is relevant if terminal rewards prove too sparse for our interactive
pilot; matching visible text alone should not be assumed to identify the same
underlying state.

[A Practitioner's Guide, version2, December6,2025](https://arxiv.org/html/2510.01132v2)
studies environment complexity, supervised initialization and reward density in
multi-turn training. It motivates starting with tasks that produce useful
feedback and tracking the training budget explicitly. Its recommendations do
not establish which optimizer is best for our model and interface; that requires
a local controlled comparison.

## Concrete decisions for our queue

Keep the accepted local-reason control and paired sufficiency training. Prepare
fresh examples before inspecting their outcomes. If an interactive benefit
survives, compare against a simple persistent-goal or public-memory control
before proposing a novel recursive controller. A training pilot should use
TRAIN games, retain zero-reward rollout costs, and evaluate new games separately.
Do not compare our tiny exposed-game rates with published full-benchmark rates.

These are research inputs, not reproduced results. No third-party code from
these papers was executed in this review. The failed v3 HiPER and v4 GiGPO HTML
lookups were corrected to the verified versions linked above.
