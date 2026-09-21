---
retrieved_utc: 2026-09-21T18:03:00Z
status: primary_source_screen_not_reproduction
decision: retain_bounded_qualification_runs_do_not_claim_generic_novelty
---

# Literature that changes our next decisions

## Answering and declining to answer

[AWA-RL, July 12, 2026](https://arxiv.org/html/2607.10738v1) studies abstention
in search agents, with query-dependent rewards and a changing refusal penalty.
Its comparisons include mixed supervised examples and static refusal rewards.
This is close prior art for our observed answer/refusal tradeoff. Its setting
includes search, larger models and external judging; it does not establish that
our small paired-label objective will work. We should retain the bounded RL
versus additional-SFT comparison as a competency diagnostic, not present learning
to abstain as a new contribution. A useful next comparison, only if that pilot
has signal, is paired reward versus separate per-input rewards on identical
TRAIN pairs and sampling budgets. This would isolate the paired objective rather
than compare unrelated recipes. No new GPU job is accepted by this note.

## Evidence editing already has direct precedent

[MAVEN, July 2, 2026](https://arxiv.org/html/2607.02073v1) trains an editable
evidence memory and assigns credit to evidence-state changes. Its verifier uses
gold-answer likelihood, with a contrastive-answer alternative also examined.
Consequently, evidence memory plus intermediate rewards is not a sufficient
novelty claim for our harness. A sharper question would be whether a small worker
can request a specific missing fact, and whether the supplied fact—not merely
another model call—causes the final answer to improve. A matched irrelevant-fact
control and the same receiver would distinguish these explanations. That is our
proposed diagnostic, not a result reported by the paper or a proven novel method.

## Recursive training is an existing research program

[Recursive Agent Optimization, May 7, 2026](https://arxiv.org/html/2605.06639v1)
trains a shared policy across recursive calls. Its node reward combines task
success with a child-success bonus; local success can use exact verification,
a judge, or a root-outcome proxy depending on the task. Thus learning when and
how to spawn helpers, or adding local credit, is not by itself novel. Our current
fixed-depth question planner is substantially narrower. Before scaling toward
recursion, we need a case where the delegated computation has identifiable value
and the model learns a useful decision rather than a preferred output format.

The paper's synthetic crafting experiments use the same Qwen3-4B-Instruct-2507
model family as our current runs, with task depth varied independently of model
size. This makes the cached official environment a relevant future testbed.
Its published training and inference resources exceed our allocation, however;
a one-A100 pilot would be an adaptation, not a reproduction. The existing
[September 11 source inspection](../../docs/research-plans/2026-09-11-rao-code-notes.md)
already records the cached commit and implementation caveats; no duplicate clone
or new training framework is needed merely to revisit the idea.

## A useful warning, but not a universal routing theorem

[Reward–SNR acquisition study, August 11, 2026](https://arxiv.org/html/2608.10441v1)
compares noisy per-example acquisition effects and deployable routing in
recommendation tasks. Its oracle-versus-noise-placebo comparison is worth borrowing.
However, its displayed threshold is a test-power approximation for a nonzero
**average** effect. We should not adopt its broader interpretation as a universal
necessary condition for useful routing.

Our counterexample: let an observed feature identify two equally common groups.
Acquisition helps one by one unit and harms the other by one unit. The overall
mean effect is zero, yet routing only the first group gives a positive gain.
This mathematical example is our critique, not an empirical result. Therefore
our earlier uncertain mean gains neither establish nor rule out a useful router.
We need held-out conditional predictability and repeated-outcome or permutation
controls. An outcome-aware oracle is not a deployment policy.

## Operational decision

Keep the accepted GPU queue running. These papers sharpen the interpretation and
follow-up criteria; they do not justify abandoning useful controls for a large new
framework implementation tonight. The nearest potentially distinctive question is
what information must cross the delegation boundary, and when acquiring it changes
the answer or next action. Basic abstention training and basic manager-worker SFT
are prerequisites or controls, not publication claims on their own.
