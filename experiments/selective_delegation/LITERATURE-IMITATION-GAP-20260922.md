---
status: literature_informed_followup
reviewed_utc: 2026-09-22T09:40:00Z
question: What would distinguish our teaching result from known information-asymmetry failures?
gpu_acceptance: none
---

# A better teacher is a useful finding, but not yet a new general principle

The completed [crafting comparison](TEXTCRAFT-PUBLIC-DISCOVERY-FINDINGS.md)
motivates a closer check of imitation-learning research. The broad failure mode
is established: a teacher can choose actions using information the learner lacks.
We should not present our result as discovering that problem.

## Closest verified primary sources

**ADVISOR, NeurIPS 2021.** Its examples show how copying a fully informed teacher
can yield poor behavior under partial observation. Its policy-averaging result
formalizes the mismatch. The proposed method uses an auxiliary policy to weight
imitation versus reinforcement-learning losses, rather than always imitating.
The paper also distinguishes this from adapting the teacher itself.
We inspected the introduction and §§2–3.1; we have not reproduced the method.
[Official paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/9fc664916bce863561527f06a96f5ff3-Paper.pdf)

**Adaptive asymmetric DAgger, ICML 2021.** This work explicitly addresses an expert
that does not account for what the learner cannot observe. It jointly trains
the expert and learner so that the expert becomes more useful to the learner.
This description is based on the official proceedings abstract, not an audit
of its implementation or all experimental details.
[Official proceedings](https://proceedings.mlr.press/v139/warrington21a.html)

**LEAP, ICLR 2025**, already covered in our [teacher-information review](LITERATURE-TEACHER-INFORMATION-20260922.md),
is the nearer language-agent comparison: student-realizable expert feedback and
corrections on student trajectories are prior art. The older results above mean
that simply calling our teacher “observation-aware” would not establish novelty.

## What our data support, and what they do not

The old crafting teacher does not expose its hidden plan as a model input.
Instead, that plan influences which query it demonstrates next. Arbitrary recipe
queries are legal. A student might learn useful associations in the fixed world;
we have not proved the old training task mathematically unrealizable.

Our [paired-world CPU example](TEXTCRAFT-OBSERVATIONAL-AMBIGUITY.md) is sharper:
identical initial public inputs receive different privileged first-query labels,
while querying the visible goal works in either world. But those different labels
are not uniquely necessary actions. This demonstrates a supervision ambiguity,
not impossibility of successful imitation or a causal proof about our model.

The current SFT comparison changes an entire demonstration procedure. Its gain
could involve discoverability, ordering, prompt history, or their interaction.
It is therefore a useful controlled teaching-package result, not yet a theorem
or a new algorithm.

## Falsifiable route toward a stronger contribution

1. **Cheap alternative:** can an explicit procedure prompt repair the old model?
   Queue061 is already running; a positive result narrows the value of retraining.
2. **Transfer:** does the gain persist for changed recipes and unseen goals?
   Queue062 changes recipes; a separate outcome-blind fresh-goal panel is in
   preparation. Neither alone is a new independent task domain.
3. **Mechanism:** on a separate training family, hold the initial public input
   fixed while varying hidden recipes. Compare demonstrations that acquire the
   missing facts with demonstrations that directly use teacher-only knowledge.
   Control action/token exposure and measure both completion and information
   gathering. Do not train on our already examined validation counterexamples.
4. **RL connection:** first check whether a public-discovery warm start produces
   both successful and failed TRAIN rollouts under terminal task reward. If so,
   compare a small RL continuation against equally budgeted extra SFT. This could
   test whether a better starting procedure makes task reward informative; it
   would not itself reproduce ADVISOR or establish an RL-specific advantage.

Items3–4 are proposals, not accepted GPU jobs. Favor a bounded one-A100 pilot
after the queued controls, with fixed held data and checkpoints. Retire the broad
transfer story if neither changed recipes nor new goals retain the advantage;
retain the narrower result and investigate execution failures instead.
