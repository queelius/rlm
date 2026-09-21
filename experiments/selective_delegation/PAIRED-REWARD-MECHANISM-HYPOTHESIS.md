---
question_id: rq:paired-reward-coverage
created_utc: 2026-09-21T18:16:00Z
status: prospective_hypothesis_before_rl_updates
trigger: completed_joint_vs_positive_only_sft_tradeoff
experiment: source037_paired_rl_with_source038_extra_sft_control
novelty: unestablished_objective_diagnostic
---

# Could paired reward protect the harder skill?

The joint supervised model often declines to answer even when the answer is
available. Its remaining weakness is therefore not simply learning to refuse.
One reason to test the accepted paired reward is that it pays only when the model
both answers the supported input correctly and declines the negative variant.
It cannot earn that reward by doing only the easier half.

For one question, let `p+` be the probability of a correct supported answer and
`p-` the probability of the correct negative decision. Our two sampled responses
are conditionally independent given the policy and their separate inputs. The
expected paired reward is consequently `p+ * p-`, with gradient
`p- * gradient(p+) + p+ * gradient(p-)`. Separately rewarding the two decisions
would instead optimize `(p+ + p-) / 2`.

When supported answering is much weaker, the paired objective gives its gradient
a larger relative coefficient. This is a mathematical property of the objectives,
not a prediction that shared language-model weights will improve both behaviors.
Sampling noise, model capacity, imperfect labels and cross-example interference
can still prevent progress. Pairing also adds variance because both sampled
responses must succeed. There is no novelty claim for multiplying task rewards.

## What the current run can establish

The accepted comparison tests paired RL against extra supervised examples, with
fixed starting weights and matched real update counts. A gain would justify
further investigation; it would not isolate the multiplicative objective from
other differences between RL and supervised learning. A null or refusal collapse
would argue against continuing the same recipe without another diagnosis.

## The next isolating comparison, only if informative

Compare paired reward with the mean of the two separately graded rewards. Keep
the initial model, TRAIN parents, candidate count, sampling schedule, optimizer,
caps and checkpoint rule fixed. Record the supported-answer and negative-decision
components separately, including which produced each nonzero advantage. Charge
all sampling, including uniform-reward groups. No reward or label is shown to the
model during generation. This is not accepted GPU work yet.

A useful result would improve paired correctness on held questions without merely
moving along the answer/refusal tradeoff. We would then test another dataset and
a fresh training seed. If both objectives behave alike, prioritize task competence
or information acquisition rather than adding reward complexity.

## Do not mistake stricter exclusions for the only valid transfer question

The current held panel excludes every official TRAIN atomic component, including
many that we never used for training; the resulting selected panel contains only
two-hop questions. Other exclusion rules also contribute to selection, so this
is not an isolated causal effect of one filter. It is a disclosed evaluation
choice, not the only defensible one.
Future panels should distinguish new questions composed from familiar facts from
questions with unfamiliar atomic components and from document-disjoint inputs.
The first is particularly relevant to compositional generalization. None proves
absence from pretraining. Do not silently loosen the current frozen panel, but do
not let its strict exclusion rule erase the deeper compositions we actually want
to study in a separately declared experiment.

The subsequent CPU feasibility audit clarifies this distinction: after exact
question exclusions,385 three-hop and189 four-hop candidate parents remain.
None shares an atomic step ID with official TRAIN, but prior study DEV panels
cover atoms in382 of the385 three-hop parents and all189 four-hop parents.
Thus prior evaluation exposure, not demonstrated overlap with TRAIN atomic IDs,
is the important obstacle in this candidate pool. A new declared panel can test
new combinations of previously evaluated components, with that limitation stated.
It must not be described as recombination of *training*-seen atoms when the audit
finds no such overlap. Exact TRAIN document reuse remains widespread and is a
different exposure measure. The existing frozen two-hop panel is unchanged.
