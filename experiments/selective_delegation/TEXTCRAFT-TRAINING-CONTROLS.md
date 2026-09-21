---
question_id: rq:recursion_or_subtask_practice
created_utc: 2026-09-21T18:25:00Z
status: prospective_not_gpu_accepted
depends_on: textcraft_bounded_inference_qualification
novelty: requires_prior_art_review_and_positive_evidence
---

# Is the benefit recursion, or practice on smaller tasks?

The immediate proposed TextCraft pilot is narrower: can an unchanged model use
recursive calls effectively under the same total budget as a flat agent? It does
not yet train anything. See [the readiness inspection](TEXTCRAFT-RECURSION-READINESS.md).

If that pilot supports further work, a stronger training question is whether
learning from smaller tasks improves the underlying model, whether the recursive
interface helps at inference time, or whether both are needed. These explanations
should not be combined into a single trained-recursive versus untrained-flat result.

A useful minimal experiment would evaluate the same trained checkpoint both flat
and recursively. A more informative follow-up would compare training on full tasks
alone with training that includes locally verified subtasks, then evaluate both
checkpoints through both interfaces. The subtask control must receive comparable
training examples and measured update/token budgets. Otherwise an advantage could
come from easier supervision rather than learning to decompose.

Task difficulty should be defined independently of outcomes, using recipe depth
and the host's trusted solution description. Generalization to deeper combinations
of familiar recipe operations is a valid target; it is different from unfamiliar
recipes. Root-target disjointness alone does not imply the latter, as the cached
validation tasks reuse TRAIN intermediate items.

Track root success first, then successful child goals, wasted or unnecessary child
work, total calls/tokens and actual service time. A child can solve a real but
irrelevant task, so child success must not replace root success as the headline.
Use the smallest informative training dose and stop if the interface merely fails
to parse, if all tasks are trivial, or if no sampled actions change outcomes.

This is a candidate controlled study, not a novelty or performance claim. It does
not authorize a larger training campaign before the bounded inference pilot and
its failure analysis are complete.
