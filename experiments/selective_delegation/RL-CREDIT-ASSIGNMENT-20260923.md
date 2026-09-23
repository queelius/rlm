---
date: 2026-09-23
status: exploratory-follow-up
question: Does terminal credit penalize useful actions inside failed attempts?
evidence_cutoff_utc: "09:23"
---

# A larger update works numerically. Does it teach the right behavior?

The five-times-larger learning-rate run finished one new optimizer update in
23.7 minutes. It reused the exact saved training attempts and starting checkpoint;
no new model rollouts were collected for training. Its task evaluation is now
complete; partial success scores were not used to change the plan.

## Completed evaluation: a larger update did not establish improvement

The smaller update solved **16/32** attempts; the larger update solved **15/32**.
All 32 pairs are known: two improvements, three regressions and 27 ties. The
task-cluster bootstrap interval for the difference is −18.75 to +12.5 percentage
points. This small exposed panel does not establish a reliable difference.

The larger update used 1,230 calls versus 1,168, produced 689 rejected actions
versus 578, and hit the context limit in seven attempts versus four. Its owner
ran 58.1 minutes versus 53.1 minutes. Both completed normally with no transport
failures or missing results. Costs are descriptive, not replicated timing effects.
See the [compact readout and full-report hash](textcraft-lr-readout-summary.json).

Decision: do not increase the learning rate again just because the saved-batch
objective improves. The positive-only comparison is now accepted after the
unchanged memory factorial, under queue019. It tests a different explanation
for weak RL progress, not an established diagnosis. Earlier queues017/018 were
waiting with no scientific jobs started; their jobs are retained after it.

## What the completed training tells us

| Quantity | Earlier update | Larger update |
|---|---:|---:|
| Learning rate | 0.00002 | 0.0001 |
| Adapter-weight change, Euclidean length | 0.058994 | 0.294971 |
| Saved-batch objective before update | −7.37471 | −7.37471 |
| Saved-batch objective after update; lower is better | −8.00974 | −11.32668 |
| Mean log-probability change per positively credited token | +0.000402 | −0.001562 |
| Mean log-probability change per negatively credited token | −0.001629 | −0.013049 |

The weight change is 5.00003 times larger. All pre-update replayed token
probabilities agree exactly between the runs. Recorded gradient norms differ
slightly (39.2610 versus 39.3127), so this should not be described as bit-identical
gradient execution. Both runs passed their unchanged finite-gradient and
training-versus-evaluation probability checks.

The larger update reduces the probability of negatively credited sequences more
strongly. But averaged over positively credited tokens, it also reduces the
probability of sequences that succeeded. Shared parameters couple these changes;
the training objective does not promise to improve every successful sequence.
These selected-token measurements are **not** KL divergence, task accuracy or
evidence of generalization. The completed evaluation above shows why a better
saved-batch objective alone is insufficient.

The eight training tasks have success counts `[2, 2, 2, 3, 3, 4, 4, 4]` out of
four attempts each. Thus reward variation is present in five task groups. Twelve
successful attempts receive positive relative credit, eight failed attempts
receive negative credit, and twelve attempts in all-success groups receive zero.
There are 12,927 positively credited and 14,006 negatively credited tokens.
This batch is not suffering from an absence of both successes and failures.

Native replay adds an important qualification: the positively credited attempts
contain 177 environment-rejected actions among 399 calls. Negative attempts
contain 110 information requests, 303 crafting requests and six finish requests
(plus one malformed call); their audits record 240 environment-action errors.
Of the 177 positive-credit errors, 79 report insufficient inventory. Seventy-one
are immediate repetitions of the preceding identical action and identical error;
failed attempts contain 89 such repetitions. For example, an agent repeatedly
tries a recipe needing eight units while the environment keeps reporting that
only six are available, before eventually producing the missing units. This is
a repeated-action loop, not an identical full prompt: history and budget change.
The [error-description receipt](textcraft-credit-error-description.json) pins the
saved public histories and checks error totals against the native audits.
An information request is not automatically useful, and an eventual success does
not make all earlier actions good. This is why simply reinforcing entire winning
attempts may retain substantial waste.

Evidence: [machine-readable diagnostic](textcraft-lr-training-diagnostic.json)
pins the two batches, saved token probabilities and committed optimizer states.
Its external copy is `R/analysis-textcraft-lr-training-diagnostic-001.json`, where
R is the artifact root in [the unattended handoff](UNATTENDED-20260923.md).
The post-update objective is reconstructed from the saved action-token
log probabilities, original leave-one-out advantages and denominator 32.

## The next small comparison

Prepare a separate one-step run from the same starting model and optimizer,
using the same batch and larger learning rate. Keep the original positive
advantages but omit negatively credited actions from the optimizer. Keep the
denominator at 32 and report the smaller credited-token count. Compare its
fixed endpoint against the completed larger signed update on identical tasks.

This tests whether negative whole-attempt credit is helpful in this setup.
It is deliberately a different, biased objective—not an unbiased policy-gradient
estimator, an equal-token-dose experiment, or a matched supervised-training
control. A positive result would motivate more precise action credit, not prove
that all negative updates are bad. It is accepted and waiting, not training yet.
In particular, it retains those 177 rejected actions from successful attempts.

## How this connects to the RLM design

The current actors do not recurse, so these results alone do not establish an
RLM advantage. If the notebook experiment helps, the next architectural question
is whether helpers should return reusable, structured facts as well as an answer.
A parent or sibling could then avoid rediscovering information already obtained.
The first comparison should keep the model, task, total call budget and available
facts fixed, changing only whether verified public facts survive a helper boundary.
That would test information transfer across decomposition, rather than claim
novelty from adding a generic memory feature.

Repeated-state credit is another possible connection: compact public state might
make repeated decisions easier to recognize. The saved-trace feasibility check
must distinguish identical current prompts from merely similar inventories.
Only fresh rollouts under a changed observation interface could test that new
policy; compressing old prompts after collection would not make the old behavior
on-policy for the new interface.

## Primary research that changes the plan

- [TRACE, July 2026](https://arxiv.org/html/2607.13988v1) uses a frozen model's
  probability of the gold answer to estimate progress at tool-call boundaries.
  Adjacent changes supply intermediate credit. This supports investigating
  useful steps inside failed attempts. However, TextCraft finishes by changing
  inventory rather than writing an informative answer: the probability of
  emitting `finish` is not a justified state-value estimate. Do not transplant
  that measurement without validating it against actual completion.
- [GACA, September 2026](https://arxiv.org/html/2609.12424v1) mixes whole-attempt
  and repeated-state credit using action uncertainty. Its method and analysis
  make assumptions about state grouping and uncertainty; token surprise alone
  does not establish decision importance. First count repeated decision contexts
  in our saved traces. Identical inventories with different remembered recipes
  are not necessarily the same decision state.
- [LEAP](https://arxiv.org/html/2410.05434v1), published at ICLR 2025, explicitly
  analyzes the tension between privileged expert performance and whether a
  student can imitate it from its own observations. Our teaching result therefore
  cannot claim novelty merely from noticing that tension. More defensible work
  would identify which information must be acquired and retained, with controls
  that separate teaching content, memory and recipe changes.

These are method-level connections from the primary papers, not replications of
their reported gains. Priority changed only after the complete result; the live
memory experiment and all sealed sources remain unchanged. The
[decision map](DECISION-MAP-20260923.md) records prospective interpretation rules.

Advisor-deck decision: keep these details in supporting research notes for now.
The main message remains that teaching is promising but meaningful TextCraft RL
improvement is not established. Do not add a positive RL headline to the slides.
