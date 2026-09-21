---
status: adaptive_exploratory_queue
created_utc: 2026-09-21T17:05:00Z
primary_question: What must cross a delegation boundary for extra work to be useful?
novelty_status: hypotheses_require_prior_art_and_empirical_support
---

# Where the current evidence points next

The short-context studies do not yet justify a claim that our trained planner
beats direct answering. They do give us a sharper research question: **what
must a helper communicate, and when should the parent ask for more?**

This develops the first two ideas in [the existing idea collection](../../new-ideas/ideas.md):
composable results and follow-up requests to a child. It does not require us to
commit to a large search algorithm before finding a useful action.

## 1. Preserve information that is needed across input groups

Imagine asking for people who bought both a bicycle and a helmet. One helper
sees bicycle purchases; another sees helmet purchases. Asking each for people
who bought both can produce two locally correct empty lists and a globally
wrong answer. Asking for each person's purchases preserves what the parent
needs to combine.

Our pending QAMPARI screen tests the first practical boundary: direct reading
of 200 retrieved passages versus four groups with an answer-list union. If
grouping helps independent answer collection but loses relationships spanning
groups, compare answer-only messages with source-linked intermediate facts.
Keep the same input pool and measure all extra reading and merging costs.

The smallest next comparison would reuse the fixed development panel, clearly
labeling it exploratory. A positive result must then survive a fresh panel and
a simple evidence-extraction or re-reading control. A harder input created only
by arbitrarily restricting the direct model is not an acceptable win.

Potential contribution: a learned choice of what information to return across
the boundary, with a measurable improvement over a fixed message. Neither
JSON schemas nor fixed passage splitting are novel by themselves.

## 2. Separate useful short-term goals from extra thinking

The completed first ALFWorld screen was dominated by inadmissible commands.
The completed follow-up gives both flat and manager-worker policies numbered
actions and explicit rejection feedback. The manager solves six of16 attempts
versus one for the flat policy, with no invalid commands in either arm. The
successes occur on three placement games; heating and cleaning remain unsolved.

If a manager advantage survives, the next control should let a single agent
state a short reason or immediate goal before choosing its action. This tests
whether the benefit requires a persistent manager-worker division or merely
more explicit deliberation. A fresh, outcome-independently selected game panel
would test transfer; the existing eight games are development material.

If both policies merely loop through admissible actions, more hierarchy is not
the immediate solution. A public-observation memory or progress signal is a
smaller intervention to qualify first. Such a signal must be computed from
what the agent observed, not hidden simulator success labels.

Candidate experiment: learn when an existing short-term goal needs revision,
rather than refreshing it every fixed number of actions. The subsequent
[primary-literature review](LITERATURE-UPDATE-1721.md) found that HiPER already
learns subgoal switching. This is not sufficient novelty. Any contribution must
go beyond that design and survive simpler fixed-rule and local-deliberation
controls; no new hierarchy claim is currently established.

## 3. Learn a necessary evidence-sensitive competency

The sufficiency baseline often makes the same answer/refuse decision on both
official variants of a question. Two small training arms are being prepared:
answering with both sufficient and insufficient examples, versus positive-only
answer training with the same number of examples and updates.

A useful result must improve the paired score without merely refusing more
questions. Report positive-answer accuracy, false refusal and official-negative
overanswer separately. Some negatives appear to retain alternate support, so
these labels cannot establish perfect grounding or justify indiscriminate
penalties for every non-abstention.

This is competency training, not a new architecture. If it works, the later
question is whether the learned distinction supports a useful information
request or decomposition decision. More accurate refusal alone is insufficient.

## 4. Resume RL where actions can change the outcome

The current root RL has real updates but no established fresh-question gain;
the extra five updates did not help. The execution-credit alternative mostly
agrees with ordinary terminal reward. We should not keep changing optimizers
while leaving an unhelpful action space unchanged.

For the next qualified task, measure valid candidate diversity, reward
differences, intermediate usefulness and fresh performance. A bounded RL pilot
should train the component responsible for the useful action. Skip all-zero
advantage updates explicitly; retain their rollout cost. A more elaborate
curriculum or search policy is warranted only after a simple pilot creates a
repeatable improvement.

## Connection to the neighboring project

Read-only inspection of `../structured-decomposition-benchmark` at commit
`c7ee38127e5f37c327b77f518a21f0bab151ff33` found a related distinction in
`research/WHY_DECOMPOSITION_AND_REVISION.md`: report correctness, whether the
combiner follows the report, and final correctness are separate measurements.
Its local status is dated August 7; it is not new September 21 RLM evidence.
No artifacts or results have been imported into this study.

## Reporting decision

Keep the completed advisor deck as a historical presentation for the meeting
that has already happened. Record these newer results in the current research
update and linked evidence reports. A future deck should use the larger
replications and limitations rather than silently preserving the earlier,
more favorable small-panel interpretation.
