---
status: exploratory_running
evidence_cutoff_utc: 2026-09-21T16:00:00Z
model: Qwen3-4B-Instruct-2507
primary_question: When does asking helpers earn its extra computation?
publication_status: promising_questions_not_established_architecture_improvement
---

# What we are learning about useful delegation

The most useful question is no longer simply whether we can train a model to
write subquestions. We can. The harder question is whether those subquestions
lead to useful work that a simpler, cheaper answerer would miss.

Our current controlled system is narrower than a full recursive language model:
a planner reads the question and document titles, helpers answer its proposed
subquestions, and a final model reads the original documents plus their answers.
The planner produces a question list, not arbitrary Python or a recursive tree.
All current question-answering inputs fit in one model call. That limitation
matters when interpreting both successes and failures.

## Three findings worth keeping

**Helper training sometimes improves answers, but formatting explains much of
the apparent gain.** On the small HotpotQA panel, trained helpers give 40 correct
answers out of 64 attempts, versus 22 with unchanged helpers. A simple formatting
reminder already raises the latter to 35. Against the strongest direct-answer
baseline, the comparison is 40 versus 36, with substantial uncertainty and about
3.5 times the tokens. A replication on 128 new questions is running.
See [the matched architecture comparison](HOTPOT-MATCHED-ARCHITECTURE.md).

**Root reinforcement learning produces real updates, but its new-question gain
is not established.** On 64 new MuSiQue questions, each sampled twice, the
supervised planner answers 53 of 128 attempts correctly. After 16 RL updates it
answers 56; direct answering gets 54. The paired RL improvement interval spans
zero. The seven changed outcomes have valid answers, so this particular contrast
is not merely repaired JSON. Some final answers nevertheless contradict their
helper chains. See [the fixed-helper results](FRESH-CONTRACT-FINDINGS.md).

**Actually executing the helpers is not consistently better than supplying the
plan alone.** Without helper reports, the supervised planner's final model gets
56 rather than 53 correct; the RL planner's final model gets 53 rather than 56.
Neither difference is established. Plan-only uses roughly one-third of the
tokens. Helpers can supply useful information or mislead the final answerer.
See [the plan-only experiment](PLAN-ONLY-FINDINGS.md).

These findings do not show that long-context RLMs are unnecessary. They show why
a short-context question-planning experiment needs strong direct and plan-only
controls before claiming that learned decomposition is useful.

## What the RL diagnostics changed

Training-set replay improved from 45 to 49 correct attempts out of 64. Repeating
execution of frozen plans revealed some reward variability, but averaging more
executions added little value for choosing candidates. We therefore tried one
bounded continuation rather than multiplying every reward evaluation.

That continuation saved five more updates, reaching checkpoint 21. The next
batch contained varied valid plans but no within-question reward differences:
12 questions were always correct and four always wrong across four candidates.
The declared pilot rule stopped the run. This is not proof of convergence;
stopping at one uninformative batch is a rule we should reconsider in future
training. A separately declared checkpoint-21 evaluation is queued.
See [the stopped-run analysis](RL-STOPPED-DOSE.md).

The next diagnostic asks whether ordinary final-answer reward agrees with the
measured benefit of executing helpers. An action-dependent counterfactual would
change the learning objective, not simply provide a free variance-reduction
trick. We will measure that distinction before adopting a new reward.
See [the execution-credit question](EXECUTION-CREDIT-QUESTION.md).

## What could become a stronger research direction

One possibility is learning **when additional work changes the available
information or useful action**, rather than generating a longer plan for a
fully visible problem. A small text-environment comparison is now queued:
ordinary action selection versus a short-term goal manager, under the same
action and output-token limits. Both receive the same kind of public feedback
and admissible-action assistance. It is an exposed development screen, not a
new hierarchy algorithm or a benchmark superiority claim.
See [the environment screen](ALFWORLD-SCREEN-DESIGN.md).

A second possibility is deciding whether evidence is sufficient before answering
or requesting more. The canonical MuSiQue paired variants support an initial
test, but change distractors and titles as well as supporting material. They
cannot be described as a clean intervention that removes only one fact.
See [the paired-data audit](MISSING-EVIDENCE-PAIR-AUDIT.md).

The publication-oriented target is a repeatable improvement that survives a
simple control, transfers to new cases, and earns its measured cost. We do not
yet have that claim for the current planner. The present evidence helps us
choose what to test next and avoid mistaking interface compliance for reasoning.
