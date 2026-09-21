---
status: exploratory_running
evidence_cutoff_utc: 2026-09-21T19:26:00Z
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
3.5 times the tokens. The larger replication on 128 new questions is now
complete: decomposition gets 151 of 256 attempts correct, versus 152 for direct
answering, at 3.44 times the tokens. Its small partial-answer-score advantage
is also uncertain. This does not establish an advantage for decomposition.
See [the matched architecture comparison](HOTPOT-MATCHED-ARCHITECTURE.md).
The completed [three-answer voting control](HOTPOT-VOTE-FINDINGS.md) also gives
152 correct: extra samples scarcely change the normalized answers at the fixed
temperature, and every voted answer equals its original direct answer.

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
training. The separately declared checkpoint-21 evaluation is complete: 54 of
128 correct, versus 56 at checkpoint16. The difference is uncertain; there is
no demonstrated benefit from this continuation. We are not extending the same
recipe again without a new reason.
See [the stopped-run analysis](RL-STOPPED-DOSE.md) and
[the completed readout and examples](RL-STOPPED-DOSE-FINDINGS.md).

The execution-credit diagnostic is now complete. On a small training panel,
helper execution gets 198 of 320 answers correct, compared with 99 when only the
plan is supplied. These 320 attempts reuse 16 questions, four plans per question,
and five execution settings; they are not 320 independent questions. The much
larger training-set effect does not establish a benefit on new questions.
Moreover, credit based on the measured helper benefit mostly agrees with ordinary
final-answer reward. It does not yet justify a more elaborate RL objective.
The completed direct-answer control gets 148 of 320 correct. Helpers therefore
beat both direct answering and the plan-only condition on this training panel;
their benefit is not just recovery from plan-induced harm. The helper-minus-direct
difference is 15.63 percentage points, with a component-clustered interval from
1.76 to 32.67 points. This is a useful training signal, not a fresh-data result.
See [the direct-control analysis](TRAIN-DIRECT-CONTROL-FINDINGS.md).
An action-dependent counterfactual changes the learning objective; it is not
simply a free variance-reduction trick.
See [the execution-credit question](EXECUTION-CREDIT-QUESTION.md).

## What could become a stronger research direction

One possibility is learning **when additional work changes the available
information or useful action**, rather than generating a longer plan for a
fully visible problem. A small text-environment comparison tested
ordinary action selection versus a short-term goal manager, under the same
action and output-token limits. Both receive the same kind of public feedback
and admissible-action assistance. The first screen completed: the flat policy
solves one of 16 attempts and the manager-worker policy solves two. Most runs
stop after repeatedly choosing commands outside the supplied admissible list.
This was too weak to support a hierarchy claim. The follow-up with numbered
choices is now complete: the manager solves six of 16 attempts, versus one for
the flat policy, with no invalid outputs in either arm. The gains occur on
three placement games, not heating or cleaning. It uses fewer tokens but slightly
more summed model-call time. The control with a single agent that briefly explains
its next action is now complete: it also solves six of 16 attempts, with one win
and one loss relative to the manager. Thus the gain is not unique to manager-worker
delegation. The manager uses about one-third as much summed native generation time
as this explanation-based control, despite using more total tokens. These small,
exposed-game results do not establish equivalence or a general speed advantage.
See [the completed control and examples](ALFWORLD-LOCAL-REASON-FINDINGS.md).
The new-task comparison is now complete: on twelve previously unused games,
sampled twice, the manager and flat policy each solve four of 24 attempts. Local
reasoning solves six, but takes 3.8 times the flat policy's summed generation
time. All paired improvement intervals include zero. These twelve games cover
only four scenes, so the evidence is less independent than the attempt count
suggests. The earlier manager advantage did not replicate. We retain the simpler
flat baseline and will test whether action training improves execution, without
assuming a hierarchy is better.
See [the fresh comparison and concrete examples](ALFWORLD-UNSEEN-FINDINGS.md).
This is not yet a new algorithm or a general result.
See [the environment screen](ALFWORLD-SCREEN-DESIGN.md).
The [completed interface follow-up](ALFWORLD-CLOSED-LOOP-FINDINGS.md) includes
concrete successful and failed action sequences.

A second possibility is deciding whether evidence is sufficient before answering
or requesting more. The canonical MuSiQue paired variants support an initial
test, but change distractors and titles as well as supporting material. They
cannot be described as a clean intervention that removes only one fact. The
baseline completed all 128 variant attempts: it refuses 24 of 64 answerable
inputs, yet answers 23 of 64 inputs labeled unanswerable. It handles both
answerability labels correctly for only 17 of 64 paired attempts, and gets the
joint exact-answer-and-sufficiency score on nine. This identifies a concrete
competency gap under the official labels, not evidence that a proposed new
method solves it. Inspection also found a labeled-negative input that appears
to retain alternate evidence for the answer. We must not equate every official
negative-label answer with an unsupported hallucination.
See [the paired-data audit](MISSING-EVIDENCE-PAIR-AUDIT.md).

Two supervised-training comparisons now show why this is not just a matter of
showing the model more examples. Training only on answerable inputs raises correct
answers on those inputs from 16 to 25 of 64, but the model then answers every
negative input too. Training on both positive and negative inputs reduces answers
on negative inputs from 23 to six, but increases refusals on answerable inputs
from 24 to 37. The combined score barely changes: ten correct pairs instead of
nine, with substantial uncertainty. All outputs are correctly formatted, so this
tradeoff is not a JSON problem. The fresh-question replication shows the same
pattern: positive-only training answers every variant; joint training refuses 41
of 64 answerable inputs. Joint correctness is 11 pairs versus six for the base,
but its paired improvement interval still spans zero. This new panel contains
only two-hop questions and has substantial document overlap with official TRAIN.
See [the replication](SUFFICIENCY-CANONICAL-FINDINGS.md). We have queued
paired-reward RL against an additional supervised-training control to
ask whether both behaviors can improve together. These are official dataset
labels, not a perfect test of whether an answer has semantic support.
See [the training comparison](SUFFICIENCY-TRAINING-FINDINGS.md).

A third screen is complete on QAMPARI, where one question can require
collecting many answers from 200 retrieved passages. It compares reading the
whole supplied pool with reading four groups and combining their answer lists.
The base model can fit the full input, so direct answering will not be artificially
truncated. Across16 questions sampled twice, splitting has a lower answer-set
score (.166 versus .186), with an uncertain paired difference. It finds slightly
more correct entities but introduces more wrong ones. Splitting uses about28%
less summed native inference time despite slightly more tokens, so token counts
alone miss an important computation trade-off. Output limits also matter:
22 calls end with incomplete lists, mostly repeated names rather than useful
unfinished answers. We will qualify decoding before more splitting experiments.
The timing benefit also needs qualification: among the 21 attempts where both
methods returned valid answers without hitting the output limit, splitting was
about 10% slower. That outcome-selected subset is descriptive, not a causal
comparison, but it shows why the aggregate speed difference cannot be credited
to more efficient attention alone. Direct-model repetition loops affect it.
See [the complete findings and concrete examples](QAMPARI-FINDINGS.md).
This known baseline does not establish a new passage-splitting contribution.

The publication-oriented target is a repeatable improvement that survives a
simple control, transfers to new cases, and earns its measured cost. We do not
yet have that claim for the current planner. The present evidence helps us
choose what to test next and avoid mistaking interface compliance for reasoning.

The [latest interactive-agent literature review](LITERATURE-UPDATE-1721.md)
also narrows the novelty boundary: learned subgoal switching and learned external
workspace use already have direct precedents. Our next result must explain a
specific useful decision or information boundary, not merely add a manager or
memory and call it a new architecture.
