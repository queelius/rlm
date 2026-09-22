# Reading the research results

## September 22: what survived stronger comparisons

The [current research update](../experiments/selective_delegation/RESEARCH-UPDATE-20260922.md)
is the starting point. We can train the model to write subquestions and execute
real reinforcement-learning updates. We have not yet established that this
system answers new questions better than simpler, cheaper alternatives.

On 64 new MuSiQue questions, sampled twice, the supervised planner gets 53 of
128 answers correct; 16 RL updates raise this to 56, while direct answering gets
54. The paired improvement interval includes zero. Five further RL updates give
54, not a demonstrated dose benefit. Giving the final model the plan without
actually executing its helpers performs similarly at roughly one-third the
token cost. These are question-list experiments, not learned Python recursion.

A promising small HotpotQA result did not establish an architecture advantage
on its larger replication: decomposition gets 151 of 256 answers correct,
versus 152 for direct answering, at 3.44 times the tokens. The completed
three-answer voting control selects exactly the same answers as the original
direct policy. We are now testing tasks where additional work can uncover
information or enable actions, rather than merely rewrite a fully visible input.

The evidence-sufficiency experiments reveal a repeatable answer/refusal tradeoff:
training only on answerable examples increases answering, while including negative
examples increases refusals, including mistaken refusals. Paired-reward RL has
completed eight real updates without service failures. The first fixed held
readout now gives18 correct pairs out of64, versus10 before continuation;
matched additional supervised training gives19. Both improve on this small
panel, but there is no established RL advantage over more supervised training.
See [the completed held comparison](../experiments/selective_delegation/PAIRED-RL-HELDOUT-FINDINGS.md).
The [deeper-question readout](../experiments/selective_delegation/PAIRED-RL-COMPOSITIONAL-FINDINGS.md)
is complete:6/7/6 for warm/RL/extraSFT, with no clear joint improvement. The
[training audit](../experiments/selective_delegation/PAIRED-RL-TRAINING-FINDINGS.md)
shows sparse rewards and several distinct answer failures. In the small household-task screen, a manager and a single agent that
briefly explains its next action both solve six of16 attempts, compared with one
for action-only prompting. On the completed new-task comparison, manager and
action-only policies each solve four of24 attempts, while local reasoning solves
six. None of the paired differences establishes an improvement, and local
reasoning takes3.8 times the action-only policy's summed model-call time. The
earlier manager advantage did not replicate. See the
[fresh household findings](../experiments/selective_delegation/ALFWORLD-UNSEEN-FINDINGS.md).

See the [reading guide](../experiments/selective_delegation/README.md),
[fresh RL comparison](../experiments/selective_delegation/FRESH-CONTRACT-FINDINGS.md),
[plan-only control](../experiments/selective_delegation/PLAN-ONLY-FINDINGS.md), and
[larger Hotpot replication](../experiments/selective_delegation/HOTPOT-FRESH-FINDINGS.md).
The earlier evidence below remains historical context, not the current run queue.

## Earlier completed research

The [broader-data RL result](research-checkpoints/2026-09-12-broader-rl-learning-signal.md)
now repeats across two training seeds: 437 and 436 correct out of 512 after
eight RL updates, versus 422 before training and 427 after supervised training.
The trained models agree on 511 answers. This is a same-panel helper result,
not learned decomposition or broader task transfer. The completed fresh-example
test is substantially weaker: base 422, supervised 426, and the two RL models
427 and 429 out of 512. The RL-versus-supervised intervals include zero. Both
panels, rather than only the more favorable one, inform the current assessment.

The [replication and next learning study](research-checkpoints/2026-09-12-replication-and-next-learning-study.md)
shows that a one-answer RL gain on new news articles did not survive a fresh-seed
repeat. It records the earlier cutoff when the broader study was still running;
its completed comparison is in the newer report above. The earlier result and
its limitations remain part of the evidence.

The [reward-feedback and input-size checks](research-checkpoints/2026-09-12-feedback-and-input-shape.md)
found no new RL gain. Giving more training questions nonzero feedback changed
none of the 256 evaluated labels, and evaluating one item at a time did not
reveal a hidden benefit. These results shift attention toward broader training
data and learning the root's procedure.

The [checking-cost and RL-saturation results](research-checkpoints/2026-09-12-checking-cost-and-rl-saturation.md)
show that selective checking saved tokens compared with three-answer voting but
did not clearly beat the cheapest single pass on fresh examples. Larger RL
updates stopped with no relative reward signal and did not improve the evaluated
answers. A numerical-data pilot exposed code and time limits before it could
test a functioning Python-assisted policy.

The [exploration and helper-contract comparisons](research-checkpoints/2026-09-12-exploration-and-helper-contracts.md)
show that higher sampling temperature created more contrasting training answers
but changed none of the 256 evaluation predictions. A separate complete experiment
found 31/48 correct category counts from full labels versus 20/48 from targeted
yes/no answers. These results narrow the next RL and harness decisions.

The [grouping and faster-RL qualification results](research-checkpoints/2026-09-12-grouping-and-fast-rollout-qualification.md)
show that changing neighboring records or their order can change a helper's
answers even at fixed group size. A separate fresh 48-sample batch passes the
original probability checks for a future faster RL update; no optimizer step
is attributed to that qualification.

The [new decomposition and four-update RL results](research-checkpoints/2026-09-12-decomposition-transfer-and-four-step-rl.md)
show why adaptation matters: smaller requests helped slightly on question
categories but hurt on news, at higher token cost. Four RL updates produced
exactly the same evaluation answers as the earlier one-update pilot. The report
explains the limited feedback and the next task-targeted helper experiment.

The [serving-probability stability probe](research-checkpoints/2026-09-12-serving-probability-stability.md)
shows that an existing vLLM option eliminated the observed probability variation
in one small controlled workload, at increased runtime. It motivates a fresh
RL qualification test, not retroactive approval of old samples.

The [first completed helper-RL readout](research-checkpoints/2026-09-12-helper-rl-first-readout.md)
separates working training mechanics from answer quality: one qualified update
gave 119→120 correct new questions, 112→112 news examples, and 245→244 familiar
answers. The original model scored 92/128 on those new questions, providing a
stronger transfer result for the earlier supervised training. The report explains
the sparse RL signal and the planned four-update test.

The [downstream helper-size comparison](research-checkpoints/2026-09-12-helper-size-downstream.md)
connects actual saved helper replies to final task results: 12, 18 and 17 correct
answers out of 24 for replies generated in groups of sixteen, four and one.
It explains the matched comparison, reporting correction, and remaining limits.

The [helper request-size comparison and training-exposure audit](research-checkpoints/2026-09-12-helper-size-and-training-exposure.md)
finds modest gains from smaller requests, with more token use and some losses.
It also establishes that these questions were in the helper's earlier supervised
training, so these results cannot demonstrate unseen-question generalization.

The [fresh-seed learning repeat](research-checkpoints/2026-09-12-learning-repeat.md)
finds 55 correct answers for the unchanged model and 57 for self-SFT, with two
unavailable answers each. It explains the mixed paired changes and why the earlier
apparent compute saving did not repeat.

The [completed helper-information intervention](research-checkpoints/2026-09-12-helper-intervention.md)
reports 26/48 correct with original helper replies, 27/48 when replaying those
replies, and 44/48 when supplying correct local labels. It explains the remaining
root mistakes, the limits of this two-context diagnostic, and the next helper
training and request-size experiments.

The [completed September 12 learning comparison](research-checkpoints/2026-09-12-learning-comparison.md)
reports 53 correct answers for the unchanged model, 52 after one root-only RL
update, and 57 after learning from successful own attempts, on the same 72-question
panel. It separates observed-answer changes from recovered unavailable attempts
and explains why the next experiment targets helper feedback.

The [September 12 helper-feedback update](research-checkpoints/2026-09-12-helper-feedback.md)
reports the first paired root-RL result (52 correct versus 53 unchanged) and the
trace evidence motivating a helper-focused diagnostic. The oracle, replay and
helper-training ideas are clearly separated from completed findings.

The [September 12 learning checkpoint](research-checkpoints/2026-09-12-onebatch-learning.md)
records a completed 48-attempt feedback diagnostic, saved RL and self-SFT updates,
and the running paired evaluation. It distinguishes successful training execution
from an answer-quality improvement that has not yet been established.

After the advisor meeting, the [learned-decomposition research proposal](research-plans/2026-09-11-learned-decomposition.md)
connects our results to RL feedback, strategy choice, recursive stopping, and tests on new task
structures and datasets. It is an exploratory proposal, not evidence that those experiments
have run. The [first diagnostic execution plan](research-plans/2026-09-11-first-diagnostic-execution.md)
turns the approved direction into bounded comparisons; the
[RAO code notes](research-plans/2026-09-11-rao-code-notes.md) distinguish root feedback from
independently scored child tasks. Consult the live queue for actual execution status.
The [post-meeting launch checkpoint](research-checkpoints/2026-09-11-postmeeting-resumption.md)
records the first new diagnostic and its external resume pointers.

The [September 11 delivery checkpoint](research-checkpoints/2026-09-11-advisor-delivery.md)
links the eight-main-slide advisor package and records the final short-experiment state.
It distinguishes materials preserved in Git from model checkpoints on project storage.
The [earlier advisor-window checkpoint](research-checkpoints/2026-09-11-advisor-two-hour-window.md)
preserves the 13:00 UTC snapshot.
The [earlier presenter checkpoint](research-checkpoints/2026-09-11-presenter.md)
preserves the 06:33 state; it is not the current run queue.

## Public reading copy

The [research notebook on GitHub](https://github.com/queelius/rlm-research)
contains a curated snapshot of questions, claims, reports, experiment scripts,
and compact evidence metadata. Start with its
[reading guide](https://github.com/queelius/rlm-research/blob/main/README.md).
Its manifest records source paths and hashes. This is not a live mirror or a
complete reproduction bundle; large artifacts have a separate
[availability statement](https://github.com/queelius/rlm-research/blob/main/ARTIFACTS.md).

## Local research store

The current research store is `/project/alex_phd/runs/rlm-research-r4`.
Its [reading guide](../../../runs/rlm-research-r4/README.md) is the entry point.
The links below work in the project layout on the research machine; the large
run store is separate from this Git repository and is not uploaded by a Git push.

## A short route through the evidence

1. Read [what we know so far](../../../runs/rlm-research-r4/analyses/CURRENT_SUMMARY.md)
   for the research question, latest completed results and their limits. The
   [cross-experiment overview](../../../runs/rlm-research-r4/analyses/cross-experiment-synthesis-2026-09-09/OVERVIEW.md)
   provides the wider history through its stated cutoff.
2. Read the [findings](../../../runs/rlm-research-r4/analyses/cross-experiment-synthesis-2026-09-09/FINDINGS.md)
   for what each result supports, possible explanations, and its limitations.
3. Read the [proposed comparisons](../../../runs/rlm-research-r4/analyses/cross-experiment-synthesis-2026-09-09/NEXT_EXPERIMENTS.md)
   for the questions that would most change our next decision.

The [complete dossier](../../../runs/rlm-research-r4/analyses/cross-experiment-synthesis-2026-09-09/README.md)
also contains an experiment catalog, limitations, research questions, decision
history and sources. Machine-readable claim and source inventories connect the
written conclusions to exact artifacts and content hashes.

## Keep results separate from current activity

The September 9 dossier covers an explicit evidence slice ending at 01:35 UTC.
It is not silently rewritten whenever a running experiment produces another row.
Use the [analysis index](../../../runs/rlm-research-r4/analyses/README.md) for later
reports and the [live queue](../../../runs/rlm-research-r4/RESEARCH_QUEUE.md) for
what is running or ready. A candidate experiment in the dossier is not itself a
launched process.

## Organization for future analyses

Use descriptive titles and complete sentences. A report should answer:

- What question did the comparison address?
- What changed, and what stayed the same?
- What did we observe, including failures and missing outcomes?
- What can we conclude, and which explanations remain uncertain?
- What should we run next, and which result would change that decision?

Keep the readable report alongside its machine-readable results, source manifest
and focused verification script when useful. Distinguish source examples from
repeated measurements. Keep operational failures separate from model mistakes,
and keep wrong answers separate from unobserved outcomes. Report training,
inference, loading and idle time separately when the measurements allow it.

Do not move or rewrite raw runs to make a report easier to read. Add an analysis
layer and preserve links to the original evidence. New evidence may revise an
interpretation; record the revision and its reason instead of hiding the earlier
result. Follow [research operations](RESEARCH_OPERATIONS.md) so this analysis work
overlaps useful GPU execution rather than delaying it.
