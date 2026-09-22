# Learning when another reasoning step is worthwhile

For a short plain-language account of the latest findings and decisions, start
with [the September22 research update](RESEARCH-UPDATE-20260922.md).

Current checkpoint, September 22, 06:06 UTC: [paired-answer training](PAIRED-RL-HELDOUT-FINDINGS.md)
improves the fixed warm start from 10 to 18 correct pairs out of64 with RL,
but extra supervised training reaches19. This is useful learning, not an
established RL-specific advantage. The [deeper-question comparison](PAIRED-RL-COMPOSITIONAL-FINDINGS.md)
is now complete:6/7/6 for warm/RL/extraSFT, with no clear paired improvement.
[Household action SFT](ALFWORLD-ACTION-SFT-FINDINGS.md) did not improve the exposed twelve-game panel: both flat
and manager-worker policies score2/24 after training versus4/24 before it.
The [crafting pilot](TEXTCRAFT-PILOT-FINDINGS.md) reached its one-hour cap with26/32 episode records; no helper
was ever called. Its extra work often continued after enough items were present.
The immediate follow-up tests a clearer one-action-and-finish instruction.
The GPU was idle after the capped pilot blocked an unrelated successor; the
independent successor was restarted without changing its scientific comparison.

Earlier completed findings: the
[fresh128-question Hotpot replication](HOTPOT-FRESH-FINDINGS.md) is complete:
151/256 planner versus152/256 direct, with 3.44 times the tokens and unresolved
differences. The completed [three-answer voting control](HOTPOT-VOTE-FINDINGS.md)
chooses exactly the same answers as single-call direct answering. Root RL
continuation saved checkpoint21 and then stopped at a batch with no within-question
reward differences. Its completed readout gives54/128 versus56 at checkpoint16.
The first environment-action screen exposed repeated inadmissible commands;
the completed indexed-action follow-up gives six successes out of 16 attempts
with a manager, versus one without it. The completed local-deliberation control
also solves six, so the gain is not unique to a manager-worker division. It takes
about 2.8 times the manager's summed native generation time despite fewer tokens.
See [the control and concrete examples](ALFWORLD-LOCAL-REASON-FINDINGS.md).
The [fresh household-task comparison](ALFWORLD-UNSEEN-FINDINGS.md) is now complete:
flat and manager each solve four of24 attempts; local reasoning solves six at
3.8 times flat's summed generation time. No paired contrast establishes an
improvement. The earlier manager advantage did not replicate.
See [the completed comparison](ALFWORLD-CLOSED-LOOP-FINDINGS.md).
See [the stopped-run analysis](RL-STOPPED-DOSE.md) and
[the new screen's design](ALFWORLD-SCREEN-DESIGN.md). These are exploratory jobs;
the original exact-checkpoint24 readout was not run or silently substituted.

A [simple helper-skipping rule](HELPER-GATING-DIAGNOSTIC.md) failed to improve
answers; its corrected analysis preserves both repeats and component clusters.
The [evidence-sufficiency baseline](SUFFICIENCY-FINDINGS.md) and
[TRAIN execution-credit diagnostic](TRAIN-EXECUTION-CREDIT-FINDINGS.md) are complete.
The [direct TRAIN control](TRAIN-DIRECT-CONTROL-FINDINGS.md) is complete:
helper execution198/320, direct148/320, and plan-only99/320, on16 reused training
questions. The [many-answer passage-reading screen](QAMPARI-FINDINGS.md) is
complete: splitting has lower precision and an uncertain overall score change,
with shorter measured native inference time. Both sufficiency-training arms and
fixed-endpoint readouts are complete: [the answer/refusal tradeoff](SUFFICIENCY-TRAINING-FINDINGS.md)
does not establish improved joint correctness. The [fresh replication](SUFFICIENCY-CANONICAL-FINDINGS.md)
is complete and repeats the answer/refusal tradeoff. Fresh household tasks are
complete. The [sampling control](QAMPARI-SAMPLING-FINDINGS.md) did not rescue
many-answer reading, so we are retiring further fan-out tuning for now.
Paired-reward RL completed eight actual updates without service failures;
[the training diagnostic](PAIRED-RL-TRAINING-FINDINGS.md) separates sparse rewards,
wrong answers and mistaken refusals. Matched extra SFT and the first held
readout are complete: [both continuations improve the warm start](PAIRED-RL-HELDOUT-FINDINGS.md),
without an established difference between them. Public-action SFT and its
household-task readout are also complete; training did not improve this panel.
The [bounded semantic audit](SUFFICIENCY-SEMANTIC-AUDIT.md) illustrates why a
literal answer mention is not necessarily sufficient evidence, and why the
official negative labels are not infallible.
The [TextCraft qualification](TEXTCRAFT-CPU-READINESS.md) has passed CPU gold
replay, and its [bounded model comparison](TEXTCRAFT-PILOT-PLAN.md) ended at its cap.
A [separately declared deeper-question readout](SUFFICIENCY-COMPOSITIONAL-PANEL.md)
will test the fixed RL/SFT endpoints on new composed questions, with prior-study
component exposure disclosed. That readout is complete. The actual queue and owners
remain in the external research store, not this narrative summary.
[Recent primary work](LITERATURE-UPDATE-1625.md)
narrows the novelty of generic answerability training and adaptive evidence trees.
The [further literature update](LITERATURE-UPDATE-1653.md) connects these results
to existing planner-training and evidence-integration research.

Prospective questions, not completed findings: [recursion versus subtask practice](TEXTCRAFT-TRAINING-CONTROLS.md),
and [terminal-reward action learning](ALFWORLD-TERMINAL-RL-DRAFT.md).
Conditional transfer controls ask whether action training learns
[command meaning rather than list position](ALFWORLD-ACTION-ORDER-CONTROL.md),
and whether crafting skills survive
[changed recipes rather than just new goals in the same world](TEXTCRAFT-CHANGED-WORLD-FEASIBILITY.md).
Two CPU calculations concern training credit:
[averaging over arbitrary answer pairings](PAIRED-REWARD-PAIRING-VARIANCE.md), and
[rarely called helpers losing their comparison group](OPTIONAL-NODE-CREDIT-QUESTION.md).
These calculations are not GPU performance improvements or established novelty.
[Fast learned decision models](JEV-ROUTING-QUESTION.md) are a prospective comparison,
not a Jev dependency or a change to the accepted research direction.
The [context-boundary feasibility check](CONTEXT-SUFFICIENCY-FEASIBILITY.md)
retired one proposed crafting comparison: its concrete example was a parent
quantity error, not a demonstrated missing constraint for the child.
The [small weighting calculation](RAO-WEIGHTING-CAVEAT.md) is a CPU training-design
caution, not a new algorithm or a GPU performance result.

This exploratory study asks whether a controller can learn when to finish an
answer, reconsider it, ask a focused evidence question, or work through proposed
subquestions. The helpers initially remain unchanged. This is a small controlled
decision, not yet a learned recursive RLM or a new state-of-the-art claim.

Start with [DESIGN.md](DESIGN.md), [LEDGER.md](LEDGER.md), and
[LITERATURE.md](LITERATURE.md). The implementation checklist is [PLAN.md](PLAN.md).
[TRAINING-DRAFT.md](TRAINING-DRAFT.md) describes conditional training work; its
existence does not mean a training run has happened.

For the current evidence and direction, read [FINDINGS.md](FINDINGS.md) first.
The latest [fresh fixed-helper comparison](FRESH-CONTRACT-FINDINGS.md) adds an
important update: after a full training pass, RL answers 56/128 correctly versus
53/128 for its supervised starting point, with a paired interval spanning no
improvement. All seven changed outcomes have valid answers on both sides, but
some correct finals still contradict their helper chains. Direct answering gets
54/128 with about 28% of the multi-call policies' inference tokens. These are
exploratory results, not an established decomposition or RL advantage.
The initial action-choice screen found little reliable advantage from choosing
among fixed helper strategies. Follow-ups now separate three possible problems:
writing the wrong subquestions, failing to execute them, and ignoring useful
helper answers. Removing an earlier wrong answer did not help, so that proposed
explanation is not confirmed. This is an adaptive study,
not a claim that the original action-router plan succeeded.

Question-plan SFT and four fresh-rollout RL updates are now complete. The initial
SFT validation gain is small and uncertain. RL scored19/64 versus18/64 for SFT
on the other32 validation questions, with only protocol-related wins and a wide
interval. This is not convincing evidence of improved reasoning.
See [RL-PLAN.md](RL-PLAN.md), [EXECUTION-FINDINGS.md](EXECUTION-FINDINGS.md), and
[AGGREGATION-PLAN.md](AGGREGATION-PLAN.md) for the distinct questions being tested.
The broader [HOTPOT-DIAGNOSTIC.md](HOTPOT-DIAGNOSTIC.md) readout is a small official
explorer sample, not a canonical benchmark result.
The completed [four-hop transfer comparison](TRANSFER-FINDINGS.md) finds no
supervised or RL answer improvement: both trained planners score24/128 versus
36/128 for a direct answer using less than a third as many tokens. The
[HotpotQA comparison](HOTPOT-FINDINGS.md) also favors the direct control on its
small exploratory sample. These inputs fit in one call: this is a limitation of
our present question-planner setup, not a refutation of long-context RLMs.

The [aggregation result](AGGREGATION-FINDINGS.md) supports retaining full-source
final answers. [Executor failure analysis](EXECUTOR-BOTTLENECKS.md) motivates
the next [helper-only training comparison](HELPER-TRAINING-DRAFT.md). A
[search-headroom diagnostic](SEARCH-HEADROOM.md) records why simply adding
tree search is not yet the first priority.

[NEXT-COMPONENT-RL.md](NEXT-COMPONENT-RL.md) describes the conditional next
training comparison. The [completed helper comparison](HELPER-FINDINGS.md)
now supports a fixed trained helper:28/64 correct versus19/64 for base helpers
and22/64 for the format reminder. Both interventions eliminate helper-format
failures, but the trained helper also has additional both-valid-answer gains.
This is a promising development result, not established generalization.
The [completed transfer checks](HELPER-TRANSFER-FINDINGS.md) find strong Hotpot
improvement but little exact-answer gain on four-hop MuSiQue. A format-reminder
control explains much of the improvement; trained-versus-reminder differences
must be reported separately from protocol recovery.
The [accepted full-pass RL comparison](FULLPASS-PLAN.md) tests whether root
learning adds value under that fixed helper. If it offers no useful signal,
[COMPOSITION-DIRECTION.md](COMPOSITION-DIRECTION.md) and the
[local asset inventory](COMPOSITION-ASSETS.md) describe a possible alternative.
Existing fixed-decomposition successes must not be relabeled as newly learned
recursive composition.

The completed [single-call trained-adapter control](DIRECT-ADAPTED-FINDINGS.md)
scores49/128 versus54/128 for base direct, an uncertain decline rather than a
generic answering benefit. The completed [plan-only comparison](PLAN-ONLY-FINDINGS.md)
finds similar accuracy at about one third of executed-policy token cost; the
uncertain differences do not establish equivalence or a decomposition advantage.
Read the
[document exposure audit](DOCUMENT-EXPOSURE.md) before calling a panel unseen:
new questions do not imply new source documents. The small
[annotation-compatible check](HELPER-ANNOTATION-COMPATIBLE.md) also cautions
against interpreting final-answer gains as uniformly better intermediate steps.
The completed [next-question screen](NEXT-QUESTION-FINDINGS.md) changes questions
with feedback but finds no exact-answer gain. It is not a trained incremental
policy or a general test of adaptive decomposition.
The completed [training replay](TRAIN-FIT-FINDINGS.md) improves45/64 to49/64,
with four both-valid wins and no losses. It is training-set fit, not generalization.
The completed [repeated-execution diagnostic](FROZEN-EXECUTION-FINDINGS.md) finds
13/64 plans change outcome across four executions, but selecting with three
executions instead of one adds only one correct held-seed answer out of64.
The [matched Hotpot direct control](HOTPOT-DIRECT-CONTROL.md) scores31/64 with
the helper adapter versus36/64 with base weights; multi-call trained helpers
score40/64. The [paired architecture analysis](HOTPOT-MATCHED-ARCHITECTURE.md)
finds the advantage over base direct uncertain, at3.50times its token cost.
The [new-data replication](HOTPOT-FRESH-FINDINGS.md) is complete.
The [bounded RL continuation](RL-CONTINUATION-DRAFT.md) preserved the optimizer
but stopped at checkpoint21 under its declared admission rule; its separately
named development readout is [complete](RL-STOPPED-DOSE-FINDINGS.md).
The [latest prior-art check](LITERATURE-UPDATE-1515.md) narrows what would be novel
beyond existing planner/executor training and adaptive retrieval methods.
The [larger Hotpot validation asset](HOTPOT-DEV-ASSET.md) is now cached with a
pinned revision; its fresh panel has now been evaluated as reported above.

## What was controlled in the initial screen

Each question produces one initial attempt. Four alternatives start from that
same attempt. Every final answerer retains all original documents. Three seeds
repeat the continuations, not the initial attempt. The three helper alternatives
have equal call counts and token limits; finishing uses one fewer call.

The first pilot exposed a possible answer-format problem. `answer_format.py`
therefore replays only the final answers with clearer short-answer instructions,
using exactly the saved initial attempt and helper report. It changes no gold
labels and applies no gold-dependent answer cleanup.

`analysis.py` counts every planned question, records incomplete work separately,
and tests whether an action chosen using two repeats also helps on the third.
That test uses past outcomes for the **same question**. It is a diagnostic of
potentially useful differences, not a deployable policy or a generalization result.

## Artifacts and execution

External study root:
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`

- `inputs-001`: immutable256 train/64 validation/64 four-hop transfer questions.
- `source-001/probe.py`: sealed pilot collector; later repository changes improve
  reporting and add optional span instructions without changing that live copy.
- `pilot-001`: plans, owner receipts, native calls, checkpoints and arm outcomes.
- `ENVIRONMENT.json`: observed environment versions and allocation information.
- `plan-probe-001`: completed model-versus-reference question diagnostic.
- `execution-probe-001`: bundled-versus-step-by-step execution diagnostic.
- `planner-sft-inputs-001`: 256 question-list training targets, with token audit.
- `planner-sft-001`: completed48-update supervised training; fixed checkpoint0048.
- `planner-eval-isolated-001`: completed matched base/SFT validation on32 parents.
- `rl-planner-001`: completed four-update root-only RL,256 fresh trajectories.
- `held-sft-001` and `held-rl-001`: completed readouts on the other32 validation parents.
- `aggregation-probe-001`: completed frozen-trace final-evidence comparison.
- `helper-sft-inputs-001`:570 train-only annotated step-answer examples, sealed.
- `helper-sft-001`: completed one-epoch helper training,36updates,584.5 training seconds.
- `helper-eval-001`: completed base/trained/reminder helper comparison,
  192episodes/613newcalls; `analysis-helper-001.json/.md` is authoritative.
- `rl-fullpass-001`: root-only RL with frozen helper-SFT36,
  completed16updates over256training parents,1,024attempts and4,335calls;
  [training-process findings](FULLPASS-TRAINING-FINDINGS.md). Its new-question
  fresh-panel comparison is complete; see [FRESH-CONTRACT-FINDINGS.md](FRESH-CONTRACT-FINDINGS.md).
- `fresh-dev-inputs-003`: new64-question panel with prior-study parent/component
  exclusions. Earlier001/002 preparation versions are superseded; see
  [FRESH-DEV-PANEL.md](FRESH-DEV-PANEL.md).

`train_planner.py` trains question-list generation by default; explicit
`--role helper` instead trains short step answers on separate prepared examples.
`eval_planner.py` compares base and trained planners under the same
title-index-only observation, with a fixed helper contract and unchanged base
final model. The original comparisons use base helpers; later comparisons must
name any reminder or helper adapter explicitly. Its `--execution`
option selects bundled or isolated helpers. The primary trained checkpoint is
fixed at update48, not chosen by validation score. See
[PLAN-SFT-DRAFT.md](PLAN-SFT-DRAFT.md) for the supervision and architecture limits.

These cluster-specific scripts reuse the existing native inference client and
owned-service launcher under `sidecars/unattended-breadth-20260914`, plus cached
official MuSiQue metrics. They are not self-contained on a fresh laptop. Their
dependencies and immutable input hashes are recorded in each run plan. Large
models, datasets and raw outputs stay outside Git; pushing source is not a backup
of those external artifacts.

The main agent is the only GPU launcher. Use the shared coordinator lock. Never
start HF training beside the inference owner. Keep sealed source copies unchanged
while owners run. `STATUS.json` is operational; completed analyses and native
receipts establish scientific outcomes.

Focused tests run with the existing inference environment:

```bash
/project/alex_phd/envs/prime-rl-5990b1b/bin/python -m pytest -q \
  experiments/selective_delegation/test_data.py \
  experiments/selective_delegation/test_probe.py \
  experiments/selective_delegation/test_analysis.py \
  experiments/selective_delegation/test_answer_format.py
```

Do not describe an accuracy increase from revised answer formatting as an
improvement from delegation or reinforcement learning. Actual training, if
performed, must be reported separately with its own held-out
evaluation, training curve and costs.
