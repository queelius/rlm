# Learning when another reasoning step is worthwhile

This exploratory study asks whether a controller can learn when to finish an
answer, reconsider it, ask a focused evidence question, or work through proposed
subquestions. The helpers initially remain unchanged. This is a small controlled
decision, not yet a learned recursive RLM or a new state-of-the-art claim.

Start with [DESIGN.md](DESIGN.md), [LEDGER.md](LEDGER.md), and
[LITERATURE.md](LITERATURE.md). The implementation checklist is [PLAN.md](PLAN.md).
[TRAINING-DRAFT.md](TRAINING-DRAFT.md) describes conditional training work; its
existence does not mean a training run has happened.

For the current evidence and direction, read [FINDINGS.md](FINDINGS.md) first.
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
The [frozen-plan transfer checks](HELPER-TRANSFER-PLAN.md) are queued to test
that limitation on harder MuSiQue questions and HotpotQA.
The [accepted full-pass RL comparison](FULLPASS-PLAN.md) tests whether root
learning adds value under that fixed helper. If it offers no useful signal,
[COMPOSITION-DIRECTION.md](COMPOSITION-DIRECTION.md) and the
[local asset inventory](COMPOSITION-ASSETS.md) describe a possible alternative.
Existing fixed-decomposition successes must not be relabeled as newly learned
recursive composition.

The queued [single-call trained-adapter control](DIRECT-ADAPTED-PLAN.md) asks
whether the helper's gain also appears without any decomposition. Read the
[document exposure audit](DOCUMENT-EXPOSURE.md) before calling a panel unseen:
new questions do not imply new source documents. The small
[annotation-compatible check](HELPER-ANNOTATION-COMPATIBLE.md) also cautions
against interpreting final-answer gains as uniformly better intermediate steps.
[INCREMENTAL-DESIGN.md](INCREMENTAL-DESIGN.md) is conditional preparation for
testing answer-conditioned next-question choice, not a completed experiment.
Two other conditional diagnostics separate
[learning on the training examples](TRAIN-FIT-DIAGNOSTIC.md) from
[unreliable rewards when the same plan is executed again](FROZEN-EXECUTION-DIAGNOSTIC.md).
Neither is evidence of improved performance on new questions.
The [larger Hotpot validation asset](HOTPOT-DEV-ASSET.md) is now cached with a
pinned revision; no new evaluation panel has been selected from it yet.

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
- `rl-fullpass-001`: accepted/running root-only RL with frozen helper-SFT36,
  up to16updates over256training parents; no held-out outcome yet.
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
