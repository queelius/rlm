# Preliminary RLM research slides

`rlm-preliminary-results.tex` is an eight-frame research update for a short, somewhat
non-specialist discussion. It uses plain language to connect three things:

1. the larger idea that an RLM scaffold may help a model build an unfamiliar solution from
   familiar pieces;
2. preliminary supervised-training findings; and
3. the planned self-training and reward-based research that will test the larger idea more
   directly.

The slides intentionally show only the salient experimental progression. The detailed datasets,
results, negative findings, limitations, GPU-time accounting, sources, and likely advisor
questions are in [`preliminary-experiment-analysis.md`](preliminary-experiment-analysis.md).

For presentation help, concrete examples, a short explanation of every slide, and plain-language
answers to likely questions, see [`speaker-guide.md`](speaker-guide.md).

## Compile

From this directory, with a standard TeX Live installation containing Beamer:

```bash
make
```

Use `make clean` to remove temporary LaTeX files while retaining the PDF, or `make distclean` to
remove the temporary files and PDF. To invoke the compiler directly instead:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error rlm-preliminary-results.tex
```

Alternatively, run `pdflatex` twice:

```bash
pdflatex -interaction=nonstopmode -halt-on-error rlm-preliminary-results.tex
pdflatex -interaction=nonstopmode -halt-on-error rlm-preliminary-results.tex
```

## Suggested timing

- Slide 1: 25 seconds -- state the motivating question and preliminary status.
- Slide 2: 40 seconds -- explain the workspace and smaller model calls.
- Slide 3: 55 seconds -- explain the compositional-generalization hypothesis.
- Slide 4: 40 seconds -- summarize the repeatable train, save, serve, and evaluate process.
- Slide 5: 60 seconds -- show the two trained model actions and the RLM observation between them.
- Slide 6: 60 seconds -- contrast a local record task with a model-to-model handoff.
- Slide 7: 50 seconds -- describe the planned direct-practice versus RLM-practice comparison.
- Slide 8: 30 seconds -- close with the larger research question and solicit input.

Total: about five to six minutes, leaving most of a brief meeting for discussion.

## Evidence behind slides 4--6

### Completed training campaign

Several supervised-training studies completed, culminating in four matched runs that tested
canonical versus varied input layouts. Those latest adapters have not yet received their held-out
paired evaluation, so the slides do not present a behavioral conclusion from them.

Primary matched-training record:

- `/project/alex_phd/runs/rlm-v2-spikes/broad-policy-sft-v3b-ops-20260825T235900Z/TRAINING_RESULT.json`

### Narrow RLM routine: 29/30

The first experiment trained Qwen3-8B on 160 controller-turn examples from 80 symbolic tasks. The
predeclared endpoint solved 29 of 30 held-back tasks, compared with 11 of 30 for direct answering.
This was a narrow shared-schema mechanism test, not broad generalization.

Primary result:

- `/project/alex_phd/runs/rlm-v2-spikes/abi-curriculum-sft/RESULTS.md`

### Fixed dispatcher: 192/192

The next clean mechanism experiment trained one byte-identical six-operation Python dispatcher on
576 examples. After preregistered checkpoint selection, the chosen checkpoint solved all 192
sealed tasks. The result establishes that supervised training can install a causally valid fixed
RLM mechanism; it does not establish broad planning or recursive decomposition.

Primary results:

- `/project/alex_phd/runs/rlm-v2-spikes/broad-policy-sft-v2b/EXECUTION_PLAN.md`
- `/project/alex_phd/runs/rlm-v2-spikes/broad-policy-sft-v2b/evaluation-v2b-development-screen/sealed-test/summary.json`

### Broader workflows: 24/36

The broadest completed training study used 2,592 controller turns covering 24 operations and six
workflow families. On a disjoint exploratory development confirmation, the selected checkpoint
solved 24 of 36 tasks. It solved all 24 tasks across four local families: one-turn dispatch,
inspect-then-compute, persistent-state verification, and repair after a visible fault. It did not
solve the 12 tasks in the two model-delegation families. This was development-only, used one
training seed, and was not a sealed or replicated result.

Primary results:

- `/project/alex_phd/runs/rlm-v2-spikes/broad-policy-sft-v3a/EXECUTION_PLAN.md`
- `/project/alex_phd/runs/rlm-v2-spikes/broad-policy-sft-v3a-development-pilot-20260825T050000Z/confirm/summary.json`

## Why the earlier 43/192 score is not the headline result

The earlier sealed score of 43/192 versus 12/192 for direct answering is a real measurement, but
the training generator chose operation-specific first-turn targets using information that the
model could not see. The checkpoints learned fixed favorite operations rather than a
task-conditioned policy. This is important negative evidence about shortcut learning and is
documented in the companion analysis rather than presented as the deck's positive training
result.

Primary diagnosis:

- `/project/alex_phd/runs/rlm-v2-spikes/SESSION_CHECKPOINT_20260824.md`

## Status of planned work

Slide 7 describes planned work, not a completed result. The research question and matched
plain-self-training control come from the sibling project:

- `/project/alex_phd/repos/rlm-bootstrap/rlm_self_training_experiment_handoff.md`
- `/project/alex_phd/repos/rlm-bootstrap/docs/superpowers/specs/2026-08-24-clean-causal-self-sft-design.md`

As of this deck, `rlm-bootstrap` has reproducible dataset and experiment-design foundations but no
self-training rollout, model training, or evaluation result. Likewise, the reward-based training
diagnostics in the current experiment never reached an optimizer update. The matched input-layout
evaluation and compact-target training remain future work.
