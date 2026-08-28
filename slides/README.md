# Preliminary RLM research slides

`rlm-preliminary-results.tex` is a five-slide research update for a short,
somewhat non-specialist discussion. It uses everyday explanations before any
method name and separates completed exploratory work from the planned
self-training experiment in the sibling `rlm-bootstrap` project.

## Compile

From this directory, with a standard TeX Live installation containing Beamer
and PGF/TikZ:

```bash
make
```

Use `make clean` to remove temporary LaTeX files while retaining the PDF, or
`make distclean` to remove the temporary files and PDF. To invoke the compiler
directly instead:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error rlm-preliminary-results.tex
```

Alternatively, run `pdflatex` twice:

```bash
pdflatex -interaction=nonstopmode -halt-on-error rlm-preliminary-results.tex
pdflatex -interaction=nonstopmode -halt-on-error rlm-preliminary-results.tex
```

## Suggested timing

- Slide 1: 30 seconds -- state the motivating question.
- Slide 2: 45 seconds -- explain that the RLM provides tools while the language
  model decides how to use them.
- Slide 3: 75 seconds -- present the exploratory evidence and immediately state
  its limitations.
- Slide 4: 60 seconds -- explain the planned one-generation self-training test.
- Slide 5: 45 seconds -- solicit advice on tasks, controls, and priorities.

Total: about four to five minutes, leaving most of a brief meeting for
discussion.

## Evidence behind slide 3

- V3A development pilot:
  `/project/alex_phd/runs/rlm-v2-spikes/broad-policy-sft-v3a-development-pilot-20260825T050000Z/pilot-result.json`
  - 36 episodes per condition across six task families.
  - The step-648 controller had macro qualified accuracy `0.6667` and 24
    qualified-correct episodes.
  - Direct, base-RLM, and earlier-controller comparison conditions each had
    macro qualified accuracy `0.0`.
- V3B paired SFT training:
  `/project/alex_phd/runs/rlm-v2-spikes/broad-policy-sft-v3b-ops-20260825T235900Z/TRAINING_RESULT.json`
  - Four successful runs: two curricula crossed with two training seeds.
  - Runtime ranged from 3,747 to 3,792 seconds.
  - Mean treatment-minus-control training-loss difference was `+0.00102`;
    mean runtime difference was `+10` seconds.
  - The artifact explicitly withholds efficacy conclusions pending held-out
    paired evaluation.
- RLVR one-step smoke:
  `/project/alex_phd/runs/rlm-v2-spikes/rlvr-one-step-smoke-v3-artifacts/V3_TERMINALIZATION.json`
  - All four stochastic rollouts had reward `1.0` and sample variance `0.0`.
  - The optimizer did not start.

## Status of slide 4

Slide 4 describes planned work, not preliminary results. The research question
and matched plain-self-training control come from the sibling project:

- `../rlm-bootstrap/rlm_self_training_experiment_handoff.md`
- `../rlm-bootstrap/docs/superpowers/specs/2026-08-24-clean-causal-self-sft-design.md`

As of this deck, `rlm-bootstrap` has reproducible dataset and experiment-design
foundations, but no self-training rollout, model training, or evaluation result.
