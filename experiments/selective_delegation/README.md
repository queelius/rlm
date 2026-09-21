# Learning when another reasoning step is worthwhile

This exploratory study asks whether a controller can learn when to finish an
answer, reconsider it, ask a focused evidence question, or work through proposed
subquestions. The helpers initially remain unchanged. This is a small controlled
decision, not yet a learned recursive RLM or a new state-of-the-art claim.

Start with [DESIGN.md](DESIGN.md), [LEDGER.md](LEDGER.md), and
[LITERATURE.md](LITERATURE.md). The implementation checklist is [PLAN.md](PLAN.md).
[TRAINING-DRAFT.md](TRAINING-DRAFT.md) describes conditional training work; its
existence does not mean a training run has happened.

## What is controlled

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
justified by these diagnostics, must be reported separately with its own held-out
evaluation, training curve and costs.
