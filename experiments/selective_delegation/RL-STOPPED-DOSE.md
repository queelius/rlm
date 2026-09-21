# Stopped RL continuation: checkpoint21, not checkpoint24

The continuation stopped under its declared admission rule at batch22, before an
optimizer update for that batch. Five additional updates were committed, taking
the root from checkpoint16 to **checkpoint21**. The exact-checkpoint24 readout was
skipped as specified. Its collector, comparator, and outputs were not relaxed or
relabeled.

The new decision `POST-RL-STOP-DECISION-001` accepts a separately named
`fresh-contract-rl21-001` readout. Checkpoint21 is the **last committed checkpoint
following the stopping rule**, chosen before its development outcomes, not a
best-checkpoint selection. Its matched fresh003 results are pending; no further
training is accepted by this note.

## Actual stopped-batch evidence

Source: `R/rl-continuation-001/batch-0022`. Here `R` is
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.

- 16 TRAIN parents × 4 root candidates = 64 episodes; official final regrading
  independently reproduced 48/64 correct, mean reward 0.75.
- Twelve groups had rewards `[1,1,1,1]`; four had `[0,0,0,0]`. **Every within-parent
  advantage was zero: 0/64 nonzero advantages.** There were zero qualifying groups,
  so no gradient/optimizer update22 was attempted.
- All 64 root plans were valid, all 64 finals passed the response protocol, and
  all 275 native calls were available: 64 roots, 147 helpers, 64 finals. Request
  digests and final scores were checked against actual native receipts.
- Exact question-list diversity was not collapsed: 11 parents had four distinct
  lists, two had three, and three had two. Plan lengths were 45 two-step and 19
  three-step lists.
- Static dependency-reference patterns differed within six parents: five parents
  had two patterns and one had three; ten had one pattern. No generated plan had
  a forward or self-reference. This measure uses the ordered `#k` references and
  number of questions, **not semantic equivalence or annotated-step correctness**.

Thus zero admission arose from **flat sampled terminal reward despite distinct
valid plans**, not invalid-only variation or identical sampled strings. Some
groups were all-right and others all-wrong. This one block does not establish
global collapse, absence of useful planning differences elsewhere, or convergence.
It supplies no held-out learning estimate. Fresh parents, reward stochasticity,
and full-source final answering remain relevant limitations.

The continuation terminal reports `admission_failed_no_update`, global optimizer
step21, five additional committed updates, no failure and no unresolved started
requests. Total continuation acquisition, including the unupdated batch22, was
1,612 native calls, 3,672,022 input tokens, 21,428 output tokens, and 1,866.7 seconds.
These are training acquisition costs, not the cost of the future readout.

## The stopping rule is not a limit of reinforcement learning

Stopping the whole run at its first uninformative batch was our conservative
pilot rule, not a mathematical requirement. A zero-advantage batch can instead
be recorded without an optimizer update and followed by fresh, predeclared
training questions. We should not infer convergence from this stop.

[DAPO, section3.2, inspected version2](https://arxiv.org/html/2503.14476v2)
describes a related established approach: collect more prompt groups and omit
all-correct/all-wrong groups when forming a learning batch. Its larger math
experiments do not establish that this will help our short question planner.
It also consumes additional sampling compute, which must be counted here.

Decision: do not alter the stopped owner or retroactively relax its rule. Before
accepting more root training, inspect the checkpoint21 readout and the proposed
[execution-credit diagnostic](EXECUTION-CREDIT-QUESTION.md). If training remains
worth pursuing, use a declared rollout/compute cap and separate sampled-batch,
zero-update, and optimizer-step counters. A single zero-gradient batch should
not automatically halt that new run. This would be a sampling/stopping change,
not a new RL algorithm or an apples-to-apples continuation of the old schedule.

## Separate matched readout implementation

`compare_rl_stopped.py` preserves the original MuSiQue native regrading,
planned-denominator accounting, strict matched source/case/seed/helper/model/cap
checks, and atomic-component bootstrap logic. It explicitly names the analysis
policies `rl16` and `rl21` while leaving native condition `rl` untouched. Additional
checks require committed checkpoint21 descending from the compared checkpoint16,
the successful admission-stop terminal, batch22's zero-admission receipt, unchanged
checkpoint file hashes and training source hashes, and no later committed checkpoint.

The original `compare_rl_dose.py` is unchanged, SHA-256
`41c7e738dc5d9417ac63ee2d69b4226f4edfa48a993a5ec4fe63e7c8932d5b78`.
The new analyzer reuses its strict contract/protocol scoring helpers; it does not
call its checkpoint24 comparison with a substituted checkpoint.

Three focused fixtures pass: stopped ancestry/source hashes; rejection of failed
terminal or changed checkpoint weights; and truthful native missing/protocol
accounting with explicit RL21 comparison labels. Ruff passes. The real stopped
ancestor check also passed, binding 17 training/checkpoint/source files without
loading a model or optimizer. No new development score has been inspected.

After separately sealing the analyzer and its dependencies, run on completed
evaluation receipts only:

```bash
CPUPY <analysis-seal>/compare_rl_stopped.py \
  --rl16-output R/fresh-contract-rl-001 \
  --rl21-output R/fresh-contract-rl21-001 \
  --cases R/fresh-dev-inputs-003/cases.jsonl \
  --report R/analysis-fresh-rl-stopped-dose-001.json
```

JSON and Markdown output paths must be unused. Missing outcomes remain
unobserved, and incomplete-panel differences are labeled differences of lower
bounds, not effect estimates. The exposed panel is balanced 32 two-hop/32 three-hop
parents, not an estimate of the natural whole-development mix. More updates also
mean more second-pass data exposure; a difference is not an isolated optimizer
dose effect.

Key immutable receipts:

| Receipt | SHA-256 |
|---|---|
| Batch22 `BATCH.json` | `fe660f1eeafdeb702816c6aa1de1f1ac5bf953ca5e4d67a55742696a8da9fb6f` |
| Checkpoint21 `COMMIT.json` | `703b30fd49b7d5bf954caa08be788a84139178aacf25f98cc07bf7e55efc5fb8` |
| `TERMINAL-02abab8fb18e.json` | `c9cde873fb252845067e87dcd98412042a5ade9a71e43515d1f4e0000e883a98` |
