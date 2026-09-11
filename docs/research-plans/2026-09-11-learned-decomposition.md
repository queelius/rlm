---
title: Learning how much decomposition a problem needs
date: 2026-09-11
status: exploratory_proposal_not_a_frozen_campaign
question_ids:
  - rq:rl-effective-feedback
  - rq:adaptive-decomposition
  - rq:depth-value
  - rq:decomposition-generalization
evidence_status: existing_results_exploratory
research_store: /project/alex_phd/runs/rlm-research-r4
hardware_at_resumption: one_A100_40GB_MIG_on_an22
allocation_end_from_environment_utc: 2026-09-15T17:30:16Z
---

# The research direction

Can a small model learn **what work to delegate, how to divide it, and when further
decomposition is worth its cost**, including on unfamiliar tasks?

Treat these as related but different abilities. Executing a supplied plan is not choosing
a plan. Making many helper calls is not necessarily using recursive depth. Increasing depth
is not itself progress: sometimes a direct calculation or a few small calls are better.

This is a ranked starting proposal, not a commitment to run every combination. Change the
queue when evidence changes which comparison is most informative.

## What our results motivate

- SFT improved the requested routine on a shared 72-task panel: 12 confirmed successes
  before the additional training, versus 55 and 53 for separately trained copies. The task
  families were familiar. This is a useful starting policy, not learned general planning.
- Training improved execution of supplied new combinations (2 to 29 confirmed successes),
  but instructions supplied the decomposition. We need a separate no-plan condition.
- Matching names improved large helper requests, but smaller requests were competitive and
  faster in the local equal-work comparison. A learned policy must compete with these simple
  alternatives, not just a weak large-request baseline.
- The longer RL pair did not improve final-answer counts: 57 before, 55 confirmed after
  answer-reward training, and 54 after adding a calculation-consistency bonus. Two answer-reward
  results were unavailable. Both arms attempted 576 training solutions, but only 108 and 147
  grouped trajectories contributed to their 19 and 21 updates. These facts motivate measuring
  useful training signal, not merely buying longer runs. They do not identify the cause of failure.
- The latest whole-task handoff pilot mostly exposed interface misuse and incomplete attempts.
  A small explicit-API recovery is an integration diagnostic, not evidence of learned decomposition.

Sources: [advisor evidence ledger](../../slides/2026-09-11-advisor-meeting/evidence-and-methods.md),
especially S3, S4, E1, R3; and the external store's
`analyses/root-authenticated-map-reward-pair-live-2026-09-11/final-pair-2026-09-11T0840Z/REPORT.md`.

## 1. Make RL informative before making it large

**Question:** Is progress limited by unreliable actions, weak exploration, the helper's reading
ability, or the reward we give the root?

Start with the existing 4B SFT policy and a working, unchanged interface. On a training/development
split, sample several solutions per task. Measure valid completion, correct final answers,
different strategies, and the fraction of groups containing different rewards. Compare real
helper replies with exact helper labels as a diagnostic. Exact labels are an oracle condition,
not a deployable baseline or a replacement for real-helper evaluation.

- If almost every attempt fails to call tools correctly, repair that skill with a small SFT set.
- If attempts are valid but all fail, simplify the curriculum or improve the leaf skill.
- If attempts all succeed, increase difficulty; constant within-group rewards give ordinary
  group-relative updates little information.
- If occasional attempts succeed but the usual answer is wrong, compare RL with self-training
  on successful attempts. Report pass@1 and multi-sample success separately; an oracle selecting
  the correct test answer is not a free deployable method.

**Initial comparison:** unchanged SFT policy, successful-trajectory self-SFT, and root-only RL.
Use the same starting checkpoint and equal collection budgets. Self-SFT uses selected successful
trajectories; RL refreshes its on-policy rollouts. Do not imply those trajectories stay identical.
Add shared parent-and-child RL only after establishing the simpler baseline. If the oracle-helper
diagnostic identifies the helper as the bottleneck, prioritize that shared/leaf training branch.

Keep final correctness primary. Do not promote the old numerical agreement bonus unchanged:
agreement with a mistaken helper is not truth. Initially use hard compute caps; introduce a
small cost penalty in a separate comparison once the policy can succeed. Do not reward making
more calls or deeper trees. Preserve timeouts as unsuccessful completions within the budget.

Proposed pilot shape: one A100, 4B adapter training, a 30–45-minute rollout diagnostic, then
bounded 60–90-minute training arms if actual throughput supports them. These are caps, not
predictions. Save every update, optimizer state, and the rollout cursor. Expand only after seeing
valid reward variation and a useful held-out signal. Separate matched opportunities from actual
optimizer dose, and report both.

## 2. Learn a strategy rather than another fixed routine

**Question:** Can the model choose a useful decomposition from the task and observations,
without receiving a worked plan in the test prompt?

Start with a small portfolio of legitimate options: solve directly, search for relevant material,
split independent records into small helper requests, or turn text into structured relations and
calculate in Python. These options are a controlled first experiment, not the final unrestricted
planner. Later allow the model to write its own decomposition program.

Create tasks where the best option varies. A simple exact lookup should not require a tree;
aggregation may favor independent calls; relational tasks may favor extracting a graph before
calculation. Include cases where splitting loses a relationship across boundaries. Otherwise,
the experiment only teaches “always split.”

SFT should demonstrate several useful strategies and how to inspect the environment, not a
single universal template. RL then rewards the final solution under the stated budget. Compare
against a prompted choice policy, each fixed strategy, and a simple input-size rule.

**Promotion criterion:** the learned choice improves the accuracy–cost tradeoff on held-out
task structures or domains. A win confined to seen templates is strategy selection within a
distribution, not general decomposition.

**A particularly promising harness question is what information a child should return.**
Suppose we count customers who bought both A and B. A customer's A purchase may be in one chunk
and B purchase in another. Adding each chunk's count of customers who bought both gives the
wrong answer. Returning each customer's observed product set and merging sets preserves the
needed relationship. Compare fixed scalar summaries, fixed sufficient structured summaries,
and a policy that learns the return representation. Vary chunk boundaries and subject matter.
Reward correct answers, not consistency alone: two decompositions can agree on the same wrong
answer. This tests a substantive extension of our handoff work beyond naming records.

## 3. Learn whether another recursive level is worth it

**Question:** At a child task, should the model finish locally, inspect more evidence, or create
children of its own?

Use a minimal shared interface: inspect, calculate, delegate, and return a result. Every node
sees its task, available evidence, and remaining global budget. It need not have a special named
expert role. Preserve record identities and evidence references across calls.

First compare no delegation, one helper layer, fixed deeper recursion, a cheap size-based rule,
failure-triggered decomposition such as [ADaPT](https://aclanthology.org/2024.findings-naacl.264/),
and learned stop/delegate decisions. Hold the total budget fixed and report actual cost too.
Longer input alone is not proof that deeper recursion is needed. With Python and loops, a flat
program can express many recursive computations; our question is empirical usefulness under
resource limits, not an assertion that a visible tree is mathematically necessary.

At selected replayable training states, sample both local completion and further delegation.
Their independently scored continuations estimate when extra decomposition helps. Charge the
extra sampling to training cost. A single pair is noisy; do not call it a definitive causal
attribution. This can first train a small stop/delegate policy or weight self-training examples
before introducing a more complicated tree-wide RL objective.

Include easy cases, cases with relevant evidence separated across inputs, and cases in which
local evidence is insufficient. Measure both unnecessary delegation and premature stopping.
The important result would be appropriate depth that transfers, not a larger average depth.

## 4. Generalization must have more than one axis

| What is held out? | What the test can establish |
|---|---|
| Source documents or generated problem instances | Transfer beyond the exact training records |
| Longer inputs with the same task structure | Length extrapolation |
| New combinations and deeper dependencies | Structural transfer, separate from merely longer input |
| Different subject matter or datasets | Domain/task transfer |
| New valid partitions, orderings, and names | Robustness to how work is divided and represented |

Use a controlled generated family with exact answers for mechanism tests, a natural-text task
with supporting evidence such as [HotpotQA](https://hotpotqa.github.io/), and an external realistic
long-context check such as selected multi-document or code tasks from
[LongBench v2](https://github.com/THUDM/LongBench). Its dataset split named `train` is not permission
to train on the benchmark evaluation items. Freeze our evaluation identities before tuning.
OOLONG remains useful as continuity with earlier work, but should not be the only task family.

Do not average these into a single headline that hides failure on one family. Separate source
groups across splits, hold out generator templates/structures, preserve official scoring, and
record known prior exposure. Public benchmarks cannot establish absence from model pretraining.
Use new confirmation data after exploratory decisions, not endless tuning against the same panel.

## Recent primary work that changes the plan

- [Recursive Agent Optimization, May 7, 2026](https://arxiv.org/abs/2605.06639v1):
  directly trains recursive agents when and how to delegate and communicate, with reported
  generalization to harder tasks. This is a central comparison, not an adjacent citation;
  “learned recursive delegation” by itself is already prior art.
- [Tree Search for LLM Agent Reinforcement Learning, ICLR 2026](https://arxiv.org/abs/2509.21240v3):
  Tree-GRPO shares rollout prefixes and derives step-level learning signals from outcome rewards.
  A tree of alternative training continuations is different from a tree of delegated subtasks
  within one solution. The method informs credit assignment, not proof that deeper delegation helps.
- [Recursive Language Models, revised May 2026](https://arxiv.org/abs/2512.24601v3):
  the external-context and recursive-call foundation already includes small-model post-training.
  Simply fine-tuning an RLM is not a new contribution.
- [Reinforcing Recursive Language Models, May 13, 2026](https://www.alphaxiv.org/blog/reinforcement-learning-for-rlms):
  an author research post with SkyRL code. It uses SFT warm starts, turn-wise training examples,
  and a shared parent/child policy with inherited advantages and normalized child losses. The
  reported multi-paper run used eight H200s and judge-based rewards; do not treat it as a proven
  one-A100 RLVR recipe. Its explicit strategy prompts also leave strategy discovery distinct.
- [Language model harnesses are compositional generalizers, July 2026](https://alexzhang13.github.io/blog/2026/harness/):
  an author research post reporting short-to-long and cross-domain training transfer. It supports
  testing shared problem structure, while also noting that short-task training can learn a
  nongeneralizing shortcut. It does not establish that every harness or strategy generalizes.
- [Training Cost-Aware Recursive Language Models with Reinforcement Learning, 2026](https://essay.utwente.nl/essays/110199):
  a thesis reporting Qwen3-4B/Prime-RL/LoRA results, including worse depth-2 than depth-1 performance.
  The repository abstract was verified; full-text retrieval failed in this pass. Treat detailed
  reproduction requirements as unresolved, rather than copying an uninspected recipe.
- [Recursive Models for Long-Horizon Reasoning, ICML 2026](https://arxiv.org/abs/2603.02112v2):
  recursion with isolated contexts, theoretical analysis, and SAT/Go experiments. This motivates
  controlled structural tests but is not evidence that our natural-text planner already benefits.

This is a targeted literature pass, not an exhaustive claim of novelty. In particular, generic
learned delegation, recursive policies, and counterfactual credit have substantial prior art.
For stopping specifically, [CaRT](https://arxiv.org/abs/2510.08517) studies counterfactual
stop/continue training examples; do not claim that general idea as new. The separate bounded
prior-art memo is in the research store under
`analyses/post-meeting-directions-2026-09-11/RECURSION_PRIOR_ART.md`.
The `new-ideas/ideas.md` citation titled “Python is the Counterfactual” could not be independently
verified in this pass; do not promote its bibliographic details without a retrievable primary source.

## Reuse the sibling projects without making them prerequisites

`rlm-bootstrap` supplies the useful self-training baseline: generate successful trajectories and
learn from the model's own actions, not environment outputs. Reuse its data and split ideas without
migrating the entire live training stack. Inspected commit: `7f79801073be7799a2318c414f29a764af1a1a0b`.

`structured-decomposition-benchmark` contributes controlled exact-versus-natural helper reports,
one-report error interventions, and matched-compute direct baselines. It explicitly distinguishes
fixed decomposition from learned routing. Inspected commit: `c7ee38127e5f37c327b77f518a21f0bab151ff33`.

## Preserve what makes later analysis possible

For each run, record question and decision IDs; code, model, adapter, data and prompt hashes;
train/development/test provenance; seed; budget; every attempted task and termination reason;
checkpoint and optimizer cursor; reward components; effective training groups and target tokens;
and training, inference, loading, and idle time separately.

For each tree node, record parent ID, depth, visible evidence, chosen action, child inputs and
outputs, global budget remaining, actual tokens/time, and final verification. Keep natural policy
choices separate from forced or oracle interventions. Log what changed our next decision,
including negative findings and alternative explanations.

The first recovery pilot and live ownership belong in the external `RESEARCH_QUEUE.md`.
On resumption, attempt004 stopped before model calls because the runtime belonged to the old
allocation. A separate new-allocation runtime recovery was assigned; it must not be described as
successful inference until actual calls exist. The old immutable runtime must not be relabeled.
The proposed training and recursion comparisons are not ready jobs until their inputs, launch
command, cap, and owner exist. A publication candidate is **cost-effective learned delegation
that transfers across structures and domains**—not merely “RL worked” or “we added identifiers.”
