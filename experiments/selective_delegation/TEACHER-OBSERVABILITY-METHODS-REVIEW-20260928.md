---
date_read: 2026-09-28
status: primary_methods_review_not_exhaustive_novelty_search
question_id: TC-TEACHING-MECHANISM
assets_downloaded: none
GPU_jobs_launched: none
---

# Teacher observability: known failure mode, narrower experimental opportunity

The general finding is established prior art: a successful privileged teacher can be a poor
supervisor for a partially informed learner. Even **identical action labels with different
conditioning** are not, by themselves, a new experimental idea. A defensible contribution would
be the more specific, still-unproved result that an **offline, executable query-schedule repair
improves tool-agent transfer while preserving every optimizer minibatch's supervised targets**.
Frame this as controlled evidence and a data transformation, not a new imitation-learning
principle, privileged-distillation objective, causal-discovery method, or RLM/decomposition gain.

## What was actually read

Primary arXiv HTML, accessed September28,2026. Versions/dates below identify the reviewed texts.
Methods and the listed limitations/results were read; proofs, repositories and experiments were
not independently reproduced. This supplements, without replacing, the earlier
[abstract-only screen](TEACHER-OBSERVABILITY-PRIOR-ART-20260928.md).

| Primary source and read scope | Actual method/claim and boundary relative to ours |
|---|---|
| [Student-Informed Teacher Training, v2, February27,2025](https://arxiv.org/html/2412.09149v2): §§3–4,5.2, AppendixA.2–A.3 | Co-trains the teacher with both a teacher–student divergence reward penalty and a direct divergence gradient. A proxy student and shared action decoder make the alignment practical. Its drone/arm behaviors make relevant objects visible before acting; this is already an observability-aware teacher. It changes the learned policy and visited states, not a fixed action multiset. Joint teacher training and paired student observations are required; evidence is maze/robotics, not a target-preserving offline LM trace transformation. |
| [Guided Policy Optimization, v2, March13,2026](https://arxiv.org/html/2505.15418v2): §§2.2,3.1–3.3,4.4–5 | TigerDoor explicitly demonstrates a privileged teacher that never listens although the student must. GPO alternates guide improvement, learner imitation and guide backtracking; practical variants use KL penalties/clipping and a learner RL term. Its idealized mirror-descent result assumes an unrestricted guide class and the stated projection/backtracking procedure; it is not a guarantee for our SFT. Poor guide information, memory capacity and divergence thresholds are discussed failure cases. Our narrower question retains the query labels and changes when they are conditioned on available evidence, rather than adding the missing exploration action. |
| [Privileged Information Distillation for Language Models, v3, February16,2026](https://arxiv.org/html/2602.04942v3): §§3–4,7–8,10 | A shared model acts as PI-conditioned teacher and unconditioned student. π-Distill combines teacher reward/KL training with teacher-sampled student updates; OPSD samples the student and adds student-to-teacher reverse KL. Action traces, tool names alone and generated hints provide PI. Thus action-only multi-turn LM transfer and teacher–student distribution mismatch are already addressed. Their §7 varies PI utility/divergence; §10 explicitly limits that mechanism analysis to observational comparisons without systematic variable control. Frontier-derived PI and models≤8B also limit scope. Our possible contribution is a tightly controlled history intervention, not discovery of the PI-transfer problem. |
| [Causal Confusion in Imitation Learning, v2, November4,2019](https://arxiv.org/html/1905.11979v2): §§3–6 | §3.2 already trains identical architectures on the same demonstrations while altering observation conditioning, including a previous-action nuisance variable. Their intervention method searches graph-conditioned feature subsets using expert queries or environment returns. It needs suitably disentangled representations; the solution is validated mainly in synthetic control settings. This directly defeats novelty based only on matched labels/different histories. Their core failure is reliance on misleading correlates; ours hypothesizes missing decision-relevant information. Do not claim we have identified the model's causal graph. |

## Exact experiment we can defend

The [fixed-cutoff finding](finding_textcraft_breadth_20260928.md) compares quantity-corrected
known-recipe and discovery supervision: all32 per-task action/string/label-token multisets match,
including8,820 answer tokens, but public query-name availability differs sharply. This is a
teacher-package association. An absent name is not automatically information-theoretically
unpredictable: a fixed training world permits memorization. The existing
[paired-world CPU diagnostic](TEXTCRAFT-OBSERVATIONAL-AMBIGUITY.md) establishes ambiguity under
its matched-start construction, not a new general impossibility theorem or trained-policy result.

The queued [stable/random-visible controls](teaching_order_20260928/README.md) are stronger:
original craft order and literal action labels stay fixed; native histories are regenerated;
training rows are mapped back to original action indices, preserving each shuffled minibatch's
targets and token denominator. Stable/random legal query schedules test whether one special
ordering drives the effect. No reviewed method reports this exact control, but four papers do
not establish global priority.

Remaining differences include history content/recency, prompt length and action position within
the native trajectory. Visibility certifies available query names, not that every gold decision
is inferable or optimal. The scheduler still uses the oracle action list; it is not a public-only
online planner. Existing undertraining/teacher-forced-fit controls remain relevant. A positive
result would show that useful supervision can be repaired without more answer labels, not that
we isolated a unique internal mechanism.

## Only two narrow follow-ups

1. **Reindex discovery without changing its prompts.** Match the known-recipe control's action
   identities, row presentation and every optimizer batch, retaining discovery's existing native
   histories. Use the same initialization/seed,23 updates and fixed checkpoint23; compare on the
   already declared raw world42/50 cells. This is an existing proposed control, not another new
   research branch. Estimate oneA100: approximately3.5min training plus30–50min readout. If its
   advantage vanishes with histories untouched, optimizer presentation order materially explains
   the package gap. Persistence plus successful stable/random repair strengthens the narrower
   history-conditioning interpretation; neither outcome isolates prompt-token dose.

2. **Evidence-necessity scoring, without another training run.** Freeze eligible query prefixes
   from public native traces on disjoint official TRAIN diagnostic goals: the target item must
   first have become visible through earlier native recipe feedback. Score identical target
   tokens under the original prefix, an equal-token mask of that revealing evidence, and a
   position/length-matched mask of irrelevant feedback. Use fixed known/discovery/repaired
   checkpoints; include all prospectively eligible prefixes. Estimate oneA100:10–30min scoring,
   no rollouts or optimizer. Selective sensitivity in repaired models supports actual use of
   public evidence; equal sensitivity to irrelevant masks or no relevance effect weakens that
   mechanism. These are explicitly counterfactual model-input probes, not native-valid trajectories,
   task-success estimates or a substitute for the queued transfer readouts.

## What defeats the proposed contribution?

Historical method novelty fails if earlier work already performs the same label/minibatch-preserving,
native-executable information-order repair; this review does not rule that out. The empirical claim
fails if benefit disappears under matched target presentation, adequate-fit controls or a second
training seed; improved first-query grounding without task success is insufficient. If only one
special schedule works, claim a schedule-specific result. Without transfer beyond the current
generator/grammar, describe a useful controlled TextCraft case study—not a general solution to
partially observable imitation. Reproducing the known failure mode is valuable evidence, but is
not by itself a novel learning method.
