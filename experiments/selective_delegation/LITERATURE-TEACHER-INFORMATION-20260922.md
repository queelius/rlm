---
status: literature_informed_proposal
retrieved_utc: 2026-09-22
related_question: public_information_prerequisite_discovery
gpu_experiment_accepted: false
---

# Teach a procedure the student can actually follow

The [training coverage audit](TEXTCRAFT-TRAINING-COVERAGE.md) found that most
teacher trajectories begin with a prerequisite chosen from a hidden plan.
The [reminder control](TEXTCRAFT-INSTRUCTION-FINDINGS.md) separately shows that
valid action formatting does not guarantee successful execution or stopping.
These observations motivate examining demonstrations, not immediately adding
another planning layer. The proposed [public-discovery comparison](PUBLIC-DISCOVERY-QUESTION.md)
is conditional on the pending trained-policy evaluation.

## Directly relevant prior art

**LEAP, ICLR 2025.** The method gathers student trajectories and obtains expert
corrections using information available only during training. It explicitly
constrains feedback to decisions the student can learn from its own observations.
Section 3.1 discusses this tradeoff; Appendix C.1 distinguishes successful-only
demonstrations from corrections at states reached by the student. It reports
that the latter help recovery from errors. Thus neither constrained expert
feedback nor adding failure recovery is a new general idea for language agents.
[Primary conference paper, §§3.1–3.2 and Appendix C.1](https://proceedings.iclr.cc/paper_files/paper/2025/file/1c60ed2b01120d383eebf12dc7a0e138-Paper-Conference.pdf)

**PACT, arXiv:2606.16215v1, June 15, 2026.** It generates rollouts without expert
hints, then uses expert traces during optimization: a trace-conditioned RL
surrogate, supervised losses on selected reasoning and tool-call tokens, and
some ordinary prompt-only RL updates. Environment observations are context,
not supervised targets. This is a different intervention from changing our
demonstration order. The paper's separation of rollout inputs from optimization
inputs is useful for our provenance records. Its full recipe should not be
treated as a single interchangeable reward change.
[Primary paper, §§4.1–4.4](https://arxiv.org/html/2606.16215v1)

## Our inference and next decision

Our narrow question is whether a student benefits when demonstrations explicitly
acquire prerequisites before using them. This is not a claim that privileged
teachers are always harmful: in a fixed recipe world, their actions may teach
useful associations. Nor do our completed traces establish why SFT will succeed
or fail; that evaluation is still pending.

The smallest informative follow-up would retain the same training tasks and
model while changing how demonstrations discover their plan. Report changed
action and token exposure; do not label unlike trajectories compute-matched.
A later changed-recipe or renamed-item test could distinguish discovery from
remembered associations. Recovery examples are a separate intervention and
should not be mixed into the first comparison.

If the current trained model solves the task reliably, prioritize fresh goals
and changed-world transfer before repairing its teacher. If it still fails to
discover prerequisites, qualify a public-query teacher on CPUs, then consider
one short SFT comparison on the single A100. Keep the existing reward-control
and trained-policy readouts ahead of that proposal. Generic imitation-learning
ideas alone are not a publication contribution; we need a reproducible effect,
a meaningful contrast, and evidence beyond the motivating traces.

Both primary sources were inspected online; no third-party implementation was
downloaded or executed. LEAP's cited sections and PACT's method were read;
we have not reproduced either paper's results.
