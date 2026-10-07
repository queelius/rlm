# What might we be missing about training RLMs?

Research discussion, 7 October 2026. These are hypotheses and proposed comparisons,
not new experimental results. Our [overview](START-HERE.md) contains the measured
math results; the [earlier RLM audit](../research-audit-2026-09-29/teaching-and-execution.md)
records the tool-training successes and failures we should build on.

We do not need one research question yet. We need a set of inexpensive experiments
that distinguish plausible explanations. The organizing idea is: **find which part
of useful behavior is missing, make it learnable, and test whether it transfers.**

## 1. A learning signal is necessary—but “some right, some wrong” is not the whole story

For our binary group-relative reward, all eight wrong answers have the same
advantage: zero. So do eight correct answers. Mixed outcomes supply a contrast
within that question. A batch containing perfectly solved easy questions and
completely failed hard questions can average 50% while providing no such contrast.

This is a property of this objective, not all reinforcement learning. A different
baseline, intermediate rewards, other training questions, or auxiliary losses can
still change the policy. Also, a mixed group is not necessarily useful: the positive
answer might be a lucky guess, or the checker might reward an irrelevant shortcut.

The missing measurements are therefore not just accuracy. Record:

- Did the action parse and actually execute?
- Did the model use the returned information correctly?
- Did the complete task succeed?
- Which questions produced all-failure, mixed, or all-success groups?
- How much generation cost produced each verified useful trajectory?

[DAPO](https://arxiv.org/html/2503.14476v1), section 3.2, filters uniform-reward
groups. That motivates selecting informative practice, but does not prove a
particular easy-to-hard sequence is best.

## 2. We should diagnose skills, not assume they are absent

A model may know Python but not our function names. It may know the function names
but fail to provide a helper with enough context. It may execute two correct calls
but fail to connect them. It may connect them yet choose an unnecessary plan.
These are different problems.

Our previous RLM work already taught executable routines. More SFT is not a blank
slate solution. The audit also found that changing which information was visible
before an action changed learnability. That is **information availability within a
trajectory**, not the same intervention as ordering easy examples before hard ones.
An impossible-to-observe answer cannot be repaired simply by a better curriculum.

Our math prompt comparison is another warning: the same base weights solved
154/500 with chat instructions and 305/500 with the question alone. Some apparent
training gain can be better access to existing ability. This does not make the gain
unimportant; it changes what we should say the gain demonstrates.

## 3. Individual tool competence is not composition

An invented example makes this concrete. A helper returns the employer **Aster Lab**.
The next lookup must ask for Aster Lab's city. A valid call asking about a memorized
employer, **Birch Lab**, can execute perfectly and still be wrong.

A useful diagnostic changes the first returned employer and checks whether the
next query changes with it. That is stronger evidence of information passing than
counting valid calls or accepting an accidentally correct final city. Use newly
specified diagnostic tasks; our old diagnostic-B outcomes remain excluded from
training and favorable checkpoint selection.

[Toolformer](https://arxiv.org/html/2302.04761v1), section 7, explicitly reports this
kind of boundary: independently generated API-call demonstrations did not teach
chained tool use. The implication is not “tools fail,” but “teach and test the
dependency that matters.”

## 4. Curriculum is more than increasing difficulty

We have several independent choices:

1. **Content:** which missing skills receive examples?
2. **Order:** are the same examples staged or shuffled?
3. **Selection:** which tasks receive more practice as the model changes?
4. **Assistance:** how much of the plan, context, or partial solution is supplied?
5. **Retention:** how often are earlier skills revisited?

Difficulty itself has several axes: document length, distractors, number of calls,
dependency depth, unfamiliar tools, uncertainty, and the difficulty of each leaf
subproblem. A longer document is not necessarily a harder planning problem.

The child-learning analogy suggests experiments; it is not evidence for the right
sequence. [E2H Reasoner](https://arxiv.org/html/2506.06632v3) reports that scheduling
and fading easy tasks matter. [Self-Evolving Curriculum](https://arxiv.org/html/2505.14970v4)
instead adapts category selection using an advantage-based proxy. Reward variation
is not the same measurement as actual improvement on new tasks. Compare schedules
against shuffled practice, rather than assuming ordering is the active ingredient.

## 5. More attempts and higher temperature solve only some exploration problems

If independent attempts succeed with probability p, the chance of at least one
success in k attempts is `1 − (1 − p)^k`. At p = 1%, eight attempts give about 7.7%;
32 give about 27.5%. More attempts can expose rare successes. Observing none does
not prove that the probability is zero.

But raising temperature changes the success probability as well as diversity.
It may produce new plans—or broken syntax and nonsense. [STaR](https://arxiv.org/html/2203.14465v2),
section 5, found that higher-temperature sampling could retain correct answers
with bad reasoning and harm subsequent generalization. That is a warning about
quality filtering, not a universal argument against temperature tuning.

For RLMs, distinguish diversity of **plans** from diversity of wording. Try several
valid decomposition plans rather than merely making each token less predictable.
Compare attempts per question at the same total token/tool budget: doubling k
otherwise buys extra compute while reducing the number of distinct questions seen.
Keep test-time attempts fixed when claiming a better trained model.

## 6. Teach recovery, and ask who deserves the credit

Perfect demonstrations show what to do along a successful path. A deployed model
also encounters missing keys, empty searches, ambiguous helper answers, and mistakes
of its own. It may need examples of noticing and repairing those failures. This is
an extension of our previous recovery experiments, not a newly discovered topic.
Keep the error visible; do not silently correct calls in the runtime and then
attribute that success to the model.

There is also a credit problem. Suppose the parent chooses the right subproblem,
but the helper answers incorrectly. A zero final reward can discourage the parent
as well. Conversely, a capable helper can rescue a poor request. First freeze the
helper and measure its reliability; later test joint training separately.

The [alphaXiv RLM training report](https://www.alphaxiv.org/blog/reinforcement-learning-for-rlms)
trains parent and child rollouts using shared outcome advantages and identifies
finer credit assignment as future work. Its reduced-strategy-prompt experiment
also became less stable. Thus scaffolding removal and reward assignment are live
questions, not solved ingredients. It is an experimental report, not our replication.

Two further checks matter. First, the best teacher is not necessarily the one whose
demonstrations this student learns from most effectively. A
[student–trajectory study](https://arxiv.org/html/2601.14249v5) finds that teacher
strength alone does not predict downstream student performance. Its evidence is
mainly mathematical; applying this to RLM traces is a hypothesis. Compare whether
the student can continue a demonstrated trajectory, not just whether the teacher's
final answer is correct.

Second, [SCoRe](https://arxiv.org/html/2409.12917v1) shows why correction data must
match the errors the student makes. Its math/code results do not establish tool
recovery, but suggest an important control: compare correction with an independent
second attempt at the same total cost, and count right-to-wrong changes too.

For finer credit, [GiGPO](https://arxiv.org/html/2505.10978v3) groups repeated
decision states to obtain step-relative signals without extra reward labels.
Before importing it, check whether our RLM trajectories actually revisit equivalent
states. Identical printed observations can conceal different Python memory or
remaining budgets. Extra intermediate reward labels and a better estimator of the
same terminal reward are different interventions.

## 7. Do not turn “RL amplifies existing skill” into an absolute ceiling

That is a useful working description of some outcomes, not a theorem that RL can
never learn a new combination. A model can improve on easier related tasks, changing
which harder trajectories it can reach. A tool environment also supplies observations
that a standalone text generator never receives.

The literature disagrees about capability expansion.
[Yue et al.](https://arxiv.org/html/2504.13837v5) find limited large-sample coverage
gains in their settings; [Wen et al.](https://arxiv.org/html/2506.14245v2) find
counterevidence using reasoning-sensitive evaluation, with its own judge limitations.
Finite pass@k under one prompt and sampler cannot establish an absolute capability
boundary. Our practical question is whether learning improves useful behavior on
new problems at a fixed affordable budget.

Likewise, “SFT then RL” is not by itself a novel RLM contribution. The
[RLM paper, version 3](https://arxiv.org/html/2512.24601v3), already includes training.
The opportunity is to identify a reproducible bottleneck, show which intervention
addresses it, and map where the benefit does and does not transfer.

## A portfolio of small experiments

All rows below are **proposals**, not ready jobs or promised improvements.

| Possible bottleneck | Small discriminating comparison | What would change our mind? |
| --- | --- | --- |
| Poor access to existing skills | Same weights: clear prompt and examples versus current prompt | If prompting fixes it, do not initially pay for extra training. |
| Protocol versus dependency skill | Matched-size independent-call versus linked-call SFT; retain both pre-RL controls | More valid calls without better information passing is not composition. |
| Teaching order | Identical demonstrations staged versus shuffled; same subsequent RL | A tie or shuffle win weakens the claimed need for staging. |
| Wrong practice distribution | Uniform tasks versus mixed-outcome selection versus a simple staged schedule | Compare held-out gain per generation cost, not just accepted updates. |
| Inadequate exploration | Attempts per question and plan diversity at equal total budget | If wider sampling already recovers good trajectories cheaply, more SFT may be unnecessary. |
| Brittle success-only training | Matched successful-only versus error-and-repair teaching | Test new errors; canned correction of the demonstrated error is insufficient. |
| Dependence on hints | Constant hints versus gradually removed hints, same tasks | Measure final unassisted performance, not just easier assisted training. |
| Parent/helper credit confusion | Fixed parent with controlled helper reliability; then fixed helper with parent RL | Determine which participant fails before jointly updating both. |
| Checker shortcuts | New counterfactual task instances and independent output checks | If reported success vanishes when irrelevant cues change, revise the reward. |
| RLM overhead without value | Direct model, simple program, fixed decomposition and learned RLM at matched cost | If a simpler method solves the task, restrict our claim or change the task. |

For transfer, separate familiar structures, longer familiar structures, and new
dependency structures matched for size. Include tasks where no delegation is needed.
Useful recursion is a decision, not a requirement to maximize calls.

A frozen helper must really be a fixed copy: excluding child tokens from the loss
does not freeze the helper if it shares weights that parent training changes.
Likewise, the math experiment does not validate how an RLM loss weights roots,
children, turns and tokens. A tiny traced example should verify that those weights
match the intended objective before a large recursive training run.

Start with cheap diagnostic and sampling screens. Promote informative differences
to small matched training comparisons, then independently repeat promising effects.
Use shared tasks and metrics across branches so breadth remains interpretable. Small
samples are fine for deciding what to investigate; they do not turn the best result
among many trials into a confirmed effect. Preserve all arms and maintain a separate
confirmation set when choosing a candidate.

## What is happening now

The ongoing math run and its fixed final evaluations are unchanged. The reviewed
512-question math control/learning-rate/rollout-reuse screen remains available after
the entire current owner releases the GPU; it has not launched. A meaningful final
math gain can instead prioritize a fresh training repeat. The RLM ideas above need
task manifests, a selected existing controller/harness, explicit compute caps and
their own run owners before launch. Literature and CPU preparation do not displace
the active GPU experiment.
