# Publication decisions: repair teaching histories, then test interface learning

September 28, 2026. Decision memo, not a novelty certification. The original assessment below
uses completed panels 00–05. This amendment adds completed intervention results through 15:50 UTC;
prepared follow-ups are not completed results.

## 15:50 UTC amendment: two positive signals warrant targeted replication

**Changing the teaching history improved completion while retaining the supervised answers.**
On the first eight validation goals, two recipe worlds and two rollout seeds, the original
known-recipe teacher produced 1/32 successful attempts. Stable visible-query ordering produced
14/32, and randomized visible-query ordering produced 12/32. The discovery reference also
produced 14/32. These are repeated measurements of eight goals, not 32 independent problems.
Both repairs retain all 23 optimizer minibatches' target-token sequences and normalization;
their input histories, input-token counts and gradients differ. This supports practical repair
at this setting, not superiority to discovery or an isolated explanation for the benefit.

The next prepared comparison repeats both repairs at a second training seed and evaluates the
existing repaired models on eight additional goals. It does **not** include the
discovery-reindexed presentation control proposed below; that mechanism question remains open.
The GPU is currently testing longer training for the original teachers. If longer training
closes the gap, the claim becomes faster learning rather than an unusable teaching strategy.

**One RL update improved completion on familiar goals.** With ordinary execution, the
raw-trained model improved from 9/16 to 13/16 attempts: four paired wins and no losses.
With ingredient-binding assistance, the binder-trained model improved from 14/16 to 16/16.
These eight goals were used in RL training; only rollout seeds were held out. Calls and output
tokens increased, and schema errors persisted. This is a promising learning signal, not yet
transfer or greater efficiency. A prepared four-cell evaluation tests these fixed checkpoints
on the already frozen diagnostic goals; the separate fresh-goal RL campaign tests new learning.

**Shorter actions reduced output tokens, but did not reliably improve success.** Compact actions
used about 39% fewer generated tokens in each world; success was 8 versus 10 and 10 versus 9
out of 16, with slightly more calls. Their legality rules and SFT target-token dose also differ.
Keep this as an interface-efficiency observation, not an RL result.

Finally, all 48 scripted continuations in the counterfactual-label diagnostic completed the
task, despite conflicting privileged next-query labels. Label ambiguity alone is therefore
not evidence that those choices cause failure. The actual ordering intervention is the more
useful signal to pursue. See [the live findings](FINDINGS-20260928-LIVE.md),
[the RL outcome analysis](rl_outcomes_20260928/README.md), and
[the counterfactual diagnostic](teacher_counterfactual_20260928/README.md).

## Strongest claim at the original six-panel cutoff

**Identical supervised answer content can accompany very different tool-agent transfer,
and deterministic execution assistance helps—but does not close—the gap.**
The [six-panel study](finding_textcraft_breadth_20260928.md) finds discovery versus
quantity-corrected known-recipe success of 323/768 versus 39/768 without assistance, and 381 versus 48
with observed-recipe ingredient binding. The raw teacher gap is 36.98 percentage points
(task-cluster 95% interval 32.94–40.89); binding adds 7.55 points for discovery (5.08–10.16).
Both teachers contain identical per-task action/string/label-token multisets: 8,820 target
tokens. Publicly available names support 167/167 discovery queries but only 32/167 known-teacher
queries. That is a compelling conditioning-history hypothesis, not isolated causation.

There are 48 task identities, four related recipe worlds and two fitted seeds—not 768 independent
problems. At that cutoff, the evidence did not establish teacher-order repair, learned interface
co-adaptation, recursion or an RL improvement. Post-cutoff panel 06 remains separate.

## Narrow opportunity and prior-art boundary

The [methods review](TEACHER-OBSERVABILITY-METHODS-REVIEW-20260928.md) already identifies close
precedents: Student-Informed Teacher Training, Guided Policy Optimization and Privileged
Information Distillation address teacher–student information mismatch; Causal Confusion already
varies observation conditioning while retaining demonstrations. Do not sell “privileged teachers
can fail” or “same labels, different observations” as discoveries.

Action grounding is also established: [Huang et al., §§3.2–3.3](https://proceedings.mlr.press/v162/huang22a/huang22a.pdf)
translate outputs into admissible actions and condition subsequent generation on them.
[Xue et al., §§2–3](https://arxiv.org/html/2406.01026v2) study symbol/content supervision and
loss weighting. Neither ingredient binding nor index-versus-command sensitivity is a new general
principle. [ALFWorld](https://arxiv.org/abs/2010.03768) already separates abstract language
policies from embodied actuation; our flat experiments cannot establish a hierarchical-RL contribution.
Additional primary reading here was limited to those Huang/Xue method sections and the ALFWorld
abstract; the earlier privileged-imitation assessment comes from the linked methods review.

The candidate contribution is **an executable offline trace repair with unusually strong
training controls**: reorder existing queries, retain original craft order, regenerate native
histories, then restore original action-index row order. Every optimizer minibatch retains its
target-token sequence and normalization—not its gradients, input tokens or FLOPs.
Stable and random legal schedules test dependence on one ordering.

Practical value would be recovering useful demonstrations without collecting additional answer
labels. Applicability is restricted: the gold action list is privileged; needed queries must
already exist; reordered queries must replay safely. Visibility does not make every decision
inferable. This is neither an online public planner nor a remedy for missing exploration actions.
The reviewed methods do not establish priority for this exact control, and missing citations
do not establish novelty. Until transfer succeeds, position it as a reproducible intervention
and diagnostic—not a general new learning algorithm.

## What the queued/prepared contrasts can falsify

| Contrast | Decision it can change; what it cannot establish |
|---|---|
| [Teaching-order repair](teaching_order_20260928/README.md) | Stable/random visible schedules versus matched known-teacher training test whether changed histories improve native completion with optimizer targets fixed. Grounding without completion refutes the proposed practical repair at this recipe; one successful schedule supports only schedule-specific utility. Prompt length/recency remain coupled. |
| [Dose and fit](teaching_dose_20260928/README.md) | Predetermined total 46/69-update endpoints test whether the gap survives additional fitting. Known-teacher catch-up revises the story toward learning efficiency, not necessarily unusable supervision. Lower teacher-forced NLL without rollout gains weakens simple undertraining; neither result alone identifies an internal mechanism. |
| [Phi](phi_transfer_20260928/README.md) | A missing/reversed original-teacher gap limits family-independent package claims. Replication supports two-family association, **not repair transfer**, because these jobs do not train repaired Phi demonstrations. Tokenization, LoRA capacity and compute differ across families. |
| [Compact RL](rl_compact_20260928/README.md) | Compare each interface's post-update minus unchanged-weight success, then their gain difference. A better compact endpoint without a larger own-interface gain is not better RL. Constant-reward/skipped updates are uninformative, not negative learning evidence. Legality rules, warm actors and token doses differ. |
| [ALFWorld representation](alfworld_representation_20260928/READINESS.md) | Command-versus-index SFT gain, each relative to its own base-interface control, tests representation-dependent learning on newly selected valid_unseen games. A command-base benefit alone is inference assistance. Longer targets change token exposure and token-mean example weights; a positive interaction does not isolate symbol binding. |

These controls could support complementary teaching-history and interface-learning results.
They do not yet justify one causal story explaining both.

## At most three next experiments, conditional on results

1. **Matched-presentation replication, if repair improves completion.** Compare known,
   stable-repaired and discovery-reindexed-to-known-target-order training on a frozen unread
   panel and second training seed. Discovery prompts stay unchanged. This tests reproducibility
   while separating its historical target-presentation advantage from conditioning histories.
2. **Repair transfer, only if that result survives dose controls.** Apply the same stable,
   target-preserving repair to Phi; retain fixed endpoints and prospectively selected inputs.
   This directly tests the method claim that the queued original-teacher Phi comparison cannot.
3. **Learning-interaction replication, only after positive compact-RL or ALFWorld interaction.**
   Replicate the relevant own-interface learning gains on fresh diagnostic goals. Match native
   legality where possible and report success against actual token/GPU budgets, not merely update
   count. Without a reproducible learning interaction, keep execution assistance as a separate result.

**Early abandonment:** if neither repair beats its matched known-teacher baseline in either
fixed world despite learning visible queries, stop expanding this repair recipe.
That is a resource-allocation rule, not proof of a universal null. Likewise, fewer errors or
shorter outputs without improved goal completion cannot promote an interface-learning claim.

Publication recommendation: prioritize a tightly controlled teaching-history case study with
audited data transformation and replication. Add co-adaptation only if its own evidence matures;
do not broaden the title into a general RLM or hierarchical-RL method to cover queued work.
