---
date: 2026-09-28
status: primary_source_review_and_proposed_experiments
literature_cutoff_utc: "2026-09-28T10:07:53Z"
local_evidence: RESEARCH-RESTART-20260928.md
questions: [TC-EXECUTION-AND-RL, TC-ADAPTIVE-DECOMPOSITION, TC-TEACHING-MECHANISM]
recommended_next: OP28-1
gpu_jobs_launched_by_this_review: 0
assets_downloaded_or_installed: none
sources:
  Harness-RL: {arxiv: "2608.29641", version: v1, submitted: 2026-08-30}
  Harness-1: {arxiv: "2606.02373", version: v1, submitted: 2026-06-01}
  PaW: {arxiv: "2606.02388", version: v2, submitted: 2026-06-01, revised: 2026-09-14}
  RAO: {arxiv: "2605.06639", version: v1, submitted: 2026-05-07}
  LATTE: {arxiv: "2605.06320", version: v1, submitted: 2026-05-07}
  distributed_teams: {arxiv: "2603.12229", version: v1, submitted: 2026-03-12}
  GACA: {arxiv: "2609.12424", version: v1, submitted: 2026-09-11}
  SRLM: {arxiv: "2603.15653", version: v1, submitted: 2026-03-07}
  NSI: {arxiv: "2605.01293", version: v1, submitted: 2026-05-02}
---

# Opportunities after the broader TextCraft result

The most immediately useful experiment is whether **training with ingredient
assistance improves decisions beyond assistance at evaluation**. Its novelty is
modest by itself. The stronger prospective contribution is a measured boundary:
which execution and information contracts make decomposition learnable, and when
shared resources make isolated subgoals misleading?

The [six-panel reconciliation](RESEARCH-RESTART-20260928.md) gives discovery-teacher
successes of 323/768 without binding and 381/768 with binding, versus 39/768 and
48/768 for the quantity-corrected known-recipe teacher. Thus executing recipe
details does not erase the teaching gap. The earlier
[memory and RL controls](MORNING-UPDATE-20260924.md) establish neither reliable
notebook gains nor meaningful RL improvement. All these actors are flat.

The new depth breakdown supplied with the reconciliation is particularly useful:
discovery successes rise 11→37/288 at depth 5, but 177→181/192 at depth 2. This
is an exploratory difficulty interaction, with a shallow-task ceiling; it does
not identify recursion, resource competition, or a causal mechanism. The
changed-world constructor intentionally supplies conservative sufficient inventory
and ignores batch-yield division when deriving base quantities. Existing depth
failures therefore do not establish scarcity.

## What the primary papers actually add

- **Harness-RL** separates action selection and argument generation by routing
  their gradients through activation-derived parameter subsets. It also preserves
  branching call contexts during training. Its conclusion asks for more diverse
  harnesses and an account of partition robustness/cost. It uses both outcome and
  process signals; reported training uses four or eight A100 80 GB GPUs. Our
  one-card proposal below is not a replication of CAPO.
  [Methods, conclusion, compute](https://arxiv.org/html/2608.29641v1).
- **Harness-1** externalizes recoverable search state while leaving search,
  curation, and stopping choices to the policy. Its limitations include engineered
  extraction/compression errors, retrieval-specific evaluation, and limited
  repeated-run uncertainty. Better external bookkeeping plus RL is already an
  explicit research program. [Methods and Appendix B](https://arxiv.org/html/2606.02373v1).
- **PaW v2**, revised September 14, trains the same policy to predict next
  observations from its own RL transitions. It selects high-action-surprise
  transitions, uses a noise-tolerant auxiliary loss, and changes no inference
  interface. Section 7 identifies one-step supervision and untested longer-horizon
  dependencies as limitations. [Methods and limitations](https://arxiv.org/html/2606.02388v2).
- **RAO** is unusually close: Qwen3-4B, TextCraft-Synth, recursive RL, local child
  rewards, and transfer to deeper tasks are already demonstrated there. Section 7
  proposes cheaper training surrogates, cross-domain delegation, and heterogeneous
  subtasks. **LATTE** already maintains dynamic dependency graphs and explicitly
  proposes learning coordination through fine-tuning/RL; its limitations include
  graph overhead on simple tasks and naturally delineated subtasks. These sharply
  limit claims based only on adding recursion or a task graph.
  [RAO §§2–7](https://arxiv.org/html/2605.06639v1),
  [LATTE §§3,6](https://arxiv.org/html/2605.06320v1).
- **GACA** already connects action surprise to adaptive credit. Its argument
  explicitly discounts multiple textual realizations of one canonical action;
  its conclusion acknowledges noisy, action-dependent weighting and leaves
  estimator learning dynamics open. **SRLM** supplies a necessary recursion
  control: extra program search/reflection can help without recursive calls;
  its future work includes earlier stopping and compute allocation. Neither
  result implies that token surprise predicts useful delegation in our game.
  [GACA §3.3, §5, Appendix B](https://arxiv.org/html/2609.12424v1),
  [SRLM §§2,4](https://arxiv.org/html/2603.15653v1).

## Ranked experiments

Ranking combines readiness, likely information gain, and potential contribution.
Durations are planning estimates for one A100 40 GB and the existing 4B/LoRA
stack, not measured promises. Each requires a new immutable run receipt; this
review does not change any live owner.

### 1. OP28-1 — Does reliable execution change what terminal-reward RL learns?

**Question.** Binding repairs arguments but leaves discovery, targets, quantities,
and order with the model. Does that make a short RL update improve those remaining
decisions, or does binding merely raise the frozen policy's score?

**Small comparison.** Start both arms from one identical discovery checkpoint.
Use eight fixed TRAIN tasks, four rollout seeds each, one existing RL update,
and raw versus observed-recipe binding during collection. Keep terminal reward,
optimizer, task schedule, and limits fixed. Evaluate warm start, raw-trained RL,
binder-trained RL, and extra SFT on the same prospective eight-goal/two-seed
panel under both execution interfaces: 128 evaluation attempts. Match updates
and declared rollout caps; report actual tokens and successful-example exposure
because these will differ. Estimated 4–6 hours; cap at six hours, saving every
episode and the post-update adapters. If a four-hour allocation fragment is all
that remains, first evaluate all four policies under binding, label the result a
training-interface pilot, and defer the crossed raw readout explicitly.

**Important loss contract.** Optimize the sampled *requested* action tokens with
their recorded behavior log-probabilities. The repaired ingredient dictionary is
an environment transformation, not a policy sample. The next prompt correctly
contains executed history. Do not train repaired strings as though sampled.

**Decision.** Improvement of binder-trained weights under a common evaluation
interface is evidence about learning; a frozen binder gain is not. Promote only
if RL beats warm start and extra SFT, then repeat a training seed and fresh recipe
graph. If gains disappear on raw deployment, report interface specialization.
If extra SFT matches RL or neither update helps, retire an RL-specific claim at
this dose. A negative one-update pilot cannot rule out longer training.

**Novelty: low to medium.** This differs from Harness-RL's parameter routing, but
“offload routine work, then train the policy” substantially overlaps Harness-1 and
existing EvoHarness-RL notes. A positive factorial interaction alone is probably
incremental. A stronger result needs a measured mediator—fewer argument-caused
failures and improved remaining decisions—and transfer, not another accuracy grid.

### 2. OP28-2 — Learn when a recursive boundary needs shared resource commitments

**Question.** At the same depth, does delegation help on independent recipe
branches but hurt when sibling goals consume the same resources? Can a policy
learn this boundary from discovered dependencies?

**Small comparison.** Prepare 12 outcome-blind recipe-graph pairs on CPUs. Pair
low/high sibling overlap at fixed depth, matching recipe/batch-size distributions
and native constructive action count where possible; disclose unmatched graph
size or query burden. Separately vary inventory slack on the *same* graph, keeping
both inventories natively solvable. This separates topology from scarcity.
Compare flat binder, one fixed child boundary, and that same boundary with a
public ledger of parent/sibling quantity commitments. All arms receive the same
observed recipes, current inventory, global goal, and total budget; the ledger
reorganizes public requirements, never reveals undiscovered dependencies. Run
two seeds, at most 144 initial attempts, roughly 3–4 hours; cap at four hours.

Use sequential children on the single GPU for the first test. Preserve native
“produce in addition to inventory at child start” semantics. Record root success,
child success, material consumed, redundant production, calls and token cost.
An offline exact solver may label whether a child's outcome leaves the root
solvable; that is a disclosed diagnostic oracle, never a policy observation.
Stratification on existing graphs is preliminary association, not a causal
topology experiment.

**Decision.** First require local skills to work on isolated prerequisites. If
fixed delegation has no conditional advantage, stop before training a router.
Otherwise collect split/continue contrasts on TRAIN states, fit a small selector
or one bounded RL update, and compare with always-flat, fixed-depth, and a simple
public-overlap rule on unseen graphs. Retire the learned-selector claim if a fixed
rule matches it. Locally successful children that reduce root solvability would
motivate comparing root-aware credit with RAO-style local success.

**Novelty: medium, potentially stronger than OP28-1.** The opportunity is a
controlled topology/resource boundary for learned decomposition, not recursion,
shared ledgers, or dependency discovery themselves. A distributed-teams paper
explicitly leaves dynamically discovered dependencies open, while LATTE already
addresses dynamic coordination. [Distributed teams §5.1](https://arxiv.org/html/2603.12229v1).

### 3. OP28-3 — Predict only consequences the policy can infer from public state

**Question.** Does auxiliary transition learning help inventory planning while
predicting undiscovered recipe answers encourages world-specific memorization?
This links the discovery-teacher result to PaW without treating an entire tool
observation as equally learnable.

**Small comparison.** Reuse a fresh fixed TRAIN rollout batch and its actual
feedback. Compare terminal RL alone, RL plus PaW-style observation prediction,
and RL plus prediction restricted to inventory changes/validity after recipes
have been observed; add matched extra SFT. The restricted targets must be
mechanically derivable from the public prefix. Full-feedback prediction may use
observed next answers as training labels, but never insert them into preceding
policy inputs. Use the same RL samples and two small updates, disclose auxiliary
token dose, then 16 prospective changed-world attempts per arm. Estimated 2–4
hours; cap at four, checkpoint after each update.

**Decision.** Measure native success, quantity errors, query-before-craft behavior,
and changed-world transition accuracy. Retire if better observation likelihood
does not improve policy behavior, or if extra SFT matches the improvement.
Only train a multi-step predictor after a one-step behavioral signal.

**Novelty: medium.** PaW already handles unpredictable observation noise. The
specific test is *epistemic availability* under recipe interventions, not the
invention of auxiliary world modeling. A positive result on one fixed world is
insufficient to support it.

### 4. OP28-4 — Measure uncertainty after the harness determines the executed action

**Question.** When binding overwrites ingredient fields, how much model output
variability still affects the task? Does token surprise mostly describe different
targets/quantities, or alternative ingredient text that the harness discards?

**CPU diagnostic first.** From saved requested/executed receipts reconstruct the
binder using only prior public `get_info` results. Count craft calls eligible for
binding, changed versus unchanged ingredient maps, raw ingredient token share,
and unique requested maps for each `(public recipe state, target, output_count)`.
Report diversity separately for action type, target, quantity, and ingredients;
do not infer entropy from isolated one-sample states. In exact repeated-prompt
groups, compare raw-action diversity with executed-action diversity and partition
NLL only when aligned rollout token probabilities are actually saved. Missing
log-probabilities are unavailable evidence, not zero surprise.

An inventory-effect signature is only a diagnostic: differing notes, prompts or
remaining token budgets can still make the subsequent policy states different.
Exact credit groups must preserve those differences. Do not merge states using
hidden recipes or inventory alone. Existing repeated-state coverage is sparse;
the [earlier feasibility audit](TEXTCRAFT-REPEATED-STATE-CREDIT-FEASIBILITY.md)
already documents that limitation.

Binding induces a distribution over executed actions by mapping sampled raw
actions through the harness. Full raw-token policy gradients remain the baseline
for this fixed mapping; simply masking ingredient-token gradients is generally
not equivalent. Ingredient text can condition later target tokens, affect parse
validity, and consume the token budget. Exact executed-action probabilities would
require summing raw preimages. A compact schema explicitly changes the policy
interface instead; this distinction is a method choice, not a new theorem.

**Small GPU comparison if the CPU audit finds substantial discarded output.**
Take 24 fixed TRAIN states and four one-step samples each, map them through the
public binder, then sample bounded continuations for representative execution
classes with two new seeds. Compare target/quantity diversity, token NLL, and
actual continuation-outcome spread. Cap at 1,536 calls, 128 output tokens/call,
and 90 minutes; missing continuations remain explicit. A subsequent compact
craft interface can omit ingredient generation only where a public recipe is
bindable, with raw crafting preserved elsewhere; compare equal adaptation against
the current post-generation binder. This tests learned decisions separately from
output-token savings.

**Decision.** Retire an uncertainty/credit claim if canonicalization does not
improve prediction of decision value. Retain a compact interface only if it saves
measured work or improves success after matched adaptation. GACA already names
textual multiplicity; the potential contribution is how a changing execution
contract changes relevant uncertainty. Likewise, NSI already uses state-grounded
variable binding in TextCraft, so compact commands are not novel by themselves.
[NSI §§4–6](https://arxiv.org/html/2605.01293v1).

## Provenance and scope

Metadata was checked on versioned arXiv pages on September 28. In particular,
[PaW's history](https://arxiv.org/abs/2606.02388v2) gives June 1 submission and
September 14 revision; [GACA](https://arxiv.org/abs/2609.12424v1) gives September
11, and [SRLM](https://arxiv.org/abs/2603.15653v1) gives March 7. Crawler-relative
ages were not used as publication dates. This is a targeted search through today's
available primary sources, not an exhaustive novelty or newest-paper claim.
The previously noted date inconsistency for arXiv:2609.20831 remains unresolved;
none of these proposals depends on assigning it a new publication date.

Official repository READMEs were inspected online, and remote HEADs were resolved
read-only with `git ls-remote`. No code was installed or executed. These are
retrieval pointers, not claims of code-level reproduction:

| Repository | HEAD observed 2026-09-28 | License status |
|---|---|---|
| [Harness-RL](https://github.com/jiangxinke/Harness-RL/tree/740f24cd86bc23d2b44941327e9254e5a5124372) | `740f24cd86bc23d2b44941327e9254e5a5124372` | Root license not located in inspected listing; paper is arXiv non-exclusive |
| [Harness-1](https://github.com/pat-jj/harness-1/tree/8ac4012167858f6478fb2a8fd840e4550e2af161) | `8ac4012167858f6478fb2a8fd840e4550e2af161` | Repository Apache-2.0 |
| [PaW](https://github.com/ColinLu50/Policy-WorldModel-Agent/tree/4feb96444ec220a7e36cbe1d739cdfa66bd70f1c) | `4feb96444ec220a7e36cbe1d739cdfa66bd70f1c` | Repository Apache-2.0 |
| [LATTE](https://github.com/emieczkowski/latte/tree/8647e3b9cead575512c1476e8acad73bf2c5a495) | `8647e3b9cead575512c1476e8acad73bf2c5a495` | Repository MIT |

RAO's local native environment is already pinned at
`d9c5857d3a0a056ebc9b047241a2a0c9515aafbe` with MIT license and file hashes in
[`textcraft_bridge.py`](textcraft_bridge.py). No new datasets, weights, or
environments were acquired, so there are no new data splits or asset checksums.
The existing literature notes on LEAP/DAgger, HiPER, DecomposeR, SkillGate,
counterfactual credit, and context isolation remain prior art, not new ideas
introduced by this review.
