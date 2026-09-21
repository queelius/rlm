# Research options after weak QA delegation signals

Status: decision note, 2026-09-21. This is not an accepted GPU job or a claim
of novelty. It assumes the ongoing MuSiQue/Hotpot replications and the small
ALFWorld screen are reported honestly first.

## What the current results say

On fresh MuSiQue, root RL16 (56/128) is close to SFT (53/128) and direct
(54/128), and its interval includes no improvement. Re-executing frozen plans
showed some execution-seed variability but little selection value (47 versus
46/64). The plan-only/executed mismatch and weak helper value make another
static question tree, more levels, or a generic planner/worker headline a poor
next direction. On Hotpot, helper execution has a suggestive but uncertain
advantage over new base direct, while applying the helper adapter directly is
not established as helpful. The actionable open question is therefore **when an
additional decision changes the information or available action, and when it
should stop**, not whether a model can emit a hierarchy.

ALFWorld is appropriate only as an affordance-assisted, exploratory development
screen. The public admissible-action list is extra executable information, not
mere syntax; both policies must see exactly the same list, feedback, history,
action cap, and 2,048 generated-token ceiling. The eight `valid_seen` games are
exposed development cases, not an unseen generalization result.

Prior art rules out broad claims: ALFWorld already explicitly separates
high-level abstract policy from low-level execution; ReAct interleaves reasoning
and environment action; Reflexion uses feedback-conditioned textual memory; and
ArCHer trains hierarchical multi-turn language agents. A useful result here
would be a narrow controlled measurement, not "hierarchical LLM agents work."

Primary sources: [ALFWorld](https://arxiv.org/abs/2010.03768),
[ReAct](https://openreview.net/forum?id=WE_vluYUL-X),
[Reflexion](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html),
and [ArCHer](https://proceedings.mlr.press/v235/zhou24t.html).

**Retrieval note.** This note's prior-art boundary was checked against the
ALFWorld arXiv abstract and official project page, the ReAct ICLR record/abstract,
the Reflexion NeurIPS proceedings page, and the ArCHer PMLR proceedings page on
September 21. No further ALFWorld hierarchy result is asserted from uninspected
prior knowledge.

## 1. Public-state change gating in ALFWorld — recommended screen

**Falsifiable question.** Does a manager help only when public feedback changes
the decision-relevant state (for example, target found, target absent at a
searched location, or a new applicable transformation), rather than merely
because it emits an extra natural-language goal?

Run the already prepared eight games × two seeds. Compare the flat reactive
policy against (a) the fixed four-action goal-manager/worker now planned and
(b), only if (a) has a signal, a manager invoked at predeclared public-state
change events. The manager receives no expert plan or hidden PDDL state. It can
select a short subgoal or stop; the worker chooses environment actions. Every
arm gets the identical public action list and total generated-token/action caps.
Count manager tokens/calls, action-list length, parse/grounding failures, native
`won`, and actions-to-success. Do not give the manager a larger hidden retry
budget.

**Cheap kill baseline and interpretation.** Flat reactive is the primary kill
baseline. A fixed periodic manager with the same maximum calls is the direct
control for "more deliberation." If flat matches gated management, or periodic
management matches it at equal realized token/action cost, there is no evidence
that public state-change gating adds value. If gated management wins only by
using more action attempts, it is a budget result, not adaptive planning.

**One-A100 screen.** 8 × 2 × 2 arms is 32 episodes for flat versus fixed manager;
the conditional gated arm is another 16 episodes. With the 2,048-token and
50-action ceilings, target an initial 30--45 minute screen and stop at one hour.
The result can justify a larger *new* panel only if it shows paired wins at equal
information and credible realized-cost accounting. It is not publishable
evidence of a new hierarchy by itself.

## 2. Evidence-sufficiency / expand-or-stop control — stronger QA direction,
but CPU preparation first

**Falsifiable question.** Can a policy use only visible evidence to distinguish
"answer now" from "request one bounded additional evidence view," and improve
end-to-end accuracy per actual token over direct answering and always-expand?
This makes delegation useful only when it changes information access.

The official MuSiQue paired sufficient/missing-evidence variants are preferable
to deleting paragraphs after the fact: they separate source-unsupported from
globally false claims and offer an official sufficiency outcome. Freeze paired
variants and a host-only document partition before calls. The model sees an
identical initial public projection and selects `answer`, `one title-indexed
evidence request`, or `insufficient evidence`; a fixed host route reveals the
requested public paragraph, never a gold answer. Score answer EM/F1 on
sufficient cases and the official sufficiency metric on paired variants. A
policy that confidently answers missing-evidence variants has failed even if a
pretrained fact happens to be true.

**Cheap kill baseline.** Direct-on-initial-context, always-expand-one-view, and
an oracle *availability* upper bound (host evaluation only; never prompt it).
If always-expand matches adaptive routing at the same or lower realized cost,
or if the routing signal does not separate official sufficient from paired
missing-evidence cases, retire it. Do not claim a new retriever, factuality
system, or generic abstention method.

**Readiness and screen.** The archive/schema audit exists, but a public,
non-leaking title/paragraph action interface and paired split manifest do not
yet. Build and test those on CPU before accepting a run. A 16--32 pair
development screen with two seeds and three arms is roughly 96--192 answer or
expand calls; cap it at one A100-hour. This is more publication-oriented than
another helper SFT only if the action changes accessible evidence and the
paired sufficiency control holds.

## 3. Evidence-carrying composition contracts — defer, do not adapt old pilots

The remaining distinctive composition question is whether a bounded message
with evidence IDs and unresolved items survives a semantics-preserving
repartition better than free-form leaf reports, with the same leaves compared
under a model versus exact combiner. That is a question about sufficient
interfaces and partition robustness, not recursive depth generalization.

It is **not ready** as a next screen. Existing purchase and B05 results already
cover local joins/singletons and do not independently vary surface, length, and
dependency depth. Re-running them would not be novel. A small fixed-depth,
new-surface/partition test could eventually be worthwhile, openly without a
depth-generalization claim, but requires a new frozen generator family and
fair actual-cost controls first. The minimal killer remains: free leaves or a
model combiner match the contract+exact-combiner arm on both-valid partition
accuracy. Until that family exists, prefer options 1--2.

## Decision rule

Finish the CPU-prepared (not yet GPU-accepted) ALFWorld flat-versus-fixed screen
and ongoing fresh QA readouts before expansion. Promote a direction only for a
repeatable paired effect that survives its simple control, keeps information
access symmetric, and reports actual calls/tokens and stopping failures. A null
is useful: it would strengthen the conclusion that added internal decomposition
has not yet shown value for this base model under short, fully observed contexts.
