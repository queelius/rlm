---
question_id: rq:recursion_or_subtask_practice
created_utc: 2026-09-21T18:25:00Z
status: prospective_not_gpu_accepted
depends_on: textcraft_bounded_inference_qualification
novelty: requires_prior_art_review_and_positive_evidence
---

# Is the benefit recursion, or practice on smaller tasks?

The immediate proposed TextCraft pilot is narrower: can an unchanged model use
recursive calls effectively under the same total budget as a flat agent? It does
not yet train anything. See [the readiness inspection](TEXTCRAFT-RECURSION-READINESS.md).

If that pilot supports further work, a stronger training question is whether
learning from smaller tasks improves the underlying model, whether the recursive
interface helps at inference time, or whether both are needed. These explanations
should not be combined into a single trained-recursive versus untrained-flat result.

A useful minimal experiment would evaluate the same trained checkpoint both flat
and recursively. A more informative follow-up would compare training on full tasks
alone with training that includes locally verified subtasks, then evaluate both
checkpoints through both interfaces. The subtask control must receive comparable
training examples and measured update/token budgets. Otherwise an advantage could
come from easier supervision rather than learning to decompose.

Task difficulty should be defined independently of outcomes, using recipe depth
and the host's trusted solution description. Generalization to deeper combinations
of familiar recipe operations is a valid target; it is different from unfamiliar
recipes. Root-target disjointness alone does not imply the latter, as the cached
validation tasks reuse TRAIN intermediate items.

Track root success first, then successful child goals, wasted or unnecessary child
work, total calls/tokens and actual service time. A child can solve a real but
irrelevant task, so child success must not replace root success as the headline.
Use the smallest informative training dose and stop if the interface merely fails
to parse, if all tasks are trivial, or if no sampled actions change outcomes.

This is a candidate controlled study, not a novelty or performance claim. It does
not authorize a larger training campaign before the bounded inference pilot and
its failure analysis are complete.

## September22 update: optional recursion after public-action competence

**Prospective CPU scout only; no collector change or GPU acceptance.** The earlier
proposal above preceded the completed pilots. Public-discovery056 now solves
10/16 original057 flat slots. A same-adapter recursion-enabled screen is useful
and small enough to consider **after accepted062 transfer,063 TRAIN readiness,
fresh-root comparison and any accepted conditional RL**. It is an optional-use
test, not a trained-recursion claim or reason to displace those priorities.

### The interface permits it, but the supervision never demonstrated it

The unchanged bridge instruction explicitly advertises
`{"action":"delegate","targets":{"item":2},"context":"brief useful public context"}`
and allows delegation when `agent_depth < max_agent_depth`. It does not tell the
model never to delegate. CPU comparison of all eight initial task frames found
exactly one public-field change between flat and recursive: max_agent_depth0→2.
Root0 can create child1 and grandchild2; no deeper node is permitted.

However, all366 public056 training rows have agent_depth0/max_agent_depth0 and
targets167 queries,167 crafts,32 finishes—**zero delegate targets or child
trajectories**. The policy learned competent flat discovery, not recursive
planning. Allowing depth2 changes a training-state feature and may be ignored
despite the advertised schema. This lowers the probability of optional helper
use, but is not a prohibition that makes the test logically pointless. Child
execution may transfer flat competence to a smaller goal; choosing that goal
and supplying useful public context are untrained components.

### Existing native mechanics already support the exact comparison

`eval_textcraft.episode` recursively invokes the **same client instance**; the
owner loads one frozen `textcraft_action` PEFT adapter and never switches it for
child calls. Thus root/child/grandchild use public056 checkpoint23, the same
base4B, tokenizer and T=.5/top-p1/top-k0. There is no base-only helper or hidden
host decomposition. Strict JSON parsing rejects invalid quantities, unknown
fields and over-depth delegates without repair; rejected calls remain charged.

All nodes share one inventory and one96-call/8192-generated-token budget, with
cap256 per call and input+cap8192 without truncation. Delegate and finish model
responses, invalids and child responses count. The parent sees a mechanical
child status/message/current-inventory return, not a free model summary or gold
success. Child contexts are fresh: structured subgoal plus parent-supplied
public context and current inventory, not automatic access to parent history.
Each child's required quantity is **additional to its own starting inventory**;
root scoring retains the original root inventory baseline. This matters for
overproduction and prevents child success from substituting for root success.

### No inspected completed run answers this trained-policy question

044 evaluated base flat/recursive: among recorded rows it had zero child nodes
in either arm. Its flat group has13 observed/3 missing, recursive12 observed/
3 missing/1 unavailable. Different base successes without any delegation are
not evidence of recursion benefit. 052,057 and061 are flat-only;062's PLANs are
also flat-only (no partial062 quality inspected). No public056 recursive result
was found in this bounded run inventory.

### Smallest prospective screen

Reuse all16 original057 slots (8 roots×2 seeds2026092204/2026092205), original
world42/task bytes, same public056 checkpoint23, original prompt and global
budgets. Collect **16 recursive-enabled episodes only**, at most1536 new calls,
with a proposed45-minute owner cap; retain all incomplete outcomes as unknown.
Reuse the completed057 original flat arm only after exact task/adapter/model/
tokenizer/sampling/seed/native-source checks; do not substitute057's reminder
arm or a vLLM baseline. Historical flat acquisition cost remains separate.
Current NativeClient and episode ASTs match source057 exactly. The renderer
later acquired a procedure-control branch, but its original branch remains the
same bridge prompt; bind/check that branch, not a claim of whole-file identity.
If any scientific generation setting changed, recollect both arms instead.

Use the existing per-call seed formula unchanged:
`repeat_seed + int(sha256(task_id)[:8],16) +1000*global_call_index`.
Seeds match by global call index, **not by semantic role after branching**.
Report root success on the full16 planned denominator, missing/unavailable,
paired8-root uncertainty, actual delegate frequency/depth, child success and
failure, context/quantity errors, and total root+child tokens/calls/service time.
Re-audit native tree/inventory receipts. Child success alone does not show that
the child was necessary or useful to the original goal.

If no delegates occur, conclude only that this flat-trained policy did not
spontaneously use the available interface under these budgets. Any success
change without children is a prompt/state effect, not a recursion gain. If
children occur, root benefit/cost can motivate a separate TRAIN-only delegation
learning/control question. Do **not** force delegation, inject an oracle plan,
add demonstrations or change the prompt within this optional-use comparison.
The exposed eight-root panel is a qualification diagnostic, not fresh-domain
generalization or an independent16-parent benchmark.

Read-only evidence: `R/analysis-textcraft-pilot-001.json`, PLANs for044/052/057/
061/062, and all366 `textcraft-public-discovery-prototype-001/rows.jsonl` rows.
Fixed public adapter SHA256
`f029d36967eb1cef30e102fe82e52f22f90d5eacdaf91e82937b60218ba5b00f`;
bridge SHA256 `1553db71f5c4dd4738e50d087c0a33d99d9ea7f97a084e136e4360837f2a7d40`.
The original eight-task bytes hash is
`16a6663385759a9a16fa7ca44e6601b4d491234f8b04ccb9f2bf522c7ced8ab3`.
