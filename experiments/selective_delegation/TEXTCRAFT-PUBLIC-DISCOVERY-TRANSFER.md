# TextCraft public-discovery transfer boundary

This is a feasibility/novelty note for source055's proposed public root-first
teacher, not an accepted experiment. It does not repeat the LEAP/PACT
privileged-information review.

## What the acquired benchmark is

The pinned official [Platoon repository](https://github.com/ApGa/platoon)
describes TextCraft-Synth as procedural, abstract-name crafting with `craft`,
`get_info`, `view_inventory`, `finish`, and optional subagent actions. The
recipe-world and task samplers each take a seed. The associated
[RAO paper](https://arxiv.org/html/2605.06639v1), read at §3.1, varies
craft-tree depth (easy 2--3, medium 4--6, hard 7--9), trains on medium, and
evaluates easy/hard; its question is recursive training/delegation under
context constraints, not distilling public prerequisite discovery.

The existing changed-world audit matters: seed43 retains the **same 1,452 item
names** as seed42 but changes 1,399/1,402 same-named recipes (and 326 batch
sizes). A world seed therefore changes recipe topology, not identifiers. A
task seed also changes root selection and inventory difficulty; it is not a
matched holdout by itself.

Our small bridge is narrower than that paper: a single flat policy has public
goal/inventory and interactive recipe lookup, then receives native success at
finish. It should not be presented as an RAO replication, a recursion result,
or a new benchmark. The original TextCraft-style setting already assumes that
recipe search is an agent action; a query-first action sequence is consequently
ordinary tool-use supervision, not a novel algorithm.

## Prior-art boundary

[Learning to Gather Information via Imitation](https://arxiv.org/abs/1611.04180)
and [Adaptive Information Gathering via Imitation](https://arxiv.org/abs/1705.07834),
both read, train partial-information policies from clairvoyant training-time
oracles. [ReAct](https://arxiv.org/abs/2210.03629), also read, evaluates
language action/observation interleaving including ALFWorld. None is this
recipe interface. In the checked official sources, I found no exact
public-root-first versus privileged-intermediate-first TextCraft comparison.
That scoped absence is not novelty: query-first behavioral cloning is routine.

## Smallest meaningful transfer screen

If source052 first shows a credible action-SFT signal, the minimum useful
changed-world screen is seed43 with the same eight held root names and target
quantities. Reconstruct feasible initial inventories and native-verified
trajectories in each world; report realized craft count, branching, prompt
length, and cap status rather than pretending nominal depth matches. Compare
the two trained adapters (privileged-order and public-root-first) under the
same two decoder seeds: 32 episodes. This tests relative robustness, not a
base-model effect.

A base arm makes that screen 48 episodes. A separate deterministic item-ID
bijection is only a **rendering-sensitivity** probe: changed tokenization and
prompt lengths mean it does not isolate memorization. The broader 96-episode
proposal (both renderings, base plus two trained arms) is optional, not the
minimum, and should follow a signal rather than precede it.

Fairness requirements: freeze seeds, generator commit/hash, task IDs, item
bijection, prompts, action/context/token budgets, and score before adapters
are read. The actor sees no recipe except a reply produced by its own prior
`get_info`; teacher labels may use only the same public state/history. Do not
give the public teacher a gold plan, future lookup, target answer, or extra
lookup budget. Score native success and diagnostic root-first-query/invalid
action rates, not only exact action overlap.

Interpretation: public-root-first beating privileged-order on a difficulty-
reported changed world would support a procedural-information hypothesis within
this generator. A single small panel cannot automatically retire the teacher;
it can reveal a reason to stop, replicate, or broaden. Neither outcome
establishes generic prerequisite discovery.

## Follow-up decision addendum: after completed052, before057 outcomes

Source056→057 is now accepted separately; the follow-ups below are **not**
accepted, frozen panels, or implemented harnesses. Preserve the original prompt
as 057's primary comparison and the reminder as secondary. The general question
worth pursuing is whether public-evidence-constrained demonstrations teach a
transferable information-gathering procedure, or whether an equivalent cheap
instruction/state presentation explains their benefit. Query-first imitation,
recipe caching and symbolic quantity bookkeeping are not themselves novel.

In particular, [CGDP](https://arxiv.org/html/2605.07042v1) already studies persistent
belief state and a programmatic stopping gate based on query similarity and
observation novelty for iterative question answering. That rules out generic
“add state and stop loops” novelty here. Its exhaustion gate is not our exact
observable crafting-completion predicate: ending an unproductive craft trajectory
does not create the missing target. A remaining-requirements planner would also
provide computation beyond persistent facts. Treat these as known harness
controls around the narrower teaching/changed-transition question, not as an
unprecedented architecture.

Completed 052 motivates a different priority from a finish-only control. Original
action SFT succeeds 3/16 and reminder 2/16, with no schema failures. Neither profile
queries its root first in more than 3/16 episodes; 5/16 never query the root.
No failed trained episode reaches sufficient public net target inventory.
Base policies had queried their root first in every observed episode. Thus
positive-goal auto-finish cannot rescue the observed trained failures, and
inventory is already present in every prompt. Source057 must demonstrate task
progress, not just a nicer first action or fewer JSON errors.

### 1. Cheap procedural-prompt control: first when discovery behavior changes

Use the **unchanged privileged-order checkpoint23**, original prompt, same eight
tasks/two seeds, with one trailing instruction such as: “First query each requested
target's recipe. Discover needed ingredients from returned recipes before
crafting. Use current inventory and returned batch sizes; do not repeatedly query
an item whose recipe is already known. Finish only after the net target is met.”
Freeze exact wording once; no recipe names, future replies or model-selected
edits. Compare 16 new episodes against complete 052 original 16, retaining all
failures. Maximum 1536 calls, same 96/8192/256/8192 budgets; propose a 45-minute cap
with partial outcomes unknown. This avoids another training-plus-readout package,
not a 16-call first-action test; readout dominates the few-minute optimization
cost here. Existing 052 original used 646 calls/18 native
minutes, but altered trajectories could take longer.

This asks whether a direct procedural instruction rescues the same adapter's
information gathering and native success. If it matches a public-teacher gain,
the new training procedure is not yet necessary for this interface. Comparing
that prompted old adapter to unprompted 057 is a policy-baseline comparison, not
an isolated training effect. Extra instruction tokens and changed salience are
part of the intervention. Run if 057 improves discovery/success, or retains
guessed-name failures that an explicit procedure could cheaply challenge. Do
not run just to repeat the already-passed JSON reminder or if 057 fails because
of an unresolved runtime defect. If only first-action choice improves without
task progress, stop wording iteration rather than search indefinitely.

### 2. Changed-recipe world: first transfer test after useful task progress

If public teaching improves **native completion** beyond privileged teaching,
prefer the already-qualified seed43 control next: same eight root names and
requested quantities, both fixed adapters, two seeds, original prompt only.
That is 32 episodes, at most 3072 calls; propose a 90-minute cap and report unknowns
if capped. No new world-generation framework is needed. CPU preparation must
reconstruct feasible initial materials and native-verify both-world tasks,
freeze before new model outcomes, and tabulate realized craft count, dependency
chain/branching, batch sizes, distractors and prompt lengths. No replacement
based on model difficulty. Equal nominal tiers do not make difficulty equal.

Seed43 changes dependency edges and batch sizes while retaining names; it is not
just renaming. A relative public-teacher advantage under those changed facts
would support procedural robustness within this generator. Same names can also
induce negative transfer, and two worlds cannot establish generality. Do not
spend this readout on a pure formatting/root-first gain with no useful task
progress, or call a randomized identifier mapping semantic generalization.

### 3. Explicit public memory/requirements: only after facts are retrieved

If 057 queries the correct prerequisites but still repeats queries or mishandles
available quantities, the smallest state-presentation screen is one new 16-episode
arm with that fixed adapter, versus its complete 057 original 16. Append a sorted
map of **only previously queried static recipes** and each root's observable
remaining net quantity, `max(0, target - (current - initial))`. Keep full history,
native calls, parser, caps and learned finish unchanged. At most 1536 new calls;
propose a 45-minute cap. Charge all added input tokens and retain earlier context-cap
failures. No context compression means this tests a presentation package, not
longer memory or more evidence. Root-deficit arithmetic and recipe caching are
separate host aids; a positive package would need a cache-only control before
attributing the gain to quantity bookkeeping.

Do **not** silently expand the host's remaining-requirements field into recursively
solved ingredient demand, shared-stock allocation, or the next craft action.
Although computable from public queries, that would offload the planning algorithm
to the host, not test memory. Such a planner is a legitimate declared baseline
only for a later model-versus-host responsibility question. Do not run a memory
screen when the root was never queried, nonexistent-name discovery dominates,
or the only problem is an already-observed missing finish.

**Ranking:** meaningful 057 task gain → procedural-prompt qualification and then
changed-world transfer; discovery gain without completion → inspect quantity/
retention errors, one targeted memory screen only if they fit; no discovery or
completion gain → one cheap procedural baseline at most, then retire this small
TextCraft teaching branch rather than add depth, training dose or infrastructure.
These are prospective decision criteria, not outcome-selected claims or GPU
acceptance. Current CPU world feasibility and frozen052/057 comparison suffice
to prepare the next two candidates once the complete readout identifies which
questions remain informative.

### Procedural control is CPU-ready, not yet GPU-accepted

Source061 now implements candidate1 with the exact frozen instruction above.
The actual saved052 first request and input token IDs were checked; only the
suffix changes. The immutable plan contains 16 matched slots, the unchanged
048 checkpoint23, and an initial maximum of833 prompt-plus-output tokens.
The native replay, suffix-tamper, pairing/missing-outcome and adapter-routing
fixtures passed. Main independently ran the four new sealed tests in5.41seconds;
the agent also ran the inherited tiny-adapter routing check. Live052/057 source
was not changed.

Authoritative launch/analyze commands and40 input/import pins are in
`R/TEXTCRAFT-PROCEDURE-CONTROL-PROPOSED-002.json` (SHA256
`62fc55bf5590e9f9df286d0d7d1645841c2d783b923ae0bae81748e622001fe5`).
The source seal is `R/source-061-textcraft-procedure-control/SOURCE.json`
(SHA256 `42e5c3a57426987fe4dcba75ccd9296d12493945ea591784313593a516a750ff`).
Version001 remains as the earlier proposal;002 adds explicit transitive import
pins without changing the planned comparison. Neither receipt is GPU acceptance.

The [identical-public-start feasibility check](TEXTCRAFT-OBSERVATIONAL-AMBIGUITY.md)
is a separate, conditional design. It must not silently replace this prompt
control or the changed-world comparison above.
