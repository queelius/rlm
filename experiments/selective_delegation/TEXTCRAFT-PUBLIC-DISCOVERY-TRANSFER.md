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
