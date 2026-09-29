---
title: "Teaching-history repair: close precedents and claim boundary"
date: 2026-09-29
status: bounded_primary_literature_review_no_run
evidence_cutoff_utc: 2026-09-29T01:14:27Z
scope: "Three precedents beyond GPO; one conditional rendering control"
---

## Decision

The result merits further study as a controlled demonstration-repair finding,
not as the discovery that privileged teachers can be hard to imitate. The
strongest defensible description is: an offline scheduler rearranges an already
correct oracle action list so prerequisite names appear publicly before being
queried; this improves this learner's tested-budget task completion while
preserving its literal taught answers and minibatch target-token counts.
The scheduler still uses an oracle plan. It is not an independently learned
public-information planner or a recursion result.

The three 32-attempt cohorts give original-teacher successes **1, 2, 3** versus
stable-repair **14, 13, 17**. They cover two fit seeds and sixteen unique goal
IDs, with repeated worlds and rollout seeds, not 96 independent tasks. All
366 answers, 23 minibatch target-token denominators and 8,820 answer tokens
including EOS are preserved. Input tokens are not: original 414,682, stable
431,274, random repair 435,710. Thus this is equal training steps and answer
supervision, not equal total tokens or FLOPs. Stable repair has not established
superiority to discovery teaching. See the
[completed synthesis](../teaching_synthesis_20260929/FINDING.md).

## Closest additional comparisons

1. **Robust Asymmetric Learning in POMDPs (A2D), ICML 2021.**
   The expert is trained to improve the partially observing trainee, while the
   trainee imitates it on mixed-policy rollouts. Its analysis shows why fitting
   a fixed privileged expert can be insufficient; exact guarantees require
   strong policy-class and update assumptions. Discussion flags approximate
   updates and non-omniscient experts as further work. This rules out novelty
   claims for either identifying teacher–learner information mismatch or
   adapting teaching to it. Our distinction is offline history repair with
   fixed answers, not joint expert/trainee optimization. Read §§3.2–5 and §7
   of the [venue paper](https://proceedings.mlr.press/v139/warrington21a/warrington21a.pdf);
   [arXiv v3](https://arxiv.org/abs/2012.15566v3) is dated 2021-07-01, first
   submission 2020-12-31. New to the inspected repository notes.

2. **Adaptive Information Gathering via Imitation Learning, RSS 2017.**
   A history-limited sensing policy learns from a map-knowing oracle using
   reward/value labels on learner or mixed-policy rollouts. Its analysis links
   this to averaging over worlds consistent with the learner's information.
   Stronger guarantees rely on adaptive submodularity; the constrained problem
   lacks that property, and improved guarantees are future work. This rules out
   novelty for learning useful queries from a clairvoyant teacher, and cautions
   against declaring privileged demonstrations intrinsically defective. It does
   not preserve a fixed list of literal next-action labels. Read §§II-C,
   III-C–D, IV-A–D and VI of the
   [RSS paper](https://www.roboticsproceedings.org/rss13/p41.pdf);
   [arXiv v1](https://arxiv.org/abs/1705.07834v1), 2017-05-22. Already cited in
   our [discovery-transfer note](../TEXTCRAFT-PUBLIC-DISCOVERY-TRANSFER.md);
   this is a method-level comparison, not a newly discovered citation.

3. **Temporal Behavior Tree-Guided Trajectory Repair, 2026 preprint.**
   This is a repair-method boundary, less directly about observability. Offline
   optimization changes state paths to satisfy formal specifications; repaired
   paths then define rewards for RL. Limitations include approximate dynamics,
   offline repair cost, and potentials that may not transfer between layouts.
   This rules out novelty for repairing demonstrations before policy learning.
   Unlike our setting, the original paths violate task specifications and the
   repaired supervision is not fixed-answer SFT. Read §5.1, the opening of §5.2,
   §6.2.3 and §7
   of [v1](https://arxiv.org/html/2604.04225v1), dated 2026-04-05; the
   [record](https://arxiv.org/abs/2604.04225v1) reports submission for possible
   publication, not verified acceptance. New to the inspected notes.

[SITT and Causal Confusion](../TEACHER-OBSERVABILITY-METHODS-REVIEW-20260928.md)
already constrain our claims; neither is a new find here. In particular,
holding demonstrations fixed while changing observation conditioning is not
itself new. [GPO/TigerDoor](../literature_decisions_20260929/README.md) is not
re-surveyed. This short search cannot establish absence of an exact precedent.

## What remains unexplained

The [counterfactual witness](../teacher_counterfactual_20260928/README.md)
found oracle next-query conflicts for all eight exactly matched public prefixes,
but all 48 public continuations succeeded. The forced queries were valid; some
cost one extra lookup. Thus exact-label unpredictability does not establish
harmful choices or explain the SFT gap. Contextual grounding, easier copying,
item-identity cues, and changed history length remain possible contributors.
Changed recipe worlds retain item names, so existing world transfer does not
remove identity cues. This is an inference from our controls, not a conclusion
proved by any cited paper.

## One conditional control: consistent item-name permutation

After the already queued repair/transfer results, use the existing original and
stable-repair fit-2026092208 checkpoint-23 actors, panel00's eight goals,
world42 and rollout seed2026092204. Reuse their 16 completed identity-rendered
controls only if their full inference contract matches. Run **16 new attempts**:
the same tasks and actors with one deterministic, episode-consistent item-name
bijection, shared across actors and frozen before outcomes (mapping seed
2026092909). This sharpens the earlier transfer note's rendering proposal; it
is not a new method or a pure memorization test.

Rename the goal, stock, observations and every action item field consistently;
decode names back before the unchanged native transition. No recipe, quantity,
query schedule, candidate list or hidden answer is added. Preserve public name
syntax where possible; do not use hidden dependency depth to choose aliases.
A CPU replay must first verify identical decoded native transitions and root
scores, strict parsing, and all actual token-length differences. Failed mapping
construction stays recorded; no replacement based on model outcomes.

Keep 96 calls, 8,192 total output tokens, 8,192 context tokens and 256 tokens
per response. Compare native completion, invalid-action counts and call/token
costs, not agreement with a single oracle label. Estimate 30–60 minutes on one
A100, hard cap 90 minutes; capped outcomes remain unknown. No new training.

If the repair advantage survives, dependence on exact item identities is less
plausible. If it disappears, narrow the generalization claim; if both actors
collapse, rendering shift/tokenization is an alternative explanation, not proof
of memorization. This control does not isolate causal observability or equalize
FLOPs. Prioritize it only if queued results justify a portable-procedure claim;
otherwise retain the present finite-budget, exposed-panel finding and stop
expanding controls. No package or queue change is requested here.

Exact versions, read scopes and local evidence hashes are in
[SOURCES.json](SOURCES.json). No paper assets, models or code were downloaded.
