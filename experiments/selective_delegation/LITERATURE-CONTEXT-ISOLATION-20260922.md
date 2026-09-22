# Context isolation and missing bridges: conditional experiments, not a pivot

Retrieved/read directly from primary arXiv full texts on **2026-09-22 UTC**
(metadata checked08:47). No repositories, datasets, models or new environments
acquired; no experiment implemented or launched. Scope: these two papers only.

## 1. Recursive Language Models Generalize Out of Domain

Yang, Li, McAllester and Srebro; exact **2609.20831v1**. The
[versioned PDF](https://arxiv.org/pdf/2609.20831v1) and
[submission history](https://arxiv.org/abs/2609.20831v1) disagree with the
September identifier: both display **23 July2026**, with history03:49:21UTC.
This inconsistency is unresolved; crawler “yesterday” is not publication evidence.
Do not describe it as newly published yesterday. License: arXiv non-exclusive
distribution, not a checked open-source implementation license.

Sections3–6/AppendixG compare identical symbolic executions and labels under
full-trace versus active-frame visibility. Six-layer, approximately10.65M-parameter
models evaluate modulo10 expressions; this is not pretrained4B tool-use evidence.
At executed lengths1.25–1.5 times595, Table1 reports99.3% recursive versus24.5% CoT;
the farthest recursive bin also deteriorates. Constructed shortcuts supply a
mechanism test. The realizable-MDL argument and Proposition7 assume the correct
target factors through the retained observation; transfer covers previously
represented local contexts, not arbitrary unseen computations. Representation
existence is not an optimizer guarantee. Equal expression pools also need not mean
equal example/token dose. [Methods, results and assumptions](https://arxiv.org/pdf/2609.20831v1).

## 2. Missing Bridges: Composition-Aware Active Imitation Learning

Jacobson, Qureshi and Xue; exact **2609.18004v1**, submitted
**2026-09-16 01:48:13UTC**; [metadata](https://arxiv.org/abs/2609.18004v1),
[full text](https://arxiv.org/html/2609.18004v1), CC BY4.0.

AALT greedily acquires demonstrations connecting existing latent hubs, ranked by
reliable start–goal connectivity. It plans hub transitions through a shared
diffusion policy. In one structured simulated72-task retrieval domain,24 initial
demonstrations yield42/72; three acquired demonstrations/five transitions reach72/72
across five adaptation seeds. Same-candidate bridge baselines help isolate
acquisition choice; all policies receive environment action masks. The6.33M-model
experiment uses19,800 post-query optimization steps, so “three demonstrations”
does not imply three updates or an equivalent LoRA dose.

The reachability-information equality assumes perfectly reliable acquired edges;
the soft case supplies a bound, not guaranteed neural-policy improvement. Fixed
hubs, grounding/matching, reusable behaviors and the edge-product reliability
model matter; greedy acquisition is not globally optimal. AppendixE explicitly
limits the approach when missing skills, rather than missing connections, dominate.
[Methods, comparisons, propositions and limitations](https://arxiv.org/html/2609.18004v1).

## Our evidence changes the priority

Our [052 query audit](TEXTCRAFT-ACTION-SFT-FINDINGS.md) shows matched original
root-first13→2, with three baseline slots unknown; trained nonexistent queries
appear despite zero malformed actions. This supports a discovery-competence
problem, not yet a diagnosed missing bridge or harmful-history mechanism.
Our [QA isolation probe](EXECUTION-FINDINGS.md) showed no demonstrated gain and
changed several interfaces simultaneously. Paired-sufficiency046 is a direct
answer policy comparison, not recursion or unseen-depth extrapolation
([findings](PAIRED-RL-COMPOSITIONAL-FINDINGS.md)). Neither paper licenses relabeling
those results as evidence for its mechanism.

## Smallest falsifiable options, conditional and not accepted

**First choice after057: a public-state context test, not more recursion.** Freeze
16 TRAIN-side crafting states without selecting model wins/losses. Compare the
same frozen checkpoint using (a) full native history plus a public fact record,
and (b) that same record without historical narrative. Both receive identical
current/initial inventory, target quantities, queried recipes and negative query
results, action contract and remaining budgets; no hidden recipe lookup. The
record must be mechanically reconstructed from actual feedback and supplied to
both arms. Shared inventory/resource dependencies cannot be discarded as
“irrelevant context.” Reject the setup if CPU checks cannot preserve all needed
public state or fit both prompts without truncation.

Use two seeds and at most eight continuation calls per state/arm:512 calls,
128 output tokens/call, one resident4B, estimated≤45minutes on our A10040GB.
Measure native target attainment/finish, feasible crafts, repeated nonexistent
queries and actual costs; preserve all states. Improvement would motivate a
same-label/local-versus-full-history SFT comparison. No improvement, or losses
from removed information, retires this projection. This is our proposed
presentation/competence diagnostic—not a replication of the paper's training
theorem. Genuine equivalent native histories would be a subsequent invariance
check; do not manufacture misleading histories to handicap the full-context arm.

**Lower priority: bridge-targeted versus random additional demonstrations.** Only
if TRAIN tests establish reliable local skills but failed combinations, use the
existing trusted teacher to freeze a small set of quantity-correct connecting
traces. Rank candidates by additional TRAIN start–goal coverage versus uniform
selection from the same candidates. Include full inventory and public knowledge
in state identity; item-name overlap alone is not state reachability. Host-side
teacher knowledge is privileged supervision, never policy input or held-out
selection. Use at most eight additional LoRA updates per arm, matched rows/updates
and explicitly unmatched tokens, then the same preselected16 TRAIN continuation
states×two seeds×two arms×eight calls (512-call cap); budget≤90minutes including
training. Promote only for better native completion per expert transition beyond
random selection; retire if local skill failures persist or no distinct bridges
exist. Do not implement AALT's latent/diffusion stack for this screen.

## Decision and novelty boundary

Finish the already bounded056/057 teacher comparison first; no new GPU acceptance
follows from this note. If discovery remains weak, improving that skill is more
direct than hiding context or expanding a delegation tree. If discovery improves
but history-sensitive execution remains, the frozen public-state comparison is
the cheaper discriminating control. Keep paired-QA credit-estimator058/059 results
separate: more useful response credit is not a context-isolation result.

Recursive context restriction and composition-aware demonstration acquisition
are prior art here. Our possible contribution would be a carefully controlled
failure-mechanism/transfer result with native costs, not invention of either idea.
