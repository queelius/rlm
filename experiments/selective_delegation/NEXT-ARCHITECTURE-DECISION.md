# Next architecture decision: execution-conditioned planning before new recursion

This is a decision memo, not authorization to interrupt the queued helper
comparison or fixed-helper RL readout. Recent short-context evidence makes a
generic "more helpers" architecture unjustified: direct answers beat the
planner on both four-hop MuSiQue (36/128 direct versus 24/128 SFT/RL) and the
Hotpot explorer transfer (36/64 direct versus 23/64 SFT and 24/64 RL). Direct
already receives the same public documents and fits the context window. An RLM
win produced by denying that access would be an artificial handicap, not a
decomposition capability.

This is not a novelty claim about incremental planning or structured execution:
DecomposeR already studies dependency-graph plans and revision, while RLM,
lambda-RLM, PyRAG, and C3 cover external recursion, typed composition,
programmed intermediates, and replay-based counterfactual comparisons. The
local question is narrower: can the current small planner avoid its *observed*
forward/self-reference and evidence-binding failures without merely converting
them to valid JSON or adding avoidable calls?

## 1. First admissible architecture test: incremental next-question planner

**Falsifiable question.** Given the public title index and actual prior helper
answers (rather than the original full documents), does choosing only the next
question reduce invalid dependencies
and increase end-to-end **both-valid-final** EM over a one-shot plan under a
matched root-token and helper-call cap? It is not enough to improve plan parse
rate: the four-hop SFT made all roots parse while answer EM fell, and Hotpot
showed both helper-format and wrong-chain losses.

**Minimal contract.** The root emits exactly one question or `FINAL` at a time.
After a strict helper answer, the root sees its own prior question/answer trace
and the unchanged public title index--not the original full documents--then
chooses the next question. The helper and final retain their existing
full-source contracts; the final also receives the trace. The executor rejects
forward references; no gold answer, annotated decomposition, answer repair, or
hidden retrieval is introduced.

**Cheapest fair comparison.** Freeze one helper contract selected by the pending
helper comparison, one released final, model/temperature/seed policy, documents,
and the existing meaningful per-parent root/helper caps. Compare one-shot root
planning against incremental roots with the same predeclared maximum number of
steps/calls already allowed by the relevant panel (rather than inventing a
two-helper limit that truncates current 2/3-hop or misses the observed four-hop
failures). Preserve the 128-token total root allowance, but disclose that its
distribution across decisions changes the policy. Keep direct answer as a third
policy rather than a straw baseline. Report actual tokens/calls separately:
equal caps do not make direct and multi-call execution compute-matched.

**Data/readiness and cost.** `fresh-dev-inputs-003` is now reserved for the
selected-helper/fixed-helper-RL full-pass readout, not a future clean architecture
panel. The already exposed four-hop panel can support a clearly exploratory
failure diagnostic because it contains the observed self-reference issue; use
new identities only after a credible signal. A 32-parent, two-repeat diagnostic
(one-shot, incremental, direct) is roughly 320--450 model calls depending on
early `FINAL` decisions, plausibly one A100-hour; verify a real full-source
helper and one native trace before extending.

**Go/no-go.** Report dependency/protocol recovery and both-valid-final EM as
separate outcomes. A repeated protocol-mediated end-to-end gain is useful
execution evidence, but not proof of better decomposition; a both-valid gain is
stronger evidence of planning value. If incremental planning mainly makes valid
but wrong chains, do not train another planner protocol. A direct short-context
advantage remains an important boundary, not an absolute veto: a meaningful,
replicated incremental effect can motivate a longer-input or genuinely
compositional follow-up where direct access is still fairly controlled.

## 2. Defer, do not launch: sufficient evidence messages on genuine composition

**Falsifiable question.** Can a leaf emit a fixed, evidence-carrying relation
message (entity ID, relation/value, record IDs, explicit unresolved items) whose
exact composition remains correct under two valid repartitions of the same
facts, and transfers to unseen surface and dependency structures? The required
comparison is free-form leaf + model final, same contract + model final, and
same contract + fixed exact combiner, with direct full-access baseline and
both-correct-across-partitions as primary outcome.

**Why defer the strong generalization claim.** This tests a real compositional
capability rather than a forced short-context weakness. A narrow fixed-depth
partition-robustness pilot need not first vary every axis, provided it says so
plainly and preserves matched facts, public access, and budgets. What is not
ready is a claim of depth/surface/length generalization: the available B05,
purchase, operator, and route assets confound those axes. Reusing their exposed
worlds would repeat prior singleton/normalization or partition findings; using a
length limit to make direct fail would manufacture the claimed advantage.

**Readiness/cost.** A narrow new fixed-depth partition pilot needs only exact
evaluation, public projections, two semantics-preserving partitions, and frozen
identities; it must not reuse the already exposed purchase/B05 findings as a new
result. A broader generator extension is needed only before claiming structural
generalization. No generic framework or new model training is justified. The
predeclared 24-case x two-partition four-arm diagnostic is at most 336 model
calls and one A100-hour; actual per-arm cost must be reported.

**Go/no-go.** For a narrow pilot, assess both-correct partition accuracy and
contract completeness without concealed final-source access. A Python-only gain
is a harness result, not learned recursion; a null/partition-fragile result
retires that fixed-depth contract. Require independent structure/length/template
holdouts only before promoting a generalization claim.

## Decision

Finish the current helper and fixed-helper root-RL decisions. Use Alternative 1
as an exploratory diagnostic on the already exposed four-hop panel only if its
observed reference failures remain decision-relevant; obtain new data after a
signal. Alternative 2 can begin as a narrowly scoped fixed-depth robustness
pilot, but needs stronger generator separation before any generalization claim.
Neither alternative supports a general RLM superiority claim on the current
short-context MuSiQue/Hotpot panels.
