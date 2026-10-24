# Decomposition: brainstorm and prior art (2026-10-24)

This is a step-back page written after the user's prompt about how a small RLM should decompose large inputs:
- how to choose a decomposition;
- whether to probe the context first;
- how the question is carried into sub-problems and merged;
- whether a dataset classifying problems by the decomposition they need is a contribution;
- the bitter lesson.

Sources:
- this session's own brainstorm;
- an independent Codex brainstorm (`codex-reviews/2026-10-24-decomposition-brainstorm.md` in the run store);
- two literature searches (academic; practitioner blogs and posts), run 2026-10-24.

Citations below were checked by the search agents against arXiv, proceedings or the source page unless marked. Many
2026 items are unreviewed preprints or blogs. Read the key ones before relying on them.

## Our own evidence that bears on the question

- **Execution generalizes; question handling as written does not.** In E59, 55 of 55 libraries survive 5 times
  longer inputs, but only 3 of 55 survive a reworded question, and changing "QUESTION:" to "Q:" breaks most of them.
- **Neither teaching fix worked with a 4B proposer.** In E60, neither mixed-wording teaching nor a prompt note
  produced a library robust to rewording.
- **A small parser reads the shifts that break regex libraries.** In E61, a 4B parser gets 95.7% exact parameters
  on an unseen rewording; a 0.6B gets 58%.
- **RL did not learn the route choice; selection memory did.** RL collapsed every time, while an input-keyed
  outcome table chose routes well (E13).
- **Choose evidence; do not summarize.** Map-reduce notes, recursion and bigger readers did not help on our tasks
  (E23, E31/E32); retrieve-then-read did.

## A framework for decomposability (ours and Codex's, which agree)

Decomposability belongs to a question–input pair. The ideal is a question-dependent partial state S_q with
S_q(A ⊎ B) = S_q(A) ⊕_q S_q(B) and answer = h_q(S_q(input)). The axes:

1. **Merge algebra and state growth.** Distributive aggregates (count, sum, max) merge local answers directly.
   Algebraic ones (average) need a fixed-size partial state such as (sum, count). Holistic ones (median, mode,
   distinct count) need a growing state; sort-and-merge works but keeps a lot of information. The taxonomy is from
   database research (Gray et al., the data cube paper).
2. **Dependency topology.**
   - Independent units: a parallel map.
   - Ordered processes: a fold with carried state.
   - Joins: repartition by key.
   - Multi-hop: adaptive evidence acquisition.
3. **Segmentation unit and how discoverable it is.** Records, messages, sections and chapters are discoverable;
   concepts and entities are latent. Token windows are execution units, not meaningful ones.
4. **Evidence density.** Sparse evidence favours retrieval; exhaustive questions require a scan.
5. **Question transformation.** Each worker needs the local obligation that supports the global answer. "Who spent
   most?" becomes "return the total for every user", not "name this chunk's winner".
6. **Leaf reliability.** A correct decomposition can still fail statistically when leaves err and the answer must
   be exact.

**Division of labour.** Code handles explicit structure: boundaries, bookkeeping, arithmetic, joins and merges. LLMs
handle unresolved meaning: question interpretation (E61), semantic predicates, latent segmentation, and proposing
unfamiliar plans.

## Prior art, by question

**Theory.**
- Xu et al., "When does divide and conquer work for long context LLM?", ICLR 2026
  ([2506.16411](https://arxiv.org/abs/2506.16411)). Error splits into task, model and aggregator noise. Three
  regimes: trivial; "silo", where task noise dominates; and "brain fog", where divide and conquer works best. Task
  noise is measured after the fact, not predicted. Predicting it from merge algebra and dependency structure is open.
- Yang, Srebro, Li, "Recursive Models for Long-Horizon Reasoning", ICML 2026
  ([2603.02112](https://arxiv.org/abs/2603.02112)). Recursive decomposition with exponentially smaller active
  context.
- Schnabel et al., "Lost in Transmission" (BAPO), NeurIPS 2025 ([2505.08140](https://arxiv.org/abs/2505.08140)).
  A bandwidth account of why global tasks are hard.
- Du et al. ([2510.05381](https://arxiv.org/abs/2510.05381)) and Chroma's "Context Rot" (2025) both measure
  degradation with length even when retrieval is perfect.

**RLM itself and its direct competitors.**
- Zhang, Kraska, Khattab, RLM ([2512.24601](https://arxiv.org/abs/2512.24601)). Models peek, filter with regex,
  chunk uniformly and merge in REPL variables. The authors "did not observe complicated partitioning strategies
  beyond uniform chunking or keyword searches". Recursion depth is capped at 1.
- λ-RLM ([2603.20105](https://arxiv.org/abs/2603.20105), preprint). Typed combinators (Split, Peek, Map, Filter,
  Reduce, Cross) plus a task-type router chosen by one LLM call; it beats RLM in 29 of 36 comparisons. This is a
  hand-written library with a router, the closest competitor to our library approach. The search agent flagged
  internal inconsistencies.
- SRLM (Apple, [2603.15653](https://arxiv.org/abs/2603.15653), preprint). Samples 8 context programs and selects by
  plurality and confidence. It argues recursion is not the main driver. This is the try-all-and-evaluate competitor.
- **Reinforcement learning for RLMs.**
  - Zhang and Khattab, blog post (July 2026): an RL-trained RLM (Qwen3-30B-A3B) generalizes to inputs 8 to 32
    times longer, but learns a shortcut (handing everything to one sub-call), and MRCR needed a nudge to decompose.
  - Kim and Ahmad, "Reinforcing Recursive Language Models" (alphaXiv, May 2026): a 4B with GRPO as both root and
    child, which still needed a hand-written strategy prompt.
  - Prime Intellect blog (January 2026): RLMs hurt on DeepDive without a given strategy, and solved synthetic
    Oolong with regex alone.
- Wang ([2603.02615](https://arxiv.org/abs/2603.02615)): depth 2 hurts, and so do RLMs on simple retrieval.

**Choosing a decomposition.**
- Pre-Route ([2605.10235](https://arxiv.org/abs/2605.10235)) routes from document type, length and the opening
  snippet, before answering. A primitive form of probing first.
- MoC, ACL 2025 ([2503.09600](https://arxiv.org/abs/2503.09600)): a small-LM router picks a chunker, and the chunkers
  emit regexes.
- DOLCE ([2409.06338](https://arxiv.org/abs/2409.06338)) classifies tasks by sampling short spans.
- Select-Then-Decompose, EMNLP 2025 ([2510.17922](https://arxiv.org/abs/2510.17922)).
- Per-query workflow construction: MaAS, FlowReasoner. Optimizers: MOAR/DocETL, Abacus.
- Librarian Agents ([2610.22273](https://arxiv.org/abs/2610.22273), October 2026): RL-trained reorganization of a
  corpus into a navigable workspace.
- Gap: no controlled comparison of these regimes (free-form, library plus router, sample-and-select, optimizer, RL)
  on the same tasks at 4B scale.

**Segmentation.**
- Semantic chunking is generally not worth its cost over fixed-size chunking: Qu, Tu, Bao, NAACL Findings 2025
  ([2410.13070](https://arxiv.org/abs/2410.13070)).
- The best method varies by domain ([2603.06976](https://arxiv.org/abs/2603.06976)).
- HiChunk ([2509.11552](https://arxiv.org/abs/2509.11552)): benchmarks with sparse evidence cannot measure chunking
  quality.
- Structure trees: ToM ([2511.00489](https://arxiv.org/abs/2511.00489)), ZenDB
  ([2405.04674](https://arxiv.org/abs/2405.04674)).
- Gap: nobody measures what cutting a unit in half costs when every chunk is processed. A boundary-shift ablation
  inside an RLM would be new.

**Question transformation, merging and label-free checks.**
- LOTUS's `sem_agg` assumes the aggregation is commutative and associative.
- PRISM ([2412.18914](https://arxiv.org/abs/2412.18914)) carries a typed schema as the partial state.
- LLM×MapReduce resolves conflicts between chunks.
- Recursive merging amplifies hallucination ([2502.00977](https://arxiv.org/abs/2502.00977)).
- Operadic consistency ([2606.13649](https://arxiv.org/abs/2606.13649), preprint): agreement between the direct
  answer and the answer composed from a decomposition correlates 0.86 to 0.94 with accuracy on multi-hop QA. Not
  tested on long context.
- Gap: nobody checks answer(A + B) against merge(answer(A), answer(B)) for long-context decomposition, or labels
  queries by aggregate class.

**Taxonomies and datasets.**
- DOLCE, HoloBench (ICLR 2025, [2410.11996](https://arxiv.org/abs/2410.11996)), Oolong (COLM 2026), Loong, ETHIC,
  UDA-Bench, PredicateLongBench, TeaBReaC (contexts generated from programs), BREAK/QDMR (question decomposition),
  RULER, and Goldman et al. 2024 (diffusion by scope).
- Gap: no dataset labels instances with the split unit, the question transform, the merge operator and its
  algebraic class, and the dependency graph, verified by running a gold decomposition. None shows that the labels
  predict which strategy wins. Codex adds: make it interventional. Hold meaning fixed while varying evidence
  dispersion, dependency distance, segmentation visibility and state size, and hold out operator compositions,
  wordings and schemas, not just instances.

**Library learning.**
- TroVE (sample, keep the majority execution output, prune rare functions), LILO, ReGAL, AWM and ASI (skills
  verified by re-execution), ReasonFlux, SkillRet ([2605.05726](https://arxiv.org/abs/2605.05726): retrieval over
  16K skills "far from solved").
- Gap: a split/map/merge library induced from RLM trajectories, verified label-free, does not exist; λ-RLM's
  library is hand-written.

**Practitioners.**
- Isolated sub-contexts help for read-only workers and clean reviewers but fail for parallel writers. Cognition's
  "Don't Build Multi-Agents" (2025) and its 2026 follow-up; Anthropic's multi-agent research system.
- Delegation needs an objective, an output format and task boundaries, and should pass references, not summaries
  (Anthropic).

## The user's end goal: a learned probing policy

The goal: train an RLM to probe a large input (head, tail, random regions, running statistics, targeted greps),
infer its structure, and decide on a decomposition, possibly none. Then write the decomposition code.

The literature says this is the open part. Frontier models only peek, grep and chunk uniformly. Small models need a
given strategy. RL-trained RLMs learn shortcuts. Our own evidence says how to make it tractable:
- **Do not learn the plan choice by RL from scratch.** Run a menu of plans offline, label each input with its
  verified best plan, and train the prober to predict it from its probes (expert iteration). The probes become a
  learned signature for selection memory, which already works with a hand-designed key.
- **Split the work.** A neural question compiler (E61), structural decomposition in code chosen by probing, and a
  merge set by the task's algebra.
- **Order probes by cost.**
  - Free statistics: line-length histograms, delimiters, heading-pattern hits, repeated prefixes, JSON detection.
  - Then a few targeted reads.
  - A sliding classification of window types only when needed. This classifies structure; it does not summarize
    content.
- **The bitter lesson.** Search plus learning that scales with inference compute is compatible with it.
  Hand-written families and taxonomies are evaluation scaffolding, not the mechanism. The real risk, larger models
  simply reading everything, is answered (if at all) on an accuracy-per-compute frontier for small models, not on
  absolute capability.

**Strongest objection** (Codex): the easy cases are ordinary programs, and the hard cases need the understanding that
decomposition was meant to avoid. The answer would be end-to-end transfer to unseen semantic tasks and dependency
structures by small planners and workers, without oracle interfaces. It should beat retrieval and
semantic-parser-plus-query-engine baselines on accuracy against GPU cost, including the cost of building the
library.

## Candidate contributions

1. **A decomposition-type benchmark** with operational labels and controlled interventions on decomposability,
   built on `family_gen.py`, with a natural-data slice.
2. **A learned probe-to-plan policy for small models,** trained on verified outcomes and compared with λ-RLM
   (router), SRLM (sample and select), free-form RLM, and the input-keyed outcome table.
3. **Label-free verification of decompositions** by partition invariance and merge laws (extending operadic
   consistency to long context), used for selection and as an RL reward.
4. **Neural question compilation plus symbolic execution** for robustness (E59 to E61, arm H).

## Next experiments (ranked; all inference or CPU first)

1. **E62, the probe-to-plan oracle gap.**
   - Inputs: generated formats plus MRCR, OOLONG, MuSiQue haystacks, and LongBench documents with chapters.
   - Plan menu (about six): direct; retrieve then read; map over records with an exact merge; fold; map over
     sections; overlapping chunks.
   - The 4B chooses a plan from (a) head and tail, (b) plus statistics, (c) plus sampled window descriptions.
   - Measure regret against the oracle plan, and probe cost.
2. **Label-free selection by partition invariance.** Use existing candidates and sealed shifted tests, including
   constant-answer and systematically wrong candidates. Compare against selection by agreement.
3. **Boundary-shift ablation.** Cut units deliberately; measure the loss in map-merge plans.
4. **Typed-state map-reduce against prose summaries**, with an oracle-leaf replay to separate extraction errors from
   decomposition errors.
5. **Reproduce λ-RLM and SRLM at 4B** on our tasks, as competitor baselines.
6. **The frozen-executor interface test.** Swap regex, helper and oracle question parsers inside frozen libraries.
   E61 measured the parsing half.
