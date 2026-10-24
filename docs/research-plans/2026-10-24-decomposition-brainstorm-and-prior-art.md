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

## Reframe (user, 2026-10-24 ~10:30 UTC): maybe it is not about recursion

**The user's observation.** With enough interrogation of the input, frontier models in a good harness do much of
this without sub-agents. They iterate on external state (files, notes, variables) and update their context as they
go.

**What supports it.**
- Our own results: recursion did not help (E23); programmatic access, verified routines and selection memory did.
- RLM runs at depth 1, and its models mostly chunk and grep.
- SRLM: recursion is not the main driver.
- Depth 2 hurts in a reproduction.
- Agentic search over a filesystem gives large gains with no recursion
  ([2603.20432](https://arxiv.org/abs/2603.20432)).
- Practitioners: Manus rewrites a todo file to keep the goal in attention. Prime Intellect's RLM "never summarizes",
  building the answer in a Python variable.

**The recentred thesis.** For small models on large inputs, the active ingredients are:
1. programmatic access, with intermediate state held outside the context window (REPL variables or files) and
   updated iteratively;
2. fresh LLM calls only as neural functions, applied from code to small pieces where code cannot decide (semantic
   maps such as OOLONG's per-item labels);
3. verified routines, selection memory, and narrow trained skills (question to parameters, probes to plan) that
   substitute for the open-ended control frontier models do zero-shot and small models do not.

Recursion becomes an ablation.

**Decisive positioning experiment (ranked above E62).** A 4B model in a frontier-style iterative harness, at matched
cost:
- a Python REPL with grep and read-range helpers;
- a scratch state the model rewrites (notes, a todo list, variables);
- no sub-calls.

It is compared with the same harness plus a neural `map` helper, with a sub-calling RLM, and with our library and
selection-memory system, on OOLONG, MRCR and LongBench-v2.

**What the outcomes would mean.**
- If tools and state alone match the library system, that is the finding.
- If not, the comparison shows what the library and the narrow skills add.

**One risk.** Small models may decay over many iterations: one preprint reports that step count, not context
length, drives decay ([2609.01660](https://arxiv.org/abs/2609.01660), preprint).

## System One classifiers and cheap structure inference (user prompt: "Jev-like" and universal classifiers)

**Jev** (TypeSafe AI, September 2026; https://docs.typesafe.ai) is a closed, hosted classifier. It takes a state plus
typed questions (choice, score, yes/no) and returns calibrated probabilities instead of text, at $0.042 per million
input tokens. It is said to be trained with "RL for calibrated decisions". There is no paper and there are no
weights.

Independent tests are mixed:
- Parallel: it matched an in-house reranker but lost to specialised classifiers.
- Qdrant: it gained +0.067 nDCG@10 in reranking. A chunker that cuts where Jev's "starts a new topic" probability
  passes a threshold beat other chunkers on QASPER.
- A blogger's roundup: complementary probabilities summed to 1.19.

**Open analogues** (found by the search, not yet checked by us):
- Simple Jev (`featherless-ai/simple-jev`): scores any Hugging Face LLM's answer options in one forward pass.
- Cloudflare Clef and Clef-flash (Apache-2.0): a Qwen base with LoRA and an option-scoring head.
- OpenJev (MIT): trained on some benchmarks' test splits; do not use it for evaluation.

**Universal and few-shot classifiers.**
- Laurer et al.'s NLI zero-shot models ([2312.17543](https://arxiv.org/abs/2312.17543)); ModernBERT-large zero-shot
  runs about 1,000 pairs per second.
- GLiClass ([2508.07662](https://arxiv.org/abs/2508.07662)) scores all labels in one pass.
- SetFit.
- Fine-tuned small encoders beat zero-shot LLMs at classification.
- Calibration: contextual calibration ([2102.09690](https://arxiv.org/abs/2102.09690)) and batch calibration
  ([2309.17249](https://arxiv.org/abs/2309.17249)). Batch calibration needs mixed-context batches, because windows
  of one document share a type.
- JEPA text encoders: no usable checkpoints, and one report says they collapse
  ([2607.23531](https://arxiv.org/abs/2607.23531)).

**Classical structure inference.** These are prior art for "probe, then write a splitter", and for library
bootstrapping:
- TextTiling and C99 for topic boundaries; Koshorek et al. 2018 for supervised sections, with Wiki-727K as free
  labels.
- Drain and LILAC (LLM plus template cache) for log templates.
- PADS, "From Dirt to Shovels", POPL 2008: infers a data format's grammar from a small sample.
- Datamaran, SIGMOD 2018: infers records and fields without supervision.
- CleverCSV for CSV dialects.
- Wrapper induction (Kushmerick) for delimiter-based extraction.
- FlashExtract, PLDI 2014: synthesizes extraction programs from about 2 examples per field. This is the
  programming-by-example precedent for bootstrapping libraries from three examples.
- Evaporate, PVLDB 2023 ([2304.09433](https://arxiv.org/abs/2304.09433)): an LLM writes extraction code from sampled
  documents, using 110 times fewer LLM tokens. This is the closest precedent to our proposer-writes-library approach
  and must be cited.
- Magika, for content-type detection.

**Routers and cascades.**
- RouteLLM gates work when labels are crisp.
- Adaptive-RAG's complexity classifier is only 54.5% accurate; an oracle router adds 9 to 12 F1.
- Online cascade learning (logistic regression → BERT → LLM) nearly matches the LLM at up to 90% lower cost.
- Escalating only low-confidence cases is the proven pattern.

**Where it fits.** Most neural leaf calls in long-context pipelines are decisions, not generation:
- window typing for probing;
- OOLONG per-item labels;
- choosing a plan from probes;
- relevance filtering;
- the choice-type parameters of a question.

A calibrated System One leaf can abstain and escalate to the generative 4B, which gives a label-free reliability
mechanism.

**Experiment (E63, to design and review).**
- Tasks: window typing over about 12 structural types, and OOLONG per-item labels.
- Conditions:
  - (a) character n-grams with logistic regression;
  - (b) a small embedding model with a linear probe, or SetFit;
  - (c) zero-shot GLiClass or ModernBERT;
  - (d) the local 4B with Simple-Jev-style calibrated option scoring;
  - (e) a cascade from (a) or (b) to (d).
- Measures: window macro-F1, decision accuracy, end-task accuracy inside a code-orchestrated map, ECE, and
  GPU-seconds.
- Optional: Jev itself, if the user has access and agrees to send synthetic data.
- Promote a cheap typer if it comes within 2 points of (d) at 10 times lower cost.

## The REPL as a structured, reproducible harness (user prompt, 2026-10-24 ~10:40 UTC)

**The user's thought.** Perhaps the Python REPL is just a very structured, reproducible harness: variables instead
of files, functions instead of tool calls.

**The field has converged on code as the harness interface.**
- CodeAct ([2402.01030](https://arxiv.org/abs/2402.01030)): code actions beat JSON tool calls, with the largest
  gains for open models.
- Anthropic, "Code execution with MCP" (engineering blog): tools become code APIs. "The agent only sees what you
  explicitly log or return." Saved working functions become skills.
- Cloudflare's "Code Mode", smolagents' CodeAgent, and `dspy.RLM` take the same approach.
- **Zhang and Khattab's July 2026 harness post** (https://alexzhang13.github.io/blog/2026/harness/) reframes RLMs
  in almost exactly the user's terms:
  - The harness supplies compositional generalization through *context offloading* (input held in a variable) and
    *programmatic sub-calling* (outputs stored in variables).
  - A good harness keeps every LM call "locally in-distribution", resembling the model's training data.
  - Recursion "is not strictly necessary".
  - They used one 30B-A3B model and make no claims about small models.

So the reframe is right and is now the RLM authors' own view. Novelty must come from the small-model regime,
verified procedural memory, selection memory, and measured principles.

**What a REPL harness gives that files and tool calls do not, for small models specifically.**
- **Typed state that never enters the context.** Counters and dicts, not serialized files.
- **Control flow in code.** Fewer model decisions per task; one preprint says agents decay with step count.
- **Functions with contracts.** These make routines verifiable units, which is why library verification works.
- **A trace that is itself a program.** It can be replayed, diffed, minimized, and lifted into a reusable routine.
  Turning a trajectory into a routine is library learning.
- **Loud failures.** Exceptions surface, unlike silent prose errors.

The costs are a sandbox and the security work that comes with it.

**"Locally in-distribution" unifies our results.**
- **E59 and E61.** Regex libraries fail once the question leaves its template's distribution. A 4B parsing a single
  question is in distribution, so it is robust.
- **OOLONG.** Per-item classification calls are in distribution for a 4B; reading the whole context is not.
- **Route choice.** RL route choice from raw long inputs collapsed, while an outcome-table lookup works.
- **Hypothesis: the smaller the model, the narrower its in-distribution region, so the finer the harness's
  decomposition must be.** With a fine enough harness, model size matters little; E14 found a 0.6B caller equal to
  the 4B with the library.

**Systems analogies with prior art.**
- ClawVM ([2604.10352](https://arxiv.org/abs/2604.10352), EuroMLSys 2026): harness-managed virtual memory.
- Pichay, "The Missing Memory Hierarchy" ([2603.09023](https://arxiv.org/abs/2603.09023)): demand paging of
  context, reporting 21.8% structural waste.
- MemGPT.
- Record and replay: agrepl ([2607.16200](https://arxiv.org/abs/2607.16200)) has a determinism invariant and
  replay fidelity 1.0. Also harnix and Belay.
- LongHarness Bench ([2609.38137](https://arxiv.org/abs/2609.38137), to verify): decomposition reduces the context
  per call "without sufficiently reducing the final decision space".

**Thesis statement, draft (2026-10-24).** Small language models can solve tasks over inputs they cannot read when a
programmable harness keeps every model call locally in distribution:
- code holds the state and the control flow;
- verified routines carry procedures;
- an input-keyed memory selects among the routines;
- the model is called only for small, typed decisions it can make reliably.

Recursion is one way to produce such calls but is rarely necessary. What matters is the granularity and verification
of the interface.

**New experiments this suggests.**
- **E65: harness-granularity scaling.**
  - Callers 0.6B, 1.7B, 4B and 8B, crossed with four granularities: direct; a coarse chunk map; a per-record neural
    map; library code with per-item neural leaves.
  - Families: OOLONG and the event log.
  - Prediction: the best granularity gets finer as the model gets smaller, and the curves converge at the finest
    granularity.
  - The figure, accuracy against granularity for each model size, could be the thesis's central plot.
- **E66: trajectory surgery.** Replay recorded REPL trajectories deterministically, swap one leaf result for the
  oracle or a perturbation, and re-execute the rest. This attributes errors to leaves, plan or merge. It is a
  methodological contribution enabled by the reproducible harness; the runtime already records traces and
  attestations.
- **E67: lifting trajectories into routines.** Turn successful REPL traces into verified library functions, as with
  TroVE or LILO, and compare them with proposer-written routines on yield and robustness.

**User (2026-10-24 ~11:00 UTC): the Python REPL as the natural home for LLMs, in training as well as inference.**
Python is Turing-complete and stateful, and its traces are consistent records usable for training and evaluation.
Models should learn under RL to use the environment well. SFT needs traces, which were not available at first.

**Agreements.**
- RL inside code environments is where the field is going: Zhang and Khattab's RL-trained RLM, Prime Intellect's
  environments, CodeAct-style training.
- Our own scripted-expert traces followed by RL lifted OOLONG.

**Corrections and nuances.**
- **Reproducibility is not free.** Codex's review of `src/rlm` found that the runtime records model responses and
  fingerprints traces. It does not provide deterministic execution or transactional rollback, and a failed cell can
  leave mutations behind. Full replay also needs control over files, dependencies and randomness, and the executor
  is not a security sandbox. Traces used as training data inherit hidden-state dependencies.
- **The action space may be too large for small models.** "Python is universal" does not mean "unrestricted Python
  is best for small models". The 4B reliably *calls* library functions, and so does the 0.6B (E14). But writing
  open-ended programs needed the 8B, or SFT followed by RL. Practitioners report small CodeAgents looping on broken
  Python.

  Open question: how much expressivity should the environment expose for a given model size? The options range
  from full Python, through a typed API or DSL inside Python (as λ-RLM's combinators do), to a single library call.
  This is the action-space side of E65's granularity question.
- **RL works when rewards are local and verifiable.** RL on the proposer, rewarded by verified teaching score,
  worked. RL on the controller's route choice collapsed. The REPL helps RL because it makes rewards checkable,
  through unit tests on functions; long-horizon credit assignment remains hard.
- **The trace problem has a known fix in this project.** The library plus a scripted driver generates SFT traces,
  and RL takes over from there.

These points sharpen the thesis: the REPL is the environment for both training and inference. The research is in
the interface it exposes and in what the model must learn within it, as a function of model size.

## Red-team review (Codex as a skeptical committee member, 2026-10-24 ~11:10 UTC)

Full text: `codex-reviews/2026-10-24-red-team-novelty.md` in the run store.

**Verdict.** A REPL with variables, helpers and reusable routines is not by itself a thesis contribution, and most
mechanisms in this project have prior art:
- CodeAct, RLM and Anthropic's code execution with MCP;
- λ-RLM, for prebound verified combinators;
- Adaptive-RAG and bandits, for outcome-table selection;
- TroVE, LILO, Voyager and Evaporate, for growing libraries;
- FlashExtract, for programs from a few examples;
- LOTUS, for semantic leaves;
- semantic parser plus query engine, for E59 to E61;
- PAL and Program of Thoughts.

Its harshest summary: "you moved the solution into a library, then showed that a small model can invoke the
solution."

**Missing critiques we must address.**
- "Library Learning Doesn't" ([2410.20274](https://arxiv.org/abs/2410.20274)): reported gains came from
  self-correction and self-consistency, not reuse.
- A compute-matched re-evaluation of TroVE ([2507.22069](https://arxiv.org/abs/2507.22069)): most of its advantage
  on MATH disappeared once compute was matched.

**What survives as narrow original findings.**
- E59 and E60, with E61 as a diagnostic: length generalization hides severe brittleness at the question interface.
- E39 (with E37, E38 and the E49 bound): a cheap acquisition protocol works within one program distribution.
- E26c, read with E43 to E48: proposer training improves discovery, with large seed dependence.

**Corrections to this page's reframes.**
- **"Locally in-distribution" is borrowed and unmeasured.** It is Zhang and Khattab's thesis. Operationalize it, or
  stop using it as an explanation.
- **Caller size is not system capability.** E14's equal callers share the library and the helpers' compute, and
  E61's gap between the 4B and the 0.6B argues against "size matters little".
- **E61 is slot filling with a supplied schema** and an already located question. It is not a question compiler.
- **"Verification" is used too strongly.** Passing three examples and running on more inputs is testing.
- **Partition invariance does not establish truth.** Doubled counts, or a consistently wrong classifier, pass it.
- **The granularity law is unsupported, and E65 confounds several factors:** semantic unit, number of calls,
  symbolic aggregation and supplied knowledge.
- **The RL-collapse claim is broader than the evidence.** E07's mechanism is unresolved.
- **E66 must specify downstream semantics.** After a leaf is swapped, do downstream calls reuse their old outputs or
  resample? Serving is nondeterministic (E22c).
- **Renaming familiar architectures is drift.** "Nobody combined these at 4B" is weak novelty, and the "nobody
  checks..." claims above need stronger literature support.

**Accepted research questions (alternatives, not all at once).**
1. Can cheap probes choose a decomposition that beats a fixed pipeline or a cheap feature router, given measured
   headroom over an oracle?
2. Can a local 4B proposer acquire procedures that stay useful under meaning-preserving shifts and repay their
   acquisition cost? Compare with an 8B proposer, primitive-only code generation and a TroVE- or Evaporate-style
   baseline.
3. Do label-free checks reduce false acceptance under held-out shifts, at matched cost and coverage?

**Decisive experiment (supersedes the E64 sketch above).**
- Hold the 4B controller and helper fixed, and cross two factors:
  - interface: a conventional iterative harness (read and search tools, scratch files, script execution) or a
    persistent Python REPL;
  - procedural knowledge: generic primitives, human-written routines, or model-acquired routines.
- Add a host pipeline that invokes the same routines through a cheap selector or typed parser, with no
  conversational controller, plus direct answering.
- Tasks: MRCR, OOLONG, LongBench-v2 or MuSiQue, and held-out generated families with rewording, layout and
  composition shifts.
- Measure device-seconds at two compute caps, the amortized cost C(N) = C_training + C_acquisition + N·C_solving,
  robust yield and false acceptance.
- Reading the outcomes:
  - REPL ≈ conventional harness and acquired routines robust and amortizing → the thesis is about acquisition.
  - Human-written routines work but acquired ones fail under shifts → task-specific programming, with acquisition
    unsolved.
  - Host pipeline ≈ the best controller → conversational control adds little.
  - Generic tools ≈ the full system → the distinctive claim is gone.

**Our response.** We accept the critique of the reframes and narrowed the README and index the same day. The thesis
is now centred on **acquiring robust procedures cheaply, and the boundaries of that acquisition**, with the interface
question settled by the crossed experiment rather than asserted.

## Revisits as results arrive

- **10:37 UTC, E60 arms MC and H.** These results support the red-team reading that the thesis is about
  *acquisition*. Told to use helper question parsing, the 4B proposer adopted it in 1 of 11 accepted libraries,
  and that library was the only one robust to an unseen rewording.
  - The interface works when present (E61, and this library), but a small proposer does not reliably produce it.
  - This weakens 'teach or instruct the small proposer to write robust interfaces'.
  - It strengthens two alternatives: supply the interface as a fixed harness component (a question-canonicalization
    or typed-parameter wrapper around frozen routines), or acquire routines with a stronger proposer.
  - Next test: a canonicalization wrapper around E39's frozen libraries, using E61's 4B parses. This is the
    end-to-end frozen-executor test both Codex reviews ranked first.
- **10:44 UTC, E68.** Supplying the question interface in the harness repairs E59 end to end: a 4B
  canonicalizer takes reworded questions from 3 to 54 of 55, matching the oracle; a 0.6B gets 22.
  - With E60 H, this splits the problem cleanly. Executors generalize, and a 4B can supply the semantic
    interface, but a 4B *proposer* does not write that interface itself.
  - The thesis centre moves further toward *acquisition*: who writes the interface and the executor, at what
    cost. The interface itself is the known semantic-parser-plus-executor pattern.
  - This is also a concrete datum for the crossed experiment's knowledge factor: a human-written interface plus
    acquired executors already works on these shifts.

## E70, the decisive crossed experiment: revised design after Codex's review (2026-10-24 ~11:00 UTC)

Review: `codex-reviews/2026-10-24-e70-design-review.md`. This is the plan for the next allocation; it is built on CPU
first.

- **Use the repository runtime, not smolagents, for the primary comparison.** smolagents' CodeAgent and
  ToolCallingAgent differ in prompts, history handling and a fallback answer call. Build an experiment adapter that
  decodes JSON tool calls into the same executor, helper ABI, observation encoder and accounting. The public
  Responses contract stays unchanged.
- **Define the treatment as a persistent REPL against scripts with explicit storage.**
  - Both arms see the same immutable request, MRCR message boundaries included.
  - Both keep large results through handles, and both have `submit(handle)`.
  - The same routine code sits behind both, through handles, with the same helper semantics (`ask_batch(...).texts`,
    system and options).
  - Both get the same truncation (6,000 characters) and the same history-retention rule.
  - Budgets are aggregate compute and helper calls, not turn caps.
- **Matched semantic demonstrations** are rendered into each action format, with no planning and no fallback answer
  call. The model revision, sampling and output limits are frozen.
- **Routine access.** Every arm uses the same input-only retriever, with no `task.family` metadata. Routine
  documentation is normalized across human-written and acquired routines, since one usage line can dominate (E25c-d).
- **Incomplete factorial.**

  | Task group | Arms | Sample |
  | --- | --- | --- |
  | OOLONG | JSON or REPL × primitives or human-written routines, plus host with human-written routines | all 293 rows (12 contexts; clustered uncertainty) |
  | LongBench-v2 | the same five arms | all 503 questions (462 contexts; clustered) |
  | Generated families | all six controller arms, plus host arms with human-written and with acquired routines | 30 families × 10 fresh instances × control, paraphrase 2 and nuisance 2 |
  | MRCR | adapter and ceiling check | 32–64 cases |

- **Failed acquisition** counts as zero robust yield, with no manual repair. Conditional utility is reported
  separately from yield.
- **Power.** At 25% discordance, about 200 independent pairs detect 10 points and about 400 detect 7. Generated
  families have only 30 independent units.
- **Budget.** About 100M processed tokens per compute cap: 18–55 GPU-hours for two caps on two slices. Start with one
  generous cap.
- **What each outcome licenses.** See the review's table. A null interface difference is inconclusive unless its
  interval excludes a prespecified meaningful advantage.

**Built on today's results.** E68 already gives the knowledge factor one datum for generated families: a
human-written interface around acquired executors is robust to the tested shifts.
- **11:12 UTC, E60 H2.** The adoption bottleneck replicates on a second 4B checkpoint: 0 of 18 accepted libraries.
  The acquisition framing is reinforced. Small proposers fit the shown templates even when told how to
  generalize. Open: does a stronger proposer (H8), or editing a working library (E69), change that?
- **12:13 UTC, E69 and E69b.** Editing helps a little: a working adopting edit for 3 of 12 families, against
  near zero when writing from scratch. Compare the hand-written wrapper: 54 of 55.
  - A 4B proposer is a poor source of robust interfaces; a 4B *parser* inside a supplied interface is a good one.
  - For the thesis: put the interface in the harness, and spend acquisition effort on executors, which do
    generalize. Or test whether the 8B proposer (H8) closes the gap.
- **12:35 UTC, E57.** From matched MRCR pools, three-label repair did at least as well as thirty-label repair (2 against 1
  of 5). This supports cheap acquisition on exact families, but descriptively only.
  - The new false acceptance is again a *length* gap (teaching at 8k to 32k, failure at the longest held-out
    inputs), as in E52b s0.
  - The two boundaries now on record: robustness to *wording* (E59) and to *length beyond teaching* (E52b, E57).
  - Both are about the teaching distribution, not the executor, and both point to the same acquisition rule:
    acceptance must cover the deployment distribution (lengths, wordings), or the interface must normalize it
    (E68).
- **13:16 UTC, E58 (pre-registered).** Arm A 75 against arm B 73 of 141, interval [-0.078, +0.106]: non-inferior at
  the 10-point margin with families as the unit. It is not robust to template clustering ([-0.113, +0.139]), it
  costs 10 false acceptances, and the weakest checkpoint does not show it.
  - This is the strongest single result for 'cheap acquisition' and supports the acquisition-centred thesis.
  - Within that thesis, the most consequential weakness of three-label acquisition is now *false acceptance*
    (10 of 85).
  - That raises the value of label-free checks that catch wrong libraries: accepted research question 3, and
    partition invariance.
- **13:37 UTC, E72.** A hand-written metamorphic check separates E58's wrong day-span libraries from the correct ones
  perfectly (3 of 3 against 0 of 22), with no labels.
  - This is the first evidence for research question 3: label-free checks can cut false acceptance.
  - The open, thesis-relevant version is automatic: can a small model derive invariances from the question
    (dates not times; renaming items keeps counts; duplicating records doubles sums) and run them at acceptance
    time?
  - Semantic totals (6 of 6 accepted were wrong in E58) need a different relation, such as helper
    classification consistency or invariance of category membership.
- **13:37 UTC, E58 diagnosis.** False acceptances have two causes: logic bugs, caught by invariances (E72), and
  unreliable neural leaves (semantic totals, 6 of 6). Three-label acceptance cannot detect leaf noise; thirty labels
  can.
  - This ties the acquisition thread to the System One leaf idea: a calibrated leaf, or a consistency check on
    the leaf, is what makes cheap acceptance safe for helper families.
  - Concrete next test: a leaf-consistency check (two phrasings, plus category exclusivity) as a label-free
    filter on E58 arm A's semantic-total libraries.
- **14:35 UTC, E54.** MRCR at a matched budget: 0 against 0 correct, plus a three-label false acceptance with the same
  held-out signature (0.786) as E52b s0. The same plausible-but-wrong algorithm keeps passing short teaching
  inputs.
  - This strengthens rank 4 in the resume doc: acceptance must span deployment lengths. A label-free version
    would run candidates on unlabelled long inputs and check output well-formedness or agreement between
    candidates.
- **15:12 UTC, E70 OOLONG reference.** A 4B with generic REPL tools scores 0.150, below direct reading (0.208), while
  the same runtime with the library scores 0.672.
  - For this model and task, the 'generic tools match the full system' outcome is clearly false: the leverage is
    in the procedures.
  - It does not yet separate REPL from tool-calling; that is E70's interface factor.
- **15:14 UTC, E73.** The semantic-total failures were the *leaf's question form*. A 4B is 97.9% right on yes/no and
  100% right on a typed five-way choice, so the task ceilings are 0.74 and 1.00. The yes/no errors are systematic
  (tools read as 'instruments').
  - This supports the user's System One point: typed choice decisions, as in Jev, beat free-standing binary
    questions.
  - It also says which label-free check can work: exclusivity across the choice set, not resampling.
  - For acquisition: proposers should be steered toward typed-choice leaves, or the harness should supply a typed
    `classify(item, options)` primitive. This is the leaf analogue of E68's question interface.
  - The 0.6B is not a usable leaf here (choice 0.87, ceiling 0.25).
- **15:36 UTC, E69b8.** The 8B editing: 7 of 12 working adopting edits, but only 3 robust; the 4B got 3 and 2. Who
  supplies a robust interface, out of 12 families (H8, the 8B writing from scratch, is still pending):
  - 4B writing: under 1;
  - 4B editing: 2;
  - 8B editing: 3;
  - hand-written wrapper with a 4B parser: about 12 (54 of 55 libraries).

  The harness-supplied interface dominates.
