# Research program and idea backlog (living; started 22 October 2026)

This page is the step-back companion to the plan of record. It states the big picture, organizes open
questions into pillars, and keeps a ranked backlog of ideas with their smallest informative test. It is
revisited at every plan revision, and new ideas, related work and follow-ups are added here first.

## The big picture

The evidence of 19 to 21 October reframed the thesis. Training a small controller to decompose long
inputs works, but every attempt to learn the *choice* of when to decompose collapsed, and putting the
routines in a verified, prebound library made the untrained controller as good as every trained one. A
0.6B caller distilled from a few hundred calls then matched the 4B. The emerging thesis:

> For small recursive language models, capability is best held outside the weights, in three kinds of
> memory: a **procedural memory** of verified, executable routines (the library); a **selection memory**
> of verified outcomes keyed by the input (the statistics table, or a learned key); and a thin
> **calling policy** in the weights, which is cheap to learn and transfers across model families.
> Training belongs where verification is local: in the proposer that writes routines, and in helpers
> whose per-item outputs can be checked.

That thesis is only as strong as the answer to "where do libraries come from", which is now the
central question, together with whether the rest of the RLM toolbox (recursion, harness search) adds
anything once routines and selection live outside the weights.

## Pillars of open questions

1. **Where libraries come from.** Bootstrapping a whole library from teaching examples (E21 MRCR, E22
   MuSiQue); proposer capacity (4B fails on MRCR, 8B partial so far); growing by verified repair;
   evolving routines rather than writing them once; training the proposer itself with verified reward.
2. **Selecting and composing.** Outcome tables and learned keys (E13), bandits (E10), recursion as one
   more selectable route (E23), multiple routines per signature on an accuracy and cost frontier.
3. **The caller and its prompt.** How small the caller can be (0.6B works, 360M misreads the note); what
   form the retrieved payload should take for a given caller (table, imperative, exemplar); evolving the
   harness text and configuration rather than hand-writing it.
4. **Where training still pays.** Helpers on verifiable leaves (E17 negative on ambiguous labels);
   proposers with verified reward; distillation of call discipline.
5. **External validity and cost.** Real benchmarks (OOLONG-real, MRCR, MuSiQue, LongBench-v2,
   BrowseComp-Plus), comparisons with frontier long-context agents on cost and size, and the honest
   classification of which benchmarks a short program solves outright (MRCR does).

## Idea backlog (ranked by information per GPU-hour and readiness)

| Id | Idea | Smallest informative test | Decision it informs |
| --- | --- | --- | --- |
| E24 | **Evolve the routine and harness configuration** (GEPA-style reflective search over the text and numeric parameters of a routine: filter and reader prompts, chunk size, retrieval depth, keep cap) | MuSiQue filter-then-read at 32k and 128k: an 8B reflector mutates a population on teaching minibatches (F1 and tokens as a Pareto front); best candidate on the 90 held-out questions against the hand-tuned reference | does automated harness search beat hand design, and at what search cost |
| E25 | **Evolve the retrieved payload format per caller size** (table of history, imperative cell, exemplar trajectory, docstring only, and LLM-mutated variants) | 360M, 0.6B and 4B callers on the event-log grown forms and MRCR; fitness = caller accuracy | does the injection format need to adapt to caller capacity (S5b says yes at 360M) |
| E26 | **Train the proposer, not the controller**: RLVR (or rejection-sampling SFT) on a procedural generator of many synthetic task families, reward = verified teaching score of the written library | 4B proposer, about 50 generated families for training; test bootstrapping on MRCR and MuSiQue, held out | can a small model learn to write libraries, the place where RL has a verifiable target |
| E27 | **Code routine vs natural-language routine executed by the helper** (ARISE-style ablation) | same families, same decomposition, routine as Python vs as an NL procedure the helper follows | is the gain from code, from decomposition, or from verification |
| E28 | **Reuse and compute-matched audit** of every library result (answering the library-learning critiques) | per-routine call counts; no-library baselines given the same helper-call budget (self-consistency, more chunks) | are library gains real once compute is matched |
| E29 | **Continual library learning in a stream** of mixed families, failures triggering the librarian, the table updating online | event-log, OOLONG, MRCR and MuSiQue in one shuffled stream; time to recover per new family | does the full system learn new families online without training |
| E30 | **Depth as a route**: recursion against a compute-matched self-consistency null (SRLM), chosen per signature by the table | E23 arms plus k-sample filter-then-read at matched tokens | does recursion add anything a cheaper search does not |
| E31 | **Frontier of routines per signature** (MAP-Elites over accuracy and cost) with the table choosing per input | keep several accepted routines per family; route by length | selection among cheap and thorough routines |
| E32 | **Consensus as verification** when no gold exists: accept a routine when independent routines agree | MuSiQue and OOLONG without labels | bootstrapping on unlabelled families |
| E33 | **Program-solvability of long-context benchmarks**: for each benchmark, the score of the best short program, the best frontier model, and our small RLM | MRCR (program 1.0), OOLONG (counting ceiling), MuSiQue | positions RLM results honestly against "a regex solves it" |
| E34 | **Combinator library plus learned composition** (λ-RLM primitives with the small caller composing them) | same families, compare with routine-per-family libraries | fixed primitives vs learned routines |
| E35 | **Library portability across helper models** (routines written against Qwen helpers run with other helpers) | swap helpers to Phi-4-mini or the 8B | is a library tied to its helper |

## Reflection log

- 2026-10-22 02:30: first entry. Running: E21 (MRCR bootstrap; the hand-written routine scores 1.000 at
  every length, so MRCR is program-solvable), E22 (MuSiQue bootstraps; reference filter-then-read 0.66 vs
  direct 0.32 on teaching at 32k, 0.56 at 128k where direct is impossible), E23 (recursion arms). Next
  ready: E24 (evolve the MuSiQue routine configuration), then E25 (payload format per caller size).
