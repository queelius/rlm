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

## Related work found at the 22 October scan, and what it changes

- **SMITH** (arXiv 2608.24571, Aug 2026) trains Qwen3-4B-Instruct with LoRA and DAPO (about 60 steps) to
  write tools from 4 examples, rewarded by the tool's pass rate on 16 held-out items; tools from the trained
  4B lift a frozen 350M model from 11.6 to 42.9 on unseen tasks. E26 is therefore not novel as "RL makes a
  small model a tool writer". What remains ours: libraries for long-context RLM families, including
  helper-dependent routines (MuSiQue, OOLONG) where the program alone cannot answer; the separation of
  reasoning from scale (E21b ladder); and transfer to real benchmarks through a distilled caller. First arm:
  rejection-sampling fine-tuning on the 8B's verified libraries for generated families (SOAR, arXiv
  2507.14172, does this for ARC), before any RL.
- **CASD** (arXiv 2609.26261) optimizes prompts with one offline pass of a coding agent over saved traces
  and beats GEPA on average at a fraction of the cost; **RLMOpt** (arXiv 2608.10471) runs the prompt search
  with an RLM. Both are comparison arms for E24. No work found on whether optimized prompts must differ by
  caller size (GEPA's smallest model was 8B), which E25 tests.
- **PARSER** (arXiv 2609.06702): parallel 4B chunk readers plus an RL-trained 4B lead over several rounds,
  85.4 on HotpotQA at 112k; our filter-then-read is the one-round version. Knobs worth adding to E22c:
  rounds and reader size. Coding agents (arXiv 2603.20432) found BM25 hurt on every benchmark tested, which
  E22c checks for our routine; tuned deeper BM25 matches dense retrieval on BrowseComp-Plus (arXiv
  2605.10848).
- **Recursion:** a reproduction (arXiv 2603.02615) finds depth-2 recursion overthinks and hurts on
  retrieval; a theory paper (arXiv 2609.20831) argues recursion helps out of distribution, not within it.
  Both agree with E23 so far (recursive library below depth-0 library).
- **MRCR:** no published critique showing MRCR is parser-solvable was found; registered as an idea in the
  idea lab.

## How the research workflow itself should improve

The costliest failures of 19 to 22 October were operational, not scientific: GPUs idle for 16.5 hours
between sessions and an hour at the end of an allocation; launches that failed silently (scrambled
arguments, 288 rows of connection errors from a stale readiness check); a train/serve chat-template
mismatch that voided a set of fine-tunes; kill patterns matching their own shell; ledger numbers typed
by hand. Changes, in order of value:

1. **A job queue that never lets a lane idle** (built: `experiments/ops/queue.py`). Ready jobs are JSON
   specs with a GPU lane, a priority and optional dependencies; a detached dispatcher starts them when a
   lane has a free slot, so follow-on work starts while no one is watching. Every session should end
   with the queue holding at least two decision-relevant jobs per lane.
2. **Checked launches** (built: `experiments/ops/launch.py`). Argument dry run, a manifest (command,
   commit, dirty flag, lanes), detached start, and an early health check that marks the run FAILED and
   stops it if the process dies or every early row is an error.
3. **One status view** (built: `experiments/ops/status.py`): rows, score per arm, errors and age for
   every recent run; monitors call it instead of bespoke one-liners.
4. **A results registry** (next): each ledger claim as a small JSON (claim, run directories, metric,
   comparison) from which a script recomputes the numbers, intervals and paired tests and renders the
   ledger table. No hand-copied numbers; every claim reproducible from its runs.
5. **Pre-registered experiment cards** (next): question, arms, primary metric, sample size and the
   decision rule written in the backlog before launch; the analysis states which branch fired.
6. **A declarative serving plan** (next): models, lanes, memory shares, ports, adapters and chat-template
   flags in one file, with a check that the shares fit and a canary request after start; training jobs
   read the same template flags, which closes the mismatch that voided the first small-caller fine-tunes.
7. **Review of generated code before it enters a library**: an independent reviewer (subagent or the
   8B) reads every accepted routine for side effects and leakage of teaching specifics (one 4B
   candidate hard-coded a teaching example's string) before registration.
8. **A live status page for the user**: a periodically republished artifact with the queue, running
   arms and latest ledger rows, so progress is visible without reading the terminal.
9. **An idea lab** (private repository `queelius/idea-lab`, run by its own Claude Code session) for
   questions noticed along the way, including ones outside RLMs, each with a prior-art verdict, the queries
   behind it and a smallest test. This session only appends to its inbox; RLM-relevant experiments it
   proposes come back as rows in the backlog above. GPU sharing: one priority-aware queue for both
   projects, lab jobs filling idle capacity, when the user allots lab time.
10. **Scheduled reflection**: at every plan revision and at least every 12 GPU-hours, a short entry in
   the reflection log below, an idea added or retired, and a related-work check.

## Reflection log

- 2026-10-22 02:10: first entry. Running: E21 (MRCR bootstrap; the hand-written routine scores 1.000 at
  every length, so MRCR is program-solvable), E22 (MuSiQue bootstraps; reference filter-then-read 0.66 vs
  direct 0.32 on teaching at 32k, 0.56 at 128k where direct is impossible), E23 (recursion arms). Next
  ready: E24 (evolve the MuSiQue routine configuration), then E25 (payload format per caller size).
- 2026-10-22 02:20: E23 first arm done: the paper-style depth-1 RLM without a library scores 0.254 F1 on
  MuSiQue (depth 0: 0.245; overflow at 128k in both), so recursion alone does not help an untrained 4B.
  Started E24 (routine evolution) and, through the new queue, E25 (does the distilled 0.6B caller call
  libraries of families it never saw: MRCR and MuSiQue reference libraries) and MuSiQue at 512k.
- 2026-10-22 02:25: E21 succeeded: the 8B wrote a reviewed MRCR library (1.000 held out to 1M tokens, 2 and
  4 needles) and the distilled 0.6B caller calls it for 1.000. Proposer capacity is the gate (4B 0/24).
  New questions this opens: (a) proposer scaling: the smallest proposer that succeeds, and with how many
  candidates (sample efficiency of library discovery as a metric); (b) E26 gets a concrete target:
  train the 4B proposer by rejection sampling on synthetic families until it bootstraps MRCR; (c) harder
  program-solvable variants (MRCR 8-needle, paraphrased asks) to find where a parser stops sufficing and
  helper calls become necessary; (d) libraries whose code depends on the helper (MuSiQue) are the real test
  of bootstrapping, since there the program alone cannot answer.
- 2026-10-22 06:20: step back after a dense night.
  **What now looks solid:** (1) a library beats reading once the input outgrows the window, and the gain is
  not bought by compute (E28: majority-of-3 sampling at 2.3 times the tokens does worse); (2) the calling
  policy is thin and transfers: a 0.6B caller distilled on two families calls libraries of families it never
  saw (MRCR 1.000, MuSiQue level with the 4B); (3) for an untrained caller, one usage line in the docstring
  decides whether a library is used at all (E25c/d, 0.18 to 0.95); (4) recursion, paper-style or per hop,
  does not help a small model on MuSiQue (E23); (5) library discovery is gated by the proposer: on 37
  generated families thinking 8B 73 percent, thinking 4B 44 percent, non-thinking 4B and 8B 11 to 14 percent;
  on MRCR only the thinking 8B, on 1 of 3 seeds.
  **What was withdrawn:** the "distractor-limited reader" reading and length routing (E31, pre-registered):
  30-question cells cannot separate close routines; per-length claims now need pooled samples.
  **What changed in method:** paired tests over all questions, pre-registration for selection claims, exit
  codes on every job, a reader-truncation check (fragments) on every QA routine.
  **New questions, ranked:** (a) can library writing be distilled into a non-thinking 4B (E26b, training
  now) and does it reach MRCR; (b) where should capacity go in a QA routine: the reader (gold-paragraph
  ceiling 0.63 with a 4B) or the selector (BM25 alone is as good as a 4B filter): an 8B-reader arm answers
  it; (c) does the generic map-reduce library carry to a heterogeneous benchmark (E32 LongBench-v2, running);
  (d) is the LLM reflector worth its cost over random search on the routine's numbers (E24c, running);
  (e) the MRCR shortcut (I-001) as a short benchmark note, since nobody has published it.
