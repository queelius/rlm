# Research program and idea backlog (living; started 22 October 2026)

This page is the step-back companion to the plan of record. It states the big picture, organizes open
questions into pillars, and keeps a ranked backlog of ideas with their smallest informative test. It is
revisited at every plan revision, and new ideas, related work and follow-ups are added here first.

## The big picture

**Revision, 24 October 2026.** The thesis below still holds, but its emphasis has moved:
- **Recursion is not the active ingredient.** It and map-reduce summaries did not help here; the RLM paper runs
  at depth 1; SRLM and a 2026 reproduction agree.
- **The working frame is harness design for small models.** Context is offloaded into a REPL and code calls
  the model only for small, typed decisions that stay locally in distribution. The question is how to
  allocate competence among weights, verified routines and selection memory as the model shrinks.
- **New evidence locates the fragile interface in question interpretation:** E59 to E61.
- **Two experiments must come first.** The positioning test against a plain tool harness (E64), and a
  factorial separating the harness, the library content and the model's own skill.

Details, prior art and the challenges raised are in
[2026-10-24-decomposition-brainstorm-and-prior-art.md](2026-10-24-decomposition-brainstorm-and-prior-art.md).

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

Note on ids: backlog ids E24 to E35 were assigned when this page was written; ledger rows E31, E32 and E35 in
`docs/RESULTS.md` later reused those numbers for different experiments (routing by length; LongBench-v2; OOLONG
bootstrapping). Backlog E31 and E35 below are still untested ideas; backlog E32 runs as ledger E37.

| Id | Idea | Smallest informative test | Decision it informs |
| --- | --- | --- | --- |
| E24 | **Evolve the routine and harness configuration** (GEPA-style reflective search over the text and numeric parameters of a routine: filter and reader prompts, chunk size, retrieval depth, keep cap) **Status: done: E24, E24c (numbers helped weakly, prompt rewrites hurt, random search nothing).** | MuSiQue filter-then-read at 32k and 128k: an 8B reflector mutates a population on teaching minibatches (F1 and tokens as a Pareto front); best candidate on the 90 held-out questions against the hand-tuned reference | does automated harness search beat hand design, and at what search cost |
| E25 | **Evolve the retrieved payload format per caller size** (table of history, imperative cell, exemplar trajectory, docstring only, and LLM-mutated variants) **Status: partly done: E25c, E25d (usage line), E25e.** | 360M, 0.6B and 4B callers on the event-log grown forms and MRCR; fitness = caller accuracy | does the injection format need to adapt to caller capacity (S5b says yes at 360M) |
| E26 | **Train the proposer, not the controller**: RLVR (or rejection-sampling SFT) on a procedural generator of many synthetic task families, reward = verified teaching score of the written library **Status: done: E26b to E26g (SFT then RL works on generated families; round four 42 of 74 held out, not distinguishable from the thinking 8B teacher, p 0.66; not MRCR; helper use the gap left). Round five (ledger E36, more helper-family teacher data from the 8B and 14B) running.** | 4B proposer, about 50 generated families for training; test bootstrapping on MRCR and MuSiQue, held out | can a small model learn to write libraries, the place where RL has a verifiable target |
| E27 | **Code routine vs natural-language routine executed by the helper** (ARISE-style ablation) | same families, same decomposition, routine as Python vs as an NL procedure the helper follows | is the gain from code, from decomposition, or from verification |
| E28 | **Reuse and compute-matched audit** of every library result (answering the library-learning critiques) **Status: partly done: E28 (sampling null at 32k).** | per-routine call counts; no-library baselines given the same helper-call budget (self-consistency, more chunks) | are library gains real once compute is matched |
| E29 | **Continual library learning in a stream** of mixed families, failures triggering the librarian, the table updating online | event-log, OOLONG, MRCR and MuSiQue in one shuffled stream; time to recover per new family | does the full system learn new families online without training |
| E30 | **Depth as a route**: recursion against a compute-matched self-consistency null (SRLM), chosen per signature by the table **Status: done: E23 (recursion does not help).** | E23 arms plus k-sample filter-then-read at matched tokens | does recursion add anything a cheaper search does not |
| E31 | **Frontier of routines per signature** (MAP-Elites over accuracy and cost) with the table choosing per input | keep several accepted routines per family; route by length | selection among cheap and thorough routines |
| E32 | **Consensus as verification** when no gold exists: accept a routine when independent routines agree **Status: running as ledger E37 (8 round-0 candidates per held-out family, no early stopping; exact-agreement and MBR-exec rules). Prior art for program selection: MBR-exec, AlphaCode clustering, CodeT; the open part is library acceptance on unlabelled long-context families.** | MuSiQue and OOLONG without labels | bootstrapping on unlabelled families |
| E33 | **Program-solvability of long-context benchmarks**: for each benchmark, the score of the best short program, the best frontier model, and our small RLM **Status: MRCR shown program-solvable (E21); audit of other benchmarks in the idea lab (I-001). OOLONG: no proposer bootstraps it (ledger E35: 30 rows span 11 question templates); per-task-group families running (E35c).** | MRCR (program 1.0), OOLONG (counting ceiling), MuSiQue | positions RLM results honestly against "a regex solves it" |
| E34 | **Combinator library plus learned composition** (λ-RLM primitives with the small caller composing them) **Status: done: E34 (primitives change what the 8B tries; composition hit or miss).** | same families, compare with routine-per-family libraries | fixed primitives vs learned routines |
| E35 | **Library portability across helper models** (routines written against Qwen helpers run with other helpers) | swap helpers to Phi-4-mini or the 8B | is a library tied to its helper |
| I-004 | **Keep input-conditioned routes through RL** (proposed by the lab session): the 13 route-RL runs of E06/E07 lost the warm start's split (direct on short, decompose on long) in every run without strong KL. Stored advantages show why: direct was positive on short inputs but never sampled on long ones, so it leaked there unopposed, failed, drew zero advantage in all-direct groups (leave-one-out baseline), then bled back and vanished everywhere. Forced per-route groups are prior art (CounterRoute, arXiv 2609.29109); the mechanism and a cheap variant are not | rl-s7-strat setup, 30 steps, 2 seeds: null vs forced route per group vs forced route only in unanimous groups vs KL 1; metric: short-minus-long direct share over the last 5 steps; about 64 GPU-h | whether route RL needs counterfactual samples where the warm start has zero mass |
| I-019 | **Train on sampled ids and keep sampler log-probabilities** (proposed by the lab session): `proposer_rl.py` re-tokenizes `/v1/responses` text (on R4 rollouts 12.8% of turns and 0.44% of tokens change), and E06/E07 stored no sampler log-probabilities | sample through `/v1/completions` with `logprobs` and `return_tokens_as_token_ids`; until then, count turns that fail to round-trip in the next E26 run | whether E26 updates are on-policy; any later collapse becomes diagnosable offline |
| I-018 | **Is the distilled caller's gain carried by a few decisions?** (proposed by the lab session): the untrained 0.6B decodes and the distilled 0.6B's token is taken only where their greedy choices differ (the math version, arXiv 2610.24663, recovers 90.5% of an RL gain from about 4% of positions; no agentic version found) | 200 MRCR and event-log tasks: untrained, distilled, lockstep swap, random-swap control; recovered fraction and the class of swapped positions (cell start, library call, chunk literals); about 8 GPU-h | what caller distillation actually teaches: a few call decisions or broad code skill |
| I-007 | **Lane-idle alarm and heartbeats** on top of the queue (proposed by the lab session): 38.8 of 141.8 reserved GPU-h (27%) idle on 19 to 22 October, 90% from one pause with nothing queued | alarm when a lane has run nothing for 10 minutes; long jobs write a heartbeat file; the lab's GPU sampler measures the idle share after the queue | whether the queue closes the idle gap |

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
  on MRCR only the thinking 8B, on 1 of 3 seeds (2 of 5 after two more seeds, 12:56).
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
- 2026-10-22 11:00: step back.
  **New solid results:** two-round retrieval is the MuSiQue library (E22f/h replicated; E25e through both callers;
  512k at 8k tokens per question); with it, caller size stops mattering (untrained 4B = distilled 0.6B).
  Distilling the 8B's libraries into the 4B transfers parsing to unseen string formats (1 to 13 of 42, p 0.0005)
  but not to MRCR (round two with chat data: best 0.214, 0 of 3 seeds). The 8B never invents retrieval for
  MuSiQue (pass-through wins even on mixed-length teaching).
  **Boundaries found:** LongBench-v2 (generic library ties direct reading); semantic families capped by the
  helper; routing by length is noise at 30 per cell.
  **Method lessons that cost the most today:** scripted harnesses must mirror the kernel's fault semantics;
  verification sets must span deployment lengths.
  **Where the thesis now stands:** capability in verified routines; capacity in evidence selection and in the
  proposer; the proposer is the bottleneck. Two levers on the proposer remain untested: giving it parts (E34,
  running: retrieval primitives in the contract) and training it with the verified teaching score as reward
  (E26c, next: RL on generated training families from the round-two SFT adapter, SMITH-style; held-out families
  and MRCR as tests). Allocation ends 2026-10-25 00:52 UTC, so E26c is the last large build of this allocation.
- 2026-10-23 05:40: step back.
  **New solid results:** proposer training works as far as code goes: the round-four 4B (more varied helper-family
  teacher data, then RL on sampled ids) solves 42 of 74 held-out family attempts, not distinguishable from its
  thinking 8B teacher (45; +9/-12, p 0.66) at a twenty-fourth of the tokens. Scale does not make MRCR discovery
  reliable: the 14B also succeeds on 2 of 5 seeds (both libraries 1.000 held out to 1M tokens).
  **Boundaries found:** helper-dependent families (8B 5 of 8 semantic families at 0.7, round-four 4B 1 of 8);
  OOLONG, the study's own family, is not bootstrapped by any proposer (8B 0.20, 14B 0.00 against the shipped
  library's 0.60 on the same teaching rows), and giving the shipped parsing and classification as primitives does
  not fix it (0.13 to 0.17): one library must cover 11 question templates from 30 examples.
  **Running, each with its decision:** (1) round five (E36): does more helper-family teacher data close the helper
  gap (semantic families at 0.7 or more, held out, against round four's 1 of 8)? (2) consensus (E37): can
  agreement among independent candidates replace gold labels when accepting a library (precision near 1 at a
  useful recall would make bootstrapping on unlabelled families possible)? (3) OOLONG by task group (E35c): is
  breadth the barrier (a group library reaching its group's reference score)?
  **Method lessons:** a "dry run" must never execute a shell pipeline (the launcher ran one for 120 s; fixed);
  early-stopped runs cannot answer agreement questions (all candidates after the first success are missing).
- 2026-10-23 10:50: re-rank after a morning of results.
  **New:** verification is cheap where answers are exact (E37: three shown examples plus a label-free validity
  check accept libraries that generalize to fresh, longer inputs 127 times in 129; MRCR exactly the 4 correct of 646)
  and expensive where they are noisy (MuSiQue: tens of labels to rank close routines; OOLONG counting: a library that
  cleared 30 teaching rows was worse held out, p 0.023). Agreement is no substitute (RL-trained proposers repeat their
  wrong programs; sent to the idea lab). Sampling does not reach the helper families (E38: 1 of 8 at 32 samples) and a
  worked example makes the 4B call the helper without adapting the pattern (E38b). OOLONG narrowed to one task group
  is bootstrappable where its questions are simple (14B counting library level with the shipped one held out), not
  where they involve date windows (timeline 0).
  **Running, ranked by what they decide:** (1) round five (E36): RL reward on helper families 0.36 and 0.48 in the
  first two blocks against round four's 0.06 to 0.17, so the extra helper libraries were learned; held-out discovery
  decides whether they transfer. (2) E39 (pre-registered): best-of-8 4B with three labels against the 8B with 30 on 30
  unseen families. (3) the remaining 8B OOLONG group seeds.
  **Next, conditional:** if round five transfers, replicate it with a second RL seed and run it in E39 as a third arm;
  if not, the helper gap is generalization across questions, and the next lever is more helper question types in
  family_gen (only semantic_count and semantic_top_user train; semantic_total is held out).
- 2026-10-23 18:30: pause (job 6599 cancelled early for new preemptible GPU access; resume steps in
  `docs/RESEARCH_RESUME_2026-10-24.md`). Since 10:50: E39 replicated (pooled 26 vs 29 of 60 at 7 percent of the
  tokens; three-label bootstrapping holds); more helper-family teacher data trades held-out code for helper skill and
  then stops helping (E36, E41, E42: code 42, 34, 28 of 58 at 20, 65, 113 helper pairs; training reward rises while
  held-out success falls); RL seed variance is about four families of 74, so RL recipes need two or more seeds.
  **Re-ranked queue for the next allocation:** (1) finish the pending E43 A evaluations and E39 arm C (cheap, queued);
  (2) helper gap with varied helper questions per pair rather than more pairs, two RL seeds per recipe; (3) graded
  verification and label needs on noisy families; (4) automatic splitting of broad families into templates;
  (5) the program-solvability note (MRCR) and idea-lab I-018 and I-004. With four GPUs on separate nodes, run one
  server set and dispatcher per node.
