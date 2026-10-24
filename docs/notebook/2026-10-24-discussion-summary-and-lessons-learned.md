# 24 October 2026: discussion summary and lessons learned

This note records a long working session between the student and the research assistant (Claude, with Codex as an
independent reviewer). It covers the experiments run that day, a broad brainstorm about what the thesis is, and the
lessons, both scientific and operational.

Companion pages:
- [decomposition brainstorm and prior art](../research-plans/2026-10-24-decomposition-brainstorm-and-prior-art.md),
  with references and the Codex reviews;
- [claims ledger](../RESULTS.md), rows E51r and E57 to E61;
- [E39 to E61 pre-registrations and addenda](assets-e39-preregistration.md).

## Part 1: what was discussed

### 1.1 Operations
- **Allocation and checkpointing.** Work continued on job 6618, ending 18:38 UTC. A resume handoff ("Start here")
  with a ranked "finding new work" list was written and pushed, so work can resume without this conversation. The
  student will get more compute later and asked that the work stay checkpointed.
- **The idle GPU.** GPU1 sat idle for about 70 minutes because the session ended a turn without a watch. A second
  8B server was started on GPU1, and E57 moved onto it.
- **Codex as a standing principle.** The student asked that Codex be used routinely as an independent second
  opinion, "to avoid monoculture". It reviewed designs before launch, analyses before unblinding, claims, and the
  big picture. Its reviews changed the work several times (Part 2).

### 1.2 Experiments run or launched
- **E51r** (32 samples, second boot seed). The 4B's best checkpoint gets 37 against the 8B's 29 of 60 (p 0.077);
  the weakest gets 32. Exploratory: these are the same 30 families, and the checkpoint was chosen after its outcome.
- **E58** (the powered, pre-registered non-inferiority test). Arm B was still running. Before unblinding, the
  analysis was amended: a completeness guard, input-token reporting, sensitivity analyses, and a corrected
  checkpoint rationale.
- **E59** (stress test of frozen libraries).
  - 55 of 55 survive inputs 5 times longer.
  - 3 of 55 survive a reworded question, and 11 of 55 a changed question prefix or field order.
  - Changing "QUESTION:" to "Q:" breaks 21 of 26 on prose and chat.
- **E60** (can bootstrapping produce robust libraries?).
  - Mixed-wording teaching and a prompt note both gave 0 of 30 robust libraries.
  - Mixed teaching cut acceptance from 12 to 4, and those 4 fit only the demonstrated renderings.
  - The note gave a suggestive gain on question-prefix and field-order shifts (7 against 2).
  - Arms MC (both), H (the helper parses the question) and H2 (H with a second checkpoint) were queued or running.
- **E61** (can a small model parse reworded questions?). Given the family's parameter schema, a 4B fills it from
  the question alone on 96% of unseen rewordings; a 0.6B on 58%. This is slot filling with a supplied schema, a
  diagnostic for E59, not a fix.
- **E57** (repair attribution with shared MRCR pools). Running on the second 8B server.

### 1.3 The big-picture conversation, in order
1. **Motivation and research questions.** Long-input capability from small, cheap, local models. Earlier evidence
   had moved capability out of the weights: library routines plus a selection table beat trained controllers, and
   RL on route choice collapsed.
2. **Decomposition.** The student described the core hypothesis: decompose a large problem into independent
   sub-problems, give each to a fresh LLM with only the context it needs, then recompose. They asked:
   - how to choose a decomposition (a library plus a router, try them all, a lookup keyed by the input, or an LLM
     that probes the input and writes a splitter);
   - how the question is carried into, or transformed for, each sub-problem;
   - which problems decompose;
   - whether a dataset of decomposition types would be a contribution.

   The framework we converged on (shared with Codex) treats decomposability as a property of a question–input pair:
   - merge algebra: distributive, algebraic or holistic aggregates, or a fold with carried state;
   - dependency topology: maps, folds, joins, multi-hop;
   - the segmentation unit;
   - evidence density;
   - the lifted sub-question each worker must answer;
   - leaf reliability.

   Prior art: RLM, λ-RLM, SRLM, Xu et al.'s noise decomposition (ICLR 2026), chunking studies, and the
   long-context benchmarks. A decomposition-type benchmark is new only if its labels are operational (a gold
   executable decomposition) and interventional.
3. **A learned probing policy (the student's end goal).** Train a model to probe the input (head, tail, sampled
   regions, statistics), infer its structure, choose a decomposition or none, and write the code. Our suggestion:
   - learn plan choice from verified outcomes (expert iteration), not RL from scratch;
   - order probes by cost: free statistics, then a few reads, then window classification.

   On the bitter lesson: search plus learning that scales with compute is compatible with it, while hand-written
   taxonomies are scaffolding.
4. **System One classifiers.** The student raised Jev (TypeSafe), a hosted calibrated classifier that answers typed
   questions with probabilities. Open local analogues exist: Simple Jev, Cloudflare Clef, GLiClass and NLI
   zero-shot models. There is also a large classical literature on inferring structure:
   - PADS, Datamaran and CleverCSV for data formats;
   - Drain for log templates;
   - TextTiling for topic boundaries;
   - FlashExtract (PLDI 2014) and Evaporate (VLDB 2023), which are direct precedents for bootstrapping libraries.

   Most neural leaf calls are decisions, so a calibrated classifier that escalates low-confidence cases is a
   plausible leaf.
5. **"Maybe it is not about recursion."** The student noted that frontier models in good harnesses do much of this
   with no sub-agents, by iterating on external state. Our evidence agrees (recursion and map-reduce did not help),
   and so do the RLM paper, SRLM and a reproduction.
6. **The REPL as a structured, reproducible harness.** Variables instead of files, functions instead of tool calls.
   - Zhang and Khattab's July 2026 harness post makes the same reframe: context offloading plus programmatic
     sub-calls, with each call "locally in-distribution".
   - The student added that the REPL is the natural environment for training too.

   Qualifications:
   - reproducibility is not free: the runtime has no deterministic execution or rollback;
   - unrestricted Python may be too large an action space for small models;
   - RL works where rewards are local and verifiable.
7. **"Is there any novelty?"** The student asked to be challenged.
   - **Our pushback:**
     - the mechanism is not novel;
     - the small-model regime is the real question, and generic scaffolds fail at 8B (0.29 on OOLONG);
     - novelty, if any, lies in amortized acquisition, its economics and limits, and measured principles;
     - beware breadth without closure, and the risk that libraries are per-family engineering.
   - **Codex, big-picture review:** the defensible claim is a measured split of competence between invoking,
     interpreting, selecting and constructing routines. It proposed a 2×2×2 factorial.
   - **Codex, red team, the harshest:** "you moved the solution into a library, then showed that a small model can
     invoke the solution." It listed prior art for almost every contribution, and named two missing critiques of
     library learning ([2410.20274](https://arxiv.org/abs/2410.20274),
     [2507.22069](https://arxiv.org/abs/2507.22069)). It flagged confused reframes: "locally in-distribution" is
     unmeasured, caller size is not system capability, E61 is slot filling, "verification" means testing, and the
     granularity law is unsupported. It proposed one decisive crossed experiment.
8. **Terminology.** "RLM" remains the community's term in 2026: the original paper, λ-RLM, SRLM, RL-trained RLMs,
   `dspy.RLM`. The mechanisms are "context offloading" and "sub-calls", and the broader frame is "harness". We keep
   "RLM" and define it as a harness, not a claim about recursion.

### 1.4 Where we landed
- **Thesis centre (provisional).** Acquiring robust procedures cheaply with small models, and the boundaries of
  that acquisition, measured against strong conventional baselines. The REPL interface question is settled by
  experiment, not asserted.
- **Decisive next experiment.**
  - Hold the 4B controller and helper fixed. Cross the interface (a conventional iterative harness or a REPL) with
    the source of procedural knowledge (generic primitives, human-written routines or model-acquired routines).
  - Add a host pipeline with no conversational controller, plus direct answering.
  - Tasks: MRCR, OOLONG, LongBench-v2 or MuSiQue, and shifted generated families.
  - Measure device-seconds, the amortized cost C(N), robust yield and false acceptance.
  - Its outcome decides whether the thesis is about the harness, about acquisition, or neither.
- **Accepted research questions (pick, do not do all).**
  1. Do cheap probes beat a fixed pipeline or a feature router, given measured oracle headroom?
  2. Can a 4B proposer acquire procedures that survive shifts and repay their cost?
  3. Do label-free checks reduce false acceptance at matched cost?
- **Gate.** No new directions until the crossed experiment has run and E58 and E60 H are analysed.

### 1.5 Open decisions for the student
- Whether to use Jev itself as an optional arm (it is closed and hosted; only synthetic data would be sent).
- Which accepted research question to prioritise after the crossed experiment.
- Whether the 0.6B or 4B "tiny caller" results stay as headline claims, given the critique that the caller's
  remaining job is easy.

## Part 2: lessons learned

### Scientific
1. **RL seed variance exceeds recipe effects.** Single-seed recipe claims (E36, E41, E42) were overturned by multiple
   seeds; one round-four seed reached 42, while four seeds averaged 31.5. Compare recipes over at least three
   seeds, and choose checkpoints by held-out discovery, never by training reward.
2. **Length generalization is not robustness.** Fresh-split evaluation kept each family's wording and surface
   settings, which hid template-boundness until E59. Every generalization claim needs an explicit shift.
3. **The question interface is the fragile part.** Libraries handled data at any length but not reworded questions.
   A 4B can fill a supplied schema from unseen wordings; a 0.6B cannot.
4. **Three labels suffice only within a template distribution with exact answers.** That is E37 and E39. Noisy
   families need tens of labels to pick a library (E49), and acceptance by three examples does not certify
   robustness (E59).
5. **Varied demonstrations did not teach abstraction to a 4B proposer.** Its accepted libraries fit the shown
   renderings (E60).
6. **Recursion and summaries were not the active ingredient.** Choosing evidence and executing code were.
7. **Cheapness claims must count all costs.** The three-label protocol's 7% figure counts proposer output tokens
   only; its input tokens are about 20 times the 8B's. Count helpers and the cost of acquiring routines too.
8. **The mechanisms are prior art.** Code actions, context offloading, skill libraries, programming by example and
   semantic operators all exist. Novelty must be a measured claim about small models, acquisition and boundaries,
   checked against strong conventional baselines and compute-matched.

### Statistical and reporting
9. **Pooling bootstrap seeds over the same families is not independent replication.** Outcome-informed follow-ups
   (E51, E52b, E55, E56) are exploratory and must keep their chronology.
10. **"Matches" requires non-inferiority.** Report observed counts, an interval and the margin.
11. **Words carry claims.**
    - "verified" means tested on specific examples;
    - "unseen" means new variants of the same generator;
    - "correct" means 0.9 or more on 30 tasks;
    - "robust" needs a named shift.
12. **Write predictions before results, with the real time from `date`.** Three timestamps were guessed and had to
    be corrected on 24 October.

### Independent review
13. **Codex reviews caught real flaws before they cost compute or credibility:**
    - E60's shown examples would all have been in the original wording, which voids the mixed-teaching arm;
    - nuisance 2 repeated nuisance 1 on four families;
    - E58's analysis would have scored unfinished families as failures, biased towards the arm that finished first;
    - the MRCR held-out scorer parsed rounded means and double-counted appended rows;
    - six overstated ledger claims, and overclaims in the README written that same day.

    Use review before launch, before unblinding, and whenever a claim changes.
14. **Ask for adversarial reviews explicitly.** The red-team prompt ("most skeptical committee member") produced
    the most useful critique of the day.

### Operations
15. **Never end a turn without a watch** on the queue log that covers started, failed, slow and idle-alarm lines.
16. **When a lane's last job will finish before its follow-on can start,** move the follow-on or swap the server
    first. Keep two follow-ons per lane.
17. **Wait on a known PID, never `pgrep -f`** with a pattern that also appears in the waiting command; it matches
    itself and never exits.
18. **Analysis code must refuse incomplete or duplicate data** rather than count it as failure. Scorers must never
    reuse output files: `verify` now deletes stale output, and the MRCR scorer refuses to reuse a work directory.
19. **Do not edit a live job's source.** Additive, gated changes for new families are acceptable; note them.
20. **Checkpoint for a reader with no memory of the session:** a "Start here" handoff, a machine-readable manifest,
    and pushed docs.

### Process
21. **Breadth without closure is a risk.** The morning produced six new framings (probing, System One leaves,
    harness, granularity scaling, trajectory surgery, trajectory-to-routine lifting) and none has run yet. Gate new
    directions behind the decisive experiment.
22. **Reframes are hypotheses until operationalized.** "Locally in-distribution" and "finer decomposition for
    smaller models" need measurable definitions and confound-free designs before they can explain anything.
