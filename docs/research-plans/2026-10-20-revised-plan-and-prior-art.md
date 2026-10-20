# Research plan of record (living document; revised 2026-10-20 10:20 UTC)

**Rule: revisit this plan periodically.** At every completed or failed arm, at every session start
or resumption, and at least once per 12 hours of GPU time, re-rank the queue below against the
newest evidence in `docs/RESULTS.md`, retire what stopped producing information, add what the
evidence or new prior art suggests, and append a line to the revision log. The queue is ordered
by expected information per GPU-hour and readiness, not by when an idea arrived.

Revision log:
- 2026-10-20 10:20: library thread promoted to the lead. L1-b/c/d: the *untrained* 4B controller with
  a prebound library matches the taught and RL controllers on both families (event-log 85/96 and
  23/24; OOLONG 43/72 against 36 and 41); L4: teaching the call pattern (SFT-009) adds only the
  cheap lookup route; L3a-lite: a note rendered from per-signature outcome statistics makes the
  untrained controller take the cheap route with no training (first 36 rows). E07-9 closed
  (all-direct by step 5). New: library growth E08 (held-out question types, verifying librarian),
  honest-library fallback arm, OOLONG route statistics. RL routing arms (balance, KL) now
  confirmatory only.
- 2026-10-20 06:30: first revision after the routing experiments; prior-art pass (routing collapse,
  length-weighting impossibility, When2Tool, Prime Intellect RLM post, SRLM, entropy collapse).
- 2026-10-20 06:45: distillation and model-capacity analysis added.
- 2026-10-20 06:05: harness-side learning (retrieval, GEPA, Meta-Harness) and role adapters added.
- 2026-10-20 06:15: queues merged into one ranked list with status; revisit rule stated.
- 2026-10-20 08:10: E07-9: RL from the correct router loses the rule within two updates; library and probe routing promoted; kernel prelude implemented, prompt-side library negative (L1-a).
- 2026-10-20 07:40: exemplar-payload variant (H1') and the router representation ladder (L3a to L3c) added.
- 2026-10-20 07:20: SFT-008 shows the rule is representable; RL from it queued ahead of balance and KL; stratified lanes at steps 3 to 4 flip toward always-direct (interim).
- 2026-10-20 06:45: library-learning thread (L1 to L4) added from the user's RAG proposal.
- 2026-10-20 06:28: cyclic task-sampling artifact found (short lengths absent for several steps at a
  time); confounded arms stopped; remedies requeued under stratified sampling in two lanes.

## Ranked queue, merged (status as of 10:20 UTC)

| Rank | Id | Arm | Status | Decision it informs |
| --- | --- | --- | --- | --- |
| done | L1-b/c/d | library prebound through the kernel prelude: scripted caller, untrained 4B, OOLONG | 87/96 oracle; untrained 85/96 and 23/24 (event-log), 43/72 (OOLONG) with no errors; see RESULTS L1-* | the routine belongs in the library, not the weights |
| done | L4 | controller taught to call the library (SFT-009) | 85/96, 24/24; takes the cheap lookup route; 4x tokens on short tasks | teaching buys the route choice only |
| running | L3a-lite | per-signature outcome statistics rendered as a harness note (`skills/stats.py`, `--stats-file`) | `dev-lib-base-stats0` 36/36 so far, lookups at 1k tokens (17k without); `teach-lib-base-stats` collects the controller's own statistics for v1 | route choice without gradients; the table is the router |
| running | E08 | library growth: three held-out question types; untrained controller fails them with the shipped library (0/22, submits the library's "unknown"); librarian proposes, verifies on cached teaching extractions, registers; before/after on dev | chains `e08_growth_gpu1.sh`, `e08b_growth_gpu1.sh`; honest-library fallback arm `e08-dev-heldout-honest` | does verified growth recover a new question type without training; does an honest library make the controller fall back to its own program |
| 1 | L3b | OOLONG route statistics (library vs direct vs own) and the note on OOLONG; 293-row held-out run with the library | needs OOLONG teaching-split rows for the library route | second family for the statistics router; the held-out number for the paper |
| 2 | E08-hard | a harder held-out family (joins, multi-item) and an 8B proposer for the librarian | after E08 | where the 4B librarian fails and verification matters |
| 3 | L3c | contextual bandit over the table (Thompson sampling with exploration) replacing the greedy note; measure regret on a stream of mixed tasks | after L3a-lite | online adaptation without weights |
| stopped | G-short, G-seq | short-weighted and sequence-normalized arms under cyclic sampling | stopped at step 5: confounded by the burst artifact | superseded by pipeline 20 |
| running | G-strat (lane A) | stratified sampling + relative cost + seq-norm, then + balance term, then + strong KL | `rl-s7-strat*`, pipeline 20 | does even per-step coverage of short tasks hold the route; then the two remedies on a clean footing |
| running | G-strat-short (lane B) | stratified + short-weighted, then + balance | pipeline 20 | signal share on a clean footing |
| done | SFT-008 | consistent teacher (direct only at 8 records) | routes by length as taught (38/40 direct at 8 records, 31/32 decompose at 32) | length-conditional routing is representable; RL failures are dynamics |
| stopped | G-s8 | RL from SFT-008 under relative cost, seq-norm, stratified sampling (`rl-s8-strat`) | all-direct by step 5 (dev 0.87 to 0.304); stopped, adapter kept (E07-9) | RL does not keep a correct routing; routing moves outside the weights |
| confirmatory | G-strat, G-bal, G-kl5, G-short-bal | stratified lanes and their balance and KL remedies (pipelines 21, 22) | running on GPU 0; expected to confirm collapse or a forced balance at an accuracy cost | last word on gradient routing in this setup |
| 4 | H1' | exemplar payload (cells of a matching teaching trajectory) vs the library | ready | what verification buys over imitation |
| 5 | P1 | probe routing (When2Tool-style) | needs hidden-state dump | representability; only if the signature table proves insufficient |
| 4 | D1 | stronger-controller teaching with local helpers (rlm-qwen3-8b, Qwen3-8B) | needs 8B serving slot | does a teacher's routing judgment transfer by SFT |
| 5 | C1 | capacity: 8B controller, 4B helpers, E06 recipe | needs vLLM restart | gap and gain vs capacity |
| 6 | H3 | GEPA over the harness hint text | ready after H1 | harness evolution vs RL |
| 7 | D2 | on-policy distillation from a local teacher | after D1 | dense signal without collapse |
| 8 | X1 | OOLONG-real transfer of the best 4B controllers | license check first | external validity |
| 9 | H4 | role adapters (controller + helper co-training) | after G-bal | revisits C07/C08 |
| 10 | X2 | 64k to 128k OOLONG-synth for the RL controllers | ready | budget-safe chunking at length |
| 11 | G-dec | decoupled route-token loss (Thinkless-style) | only if G-bal fails | last RL remedy |
| 12 | G-alpha | length-weighting sweep (alpha = 0.5) | after G-seq | fills the tradeoff curve |
| retired | absolute cost 0.3 from SFT-007; weak KL; more seeds beyond two per arm | | |

## What the last 24 hours established

1. **Teaching plus RLVR works for the decomposition routine** (E06, replicated across four RL
   runs): 0.567 to 0.63-0.65 on held-out OOLONG-synth, driven by a length-generalizing fix of helper
   budgets at 32k tokens; the cost term is nearly inert at the strengths tried.
2. **Learned selective recursion fails by routing collapse** (E07): from a warm start that holds
   both routes, on-policy RL removes the direct route within three to five updates under an
   absolute token cost, under a group-relative cost (two seeds), and under a weak KL penalty. The
   collapse happens even when the net advantage on direct rollouts is near zero, which points at
   token-sum loss weighting and at the rarity of short tasks in the draw.

## Prior art that bears on this directly (searched 2026-10-20)

- **Routing collapse is a named phenomenon.** Kassenaar, Yang and François-Lavet, *Learning When
  to Think: Adaptive Reasoning for Test-Time Compute Allocation* (arXiv:2608.20256, 2026) show that
  a single routing token under policy gradient collapses to one mode within 30 to 50 steps, that
  all-one-mode states are stable while mixed splits are not, and that the effect is stronger on
  small distilled models with low output entropy. Their remedies: an outcome-gated **balance term**
  on advantages, A_i <- A_i + beta_bal (p* - f_mode(i)) with p* the target share and f the batch
  share of that mode (positive nudges only to correct rollouts, negative only to incorrect, annealed
  to zero late), **per-mode token caps**, a forced-rollout warmup, token-mean loss aggregation, no
  standard-deviation normalization, and no KL. On a 1.5B model this held a 20/32/47 mode split with
  a 41 percent length cut on MATH-500 at a 1.4-point accuracy cost. Thinkless (Fang, Ma, Wang,
  arXiv:2505.13379) decouples the control-token loss from the response loss for the same reason;
  AutoThink (arXiv:2505.10832, NeurIPS 2025) enforces a batch-level target thinking ratio.
- **Length weighting cannot be made innocent.** Ding et al., *On the Impossibility of Unbiased and
  Length-Invariant Policy Optimization with Outcome Rewards* (arXiv:2607.23364, 2026): no
  length-based weighting is both gradient-unbiased and length-invariant; Dr. GRPO-style constant
  normalization (our token-sum loss) lets longer trajectories dominate updates in proportion to
  the length ratio, which is the mechanism we suspected for the seed-1 collapse. Our
  sequence-normalized arm sits at the other end of their f_alpha(L) = L^(alpha-1) family.
- **Models may already know when a call is unnecessary.** Sun et al., *LLM Agents Already Know
  When to Call Tools* (When2Tool, arXiv:2605.09252, 2026): tool necessity is linearly decodable
  from the pre-generation hidden state (AUROC 0.89 to 0.96) far better than the model's own
  verbalized decision; a probe plus a prefilled steering sentence cuts tool calls 48 percent at
  1.7 points of accuracy. No RL baseline. This suggests a non-RL route for selective recursion:
  probe the controller's state after the peek and inject the decision through the harness.
- **RL training of RLM controllers is an acknowledged open problem.** Prime Intellect's post
  (January 2026) evaluates API models in an RLM environment (verifiers `RLMEnv`, Oolong-real,
  DeepDive, verbatim-copy) and states that training models to use the RLM, starting with small
  ones, is the next step; no RL runs are reported. The RLM paper (Zhang, Kraska, Khattab,
  arXiv:2512.24601, NeurIPS 2026) post-trains RLM-Qwen3-8B and mentions RLVR on a small split.
  Alizadeh et al., *Recursive Language Models Meet Uncertainty* (arXiv:2603.15653, 2026) add
  inference-time self-reflective program search (+22 percent over RLM at equal time) on
  OOLONG-synth from 1k to 8M tokens, inference only.
- **Entropy collapse remedies** (STEER arXiv:2510.10150; OPEFO arXiv:2605.11491; Anchored Policy
  Optimization arXiv:2602.05717; Pass@k training) target token-level or entropy-flow control
  rather than global entropy bonuses. Our weak-KL result agrees that a global per-token penalty
  cannot protect a few decision tokens.

Our position in this landscape: the only study we found that trains a small RLM controller with
RLVR, and the first to show that the think/no-think routing collapse carries over to
decompose/answer-directly routing, with a gradient-source analysis. The remedies above are the
natural next arms.

## Ranked queue (each: question, smallest informative design, GPU shape, decision rule)

1. **Balance-term RL (Kassenaar-style) from SFT-007.** Does an outcome-gated balance toward a
   global direct share (p* = 0.3, beta_bal = 1, annealed over the last third) keep the direct
   route alive, and does the policy then allocate it to the lengths where it is cheap and
   correct? Relative cost 0.5, sequence-normalized loss, no KL. Two seeds, 30 updates each,
   about 2.5 h per run in the concurrent slot. Decision: direct share at 8 records at temperature
   1 after training, direct share at 32 records and OOLONG 2k, held-out accuracy and tokens.
   Implemented as `--balance-target/--balance-coef` in `rl_train.py`.
2. **Probe routing without RL (When2Tool-style).** Extract the controller's hidden state after the
   peek observation for the teaching rollouts, fit a linear probe for "direct answer would be
   correct", and route at inference through the harness hint. Oracle-router upper bound and a
   training-free baseline for the paper. About 1 GPU-hour. Decision: probe AUROC by length; routed
   accuracy and tokens against always-decompose.
3. **Consistent-teacher SFT-008.** Teach direct only at 8 records and decomposition elsewhere;
   test at temperature 0 and 1 whether SFT alone yields length-conditional routing. 30 minutes plus
   evaluations. Decision: if SFT cannot represent the rule, routing is a representation problem,
   not only an exploration one.
4. **Decoupled route loss (Thinkless-style).** Weight the tokens of the route-deciding line
   separately from the rest of the cell. Medium implementation effort; run only if 1 fails.
5. **Length-weighting sweep.** alpha = 0.5 between token-sum and sequence-mean, after the two
   endpoints have reported.
6. **Capacity: Qwen3-8B controller.** Repeat E06's teaching, SFT and one RL run at 8B (the RLM
   paper's own post-trained size). Needs a vLLM restart with the 8B model, so it is scheduled after
   the 4B routing arms end; about 7 GPU-hours end to end. Decision: expert/SFT gap and RL gain at 8B.
7. **External validity: OOLONG-real.** Evaluate the best 4B controllers on Oolong-real after
   checking its license; about 1 to 2 GPU-hours. Decision: does the synth-trained routine transfer.
8. **Longer contexts.** 64k to 128k OOLONG-synth rows for the RL controllers, where budget-safe
   chunking matters most.

Retired or deprioritized: more absolute-cost strengths (inert), weak KL (inert), the single-model
seed studies beyond two seeds per arm, and breadth across new task families before the routing
story is settled.

## Data quality and distillation (added 06:45 UTC after the user's question)

What we did before: all teaching came from a deterministic scripted expert run through the real
kernel (native replay), filtered to correct trajectories. That gave a warm start that matches the
expert in distribution on event logs but sits 10 points below it on OOLONG (0.567 vs 0.672), and
it carries no routing judgment at all; the earlier Codex-era self-SFT control (ledger C32) and the
TextCraft teaching work (C79 to C81) are the other data-quality experiments. The RLM paper
post-trained RLM-Qwen3-8B on filtered demonstration trajectories (the public abstract gives no
count, teacher or filter), and Prime Intellect names training small models to use the RLM as the
open problem. Relevant method: Thinking Machines' *on-policy distillation* (October 2025): the
student samples, a teacher scores every sampled token with a reverse KL, so the signal is dense
and on the student's own states; a Qwen3-8B student went from 60 to 70 percent on AIME'24 in about
150 steps at a claimed 9 to 30 times less compute than SFT; follow-ups (arXiv:2604.13016) report it
depends on student and teacher sharing compatible patterns.

Options, ranked by expected information per GPU-hour and cost:

1. **Stronger-controller teaching with local helpers.** Drive our kernel with a stronger model as
   controller while the helper calls still go to the local 4B, so the student later sees the
   same helper behaviour it was taught against. Teachers available without an API: the RLM paper's
   own post-trained `mit-oasys/rlm-qwen3-8b-v0.1` (local, MIT; trained for a different scaffold,
   so it must first be checked for compatibility with our ABI), Qwen3-8B, and re-downloadable
   Qwen3-14B (bf16 fits one 40 GB card for inference) or Qwen3-32B-AWQ (4-bit, about 18 GB,
   inference only). An API teacher (Claude or GPT class) would cost roughly 15k controller tokens
   per trajectory, so about 1,000 trajectories for 15M tokens, tens of dollars at mid-tier
   pricing; there is no API key on the cluster today. The point of a strong teacher is not only
   program quality: a strong model decides *when to answer directly* on its own, so its filtered
   traces contain the routing judgment that RL could not discover. Filter by correctness, cost and
   program brevity (our teaching lesson: short composed programs reproduce, long ones drift).
2. **On-policy distillation with a local teacher.** Student samples through the kernel; the
   teacher (8B or 14B served on the second GPU) scores the student's cells per token; train on
   reverse KL. Dense signal without the collapse dynamics of outcome rewards, and the routing
   token gets a direct teacher signal. Implementation: a second logprob pass with the teacher's
   weights, which the trainer's KL code path already resembles. Needs the teacher to be competent
   on our tasks first (item 1 measures that).
3. **Rejection-sampled self-improvement (STaR/RFT) from the RL controllers.** Sample the 0.65
   controller many times, keep correct-and-cheap trajectories, SFT the warm start on them. Cheap,
   but the September self-SFT control (C32) was inconclusive; run only as a baseline for 1 and 2.
4. **Reasoning in traces.** Our kernel takes one Python cell per turn and shows only the last
   pair, so long reasoning per turn is not retained; "high-quality reasoning traces" here means
   brief rationale comments plus the decision itself. A teacher's hidden reasoning can still shape
   the cell it writes; what we distill is the cell.

Decision rule for the whole thread: a distilled warm start must beat SFT-007 on held-out OOLONG and
event-log accuracy at equal or lower tokens *and* keep a nonzero direct share on short inputs at
temperature 0; otherwise the scripted expert remains the better teacher per dollar.

## What fits on two A100 40GB (MIG, no NVLink, no tensor parallel across cards)

| Model | Serve (vLLM, bf16 unless noted) | Train (LoRA, bf16 base) |
| --- | --- | --- |
| Qwen3-4B (current) | 8 GB weights, ample KV | 9 to 12 GB per trainer; two trainers fit |
| Qwen3-8B / rlm-qwen3-8b | 16 GB weights, KV fine at 32k | about 24 to 30 GB with checkpointing; one trainer |
| Qwen3-14B | 28 GB weights, KV tight above 16k | QLoRA (4-bit base) only, about 20 GB; slow |
| Qwen3-32B | AWQ 4-bit about 18 GB, inference only | no |

The one-card limit is the serving card: controller and helpers share it, so an 8B controller with
8B helpers halves rollout throughput relative to 4B. The capacity study (item 6 above) therefore
uses an 8B controller with 4B helpers first.

## Harness-side learning and role adapters (added 06:05 UTC after the user's questions)

Serving note: a second model does not require unloading the first. Two vLLM processes can share
one 40 GB card with reduced memory fractions (4B at about 0.25, 8B at about 0.55), each started
once (about 90 s); per-request LoRA adapters on one base are the cheapest form of this and are
what the controller adapters already use, with the un-adapted base as helper.

Prior art: GEPA (Agrawal et al., arXiv:2507.19457, ICLR 2026 oral; repository already cloned under
`research-cache/repos/gepa`) evolves prompts and other text parameters by reflection and beats
GRPO by 6 points on average with up to 35 times fewer rollouts, though GRPO still wins on AIME;
Meta-Harness (arXiv:2603.28052, cloned) and Harness-1 (arXiv:2606.02373, cloned) search the code
around a fixed model end to end; self-generated in-context exemplars (NeurIPS 2025) raise ALFWorld
from 73 to 89 percent by retrieving the agent's own successful trajectories, 93 with curation;
ExpeL, JARVIS-1, SkillRL (2026), AutoRefine (2026) and SkillsVote (2026) distill trajectories into
retrievable insights or skills. The RLM state is unusually retrieval-friendly: request metadata,
the peek output and a namespace summary are a few hundred tokens, and the same keys apply to a
sub-problem in a recursive call.

| Id | Question | Design | Cost | Decision |
| --- | --- | --- | --- | --- |
| H1 | Does retrieving one exemplar cell by state (family, length bucket, question type) from our teaching traces let the *untrained* 4B controller execute the routine? | Library = `teach-003/004` and OOLONG cells; inject the exemplar through the harness hint; evaluate on the held-out splits against base, SFT-005 and RL | no training; about 1 GPU-hour of evaluation | retrieval-augmented base within 5 points of SFT-005 means the routine is a prompting problem, not a weights problem |
| H2 | Does retrieval give routing without gradients? | Same library keyed by length; at 8 records the retrieved exemplar is a direct answer; on top of SFT-007 and on top of the base | about 1 GPU-hour | direct share at 8 records with no loss elsewhere |
| H3 | Can GEPA evolve the harness hint text to do what RL could not? | GEPA over the hint and bootstrap wording, metric = held-out correctness minus a token cost, budget 500 rollouts | about 2 GPU-hours of rollouts | beats SFT-007 at equal tokens, or routes correctly |
| H4 | Role adapters: does co-training a helper adapter with the controller adapter beat a fixed base helper? | Two LoRA adapters on one base, alternating updates, helper reward = extraction correctness, controller reward as now | one RL run, about 4 GPU-hours | revisits ledger C07/C08 (helper SFT improved labels but the root did not exploit them) |

Recursive calls at depth above one are not exercised by the current expert (helpers are
`ask_batch` calls, not nested controllers); H1 and H2 apply unchanged to a nested controller because
the retrieval key is the state shape, which is worth testing once a depth-2 teacher exists.

## Library learning for RLM controllers (added 06:45 UTC after the user's RAG proposal)

Proposal: make the decomposition routine, and every other reusable pattern, a *retrievable,
executable program* keyed by properties of the input, grown automatically from successful runs,
rather than something the weights must memorize. The controller's job shrinks to: characterize
the input, retrieve candidate programs, adapt and run them, and submit. Capacity moves from
weights into a library of verified programs, which is the right place for it when the model is
small.

Why it fits the RLM kernel specifically: the environment is a stateful Python process, so a
"skill" is a function that can be defined once and reused across turns and across recursive calls;
the peek cell already computes the natural retrieval key (size, line structure, markers, the
question); helper outputs can be cached in the namespace; and every run is verified and traced, so
admission to the library can be evidence-gated. The routing decision (direct answer versus a
routine) can live in the library's outcome statistics per input bucket, a bandit over verified
options, which cannot collapse the way a policy gradient does; this is a principled version of the
small-policy idea the user once abandoned, restricted to choosing among programs the LLM executes.

Prior art: Voyager's skill library of executable code (Wang et al. 2023, arXiv:2305.16291),
DreamCoder and LILO on learning program libraries by abstraction (arXiv:2310.19791), Agent
Workflow Memory (arXiv:2409.07429), the self-generated in-context exemplar result (NeurIPS 2025),
SkillRL, AutoRefine and SkillsVote (2026) on distilling trajectories into evidence-gated skills.
Our angle is new in three ways: skills are typed programs over a versioned ABI in a persistent
REPL, the retrieval key is the input signature produced by the agent's own first cell, and
selection statistics replace gradient routing.

Patterns beyond decomposition that the library should hold: regex-first extraction before any
model call (Prime Intellect reports OOLONG-synth-with-labels solved by regex); chunk-size
estimation from a cost model before committing a budget; index building over the input for
repeated lookups; memoized helper calls; verification routines (re-ask a sample, consistency
checks); recovery routines after a helper budget error; and recursion (call the kernel on a
sub-problem with the same library).

Ranked experiments (added to the queue as L1 to L4):

| Id | Question | Design | Cost | Decision |
| --- | --- | --- | --- | --- |
| L1 | Does a static library of our expert cells, retrieved by input signature and injected as prebound functions plus a usage note, let the untrained 4B and SFT-005 match or beat the RL controllers? | Library from `expert_cells/` and `oolong_cells/` as functions with notes; signature = peek output features; top-1 injection through the harness first, an ABI helper (`skills_search`, `skills_load`) second | about 1 GPU-hour per evaluation, no training | within 5 points of the best RL controller at equal or lower tokens means the routine belongs in the library |
| L2 | Does the library grow and transfer? | Admit programs from successful runs on the teaching split only (generalize constants to parameters, store outcome stats); evaluate on a new family never seen (OOLONG-real, or a new synthetic family) with and without growth | about 3 GPU-hours | OOD gain over the frozen library |
| L3 | Does bandit selection over (direct, routine) per signature bucket give selective recursion without gradients? | Thompson sampling on verified outcomes minus cost, bucketed by length and family, frozen policy weights | about 1 GPU-hour | direct share at 8 records, none at 32 and OOLONG 2k, no accuracy loss |
| L4 | Is a controller trained to *use* the library better than one trained to *contain* the routine? | SFT on library-using traces vs SFT-005; then RL from each | one SFT plus one RL run | accuracy, tokens, and robustness to a new family |

Two refinements from the user (07:40 UTC):

- **Retrieval payloads.** L1 retrieves executable routines (functions with contracts; they call
  `ask_batch` internally and may call the kernel recursively). H1' retrieves exemplar cells
  instead (demonstrations the controller imitates): same key, same splits, more tokens per turn,
  no verification of the artifact. Run both; the difference measures what verification buys.
- **A small router over a representation of the state (L3a to L3c).** Actions are the verified
  strategies (null, lookup, chunked extraction, later the others), never tokens. Representations,
  cheapest first: the explicit signature from the peek cell; a mean-pooled controller embedding of
  the peek-stage context; the controller's last-layer state at the decision token. Training data
  already exists: thousands of logged rollouts with route, outcome and cost on the teaching split.
  Fit offline (supervised or contextual bandit on outcome minus cost), evaluate by injecting the
  chosen action through the harness on held-out splits. This is the precise form of the earlier
  small-policy idea that was abandoned: the policy only chooses among verified programs and reads
  the LLM's own representation.

Leakage rule: the library is built only from the teaching split; held-out windows never seed it.

Status at 10:20 UTC (results in `docs/RESULTS.md`, L1-b to L1-d, L4; notebook of 2026-10-19):

- Implemented: kernel prelude (`PreludeSpec`), skill registry with learned-skill discovery and
  variants, `skills/stats.py` (signature from the input text, routine classification from traces,
  per-signature table, rendered note with a recommendation by accuracy minus 0.05 per 10k tokens),
  `build_stats.py`, `extract_cache.py`, `librarian.py` (propose, verify on cached model
  extractions of the teaching split, register with a verification pseudo-run so the routine
  enters the statistics), three held-out question types in `taskgen.py`.
- Found: with the routine in the library, the untrained controller is as good a caller as the
  taught and RL-trained ones in both families; teaching adds only the route choice; the
  statistics note adds the route choice without training; a prompt-side library (exec in the
  prompt) breaks the taught controller; an overconfident library docstring makes the untrained
  controller submit "unknown" on uncovered questions (E08 before-growth, 0/22), which the honest
  variant and growth address.
- Open: the librarian's 4B proposer verified the first held-out routine at 1.0 on 12 tasks (dry
  run), so the first two held-out types test the loop, not the proposer; `busiest_day` (tie
  rule) is the harder third; a family with joins and an 8B proposer come next if the loop holds.


## Time budget

Allocation ends 2026-10-22 00:39 UTC (about 38 hours from this revision). GPU 0 runs the
confirmatory RL lanes (pipelines 21 and 22) until about 16:00 UTC; GPU 1 serves every library,
statistics and growth evaluation, which are the binding constraint now (six to eight concurrent
sweeps share it). After the RL lanes end, GPU 0 is free for an 8B serving slot (librarian proposer,
capacity study C1) or the OOLONG 293-row library run. Before the allocation ends: rebuild the
training environment inside the monorepo, prune per-step adapters, publish the handoff.
