# Revised research plan after the routing experiments (2026-10-20, 06:30 UTC)

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

## Time budget

Allocation ends 2026-10-22 00:39 UTC. Pipelines 14, 16 and 17 (short-weighted, sequence-normalized,
strong KL) end around 14:00 UTC today; items 1 to 3 fit by about 20:00 UTC; item 6 overnight; items
7 and 8 on 21 October. Two trainers share GPU 0 throughout so neither GPU idles.
