# Idea inbox

Append-only list written by the experiment session. The registry session moves items into
`registry.md` and marks them here. Format: date, short title, origin with evidence pointer, one-line
question.

- 2026-10-22. Seven seed ideas were written directly into `registry.md` as I-001 to I-007 (unchecked). A
  first novelty check of I-001 to I-004 and I-007 was started from the experiment session; its result will
  be appended here when it arrives.

## 2026-10-22 03:00 UTC: novelty check of I-001 to I-004 and I-007 (from the experiment session)

One subagent, mostly abstract-level reading; all five came out `partial`. For the registry session to
fold into `registry.md` (verdicts, closest works, queries, follow-ups).

- **I-001 partial.** Closest: Coding Agents are Effective Long-Context Processors (2603.20432); LOFT
  (2406.13121, specialized retrieval/SQL vs long-context models); 100-LongBench (2505.19293, separates
  baseline knowledge from long-context ability); NoLiMa (2502.05167). No critique found that MRCR or
  Michelangelo is parser-solvable, and no program-solvability or shortcut-resistance metric; one
  non-peer-reviewed post calls MRCR "benchmaxed". Queries: "MRCR solvable by regex parsing shortcut",
  "long-context benchmark regex OR grep baseline", "shortcut resistance OR programmatic solvability
  long-context", "Michelangelo Latent List code execution shortcut", "MRCR agent code scaffold 1M".
  Follow-ups: (1) program-solvability curve for MRCR, Michelangelo (Latent List may be directly
  executable), RULER, OOLONG-synth, NoLiMa at 128k with the same 8B writer, k=3, about 50 instances each;
  (2) does program-solvability predict the RLM-minus-direct gain across existing paired runs.
- **I-002 partial.** Closest: Hsieh et al. tool documentation (2308.00675); When LLMs Meet API
  Documentation (2503.15231: example code matters most, weaker models copy examples verbatim); Meta-Tool
  (2604.20148: few-shot +21.5% vs docs +5.0% on a 3B); MCP Tool Descriptions Are Smelly (2602.14878). No work
  crosses documentation form with caller size or training, or measures whether a library is called at all.
  Queries: "tool documentation usage examples small models size ablation", "tool description wording
  selection model scale", "MCP tool description smells", "API documentation components examples vs
  signatures", "small model ignores available tool invocation rate". Follow-ups: (1) usage line x caller
  size grid (Qwen3 0.6B to 14B untrained, plus the distilled 0.6B), first-cell library-call rate and
  accuracy; (2) documentation form arms on the 4B (usage line, full example, signature only, prose).
  Experiment evidence so far (E25c): 8B library 0.601 without the line vs 0.953 with it (4B caller, MRCR 2
  needles); reference library 0.952 with vs about 0.51 without (partial).
- **I-003 partial.** Closest: Rogan-Gladen correction for LLM-judge labels (2601.05420); LLM prevalence of
  policy violations (2602.18518); multicalibration for prevalence under shift (2604.21549); PPI and
  confidence-driven inference (2408.15204). Not found: adjusted counts inside long-context aggregation.
  Caveat: OOLONG (2511.02817) filtered items that strong models misclassify, so room for correction is
  larger with small classifiers. Queries: "LLM quantification adjusted classify and count", "quantification
  LLM zero-shot ACC PACC EMQ", "PPI LLM annotations proportion", "Rogan-Gladen LLM annotation prevalence",
  "OOLONG counting correction calibration". Follow-ups: (1) OOLONG-synth trec_coarse, small classifiers,
  confusion matrix from k in {8, 32} labelled rows, CC vs ACC vs PACC count error; (2) when it hurts, by
  prevalence and separability.
- **I-004 partial.** Closest: Yue et al. (2504.13837, RLVR does not widen pass@large-k coverage); The
  Invisible Leash (2507.14843); ARM (2505.20258) and Thinkless (2505.13379) on GRPO mode collapse; ParaVT
  (2605.20342: weak tool prior gives zero tool calls under RL). No quantitative pre-RL reachability
  threshold found. Queries: "invisible leash support constraint", "Ada-GRPO format collapse", "predict
  before RL which behaviors learnable SFT prior rebalance", "base pass@k predicts RL gains", "RL tool-use
  collapse initial tool-call rate". Follow-ups: (1) dose-response: warm-start mass on option B in {0.01,
  0.05, 0.2, 0.5}, B optimal on a known subset, same GRPO budget on 0.6B; (2) mass vs expected advantage.
- **I-007 partial, close to open on operations.** Closest: Kirgis et al. (2607.27191, six days, five
  failure modes incl. poor resource awareness); Safdar and Saadeldin (2608.01995, 10-week case study, about
  2.4k GPU-hours); Deep Researcher Agent (2604.05854, 30 days); MLE-bench (2410.07095); MAST (2503.13657).
  None quantify lost GPU-hours by operational failure type or test guards. Queries: "longitudinal case study
  autonomous AI research agent operational failures GPU", "failure taxonomy MLE-bench RE-Bench timeouts
  process management", "Claude Code OR Codex autonomous research weeks incidents". Follow-ups: (1) replay
  the timestamped logs to attribute lost GPU-hours to failure types before and after each guard; (2) share
  of idle time at session handoffs vs within sessions.
