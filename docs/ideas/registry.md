# Idea registry

Research questions noticed along the way, including ones unrelated to RLMs, kept so they can be pursued
later. Each entry records where the idea came from, a falsifiable question, a novelty status from a
prior-art search, and the smallest test. "No prior art found" is only as good as the search, so every
check records its queries and date.

Novelty status: `unchecked`; `checked: open` (no close prior art found); `checked: partial` (related work
exists, a stated gap remains); `checked: prior art` (answered; follow-ups listed if any remain open).
Promotion: an entry that is `open` or `partial` with a cheap smallest test moves to the backlog in
[the research program page](../research-plans/2026-10-22-research-program-and-idea-backlog.md).

| Id | Idea (short) | RLM? | Origin | Novelty | Smallest test |
| --- | --- | --- | --- | --- | --- |
| I-001 | Program-solvability audit of long-context benchmarks; a "shortcut-resistance" score | adjacent | E21 | unchecked | best few-shot program per benchmark at fixed proposer budget |
| I-002 | Documentation as an affordance: one usage line decides whether a small caller uses a library | yes | E25c | unchecked | 2x2 usage line x library, 4B and 0.6B callers (running) |
| I-003 | Quantification correction for LLM aggregation (classify-and-count bias from per-item confusion) | no | E12, OOLONG ceiling | unchecked | adjusted counts on OOLONG with a confusion matrix from teaching rows |
| I-004 | Reachability before RL: warm-start probability of each taught option predicts what on-policy RL can learn | general RL | E07 collapses | unchecked | measure option probabilities pre-RL across past runs, correlate with outcome |
| I-005 | Calling discipline as a transferable skill: a 0.6B caller distilled on two families calls unseen libraries | yes | E21, E25 | unchecked | unseen-library calls vs caller size and number of training families |
| I-006 | Sample efficiency of verified program discovery (candidates to first verified program) as a capability metric | no | E21, E26 | unchecked | discovery benchmark across proposers (running) |
| I-007 | Longitudinal operational failures of an autonomous research agent (idle GPUs, silent launches, template mismatches) | no | ops logs | unchecked | taxonomy and rates from this project's IDLE-LOG, FAILED and NOTE files |

## I-001 Program-solvability audit of long-context benchmarks

**Origin.** E21: a 40-line parser written by a 4-bit 8B model from three examples scores 1.000 on OpenAI MRCR
at every length from 8k to 1M tokens, while frontier models are reported far below that at long lengths.

**Question.** For each long-context benchmark, what score does the best program reach when a model writes it
from k examples under a fixed budget, and how does that compare with the best reading model? A benchmark
whose program score approaches 1.0 measures literal retrieval, not long-context understanding.

**Smallest test.** The bootstrapper on MRCR (done), RULER subsets, OOLONG-synth and NoLiMa, 8 candidates each;
report the program score next to published model scores.

## I-002 Documentation as an affordance for small callers

**Origin.** E21/E25c: with the 8B-written MRCR library, whose docstring lacks a "Typical use" line, the
untrained 4B often ignored the library and sent the whole input to a sub-model (overflowing at 128k and
above), or retyped the routine's output by hand; with a docstring carrying the line it called the routine in
one cell. The distilled 0.6B caller was unaffected.

**Question.** How much of a small model's tool or library use is decided by documentation form (usage
example vs signature only), and does the needed form depend on caller size or training?

## I-003 Quantification correction for LLM aggregation

**Origin.** OOLONG count questions are capped by per-row label ambiguity; classify-then-count inherits the
helper's confusion. Classical quantification (adjusted classify-and-count) corrects aggregate counts with an
estimated confusion matrix.

**Question.** Does confusion-matrix correction, estimated on a few teaching rows, raise accuracy on
LLM aggregation benchmarks, and when does it hurt (small samples, shift between teaching and test)?

## I-004 Reachability before RL

**Origin.** E07: every attempt to learn when to decompose collapsed; a taught option to which the warm start
assigned near-zero probability was never found by on-policy RL.

**Question.** Does the pre-RL probability mass on each taught option predict which options on-policy RL can
learn, well enough to decide before training whether to rebalance the teaching prior?

## I-005 Calling discipline as a transferable skill

**Origin.** The 0.6B caller distilled from 436 one-cell traces on event-log and OOLONG calls MRCR and MuSiQue
libraries it never saw (1.000 on MRCR with the 8B-written library).

**Question.** Is "call the provided routine correctly" a separable skill that transfers across libraries, and
how does transfer depend on caller size and the number of training families?

## I-006 Sample efficiency of verified program discovery

**Origin.** E21 and the E26 discovery benchmark: the 4B failed 24 of 24 MRCR candidates and the 8B succeeded on
the fourteenth; generated families give a success rate and a candidates-to-success count per proposer.

**Question.** Is candidates-to-first-verified-program a useful capability metric, separating reasoning
(thinking on or off) from scale, and does it predict success on real families?

## I-007 Longitudinal operational failures of an autonomous research agent

**Origin.** This project's operations notes: GPUs idle between sessions, launches that failed silently,
a train/serve chat-template mismatch, self-matching kill patterns, a dispatcher that never reaped its
wrappers. These are logged with times in IDLE-LOG, FAILED, STOPPED and NOTE files.

**Question.** Which operational failures dominate lost compute in weeks-long autonomous research by an AI
agent, and which cheap guards (checked launch, job queue, canary request) remove them?
