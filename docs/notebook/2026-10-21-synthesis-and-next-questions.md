# Synthesis and next questions (21 October 2026)

This note takes stock of the selective-recursion study after three days on two A100s (19 to 21
October 2026). It states what the evidence supports, how strongly, what it does not show, and
which research questions it opens. Ledger entries are in [the claims ledger](../RESULTS.md); the
day-by-day record is the [note of 19 October](2026-10-19-selective-recursion-and-broader-teaching.md).

## 1. What the evidence supports

**Setting.** A recursive language model (RLM) in which a controller writes one Python cell per
turn in a persistent interpreter, never sees the long input, and calls helper models on pieces of
it. Two task families: a synthetic store log (8 to 512 lines, nine question forms, five held out)
and OOLONG-synth (aggregation over labelled instances, 1k to 2M tokens). Helpers are always
Qwen3-4B-Instruct-2507; scoring is exact match.

| Claim | Strength | Key numbers |
| --- | --- | --- |
| Teaching then RL improves a 4B controller that carries the routine in its weights | replicated (four RL runs) | held-out OOLONG 0.567 to 0.63 to 0.65 |
| Policy-gradient RL cannot hold a length-conditional route choice | replicated across every variant tried | collapse within 2 to 5 updates, even from a correct router |
| With the routines prebound as a library, the *untrained* 4B equals the scripted expert | replication-grade n | 0.672 [0.619, 0.726] on 293 held-out rows |
| ... and equals the best RL controller, row by row | paired | 197 vs 191 correct; 8 discordant rows (7 to 1), p = 0.07 |
| A table of verified outcomes keyed by the input recovers route choice without training | one family | 87/96 and 24/24 at the lowest cost of any controller |
| The controller's own hidden state (plus length) is as good a retrieval key as hand features | offline, two routes | 0.705 vs 0.695 at moderate cost pressure; equal otherwise |
| Verified library growth recovers held-out question forms without training | four forms | 0/24 to 22 to 24/24 each; a buggy proposer is kept out; an 8B proposer fixes it |
| With library and table, caller size stops mattering | 293 rows, two seeds, two families | 0.6B, 1.7B and 4B correct on **identical** rows (0 discordant of 293) |
| A 0.6B caller learns the calling discipline from a few in-family traces | one to two seeds | 55 traces (19 OOLONG) suffice; event-log-only traces do not transfer |
| The recipe is not specific to Qwen | one seed | Phi-4-mini 0/72 untrained, 43/72 after the same 436 traces |
| A 0.6B caller handles inputs far beyond any model's window | 80 rows, 64k to 512k; 1M and 2M in progress | cost linear in length; accuracy falls through count questions |
| OOLONG count errors are per-row label ambiguity, not routine choice | measured against per-row gold | helper agrees 0.903 (spam), 0.789 (TREC); systematic; not correctable by inversion |

## 2. What the results mean

**The division of labour inverts the original plan.** The study set out to train a small
controller to decompose. The evidence says the decomposition routine is better placed in a
verified library, the choice among routines in a table of verified outcomes, and the weights need
only learn to call the library correctly, which a 0.6B model does from a few dozen traces. With
those three pieces in place, the controller contributes nothing measurable beyond selection: the
0.6B, 1.7B and 4B callers are right on exactly the same 293 held-out rows.

**What RL was doing.** The best RL controller and the library agree, correct or wrong, on 285 of
293 rows: RL taught the 4B to execute, approximately, the routine the library supplies outright.
Its gains over the taught warm start (fixing helper budgets at 32k tokens) are the same gains the
library delivers by construction. Where RL was asked for something not verifiable per step, the
route choice, it collapsed in every form tried, in line with the routing-collapse literature.

**Where the remaining error lives.** Once the routine and the route are fixed, all residual error
is in the helpers: OOLONG count questions fail because about one row in ten (spam) to one in five
(TREC) is labelled differently from gold, systematically, and exact counts over hundreds of rows
cannot survive that. Neither routine variants, a larger helper, nor statistical correction
removed it. This is the first place in the study where training (of the helper, against
verifiable per-row labels) has a clear, measurable target.

## 3. What the evidence does not show

- **The library encodes our expert's knowledge.** For both families the base routines were
  written by us; "untrained 4B equals the expert" is partly by construction. The open question
  is where libraries come from. Growth has only been shown for new aggregators over an existing
  extraction step, not for bootstrapping a whole new family.
- **Narrow external validity.** Two task families; OOLONG-synth at up to 32k covers only two
  label sets (SMS spam, TREC coarse); no OOLONG-real, no multi-hop or retrieval benchmark, no
  comparison with the RLM paper's released RLM-Qwen3-8B or its frontier-model numbers.
- **Small samples outside OOLONG-293.** Most event-log and 72-row OOLONG comparisons are single
  seeds; differences of a few rows are within noise (the 293-row interval is about plus or
  minus 0.054).
- **Route choice has little headroom here.** With the library dominating direct answering
  almost everywhere, the oracle router is only 3 points above always-library; the router results
  matter mainly under cost pressure.
- **Depth zero only.** No library routine calls the kernel recursively; the "recursive" part of
  recursive language models is not yet exercised.

## 4. Research questions and proposed experiments

Ranked by expected information per GPU-hour and by how directly they test the thesis claim.

1. **Can a library for a new task family be bootstrapped automatically?** (the central open
   question). Give the librarian (8B proposer, thinking on) a new family with only teaching
   examples: it must propose the parse, extraction and aggregation routines, verified end to
   end on the teaching split, then the 0.6B caller is distilled from the 4B's calls. Candidates:
   OOLONG-synth's other label sets above 32k (more datasets appear there), a structured family
   from LongBench-v2, or OOLONG-real after a licence check. Decision: held-out accuracy of the
   bootstrapped library against a hand-written one and against direct answering.
2. **Does training the helper, not the controller, move the remaining error?** OOLONG supplies
   per-row gold labels for teaching windows, a verifiable per-row reward. Fine-tune (SFT, then
   RL) the 4B helper on teaching-window rows, after removing any instance text that also occurs
   in held-out windows, and measure held-out numeric-one-class and long-context accuracy.
   This is the test of "RL belongs where verification is local".
3. **How does this compare with RL-trained RLMs from the literature?** Run the RLM paper's
   released RLM-Qwen3-8B (cached locally) on the same 293 held-out rows and long contexts, in
   its own scaffold and as a caller in ours, against our 0.6B caller plus library.
4. **Recursion.** Add library routines that call the kernel on a sub-problem (depth one): for
   example per-document multi-hop reasoning over MuSiQue or HotpotQA contexts (cached), where
   per-chunk work itself needs decomposition. Question: does the library-plus-table recipe
   extend to recursive routines, and does a small caller still suffice?
5. **Continual library learning.** One stream in which new question forms appear unannounced:
   failures trigger the librarian, accepted routines enter the table, the bandit adapts. Measure
   time to recovery and regret, the online version of growth plus selection.
6. **Retrieval at scale.** With many routines across several families the hand signature
   cannot be maintained; test the learned key (controller hidden state plus length) for routine
   retrieval among ten or more routines, against the hand signature and an oracle.
7. **How small can the caller be, really?** Below 0.6B (a 135M to 360M model), and with no
   in-family traces but an executable recipe in the note, to separate protocol discipline from
   composition.

Hygiene to apply before any claim is promoted: two or more seeds per arm, the 293-row OOLONG set
or equivalent n for every headline comparison, paired tests, and the helper fixed and recorded.

**Update, 18:05 UTC (E17).** Question 2 was tried in supervised form. OOLONG-synth's windows
share one item pool, so its teaching labels leak into every evaluation window; a helper LoRA was
instead trained on SMS-spam and TREC items absent from all OOLONG windows. It did not help (TREC
per-row agreement 0.759 to 0.750; spam collapsed from a 12-spam training pool). For these two label
sets the gold labels themselves bound the helper; the question needs label sets with less
instance-level ambiguity, or tasks whose per-row verification is exact.

## 5. Remaining time on this allocation

The allocation ends 22 October 00:39 UTC. Running: OOLONG at 1M and 2M tokens with the 0.6B
caller. Feasible before the end: question 2 in its supervised form (helper SFT on teaching-window
labels with an overlap filter, then the 293-row and 64k to 512k evaluations), and the external
baseline of question 3 at up to 32k. Questions 1, 4 and 5 need a fresh allocation.
