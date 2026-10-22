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
| A 0.6B caller handles inputs far beyond any model's window | OOLONG 64k to 2M; event-log 20k to 80k tokens | cost linear in length; OOLONG falls through count questions; event-log 98/108 with no loss by length (direct 3/108) |
| OOLONG count errors are per-row label ambiguity, not routine choice | measured against per-row gold | helper agrees 0.903 (spam), 0.789 (TREC); systematic; not correctable by inversion |
| The paper's released RLM-Qwen3-8B, in its own scaffold, is far behind the library | 293 rows, paired | 0.290 vs 0.672 (130 to 18 discordant); above direct (0.208); a 131k window lifts it only to about 0.32 |
| On the paper's own 131K TREC setting, a 0.6B caller with the library is level with GPT-5 base | 50 tasks | 45.3 vs 44.0 (RLM with GPT-5: 56 to 58) |

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
- **Narrow external validity.** Two task families; OOLONG-synth's validation split covers only
  two label sets (SMS spam, TREC coarse) at every length; no OOLONG-real, no multi-hop or retrieval benchmark, no
  comparison with frontier-model RLMs beyond the paper's published numbers (the released RLM-Qwen3-8B was run, E18).
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
   OOLONG-synth's test split if it carries other label sets (the validation split used throughout
   has only SMS spam and TREC coarse, 650 rows each, at every length), a structured family from
   LongBench-v2, or OOLONG-real after a licence check. Decision: held-out accuracy of the
   bootstrapped library against a hand-written one and against direct answering.
2. **Does training the helper, not the controller, move the remaining error?** OOLONG supplies
   per-row gold labels for teaching windows, a verifiable per-row reward. Fine-tune (SFT, then
   RL) the 4B helper on teaching-window rows, after removing any instance text that also occurs
   in held-out windows, and measure held-out numeric-one-class and long-context accuracy.
   This is the test of "RL belongs where verification is local".
3. **How does this compare with RL-trained RLMs from the literature?** Done in part (E18, E19):
   the released RLM-Qwen3-8B in its own scaffold scores 0.290 on the 293 held-out rows against 0.672
   for the library, and on the paper's 131K TREC setting the 0.6B caller is level with GPT-5 base.
   Open: frontier-model RLMs on the same rows, and OOLONG-real.
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

## 6. Handoff at the end of the allocation (22 October 00:39 UTC)

**Where things are.**

- Code: `experiments/selective_recursion_20261019/` in this repository (all scripts named in the
  ledger); kernel prelude in `src/rlm/`.
- Run outputs: `/project/alex_phd/runs/selective-recursion-20261019/` (symlinks into
  `/home/atowell/research-runs/selective-recursion-20261019/`), one directory per run named as in
  the ledger; failed and partial attempts carry `FAILED.txt`, `STOPPED.txt` or `NOTE.txt`.
- Adapters: `.../adapters/` (SFT adapters; RL runs keep step 0, every fifth and final step after
  storage tranche 3). Statistics tables: `stats-teach-v0.json` to `stats-teach-v5.json`. Distillation
  sets: `sft-data/`.
- Datasets: `/project/alex_phd/research-cache/datasets/oolong-synth-20261019/` (up to 32k),
  `oolong-synth-20261021-long/` (pinned revision f0d59eaf, 64k to 2M exports),
  `helper-label-sources-20261021/` (SMS spam, TREC). Models: `/project/alex_phd/research-cache/models/`
  with `PROVENANCE.json` beside the newly downloaded ones (Qwen3 0.6B, 1.7B, 8B-AWQ, SmolLM2-360M).
- Environments: training `/project/alex_phd/envs/selective-recursion-train-5b1e2b4` (rebuild recipe in
  `EXPERIMENT.md`), serving `/project/alex_phd/envs/vllm`, official RLM scaffold
  `/project/alex_phd/envs/rlm-official-854e688`.

**State at the end.** Every run reported in the ledger finished and is recorded, except the
window-sensitivity rerun of the external baseline (32 of 48 rows, recorded as partial in E18).
GPU 1 was idle from 23:22 UTC, when the long store-log run finished, to the end of the allocation;
no ready job could have been completed and recorded in the remaining hour. The idle log has the entry.

**Next allocation.** A single A100 suffices for questions 1, 4 and 5 (inference dominated): serve the
4B helper at about 0.55 of the card, the 8B-AWQ proposer at 0.28 when the librarian runs, and the
0.6B caller at 0.08; run small-caller fine-tunes between evaluations. A second card mainly buys
throughput; ask for two again only if helper training at scale or large 8B experiments resume.

**Resuming.** On a new allocation, serve Qwen3-4B-Instruct-2507 as in `EXPERIMENT.md` (port 8001,
LoRA enabled), serve small callers with thinking disabled, and run any ledger entry's command from
the pipeline scripts in the run directory (`*_gpu0.sh`, `*_gpu1.sh`, `s*.sh`, `e*.sh`). The first
experiments to run are questions 1 (bootstrap a library for a new family) and 4 (recursive
routines), which this allocation did not reach.

**Released.** The distilled 0.6B caller (`queelius/rlm-qwen3-0.6b-onecell-caller`) and its 436 training
pairs (`queelius/rlm-onecell-library-calls`), public on Hugging Face since 21 October; see the release list.

## 7. Update after 22 October (allocation 6599, to 15:30 UTC)

The questions of section 4 now have answers or sharper forms. Ledger rows E21 to E34 carry the numbers.

**Where libraries come from (question 1).** A capable proposer can write a verified library for a new real family
from three examples: the thinking Qwen3-8B wrote MRCR's library on 2 of 5 seeds (1.000 held out to 1M tokens). A 4B
proposer cannot, however trained (SFT on the 8B's code, RL with the verified teaching score as reward, test-time RL on
MRCR's own teaching rows: 0 accepted). On 37 generated held-out families, training does work: SFT then RL takes the
4B from 9 to 30 of 74 attempts (replicated with a second RL seed; RL without SFT does not help), matching the
thinking 4B at a twenty-fifth of the tokens. Trained only on code-solvable families the 4B becomes a parser writer
that ignores the helper; helper-dependent training families are being added (E26e). For helper-dependent real
families (MuSiQue), even the 8B does not invent retrieval unless given retrieval primitives, and then composes a
working routine on 1 of 2 seeds. Verification must cover the lengths the library will face, and run under the
kernel's semantics.

**Selection and composition (questions 2, 3).** Routing by length among close routines failed a pre-registered test
(30-question cells are noise). Recursion does not help a small model on MuSiQue, paper-style or per hop.

**The caller (question 3).** A thin caller is enough: the distilled 0.6B uses libraries of families it never saw
(MRCR 1.000, MuSiQue and LongBench-v2 level with the 4B). For an untrained caller, one usage line in the docstring
decides whether the library is used at all.

**Where capacity belongs.** In choosing evidence: two-round retrieval is the best and cheapest MuSiQue library (also
at 512k), an 8B reader adds nothing on MuSiQue or LongBench-v2, and on LongBench-v2 a retrieve-then-read library
beats direct reading by 0.11 at a quarter of the tokens, through the 0.6B caller. Map-reduce notes do not help.

**Withdrawn on 22 October:** the distractor-limited reader (E22c), length routing (E31), and "libraries stop helping
on LongBench-v2" (E32; it was the map-reduce routine).

**Open, ranked:** (1) proposer training with helper-dependent families (running); (2) primitives plus proposer
training for retrieval families; (3) why MRCR resists small proposers (the 0.214 partial program is a local optimum
that RL does not leave); (4) selection memory with pooled priors; (5) a note on program-solvable long-context
benchmarks (idea lab I-001).
