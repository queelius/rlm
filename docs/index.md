# RLM research

This site documents a PhD research program on **Recursive Language Models** (RLMs): language
models that do not read a long input directly but write code to inspect it, delegate pieces to
helper model calls, and submit an answer. The runtime in this repository is a small,
request-preserving RLM kernel; the research asks how a *small* model can use that kernel well: by training it, or by giving it a
verified library of routines and a record of what worked.

!!! tip "If you are new here"
    Read the [Primer](primer.md) first: what an RLM is, how our kernel works, one worked
    trajectory, and the vocabulary used everywhere else.

## Headline results (as of 22 October 2026)

The study set out to train a small controller to decompose long inputs, and found that the
capability is better placed outside the weights. The full assessment, with limits and the next
research questions, is the [synthesis of 21 October](notebook/2026-10-21-synthesis-and-next-questions.md)
(section 7 updates it after 22 October).

| Finding | Status | Where |
| --- | --- | --- |
| Teaching then on-policy RL lifts a 4B controller from 0.567 to about 0.65 on held-out OOLONG-synth (direct answering 0.21) | replicated | [Claims ledger](RESULTS.md) |
| RL cannot hold the choice of *when* to decompose: every variant collapses to one route, even from a correct router | replicated | [note of 19 October](notebook/2026-10-19-selective-recursion-and-broader-teaching.md) |
| With the routines prebound as a library, the **untrained** 4B scores 0.672 on the 293 held-out rows, equal to the scripted expert and row by row to the best RL controller | replication-grade n | ledger L1, E09 |
| A table of verified outcomes keyed by the input (retrieval, RAG-style) recovers route choice with no training; the controller's own hidden state works as the key | exploratory | ledger L3a, E13 |
| Verified library growth recovers new question forms without training; a stronger proposer fixes what a weak one cannot | exploratory | ledger E08, E11 |
| With library and table, caller size stops mattering: a 0.6B caller tuned on 436 of the 4B's own calls is right on exactly the same 293 rows as the 4B; the recipe transfers to Phi-4-mini | replication-grade n | ledger S1 to S4 |
| The 0.6B caller works from 64k to 2M tokens on OOLONG (cost linear in length) and the library holds at 20k to 80k-token store logs (98/108 against 3/108 direct); OOLONG's remaining error is per-row label ambiguity in the helpers | exploratory | ledger E14, E15 to E17, E20 |
| The RLM paper's released RLM-Qwen3-8B, run in its own scaffold, scores 0.290 on the same 293 rows (library: 0.672); on the paper's 131K TREC setting the 0.6B caller with the library scores 45.3, level with GPT-5 base (44.0) | exploratory | ledger E18, E19 |
| A 4-bit Qwen3-8B with thinking writes a verified MRCR library from three examples (2 of 5 seeds; 1.000 held out to 1M tokens), and the 0.6B caller, never trained on MRCR, calls it correctly; no 4B proposer finds it | exploratory | ledger E21, E21b, E26b to E26d |
| Training the proposer: SFT on the 8B's verified code then RL with the verified teaching score as reward lifts a 4B from 9 to 30 of 74 held-out family attempts (two RL seeds), and, with more varied teacher data, to 42 on one RL seed (level with the 8B) but 31.5 on average over four seeds (range 22 to 42); RL without SFT does not help | replicated (9 to 30); 42 is the best of four RL seeds | ledger E26b, E26c, E26g, E43, E45, E46 |
| Choose evidence: two-round retrieval is the best and cheapest MuSiQue library (512k tokens at 8k helper tokens per question); a retrieve-then-read library beats direct reading on all 503 LongBench-v2 questions (0.43 vs 0.36, +0.07, se 0.02), also through the 0.6B caller; bigger readers, map-reduce and recursion do not help | exploratory | ledger E22c to E22h, E23, E25e, E32 to E32e |
| One docstring usage line decides whether an untrained caller uses a library (0.18 to 0.95 on MRCR); the distilled caller does not need it | exploratory | ledger E25c, E25d |
| Where answers are exact, bootstrapping needs three labels: a cleanly running candidate that reproduces the three examples in its prompt is almost always correct (127 of 129, generalizing to fresh inputs); pre-registered on 30 unseen families, the trained 4B with eight samples and three labels matches the thinking 8B with thirty (12 vs 13) at 7 percent of the tokens; agreement among candidates is no substitute | pre-registered, replicated (pooled 26 vs 29 of 60, p 0.61), robust across 8 proposer checkpoints (12 to 16 vs 13) | ledger E37, E39 |
| More helper-family teacher data gives recipes that match round four on solves (means 33.7 and 31.7 vs 31.8 of 74, over three to five runs each) and reach the 8B on helper families (mean best 0.33 to 0.58 vs 0.10 to 0.25; 8B 0.46); round four varies far more across RL seeds; training reward does not track held-out success | exploratory (multi-seed) | ledger E36, E41, E43 to E48 |
| Boundary: OOLONG is not bootstrapped as one family (eleven question templates from thirty examples); narrowed to one task group, the 14B writes a counting library level with the hand-written one, not timeline or user | exploratory | ledger E35, E35c |

## How this site is organized

- **[Claims ledger](RESULTS.md)**: every empirical claim with status (exploratory, replicated,
  confirmed, retracted) and evidence. The single place to check what is load-bearing.
- **[Experiments](experiments/)**: one page per experiment with question, design, immutable
  inputs, metrics and compute.
- **[Lab notebook](notebook/)**: dated notes in the order the work happened, including failures.
- **[Plan of record](research-plans/2026-10-20-revised-plan-and-prior-art.md)**: the ranked queue,
  revised at every completed arm, with a revision log and the prior art that shaped it.
- **[Links and prior art](links.md)**: our models and datasets on Hugging Face, the companion
  reproduction repositories, and the papers and posts we build on.

## Code and artifacts

- Runtime and experiments: https://github.com/queelius/rlm (`src/rlm` is the kernel;
  `experiments/` holds each study's scripts, manifest and environment lock).
- Models and data: [Hugging Face releases](hf/RELEASE_LIST.md).
- Companion reproductions: [SPO on GSM8K](https://github.com/queelius/spo-gsm8k-reproduction)
  (with its own [walkthrough](https://queelius.github.io/spo-gsm8k-reproduction/)) and
  [R1-style Dr. GRPO and CURL](https://github.com/queelius/rl-reproductions).
