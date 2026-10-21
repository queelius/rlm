# RLM research

This site documents a PhD research program on **Recursive Language Models** (RLMs): language
models that do not read a long input directly but write code to inspect it, delegate pieces to
helper model calls, and submit an answer. The runtime in this repository is a small,
request-preserving RLM kernel; the research asks how a *small* model can use that kernel well: by training it, or by giving it a
verified library of routines and a record of what worked.

!!! tip "If you are new here"
    Read the [Primer](primer.md) first: what an RLM is, how our kernel works, one worked
    trajectory, and the vocabulary used everywhere else.

## Headline results (as of 21 October 2026)

The study set out to train a small controller to decompose long inputs, and found that the
capability is better placed outside the weights. The full assessment, with limits and the next
research questions, is the [synthesis of 21 October](notebook/2026-10-21-synthesis-and-next-questions.md).

| Finding | Status | Where |
| --- | --- | --- |
| Teaching then on-policy RL lifts a 4B controller from 0.567 to about 0.65 on held-out OOLONG-synth (direct answering 0.21) | replicated | [Claims ledger](RESULTS.md) |
| RL cannot hold the choice of *when* to decompose: every variant collapses to one route, even from a correct router | replicated | [note of 19 October](notebook/2026-10-19-selective-recursion-and-broader-teaching.md) |
| With the routines prebound as a library, the **untrained** 4B scores 0.672 on the 293 held-out rows, equal to the scripted expert and row by row to the best RL controller | replication-grade n | ledger L1, E09 |
| A table of verified outcomes keyed by the input (retrieval, RAG-style) recovers route choice with no training; the controller's own hidden state works as the key | exploratory | ledger L3a, E13 |
| Verified library growth recovers new question forms without training; a stronger proposer fixes what a weak one cannot | exploratory | ledger E08, E11 |
| With library and table, caller size stops mattering: a 0.6B caller tuned on 436 of the 4B's own calls is right on exactly the same 293 rows as the 4B; the recipe transfers to Phi-4-mini | replication-grade n | ledger S1 to S4 |
| The 0.6B caller works at 64k to 512k tokens (and beyond), cost linear in length; the remaining error is per-row label ambiguity in the helpers | exploratory | ledger E14, E15 |

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
