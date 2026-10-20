# RLM research

This site documents a PhD research program on **Recursive Language Models** (RLMs): language
models that do not read a long input directly but write code to inspect it, delegate pieces to
helper model calls, and submit an answer. The runtime in this repository is a small,
request-preserving RLM kernel; the research asks whether a *small* model can be taught and then
trained by reinforcement learning to use that kernel well.

!!! tip "If you are new here"
    Read the [Primer](primer.md) first: what an RLM is, how our kernel works, one worked
    trajectory, and the vocabulary used everywhere else.

## Headline results (as of 20 October 2026)

| Finding | Status | Where |
| --- | --- | --- |
| A 4B controller taught by replaying a scripted expert through the real kernel executes the decomposition routine, and 30 updates of on-policy RL raise held-out OOLONG-synth accuracy from 0.567 to 0.63 to 0.65 (direct answering 0.21, expert 0.67), replicated across four RL runs | replicated | [Claims ledger](RESULTS.md), [note of 19 October](notebook/2026-10-19-selective-recursion-and-broader-teaching.md) |
| The RL gain is a length-generalizing fix of helper budgets at 32k tokens; the token-cost term is nearly inert | replicated | same |
| Learning *when not to decompose* fails by routing collapse: from a warm start that holds both routes, RL removes the direct route within a few updates under every reward tried; the mechanism (gradient from lengths where direct answering fails, token weighting, bursty task sampling) is being dissected and remedies are running | in progress | [Plan of record](research-plans/2026-10-20-revised-plan-and-prior-art.md) |
| Teaching-history repair on TextCraft beats a weak teacher but not stronger controls; every pre-October attempt to learn routing by RL was exploratory or null | replicated / null | [Full ledgers](results/) |

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
