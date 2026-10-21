---
license: apache-2.0
base_model: Qwen/Qwen3-0.6B
library_name: peft
tags: [recursive-language-model, rlm, lora, long-context, tool-use]
---

# rlm-qwen3-0.6b-onecell-caller (draft, not yet published)

A LoRA adapter (r 32) for Qwen3-0.6B that acts as the **controller** of a recursive language model
(RLM) from [queelius/rlm](https://github.com/queelius/rlm): it reads only metadata about the input,
writes one Python cell that calls a prebound library of verified routines (which call a separate
helper model on chunks), and submits the answer. It never sees the long input itself.

**Training.** 436 correct one-cell library calls written by the untrained Qwen3-4B-Instruct-2507
controller on the teaching splits of two task families (synthetic store logs; OOLONG-synth
teaching windows), three epochs, learning rate 2e-4, about six minutes on one A100. Trained with
the serving template (thinking disabled). Dataset: `queelius/rlm-onecell-library-calls` (draft).

**Results** (helpers: Qwen3-4B-Instruct-2507; library and statistics note from the repository):

| | Store-log dev | Four grown question forms | OOLONG held-out (293 rows) |
| --- | --- | --- | --- |
| This adapter | 110/120 (seed 1: 107) | 94/96 | 0.672 |
| Untrained Qwen3-0.6B | 11/120 | 18/96 | 1/72 (sample) |
| Untrained Qwen3-4B | 111/120 | 94/96 | 0.672 |

Long contexts: see the repository's results ledger (E14) for 64k to 1M-token OOLONG inputs.

**Use.** Serve with vLLM (`--enable-lora`, `--default-chat-template-kwargs '{"enable_thinking": false}'`)
and point the RLM controller at it; the helpers and the library come from the repository
(`experiments/selective_recursion_20261019/`). Outside that harness the adapter is not useful.

**Limits.** Two task families; exact-match scoring; one or two training seeds; OOLONG counts are
limited by the helper's per-row labels, not by the caller.
