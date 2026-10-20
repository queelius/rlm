---
license: apache-2.0
base_model: Qwen/Qwen3-4B-Instruct-2507
library_name: peft
tags: [rlm, recursive-language-model, lora, controller, long-context]
datasets: []
---

# {MODEL_NAME}

LoRA adapter (rank 32, all projection matrices) for **Qwen/Qwen3-4B-Instruct-2507** (revision
`cdbee75f17c01a7cc42f958dc650907174af0554`) that acts as the *controller* of a Recursive Language
Model: given only request metadata, it writes one Python cell per turn in a persistent IPython
session, inspects the input, delegates chunks to helper calls (`ask_batch`) or answers directly
(`model_complete`), and submits a final answer with `FINAL_TEXT`. It is meant to be served with
the `rlm` runtime (https://github.com/queelius/rlm) and vLLM with LoRA enabled; it is not a chat
model and does nothing useful without the harness.

## Training

{TRAINING_PARAGRAPH}

Environment: see `experiments/selective_recursion_20261019/env/` in the repository (torch 2.13,
transformers 5.15.1, peft 0.20.0; vLLM 0.27.1 for serving and rollouts).

## Evaluation (held out; from `docs/RESULTS.md`)

| Benchmark | This adapter | Taught warm start | Direct answering | Scripted expert |
| --- | ---: | ---: | ---: | ---: |
| OOLONG-synth validation, odd windows, 1k to 32k tokens (293 rows) | {OOLONG} | 0.567 | 0.208 | 0.672 |
| Synthetic event logs, 8 to 512 records, aggregation (96) | {EVLOG} | 86/96 | varies by length | 93/96 |

Verifier: format-tolerant exact match on the final answer line (not the official OOLONG scorer).
Costs are endpoint-reported tokens including the harness prompt repeated every turn.

## Intended use and limitations

Research artifact for studying taught and learned decomposition in small controllers. One model
family and size; two task families; synthetic event logs; routing between direct answering and
decomposition is *not* learned by this adapter (see the routing study in the repository).

## Provenance

Code commit `{COMMIT}`; adapter sha256 `{SHA}`; run store `{RUN}`. Teaching data: the
`rlm-teaching-*` datasets (OOLONG-derived rows carry ids only; OOLONG-synth text is not
redistributed because the source dataset declares no license).
