# Hugging Face release list (draft, 2026-10-20)

Fill from `experiments/selective_recursion_20261019/manifest.json` (hashes, run paths) and
`docs/RESULTS.md` (numbers). Login done 2026-10-20 (`oauth-queelius`). Uploaded and **public** since 2026-10-20 04:00 UTC: `queelius/rlm-qwen3-4b-sft-005` and `queelius/rlm-qwen3-4b-rl-mixed-001` (adapter_config.json, adapter_model.safetensors 252 MB, README card from the template).

| HF repo | Local artifact | Training paragraph | OOLONG 293 | Event-log 96 |
| --- | --- | --- | ---: | ---: |
| `queelius/rlm-qwen3-4b-sft-005` | `adapters/sft-005/final` | SFT (2 epochs, lr 1e-4) on native-replay trajectories of a scripted expert: unified peek-first expert, composed short question-specific cells, event-log seeds 16-23 plus OOLONG even windows. | 0.567 | 86/96 |
| `queelius/rlm-qwen3-4b-rl-mixed-001` | `rl-mixed-001/adapters/step-0030` | SFT-005 plus 30 updates of REINFORCE (leave-one-out baseline, 16 tasks x 4 samples, T=1, lr 2e-5), reward = correct - 0.1 x capped tokens / 10k, mixed event-log and OOLONG tasks up to 8k tokens. | 0.652 | 88/96 |
| `queelius/rlm-qwen3-4b-sft-007` | `adapters/sft-007/final` | As SFT-005 but on the route-balanced teaching set (equal direct and decompose trajectories on short inputs). | 34/72 sample | 85/96 |
| `queelius/rlm-qwen3-4b-rl-route-*` | best informative routing arm | To be chosen after pipeline 15. | | |

Datasets (CC-BY-4.0), uploaded and public 2026-10-20 05:05 UTC:
`queelius/rlm-teaching-eventlog` (1,234 message-cell pairs, synthetic), `queelius/rlm-teaching-oolong-cells`
(1,174 cells by OOLONG row id, no benchmark text), `queelius/rlm-oolong-split-and-verdicts` (600-row split
definition and 1,758 per-row verdicts for the six full held-out evaluations). Planned:

| HF repo | Content |
| --- | --- |
| `queelius/rlm-teaching-eventlog` | `teach-003`, `teach-004` sft.jsonl (messages, completion, variant, correct) |
| `queelius/rlm-teaching-oolong-ids` | `teach-oolong-004/005` with OOLONG text replaced by row id and window id |
| `queelius/rlm-oolong-split` | window-id split definition and per-row verdicts for every evaluated controller |
| `queelius/rlm-eval-traces` | kernel traces for the headline comparisons (OOLONG text stripped) |

## Released 2026-10-21 (public)

| HF repo | Content | Local source | sha256 (first 16) |
| --- | --- | --- | --- |
| [`queelius/rlm-qwen3-0.6b-onecell-caller`](https://huggingface.co/queelius/rlm-qwen3-0.6b-onecell-caller) (model, Apache-2.0, commit 79789fc0) | LoRA r32 on Qwen3-0.6B, distilled from the untrained 4B's 436 one-cell library calls; event-log dev 110/120, grown forms 94/96, OOLONG-293 0.672 (rows identical to the 4B); 64k to 2M-token OOLONG; paper's 131K TREC setting 45.3 | `adapters/sft-06-onecell-v2/final` (base path rewritten to `Qwen/Qwen3-0.6B`) | `8902798fa4920ca8` |
| [`queelius/rlm-onecell-library-calls`](https://huggingface.co/datasets/queelius/rlm-onecell-library-calls) (dataset, MIT, commit 5aed956f) | 436 (messages, completion) pairs, one controller turn each, from correct teaching-split runs of both families; no task input text (checked: no OOLONG instance lines, no log lines, no local paths) | `sft-data/onecell-4b-lib.jsonl` | `0d0c96a0133f142d` |

Uploaded files were downloaded again and their hashes matched the staged copies.
