# Storage plan for the cluster stores (inventory of 2026-10-20 03:03 UTC)

The project volume `/project/alex_phd` is at its 1.1 TB quota; new outputs go to the home volume
and are symlinked. Nothing below is deleted without the user's explicit approval, and nothing is
moved under a live process. Proposed classes: **archive** (immutable, hashed manifest, referenced
by the claims ledger), **prune** (keep final and referenced checkpoints, delete intermediates),
**delete** (reproducible or superseded), **keep** (active or required).

## Run stores

| Path | Size | What it is | Proposal |
| --- | ---: | --- | --- |
| `runs/rlm-research-r4` | 325 GB | Codex-era structured store (Aug 24 to Sep 29): questions, claims, analyses, sidecars with checkpoints of many one-to-eight-update pilots. The claims ledger cites its reports and analyses, which are small. | **Prune**: keep `questions/`, `claims/`, `analyses/`, every `REPORT.md`/`FINDINGS.json` and the checkpoints behind replicated claims (C06, C50, C60, C62, C68); delete the rest after a sub-inventory. Expected recovery: most of 325 GB. |
| `runs/curl-*` (8 dirs) | 234 GB | CURL reproduction training runs (nine fresh-seed models and controls). Findings and 2.3 MB of curves are preserved in `rl-reproductions`. | **Delete** after the reproductions repository is on GitHub, keeping at most one final checkpoint per arm if the user wants weights (about 100 MB each). |
| `~/research-runs/rl-paper-reproduction-20261012` | 292 GB | SPO reproduction, restarted run (the audited 58.19 percent result) with periodic full checkpoints and optimizer states. | **Prune**: keep the final audited checkpoint (for Hugging Face), its evaluation outputs and receipts; delete intermediate optimizer states. |
| `~/research-runs/spo-reproduction-20261008` | 171 GB | SPO first attempt (diagnosed and restarted 2026-10-12). | **Prune** to receipts and evaluations; delete checkpoints. |
| `~/research-runs/r1-zero-replication-20261007` | 185 GB | R1-style math RL; checkpoints 208 and 216 preserved by the user's stop decision. | **Prune**: keep updates 208 and 216 with their evaluations; delete other checkpoints. |
| `runs/rlm-v2-spikes` | 20 GB | August interface-competence SFT spikes (C01 to C05). | **Archive** the evaluation summaries; delete adapters except the v2b dispatcher (C03). |
| `runs/rlm-teaching-goal-20261014`, `runs/rlm-bootstrap` | 4 GB | October teaching campaign receipts and adapters (E04, E05). | **Archive** as is. |
| `~/research-runs/selective-recursion-20261019` + `runs/selective-recursion-20261019` | 59 GB | Active study (E06, E07): adapters per RL step, traces, evaluations. | **Keep** while running; afterwards prune per-step adapters to the final and the hashed ones in `manifest.json`, and traces to the evaluations cited in the ledger. |
| `runs/rlm-research`, `-r2`, `-r3` | 36 MB | Earlier notebook stores. | **Archive**. |

## Caches and environments

| Path | Size | Proposal |
| --- | ---: | --- |
| `cache/huggingface` | 98 GB | Re-downloadable hub cache (Qwen3-14B, Qwen3.5-9B, datasets). **Delete** entries not used by an active experiment; the OOLONG snapshot is already mirrored under `~/research-cache/hf`. |
| `research-cache/models` | 71 GB | Pinned base-model snapshots (Qwen3-4B-Instruct-2507 in use; Qwen3-8B, Phi-4-mini, Mistral-7B, others). **Keep** the ones named in experiment manifests; delete the rest once a manifest lists them. |
| `envs/prime-rl-*`, `envs/r1-zero-*`, `envs/spo-*`, `envs/curl-*`, `cache/uv-prime` | 49 GB | Reproduction environments. **Delete** after each reproduction repository carries an exported lock (as done for the training environment here). |
| `envs/vllm`, `envs/rlm` | 8 GB | Active serving and runtime environments. **Keep**. |
| `research-cache/runtime-images`, `research-cache/spo`, `research-cache/datasets` | 30 GB | Container images, SPO data, benchmark exports. **Keep** datasets with provenance records; review images. |

## Order of operations

1. Sub-inventory `rlm-research-r4` and the three reproduction stores to list exactly which
   checkpoints the ledger and the reproduction reports cite (in progress).
2. Write `MANIFEST.json` (path, size, sha256 of small files, citation) for each archive class.
3. User approves a deletion list; deletions run with the list recorded in this file.
