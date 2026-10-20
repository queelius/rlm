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

## Concrete candidates from the second-level inventory (2026-10-20)

| Path | Size | Classification | Condition |
| --- | ---: | --- | --- |
| `runs/rlm-research-r4/sidecars/official-rlm-prime-pilot-v1` | 67 GB | delete | Not cited by any ledger entry; a retired prime-rl pilot with its own manifest. Keep `MANIFEST.json` and `RUNBOOK.md`. |
| `runs/rlm-research-r4/sidecars/selective-delegation-20260921` | 134 GB | prune | Cited by the ledger (threads 11 to 15). Keep reports, receipts and the checkpoints behind C79, C84 and C87; delete per-update adapters of retired pilots. |
| `runs/rlm-research-r4/shared/renderer-gate-launch` | 16 GB | delete | Not cited. |
| `runs/rlm-research-r4/campaigns/adaptive-context` | 9.7 GB | prune | Reports behind C05 and C06 stay; adapters can go. |
| other `runs/rlm-research-r4/sidecars/*` (about 25 pilots, 1.6 to 9.5 GB each) | about 70 GB | prune | Keep `REPORT.md`, `FINDINGS.json`, `MANIFEST.json`; delete adapters except those behind replicated claims (C07 trec-leaf-sft-v1, C50 openai-mrcr-procedural-sft-continue32-v1, C40 helper-agnews-*). |
| `~/research-runs/rl-paper-reproduction-20261012/preserved-checkpoints/.../ckpt--iter_0690--epoch_2.00--step_11040` | part of 181 GB | **keep** (archive, Hugging Face) | The audited SPO final state with its fourteen-file hash receipt. |
| rest of `~/research-runs/rl-paper-reproduction-20261012` | about 250 GB | prune | Intermediate SPO checkpoints and optimizer states. |
| `~/research-runs/spo-reproduction-20261008` | 171 GB | prune | First SPO attempt (restarted 2026-10-12); keep receipts and evaluations. |
| `~/research-runs/r1-zero-replication-20261007` | 185 GB | prune | Keep updates 208 and 216 of the stopped run and their evaluations; the four `subset512-*` runs (96 GB) and `archive` (47 GB) are candidates. |
| `runs/curl-*` | 234 GB | delete | After `rl-reproductions` is on GitHub with the curves asset. |
| `cache/huggingface/hub` (Qwen3-14B, Qwen3-14B-Base, Qwen3.5-9B, Qwen2.5-7B-Instruct, a duplicate Qwen3-4B-Instruct-2507) | 98 GB | delete | All re-downloadable; none used by a live experiment. |
| reproduction environments (`envs/prime-rl-*`, `envs/r1-zero-*`, `envs/spo-*`, `envs/curl-*`, `cache/uv-prime`) | 49 GB | delete | After each reproduction repository carries an exported lock. |
| `~/research-runs/selective-recursion-20261019/rl-*` (per-step adapters, 7.5 GB per run) | about 60 GB | prune later | After the campaign: keep the final adapter and any step cited in the ledger or `manifest.json`. |

**First tranche, approved by the user on 2026-10-20 and executed the same day (log: `/project/alex_phd/runs/DELETIONS-2026-10-20.jsonl`; small CURL metadata files preserved under `~/research-runs/archive/curl-runs-metadata-20261020`):** the HF hub cache (98 GB),
the CURL runs (234 GB, after the GitHub push), the prime pilot (67 GB) and renderer-gate (16 GB),
and the reproduction environments (49 GB). Everything in it is either re-downloadable,
reconstructible from a lock, or uncited by any claim.

## Order of operations

1. Sub-inventory `rlm-research-r4` and the three reproduction stores to list exactly which
   checkpoints the ledger and the reproduction reports cite (in progress).
2. Write `MANIFEST.json` (path, size, sha256 of small files, citation) for each archive class.
3. User approves a deletion list; deletions run with the list recorded in this file.
