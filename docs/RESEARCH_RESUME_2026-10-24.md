---
date: 2026-10-24
status: allocation_ending
allocation: job 6599, two A100 40GB (MIG 7g.40gb each), ends 2026-10-25 00:52 UTC
research_cutoff: 2026-10-23 (draft 14:45 UTC; final section added at handoff)
branch: research/r1-replication-20261007, pushed to origin/main
---

# Selective recursion and library bootstrapping: resume point

This allocation (19 to 25 October) studied how small recursive language models get capability from verified
libraries: who can write a library (proposers), how a library is accepted (verification), how a small caller uses
it, and where the approach stops working. Everything below is reproducible from the repository plus the run store;
model weights and adapters live only on `/project/alex_phd` and `/home/atowell` (not in git).

## Read first

1. `docs/RESULTS.md`, ledger rows E21 to E42: every result with its evidence status, run names and caveats,
   including withdrawn claims.
2. `docs/notebook/2026-10-19-selective-recursion-and-broader-teaching.md`: the narrative record, newest sections
   near the top of each part.
3. `docs/research-plans/2026-10-22-research-program-and-idea-backlog.md`: ranked backlog and the reflection log
   (step backs at 2026-10-22 02:10, 06:20, 11:00; 2026-10-23 05:40, 10:50).
4. `slides/2026-10-22-bootstrapping-and-boundaries/research-update.pdf` (9 slides, notes in `speaker-notes.json`
   and `research-update.pdfpc`): the advisor summary.

## What this allocation established (details and caveats in the ledger)

- Libraries can be bootstrapped, and a 0.6B caller uses them: MRCR's library was written by a thinking 8B
  (2 of 5 seeds) and a 4-bit 14B (2 of 5); 1.000 held out to 1M tokens (E21). MRCR is program-solvable.
- Proposer training works on generated families: SFT on the 8B's verified code, then RL with the verified
  teaching score as reward; the round-four 4B is statistically level with the thinking 8B and 14B on held-out
  discovery (42, 45, 47 of 74) at a twentieth of the tokens (E26 to E26g).
- Verification is cheap where answers are exact: three shown examples plus a label-free validity check accept
  libraries that generalize (127 of 129); pre-registered, a 4B with eight samples and three labels bootstraps as
  many correct libraries as the 8B with thirty (12 vs 13 of 30) at 7 percent of the tokens (E37, E39). Agreement
  among candidates is not a substitute (RL-trained proposers repeat wrong programs).
- The helper gap: more helper-family teacher data brings the 4B to the 8B's mean on helper families (0.46) at a
  non-significant cost in code families (E36, E41); RL seed variance is about four families of 74.
- Retrieval libraries carry the long-document results: MuSiQue two-round BM25 library; LongBench-v2 retrieve then
  read 0.427 vs direct 0.356 on all 503 (E22, E32).
- Boundaries: OOLONG, the study's own family, is not bootstrapped as a whole (11 question templates from 30
  examples); narrowed to one task group, the 14B writes a counting library level with the shipped one held out,
  but not timeline or user (E35, E35c). Recursion does not help a small model on MuSiQue (E23).
- Documentation: one usage line decides whether an untrained caller uses a library (E25c, E25d).

## Where everything lives

| What | Where |
| --- | --- |
| Code (experiments) | `experiments/selective_recursion_20261019/` (code map in its `EXPERIMENT.md`; pipelines in `scripts/`) |
| Job queue and launcher | `experiments/ops/` (`queue.py`, `launch.py`, `status.py`; README) |
| Run store (immutable outputs) | `/project/alex_phd/runs/selective-recursion-20261019/` (symlinks into the home store) |
| Run homes, adapters, SFT data | `/home/atowell/research-runs/selective-recursion-20261019/` (`adapters/`, `sft-data/`, RL run dirs with `step-*`) |
| Final RL adapters | round four `e26g-proposer-rl-r4/step-40` (served as `proposer-rl-5-40`); round five `e36-proposer-rl-r5/step-40`; E41 A `e41-rl-r5sft-a/step-40`, B `e41-rl-r5sft-b/step-40` |
| SFT-only adapters | `adapters/proposer-4b-r{3,4,5}/final`, `adapters/proposer-4b-8bcode{,-r2}/final` |
| Environments | serving `/project/alex_phd/envs/vllm` (vLLM 0.27.1; put its `bin` on PATH for ninja); training `/project/alex_phd/envs/selective-recursion-train-5b1e2b4`; evaluation the repo `.venv` (`uv sync --extra dev`) |
| Models | `/project/alex_phd/research-cache/models/` (Qwen3-4B-Instruct-2507, Qwen3-8B-AWQ, Qwen3-14B-AWQ, Qwen3-0.6B) |
| Datasets | `/project/alex_phd/research-cache/datasets/` and `derived/` (OOLONG-synth, MuSiQue, LongBench-v2 truncation, MRCR, HotpotQA; each with an acquisition record) |
| Idea registry | private `queelius/idea-lab` (`/project/alex_phd/repos/idea-lab`), run by its own session; this session only appends to `inbox.md` |
| GPU idleness log | `/project/alex_phd/runs/selective-recursion-20261019/IDLE-LOG.txt` |

Servers at handoff (all `--api-key local`): GPU0 8002 Qwen3-8B-AWQ (thinking), 8004 Qwen3-0.6B with the
ctrl-06-onecell-v2 adapters, 8006 Qwen3-4B Instruct (32k); GPU1 8001 Qwen3-4B Instruct with LoRA (64k; proposer
adapters hot-loaded), 8008 Qwen3-14B-AWQ (fp8 KV). Restart commands are in the pipeline scripts
(`scripts/e27b_gpu1_servers.sh`, `scripts/e42_round7b.sh`).

## Open questions, ranked

1. Does the 4B's helper skill reach the 8B's breadth (semantic families at 0.7: 3 vs 5 of 8) with more helper
   question types (E42, round seven), and can code generalization be kept by weighting code pairs?
2. Label needs on noisy families: graded verification for MuSiQue- and OOLONG-like families (E37 shows exact
   families need about three labels, noisy ones tens).
3. Breadth: decompose a broad family (OOLONG) into question templates automatically, then bootstrap each
   (E35c shows narrowing helps only where the questions parse simply).
4. RL recipe comparisons with at least two RL seeds (E41: seed variance is about the size of the effects).
5. Idea lab I-018 (few decisions carry the distilled caller's gain), I-004 (keeping input-conditioned routes
   through RL); the RL-correlated-errors question sent to the lab inbox on 23 October.

## Handoff status

(Filled in at the end of the allocation: which runs finished, which were cut off, and the last commit.)
