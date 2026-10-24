---
date: 2026-10-24
status: allocation_ending
allocation: job 6599, two A100 40GB (MIG 7g.40gb each), ends 2026-10-25 00:52 UTC
research_cutoff: 2026-10-23 18:20 UTC (job cancelled early; see Handoff status)
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

## Working principles (user's standing instructions)

- **Independent review (avoid monoculture).** Get a read-only Codex review of each new design before launch, of each
  pre-registered analysis before unblinding, of every changed headline claim, and of what to run next.
  - Command: `codex exec -s read-only -C <worktree> --skip-git-repo-check -o <run store>/codex-reviews/<date>-<topic>.md "<prompt>"`.
  - Verify its findings before acting, and record what changed and any disagreement.
  - Reviews so far are in `codex-reviews/` in the run store.
- **Never leave a turn unwatched.** While jobs run, keep a monitor on `queue/dispatcher.log`, covering started, FAILED,
  SLOW and IDLE ALARM lines. GPU1 idled 70 minutes on 2026-10-24 because this was missing.

## Start here: handoff from job 6618 (latest; supersedes the 2026-10-23 handoff below)

Use the newest checkpoint manifest, `RUNS/CHECKPOINT-2026-10-24-*.json`, written by `experiments/ops/checkpoint.py`.
It records every server's command line and device, the queue in full, and per-run progress.
`RUNS` = `/project/alex_phd/runs/selective-recursion-20261019`; `H` = `/home/atowell/research-runs/selective-recursion-20261019`.

### State at the 08:50 UTC checkpoint

- **Finished, not yet analysed:**
  - E58 arm A (`e58a-4br5b-s16-best8`, 141/141).
  - The E58 secondary arm (`e58a2-4br4-s9-best8`, 141/141).
  - E54's finished runs (labelled4 s5, s7, s8; shown4 s5, s6, s8, s9).
- **Being scored:** E51r seed 17 (`H/e51r-4br5b-s17-analysis`).
- **Running:**
  - E58 arm B (`e58b-8b-standard`, 94/141 families). It resumes per family: finished families are skipped, and
    unfinished ones continue from their saved candidates.
  - The E57 pools (`e57-mrcr-8b-pool-s10..14`, on the second 8B server, port 8010).
- **Queued:**
  - E54 labelled4 s6, s9 and shown4 s7, after `e58b`. They resume from saved candidates.
  - The E57 labelled and shown branches, each after its pool.
- **Servers:**
  - GPU0: 8002 8B (E58 B, E54), 8004 0.6B, 8006 4B helper (used by every verification).
  - GPU1: 8001 4B with LoRA adapters, 8010 second 8B for E57 (`--gpu-memory-utilization 0.44`). The 14B (8008) is
    stopped; it is only needed for new 14B work.
- **No-8010 case:** if a new allocation has only one 8B, rewrite the E57 specs' `8010` to `8002`. Keep pool and both
  branches of a seed on the same server.

### On return: finish these analyses first

1. Read `H/codex-reviews/2026-10-24-design-and-next-experiments.md`. It is a blind review of the E58 analysis code
   against its pre-registration, of E57's design, and of proposed experiments. Fix anything it flags in the E58
   analysis before unblinding. Record such a fix in `docs/notebook/assets-e58-preregistration.md` as an amendment made
   before unblinding.
2. E58 primary, as pre-registered (report whatever the outcome), from `experiments/selective_recursion_20261019/`:
   `e39_analysis.py --a RUNS/e58a-4br5b-s16-best8 --b RUNS/e58b-8b-standard --families H/e58-prereg/families.txt --work H/e58-analysis --margin 0.10`.
   Secondary: the same with `--a RUNS/e58a2-4br4-s9-best8 --work H/e58a2-analysis`.
3. E51r: pool seed 17 with seed 9 (16 vs 16) and E51 (32 vs 29 of 60).
4. E54 and E57 held out: `mrcr_heldout_batch.py --work H/mrcr-heldout/work --out H/mrcr-heldout/<name>.jsonl RUNS/e54-* RUNS/e57-*-labelled-* RUNS/e57-*-shown-*`.
   E57 compares labelled and shown branches per seed on the same pool.
5. Record in `docs/RESULTS.md`, the notebook, the E39 pre-registration addenda and the deck.
6. Run `collect_results.py`, get a Codex review of the changed claims, then push.

### Finding new work (ranked; re-rank after E58 and the Codex review)

0. **Template robustness (E59, E60; running since 09:05 UTC on lane `gpu1b`, port 8001).**
   E59 found the accepted libraries template-bound: 55 of 55 survive 5 times longer inputs, but 3 of 55 a
   reworded question and 11 of 55 a changed label or field order. E60 tests two fixes against frozen E39 A:
   mixed-wording teaching (M) and a contract note (C).
   - Score with `e59_stress.py --prefix mix: ...` for M, and without a prefix for C and A0, on all six
     conditions, with output to `H/e60/rows.jsonl`.
   - The primary outcome is robust yield over all 30 families, as pre-registered.
   - Next candidates: Codex's repair-of-frozen-successes arm (96 candidates), the M+C combination, and the 8B
     under mixed teaching.

1. **Act on E58's outcome.**
   - If the 4B three-label protocol is non-inferior: freeze accepted libraries and test generalization on a withheld
     stress panel (longer inputs, paraphrased questions, format variation, edge cases). Then bootstrap one genuinely
     new real exact-answer family. This is the earlier Codex review's priority 2.
   - If it is not non-inferior: find which family types the 4B loses, and test whether best-of-32 (E51) closes the
     gap on the 141-family panel.
2. **Repair attribution (E57).** With 5 seeds, read it as an estimate, not a test. If labelled and shown repair
   differ, repeat E57 on another day and with the 14B. The 14B's round-0 success varied across days (7/41 vs 1/35).
3. **Label needs on noisy families.** E49's curve is specific to OOLONG counting, with prefiltered candidates. Rerun
   the curve without the prefilter, and on a second noisy family.
4. **Helper breadth.** The round-seven recipe (extra helper question types) needs at least three RL seeds before any
   recipe claim; RL seed variance dominates recipe effects.
5. **Idea lab items:** I-018 (whether a few decisions carry the distilled caller's gain), I-004 (keeping
   input-conditioned routes through RL), and RL-correlated errors.
6. **Codex proposals:** whatever the review above ranks highest, verified first.

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

## Open questions, ranked (as of 2026-10-23; superseded by "Finding new work" above)

1. Does the 4B's helper skill reach the 8B's breadth (semantic families at 0.7: 3 vs 5 of 8) with more helper
   question types (E42, round seven), and can code generalization be kept by weighting code pairs?
2. Label needs on noisy families: graded verification for MuSiQue- and OOLONG-like families (E37 shows exact
   families need about three labels, noisy ones tens).
3. Breadth: decompose a broad family (OOLONG) into question templates automatically, then bootstrap each
   (E35c shows narrowing helps only where the questions parse simply).
4. RL recipe comparisons with at least two RL seeds (E41: seed variance is about the size of the effects).
5. Idea lab I-018 (few decisions carry the distilled caller's gain), I-004 (keeping input-conditioned routes
   through RL); the RL-correlated-errors question sent to the lab inbox on 23 October.

## Handoff status (2026-10-23 18:20 UTC; job 6599 cancelled early for the new gpu-preempt access)

The user is moving to the new preemptible partition (`gpu-preempt`, QoS `gpu-preempt-qos`, two GPUs on top of the
normal two), which required cancelling job 6599 about 30 hours early. Work was checkpointed first:

- Finished and recorded before the stop: E35 to E42, E39 and its replication (pooled 26 vs 29 of 60).
- E43 A (round four's RL repeated with seed 6) finished training; adapter `e43-rl-r4-s6/step-40`, served name
  `proposer-rl-6-40`. Its held-out evaluations are queued, not run.
- E43 B (round seven's RL with seed 27) was never started. Optional.
- E39 arm C (exploratory: the 8B under the three-label protocol) paused at 7 of 30 families (71 of 240 candidates);
  it resumes from its saved candidates.

### Machine-readable state

- `docs/checkpoints/2026-10-23-job6599.json` (copy of `RUNS/CHECKPOINT-2026-10-23.json`), written by
  `experiments/ops/checkpoint.py`: git HEAD, Slurm job facts, every server with its device, full command line and
  served models or adapters, the queue (pending and running specs in full), environment and package versions, and
  the progress of the in-flight runs.
- `RUNS/RESUME-HERE.md`: an agent-neutral pointer back to this document.
- `experiments/selective_recursion_20261019/results/`: the result data behind the ledger, in git (rerun
  `collect_results.py` after new runs finish to refresh it).
- `RUNS/queue/specs-archive-2026-10-23/`: every job spec of the campaign (the originals lived in a temporary
  scratch directory).
- Versions at the stop: Python 3.12.12; serving vLLM 0.27.1, torch 2.13.0+cu130, transformers 5.15.1; training torch
  2.13.0+cu130, transformers 5.15.1, peft 0.20.0; uv 0.9.7; NVIDIA driver 580.126.09.

### To resume

1. Start the model servers. Exact command lines, devices and loaded adapters at the stop are in
   `/project/alex_phd/runs/selective-recursion-20261019/serving/servers-at-handoff-2026-10-23.txt`. Put
   `/project/alex_phd/envs/vllm/bin` on PATH, and set `CUDA_VISIBLE_DEVICES` to the new job's GPU for each lane
   (GPU0 lane: 8002 8B, 8004 0.6B, 8006 4B; GPU1 lane: 8001 4B with LoRA). Check each server answers a real request.
   Lessons from the 2026-10-23 restart: launch servers that take adapters (8001, 8004) with
   `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`, or `load_lora_adapter` returns Not Found; start the GPU0 servers one at a
   time and wait for each to answer (8006 ran out of memory twice while the 8B was still starting; it now runs at
   `--gpu-memory-utilization 0.36`). Also start 8001 with `--max-loras 4` (the default holds one adapter at a time, so runs that
   sample different proposer checkpoints at once are serialized; seen 2026-10-24 when three checkpoints shared it).
2. On 8001, load the adapter the pending evaluations use:
   `curl -s -X POST http://127.0.0.1:8001/v1/load_lora_adapter -H "Authorization: Bearer local" -H "Content-Type: application/json" -d '{"lora_name": "proposer-rl-6-40", "lora_path": "/home/atowell/research-runs/selective-recursion-20261019/e43-rl-r4-s6/step-40"}'`
3. Start the dispatcher from the repository root: `nohup .venv/bin/python experiments/ops/queue.py run > /dev/null 2>&1 &`.
   It launches what is in `RUNS/queue/pending/`: `e39c-8b-best8` (resumes) and the E43 A evaluations
   (`e21-boot-4br4-s6-s0`, `e26-disc-4br4-s6-heldout{,-r2}`). Lanes `gpu0` and `gpu1` assume both GPUs are on one
   node with the ports above; for jobs on separate nodes, start one server set and one dispatcher per node.
4. When they finish: compare E43 A with round four (42 of 74) as the second RL seed of the headline result; score
   E39 arm C with `e39_analysis.py --a RUNS/e39c-8b-best8 --b RUNS/e39b-8b-standard ...` (C against B) and the
   arm A rule on both A and C (A against C).

### Resumed 2026-10-23 ~18:55 UTC on job 6618 (an21, partition gpu, ends 2026-10-24 18:38 UTC)

Servers 8001, 8002, 8004, 8006 restarted (logs in `RUNS/serving/restart-2026-10-23/`), `proposer-rl-6-40` loaded on
8001, the dispatcher restarted on the default queue; the four pending jobs resumed. A second allocation on the
preemptible partition must use its own queue (`RLM_QUEUE`).
