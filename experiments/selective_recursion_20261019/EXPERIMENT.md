# E06/E07: selective recursion, teaching and cost-aware RL of a 4B RLM controller

**Question.** Can a small controller (Qwen3-4B-Instruct-2507) be taught the RLM decomposition
routine from a scripted expert's native-replay trajectories and then improve it with on-policy RL
under a verifiable, cost-aware reward (E06)? And can RL from a route-balanced warm start learn
*when not to decompose* (E07)?

**Design.** Two task families: synthetic event logs (8 to 512 records, five question types,
generator in `taskgen.py`, verifier exact) and OOLONG-synth validation rows up to 32k tokens with a
per-length window split (even windows teach/train, odd windows held out). Policies: direct
answering, a programmed split, the untrained controller, scripted experts (`expert.py`,
`expert_cells/`, `oolong_cells/`), SFT warm starts (`sft_train.py`, LoRA r=32, lr 1e-4, 2 epochs)
and RL (`rl_train.py`: REINFORCE with leave-one-out baseline, 16 tasks x 4 samples per update,
30 updates, lr 2e-5, T=1, adapters hot-swapped into vLLM). Rewards: correct minus lambda times
capped tokens per 10k (`--cost-mode absolute`), or correct minus lambda times tokens over the
task group's mean (`--cost-mode relative`); optional k3 KL to the warm start (`--kl-coef`).

**Immutable inputs.** Base model snapshot and OOLONG export with hashes in `manifest.json`;
teaching sets `teach-003`, `teach-004`, `teach-oolong-004`, `teach-oolong-005` and the balanced
`teach-balanced-007` (`balance_routes.py`) with hashes in `manifest.json`.

**Metrics.** Accuracy on held-out rows (72-row OOLONG sample for screening, all 293 rows for
claims; 120-row event-log dev split; 20-task 1024-record check), endpoint-reported tokens and
helper calls, and route shares classified from traces (`route_analysis.py`, `route_shares.py`).

**Compute.** Two A100 40GB MIG instances (GPU 0 training, GPU 1 vLLM). An RL run is about 2.5 h.

**Status.** E06 replicated (four RL runs); E07 closed (route RL collapses; see the ledger). The study then
moved the routine into a library (E08 to E20) and, on 22 October, to where libraries come from and where they
help (E21 to E34; section below). The narrative record is
`docs/notebook/2026-10-19-selective-recursion-and-broader-teaching.md`; the claims
are in `docs/RESULTS.md`. Run outputs, traces and adapters are in the run store named in
`manifest.json` (not in git); pipeline scripts there are copied under `scripts/` (snapshot of 22 October; see `scripts/README.md`).

## Environments (rebuilt 2026-10-21)

- **Training** (SFT, RL, embeddings): `/project/alex_phd/envs/selective-recursion-train-5b1e2b4`,
  Python 3.12.12, built with uv from `env/training-requirements-monorepo.txt` (frozen; torch
  2.13.0+cu130, transformers 5.15.1, peft 0.20.0) plus this repository's `rlm` in editable mode.
  Rebuild:

  ```bash
  uv venv --python 3.12.12 /project/alex_phd/envs/selective-recursion-train-<commit>
  uv pip install --python <env>/bin/python -r env/training-requirements-monorepo.txt \
      --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
  uv pip install --python <env>/bin/python --no-deps -e <repo root>
  ```

  Verified by a one-epoch 0.6B LoRA fine-tune. It replaces the training venv inside the archived
  `rlm-bootstrap` clone (`training-requirements.txt` is that environment's original freeze, kept for
  provenance; its `rlm` and `rlm-bootstrap` pins are local packages).
- **Serving**: `/project/alex_phd/envs/vllm` (vLLM 0.27.1). Hybrid-thinking callers (Qwen3 0.6B,
  1.7B, 8B) are served with `--default-chat-template-kwargs '{"enable_thinking": false}'` and must be
  trained with `SR_NO_THINK=1` so the training prompt matches; non-Qwen callers need
  `SR_LORA_TARGETS` (Phi-4-mini: `qkv_proj,o_proj,gate_up_proj,down_proj`).
- **Evaluation and orchestration**: the repository `.venv` (`uv sync --extra dev`).

## Extensions of 21 and 22 October (E21 to E34)

| Purpose | Code |
| --- | --- |
| New families: OpenAI MRCR, MuSiQue in long haystacks, LongBench-v2 (exact Qwen3 truncation) | `mrcr_tasks.py`, `musique_tasks.py`, `longbench_tasks.py`, `make_longbench_trunc.py` |
| Evaluate direct and RLM policies on any family; run a library routine alone (optionally with the kernel's fault semantics) | `family_eval.py`, `run_routine.py` (`--kernel-semantics`) |
| Bootstrap a library from teaching examples, verify in a subprocess, repair; optional primitives and worked example | `bootstrap_library.py`, `bootstrap_verify_worker.py` (`BOOT_PRIMITIVES`, `BOOT_EXEMPLAR`, `KERNEL_SEMANTICS`) |
| Generated families for library discovery (formats x questions x answer formats; helper-dependent questions) | `family_gen.py`, `discovery_bench.py` |
| Proposer training: SFT pairs from verified libraries; on-policy RL with the verified teaching score as reward | `build_proposer_sft.py`, `sft_train.py`, `proposer_rl.py` (`--sample-ids`, `--families mrcr`, `--with-helper-families`) |
| Routine search and ablations | `evolve_routine.py` (reflective or random), `musique_ablate.py` (retrieval arms, reader server, iterative rounds) |
| Reference, bootstrapped and evolved libraries | `skills/reference/`, `skills/bootstrapped/`, `skills/evolved/`, `skills/primitives/`, `skills/exemplars/`; selected with `RLM_SKILLS_VARIANT` (`skills/registry.py`) |

Runs were launched through the job queue and checked launcher in `../ops/` (see its README). Results, with
caveats and withdrawn claims, are ledger rows E21 to E34 in `docs/RESULTS.md`.
