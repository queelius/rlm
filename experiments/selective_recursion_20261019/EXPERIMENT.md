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

**Status.** E06 replicated (four RL runs); E07 in progress. The narrative record is
`docs/research-checkpoints/2026-10-19-selective-recursion-and-broader-teaching.md`; the claims
are in `docs/RESULTS.md`. Run outputs, traces and adapters are in the run store named in
`manifest.json` (not in git); pipeline scripts there are copied under `scripts/` at campaign end.
