# Early transfer of the existing familiar-goal RL endpoints

Question: does the one-update improvement on eight familiar TRAIN goals carry to the already frozen fresh diagnostic B goals, and does it depend on the interface used for training? This adds no optimization and does not wait for fresh-A training.

Four new cells cross the actual raw-trained and binder-trained one-step actors with raw and binder execution. Each uses the same eight B tasks and seeds 202609280900/901: 16 attempts per cell, 64 new attempts total. The two unchanged public-cp23 B controls belong to the accepted fresh-RL campaign and are reused, never rerun or duplicated here. B remains an exposed official TRAIN diagnostic, not held-out confirmation.

The thin private adapter reuses `rl_fresh_20260928` plan construction and the original `rl_resume_20260928` native collection/metrics/replay loop. Only endpoint admission and truthful cross-interface lineage are added. It does not modify either source. Runtime retains world42, original prompts, FP16 base/FP32 LoRA, temperature0.5/top-p1/top-k0, 96 calls, 8192 total output tokens, 256 tokens per response and 8192 input-plus-output context. Binder arguments remain deterministic transitions; original sampled tokens remain in receipts. Both checkpoints are admitted from actual usable SUMMARY/STATE/COMMIT and adapter bytes, not predicted hashes.

## Parent dispatch

`R` below is `/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`; `E` is this worktree's `experiments/selective_delegation`. Use the accepted GPU interpreter at `/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python`.

The generic receipt is `R/textcraft-rl-transfer-20260928-001/PREPARED-JOBS.json`. Its `jobs` list has four GPU descriptors and a final CPU comparison. Only scientific GPU descriptors have `output`; each has `name`, `argv`, `cap_seconds` and `pins`. No launch function is included. Root alone installs descriptors in its authenticated queue.

Individual scientific commands are the accepted interpreter followed by:

```sh
E/rl_transfer_20260928/collect.py --actor raw --mode raw
E/rl_transfer_20260928/collect.py --actor binder --mode binder
E/rl_transfer_20260928/collect.py --actor raw --mode binder
E/rl_transfer_20260928/collect.py --actor binder --mode raw
```

Paths represented by `E` must be expanded, not supplied literally. The immutable default study is `R/textcraft-rl-transfer-20260928-001`; `--study` makes it explicit. `--prepare-only` regenerates and checks the same plan without loading a model. Do not alter these accepted descriptors after dispatch.

Ninety-minute scientific caps and 95-minute owner caps match existing B controls. Four cells should use roughly100–200 GPU minutes from the existing depth-conditioned call costs; this is an estimate, not guaranteed completion. Maximum descriptor caps total6.5 hours including the10-minute CPU comparison. No retry, optimizer, favorable-checkpoint selection or partial-result substitution is added. The existing coordinator lock and authenticated release behavior are inherited unchanged.

## Why warm B can move earlier unchanged

`rl_fresh_20260928/stage.py:resolve` directly selects `f.WARM` for `kind=readout, actor=warm`; only the RL and SFT branches read trained endpoints. The collector then consumes the frozen diagnostic dataset and fixed readout seeds. No A collection, training SUMMARY, checkpoint or reward is read. Existing warm pins include training source/data files, not future A outputs. A focused test replaces `endpoint()` with a failing function and still resolves both warm commands in an empty study directory.

The preparation receipt records `warm_controls_independent=true`, source hashes, the accepted original campaign receipt hash and both exact original descriptors as evidence. They are absent from the new `jobs` list. Root may move each original warm descriptor earlier once without changing it.

## Analysis and limits

Every complete readout receives independent native replay inside its collector. The CPU comparison verifies native receipt hashes, task/seed pairing, numerical/runtime contracts and actual actor lineage. It reports each trained actor minus the matching-interface warm control, execution-interface differences, trained-actor differences, and the training-interface-by-execution interaction. Missing or incomplete cells remain unknown, with no effect estimate. Successes, calls, errors, sampled-token entropy, response diversity and terminal status counts accompany paired task-cluster intervals.

Only eight task clusters, shared world/recipes, prior campaign adaptation and repeated diagnostic exposure limit inference. Identical seeds do not imply identical trajectories after policy divergence. Training token doses and trajectories differ between actors. There is no extra-SFT control in this probe; transfer is not proof that RL beats another gradient update, nor evidence of recursive decomposition.

The CPU fixture uses the first frozen B public native trace, real tokenizer, actual `MetricsClient`, saved request/token receipts and independent native replay. Binder receives one intentionally wrong ingredient payload, which remains in original sampled tokens while execution succeeds. Scripted generation and uniform synthetic score tensors test the seam only; they are not model probabilities, entropy measurements or learned performance. Scientific analysis explicitly rejects fixture-marked plans.
