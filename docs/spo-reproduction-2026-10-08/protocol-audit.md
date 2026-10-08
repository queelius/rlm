# SPO-chain int5 / RhoMath 1.1B / GSM8K protocol audit

Audit date: 2026-10-08. Official source inspected at commit `6c5f94723c20e4de68578545b8bb98186e956797` (2025-09-19). CPU-only audit and asset acquisition; no source checkout modifications or GPU work.

## Decision

The historical Figure 3 target is **56.7% GSM8K test pass@1**, compared with SFT **40.3%** and published GRPO **44.6%**. The author evaluation log identifies the precise matching result: **0.5672384382107657 at checkpoint `ckpt--iter_0690--epoch_2.00--step_11040`**. Current README training defaults differ materially from that historical training recipe; use the recovered historical config rather than claim those defaults are exact reproduction. The native current test evaluation remains semantically compatible with the historical test evaluation.

Sources: [official Figure 3 bar chart](https://github.com/AIFrameResearch/SPO/blob/6c5f94723c20e4de68578545b8bb98186e956797/figures/compare-baselines-short-CoT.png), [paper Section 6](https://arxiv.org/html/2505.23564v2#S6), [author target checkpoint evaluation run](https://wandb.ai/my-wandb-team/SPO-experiments/runs/ytu1arac).

## Immutable inputs and recovered artifacts

- SFT input: `realtreetune/rho-1b-sft-GSM8K`, HF revision `b28fda3216f8178c1f37c3c764196e3ca2439a30`.
- Author released policy: `gyr66/spo-chain-int5-rho1.1B-gsm8k`, revision `a806ef8ef83beca009725da026fb7bde467a9dde`. Model card supplies no training/selection provenance beyond a generated template; correspondence to checkpoint 690 is supported by the matching author evaluation metric, not proved by Hub metadata.
- Original dataset extracted safely to `/project/alex_phd/research-cache/spo/assets/gsm8k-author-20241002/gsm8k`. The archive contains the author's existing validation split; do not recreate one from today's HF dataset.
- Original training config: [author-training-config.json](evidence/author-training-config.json). It records original run `rbtvnfqm`; author local model path is replaced by the identical HF model name, and placeholder API keys remain `EMPTY`. Its original `directory` and `exp_name` require local staging overrides.
- Original evaluation pipelines: [author-evaluation-pipelines.json](evidence/author-evaluation-pipelines.json), from run `ytu1arac`.
- Later 1000-iteration rerun config: [author-training-rerun-config.json](evidence/author-training-rerun-config.json), from `yc23440y`; distinguish this from original Figure 3 training.
- Public W&B GraphQL config/summary receipt: `/project/alex_phd/research-cache/spo/data/author-wandb-chain-runs.json`, SHA256 `8b11ac87b4df716ee8b5f5a3c4a6349d3d826495a52ab4cd5d6f8f0b9a5f6a5b`. Downloaded anonymously; no credential or dotenv files were retrieved.

## Recipe and inheritance

Current documented merge order is `polIter_rho1bSft2_spo_chain_GSM8K.jsonnet` + `episode_generators/interval5.jsonnet` + `gpus/gpu_0.jsonnet` (README:79–85). GSM8K config imports chain-MATH, which imports rho PPO-MATH and `trainers/no_critic.jsonnet`; PPO-MATH imports `trainers/ppo_MATH.jsonnet`, then overrides actor DeepSpeed to zero stage 0 and imports `lam1`, `refKl0.0001`, `klLoss`; final GSM8K evaluation import replaces the MATH pipelines. See configs GSM8K:1–22, chain-MATH:1–35/99, PPO-MATH:20–25/111–123/186–189.

Shared scientific settings: 64 questions × 8 trajectories = 512 episodes/iteration; 9 MC continuations/state; every fifth low-probability cutpoint; threshold 0.9; train temperature 0.6/top-p 0.9; max generated tokens 1024, prompt+generation context 2047; two epochs/iteration; global batch 64; AdamW LR 1e-6, KL coefficient 1e-4, gamma/lambda 1, clipped PPO ratio 0.2; bf16 and FlashAttention2; actor gradient checkpointing; frozen reference model, no critic. These correspond to [paper Appendix H](https://arxiv.org/html/2505.23564v2#A8), which explicitly specifies a single A100 40GB for this experiment. Exact optimizer/scheduler settings remain in the recovered config.

## Material differences: original experiment versus current source

| Field | Original `rbtvnfqm` / `2xw628n7` | Current README merge | Evidence |
|---|---|---|---|
| KL loss maximum clipping | 10 | 1e9 | recovered config:18; `trainers/klLoss.jsonnet`:5 |
| Unfinished response reward | -1 | 0 | recovered config:370; `episode_generators/math_episode_generator.jsonnet`:30 |
| Training microbatch | 16 | 8 | recovered config:53; PPO-MATH:119 |
| Post-generation sequence limit | 2048 | null | recovered config:419; PPO-MATH:49 |
| vLLM requested memory / swap | auto / 8GB | 0.3 / 12GB | recovered config:356–358/427; PPO-MATH:42/47 |
| Evaluation pipeline definitions | test, validation, small train portion | test only | recovered config:491/558/625; GSM8K-eval:79–81 |
| Multiple-conclusion penalty flag | true, penalty -2 | defaults false | recovered config:426/429; `math_episode_generator.py`:121–122 |

All recorded scientific constructor keys are accepted by the inspected source. However, accepted keys do not guarantee historical behavior: the current MC subclass computes reward directly at `math_episode_generator_with_mc_advantages.py`:639–648, bypassing the parent's multiple-conclusion penalty branch (`math_episode_generator.py`:200–201). Preserve the recorded flag in the overlay and label the missing historical source revision as a limitation.

**Correction from native evidence, 09:35 UTC:** bypassing that parent branch does **not** mean the repeated-answer penalty is absent. The shared `MATHRewardFunction.__call__` at `math_episode_generator.py`:53–56 independently returns `(-2, False)` for more than one `####` marker, before grading. We reproduced this return on the exact saved iteration9/episode366 response; its trajectory, episode score, and training data agree. The earlier audit stopped one layer too early. Preserve the configured flag and missing historical-source caveat, but do not count an absent multiple-conclusion penalty as an established mismatch. No live source was changed. See [native fixture evidence](evidence/progress-176-updates.json).

The published `max_step_for_value_estimation:25` is **not an active MC-state limit** in current code: old step-limit code is commented at MC-generator:962–963, while lines 985–1000 generate requests for every selected cutpoint. Do not estimate runtime assuming only 25 states.

## Evaluation metric and effective decoding

Pass@1 is `correct_frac` (alias `exact_match_frac`): average per-problem fraction of correct answers among the 16 independently sampled responses. With 1319 test problems, the full evaluation has 21,104 candidate responses. `once_hit` / `exact_match` means at least one correct among 16 and is a different metric; majority vote is also separate. See `src/treetune/tasks/gsm8k.py`:174–215. The target checkpoint's once-hit is about 79.6%, while pass@1 is 56.724%; mixing these would invalidate the comparison.

Native scoring extracts the final text after the last `####`, trims it, removes commas from the prediction, and compares strings case-insensitively. Missing marker is incorrect; this is not a generalized numeric or symbolic answer checker (GSM8K task:42–51/144–159).

Evaluation temperature is 0.35, 16 samples, context 2047, max new tokens 1024, seed 42, stop string `\n\n\nProblem:`. **Although paper/config say top-p 0.9, the actual historical and current evaluation template omits the top-p argument**, so guidance `gen` defaults to **1.0**. Evidence: `prompt_library/generic_GSM8K_step_by_step.jsonnet`:1, efficient expander:357–358, `src/guidance/library/_gen.py`:20. Training template includes top-p correctly. Keep native effective 1.0 for the exact author reference; a 0.9 run is a separately labeled paper-stated decoding variant.

Compiled current test pipeline versus recovered historical test pipeline: same task, prompt, answer extraction, sample count, temperature, effective top-p, limits, stop and strategy seed. Remaining differences are author local paths, `max_depth` 10→100 (no effect here because the selected efficient expander sets every child `stop_text:None`, expansion.py:392, terminating expansion after one completion), `logprobs` 1→0 (current efficient expander forces 0 at expansion.py:287; scoring ignores it), and historical extra validation/train pipelines. Therefore the current full-test reference comparison is valid; it does not by itself validate historical training equivalence.

## Seeds, iterations, selection and runtime

Original configs record training/global seed **42** and schedule length **1000 iterations**. On one process, prompt shuffling and trajectory-server seed vary as `42 + iteration`; MC seed adds one (on-policy generator:291/330–331; MC generator:112). Evaluation strategy seed is 42; runtime pipeline constructor also passes fixed seed 2746318213 (runtime:349). No multi-seed aggregate or a historical git commit was recovered. Public original run file listing has no `wandb-metadata.json`; config has no git/version fields.

Current runtime evaluates after training iterations 0,10,...,990 (runtime:262–265), so iteration 0 is **after an update**, not the initial SFT policy. Evaluate SFT separately. Original training summaries contain validation-only iteration metrics despite defining three pipelines; published Figure 3 validation chronology cannot be reconstructed by blindly running current test-only defaults.

The original author training is split across [rbtvnfqm](https://wandb.ai/my-wandb-team/SPO-experiments/runs/rbtvnfqm) and [2xw628n7](https://wandb.ai/my-wandb-team/SPO-experiments/runs/2xw628n7), ending with checkpoint 690. The target test result occurs at that endpoint in [ytu1arac](https://wandb.ai/my-wandb-team/SPO-experiments/runs/ytu1arac). Validation maximum among recovered original training checkpoints occurs near 520, not 690. No evidence establishes that 690 was a predetermined cap, a validation-selected checkpoint, or retrospectively selected for test performance; preserve this as an unresolved selection gap.

Author elapsed `_runtime` receipts are 91,301.54s and 124,193.31s for the two original segments: **59.86h total**, plus target evaluation sweep 7,220.51s. A later [1000-iteration rerun](https://wandb.ai/my-wandb-team/SPO-experiments/runs/yc23440y) records **101.93h**, last test pass@1 57.16%, best reported test 57.42%. These are observed elapsed run durations, not guaranteed runtime on the present allocation. A 47h allocation may require a checkpoint and continuation to reproduce the original 690-iteration endpoint.

Keep `num_iterations:1000` when bounding a partial attempt: it defines 16,000 planned optimizer updates and 480 warmup updates, followed by WarmupDecayLR. Shrinking it changes the LR trajectory. Current `early_stop_iteration` stops the loop without changing that schedule (runtime:204–212; PPO trainer:327–332; trainer config warmup:74; scheduler config:2).

Checkpoint settings save full state every 10 iterations and retain model checkpoints every 40 (PPO-MATH:122–123; PPO trainer:432–451/2560–2570). `checkpoints/final` is a text pointer to the last permanent checkpoint, not a metric-selected best (PPO trainer:2684–2687). APP_MINIMIZE_STORED_FILES cleanup strips optimizer states when the loop completes (runtime:274–282); protect a verified resumable checkpoint before an allocation or bounded-run handoff.

Operational update: [train-historical-paths.jsonnet](evidence/train-historical-paths.jsonnet) fixes `early_stop_iteration:690` prospectively while preserving `num_iterations:1000`. [training-launch.json](evidence/training-launch.json) records prepared argv/environment and resume rules. `APP_MINIMIZE_STORED_FILES=False` prevents end-of-loop stripping, so normal completion at690 preserves its full state. The PPO override final-save only writes a pointer. However, automatic permanent saves unconditionally clean prior checkpoints **before** writing the next (`deepspeed_policy_trainer.py`:397–408), independent of APP_MINIMIZE_STORED_FILES: interruption during this gap can remove the previous resumable state. If continuing after690, preserve that complete checkpoint externally before700 because690 is not divisible by40. Default relaunch resumes automatically from latest full state; never use `force_rerun=True` as a resume flag, since it disables checkpoint loading. A completed run's final marker causes default launch to skip, so continuation must use a separate owner/output with a preserved full690 checkpoint and no final marker.

09:35 UTC update: full state160 was copied and all10 files hash-verified; trainer state160/iteration10/epoch2 and DeepSpeed scheduler159/skipped0 were inspected. A project-quota incident required relocating the backup to home storage and resuming there in a fresh native output directory, with unchanged scientific settings. No cached episodes/evaluations were copied; repeated collection10 question identities/order match all64 original questions. This verifies checkpoint contents and sampling identity, not bitwise RNG continuity. The interrupted attempt's176-update result remains separate from resumed results. [Preservation manifest](evidence/checkpoint-160-preservation.json).

GSM8K GRPO is available in `configs/polIter_rho1bSft2_grpo_GSM8K.jsonnet`:1–10: it switches to group advantages and disables the mask. The paper Figure 3 baseline scores are inherited from VinePPO, not shown to arise from rerunning this current GRPO config. A locally reproduced GRPO comparison must share the recovered SFT/data/evaluation protocol and record its own training provenance.

## Dataset provenance

Author source URL: `https://rl-llm.oss-cn-hongkong.aliyuncs.com/wandb_export_root.zip`; retrieval 2026-10-08. Outer SHA256 `38026cae756c03accb921d79a178e4e2dd34b5efab797768be078bffd504ac3c`; inner member `wandb_export_root/datasets/data-gsm8k-w_valid_split.zip`, SHA256 `e6da4e7c044fc273027ad11d476ee70b410427a56b4d9ce0fa0fc144a4601d2b`. Only its ten GSM8K members were extracted; no author cleanup script was executed.

Verified with the campaign's existing environment: **7100 train, 373 validation, 1319 test**, two columns question/answer, zero exact question overlap across every split pair. Saved dataset metadata traces source HF revision `e53f048856ff4f594e959d75785d2c2d37b678ee`; the existing validation split is a subset of original training (`_split:train`). Metadata still reports original training size 7473; use observed Arrow row counts. Per-file hashes and boundary question hashes are in [dataset-provenance.json](evidence/dataset-provenance.json). Saved raw fingerprints differ from loaded fingerprints under the newer library; bytes/hashes and rows establish identity.

Dataset saved metadata has an empty license field. The [original GSM8K repository license](https://github.com/openai/grade-school-math/blob/master/LICENSE) is MIT; no additional author archive license was found. Use an additive staging `data/gsm8k` symlink pointing to the extracted asset to preserve native relative config paths, without adding data to the clone or Git.
