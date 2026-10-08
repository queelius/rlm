# SPO reproduction implementation plan

> **For agentic workers:** Use the existing experiment workflow and focused verification. The user approved the proposed reproduction and unattended execution on 8 October 2026. Do not wait for further design approval.

**Goal:** Reproduce the published SPO-chain RhoMath 1.1B GSM8K RL experiment using the authors' implementation, then compare with its GRPO control.

**Architecture:** Keep the official source and reusable assets in the external research cache. Put immutable attempt inputs, native outputs, checkpoint pointers, and the serial GPU owner in `/project/alex_phd/runs/spo-reproduction-20261008`. Reuse the existing bounded owner and exact-thread review queue; publish readable milestones to GitHub main.

**Tech stack:** Authors' Python, PyTorch, DeepSpeed, customized vLLM, Jsonnet, GSM8K evaluator; existing local Python observer.

**Spec:** The user's approved five-part proposal in this chat: exact target, reference evaluation, unchanged recipe, matched comparison, later separate lessons.

## Constraints

- One A100 40 GB. Inspect existing owners before GPU admission. Old Dr. GRPO remains stopped.
- Initial source: `AIFrameResearch/SPO` commit `6c5f94723c20e4de68578545b8bb98186e956797`.
- Main target: SPO-chain interval 5, Rho SFT GSM8K starting checkpoint. Audit recovered the Figure3 checkpoint690/11040updates within the1000-iteration schedule. Fix690 prospectively for comparison; preserve1000 schedule. Historical checkpoint-selection rationale/source revision remain unresolved.
- Do not change data, group size, loss, sampling temperature, or scientific endpoint to chase a test score.
- Reference and trained-checkpoint evaluation must use the same native metric. Sixteen evaluation samples do not mean best-of-16 accuracy or sixteen training replicates.
- Failed or capped runs remain incomplete, not reproduced endpoints. Keep model and optimizer states when supported and audit resume rather than assume it.
- Retain at least 10% account allowance; dispatch only above 12%. No secret, model weight, environment, or large dataset in Git.
- Allocation end previously recorded as epoch 1791621574; use a conservative cutoff at 1791621274 unless current evidence supersedes it.

## Tasks

- [x] Audit full configuration inheritance, dataset identity, metric, checkpoint rule, and available GRPO control. Record historical baseline/selection/source gaps without inventing the authors' choices.
- [x] Prepare isolated pinned environment and public assets. Inspect downloaded code before execution. Exercise native imports and a real scientific response before accepting a long job.
- [x] Evaluate released starting and trained weights. Run bounded native jobs, inspect actual answers and grading, and preserve complete failures.
- [ ] Launch the unchanged training recipe under one exclusive, detached owner with allocation cap. Include native periodic evaluation and checkpoints; prepare independent control and endpoint evaluation as follow-on work.
- [x] Adapt the existing review mechanism with tests for terminal receipts, deadlines, pending acknowledgments, and quota. Deliver a one-time probe to verify future reviews are queued to this exact thread.
- [ ] Publish a plain-language overview, protocol, live queue, and verified evidence. Record the true launch/idle intervals and remaining limits.

## Review focus

1. A runnable example may differ from the figure's historical experiment: compare configuration and paper.
2. Test evaluation may run during training: preserve upstream behavior, but do not silently select the highest test checkpoint.
3. A training process exit may precede evaluation: GPU ownership covers the entire sequence.
4. A checkpoint may contain weights but not optimizer state: do not claim exact restartability without inspection.
5. An inference service may be alive yet return only errors: check actual returned answers promptly.

## Progress

07:42 UTC: GPU is idle with no compute process; no ready SPO job exists yet. Source acquired; environment and protocol preparation dispatched in parallel. Main account reports 31% remaining. Current worktree is already isolated; unrelated untracked files are preserved. No changes to live or prior campaign source.

08:02 UTC: Full starting-model reference39.77445%, all21104answers and7native metrics source-checked/recomputed. Post-evaluation launcher local_rank error preserved; supported --no_local_rank option used in running author reference. Observer deployed with delivery probe admitted to exact thread;31focused CPU tests and independent automation review passed. Allocation deadline validated from Slurm environment. Mainquota30%at07:55.

08:08 UTC: Author reference56.68594% versus56.72384%historicalreference, full21104responsevalidation and exit0cleanup. Fresh training launched08:06:24under45hcap+allocationdeadline; first-collection monitoring inprogress. Prospective690endpoint is preserved within1000schedulerhorizon; no test-score selection. Two follow-ons remain fixedendpoint evaluation and historicalbaseline/control provenance audit.
