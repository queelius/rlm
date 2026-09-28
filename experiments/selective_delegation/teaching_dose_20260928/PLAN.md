# Matched teacher continuation implementation plan

> Inline CPU preparation; parent owns GPU launches. No commits or live-source edits.

Goal: test whether two more optimization epochs rescue the quantity-corrected
known-recipe teacher, against matched discovery continuation. The accepted parent
brief fixes original seed2026092208, BF16 base/FP32 LoRA, restored AdamW/RNG,
46 new updates and prospective cumulative checkpoints46/69. Existing source048
supplies tokenization, target loss, epoch shuffle and checkpoint serialization.

## Tasks

- [x] Test and implement `dose_common.py` plus `train.py`: exact epoch1/2 shuffles,
  cumulative update/row/token state, prior checkpoint authentication and AdamW/RNG
  restoration. Fixed LR1e-4, batches16/last14, native source048 target-token mean.
- [x] CPU-prepare both immutable plans and summarize all23 original training-loss
  receipts. No large base-model ancestry walk; verify small manifests and the
  actual adapters/optimizer boundaries once per invocation.
- [x] Test and implement `readout.py`: qualify cumulative checkpoints46/69 using
  new truthful dose schema, then import the existing native breadth collector.
  Reuse panel00/world42 or50, original sampling seeds/caps, raw or binder execution.
- [x] Run focused CPU tests and real prepare-only commands; document ready GPU
  commands, estimates, fixed endpoint selection and underfitting limitations.

Review focus: replaying epoch0; resetting Adam state; wrong total update or
microbatch counts; falsely calling a partial checkpoint complete; readout adapter
mislabeling. Tests exercise the schedule, state boundaries, optimizer roundtrip and
fixed-endpoint admission. A 30-minute cumulative owner cap is retained per teacher.

Ruling: the user's GPU-first autonomous research instructions and parent-provided
bounded design replace additional approval gates, broad suite runs, branch
creation and commits. The already-isolated supplied worktree is preserved.
