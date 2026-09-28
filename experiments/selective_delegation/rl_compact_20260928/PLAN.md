# Compact first-cycle RL implementation plan

> **For agentic workers:** Use superpowers:executing-plans for this CPU-only, native execution.

**Goal:** Prepare one compact-interface native-success RLOO cycle and its own-interface controls.

**Architecture:** Privately reuse the frozen compact collector/auditor and existing RL
collection/optimization loops. Add truthful compact manifests, future-checkpoint admission,
four parent-dispatched GPU stages, and one CPU comparison. No existing source changes.

**Tech stack:** Existing Python/torch/transformers/PEFT environment; native TextCraft bridge.

**Spec:** Parent's September28 task: groupA8×4, groupB8×2, compact SFTcp23, one fresh AdamW2e-5
native-success update, full sampled-token T0.5 signed RLOO/32, no sweep or GPU launch here.

## Global constraints

- Same first-fresh-v002 task identities/order/seeds and original call/token/context caps.
- `execution_mode=compact_observed`; strict observed-only bridge and compact action histories.
- Real compact checkpoint bindings only after authenticated SFT completion; no invented hashes.
- Collection150min, training120min, each readout90min; four GPU stages maximum.
- Missing/incomplete/flat training skips; warm readout remains independently admissible.
- Report own-interface native-success gain and difference from full+binder first-step gain.
- Matched SFT steps are not equal token exposure; unseen-recipe semantics also differ.
- No edits to accepted sources, no GPU launch, commit, push, broad suite or environment changes.

## Review focus

Absent warm checkpoint; wrong dataset/mode lineage; compact requested versus expanded targets;
native replay of charged failures; accidental warm-readout suppression after failed training.

## Files and steps

- [x] `compact_common.py`: private dependency injection, dataset/schedule/source identities,
  actual warm/update bindings, collection/trainer PLAN builders, native-audit reuse.
- [x] `compact_stage.py`: collect/train/warm/updated resolution with truthful skips and unchanged locks.
- [x] `prepare_compact_rl.py`: CPU qualification and four GPU descriptors plus CPU analysis; pending
  checkpoint expressed only in a preparation contract, not a scientific PLAN.
- [x] `compact_compare.py`: paired compact gain versus corresponding first-fresh full+binder gain;
  success/calls/errors/token entropy/diversity and explicit missing cells.
- [x] Write failing focused tests before implementation, including a real saved compact
  request/response/token/native-audit path through the new collection dependency injection.
- [x] Implement only the thin seams, pass focused CPU tests/Ruff, prepare immutable descriptors
  and README, verify every pin and original-source immutability, hand off for parent dispatch.

No substantial new trainer or harness architecture is planned. Parent review approves this
bounded implementation; no external research job is accepted by this document.
