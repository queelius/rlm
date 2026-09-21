# Bounded public-action SFT plan

**Status, 18:20 UTC:** Main accepted the completed source041b implementation
after code review and six sealed CPU tests. Supervisor51352 waits for the resolved
sufficiency readout. The exact recipe is one epoch, 33 updates (last batch 12),
learning rate 1e-4, seed 2026092200, and a 30-minute cap. Output is
`alfworld-action-sft-002`; checkpoints include optimizer and RNG state.
The earlier source041 proposal was never run on GPU and is retained separately.

Question: can a small supervised prior map the existing public indexed-action
interface to useful next actions, before attributing closed-loop failure to
hierarchy or RL?

Frozen input is `alfworld-train-sft-inputs-001` (524 targets from the 30
selected TRAIN games that won within 50 native expert actions). This is a
success-filtered TRAIN demonstration set, so it tests interface competency, not
an unbiased ALFWorld policy estimate. Inputs use the exact source026 flat prompt;
targets are strict indexed-action JSON plus EOS. No hidden expert state is in an
input. The six selected-but-capped games remain in raw provenance and are not
quietly replaced.

Run exactly one epoch from the released Qwen3-4B base with a fresh
rank-8, alpha-16, dropout-0 LoRA; base weights frozen; only adapter parameters
FP32/trainable. Fix order seed, batch/token accounting, optimizer steps, and
checkpoint/resume cursor before launch. Retain every target in the loss mask
through EOS, and preserve raw/processed manifests and tokenizer/source hashes.
This is a queued bounded GPU job, not a completed training or quality claim.

The later readout is deliberately exploratory: base and trained flat, and base
and trained manager-worker, on the already selected source035 games. The manager
stays at the base model; only flat actors and workers use the trained adapter.
This role separation prevents action-format training from silently changing the
manager's goal-writing behavior. The comparison must
retain the same public bridge, action/token caps, temperature and game/seed
schedule; report each slot, invalid action outputs, tokens and native wins. The
panel is exposed by this planning process, so it is not fresh generalization.
Primary decision signal is whether trained flat improves valid indexed action
execution and native wins over base flat. A manager comparison can show whether a
flat-action prior transfers to the hierarchical interface, but cannot isolate
planning from the shared action-format SFT. A null or format-only gain retires
this SFT as evidence for hierarchy/RL and argues against scaling it.
