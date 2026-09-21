# Bounded paired-outcome pilot: implementation plan (not GPU accepted)

**Status update, 18:08 UTC:** Main accepted sources037/038 after reading the
training and launch paths and rerunning six sealed CPU tests. The conditional
launcher is supervisor82971. The plan below records the pre-implementation design;
its earlier pending-acceptance statements are historical. Fresh replication is a
diagnostic, not a requirement to observe a positive gain before the finite RL run.

The conservatism amendment supersedes the earlier SFT-gain gate. Question: can
official paired-outcome optimization repair excessive refusal better than additional
imitation? Joint32 is the fixed initialization, not a selected checkpoint. A separate
held panel must be frozen before any accepted update; canonical replication034 is
not silently reused as untouched evaluation.

1. Add one thin shared runner with explicit `rl` and `sft_control` modes. Reuse
   audited PEFT loading, target tokenization, native generation conventions, RLOO
   replay and atomic adapter/Adam/RNG checkpoint writer. No helper/root/final roles:
   only the whole answer+sufficiency adapter trains; base weights remain frozen.
2. Read the immutable128-parent/256-variant proposal and fixed eight16-parent blocks.
   RL generates four paired candidates (128 calls/block), T.8/top-p1/top-k0/cap128,
   preserves emitted IDs/EOS and request/model/adapter identity. Each paired candidate
   gets official paired EM; loss uses both response logp sums, RLOO and denominator64.
   Weight updates occur only after a complete block, with one replay pass.
3. Keep sampled cursor, optimizer step and zero streak separate. No Adam call on
   uniform groups/batches; zero blocks remain in acquisition costs and persisted
   boundaries. Stop at four consecutive zero blocks, eight blocks, error, or45min.
   Errors are unknown, not zero. Save every completed boundary under its sampled
   index with the actual optimizer step inside, including adapter/Adam/RNG. Explicit
   resume rejects changed PLAN or unresolved partial blocks; no automatic retry.
4. Extra-SFT control starts the same joint32 with fresh Adam2e-5. It reads the exact
   committed RL block/update inventory, skips the same zero blocks, teacher-forces
   each participating positive/negative gold target four times, and takes one
   accumulated update per RL real update. Target+EOS only, microbatch1, FP32 rank8
   LoRA, clip1, no new optimizer settings. Thirty-minute cap. This matches parent,
   response and step dose, not token lengths, information or generation FLOPs.
5. Focused CPU fixtures: actual tiny Qwen3/PEFT sampling and gradient routing;
   pair reward versus individual-label shortcuts; emitted-EOS logp alignment;
   sampled/optimizer cursor separation, four-zero stop and skipped Adam invariance;
   control schedule/target count. Preserve per-call/per-pair/per-block receipts,
   gradients/deltas, likelihood movement, protocol and missing categories.
6. Separate proposed seals037/038 only. Main reviews loss/owner and chooses acceptance
   after both completed SFT readouts and pending replication evidence. Eventual queue
   follows036. Terminal boundary is the declared endpoint even when fewer than eight
   real steps occurred; no held-score checkpoint selection or dose extension.

Completed SFT three-arm native analysis takes priority over implementation whenever
main announces the positive-only readout is terminal. No GPU launch is part of prep.
