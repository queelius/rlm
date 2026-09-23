# Conditional positive-only terminal-credit diagnostic

Proposed CPU preparation, not GPU accepted. Main decides acceptance only after
the complete signed high-LR readout. No partial evaluation was inspected here.

Restore the same original checkpoint1, FP32 LoRA, Adam step1 moments and RNG.
Use the same FP16 base and all909 saved on-policy sample2 calls. Full32-episode
native replay and every saved generation/full-forward log-probability check remain
mandatory. Only the optimizer-credit mask changes from advantage !=0 to advantage >0.
The original positive RLOO advantage weights and denominator32 are retained exactly.
No resampling, rescoring, advantage renormalization, action-validity filter or
multiple gradient steps are introduced. LR1e-4, clipping1 and max/mean replay
tolerances .25/.025 remain the same as the completed signed high-LR update.

The real saved-batch fixture verifies12 positive-advantage episodes,399 credited
calls and12,927 emitted tokens, versus819 calls/26,933 tokens under signed credit.
All32 episodes, including negative and zero-credit trajectories and invalid actions,
remain in the native audit. CREDIT-MASK.json records every original advantage and
whether it contributes to the update. Historical909-call collection cost is reused,
not counted as new generation. Actual update runtime/gradient norm/adapter movement
and credited-token differences must be reported.

This objective is biased: it is not unbiased RLOO or an SFT-matched control.
It changes gradient direction, norm and token dose; retaining clip1 does not match
the effective per-token update. A benefit would motivate investigating harmful
negative trajectory credit, but would not prove that any particular useful shared
action was suppressed. Success/failure trajectories may differ in many ways.

Exactly one new Adam update may produce cumulative checkpoint2. Require normal
completion, finite update and actual committed Adam2; failed/capped initial
checkpoint1 is not eligible. Training cap90minutes; fixed BF16 fresh16×2 readout
cap60minutes with unchanged per-episode sampling/prompts/budgets. Primary paired
native analysis is positive-only minus completed signed1e-4 on the same32 slots,
preserving unknowns and task-parent clustering. This exposed-panel adaptive
mechanism screen is not an independent replication or new evidence from32 independent worlds.
