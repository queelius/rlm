# Fresh-A reward diagnosis — 29 September 2026

The first raw **training collection**, not a learning-gain evaluation, contains 5 native successes in 32 attempts. Three of eight four-sample groups provide nonzero terminal-RLOO advantages. Five all-failure groups contribute no direct policy gradient despite using 1,295/1,831 calls (70.7%) and 41,251/57,739 sampled tokens (71.4%). Shared-parameter updates from other roots can still affect them.

| A task suffix / root | Successes / 4 | Calls | Advantage pattern, repeats 0–3 |
|---|---:|---:|---|
| 132 / t9_i3 | 2 | 136 | +2/3, −2/3, +2/3, −2/3 |
| 1423 / o8_i3_18 | 0 | 213 | 0, 0, 0, 0 |
| 2281 / m0_i4 | 1 | 173 | −1/3, −1/3, −1/3, +1 |
| 1753 / a3_i5 | 2 | 227 | +2/3, −2/3, −2/3, +2/3 |
| 1921 / o2_i4_22 | 0 | 233 | 0, 0, 0, 0 |
| 1051 / o6_i5_16 | 0 | 272 | 0, 0, 0, 0 |
| 2338 / o8_i5 | 0 | 278 | 0, 0, 0, 0 |
| 2026 / m5_i5 | 0 | 299 | 0, 0, 0, 0 |

The actual [credit function](../textcraft_trajectory_loss.py) uses `A_i = success_i − mean(other three successes)`. The [active trainer](../rl_resume_20260928/train.py) accumulates `−A_i × sum(original emitted-token log probabilities at T=0.5) / 32`, then takes one optimizer step. Zero-advantage trajectories remain in the denominator. There is no reward standardization, per-trajectory length normalization, valid-action filter or local error attribution. All sampled tokens, including erroneous responses, receive their trajectory's sign. This is a sparse *available update signal*, not evidence of a broken optimizer or of RL being ineffective.

The native audit reports 837 action errors plus 17 invalid schemas. Public error strings include 307 extra-ingredient errors, 186 missing-required-ingredient errors and 254 insufficient-inventory errors; 90 others remain separate. Of 27 unsuccessful episodes, 23 explicitly finish and four hit context limits. Successful episodes also contain 23 calls after the root quantity was first met. Thus both execution arguments and stock/termination behavior remain plausible bottlenecks; terminal reward alone cannot assign responsibility. The existing binder, error-cost and payload-mask branches already test adjacent mechanisms and should not be duplicated.

## Public recovery is feasible at the checked early prefixes

A CPU-only fixture selected repeat 0 immediately after its first native error for **every** A root, without inspecting terminal success for selection. Starting from each actual next saved request, it reconstructed the same prompt bytes, restored original initial/current inventory and charged prefix budgets, then used the existing [public teacher](../prepare_textcraft_public.py). All eight scripted continuations finished natively within the unchanged caps: at most 68 total calls and 6,220 prompt-plus-response-cap tokens. This includes all five roots with flat-zero rewards. Only goals, inventories and already observed recipe replies entered teacher decisions; no gold trajectory, future reply, task ID, native score or hidden recipe graph did.

This establishes a usable *early* recovery-teaching opportunity, not late-state recoverability, model competence or unique optimal actions. See [RESULTS.json](RESULTS.json) for all eight actual request receipts and the precise data flow.

## Recommendation and status

Choose one contingent comparison: **A-only learner-state recovery SFT versus matched ordinary A public-teacher SFT**, described in [EXPERIMENT.md](EXPERIMENT.md). The CPU feasibility check passed; the GPU package is **not prepared, accepted or queued**. Wait for the unchanged first-A updates, binder collection and already queued extra-SFT controls before ranking it. No B outcomes were inspected or used.

Verification scope: read the saved native audit, 32 small episode/root-node pairs and eight selected requests; verified all 64 episode/node receipts and eight request receipts. Read the actual objective and preparation helpers. No full 1,831-call replay, model/ancestry rehash, GPU/model load, environment mutation, queue edit or Git operation. All new retained files are in this directory.
