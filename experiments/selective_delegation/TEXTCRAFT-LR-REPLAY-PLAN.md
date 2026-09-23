# Same-batch update-size diagnostic

Proposed, not GPU accepted. Restore original RL002 checkpoint1, its FP32 LoRA,
Adam step1 moments and RNG; use the same FP16 base arithmetic as completed
`textcraft-terminal-fp16-continuation-002`. Reuse **all** its saved sample2:
32 TRAIN episodes, 909 calls, 819 nonzero-advantage calls and 26,933 credited tokens.
An independent CPU native replay reconstructed all 32 episodes and 909 requests.
The current corrected source065-v2 auditor and original task insertion order are used.

Change only the next Adam learning rate from 2e-5 to 1e-4. Recompute action log
probabilities, require the unchanged generation/full-forward and train/eval guards
(max .25, mean .025), apply terminal task-group RLOO with token-sum/32 at T=.5,
clip gradient at1, and take exactly one step. All errors and flat groups remain.
No new rollout, reward-based subset or repeated off-policy gradient steps.
The saved data are on-policy for the restored FP16 checkpoint1 actor, not for
the updated checkpoint2. The completed reference used gradient norm39.261 and
adapter L2 change .058994; the new run records its own values and full logps.

Collection cost is historical/reused, never charged as new inference. The new
90-minute training cap covers native audit, numeric backward qualification,
full replay, one optimizer update, atomic checkpoints and post-update logps.
Failure/cap preserves receipts and checkpoints but does not promote a partial
endpoint. Only a clean committed Adam2 endpoint receives the fixed BF16 fresh16
task ×2-seed readout (unchanged per-episode limits; original60-minute collection cap).

Primary comparison is new1e-4 versus completed2e-5 on the same 32 held-out slots.
Use native replay and parent-paired uncertainty, preserving missing outcomes as
unknown. This adaptively chosen exposed-panel update-size screen is not a fresh
replication, an off-policy multi-step experiment, or an RL-versus-SFT causal result.
Do not continue training from the larger step without inspecting its completed
readout. This test addresses whether the single small update is dose-limited;
it does not establish broad learning efficacy.
