# A trailing reminder fixes most JSON errors, not TextCraft success

Source049 completed all 16 fixed flat episodes on eight exposed VAL tasks × two seeds. The independent sealed analyzer passed native prompt/token/seed/model checks and trusted inventory/action/checker replay. Its terminal is clean, with no transport errors, missing episodes, unresolved starts or unlinked calls. No outcomes were inspected before authenticated terminal/dead-owner release.

**Success: 0/16.** Against the 13 observed flat episodes from source044, the matched result is **0 wins, 1 loss and 12 ties**. The other three original-flat slots remain unknown; they are not imputed failures, and there is no full-panel effect estimate or CI. Original flat recorded one success and 12 failures, with three missing of 16 planned. The recursive-interface arm from source044 is not part of this comparison.

The intervention appended one fixed reminder about exactly one JSON action, current rather than historical inventory, recipe availability, net target quantities and explicit finish. All other prompt history, model weights, seeds, strict parsing, native actions and per-episode limits were unchanged. This is a packaged post-hoc instruction control on exposed tasks, not an isolated wording mechanism or independent confirmation.

## Protocol versus actual execution

| Measure on the same 13 observed baseline slots | Original flat | Trailing reminder |
|---|---:|---:|
| Native successes | 1 | 0 |
| Calls | 650 | 1,085 |
| Invalid JSON/schema replies | 118 | 4 |
| Returned native action errors | 212 | 692 |
| Prompt tokens | 2,785,367 | 4,557,351 |
| Output tokens | 48,380 | 29,409 |
| Native seconds | 2,120.16 | 1,472.48 |

Schema-error frequency drops from 18.15% to 0.37% of calls on these matched slots. Yet correctly parsed actions increasingly repeat unsuccessful operations. More calls and input tokens coexist with fewer output tokens and lower native time; this is a measured generation/workload tradeoff, not evidence of better reasoning efficiency or attention scaling. Trajectories differ after the prompt intervention.

Across all 16 reminder episodes, 1,000 craft actions include 890 native rejections: 842 insufficient-inventory errors, 16 missing-ingredient errors, 30 extra-ingredient errors and two batch-divisibility errors. There are 81 recipe queries, 243 inventory views, four schema errors and **zero finish actions**. Thirteen episodes end at the context cap and three at the 96-call cap. Valid JSON is therefore not a useful proxy for task completion here.

## Goal readiness did not trigger finish

Three episodes reach sufficient net target inventory: both repeats of t00 and repeat1 of t03. They then consume 259 calls: 217 `view_inventory`, 31 `get_info` and 11 `craft`, with no finish. On copies of their final public inventories, the trusted native checker returns success after an explicit hypothetical finish. This confirms the public net-goal criterion is meaningful; it does **not** change their observed score or claim the model would choose finish.

- **t00, both repeats:** the requested three `c3_i2_23` units exist after call011. Instead of finishing, each trajectory runs to 96 calls, mostly viewing inventory, and eventually overproduces 18 units. This improves goal attainment in repeat0 relative to original flat but still produces no benchmark success. The two repeats are correlated trajectories, not independent task evidence.
- **t03, repeat0—the sole paired loss:** original flat eventually succeeded. With the reminder, call003 successfully produces two `m0_i1`, then the model repeatedly tries the same craft after its ore is exhausted, rather than using that intermediate in the root recipe. It reaches the context cap without the root target.
- **t03, repeat1:** the reminder run successfully crafts its root at call004, then repeatedly views inventory and ultimately requests information about exhausted ore. Original flat also reached sufficient target stock and failed to finish. This is a retained termination failure despite explicit trailing instructions, not an insufficient-information explanation.

These examples do not prove that repeated history causes the loops, or that a hidden stopping mechanism is needed. The experiment did not change retention or execute host-selected actions. Native feedback and current inventory were present, and the host executed/rejected actions exactly as specified.

## Decision

The cheap reminder qualifies the baseline: the original result was partly a formatting failure, but suppressing that failure does not rescue basic task progress or termination. Do not extend this wording experiment, force delegation, or interpret any result as recursion gain.

A future cheap harness baseline could explicitly test the public net-goal condition from `target_items`, `inventory_at_task_start` and `current_inventory`, then terminate through the documented finish action. That is a host termination policy using already-public state, not secret native reward or gold recipes—and it is distinct from learning to choose finish. It would need separately declared action/cost accounting. The copied-state checks above are only counterfactual audits, not a measured new GPU policy result. Merely caching state may not fix this bottleneck: current inventory is already present, yet no finish is emitted even with the explicit reminder. Neither this termination baseline nor a cache change is implemented here, and source052 remains unchanged.

Proceed with the already-fixed source048 action-SFT checkpoint23 and source052 original/reminder readout without changing their data, dose, endpoint or prompt definitions. That training was accepted independently of source049 outcomes. It teaches successful query/craft/finish sequences and provides the next meaningful competence test; its absence of error-recovery demonstrations remains a limitation. Any extra SFT benefit over the reminder should be assessed on task success and native action/finish behavior, not JSON alone. The same exposed eight tasks and original missing slots limit generalization claims.

## Cost and provenance

All 16 new episodes cost 1,328 native calls, 5,605,142 input tokens, 37,447 output tokens and 1,864.78 native seconds. Owner wall time was 1,901.03 seconds, below the 45-minute cap. All model calls are new; historical baseline costs are reported separately rather than charged as new work.

Artifacts under `/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/`:

- `analysis-textcraft-instruction-control-001.json` SHA256 `615740e685636c2f870cb4f3b2fcd601a4935f91eba9cd6d57f0efbd81e41cd6`; companion `.md` and per-episode native audits retain coverage and costs.
- `textcraft-instruction-control-001/TERMINAL-193bd390e9ca.json` SHA256 `4c5021080aadf891ab2ea3b917f06f8bce246ae3838a57d041ad6ddecf8ef174`; ended 2026-09-22 06:43:08 UTC, failure null, stopped false.
- PLAN SHA256 `e43ecd64f6e78fb84197186151b1542db1362e254ecaaa2e67f81629151a463a`.
- Source049 collector SHA256 `3896a1ef5ca6f13ae499ea212101a8f660bb18ab76ef1f0bdc28937ebb9b6b6f`.
- Sealed profile analyzer SHA256 `1c4062a2e06a307a38173d28dab405e497aba8576fc5d170dc3f9f29acaca88b`.
- Baseline `analysis-textcraft-pilot-001.json` SHA256 `ea875de29d9e2e1af999027ff1eaf20080a1237fb94981d71a336ec35542aa79`.

The bounded CPU watcher used source054 `resource_released` before invoking `analysis-source-textcraft-profiles-001/analyze_textcraft_profiles.py`, with `--expected-collector-sha256` set to the source049 hash above and `CUDA_VISIBLE_DEVICES=''`. Analysis exited successfully. No GPU work or accepted-source modification was performed for this readout.
