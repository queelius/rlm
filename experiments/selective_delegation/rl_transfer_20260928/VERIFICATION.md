# Verification — September 28, 2026

Prepared, not launched. No GPU model, optimizer, new scientific collection, Git operation, environment change, or accepted-source edit occurred in this task.

`R` is `/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.

- Dispatch receipt: `R/textcraft-rl-transfer-20260928-001/PREPARED-JOBS.json`, SHA256 `a484f4695be7b3c03f130db2e55e3e3703852fcc81ace8793bb022352ecaf923`.
- Four scientific descriptors, one CPU analysis descriptor; 46 unique input/source pins independently rehashed and matched. The CPU descriptor has no scientific `output` field.
- All four exact scientific commands ran with `--prepare-only`; all reproduced their prepared PLANs without GPU load.
- Existing warm descriptors match the original fresh-campaign receipt as complete objects and are absent from the new jobs list. Original receipt SHA256 `4e10be9cd54480c7deee72118b31f5d49ee56042a9889213dc08fd4a6de7f3d6`.
- Eight focused tests passed using the accepted GPU interpreter with CUDA disabled and bytecode writes disabled. Six initial tests were observed failing before implementation; explicit scripted-fixture rejection was also observed failing before its guard. Arithmetic check independently expects a two-win/16-attempt contrast of0.125 and no estimate after one unknown.
- Direct `/project/alex_phd/envs/rlm/bin/ruff check` and `ruff format --check` passed. No `uv` invocation or environment mutation.

Actual endpoint model hashes: raw `856fb4755c3a2b3297a38d6bed7d6f8e638535eba6723dc01cc0ee4a916db62e`; binder `2621e7d47aa7af51736640af22fd6846081a82f60fffdcfa8178151a6e3854ab`. Both usable SUMMARY records report exactly one new/committed optimizer step; STATE step and training-plan linkage agree with COMMIT. All readout-relevant committed adapter/config/state bytes matched. Unused optimizer/RNG tensors and multi-gigabyte base ancestry were not repeatedly traversed.

External fixture: `R/textcraft-rl-transfer-cpu-fixture-20260928-001/FIXTURE.json`, SHA256 `e40fb4efab3f5144dcf856086667decae77d6418280178c032def6aa09b39eab`. Raw and binder each made15 scripted CPU calls and saved449 original output tokens; native replay success1 for each. The first B prompt matched its frozen qualification hash. The binder fixture retained the deliberately wrong emitted ingredient counts while native execution succeeded. Every metrics sidecar matched its saved call hash/token lengths. This exercises actual runtime plumbing with scripted generation, not a scientific model, real probability distribution or learned performance.

`PREPARED-SMOKE-COMPARISON.json` reports all six absent scientific cells unavailable, with16 unknown paired outcomes and null effects in every learning comparison. It is explicitly a prelaunch smoke artifact, not the final `COMPARISON.json`.

The planning/TDD skills drove test-first endpoint/lineage/seam checks; verification remains proportional to this additive exploratory adapter. User CPU-only/no-Git and narrow-test constraints override skill defaults for broad suites, commits and additional worktree creation. No project-wide test claim is made. Parent owns independent integration review and GPU dispatch.
