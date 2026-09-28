# CPU verification — September 28, 2026

Eight focused tests pass in the accepted GPU interpreter with CPU-scripted tensors only.
Direct `/project/alex_phd/envs/rlm/bin/ruff check` and `ruff format --check` pass for this folder.
The real saved-response fixture is external at
`textcraft-compact-rl-20260928-001/runtime-fixture-001/VERIFICATION.json`.

The fixture runs compact requested actions through the existing collector, generation-score
capture, causal token reconstruction, actual signed loss, and independent native episode replay.
It includes one invalid-schema response, one strict unobserved-recipe rejection, one native stock
failure and eventual native success. Original compact and invalid emitted tokens—including EOS—
remain likelihood targets. Scripted uniform logits are a seam test, not a model capability result.

Only the new `rl_compact_20260928/` source folder was edited for this task. Accepted full/native,
fresh-RL, compact-SFT and compact-bridge source pins are checked separately before handoff.
No pretrained inference, GPU job, commit or push was performed.

Final preparation receipt SHA256:
`4603f4e1ab13318bf30933555377b0ef5ecdb6a998ceae93d531ab0b9cbed0d0`.
All 43 descriptor pins, all 19 prior fresh-binder collection source pins and all six compact-SFT
contract source pins verified unchanged. The external fixture has 14 native calls, 312 emitted
tokens, native success 1, and zero generation/replay log-probability gap. Both signed derivative
checks pass. Initial compact prompt plus 256-token response caps are at most 848/886 tokens for
groups A/B; these are not full-history guarantees.

## Local environment incident

One verification command mistakenly used `uv run ruff check ...`. `uv` warned that the inherited
`VIRTUAL_ENV=/project/alex_phd/envs/rlm` differed from the worktree project environment and installed
31 packages into this worktree's `.venv` before invoking Ruff. The resolved path is
`/project/alex_phd/repos/rlm/.worktrees/selective-delegation-20260921/.venv`, a real directory, not a
symlink. This was an unintended preparation-environment mutation. It did not target the accepted
GPU interpreter at
`/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python`.
The parent was notified. No rollback, cleanup or deletion was attempted. All subsequent Ruff
commands use the existing `/project/alex_phd/envs/rlm/bin/ruff` directly.
