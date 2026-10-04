# CURL single-GPU reproduction implementation plan

> For agentic workers: use the execution and test-driven-development skills.
> The user explicitly requests continuous autonomous research, no approval pauses,
> and focused scientific checks rather than broad production hardening.

**Goal:** Run a faithful-as-practical CURL reference and matched no-contrastive
comparison, with interpretable evaluation, resumable training, and a learning guide.

**Architecture:** Keep the official agent and replay implementation unchanged.
Use a small modern dm_control adapter and a separate driver for evaluation,
metadata and full-state checkpointing. Keep environments and raw outputs external.

**Tech stack:** pinned upstream CURL, isolated Python/PyTorch/dm_control environment,
NumPy, JSONL records, LaTeX guide. **Spec:** README.md and manifest.json here.

## Global constraints

One assigned A100, no modifications to existing RLM environments or live sources.
100k environment steps = 12500 decisions at repeat8; seed123 pilot excluded from
fresh reference means. Subsequent seeds456/789 fixed. Two-hour run cap, 15-minute
checkpoint target, allocation end2026-10-07T15:36:06Z from SLURM_JOB_END_TIME.
Use a separate eval simulator and preserve training RNG. No claims from missing
results. User's existing work is untouched. Publish milestones on main after review.

## Review focus

- Action repetition must sum rewards and count actual simulator steps, including endings.
- Evaluation reseeding must reach the task RNG, not just Gym spaces.
- Resume must retain optimizer/RNG/replay state and clearly state trajectory continuity.
- No-CURL must retain the same augmentation and SAC schedule.
- Published results, adaptations, pilot diagnostics and actual scores remain separate.

## Tasks and progress

- [x] Pin official source, inspect core code, establish isolated worktree and environment.
- [x] Env adapter: write failing real-simulator tests, implement make_env(domain,task,
  seed,action_repeat), old-Gym reset/step, seed/close, get_rng_state/set_rng_state.
  Run tests for reward sums, frame order,1000-step episode end and deterministic resets.
- [x] Training/checkpoint driver (agent): tests first for complete state round-trip and
  fixed-batch next update; retain upstream update behavior and explicit terminal eval.
- [x] Run1200decision pilot, inspect actual rewards/losses/weights promptly, retain failure
  records; only then start fresh reference/control seed123 and follow-on seeds.
- [ ] Analysis: extract episode means and independent training-seed comparisons; retain
  failed/incomplete endpoints; update guide with actual evidence and limitations.
- [ ] Scoped independent review and publication; no broad unrelated RLM test campaign.

## Decisions

2026-10-04 17:00 UTC: execution access restored; old 'blocked' documents are historical
and will be updated at the first run milestone. Main quota99%remaining at16:58UTC.
Ruling: use an isolated worktree based on current origin/main; leave old root/main
and other research worktrees untouched. Copy the current guide/protocol into this worktree.
Ruling: modern dm_control adapter is a documented compatibility adaptation, not an
exact recreation of the unpinned2020 simulator stack. Preserve reward/action/crop behavior.
