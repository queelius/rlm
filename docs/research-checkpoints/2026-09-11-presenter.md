---
title: Research and advisor-package resume checkpoint
recorded_utc: 2026-09-11T06:33:00Z
status: operational_checkpoint_not_new_scientific_claim
audience_deck_evidence_cutoff_utc: 2026-09-11T03:45:00Z
---

# Preserved in Git

The [advisor package](../../slides/2026-09-11-advisor-meeting/README.md) contains
the 14-slide audience PDF, Beamer source, portable numerical evidence, figures,
speaker guide, methods and publication notes. The pdfpc update adds short
notes for every slide and [laptop instructions](../../slides/2026-09-11-advisor-meeting/PRESENTING.md).
Run `make -C slides present`, then click the presenter window and press `w`.
Share only the audience window. `make -C slides rehearse` opens the presenter
fullscreen for private practice. No GPU or research-store access is needed.

The deck includes reviewed fresh-input SFT and growing-batch matching results;
its numerical cutoff remains 03:45 UTC. New reward-training results are not yet
adopted. See the [verification record](../../slides/2026-09-11-advisor-meeting/VERIFICATION.md).

# Research on project storage

Research store: `/project/alex_phd/runs/rlm-research-r4`.
Inspect live processes and its `RESEARCH_QUEUE.md` before resuming; this is a
dated snapshot, not current process ownership.

At 06:33 UTC the one-A100 allocation was running the second arm of
`sidecars/root-question-sensitive-authenticated-map-reward-pilot-v3`.
The terminal-answer-only control finished 24 collection windows and 19 weight
updates, then completed its fixed-last evaluation and released its service.
The calculation-bonus comparison began at approximately 06:20 and reached
window 3. Both start independently from the same original question-sensitive
adapter; the second does not continue the first one's weights.

Active operation:
`operations/2026-09-11-after-reward-v2-failure-authenticated-map-reward-v3`.
Accepted successors: the 16-episode API-transfer check, then the 240-call Mistral
batch-size comparison. Their operations are
`2026-09-11-after-reward-v3-api-transfer16` and
`2026-09-11-after-api-transfer16-mistral-nested240-v2`.
The compound-144/output-order-96 proposal and equal-work-848 candidate were
CPU-ready but not accepted or GPU-launched. A proposal directory is not a launch.

Saved control checkpoint:
`sidecars/root-question-sensitive-authenticated-map-reward-pilot-v3/outputs/control-attempt-001/window-24/training/checkpoint-19`.
Adapter SHA-256:
`463671442be6c4867572a77effc2b5ff34e2bb058877dc7b50d0fdf2d0784b4d`.
Optimizer SHA-256:
`31de744821b702c3d1efdd5dc10eeb36776efc85bf862b361d5982a0d5090825`.
State and RNG are saved beside them. Do not change live experiment sources or
select checkpoints based on evaluation outcomes.

An additive CONTROL-only audit is under
`analyses/root-authenticated-map-reward-pair-live-2026-09-11/control-interim-2026-09-11T062759Z`.
It is an agent-produced interim, with MAIN adoption and semantic calculation
review pending; it is not the completed two-arm comparison.

# Resume priorities

1. Check GPU use, allocation end, owners, checkpoints and the ready queue.
2. Read shared Codex quota from
   `operations/2026-09-11-codex-usage-reserve/LATEST.json`; refresh if stale.
   Preserve the user's slide-editing reserve under AGENTS.md.
3. Analyze completed runs and decide whether they change the next experiment
   or belong in the slides. Update evidence, figures, notes and guide together.
4. Keep accepted useful GPU jobs moving. Review later candidates on CPUs without
   altering the active comparison.
5. Push verified source/document milestones with non-force Git pushes.

GitHub does **not** contain these model checkpoints, raw runs, environments or
model caches. These pointers are not a backup of their contents. The
[public research notebook](https://github.com/queelius/rlm-research) has its own
dated artifact-availability statement. On resumption, verify whether project
storage remains accessible independently of the GPU allocation.
