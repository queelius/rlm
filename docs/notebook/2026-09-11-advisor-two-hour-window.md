---
title: Eight-slide advisor update and short-experiment handoff
recorded_utc: 2026-09-11T13:00:00Z
status: dated_operational_snapshot
audience_deck_evidence_cutoff_utc: 2026-09-11T12:55:00Z
---

# Advisor package

The [current deck](../../slides/2026-09-11-advisor-meeting/README.md) contains eight
main slides for a ten-minute discussion and six optional backups. It connects a
visual RLM explanation, supervised-training progress, two-model helper matching,
measured accuracy/time tradeoffs, and a proposed record-level handoff/group-size
component. The old August deck is unchanged. Source, PDF, figures, portable data,
private pdfpc notes, and slide-indexed teaching guide are in Git.

Compilation, all-page visual inspection, real pdfpc checks at1280×720, 14 focused
tests, and source pins are recorded in
[VERIFICATION.md](../../slides/2026-09-11-advisor-meeting/VERIFICATION.md).
Present with `make -C slides present`; click the presenter and press `w`.
Share only the audience window. `make -C slides rehearse` is private practice.

# New evidence adopted this window

- S4: described compound operations, faithful-and-correct2/72→29/72; missing
  bounds2–13→29–30. Supplied plan and previously used records, not autonomous planning.
- H9: same-panel second-model replication. At64records, matching names changed
  Qwen44%→83% and Mistral35%→52%; three malformed Mistral batches count wrong.
- H10: requested output orders83/82/81%, all96calls available. Not equivalence.
- E1: same768records, no-names48 / named48 / no-names16 / singleton accuracy
  49/85/81/87% in19/91/19/31seconds. Four concurrent requests; input/output costs
  differ. This changes the next experiment: include smaller-call baselines.
- R3: secondary final-answer counts57before,55–57answer-reward,54extra-check.
  Each attempted576solutions;19/21updates, unequal realized dose. Full primary
  faithful-calculation review is unfinished; do not claim reinforcement learning
  cannot work. Both runs together used about4h39m.

Detailed artifact paths and hashes are in the methods/portable data beside the
slides. MAIN reran native evidence readers for H9,H10,E1 and S4 extraction;
all-path semantic judgments remain explicitly scoped to agent review.

# Research state and resumption

Research store: `/project/alex_phd/runs/rlm-research-r4`. Inspect its live
`RESEARCH_QUEUE.md`, GPU processes and owners before any launch. This file is a
snapshot, not live ownership. User's two-hour presentation-focused window ends
around13:44UTC, earlier than the scheduler allocation end.

The Qwen compact-output comparison is complete:660/768objects versus654/768
compact,113.48versus86.79workload seconds,18,210versus14,104output tokens.
MAIN read/reran the32-response native auditor exactly. Analysis directory:
`analyses/leaf-mnli-compact-keyed-reply-live-2026-09-11`.
This supports a smaller output representation, not a whole-RLM improvement.

At this snapshot, its Mistral replication is active under PTY83835:
`sidecars/leaf-mnli-compact-keyed-reply-mistral7b-v1/outputs/attempt-001`.
Owner start12:55:02.012675UTC;720-second outercap,690owned,570work.
READY identity:`b9565972d9590f98913123e2fe15cd0df5a7c5cd898943079f69cb814ff8e7ab`.
The compact representations share all prompt fields except output schema.

A final32-root/32-helper comparison is being prepared, not accepted or launched:
large named calls versus smaller unnamed calls feeding the same record-map API.
The proposed1200-second cap must leave time to audit and update the deck. No
new training or broad refactor is needed for this meeting window.

The shared Codex account had31% remaining at12:55:59. Read
`operations/2026-09-11-codex-usage-reserve/LATEST.json`; another session consumes
the same allowance. Wind down at20%, pause research at15%, preserve roughly10%
for user slide editing. There is no hard cross-session reservation.

GitHub contains the presentation/source/summaries, **not** model weights,
optimizer checkpoints, raw trajectories, environments or external caches.
Those remain on project storage; pointers are not a backup of their contents.
