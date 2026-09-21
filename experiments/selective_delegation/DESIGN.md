---
date: 2026-09-21
status: exploratory-approved
question: Can a small controller learn when a further evidence or decomposition step is worth its cost?
---

# Selective delegation: first decision-value experiment

The user approved autonomous overnight implementation after brainstorming. Main owns
the sole GPU. Freeze helpers at Qwen3-4B-Instruct-2507. Use the existing native
inference runner; never execute generated code. Preserve original documents in ALL
final-answer arms, so this is not another lossy-summary comparison.

Each parent first produces a shared checkpoint containing provisional answer,
one evidence question, and two proposed subquestions. Fork finish, reconsider,
targeted-evidence, and decomposition arms from exactly that checkpoint. Finish
uses one final call. Other arms use one helper and one final call, with the same
helper cap. Final receives full source, checkpoint, and actual helper report.
Malformed checkpoints are recorded failures, not repaired or silently replaced.
Three continuation seeds estimate repeatability; initial checkpoint is fixed.

Public data: question and paragraph id/title/text only. Host-only labels: gold
answer/aliases, supporting paragraph flags, reference decomposition, hop count,
source ID. Never expose hop-bearing source IDs in prompts. All variants and seeds
of a parent stay in one split. Freeze official-train 2/3-hop training cases,
official-dev 2/3-hop validation, and dev 4-hop transfer cases. Exclude prior breadth
dev IDs. Remove shared atomic-question components between new splits where feasible;
document exclusions and avoid claiming pretraining noncontamination.

First pilot: 32 training parents, three continuation seeds, 704 planned calls.
Follow-ons: a fresh 32-parent validation block, then expand training to 256 parents
if collection works and action differences exist. Transfer remains unopened until
training decisions are fixed. Frozen selection is independent of model outcomes.

Metrics: official MuSiQue answer EM and token F1 with aliases; invalid output and
transport failure separately; paired arm differences by parent; observed
heterogeneity and repeatability, not just optimistic best-arm selection. Count
shared checkpoint once physically and once per deployed policy; measure all
helper, final, router and training costs. Report accuracy/cost curves.

Training, conditional on headroom: compare simple fixed choice and observation-only
heuristics, SFT, and real policy-gradient updates on a categorical controller.
If using a cached intervention table, label it offline contextual-bandit training,
not fresh end-to-end RL. Keep all gold and arm outcomes outside controller input.
Use separate held-out parents and fresh continuation seeds. Run actual learning
curves, checkpoint optimizer/RNG/cursor, and promote to online rollout RL only when
the acquisition decision has a useful signal. PUCT/KLPO are not initial dependencies.

Runtime: project-owned environments, existing exclusive coordinator flock, native
token IDs, immutable input hashes, per-call receipts, real-response check within
90 seconds, bounded attempts, shutdown only authenticated owned processes. Initial
pilot cap 2 hours; all campaigns stop at allocation end minus 10 minutes. Research
artifacts live in /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921.
No model weights or raw datasets enter Git.
