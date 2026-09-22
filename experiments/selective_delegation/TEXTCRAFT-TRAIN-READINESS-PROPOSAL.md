# TRAIN rollout-readiness screen

**Admission update, September22 10:00 UTC:** accepted under immutable
`INDEPENDENT-TRAINING-QUEUE-006.json`, SHA256
`cd73d0c92b9bc4764c743479e3a896e0789009e6aeb2106aad7b431a5a3799c6`.
GatePID117795 (exec69958) waits for the entire authenticated queue005 and its
existing GPU owners to release. The actual TRAIN readout has not started yet.
The original proposed receipts below remain unchanged. No RL updates are authorized
by this readout admission; choose them only after interpreting the collected rewards.

Question: does fixed public-discovery checkpoint056/23 produce usable native execution and both successful and unsuccessful rollouts on the same training task? This is a feasibility check for a later RL objective, not an RL update or a held-out evaluation.

Eight tasks are selected from the exact32 SFT TRAIN tasks. Within annotated recipe-depth strata, take the first SHA256(`2026092212:official_id`):2 tasks at depth2,2 at depth3,4 at depth4. The source contains11/11/10 tasks, so no quota adjustment or replacement was needed. Selection does not consult model outcomes or057 failures. Declared recipe depth is not necessarily the remaining inventory-conditioned execution depth.

Frozen task suffixes:2040,1791,2294,1680,429,404,465,860. All belong to `textcraft_synth.train`; none is a VAL task. This is deliberately in-sample, with shared seed42 recipes and exact SFT task exposure.

Four samples per task use seeds2026092213–2216, with all eight tasks visited before the next sample. The original flat public interface is unchanged: T0.5, top-p1, top-k0;96 calls,8192 generated tokens,256 tokens/request,8192 context, no truncation or retries. The fixed public056 checkpoint23 is frozen and enabled for every action. The run cap is60 minutes;32 planned episodes and at most3072 model calls. Initial prompts are435–483 tokens (largest prompt plus output cap739).

Report every task's0–4 native-success count, mixed and flat groups, action/protocol failures, missing or unavailable episodes, execution diversity, and actual costs. A returned native zero is not a transport failure. Capped/missing groups remain incomplete, not all-zero. Do not select favorable groups for a claimed RL gain or automatically start training. Diversity in exact training tasks is only a prerequisite signal, not evidence that reward-based training will generalize.

Artifacts under `/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/`:

- Source:`source-063-textcraft-train-readiness`, manifest SHA256`cea3404adee64ad4d718476869bc3d93937044277e8c4c4fe560b48c047146c6`.
- Inputs:`textcraft-train-readiness-inputs-001`; tasks SHA256`92eae4ca292f92b3e19780b41370702cc8057196b76eec27f05dc01f1f0b5a49`.
- Output PLAN:`textcraft-train-readiness-001/PLAN.json`, SHA256`e96a7ac7e9ded7695217378cd1668607c68cc5c0bbe244d40a88842b4c691444`.
- Proposal:`TEXTCRAFT-TRAIN-READINESS-PROPOSED-001.json`, SHA256`6e19ba2252f63802c18896d992a0f9f67dd5b879c564f3fc3ece7f6258a5edd5` contains exact runtime and native-analysis commands and dependency pins.

Qualification: three focused sealed CPU tests passed in5.24 seconds, covering outcome-independent selection,32-slot/interleaved sampling and missing accounting, and exact saved public-training prompt/token reconstruction plus native return/terminal-score receipts. Ruff passed. No4B weights were loaded and no GPU run was launched. This wrapper adds no shared-module changes; it copies the separately qualified dynamic inventory summary/analyzer into its additive seal.

Analysis amendment: `TEXTCRAFT-TRAIN-READINESS-PROPOSED-002.json` (SHA256`3c2b63e2e7aa49d4d77811a61ad38eefbadd285d6924be2f7137583725174c81`) binds the separate `analysis-source-textcraft-train-readiness-001` seal (SHA256`6bbdbf89b5a86aca80275d353c7c1a41a3c5518147d815854fa0c50691ee63f3`). The collector/input PLAN is unchanged. The terminal-only native auditor now produces all eight per-task classifications, observed successes/failures, missing/unavailable counts, error-path diagnostics and actual costs, without the generic two-seed bootstrap wording. Two sealed focused tests passed in5.68 seconds, including a real saved73-call native episode replay. Invalid-action paths remain included in terminal reward; incomplete groups cannot be classified all-zero.
