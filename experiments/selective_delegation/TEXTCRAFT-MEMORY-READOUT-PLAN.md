# Public recipe-memory factorial

CPU prepared, not GPU accepted. Fixed endpoints are public teacher seed1,
public teacher seed2026092291, and completed FP16 continuation checkpoint2
(inference remains BF16). Each endpoint receives all16 frozen fresh goals ×2
existing seeds in four independently owned readouts:

- `history_full`: full native action/feedback history.
- `ledger_full`: full history plus a notebook of actual prior public recipe replies.
- `recent4`: last four complete history entries, with current goal and inventory retained.
- `ledger_recent4`: recent4 plus the same notebook.

All four add the same memory-description text. The notebook contains no hidden
recipes, inferred demands, inventory reservations, suggested actions or stale
inventory. Native tool actions, strict parser, stopping, scores and initial states
are unchanged. Prior get_info metadata already publicly returned may be retained;
this is not a new private recipe lookup. No prompt/code execution from the model.

Every arm has the same two-hour collection limit and unchanged per-episode limits:
96 calls,8192 emitted tokens,256 per call,8192 context and original sampling/seeds.
The known1hour historical runs are supplemental, not the primary matched control.
Notebook token overhead and loss of old observations under recent4 are part of the
intervention. No silent truncation, prompt repair or answer fallback is introduced.

The same scoped memory hook surrounds collection and native auditing. The actual
sealed collector path/hash is explicitly pinned in every PLAN, alongside the
memory implementation and fixed endpoint. A CPU fixture replayed all five saved
native outputs of one real successful episode through the new prompt format and
exact native auditor; auditing without the hook rejects its changed prompts.
This tests request/rendering/scoring integration, not model quality under new prompts.

Audit each completed arm before its next companion if practical, but do not change
the frozen remaining arms based on partial outcomes. After four arms, report all
six paired differences plus `(ledger_recent4−recent4)−(ledger_full−history_full)`.
Use task-parent bootstrap retaining both seeds; one shared world is not independent
recipe-world replication. Missing/interrupted attempts remain unknown, never losses;
complete-panel CIs require all pairs. Report native calls,tokens,actual wall time,
context caps,protocol/action errors and success bounds, not only maximum budgets.

Priority is public seed1, public seed2291, then RL checkpoint2. Twelve2hour caps
sum to24hours before CPU analysis, exceeding a20hour lease in the worst case;
actual runs may be shorter, and allocation-margin clamps remain authoritative.
Do not imply all12 will finish. There is no new GPU-launch authority in this proposal.
