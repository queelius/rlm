# Flexible-discovery admission implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline. Preserve the completed v1 experiment.

**Goal:** Test actual helper uptake without rejecting useful discovery queries merely because of their order.

**Architecture:** A new thin adapter subclasses the frozen v1 routing frame. Recipe queries become optional suggestions; actual delegate boundaries retain the original enforcement and accounting. Reuse the native collector and independent replay auditor.

**Tech Stack:** Python, existing native TextCraft evaluator, Qwen public-SFT checkpoint23.

**Spec:** The concrete failure in `textcraft-decomp-flat-20260928-001` is the spec: after querying the root, the model requested two real, visible prerequisites in a different order; both were rejected before any delegation opportunity.

## Global constraints

- Preserve old source, PLANs and results byte-for-byte.
- One reused exploratory VAL task; three policies, one seed; no new efficacy claim.
- At most32 model responses/arm and15minutes/arm. Two rejected delegate instructions stop the screen with UNKNOWN root outcome.
- Use only returned recipes/current stock. No host-generated model actions or free queries.
- Shared native inventory, native child snapshots, global costs and original helper rule remain unchanged.

## Review focus

- A non-lexical valid recipe lookup must execute, not count as a refusal.
- An unrelated native action error must not become a delegate refusal.
- A required delegate must still create a real child sharing inventory and budget.
- The saved-request replay must use the new prompt and the exact implementation.
- A planned screening cutoff must not be counted as a failed root task.

### Task 1: Narrow routing adapter and saved-request fixture

**Files:** Create `routing_flexible.py`, `run_flexible.py`, `test_flexible.py`, `fixture_flexible.py` in this directory.

**Interfaces:** Reuse v1 `make_bridge(mode)`, `build(args, fixture=False)`, `audit(output, require_terminal=True)`; provide the same interfaces with optional discovery queries and exact source pins.

- [x] Write regression tests for non-lexical query execution, optional-query control, delegate rejection/uptake, and child snapshots.
- [x] Run tests against v1; expect the non-lexical query and optional-query tests to fail.
- [x] Implement the new adapter without editing v1.
- [x] Run focused tests and the real CPU request/decode/native replay fixture; expect all cases to pass.
- [x] Prepare three new immutable plans; review before parent-only admission.
- [ ] Commit the adapter, decision record and focused verification at the next research checkpoint.

## Execution decisions

Use the existing isolated research worktree. The user has already authorized autonomous scoped decisions, so there is no approval pause. Use focused tests, not a repository-wide campaign, per the repository's exploratory-research instructions. Independent reviewer may run alongside preparation; an unrelated ready GPU job continues throughout.

## Verification and admission

Five focused tests pass. Actual CPU fixture002 exercises saved requests, native
decode, child state changes and independent replay: flat/fixed/adaptive succeed
in23/29/28 scripted responses; fixed/adaptive have real scripted child responses;
two-refusal and three-response cap cases preserve UNKNOWN root scores.
Independent reviewer `rl_next_probe_20260928` reran the tests and verified all
fixture source pins, with no blocking issue found. This is runtime evidence,
not pretrained-model competence.

The initial fixture exposed an unrelated cached repository's generic `run.py`
import; scoped module binding fixes that seam without changing either repository.
V2 also corrects the inherited summary's16-slot count to the actual PLAN count.
All v1 artifacts remain unchanged.

Parent accepted the three GPU screens in
`R/compact-delegation-queue-20260928-001/ACCEPTED.json`, SHA256
`9452210ee2b4700ebc0b0c5f6d8a5b6c5b3668a657dea4f407c9cb809b5bda35`.
They wait behind existing scientific owners. Forced public subgoal selection is
not learned decomposition; no helper may also mean no qualifying candidate.
