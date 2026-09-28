# Early RL transfer implementation plan

> Execute inline with the executing-plans and test-driven-development skills. Parent performs independent integration review and dispatch.

**Goal:** Evaluate the two existing familiar-TRAIN one-step RL actors through both raw and binder interfaces on the already frozen diagnostic B panel; reuse its two warm controls.

**Architecture:** A thin private adapter reuses fresh-plan construction, original native collection, token capture and native replay. New code admits actual committed endpoints, records cross-interface lineage, prepares four generic jobs and analyzes paired transfer.

**Tech stack:** Existing native TextCraft runtime, Python, Transformers tokenizer, focused pytest.

**Spec:** Parent's September 28 bounded early-transfer task in this thread.

## Constraints and review focus

CPU preparation only; no GPU launch, Git operation, optimizer, task selection, or accepted-source edits. Diagnostic B remains official TRAIN, with eight frozen tasks, seeds 202609280900/901, world 42, FP16 base/FP32 adapter, original templates and existing 96-call/8192-token/256-response/8192-context limits. Ninety-minute scientific caps, 95-minute supervisor caps. Preserve original response tokens and deterministic executed-action distinction. Unusable endpoints and incomplete readouts must not become successful results or zero-valued failures. New descriptors exclude both inherited warm jobs.

## Tasks

- [x] Write focused tests for actual endpoint admission, state/commit mismatch rejection, cross-interface diagnostic pairing, warm independence, incomplete-result handling, and the real saved-request/token/native-replay seam. Run and observe the new feature missing.
- [x] Implement `transfer_common.py`, `collect.py`, `prepare.py`, `compare.py`, and a CPU fixture utility. Reuse the frozen planner privately without first writing a misleading fresh-A lineage. Run focused tests red to green.
- [x] Prepare immutable four-cell plans and generic `PREPARED-JOBS.json`, run the saved native fixture externally, verify pins, and report exact commands, estimates and limitations. Parent reviews before queue dispatch.

## Progress and rulings

- Ruling: user-requested location, CPU-only/no-Git scope and proportional tests override skill defaults for separate worktrees, commits, broad suites and interactive approval gates. Existing assigned worktree is retained; verification is focused rather than a project-wide claim.
- Preflight: endpoint admission supplies checkpoint identity to every plan; diagnostic seed/contract identity supplies pairing to analysis; original warm descriptors are references only, never newly queued jobs.
- Complete: six initial tests failed for the missing adapter, then passed. Added explicit fixture rejection (failed before its guard, passed after) and a hand-calculated paired-contrast check. Final focused suite: eight passed. Ruff check and format check passed. Forty-six unique descriptor pins matched; all four prepare-only commands reproduced their immutable plans. External CPU fixture and unknown-result analysis smoke check passed. See VERIFICATION.md.
- Review handoff: parent receives frozen receipt and source for independent integration review. No source changes follow preparation; documentation only. No deferred implementation issue identified.
