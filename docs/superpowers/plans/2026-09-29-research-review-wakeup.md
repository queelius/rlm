# Research Review Wakeup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resume actual research reasoning on result events and periodic checks.

**Architecture:** A stdlib observer queues messages to the existing exact Codex
thread. Explicit model acknowledgements bound the queue to one pending review;
GPU owners and accepted experiments are untouched.

**Tech Stack:** Python 3.10+, installed Codex CLI, existing quota sampler.

**Spec:** `docs/superpowers/specs/2026-09-29-research-review-wakeup.md`

## Global Constraints

- Exact existing thread UUID only, no `--last` or new model session.
- 20-minute fallback, five-minute routine event coalescing, immediate new service
  alerts, 15-minute quota freshness.
- Stop submissions at 12% remaining or allocation end minus ten minutes.
- One pending review; delivery requires a model-written acknowledgement.
- No GPU/process ownership changes or edits to scientific inputs.

## Review Focus

- An accepted native queue request may not have reached the model yet.
- Incomplete JSON writes or transient quota failures must not become successes.
- A stage can finish while the previous review is still running.
- Repeated service errors must not flood the conversation.
- Restart or ambiguous queue timeout must not duplicate pending messages.

### Task 1: Dispatcher, review receipt, and bounded live activation

**Files:**
- Create `experiments/selective_delegation/research_review_20260929/monitor.py`.
- Create `experiments/selective_delegation/research_review_20260929/test_monitor.py`.
- Create `experiments/selective_delegation/research_review_20260929/README.md`.
- Update `AGENTS.md` and `docs/RESEARCH_OPERATIONS.md`.

**Interfaces:**
- Consumes explicit config paths, accepted queue stage receipts, STATUS.json,
  quota sampler's `sample()`, and model-written acknowledgement JSON.
- Produces STATE.json, requests/ and acknowledgements/ records and the exact
  `codex queue` invocation. CLI `--config PATH --ack ID --decision TEXT` records
  a reviewed decision. The observer itself cannot acknowledge a review.

- [x] Write focused tests for due events/timer, unknown/low quota, pending
  coalescing, exact thread invocation, acknowledgement and failed admission.
- [x] Run tests; expect missing module before implementation.
- [x] Implement the small observer and acknowledgement CLI.
- [x] Run tests and scoped Ruff; expect all passing.
- [x] Review the implementation independently; fix material findings.
- [x] Launch with the existing native probe pending; preserve its admission ID.
- [ ] Update operating docs and checkpoint, commit/push scoped files.
- [ ] Yield to the queued probe; verify actual model delivery and acknowledge it.

## Decisions

The user explicitly requests uninterrupted autonomous execution, so implement
inline without an approval pause. Reuse the existing isolated research worktree.
The most recent user-supplied AGENTS policy preserves 10%; a 12% dispatch gate
adds headroom despite a newer local historical note permitting more spending.
Do not claim periodic delivery before the live next-turn test succeeds.

Independent review: retain immediate new-service alerts as an explicit exception
to routine coalescing, to catch wasted all-error runs promptly. A dedicated test
pins deduplication. Recheck deadline/STOP after slow quota sampling; two fixtures
reproduced erroneous admission and are now fixed. Evidence-free acknowledgement
was upgraded from minor to material because a review receipt must support the
claimed reasoning; its missing-pointer fixture failed before the fix.
