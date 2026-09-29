# September29 bounded follow-on admission plan

> For agentic workers: use superpowers:executing-plans for the small admission
> composer. Root owns process/Git actions; agents prepare independent studies.

**Goal:** Accept reviewed Phi repair, complete-goal delegation and baseline
restart descriptors after the whole current queue, without editing it.

**Architecture:** Reuse `queue_transfer_20260928.accept` and the frozen generic
executor. Compose immutable prepared jobs, pin their receipts and the admission
code, and preserve scientific commands/output/caps. No new scheduler.

**Tech stack:** Existing Python environments and JSON receipts.

**Spec:** `../FINDINGS-20260929.md`, `../phi_base_restart_20260929/README.md`,
`../delegation_complete_20260929/README.md` and the new Phi repair preparation.

## Constraints and review focus

One A100; existing owner and tail002 remain untouched. No duplicate scientific
output; conflicting source pins are errors. Receipt hashes must match reviewed
bytes. Endpoint-dependent Phi evaluation remains behind its native training
gate. Prepared does not mean launched; conditional/missing outcomes stay unknown.
The user explicitly requests autonomous, proportionate research: no approval
pause, new branch, broad suite, or production scheduler work.

## Task1: Compose and admit

Files: `admit.py`, `test_admit.py`, this record.
Interface: `load_jobs(path, expected_sha)` returns unmodified descriptors except
for additive receipt pins. `combine(groups)` rejects duplicate names, duplicate
scientific outputs and inconsistent pins. `base_jobs()` describes the two
already reviewed restart_v2 preparations. CLI requires the exact reviewed Phi
manifest hash; `--launch` alone allows process creation.

- [x] Write focused receipt-drift, duplicate-output and conflicting-pin tests;
  observe them fail before implementation.
- [x] Implement only composition and calls to the existing accept function.
- [x] Run focused tests and scoped Ruff; dry-run against all real manifests.
- [x] Check a second reader's study reviews and exact commands; admit once.
- [ ] Record PID/receipt hash/expected useful coverage and push scoped sources.
- [x] Start a read-only journal for this receipt; verify its invocation and first
  snapshot. It reports responses/errors, never launches or stops models.

## Execution ledger

Pre-flight: prepared study jobs all consume the same generic descriptor schema.
Ruling: preserve the existing linked research worktree and active queue. Append
follow-ups instead of editing sealed jobs; this delays them but cannot interrupt
current evidence. No meaningful GPU idle period is introduced.

Composition: six RED-to-GREEN tests cover receipt drift, duplicate names/outputs,
conflicting pins and unchanged scientific descriptors. Existing executor reused.
Ruling: the live September28 health watch only discovers that date's receipts.
Add a tiny private wrapper for the new exact receipt, leaving the live watch
unchanged. Two RED-to-GREEN tests cover exact receipt selection and hash drift.

Accepted once under PID1218112, behind unchanged tail002. Receipt SHA
`d7123d5e7e264e164dd07b2e9ac82f18993db38a5a04e5ddca71e5cbaaa64d5a`.
Eighteen descriptors, eight scientific outputs,88unique pins;4.35hours summed
caps versus an estimated0.75–1.5hours useful GPU work. No new scientific owner
exists yet. Read-only health journal PID1218542 watches this receipt. Both
invocations are live; actual science still belongs to the predecessor.
