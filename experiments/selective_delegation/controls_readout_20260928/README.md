# Fixed teaching-controls readout

Approved bounded design: add a one-shot CPU reader around existing native-audit
receipts and the public-demand comparator's pure pairing function. No collector,
training/runtime imports, model ancestry checks, new treatment, GPU launch, or Git.

Implementation plan (native execution in the existing worktree):

- [x] Write failing saved-schema fixtures for Qwen16, Phi8, reordered IDs,
  unknown/absent cells, explicit known baseline and incorrect dose lineage.
- [x] Implement explicit cell/contrast inventory, two native-audit adapters,
  `(task_id, seed)` joins and small-receipt provenance.
- [x] Add JSON/Markdown reports with fixed per-world comparisons, unknown bounds,
  and conditional research decisions, never best-checkpoint selection.
- [x] Add a lease-bounded 60-second watcher that reacts only to
  expected audit/terminal evidence changes. Parent alone launches it.
- [x] Run focused fixtures, one-shot against current saved receipts, and Ruff.

The scientific baseline is explicitly named in the contrast inventory. In
particular `reference_plan` is only a scheduling template: it must never choose
the known-recipe behavioral baseline. Original23, cumulative46/69 and Phi23 stay
separate. Same tasks across worlds remain correlated; no pooled independence or
novelty claim follows from the readout.

## Fixed inventory and authority

Exactly 24 cells and 24 contrasts: four original Qwen baselines; four visible-order
cells, each compared with corrected-known and discovery; eight dose cells at
cumulative updates 46/69, each compared with its own teacher's original update 23;
and eight Phi cells, compared teacher-within-interface and interface-within-teacher.
All comparisons are within world 42 or 50, panel 00, fit seed 2026092208. Qwen has
two rollout seeds (16 attempts); Phi has one (8 attempts), not the legacy 16-slot
summary denominator. No cross-family efficacy contrast is reported.

The reader consumes Qwen `NATIVE-AUDIT.json` and Phi **`PHI-AUDIT.json`**, their
small PLAN/training-PLAN receipts, and terminal receipts. It never opens raw
requests, trajectories, model weights or optimizer files. Native auditor output
is the scoring authority; this reader is not a new replay/integrity audit.
Comparisons join `(task_id, seed)`, allowing changed/reordered episode IDs. PLAN
hashes, input/world/sampling/budget pairing, explicit teacher/checkpoint identity,
same-actor interface comparisons and dose-to-original checkpoint lineage are
checked. Bad identities yield an unavailable contrast, not an inferred baseline.

Missing native audits remain wholly unknown, even if unaudited episode files are
present. No-PLAN cells use a clearly labeled fixed schedule until the actual PLAN
arrives. JSON retains success bounds, physical cost, error counts, terminal
failures, small receipt hashes and cutoff. Paired point estimates/task bootstrap
intervals require all pairs known; partial comparisons retain conservative bounds
and the full denominator. Eight reused roots share recipes: these are descriptive
conditional intervals, not independent-compositional or confirmatory evidence.

## Ready commands

Run from the worktree; choose a fresh one-shot filename (exclusive creation):

```bash
PYTHONDONTWRITEBYTECODE=1 /project/alex_phd/envs/rlm/bin/python \
  experiments/selective_delegation/controls_readout_20260928/readout.py \
  --output /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/controls-readout-20260928-001/manual-001.json
```

This creates the JSON and same-stem Markdown. A verified saved readout is at
`R/controls-readout-20260928-001/verified.json` and `.md`; its receipt cutoff is
authoritative. At verification, four baselines were complete and 20 treatment
cells had no PLAN: discovery 7/16 at each world, corrected-known 0/16 at world 42
and 1/16 at world 50. No treatment effect was estimated.

Parent-only observer launch command, after review:

```bash
PYTHONDONTWRITEBYTECODE=1 /project/alex_phd/envs/rlm/bin/python \
  experiments/selective_delegation/controls_readout_20260928/readout.py \
  --watch --deadline-epoch "$SLURM_JOB_END_TIME" \
  --output /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/controls-readout-20260928-001/watch
```

The explicit deadline is clamped to `SLURM_JOB_END_TIME` when set. It checks only
the 24 expected native-audit paths and their directories' `TERMINAL-*.json` every
60 seconds. Inode/size/mtime cache avoids rereading unchanged receipts; identical
receipt bytes and unrelated raw-file changes produce no new snapshot. A stable
changed evidence digest creates `snapshot-<digest-prefix>.json` and `.md`; it does
not modify source/run artifacts or existing snapshots. Completed terminal changes
can produce a snapshot before an audit, but unaudited scores stay unknown. No
long-lived observer was launched during preparation.

Runtime source pins are only this `readout.py` and the existing pure
`../inventory_bottleneck_20260928/compare.py`; both hashes are recorded in every
report. Keep both bytes fixed while observing. `test_readout.py` and this README
are checkpoint/review inputs, not runtime imports. Immutable PLAN changes alone
do not trigger a watcher tick; manual one-shot verification is the escape hatch.

## Focused verification

```bash
PYTHONDONTWRITEBYTECODE=1 /project/alex_phd/envs/rlm/bin/python -m pytest -q \
  -p no:cacheprovider experiments/selective_delegation/controls_readout_20260928/test_readout.py
/project/alex_phd/envs/rlm/bin/ruff check experiments/selective_delegation/controls_readout_20260928/
```

Sixteen focused fixtures cover both saved audit shapes, dedicated Phi filename,
actual 8/16 counts, reordered IDs, absent/partial outcomes, wrong teacher/world,
changed PLAN, same-actor binding, wrong dose ancestry and watcher evidence/deadline
semantics. Initial red failures preceded implementation. Source cross-check found
and repaired the Phi filename mismatch in this new reader before handoff; the
existing Phi auditor was not changed. No GPU, Git or frozen live-source edits.
