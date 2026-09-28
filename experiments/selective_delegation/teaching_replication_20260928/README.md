# Teaching-order replication, 2026-09-28

Approved bounded design: two repaired-teacher trainings at existing fit seed
2026092291, read on panel 00/worlds 42 and 50; four further readouts of the
existing seed 2026092208 repaired actors on panel 01/worlds 42 and 50. Both
stable-visible and random-visible variants remain fixed; no best-arm selection.
Exactly 2 new trainings and 8 new 16-attempt readouts, plus explicit CPU audits.
Root alone launches. All existing source bytes and completed run artifacts stay
unchanged. No GPU or Git operation during preparation.

## Implementation ledger

- [x] Reproducible compact initial MD/JSON from the completed native audits,
  per-task pairing and exact target-token/initialization controls.
- [x] Failing tests for fixed schedule, explicit baseline convention, endpoint
  gates, paired task identity and unknown outcomes; minimal thin implementation.
- [x] Copy exact immutable teacher rows into the new study root and invoke the
  frozen trainer's CPU prepare-only path for the second seed.
- [x] Prepare actual first-seed/panel-01 PLANs; keep second-seed endpoints pending
  until committed checkpoint 23 and released training owner are authenticated.
- [x] Saved-request/response/native replay fixture and generic-executor-ready
  descriptors with scientific outputs only on training/readout stages.
- [x] Focused tests, Ruff, actual CPU preparation and source/receipt pins; parent
  targeted review before launch.

Ruling: the supplied design and explicit autonomous instruction authorize this
bounded implementation without a second approval pause. Reuse the existing
worktree and keep this ledger here; no Git-dependent skill scripts or broad suite.
Existing `s2291` means fit seed 2026092291, not rollout seed. Rollout seeds stay
2026092204/2026092205. Panel 01 adds eight distinct root goals but is an exposed
breadth panel, not a new held-out benchmark.

## Current finding

At the fixed 15:11:13 UTC native-evidence cutoff, stable-visible scored 6/16 and
8/16 in worlds 42/50; random-visible scored 6/16 in both. Corrected-known scored
0/16 and 1/16; discovery scored 7/16 in both. Across the fixed worlds, stable adds
13 successes and random adds 11 against corrected-known, with no paired losses.
The descriptive eight-root bootstrap intervals for their +40.625/+34.375-point
differences are [15.625,65.625]/[9.375,68.75]. Worlds/repeats stay together within
root clusters; shared recipes are still a dependence limitation.

Stable matches discovery's 14/32 total while using 767 versus 1,201 calls; this
does not establish accuracy superiority or general efficiency dominance. Known,
stable and random have identical recorded checkpoint-0 adapter hashes, exact
target/label row order and all 23 actual minibatch target-token denominators.
Input histories/token dose differ. See [FINDING.md](FINDING.md),
[FINDING.json](FINDING.json), and the reproducible `report.py`.

## Fixed baselines and outputs

`R` is `/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.
New study root: `R/textcraft-teaching-replication-20260928-001`.

| Slice | Explicit baseline readout paths below R | Baseline checkpoint parents below R |
| --- | --- | --- |
| Seed 2026092291, panel 00 | `textcraft-breadth-p00-w{42,50}-s2291-{raw,corrected-raw}-001` | Discovery: `textcraft-teacher-seed2291-public-001`; known: `textcraft-quantity-matched-seed2026092291-001` |
| Seed 2026092208, panel 01 | `textcraft-breadth-p01-w{42,50}-soriginal-{raw,corrected-raw}-001` | Discovery: `textcraft-public-discovery-sft-001`; known: `textcraft-quantity-matched-seed2026092208-001` |

All endpoints are fixed `checkpoint-0023`. New second-seed actors are under
`train-{stable_visible,random_visible}-seed2026092291` in the new root. Existing
first-seed actors stay in the original `textcraft-teaching-order-20260928-001`
root. New readout outputs are always in the new root:
`eval-{mode}-s{seed}-p{panel:02d}-w{world}-raw`.

The eight new root task IDs in panel 01 are `val.28`, `val.168`, `val.22`,
`val.498`, `val.555`, `val.181`, `val.341`, `val.147`, each with the
`textcraft_synth.` prefix. No recipe-universe or untouched-holdout claim is made.

## Queue-ready descriptors

`PREPARED-JOBS.json` in the new root is generic-executor-ready. Consume its
`jobs` unchanged: each has `name`, `argv`, `cap_seconds`, `pins`; only the ten
scientific owner stages have `output`. It contains 2 trainings, 2 CPU endpoint
audits, 8 readouts, 8 CPU native audits and 1 CPU paired report: 21 jobs total,
128 new rollout attempts. Native reports remain `NATIVE-AUDIT.json` per readout;
the final aggregate is `REPLICATION-REPORT.{json,md}`.

The frozen trainer is called directly with `--root NEW_ROOT --seed 2026092291
--resume`. `--resume` is required because CPU preparation already wrote PLANs;
there are no new checkpoints or owners yet. Learning recipe remains 23 updates,
LR1e-4, effective batch16/last14, base BF16/LoRA FP32, rank8/alpha16/dropout0.
Readouts retain the exact original sampling, 96-call/8,192-output-token global
caps, 8,192-token context without truncation, and 2 paired rollout seeds.

Four first-seed/panel-01 PLANs bind actual released checkpoint receipts now.
Second-seed PLANs are intentionally absent until actual complete checkpoint 23
exists. The endpoint audit and every readout independently authenticate the
complete checkpoint, row/PLAN hashes and released successful training owner;
missing/failed training cannot substitute a baseline or partial endpoint.
`reference_plan` remains a schedule template, never a behavioral baseline choice.

Scientific hard caps: 30 minutes/training and 45 minutes/readout. Executor caps:
2,100 seconds/training, 3,000/readout, 300/endpoint audit, 600/native audit,
300/final report. Sum of executor caps is 33,900 seconds (9h25m), **not ETA**.
Observed first-seed fits took 205.09/205.70 seconds; repaired readout owner times
were 840.69,703.83,1133.98,1174.21 seconds. Doubling those four readouts plus two
fits suggests 8,116 seconds (2h15m), with a planning range of 2.2–3.5 hours;
panel/seed differences can change runtime.

Final descriptor SHA256:
`eb0214d13a25304919f1f5a9c3adfba59c79af38e87af1df58c3f0de8ee403ab`.
All 69 unique pins verified, 16.2 MB total; no model ancestry rehash. The initial
unaccepted descriptor was retained as `PREPARED-JOBS-PREFLIGHT-001.json` before
final report wording/formatting; only the final filename/hash above is ready.

## Exact commands and verification

From the worktree, to reproduce the current finding with a fresh output filename:

```bash
PYTHONDONTWRITEBYTECODE=1 /project/alex_phd/envs/rlm/bin/python \
  experiments/selective_delegation/teaching_replication_20260928/report.py \
  --output /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/textcraft-teaching-replication-20260928-001/FINDING-RECHECK.json
```

Parent-only first training command (the complete authoritative sequence is in
`PREPARED-JOBS.json`, not this single-job example):

```bash
/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python \
  /project/alex_phd/repos/rlm/.worktrees/selective-delegation-20260921/experiments/selective_delegation/teaching_order_20260928/train.py \
  --root /project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921/textcraft-teaching-replication-20260928-001 \
  --mode stable_visible --seed 2026092291 --resume
```

Seven focused red→green tests and Ruff pass. `CPU-FIXTURE.json` records exact
native replay of two real completed episodes (20 saved request/response calls),
including response decoding and native scores; all four new-panel initial
prompts/token IDs match actual saved baseline requests. Both future endpoints
were rejected before training. `CPU-REPLICATION-READOUT.json` exercises all 16
planned contrasts with unknown treatment outcomes; no missing result is imputed.
No broad suite, GPU launch, background process or Git operation was performed.

Runtime new-source pins:

- `replication.py`: `f1a13a555c96c197d66de8647d3437e026b18847f982db2009383b647f51a0ed`
- `report.py`: `2a326fdfd7483f95f7d5024437d0d416fb741879b1ae329bfe97e1c13ad58f9c`
- `fixture.py`: `f58141950fd710db75a17b8b510cbafc504fe66a4bd84d1c45670b7498b7c317`

Remaining gate: parent's independent admission review. Code and descriptors are
frozen; README/finding copies are documentation only and not runtime imports.
