---
date: 2026-09-29
status: accepted_waiting_for_predecessor
question_ids: [teaching-repair-model-transfer, complete-goal-delegation]
queue: evidence-followups-20260929-001
receipt_sha256: d7123d5e7e264e164dd07b2e9ac82f18993db38a5a04e5ddca71e5cbaaa64d5a
scientific_stages: 8
expected_useful_gpu_hours: [0.75, 1.5]
sum_of_executor_caps_hours: 4.35
---

# Accepted follow-ups, not completed results

Supervisor1218112 waits for `R/information-first-tail-20260928-002` to settle
and for every acquired scientific owner to release. No active job or sealed
source was changed. The allocation ends October1 at09:50:35UTC; the executor
retains its allocation margin and refuses unresolved ownership. Summed caps
are ceilings, not an estimate of useful coverage.

| Order | Scientific question | New GPU jobs | Actual comparison |
|---|---|---:|---|
|1|Does the teaching repair transfer to Phi?|3|One fixed23-update fit;8roots×one seed in each of two worlds; raw execution.|
|2|Does an admitted helper improve the whole task?|3|Flat/fixed/adaptive,2fresh attempt seeds each, ONE exposed goal; shared96-call budget.|
|3|Restore missing base references.|2|Unchanged2episodes/world; original attempts failed before model load.|

Separate CPU audits and two paired reports bring the queue to18descriptors.
Phi readouts require an actual released, complete checkpoint23 and endpoint
audit. No future actor hash was guessed. Missing dependencies remain missing;
unknown outcomes are not silently converted to failures. Helpers' local success
is not the primary success metric. The tiny base controls are descriptive only.

Inputs and provenance:

- [Phi repair](../phi_repair_20260929/README.md), preparation receipt SHA
  `1e881eb1dfd38c009560197279143f5ba542250daba8a3a9d9b5779539c08ebd`.
- [Whole-task delegation](../delegation_complete_20260929/README.md), SHA
  `25a8051b79804e5e393e18a1c8fb50941e10c2f9db680450f8d110dde978cd14`.
- [Base restart v2](../phi_base_restart_20260929/README.md), new PLAN hashes
  `044448fd919e6d1f0c00f40eec5eabe8069bc96718319a10b82960dcfc93dbd9`
  and `3ecd555109a9631ac19fb105fe3c59293ec2a9e11cf73397dea311375028654f`.

All88unique source/input pins matched at admission. Six composition and two
watch-selection tests passed; original scientific commands, output paths and
caps are retained. Separate study reviews checked actual native replay and
training/readout interfaces. No broad runtime test suite or model-weight
ancestry rehash was required.

Read-only health journal1218542 writes `R/evidence-followups-20260929-001/health/`.
It checks first returned responses and records status/errors every15seconds.
It neither trains models nor stops owners or invents follow-ups. The original
September28 watcher is unchanged; its date glob would not discover this new
receipt. Local monitoring is not a scheduled Codex reasoning wakeup.

Root alone launched the serialized queue. The explicit admission command is
saved here for provenance, **not** as a command to rerun into the same directory:

```text
admit.py --phi-receipt-sha256 1e881eb1dfd38c009560197279143f5ba542250daba8a3a9d9b5779539c08ebd --launch
```

`R=/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.
Use the live external SESSION_CHECKPOINT/RESEARCH_QUEUE to resume, not the fixed
status in this dated admission receipt. Weights and raw traces are not in Git.
