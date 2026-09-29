# Phi stable-visible repair

Parent admitted the unchanged seven descriptors into
[the September29 follow-on queue](../followup_admission_20260929/ADMISSION.md).
They are waiting behind existing work. No repaired Phi model or score exists
at this documentation cutoff; the preparation evidence below is CPU-only.

Approved bounded design: one seed2026092208, 23-update Phi-4-mini SFT and two raw
panel00/world42+50 readouts, eight roots and rollout seed2026092204 per world.
Stable repair retains the exact 366 known-teacher target strings and training row
order. Native Phi target IDs and each optimizer minibatch denominator must match
the existing Phi-known teacher. Input histories/input token dose differ. No binder,
new data selection, checkpoint selection, hyperparameter grid or GPU preparation.

## Execution ledger

- [x] Data: copy only public stable-history text plus original targets; drop stale
  Qwen caches. Native-tokenize against the existing Phi-known rows, assert exact
  target/row identities and all 23 matched denominators; pin compact receipts.
- [x] Trainer: privately reuse the qualified Phi wrapper and fixed optimizer;
  prepare the actual PLAN using the same training environment. Endpoint audit
  requires actual completed cp23, matched initialization and actual target dose.
- [x] Readout: privately reuse native Phi raw collection/audit, canonicalize JSON
  PLAN fields, bind actual endpoint only after completion, include
  `prepare_only=False` in the real runtime namespace.
- [x] Verify: focused RED→GREEN seam checks plus an actual saved-request/decode/
  native-score CPU fixture. Produce generic-executor-ready pinned descriptors.

Pre-flight interfaces: stable row IDs/order → Phi target tokenization → fixed
training PLAN → actual endpoint audit → two fixed pending readout templates →
native audits → paired report. A pending template is never an actual actor PLAN.

Ruling: the user/parent-approved isolated research package supplies implementation
authority; existing worktree and this ledger replace Git-based skill bookkeeping.
Parent owns fresh admission review and all launches. No sealed source, live owner,
Git state or model ancestry is changed. Expected GPU wall15–30minutes from existing
Phi timings (SFT141–146s; paired raw worlds696–1512s); scientific caps30minutes per
owner, three owners total. Caps are not expected runtime.

Data preflight ruling: repaired `step` is native execution order, not the original
known-teacher action index. All366 target strings and `source_action_index` values
match the original rows in file order. Preserve both semantic fields and compare
original target identity using `source_action_index`; neither prompts, targets,
training order nor native steps are changed. The failed CPU check is retained as
`PREPARATION-FAILURE-001.json`; its regression test went RED before the fix.

## Ready artifact and commands

External root `R/textcraft-phi-repair-20260929-001` contains immutable
`PREPARATION.json`, `PREPARED-JOBS.json`, `stable_visible/{rows.jsonl,tasks.jsonl,
MANIFEST.json,TOKEN-AUDIT.json}`, the actual training PLAN/PHI-CONTRACT, two
`eval-stable_visible-cp23-p00-w{42,50}-raw/PENDING-PLAN.json` templates, and the
saved-request fixture. Here `R` is
`/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921`.

The generic receipt has seven ordered jobs with only `name`, `argv`,
`cap_seconds`, `pins`, and (scientific owners only) `output`. Its SHA256 is
`1e881eb1dfd38c009560197279143f5ba542250daba8a3a9d9b5779539c08ebd`.
All38 unique pins were checked after preparation. Three science owners have
1800-second internal caps and2100-second executor caps; four CPU stages have
300-second caps. The7500-second executor cap sum is not the expected runtime.

Parent-only serialized launch equivalents:

```bash
P=/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python
S=/project/alex_phd/repos/rlm/.worktrees/selective-delegation-20260921/experiments/selective_delegation/phi_repair_20260929/experiment.py
"$P" "$S" train
"$P" "$S" endpoint
"$P" "$S" readout --world 42
"$P" "$S" audit --world 42
"$P" "$S" readout --world 50
"$P" "$S" audit --world 50
"$P" "$S" report
```

`train` passes `resume=True` to the exact existing recipe because CPU preparation
already wrote its PLAN. It restores only this new training output if interrupted;
initially no checkpoint exists. It is a fresh Phi base fit, not continuation from
known-teacher cp23. Checkpoints0–23 include optimizer/RNG state. The endpoint stage
requires an authenticated completed owner, exact actual cp23 and all23 actual
minibatch token counts; it also checks recorded cp0 initialization against the
existing Phi-known actor and actual FP32 tensor shapes. Its checkpoint hash is
unknown until completion. Each readout requires this endpoint receipt and exact
pending-template equality before writing its actual PLAN. Later owners reuse the
verified adapter's unchanged stat and small receipts, not base-weight rehashes.

Outputs are `train-stable_visible-seed2026092208/`,
`eval-stable_visible-cp23-p00-w42-raw/`, and
`eval-stable_visible-cp23-p00-w50-raw/` under the external root. CPU stages write
`ENDPOINT-AUDIT.json`, each cell's `PHI-AUDIT.json`, and `PAIRED-REPORT.json`.
Missing endpoint/audit dependencies fail explicitly; no fallback actor, imputed
success or successful aggregate is manufactured. The native audit preserves
unknown episodes. Do not use the inherited generic16-slot SUMMARY display.

## Controls and interpretation

Native Phi label dose is9,081 tokens, exactly matched row by row and at every
minibatch to original Phi-known. Input dose is421,776 versus405,612 tokens
(+3.98%); max full context3238 and max target51, with no truncation. Both native
assistant turn-end/EOS IDs `[200020,199999]` are supervised. The same seed,
one epoch/23 updates, effective batch16, LR1e-4, rank8/alpha16/dropout0,
baseBF16/LoRAFP32 and128 fused Phi projections are retained.

The baseline cp23 actors are under
`R/textcraft-phi-{known,discovery}-sft-20260928-001/checkpoint-0023` and behavioral
baselines under `R/textcraft-phi-{known,discovery}-raw-w{42,50}-20260928-001`.
Both are explicit; no `reference_plan` selects a behavioral baseline. Known raw
scored1/16 and discovery4/16 across these worlds in the prior completed contrast.
The new repair result is unknown. The planned aggregate pairs exact task/world/
rollout identities and resamples eight roots, retaining both worlds per root.

A repair gain would support transfer of this teacher-history repair package to
Phi, not visibility-only causality or universal cross-model transfer. A null or
reversed gain narrows that claim; it must remain in the report. These are exposed
eight roots, one fit seed and one rollout seed, not16 independent novel problems.
Input history/dose, original Qwen-era prompt budget counters and offline gold-action
scheduling are explicit limitations. No binder cell or checkpoint selection is
part of this package.

## Verification and freeze

Five focused tests pass; scoped Ruff check and format check pass. The CPU fixture
made eight scripted native calls, scored1, exercised both stop IDs and charged the
one injected raw craft error. Independent replay of a real prior Phi discovery
episode reproduced score1 over seven saved calls. The production dispatch function
was exercised with only owner launch replaced; `prepare_only=False` and endpoint
forwarding were checked. No actual readout PLAN was written by this fixture.
These checks establish the runtime seam, not the new model's quality.

The existing Phi tensor header was read without loading weights:256 tensors,
11,534,336 parameters, allFP32. No GPU, scientific model call, process control,
Git operation, prior-source edit, or large ancestry rehash occurred. The initial
metadata preflight failure is preserved. Runtime source is frozen for parent
admission review:

| Source/artifact | SHA256 |
| --- | --- |
| common.py | `b2ae349dea7de8bd41b80479a694dc4798ecdceb7979a253a5b3e1d1815ed578` |
| experiment.py | `4ed3cd72805a48ff5059c3bb6fe7043a5601311c4940dc9389330b8f3f67061c` |
| seam_fixture.py | `1e9dd20fb745381a621020b40f1374fb865d37078bd489a721be27d5d5e2ffec` |
| Training PLAN | `7f50fc2f463f485cc24637e43bbb3227ffe03a2bb616ab4da1dc6836d8310dd5` |
| Prepared rows | `a28f971956299852860a5b3d96a7b4e1d31442f79d35f877f1fa3d8035e0eb7a` |

Environment matches the prior Phi fit: Python3.12.12, Torch2.13.0+cu130,
Transformers5.15.1, PEFT0.20.0; no installs or environment mutation. The pinned Phi
model/revision remains `microsoft/Phi-4-mini-instruct@cfbefacb99257ffa30c83adab238a50856ac3083`.
