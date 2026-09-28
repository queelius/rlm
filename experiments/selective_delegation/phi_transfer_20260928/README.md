---
question_id: OP28-family-transfer
date_utc: 2026-09-28
status: cpu_qualified_weights_acquired_parent_gpu_only
scientific_model_calls: 0
---

# Phi: does teaching-history and assistance transfer beyond Qwen?

Ready: two fixed23-update Phi SFT jobs, four tiny base-control episodes, and a
64-episode paired readout. **No Phi GPU/model generation has run in preparation.**
This extends a Qwen-only result, not an existing Phi control: all298 inspected
top-level TextCraft PLANs identify Qwen3-4B. The earlier
[base control](../TEXTCRAFT-FRESH-BASE-FINDINGS.md) also used Qwen and had substantial
format/context failures, motivating a small untrained-Phi reference.

## Compatibility and acquisition

Official [Phi-4-mini-instruct model card](https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/cfbefacb99257ffa30c83adab238a50856ac3083/README.md),
[config](https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/cfbefacb99257ffa30c83adab238a50856ac3083/config.json)
and [MIT license](https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/cfbefacb99257ffa30c83adab238a50856ac3083/LICENSE)
are pinned to revision `cfbefacb99257ffa30c83adab238a50856ac3083` (upstream last
modified2025-12-10; retrieved2026-09-28). Despite the card's older remote-code
examples, installed Transformers5.15.1 successfully constructs its **built-in
Phi3ForCausalLM with `trust_remote_code=False`**, entirely on meta.

Observed locally:3,836,021,760 parameters; checkpoint index names match the built-in
model, with only the tied `lm_head.weight` omitted from stored keys. Rank8/alpha16
LoRA targets are `qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`:128 projections,
11,534,336 FP32 trainable parameters. No remote model Python was downloaded or
executed; no packages or environments were changed.

The external cache is
`/project/alex_phd/research-cache/models/microsoft--Phi-4-mini-instruct--cfbefacb99257ffa30c83adab238a50856ac3083`.
Small assets21,907,605 bytes were inspected first. Only after CPU qualification,
two safetensor shards totalling7,672,066,216 bytes were acquired; both match their
official LFS SHA256s. All small Git blobs and LFS hashes were checked. Acquisition
receipts contain exact URLs, revision, license, retrieval time and per-file hashes.

## Exact teaching text, native tokenization

| Phi package | TRAIN tasks/rows | Prompt tokens | Supervised tokens | Max full/target |
|---|---:|---:|---:|---:|
| Public discovery |32/366|428,467|9,081|3,238/51|
| Corrected known-recipe |32/366|405,612|9,081|3,238/51|

Prompt and JSON action-target strings, row order, task IDs and corrected goal
quantities are unchanged. Only stale Qwen token caches are dropped from the copied
records. No truncation/removal occurred. Native Phi supervision includes assistant
turn-end then EOS (`[200020,199999]`); live generation stops on either, as declared
by the official generation config. `native_adapter.py` makes counted, fail-closed
changes to inspected local client/auditor functions and records both transformed
source hashes. Stop IDs appear in every saved request and its independent audit.
Native parsing, actions, rewards, budgets and error accounting remain unchanged.

Equal23 updates/366 rows within Phi is **not equal input dose or compute**; neither
tokenization nor LoRA parameter count is matched to Qwen. Original teaching prompt
text includes its original Qwen-era remaining-token counters, intentionally held
fixed with the package. Live Phi counters use actual Phi outputs. This is another
model/package setting, not isolation of a single architectural or history mechanism.

## Smallest readout and decision

Train both packages from the same Phi base, seed2026092208, one epoch, effective
batch16, LR1e-4, rank8/alpha16, fixed checkpoint23; checkpoints every step. The
accepted optimizer, target-only loss, update ordering and owner implementation are
reused through a process-local binding. An explicit `PHI-CONTRACT.json` records
model/projection/native-template differences from its legacy PLAN terminology.

Then evaluate discovery/known × raw/binder × worlds42/50 × the same eight panel00
roots × **one** paired seed2026092204 =64 episodes. These are eight distinct roots,
not64 independent tasks. Keep the existing96-call/8192-output-token global cap,
256 tokens/response, input+cap8192 without truncation, original instructions and
native scoring. The binder uses only recipes observed in that episode. All actors
are flat. Initial Phi prompt+256 is at most781(world42)/835(world50).

The raw-only base reference is four episodes: panel positions0/7 (VAL313 depth2,
VAL294 depth5) in both worlds. It is descriptive, not a broad base performance
estimate. Primary comparisons are discovery-minus-known root success and
binder-minus-raw root success/cost, retaining all failures and unknowns.

Replication in Phi supports two-family transfer, not universality. A missing or
reversed teacher gap revises the family-independent claim at this training dose;
it does not establish a causal architectural explanation. If training fails or
native model responses cannot be obtained, stop that dependency, preserve the
attempt and use an independent ready job. No hyperparameter/model survey or
outcome-selected task replacement is authorized by this package.

## Ready commands

The parent schedules all GPU commands serially through its current allocation.
Expected SFT cost: roughly4–8minutes each, hard cumulative cap30minutes each.
Expected readout:10–20minutes per trained8-episode arm, cap30minutes; base reference
roughly3–8minutes/world, cap15minutes. These are estimates, not measured Phi GPU
throughput. Verify a real response within90seconds, and the first nonzero adapter
update through the unchanged trainer receipt.

```bash
P=/project/alex_phd/repos/rlm-bootstrap/.worktrees/a100-lora-roundtrip/gpu/training/.venv/bin/python
R=/project/alex_phd/runs/rlm-research-r4/sidecars/selective-delegation-20260921
E=/project/alex_phd/repos/rlm/.worktrees/selective-delegation-20260921/experiments/selective_delegation/phi_transfer_20260928
"$P" "$E/evaluate.py" --teacher base --world 42 --hours .25 --output "$R/textcraft-phi-base-w42-20260928-001"
"$P" "$E/evaluate.py" --teacher base --world 50 --hours .25 --output "$R/textcraft-phi-base-w50-20260928-001"
"$P" "$E/train.py" --teacher discovery --output "$R/textcraft-phi-discovery-sft-20260928-001" --resume
"$P" "$E/train.py" --teacher known --output "$R/textcraft-phi-known-sft-20260928-001" --resume
```

The base and training PLANs already exist from CPU `--prepare-only`; training
therefore requires `--resume`. After each completed SFT, use the readout command
below for its four combinations of world42/50 and raw/binder; it requires the real
authenticated fixed checkpoint23 and will not fall back to base/partial weights.

```bash
"$P" "$E/evaluate.py" --teacher discovery --assistance raw --world 42 --output "$R/textcraft-phi-discovery-raw-w42-20260928-001"
```

After each readout owner terminates, repeat its exact command with `--audit`.
All eight trained input combinations passed CPU checks with endpoint explicitly
pending. Their final PLANs bind the actual checkpoint only once it exists.
After each SFT owner terminates, use `check_endpoint.py --teacher discovery`
(or `known`) with `--output R/textcraft-phi-TEACHER-sft-20260928-001/ENDPOINT-AUDIT.json`.
It checks the authenticated complete23-update endpoint, actual adapter shapes and
the9,081 supervised-token dose. `PREPARED-JOBS.json` supplies all12 full argv lists
and their CPU audit argv, dependencies, caps and source/data pins.

**Do not use the inherited collector's group-level `SUMMARY.json` denominators:**
its generic display assumes16 slots, whereas these runs contain2 or8. The dedicated
`PHI-AUDIT.json` loops over the actual immutable PLAN jobs and reports the correct
observed/success/unknown counts. No frozen collector was edited for display.

## Evidence and limitations

`R/textcraft-phi-transfer-20260928-001/QUALIFICATION.json` binds the official assets,
meta module shapes, environment, exact-text per-row token audits and panel inputs.
`native-fixture-001/VERIFICATION.json` passes actual NativeClient request,
GenerationConfig, tokenize/decode, native episode/score and independent saved-call
replay on scripted CPU tensors for raw and binder. Both stop IDs are exercised.
An injected ingredient error is charged by raw and corrected only from observed
recipes by binder. This is runtime evidence, **not Phi model-quality evidence**.

Four focused unit tests pass. Core acquisition/qualification/training/client/
evaluation files pass scoped Ruff. The already-pinned fixture has one nonblocking
import-order warning; it is not claimed whole-package Ruff-clean. Real weight
loading, GPU forward/backward and model trajectories remain to be tested by the
parent. Transformers emits a legacy LongRoPE-field warning; the official config
was not rewritten and no model/library fallback was used.

Environment: Python3.12.12, Torch2.13.0+cu130, Transformers5.15.1, PEFT0.20.0,
Accelerate1.14.0; existing `uv`0.9.7. Existing training `uv.lock` SHA256
`d18054c5eaf69dbc69bc71f14c603df462a6d660001f044915814a9157552158`.

| Artifact | SHA256 |
|---|---|
| Model local manifest |`47ff4e5c0e5b979dfa102f2b92edbeb0bfed41f19a99fd26918552b2cd2fcde9`|
| CPU qualification |`006cb29676dd92e0fe4ce56e531c34e2a10be88a6653fcf10998ab4f61bea537`|
| Native runtime fixture |`dc60adec9b42bc2e3d470287253e0a346631d94d253f5be17aa3ae9818218da3`|
| Discovery SFT PLAN |`c0294328161bbd0cd603140fc950c821c6581f617336e5941f9d80d06bd20321`|
| Corrected-known SFT PLAN |`4eda31f6857a6a823d3d2bac61c4a19b43da3cd9dc5e494fe129f879b8b5ea71`|
| Base world42 PLAN |`befe6509a4e8f88f0b943ffff8708192315f103c10c227dd7ae16d14d962dccc`|
| Base world50 PLAN |`45effd3acf39f9c2d8487dea60abab7f631c7104fa723e9aa04822a5c6cdc98b`|
