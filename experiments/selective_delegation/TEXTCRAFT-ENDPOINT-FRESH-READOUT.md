# Fixed-final TextCraft fresh16 readout

**September 22, 11:26 UTC:** The conditional control/readouts are now accepted
in queue008, following corrected terminal RL output `textcraft-terminal-rl-002`.
Actual endpoints do not yet exist. The queue validates them before loading each
model; the earlier prospective receipt's `textcraft-terminal-rl-001` command is
superseded by queue008. All fixed tasks, seeds, interfaces and caps below remain
unchanged. See [the accepted training plan](TEXTCRAFT-TERMINAL-RL-READINESS.md).

## Original CPU preparation

No GPU acceptance, new task selection, or checkpoint selection. Reuse the exact
accepted queue007 public-arm PLAN as immutable interface template (SHA256
`7296007858d36d651e3ce6a7c5836f10351c3e25bff4c2b28896a96253f784b9`).
Sixteen frozen world42 root tasks, two seeds2026092204/2026092205,32 slots per
new arm, original flat prompt, same HF4B model, T0.5/p1/k0,96 global calls,
8192 generated tokens,256 per call,8192 input+cap,60 minutes per arm.
The only scientific change from007 is the final adapter. Conditions explicitly
say `fresh_rl_terminal` or `fresh_matched_sft_terminal`; no old teacher labels.

`eval_textcraft_endpoint_fresh.py` reuses the unmodified native collector via
its existing `prepared_run`/adapter seam. It validates the actual final training
endpoint before creating any readout PLAN. `--validate-inputs-only` validates
the real frozen panel/source without pretending065/control completion.

Endpoint schemas are deliberately not delegated to the old checkpoint23 reader:

| Policy | Authoritative endpoint and dose |
|---|---|
|065 terminal RL|`TERMINAL-*.json.endpoint` must be `boundaries/sample-N/checkpoint-000K`; normal complete, usable, released owner, actual=committed K, K1–2. Sample cursor N need not equal K. BATCH/state/COMMIT/action-token credits are checked through the sealed control validator.|
|Matched extra-SFT|`TERMINAL-*.json.endpoint` must be root `checkpoint-000K`; K equals actual065 committed updates. PLAN schedule must equal deterministic whole-row token matching; every actual update receipt must match. No partial-dose endpoint.|

Both require the same public056 warm identity and frozen063 TRAIN inventory;
failed/capped/zero-update RL is not a substitute warm-policy readout. The adapter
is loaded frozen as `textcraft_action` for every call. Verify final adapter,
config,STATE,COMMIT and small receipts once; do not rehash base weights or load
optimizer tensors for evaluation. Neither endpoint exists yet, so real terminal
integration and4B execution are still unqualified.

After main separately accepts completed endpoints:

```text
TRAINPY eval_textcraft_endpoint_fresh.py --kind rl \
  --training-output R/<completed065> --output R/<new-rl-fresh-readout> --prepare-only
TRAINPY eval_textcraft_endpoint_fresh.py --kind matched_sft \
  --training-output R/<completed-control> --output R/<new-sft-fresh-readout> --prepare-only
```

Only a later accepted invocation omits `--prepare-only`. No launcher or automatic
queue advance is added here. The fixed one-hour limit has no CLI extension.

`analyze_textcraft_endpoint_fresh.py` reuses the existing native action/inventory
auditor and paired comparator. After all owners release, audit32 planned slots
for each of warm007/public056, terminal065, and matched-SFT. Report RL−warm,
SFT−warm and RL−SFT, costs, unknowns, and16-root bootstrap (20,000 draws,
seed2026092206; both seeds kept together). Missing/unavailable never becomes zero;
any unknown pair suppresses the complete-panel effect/CI. Shared recipe-world
dependencies are not independently sampled roots. Retain native per-arm receipts.

```text
CUDA_VISIBLE_DEVICES='' TRAINPY analyze_textcraft_endpoint_fresh.py \
  --rl-output R/<new-rl-fresh-readout> --sft-output R/<new-sft-fresh-readout> \
  --report R/<new-immutable-comparison>.json
```

This panel was frozen prospectively for007 but is reused for the subsequent
fixed-endpoint comparison, not claimed newly untouched. Extra-SFT token/update
matching is not matching states, prompt compute, information, temperatures or
FLOPs. Preserve these limitations regardless of eventual outcomes.

Training control seal: `R/source-textcraft-matched-sft-001/SOURCE.json`, SHA256
`4f7c8be8352e3bbaed735cc1864415d0c8f37dcc084fb831de75a23ec8a8d934`;
prospective receipt `R/TEXTCRAFT-MATCHED-SFT-PROPOSED-001.json`. Frozen row order
is unchanged. Reader/analyzer source is separately preserved in
`R/source-textcraft-endpoint-fresh-001`, with prospective commands/pins in
`R/TEXTCRAFT-ENDPOINT-FRESH-PROPOSED-001.json`. Main reviewed the wrapper; three
focused endpoint/schedule tests and the existing actual saved native-call fixture
pass. CPU `--validate-inputs-only` confirms the real frozen16-task panel without
an endpoint or GPU. No fake future readout PLAN is emitted; no GPU acceptance.
