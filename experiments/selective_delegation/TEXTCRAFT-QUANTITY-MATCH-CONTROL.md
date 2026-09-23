# Quantity-matched privileged-teacher control

## Exact confound and correction

The original privileged teacher's `textcraft_synth.train.1029` trace has a surplus first craft:

```json
{"action":"craft","ingredients":{"raw_t8":6},"target_item":"t4_i1","output_count":12}
```

Its gold step is `target=["t4_i1",3]`, `result_count=12`. The public teacher, while still
reaching the same intended root goal `1x t6_i4`, uses the sufficient two batches:

```json
{"action":"craft","ingredients":{"raw_t8":4},"target_item":"t4_i1","output_count":8}
```

The smallest principled control changes only that gold step in a copied task: target quantity 3→2,
ingredients `raw_t8` 6→4, and result count 12→8. It retains the original privileged action order,
the remaining 30 action targets, all 32 TRAIN task identities, prompts/bridge, and root goal. It
does not use validation/fresh outcomes to select the correction.

## Why a target-only row edit is invalid

This is not one target-token edit. The corrected craft's public feedback changes from “crafted 12”
to “crafted 8”; the original teacher serializes public action/feedback history into every later
prompt. A CPU native replay using the pinned original teacher/bridge found:

| Replay property | Original | Corrected |
|---|---:|---:|
| Native-successful TRAIN traces | 32/32 | 32/32 |
| SFT rows | 366 | 366 |
| Changed action target strings | 0 | 1 |
| Changed prompt strings | 0 | 29 |
| Total teacher output tokens | 8,821 | 8,820 |

For train1029 alone, both traces are 31 calls, complete successfully, and have 766 versus 765
teacher output tokens. The other 31 tasks are byte-identical under this deterministic replay.
Thus a new immutable all-32 replayed dataset is required; editing the old `rows.jsonl` would leave
29 mismatched conditioning histories.

## Smallest informative comparison

If admitted, freeze the corrected 366-row privileged dataset with its task/trace/row hashes and run
the existing original-action SFT recipe at the **two already fixed seeds**, 23 updates each, with a
fresh rank-8 adapter. Read both endpoints on the already exposed fresh16 × two-seed panel against
the existing public teacher as a reference. The primary quantity question is corrected-privileged
versus existing privileged; public is contextual rather than a new matched treatment. Preserve all
32 replayed traces, including any unexpected failure; no post-replay filtering.

This is a narrow teacher-data confound control, not a clean causal isolation of teacher order:
the corrected histories, target token count, and optimizer trajectory necessarily differ. It is
also not a fresh evaluation confirmation because fresh16 has been read repeatedly. A positive
result would show that this known quantity surplus mattered; a null would only retire this one
concrete explanation.

## Readiness

The CPU replay demonstrates that the correction is technically feasible and native-successful, but
no corrected dataset, SFT wrapper, source seal, proposed job, or GPU launch was created in this
time-box. Those require an additive immutable data-preparer and a trainer binding new hashes; the
existing trainer deliberately hard-codes the old input receipt. This avoids silently treating
changed downstream histories as the old training dataset.

Pinned inputs used for the CPU replay:

- `textcraft-train-inputs-001/tasks.jsonl`:
  `390dff9bb19d0fe71c7bec0505c97608013c66c65aea90a821839f01b615ab30`
- Existing original rows: `caa78390f9d4ac28e600674b26e56375203b72d8cdad1c3f9471da3fb25776a9`
- Original deterministic trajectory generator:
  `e0a093ebc48e8a9a250770825f93bcc7305f67bfd494616b6705862d9d1b0af3`
- Public bridge: `1553db71f5c4dd4738e50d087c0a33d99d9ea7f97a084e136e4360837f2a7d40`
